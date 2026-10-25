//! Snapshot store behaviour under concurrent readers and mixed mmap/heap indexes.

use std::sync::atomic::{AtomicBool, Ordering};

use hnswops::persist::LoadOptions;
use hnswops::{Hnsw, HnswConfig, SnapshotStore};

fn index(n: u64, offset: f32) -> Hnsw {
    let mut idx = Hnsw::new(HnswConfig::new(2).with_m(6).with_ef_construction(24)).unwrap();
    for i in 0..n {
        idx.insert(i, &[(i % 17) as f32 + offset, (i / 17) as f32]).unwrap();
    }
    idx
}

#[test]
fn readers_always_see_a_complete_version_while_writer_commits() {
    let dir = tempfile::tempdir().unwrap();
    let store = SnapshotStore::open(dir.path()).unwrap();
    store.commit(&index(50, 0.0), "seed").unwrap();
    let done = AtomicBool::new(false);
    std::thread::scope(|s| {
        for _ in 0..3 {
            s.spawn(|| {
                let reader = SnapshotStore::open(dir.path()).unwrap();
                let mut loads = 0;
                while !done.load(Ordering::Acquire) || loads == 0 {
                    let (v, idx) = reader.load_current(LoadOptions::default()).unwrap();
                    // Version v holds exactly 50 + v - 1 points.
                    assert_eq!(idx.len() as u64, 49 + v);
                    loads += 1;
                }
            });
        }
        for v in 2..=12u64 {
            store.commit(&index(49 + v, 0.0), &format!("v{v}")).unwrap();
        }
        done.store(true, Ordering::Release);
    });
    assert_eq!(store.current().unwrap(), Some(12));
    assert_eq!(store.prune(3).unwrap().len(), 9);
    assert_eq!(store.versions().unwrap(), vec![10, 11, 12]);
}

#[test]
fn mapped_index_with_heap_tail_commits_and_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let store = SnapshotStore::open(dir.path()).unwrap();
    store.commit(&index(100, 0.0), "base").unwrap();
    let (_, mut loaded) = store.load_current(LoadOptions { verify_vectors: false }).unwrap();
    assert_eq!(loaded.mapped_vectors(), 100);
    for i in 100..130u64 {
        loaded.insert(i, &[i as f32, -1.0]).unwrap();
    }
    loaded.delete(5).unwrap();
    let info = store.commit(&loaded, "tail").unwrap();
    assert_eq!((info.version, info.parent, info.live), (2, Some(1), 129));
    let (_, v2) = store.load_current(LoadOptions::default()).unwrap();
    assert_eq!(v2.len(), 129);
    assert!(!v2.contains(5) && v2.contains(129));
    assert_eq!(v2.search(&[120.0, -1.0], 1).unwrap()[0].key, 120);
    // Vacuum turns the mapped store back into a compact heap store.
    let mut compacted = v2;
    compacted.vacuum();
    assert_eq!(compacted.mapped_vectors(), 0);
    store.commit(&compacted, "vacuumed").unwrap();
    store.rollback(1).unwrap();
    let (cur, v1) = store.load_current(LoadOptions::default()).unwrap();
    assert_eq!((cur, v1.len()), (1, 100));
    let labels: Vec<String> = store.list().unwrap().into_iter().map(|i| i.label).collect();
    assert_eq!(labels, ["base", "tail", "vacuumed"]);
}
