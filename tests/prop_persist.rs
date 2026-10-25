//! Property tests for persistence.
//!
//! * Round trip: any index (with tombstones) saved and loaded answers every
//!   query identically, and — because the level-sampling RNG state is
//!   persisted — keeps evolving *identically* to the original when both
//!   receive the same further inserts (the loaded one on top of an mmap).
//! * Integrity: flipping any byte of `graph.bin` or `vectors.bin`, or
//!   truncating either, makes `load` fail with an error instead of panicking
//!   or returning a silently corrupted index.

use std::fs;
use std::path::Path;

use hnswops::persist::{self, LoadOptions, GRAPH_FILE, VECTORS_FILE};
use hnswops::{Error, Hnsw, HnswConfig};
use proptest::prelude::*;

fn build(points: &[(u64, [i8; 3])], deletes: &[usize]) -> Hnsw {
    let mut idx = Hnsw::new(HnswConfig::new(3).with_m(4).with_ef_construction(16)).unwrap();
    for (k, v) in points {
        idx.insert_with_tags(*k, &v.map(f32::from), *k % 5).unwrap();
    }
    let keys: Vec<u64> = idx.keys().collect();
    for &d in deletes {
        if !keys.is_empty() {
            let _ = idx.delete(keys[d % keys.len()]);
        }
    }
    idx
}

fn queries() -> Vec<[f32; 3]> {
    (0..8).map(|i| [i as f32 * 4.0 - 16.0, (i * i) as f32 % 11.0, 2.0]).collect()
}

fn assert_same(a: &Hnsw, b: &Hnsw) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.tombstones(), b.tombstones());
    assert_eq!(a.health(), b.health());
    for k in a.keys() {
        assert_eq!(a.vector(k), b.vector(k));
        assert_eq!(a.tags(k), b.tags(k));
    }
    for q in queries() {
        assert_eq!(a.search(&q, 6).unwrap(), b.search(&q, 6).unwrap());
    }
}

fn points() -> impl Strategy<Value = Vec<(u64, [i8; 3])>> {
    prop::collection::vec((0u64..200, prop::array::uniform3(-30i8..30)), 1..120)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

    #[test]
    fn round_trip_and_lockstep_evolution(
        pts in points(),
        dels in prop::collection::vec(any::<usize>(), 0..40),
        more in prop::collection::vec((200u64..400, prop::array::uniform3(-30i8..30)), 0..40),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let mut original = build(&pts, &dels);
        let m = persist::save(&original, dir.path()).unwrap();
        prop_assert_eq!(m.live, original.len());
        let mut loaded = persist::load(dir.path(), LoadOptions::default()).unwrap();
        prop_assert_eq!(loaded.mapped_vectors(), m.slots);
        assert_same(&original, &loaded);

        // Same writes on both: the mmap'd + heap-tail index must stay identical.
        for (k, v) in &more {
            let v = v.map(f32::from);
            original.insert(*k, &v).unwrap();
            loaded.insert(*k, &v).unwrap();
        }
        assert_same(&original, &loaded);

        // Saving a mixed (mapped + tail) index and loading it again is exact too.
        let dir2 = tempfile::tempdir().unwrap();
        persist::save(&loaded, dir2.path()).unwrap();
        let reloaded = persist::load(dir2.path(), LoadOptions::default()).unwrap();
        assert_same(&original, &reloaded);
    }

    #[test]
    fn any_single_byte_corruption_is_detected(
        pts in points(),
        which in 0usize..2,
        pos in any::<usize>(),
        mask in 1u8..=255,
    ) {
        let dir = tempfile::tempdir().unwrap();
        persist::save(&build(&pts, &[]), dir.path()).unwrap();
        let file = dir.path().join(if which == 0 { GRAPH_FILE } else { VECTORS_FILE });
        let mut bytes = fs::read(&file).unwrap();
        let i = pos % bytes.len();
        bytes[i] ^= mask;
        fs::write(&file, &bytes).unwrap();
        let err = persist::load(dir.path(), LoadOptions::default()).unwrap_err();
        prop_assert!(matches!(err, Error::Corrupt { .. }), "{}", err);
    }

    #[test]
    fn truncation_is_detected(pts in points(), which in 0usize..2, cut in any::<usize>()) {
        let dir = tempfile::tempdir().unwrap();
        persist::save(&build(&pts, &[]), dir.path()).unwrap();
        let file = dir.path().join(if which == 0 { GRAPH_FILE } else { VECTORS_FILE });
        let bytes = fs::read(&file).unwrap();
        let keep = cut % bytes.len();
        fs::write(&file, &bytes[..keep]).unwrap();
        prop_assert!(persist::load(dir.path(), LoadOptions::default()).is_err());
    }
}

#[test]
fn unverified_load_skips_vector_checksum_but_still_validates_shape() {
    let dir = tempfile::tempdir().unwrap();
    let idx = build(&[(1, [1, 2, 3]), (2, [4, 5, 6])], &[]);
    persist::save(&idx, dir.path()).unwrap();
    let path = dir.path().join(VECTORS_FILE);
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x01;
    fs::write(&path, &bytes).unwrap();
    // Fast path: O(1) start-up, payload not hashed.
    assert!(persist::load(dir.path(), LoadOptions { verify_vectors: false }).is_ok());
    // Full verification catches it.
    assert!(persist::verify(dir.path()).is_err());
    fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
    assert!(persist::load(dir.path(), LoadOptions { verify_vectors: false }).is_err());
    assert!(!Path::new(&dir.path().join("nope")).exists());
}
