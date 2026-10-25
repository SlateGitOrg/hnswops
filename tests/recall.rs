//! Recall against exact ground truth on held-out queries.
//!
//! * The committed synthetic fixture always runs (no network).
//! * SIFT10K (ANN_SIFT10K, INRIA TEXMEX) runs when `make fetch-data` has
//!   placed it in `data/siftsmall` (or `HNSWOPS_SIFT_DIR`); checksums are
//!   verified before use. Otherwise the test prints why it skipped.

use std::path::PathBuf;

use hnswops::brute::recall_at_k;
use hnswops::dataset::{self, Dataset};
use hnswops::{Hnsw, HnswConfig, SearchParams};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn build(ds: &Dataset, m: usize, efc: usize) -> Hnsw {
    let mut idx = Hnsw::new(HnswConfig::new(ds.base.dim()).with_m(m).with_ef_construction(efc)).unwrap();
    for (i, v) in ds.base.iter().enumerate() {
        idx.insert(i as u64, v).unwrap();
    }
    idx
}

fn recall(idx: &Hnsw, ds: &Dataset, k: usize, ef: usize) -> f64 {
    let mut total = 0.0;
    for (qi, q) in ds.queries.iter().enumerate() {
        let found: Vec<u64> = idx
            .search_with(q, SearchParams::top(k).ef(ef))
            .unwrap()
            .hits
            .iter()
            .map(|h| h.key)
            .collect();
        let truth: Vec<u64> = ds.groundtruth[qi].iter().map(|&x| u64::from(x)).collect();
        total += recall_at_k(&found, &truth, k);
    }
    total / ds.queries.len() as f64
}

#[test]
fn fixture_ground_truth_matches_exact_search() {
    let ds = dataset::load_dataset(&root().join("fixtures"), &dataset::FIXTURE).unwrap();
    let idx = build(&ds, 8, 32);
    for (qi, q) in ds.queries.iter().enumerate().take(25) {
        let exact: Vec<u64> = idx.exact_search(q, 10, None).unwrap().iter().map(|h| h.key).collect();
        let truth: Vec<u64> = ds.groundtruth[qi][..10].iter().map(|&x| u64::from(x)).collect();
        assert_eq!(exact, truth, "query {qi}");
    }
}

#[test]
fn fixture_recall_at_10_meets_floor_and_grows_with_ef() {
    let ds = dataset::load_dataset(&root().join("fixtures"), &dataset::FIXTURE).unwrap();
    let idx = build(&ds, 16, 200);
    let curve: Vec<f64> = [10, 32, 64, 128].iter().map(|&ef| recall(&idx, &ds, 10, ef)).collect();
    eprintln!("fixture recall@10 at ef=10,32,64,128: {curve:?}");
    assert!(curve[2] >= 0.99, "recall@10 at ef=64 was {}", curve[2]);
    assert!(curve[3] >= 0.995, "recall@10 at ef=128 was {}", curve[3]);
    assert!(curve.windows(2).all(|w| w[1] + 1e-9 >= w[0]), "recall not monotone in ef: {curve:?}");
}

#[test]
fn fixture_recall_survives_heavy_deletion_and_vacuum() {
    let ds = dataset::load_dataset(&root().join("fixtures"), &dataset::FIXTURE).unwrap();
    let mut idx = build(&ds, 16, 100);
    for key in (0..ds.base.len() as u64).filter(|k| k % 10 < 4) {
        idx.delete(key).unwrap();
    }
    let live_recall = |idx: &Hnsw| {
        let mut total = 0.0;
        for q in ds.queries.iter() {
            let got: Vec<u64> = idx.search_with(q, SearchParams::top(10).ef(64)).unwrap().hits.iter().map(|h| h.key).collect();
            let truth: Vec<u64> = idx.exact_search(q, 10, None).unwrap().iter().map(|h| h.key).collect();
            total += recall_at_k(&got, &truth, 10);
        }
        total / ds.queries.len() as f64
    };
    let tombstoned = live_recall(&idx);
    idx.vacuum();
    let vacuumed = live_recall(&idx);
    eprintln!("after deleting 40%: recall {tombstoned:.4} (tombstones), {vacuumed:.4} (vacuumed)");
    assert!(tombstoned >= 0.98 && vacuumed >= 0.98);
    assert_eq!(idx.health().unreachable_l0, 0);
}

#[test]
fn sift10k_recall_at_10() {
    let dir = std::env::var_os("HNSWOPS_SIFT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root().join("data").join("siftsmall"));
    if !dir.join(dataset::SIFT_SMALL.base.name).exists() {
        eprintln!("SKIP sift10k_recall_at_10: {} not present (run `make fetch-data`)", dir.display());
        return;
    }
    let ds = dataset::load_dataset(&dir, &dataset::SIFT_SMALL).expect("pinned SIFT10K checksums");
    assert_eq!((ds.base.len(), ds.base.dim(), ds.queries.len()), (10_000, 128, 100));
    let idx = build(&ds, 16, 200);
    let r64 = recall(&idx, &ds, 10, 64);
    let r128 = recall(&idx, &ds, 10, 128);
    eprintln!("SIFT10K recall@10: ef=64 {r64:.4}, ef=128 {r128:.4}");
    assert!(r128 >= 0.99, "SIFT10K recall@10 at ef=128 was {r128}");
}
