//! Model-based property tests for insert/upsert/delete/vacuum.
//!
//! A `BTreeMap` is the reference model. After every random operation sequence
//! the index must agree with it exactly on membership and stored vectors, never
//! return a deleted key, return results sorted and duplicate-free, answer
//! `exact_search` identically to a model brute force, and — after `vacuum` —
//! have no tombstones, no dangling edges and every live node reachable.

use std::collections::BTreeMap;

use hnswops::{Hnsw, HnswConfig, SearchParams};
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Upsert(u64, [i8; 4], u8),
    Delete(usize),
    Vacuum,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (0u64..48, prop::array::uniform4(-20i8..20), any::<u8>())
            .prop_map(|(k, v, t)| Op::Upsert(k, v, t)),
        3 => any::<usize>().prop_map(Op::Delete),
        1 => Just(Op::Vacuum),
    ]
}

fn vec_of(v: &[i8; 4]) -> Vec<f32> {
    v.iter().map(|&x| f32::from(x)).collect()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn model_topk(model: &BTreeMap<u64, (Vec<f32>, u64)>, q: &[f32], k: usize) -> Vec<f32> {
    let mut d: Vec<f32> = model.values().map(|(v, _)| l2(v, q)).collect();
    d.sort_by(f32::total_cmp);
    d.truncate(k);
    d
}

fn check_agreement(idx: &Hnsw, model: &BTreeMap<u64, (Vec<f32>, u64)>, queries: &[Vec<f32>]) {
    assert_eq!(idx.len(), model.len());
    let mut keys: Vec<u64> = idx.keys().collect();
    keys.sort_unstable();
    assert_eq!(keys, model.keys().copied().collect::<Vec<_>>());
    for (k, (v, t)) in model {
        assert_eq!(idx.vector(*k).unwrap(), v.as_slice());
        assert_eq!(idx.tags(*k), Some(*t));
    }
    for q in queries {
        let hits = idx.search(q, 5).unwrap();
        assert!(hits.len() <= model.len().min(5));
        let mut seen = std::collections::HashSet::new();
        for w in hits.windows(2) {
            assert!(w[0].distance <= w[1].distance, "unsorted results");
        }
        for h in &hits {
            assert!(
                model.contains_key(&h.key),
                "deleted/unknown key {} returned",
                h.key
            );
            assert!(seen.insert(h.key), "duplicate key {}", h.key);
            assert_eq!(h.distance, l2(&model[&h.key].0, q));
        }
        let exact: Vec<f32> = idx
            .exact_search(q, 5, None)
            .unwrap()
            .iter()
            .map(|h| h.distance)
            .collect();
        assert_eq!(exact, model_topk(model, q, 5));
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn index_agrees_with_model(ops in prop::collection::vec(op(), 1..160), m in 2usize..8) {
        let mut idx = Hnsw::new(HnswConfig::new(4).with_m(m).with_ef_construction(16).with_ef_search(16)).unwrap();
        let mut model: BTreeMap<u64, (Vec<f32>, u64)> = BTreeMap::new();
        let queries: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32 * 5.0 - 12.0, 3.0, -(i as f32), 1.0]).collect();
        for op in &ops {
            match op {
                Op::Upsert(k, v, t) => {
                    let v = vec_of(v);
                    idx.insert_with_tags(*k, &v, u64::from(*t)).unwrap();
                    model.insert(*k, (v, u64::from(*t)));
                }
                Op::Delete(i) => {
                    if model.is_empty() {
                        prop_assert!(idx.delete(*i as u64 % 48).is_err());
                        continue;
                    }
                    let key = *model.keys().nth(i % model.len()).unwrap();
                    idx.delete(key).unwrap();
                    model.remove(&key);
                    prop_assert!(idx.delete(key).is_err(), "double delete must fail");
                }
                Op::Vacuum => {
                    idx.vacuum();
                    let h = idx.health();
                    prop_assert_eq!(h.tombstones, 0);
                    prop_assert_eq!(h.dangling_edges, 0);
                    prop_assert_eq!(h.unreachable_l0, 0);
                }
            }
        }
        check_agreement(&idx, &model, &queries);
        idx.vacuum();
        check_agreement(&idx, &model, &queries);
        let h = idx.health();
        prop_assert_eq!((h.tombstones, h.dangling_edges, h.unreachable_l0), (0, 0, 0));
        // On a connected graph a beam as wide as the index is exhaustive.
        if !model.is_empty() {
            for q in &queries {
                let wide: Vec<f32> = idx
                    .search_with(q, SearchParams::top(5).ef(model.len().max(5)))
                    .unwrap()
                    .hits
                    .iter()
                    .map(|h| h.distance)
                    .collect();
                prop_assert_eq!(wide, model_topk(&model, q, 5));
            }
        }
    }

    #[test]
    fn deleted_keys_never_resurface_under_heavy_churn(seed in any::<u64>()) {
        let mut rng = hnswops::rng::SplitMix64::new(seed);
        let mut idx = Hnsw::new(HnswConfig::new(3).with_m(4).with_ef_construction(12)).unwrap();
        let mut live = std::collections::BTreeSet::new();
        for round in 0..400u64 {
            let key = rng.below(80);
            if live.contains(&key) && rng.below(2) == 0 {
                idx.delete(key).unwrap();
                live.remove(&key);
            } else {
                let v = [rng.below(9) as f32, rng.below(9) as f32, (round % 7) as f32];
                idx.insert(key, &v).unwrap();
                live.insert(key);
            }
        }
        for x in 0..9 {
            for h in idx.search(&[x as f32, 4.0, 3.0], 10).unwrap() {
                prop_assert!(live.contains(&h.key));
            }
        }
        prop_assert_eq!(idx.len(), live.len());
    }
}
