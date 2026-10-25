//! Exact (brute-force) k-NN over a [`VecSet`] and recall computation.
//!
//! Used to produce ground truth for synthetic data and to score ANN results.

use std::collections::BinaryHeap;

use crate::dataset::VecSet;
use crate::distance::Metric;
use crate::error::Result;

#[derive(PartialEq)]
struct Cand(f32, u32);
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

/// Exact top-`k` row indices of `base` for every query, nearest first.
/// Work is split across `threads` scoped threads; cost O(|Q| · |B| · d).
pub fn exact_knn(
    base: &VecSet,
    queries: &VecSet,
    k: usize,
    metric: Metric,
    threads: usize,
) -> Result<Vec<Vec<u32>>> {
    let prep = |v: &[f32]| metric.prepare(v);
    let base_p: Vec<Vec<f32>> = base.iter().map(prep).collect::<Result<_>>()?;
    let queries_p: Vec<Vec<f32>> = queries.iter().map(prep).collect::<Result<_>>()?;
    let threads = threads.clamp(1, 64);
    let chunk = queries_p.len().div_ceil(threads).max(1);
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); queries_p.len()];
    std::thread::scope(|s| {
        for (qs, slots) in queries_p.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let base_p = &base_p;
            s.spawn(move || {
                for (q, slot) in qs.iter().zip(slots.iter_mut()) {
                    *slot = top_k(base_p, q, k, metric);
                }
            });
        }
    });
    Ok(out)
}

fn top_k(base: &[Vec<f32>], q: &[f32], k: usize, metric: Metric) -> Vec<u32> {
    let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
    for (i, b) in base.iter().enumerate() {
        let c = Cand(metric.distance(q, b), i as u32);
        if heap.len() < k {
            heap.push(c);
        } else if heap.peek().is_some_and(|w| c < *w) {
            heap.pop();
            heap.push(c);
        }
    }
    heap.into_sorted_vec().into_iter().map(|c| c.1).collect()
}

/// recall@k: fraction of the true `k` nearest neighbors present in the first
/// `k` returned ids. Returns 1.0 when `k == 0` or the truth is empty.
pub fn recall_at_k(found: &[u64], truth: &[u64], k: usize) -> f64 {
    let truth = &truth[..k.min(truth.len())];
    if truth.is_empty() {
        return 1.0;
    }
    let found = &found[..k.min(found.len())];
    let hits = truth.iter().filter(|t| found.contains(t)).count();
    hits as f64 / truth.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::gaussian_mixture;

    #[test]
    fn exact_knn_finds_self_first_and_is_thread_invariant() {
        let base = gaussian_mixture(300, 8, 4, 1.0, 5);
        let q = base.head(17);
        let one = exact_knn(&base, &q, 5, Metric::L2, 1).unwrap();
        let many = exact_knn(&base, &q, 5, Metric::L2, 4).unwrap();
        assert_eq!(one, many);
        for (i, row) in one.iter().enumerate() {
            assert_eq!(row[0], i as u32);
            assert_eq!(row.len(), 5);
        }
        let cos = exact_knn(&base, &q, 3, Metric::Cosine, 2).unwrap();
        assert_eq!(cos[0][0], 0);
    }

    #[test]
    fn recall_counts_overlap() {
        assert_eq!(recall_at_k(&[1, 2, 3], &[3, 2, 9], 3), 2.0 / 3.0);
        assert_eq!(recall_at_k(&[1], &[1, 2], 2), 0.5);
        assert_eq!(recall_at_k(&[], &[], 10), 1.0);
        assert_eq!(recall_at_k(&[5, 6], &[5, 7], 1), 1.0);
    }
}
