//! The HNSW graph: insertion, search and neighbor selection.
//!
//! Implements Malkov & Yashunin, *Efficient and robust approximate nearest
//! neighbor search using Hierarchical Navigable Small World graphs* (TPAMI 2018):
//! Algorithm 1 (insert), Algorithm 2 (search layer), Algorithm 4 (neighbor
//! selection heuristic) and Algorithm 5 (k-NN search). Deletion, repair and
//! compaction live in `maintenance.rs`; filtered search in `filter.rs`.

mod filter;
mod maintenance;

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use serde::{Deserialize, Serialize};

pub use filter::{Filter, FilterPlan};
pub use maintenance::{HealthReport, VacuumReport};

use crate::config::{HnswConfig, MAX_LEVEL};
use crate::error::{Error, Result};
use crate::rng::SplitMix64;
use crate::storage::VectorStore;
use crate::visited::{VisitedList, VisitedPool};

/// One search result.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    /// Caller-supplied key.
    pub key: u64,
    /// Distance under the index metric (smaller is closer).
    pub distance: f32,
}

/// How a query was answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Plain graph traversal.
    Graph,
    /// Graph traversal admitting only filter matches, with a widened beam.
    FilteredGraph,
    /// Exact scan of the matching set (filter too selective for the graph).
    BruteForce,
}

/// Per-query work counters, useful for capacity planning and for tests that
/// assert sub-linear behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchStats {
    /// Strategy the planner chose.
    pub strategy: Strategy,
    /// Beam width actually used on layer 0 (0 for brute force).
    pub ef_used: usize,
    /// Number of distance evaluations performed.
    pub distance_evals: usize,
}

/// Hits plus the work it took to find them.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Up to `k` hits, closest first.
    pub hits: Vec<Hit>,
    /// Work counters.
    pub stats: SearchStats,
}

/// Per-query options.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchParams<'a> {
    /// Number of neighbors to return.
    pub k: usize,
    /// Beam width; defaults to the index's `ef_search`. Raised to `k` if smaller.
    pub ef: Option<usize>,
    /// Only return points matching this filter.
    pub filter: Option<&'a Filter>,
}

impl<'a> SearchParams<'a> {
    /// Top-`k` with default beam width and no filter.
    pub fn top(k: usize) -> Self {
        Self {
            k,
            ef: None,
            filter: None,
        }
    }

    /// Override the beam width.
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    /// Restrict results to `filter` matches.
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }
}

/// A graph node. `links[l]` is the adjacency list on layer `l`; the node's
/// level is `links.len() - 1`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Node {
    pub(crate) key: u64,
    pub(crate) tags: u64,
    pub(crate) deleted: bool,
    pub(crate) links: Vec<Vec<u32>>,
}

impl Node {
    pub(crate) fn level(&self) -> usize {
        self.links.len() - 1
    }
}

/// (distance, node) ordered by distance with a total order on floats and the
/// node id as tie-breaker, so heaps behave deterministically.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Scored {
    pub(crate) dist: f32,
    pub(crate) id: u32,
}

impl Eq for Scored {}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Hierarchical Navigable Small World index.
///
/// `&Hnsw` is `Sync`: any number of threads may search concurrently. Mutation
/// needs `&mut Hnsw`; wrap it in a `RwLock` for a single-writer/multi-reader
/// service (see [`crate::monitor::MonitoredIndex`]).
#[derive(Debug)]
pub struct Hnsw {
    pub(crate) config: HnswConfig,
    pub(crate) vectors: VectorStore,
    pub(crate) nodes: Vec<Node>,
    pub(crate) key_to_id: HashMap<u64, u32>,
    pub(crate) entry_point: Option<u32>,
    pub(crate) max_level: usize,
    pub(crate) rng: SplitMix64,
    pub(crate) live: usize,
    visited: VisitedPool,
}

impl Hnsw {
    /// Create an empty index.
    pub fn new(config: HnswConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            vectors: VectorStore::new(config.dim),
            rng: SplitMix64::new(config.seed),
            config,
            nodes: Vec::new(),
            key_to_id: HashMap::new(),
            entry_point: None,
            max_level: 0,
            live: 0,
            visited: VisitedPool::default(),
        })
    }

    /// Reassemble an index from persisted parts, validating graph invariants.
    pub(crate) fn from_parts(
        config: HnswConfig,
        vectors: VectorStore,
        nodes: Vec<Node>,
        entry_point: Option<u32>,
        rng_state: u64,
    ) -> std::result::Result<Self, String> {
        config.validate().map_err(|e| e.to_string())?;
        if vectors.len() != nodes.len() {
            return Err(format!(
                "{} vectors but {} nodes",
                vectors.len(),
                nodes.len()
            ));
        }
        let n = nodes.len();
        let mut key_to_id = HashMap::with_capacity(n);
        let mut live = 0;
        for (i, node) in nodes.iter().enumerate() {
            if node.links.is_empty() || node.level() > MAX_LEVEL {
                return Err(format!("node {i} has invalid level"));
            }
            for (l, list) in node.links.iter().enumerate() {
                if list.len() > config.max_degree(l) {
                    return Err(format!("node {i} exceeds max degree on layer {l}"));
                }
                for &t in list {
                    let ok = (t as usize) < n && t as usize != i && nodes[t as usize].level() >= l;
                    if !ok {
                        return Err(format!("node {i} has invalid link {t} on layer {l}"));
                    }
                }
            }
            if !node.deleted {
                live += 1;
                if key_to_id.insert(node.key, i as u32).is_some() {
                    return Err(format!("duplicate live key {}", node.key));
                }
            }
        }
        let max_level = match entry_point {
            Some(ep) => {
                let node = nodes
                    .get(ep as usize)
                    .ok_or_else(|| format!("entry point {ep} out of range"))?;
                if node.deleted {
                    return Err("entry point is deleted".into());
                }
                node.level()
            }
            None if live > 0 => return Err("live nodes but no entry point".into()),
            None => 0,
        };
        Ok(Self {
            config,
            vectors,
            nodes,
            key_to_id,
            entry_point,
            max_level,
            rng: SplitMix64::new(rng_state),
            live,
            visited: VisitedPool::default(),
        })
    }

    /// Index parameters.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Change the default beam width used by [`Hnsw::search`].
    pub fn set_ef_search(&mut self, ef: usize) -> Result<()> {
        if ef == 0 {
            return Err(Error::InvalidConfig("ef_search must be >= 1".into()));
        }
        self.config.ef_search = ef;
        Ok(())
    }

    /// Number of live (searchable) points.
    pub fn len(&self) -> usize {
        self.live
    }

    /// True when no live points exist.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Number of tombstoned slots awaiting [`Hnsw::vacuum`].
    pub fn tombstones(&self) -> usize {
        self.nodes.len() - self.live
    }

    /// Vectors currently served from a memory-mapped snapshot file.
    pub fn mapped_vectors(&self) -> usize {
        self.vectors.mapped_count()
    }

    /// Whether `key` is live.
    pub fn contains(&self, key: u64) -> bool {
        self.key_to_id.contains_key(&key)
    }

    /// Stored (metric-prepared) vector of a live key.
    pub fn vector(&self, key: u64) -> Option<&[f32]> {
        self.key_to_id.get(&key).map(|&id| self.vectors.get(id))
    }

    /// Tag bitmask of a live key.
    pub fn tags(&self, key: u64) -> Option<u64> {
        self.key_to_id
            .get(&key)
            .map(|&id| self.nodes[id as usize].tags)
    }

    /// Live keys in internal order.
    pub fn keys(&self) -> impl Iterator<Item = u64> + '_ {
        self.nodes.iter().filter(|n| !n.deleted).map(|n| n.key)
    }

    /// Insert or replace `key` with no tags.
    pub fn insert(&mut self, key: u64, vector: &[f32]) -> Result<()> {
        self.insert_with_tags(key, vector, 0)
    }

    /// Insert or replace `key`. Replacing tombstones the old point (with local
    /// repair) and inserts the new one, so readers never see a half-updated node.
    pub fn insert_with_tags(&mut self, key: u64, vector: &[f32], tags: u64) -> Result<()> {
        self.check_dim(vector)?;
        let prepared = self.config.metric.prepare(vector)?;
        if self.nodes.len() >= (u32::MAX - 1) as usize {
            return Err(Error::Capacity((u32::MAX - 1) as usize));
        }
        if let Some(&old) = self.key_to_id.get(&key) {
            self.delete_slot(old);
        }
        let id = self.nodes.len() as u32;
        let level = self.random_level();
        self.vectors.push(&prepared);
        self.nodes.push(Node {
            key,
            tags,
            deleted: false,
            links: vec![Vec::new(); level + 1],
        });
        self.key_to_id.insert(key, id);
        self.live += 1;

        let Some(ep) = self.entry_point else {
            self.entry_point = Some(id);
            self.max_level = level;
            return Ok(());
        };

        let mut evals = 0usize;
        let mut cur = Scored {
            dist: self.dist_to(&prepared, ep),
            id: ep,
        };
        for l in (level + 1..=self.max_level).rev() {
            cur = self.greedy_closest(&prepared, cur, l, &mut evals);
        }
        let mut entry = vec![cur];
        let mut visited = self.visited.take(self.nodes.len());
        for l in (0..=level.min(self.max_level)).rev() {
            let found = {
                let nodes = &self.nodes;
                self.search_layer(
                    &prepared,
                    &entry,
                    self.config.ef_construction,
                    l,
                    &|n| !nodes[n as usize].deleted && n != id,
                    &mut visited,
                    &mut evals,
                )
            };
            visited.reset(self.nodes.len());
            let chosen = self.select_neighbors(&found, self.config.m);
            self.nodes[id as usize].links[l] = chosen.iter().map(|s| s.id).collect();
            for s in &chosen {
                self.add_link(s.id, id, l);
            }
            if !found.is_empty() {
                entry = found;
            }
        }
        self.visited.give(visited);
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id);
        }
        Ok(())
    }

    /// k-NN search with the default beam width.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>> {
        Ok(self.search_with(query, SearchParams::top(k))?.hits)
    }

    /// k-NN search with explicit parameters and work statistics.
    pub fn search_with(&self, query: &[f32], params: SearchParams<'_>) -> Result<SearchResult> {
        self.check_dim(query)?;
        let q = self.config.metric.prepare(query)?;
        let ef = params.ef.unwrap_or(self.config.ef_search).max(params.k);
        match params.filter {
            None => Ok(self.graph_search(&q, params.k, ef, None, Strategy::Graph)),
            Some(f) => Ok(self.filtered_search(&q, params.k, ef, f)),
        }
    }

    /// Exact k-NN over live points (optionally filtered) by linear scan.
    /// This is the ground truth the recall monitor compares against.
    pub fn exact_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Hit>> {
        self.check_dim(query)?;
        let q = self.config.metric.prepare(query)?;
        Ok(self.brute_force(&q, k, filter).hits)
    }

    fn check_dim(&self, v: &[f32]) -> Result<()> {
        if v.len() != self.config.dim {
            return Err(Error::DimensionMismatch {
                expected: self.config.dim,
                got: v.len(),
            });
        }
        Ok(())
    }

    /// Algorithm 5: greedy descent through the upper layers, then a beam search
    /// of width `ef` on layer 0 admitting only nodes `accept` approves.
    pub(crate) fn graph_search(
        &self,
        q: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&Filter>,
        strategy: Strategy,
    ) -> SearchResult {
        let mut evals = 0usize;
        let mut stats = SearchStats {
            strategy,
            ef_used: ef,
            distance_evals: 0,
        };
        let Some(ep) = self.entry_point else {
            return SearchResult {
                hits: Vec::new(),
                stats,
            };
        };
        if k == 0 {
            return SearchResult {
                hits: Vec::new(),
                stats,
            };
        }
        evals += 1;
        let mut cur = Scored {
            dist: self.dist_to(q, ep),
            id: ep,
        };
        for l in (1..=self.max_level).rev() {
            cur = self.greedy_closest(q, cur, l, &mut evals);
        }
        let nodes = &self.nodes;
        let accept = |n: u32| {
            let node = &nodes[n as usize];
            !node.deleted && filter.is_none_or(|f| f.matches(node.key, node.tags))
        };
        let mut visited = self.visited.take(self.nodes.len());
        let found = self.search_layer(q, &[cur], ef, 0, &accept, &mut visited, &mut evals);
        self.visited.give(visited);
        stats.distance_evals = evals;
        SearchResult {
            hits: found
                .into_iter()
                .take(k)
                .map(|s| Hit {
                    key: self.nodes[s.id as usize].key,
                    distance: s.dist,
                })
                .collect(),
            stats,
        }
    }

    /// Exact scan of live (and filter-matching) nodes.
    pub(crate) fn brute_force(&self, q: &[f32], k: usize, filter: Option<&Filter>) -> SearchResult {
        let mut heap: BinaryHeap<Scored> = BinaryHeap::with_capacity(k + 1);
        let mut evals = 0usize;
        if k > 0 {
            for (i, node) in self.nodes.iter().enumerate() {
                if node.deleted || !filter.is_none_or(|f| f.matches(node.key, node.tags)) {
                    continue;
                }
                evals += 1;
                let s = Scored {
                    dist: self.dist_to(q, i as u32),
                    id: i as u32,
                };
                if heap.len() < k {
                    heap.push(s);
                } else if heap.peek().is_some_and(|w| s < *w) {
                    heap.pop();
                    heap.push(s);
                }
            }
        }
        SearchResult {
            hits: heap
                .into_sorted_vec()
                .into_iter()
                .map(|s| Hit {
                    key: self.nodes[s.id as usize].key,
                    distance: s.dist,
                })
                .collect(),
            stats: SearchStats {
                strategy: Strategy::BruteForce,
                ef_used: 0,
                distance_evals: evals,
            },
        }
    }

    /// Algorithm 2: beam search on one layer. Every reached node is used for
    /// navigation, but only `accept`ed nodes enter the result set — this is
    /// what lets tombstoned and filtered-out nodes keep the graph connected.
    /// Returns up to `ef` accepted nodes sorted by ascending distance.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn search_layer(
        &self,
        q: &[f32],
        entry: &[Scored],
        ef: usize,
        level: usize,
        accept: &dyn Fn(u32) -> bool,
        visited: &mut VisitedList,
        evals: &mut usize,
    ) -> Vec<Scored> {
        let mut candidates: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        let mut results: BinaryHeap<Scored> = BinaryHeap::with_capacity(ef + 1);
        for &e in entry {
            if visited.insert(e.id) {
                candidates.push(Reverse(e));
                if accept(e.id) {
                    results.push(e);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        while let Some(Reverse(c)) = candidates.pop() {
            if results.len() >= ef && results.peek().is_some_and(|w| c.dist > w.dist) {
                break;
            }
            for &n in &self.nodes[c.id as usize].links[level] {
                if !visited.insert(n) {
                    continue;
                }
                *evals += 1;
                let s = Scored {
                    dist: self.dist_to(q, n),
                    id: n,
                };
                let bound_ok =
                    results.len() < ef || results.peek().is_some_and(|w| s.dist < w.dist);
                if bound_ok {
                    candidates.push(Reverse(s));
                    if accept(n) {
                        results.push(s);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        results.into_sorted_vec()
    }

    /// Greedy walk (beam width 1) on an upper layer.
    pub(crate) fn greedy_closest(
        &self,
        q: &[f32],
        start: Scored,
        level: usize,
        evals: &mut usize,
    ) -> Scored {
        let mut cur = start;
        loop {
            let mut improved = false;
            for &n in &self.nodes[cur.id as usize].links[level] {
                *evals += 1;
                let d = self.dist_to(q, n);
                if d < cur.dist {
                    cur = Scored { dist: d, id: n };
                    improved = true;
                }
            }
            if !improved {
                return cur;
            }
        }
    }

    /// Algorithm 4 (without candidate extension): walk candidates nearest-first
    /// and keep one only if it is closer to the base than to every neighbor
    /// already kept. This prunes redundant edges inside a cluster and keeps
    /// long-range edges between clusters, which is what makes HNSW robust on
    /// clustered data. `candidates` must be sorted ascending.
    pub(crate) fn select_neighbors(&self, candidates: &[Scored], m: usize) -> Vec<Scored> {
        let mut kept: Vec<Scored> = Vec::with_capacity(m);
        for &c in candidates {
            if kept.len() >= m {
                break;
            }
            let diverse = kept.iter().all(|k| self.dist_between(c.id, k.id) >= c.dist);
            if diverse {
                kept.push(c);
            }
        }
        kept
    }

    /// Add edge `from -> to` on `level`, shrinking `from`'s list with the
    /// heuristic if it overflows.
    pub(crate) fn add_link(&mut self, from: u32, to: u32, level: usize) {
        let cap = self.config.max_degree(level);
        let list = &self.nodes[from as usize].links[level];
        if list.contains(&to) {
            return;
        }
        if list.len() < cap {
            self.nodes[from as usize].links[level].push(to);
            return;
        }
        let mut cands: Vec<Scored> = list
            .iter()
            .chain(std::iter::once(&to))
            .map(|&n| Scored {
                dist: self.dist_between(from, n),
                id: n,
            })
            .collect();
        cands.sort_unstable();
        let kept = self.select_neighbors(&cands, cap);
        self.nodes[from as usize].links[level] = kept.into_iter().map(|s| s.id).collect();
    }

    #[inline]
    pub(crate) fn dist_to(&self, q: &[f32], id: u32) -> f32 {
        self.config.metric.distance(q, self.vectors.get(id))
    }

    #[inline]
    pub(crate) fn dist_between(&self, a: u32, b: u32) -> f32 {
        self.config
            .metric
            .distance(self.vectors.get(a), self.vectors.get(b))
    }

    /// Draw a level from the exponentially decaying distribution
    /// `floor(-ln(U) * mL)`, capped at [`MAX_LEVEL`].
    fn random_level(&mut self) -> usize {
        let u = self.rng.next_f64_open();
        let level = (-u.ln() * self.config.level_multiplier()).floor() as usize;
        level.min(MAX_LEVEL)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::distance::Metric;

    pub(crate) fn grid_index(n: usize, m: usize) -> Hnsw {
        let mut idx = Hnsw::new(
            HnswConfig::new(2)
                .with_m(m)
                .with_ef_construction(32)
                .with_ef_search(32),
        )
        .unwrap();
        for i in 0..n {
            let (x, y) = ((i % 10) as f32, (i / 10) as f32);
            idx.insert(i as u64, &[x, y]).unwrap();
        }
        idx
    }

    #[test]
    fn empty_index_returns_nothing() {
        let idx = Hnsw::new(HnswConfig::new(2)).unwrap();
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0, 0.0], 5).unwrap().is_empty());
        assert!(idx.exact_search(&[0.0, 0.0], 5, None).unwrap().is_empty());
    }

    #[test]
    fn finds_exact_neighbors_on_small_grid() {
        let idx = grid_index(100, 8);
        assert_eq!(idx.len(), 100);
        let hits = idx.search(&[3.1, 4.0], 3).unwrap();
        assert_eq!(hits[0].key, 43);
        let exact = idx.exact_search(&[3.1, 4.0], 3, None).unwrap();
        assert_eq!(hits, exact);
        assert!(idx.search(&[1.0, 1.0], 0).unwrap().is_empty());
    }

    #[test]
    fn rejects_wrong_dimension_and_nan() {
        let mut idx = grid_index(4, 4);
        assert!(matches!(
            idx.insert(1, &[1.0]),
            Err(Error::DimensionMismatch {
                expected: 2,
                got: 1
            })
        ));
        assert!(idx.insert(9, &[f32::INFINITY, 0.0]).is_err());
        assert!(idx.search(&[1.0, 2.0, 3.0], 1).is_err());
        assert!(idx.set_ef_search(0).is_err());
        idx.set_ef_search(5).unwrap();
        assert_eq!(idx.config().ef_search, 5);
    }

    #[test]
    fn upsert_replaces_vector_and_tags() {
        let mut idx = grid_index(20, 4);
        idx.insert_with_tags(5, &[100.0, 100.0], 0b10).unwrap();
        assert_eq!(idx.len(), 20);
        assert_eq!(idx.tombstones(), 1);
        assert_eq!(idx.vector(5).unwrap(), &[100.0, 100.0]);
        assert_eq!(idx.tags(5), Some(0b10));
        let hits = idx.search(&[99.0, 99.0], 1).unwrap();
        assert_eq!(hits[0].key, 5);
        assert!(idx.keys().any(|k| k == 5));
    }

    #[test]
    fn degrees_respect_limits() {
        let idx = grid_index(100, 4);
        for node in &idx.nodes {
            for (l, list) in node.links.iter().enumerate() {
                assert!(list.len() <= idx.config.max_degree(l));
                assert!(!list.is_empty() || idx.nodes.len() == 1 || l > 0 || node.level() > 0);
            }
        }
    }

    #[test]
    fn cosine_and_inner_product_metrics_rank_correctly() {
        let mut cos = Hnsw::new(HnswConfig::new(2).with_metric(Metric::Cosine)).unwrap();
        let mut ip = Hnsw::new(HnswConfig::new(2).with_metric(Metric::InnerProduct)).unwrap();
        for (k, v) in [(1u64, [1.0f32, 0.0]), (2, [0.0, 1.0]), (3, [10.0, 9.0])] {
            cos.insert(k, &v).unwrap();
            ip.insert(k, &v).unwrap();
        }
        assert_eq!(cos.search(&[5.0, 0.1], 1).unwrap()[0].key, 1);
        assert_eq!(ip.search(&[1.0, 0.0], 1).unwrap()[0].key, 3);
    }

    #[test]
    fn search_stats_report_strategy_and_work() {
        let idx = grid_index(100, 8);
        let r = idx
            .search_with(&[1.0, 1.0], SearchParams::top(5).ef(10))
            .unwrap();
        assert_eq!(r.stats.strategy, Strategy::Graph);
        assert_eq!(r.stats.ef_used, 10);
        assert!(r.stats.distance_evals > 0 && r.stats.distance_evals < 100);
    }

    #[test]
    fn from_parts_rejects_broken_graphs() {
        let idx = grid_index(10, 4);
        let cfg = idx.config.clone();
        let mut nodes = idx.nodes.clone();
        nodes[0].links[0].push(99);
        let store = idx.vectors.compacted(&(0..10).collect::<Vec<_>>());
        assert!(Hnsw::from_parts(cfg.clone(), store, nodes, idx.entry_point, 1).is_err());
        let store = idx.vectors.compacted(&(0..10).collect::<Vec<_>>());
        assert!(Hnsw::from_parts(cfg.clone(), store, idx.nodes.clone(), None, 1).is_err());
        let store = idx.vectors.compacted(&(0..10).collect::<Vec<_>>());
        let mut dup = idx.nodes.clone();
        dup[1].key = dup[0].key;
        assert!(Hnsw::from_parts(cfg.clone(), store, dup, idx.entry_point, 1).is_err());
        let store = idx.vectors.compacted(&(0..9).collect::<Vec<_>>());
        assert!(Hnsw::from_parts(cfg, store, idx.nodes.clone(), idx.entry_point, 1).is_err());
    }
}
