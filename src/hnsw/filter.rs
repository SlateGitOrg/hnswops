//! Filtered search: metadata predicates plus a cost-based planner.
//!
//! Each point carries a 64-bit tag bitmask (64 categories) and its key. A
//! [`Filter`] is a boolean expression over those. Two execution strategies:
//!
//! * **Filtered graph walk** — traverse the graph normally but admit only
//!   matches into the result set. Cheap when matches are common, but when they
//!   are rare the beam must widen roughly by `1 / selectivity` to fill `k`
//!   results, and in the limit it degenerates into visiting the whole graph.
//! * **Brute force over matches** — scan every node, evaluate the predicate and
//!   compute distances only for matches. Exact, and cheap when matches are rare.
//!
//! The planner estimates selectivity from a deterministic strided sample of up
//! to [`SELECTIVITY_SAMPLE`] nodes, prices both strategies in distance
//! evaluations and picks the cheaper one; below the configured
//! `filter_brute_force_below` selectivity it always scans.

use serde::{Deserialize, Serialize};

use super::{Hnsw, SearchResult, Strategy};
use crate::error::{Error, Result};

/// Nodes sampled to estimate filter selectivity.
pub const SELECTIVITY_SAMPLE: usize = 512;

/// Boolean predicate over a point's key and tag bitmask.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    /// At least one of these tag bits is set.
    TagsAny(u64),
    /// All of these tag bits are set.
    TagsAll(u64),
    /// Key in the half-open range `start..end`.
    KeyRange {
        /// Inclusive lower bound.
        start: u64,
        /// Exclusive upper bound.
        end: u64,
    },
    /// Negation.
    Not(Box<Filter>),
    /// Conjunction (empty = true).
    And(Vec<Filter>),
    /// Disjunction (empty = false).
    Or(Vec<Filter>),
}

impl Filter {
    /// Evaluate the predicate.
    pub fn matches(&self, key: u64, tags: u64) -> bool {
        match self {
            Filter::TagsAny(mask) => tags & mask != 0,
            Filter::TagsAll(mask) => tags & mask == *mask,
            Filter::KeyRange { start, end } => (*start..*end).contains(&key),
            Filter::Not(inner) => !inner.matches(key, tags),
            Filter::And(fs) => fs.iter().all(|f| f.matches(key, tags)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(key, tags)),
        }
    }

    /// Parse the compact CLI syntax: comma-separated clauses that are ANDed,
    /// each one of `tags_any=<mask>`, `tags_all=<mask>`, `key=<start>..<end>`,
    /// optionally prefixed by `!` for negation. Masks accept decimal or `0x` hex.
    ///
    /// ```
    /// use hnswops::Filter;
    /// let f = Filter::parse("tags_any=0x3,!key=0..10").unwrap();
    /// assert!(f.matches(42, 0b01));
    /// assert!(!f.matches(5, 0b01));
    /// ```
    pub fn parse(s: &str) -> Result<Filter> {
        let mut clauses = Vec::new();
        for raw in s.split(',').map(str::trim).filter(|c| !c.is_empty()) {
            let (negate, clause) = match raw.strip_prefix('!') {
                Some(rest) => (true, rest),
                None => (false, raw),
            };
            let (name, value) = clause
                .split_once('=')
                .ok_or_else(|| bad(raw, "expected name=value"))?;
            let f = match name.trim() {
                "tags_any" => Filter::TagsAny(parse_u64(value).ok_or_else(|| bad(raw, "mask"))?),
                "tags_all" => Filter::TagsAll(parse_u64(value).ok_or_else(|| bad(raw, "mask"))?),
                "key" => {
                    let (a, b) = value
                        .split_once("..")
                        .ok_or_else(|| bad(raw, "expected start..end"))?;
                    let start = parse_u64(a).ok_or_else(|| bad(raw, "range start"))?;
                    let end = parse_u64(b).ok_or_else(|| bad(raw, "range end"))?;
                    Filter::KeyRange { start, end }
                }
                other => return Err(bad(raw, &format!("unknown field '{other}'"))),
            };
            clauses.push(if negate { Filter::Not(Box::new(f)) } else { f });
        }
        match clauses.len() {
            0 => Err(bad(s, "empty filter")),
            1 => Ok(clauses.remove(0)),
            _ => Ok(Filter::And(clauses)),
        }
    }
}

fn parse_u64(s: &str) -> Option<u64> {
    let s = s.trim();
    match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => s.parse().ok(),
    }
}

fn bad(clause: &str, why: &str) -> Error {
    Error::InvalidConfig(format!("bad filter clause '{clause}': {why}"))
}

impl Hnsw {
    /// Estimated fraction of live points matching `filter`, from a strided
    /// sample of at most [`SELECTIVITY_SAMPLE`] live nodes. Deterministic.
    pub fn estimate_selectivity(&self, filter: &Filter) -> f64 {
        if self.live == 0 {
            return 0.0;
        }
        let stride = (self.nodes.len() / SELECTIVITY_SAMPLE).max(1);
        let (mut seen, mut hit) = (0usize, 0usize);
        for n in self.nodes.iter().step_by(stride).filter(|n| !n.deleted) {
            seen += 1;
            if filter.matches(n.key, n.tags) {
                hit += 1;
            }
        }
        if seen == 0 {
            // Every sampled slot was a tombstone; fall back to a full count.
            let live_hits = self
                .nodes
                .iter()
                .filter(|n| !n.deleted && filter.matches(n.key, n.tags))
                .count();
            return live_hits as f64 / self.live as f64;
        }
        hit as f64 / seen as f64
    }

    /// Choose a strategy for `filter` at beam width `ef` (see [`FilterPlan`]).
    pub fn plan_filter(&self, filter: &Filter, ef: usize) -> FilterPlan {
        let selectivity = self.estimate_selectivity(filter);
        let live = self.live as f64;
        let expected_matches = selectivity * live;
        // Widen the beam so that ~ef matches are expected among admitted nodes.
        let widened = if selectivity > 0.0 {
            ((ef as f64 / selectivity).ceil() as usize).clamp(ef, self.live.max(ef))
        } else {
            self.live.max(ef)
        };
        // Cost in distance evaluations. A beam of width W expands about W
        // nodes, each costing ~M neighbor distances; the scan pays one
        // distance per match plus a predicate per live node (~1/32 of a
        // distance for typical dimensions).
        let graph_cost = widened as f64 * self.config.m as f64;
        let scan_cost = expected_matches + live / SCAN_PREDICATE_RATIO;
        let strategy = if selectivity < self.config.filter_brute_force_below
            || expected_matches <= ef as f64
            || scan_cost <= graph_cost
        {
            Strategy::BruteForce
        } else {
            Strategy::FilteredGraph
        };
        FilterPlan {
            selectivity,
            widened_ef: widened,
            graph_cost,
            scan_cost,
            strategy,
        }
    }

    /// Plan and run a filtered query on a prepared vector.
    pub(super) fn filtered_search(
        &self,
        q: &[f32],
        k: usize,
        ef: usize,
        filter: &Filter,
    ) -> SearchResult {
        let plan = self.plan_filter(filter, ef);
        match plan.strategy {
            Strategy::BruteForce => self.brute_force(q, k, Some(filter)),
            _ => self.graph_search(q, k, plan.widened_ef, Some(filter), Strategy::FilteredGraph),
        }
    }
}

/// How many predicate evaluations cost as much as one distance evaluation
/// in the planner's cost model.
const SCAN_PREDICATE_RATIO: f64 = 32.0;

/// The planner's decision for one filtered query, exposed for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FilterPlan {
    /// Estimated fraction of live points matching.
    pub selectivity: f64,
    /// Beam width a filtered graph walk would use.
    pub widened_ef: usize,
    /// Estimated distance evaluations of the filtered graph walk.
    pub graph_cost: f64,
    /// Estimated cost of scanning the matching set exactly.
    pub scan_cost: f64,
    /// Chosen strategy.
    pub strategy: Strategy,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HnswConfig;
    use crate::hnsw::SearchParams;

    fn tagged_index(n: usize) -> Hnsw {
        let mut idx = Hnsw::new(HnswConfig::new(2).with_m(6).with_ef_construction(40)).unwrap();
        for i in 0..n {
            let tags = if i % 50 == 0 { 0b100 } else { 1 << (i % 2) };
            idx.insert_with_tags(i as u64, &[(i % 40) as f32, (i / 40) as f32], tags)
                .unwrap();
        }
        idx
    }

    #[test]
    fn predicate_semantics() {
        assert!(Filter::TagsAny(0b11).matches(0, 0b10));
        assert!(!Filter::TagsAll(0b11).matches(0, 0b10));
        assert!(Filter::KeyRange { start: 5, end: 6 }.matches(5, 0));
        assert!(!Filter::KeyRange { start: 5, end: 6 }.matches(6, 0));
        assert!(Filter::Not(Box::new(Filter::TagsAny(1))).matches(0, 2));
        assert!(Filter::And(vec![]).matches(0, 0));
        assert!(!Filter::Or(vec![]).matches(0, 0));
    }

    #[test]
    fn parser_accepts_and_rejects() {
        assert_eq!(Filter::parse("tags_all=5").unwrap(), Filter::TagsAll(5));
        assert_eq!(
            Filter::parse("key=1..0x10").unwrap(),
            Filter::KeyRange { start: 1, end: 16 }
        );
        assert!(matches!(
            Filter::parse("tags_any=1, !tags_all=2").unwrap(),
            Filter::And(v) if v.len() == 2
        ));
        for bad in ["", "tags_any", "color=red", "key=5", "tags_any=zz", "key=a..2", "key=1..b", "tags_all=q"] {
            assert!(Filter::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn common_filter_uses_graph_and_rare_filter_uses_scan() {
        let idx = tagged_index(2000);
        let common = Filter::TagsAny(0b01);
        let rare = Filter::TagsAny(0b100);
        let sel = idx.estimate_selectivity(&common);
        assert!((0.4..0.6).contains(&sel), "selectivity {sel}");
        let q = [10.0, 10.0];
        let r = idx.search_with(&q, SearchParams::top(10).filter(&common)).unwrap();
        assert_eq!(r.stats.strategy, Strategy::FilteredGraph);
        assert!(r.stats.ef_used >= 64);
        let exact = idx.exact_search(&q, 10, Some(&common)).unwrap();
        let overlap = r.hits.iter().filter(|h| exact.iter().any(|e| e.key == h.key)).count();
        assert!(overlap >= 9, "overlap {overlap}");
        assert!(r.hits.iter().all(|h| idx.tags(h.key).unwrap() & 1 != 0));

        let r = idx.search_with(&q, SearchParams::top(10).filter(&rare)).unwrap();
        assert_eq!(r.stats.strategy, Strategy::BruteForce);
        assert_eq!(r.hits, idx.exact_search(&q, 10, Some(&rare)).unwrap());

        let plan = idx.plan_filter(&common, 64);
        assert!(plan.graph_cost < plan.scan_cost, "{plan:?}");
        // A filter matching nothing plans a scan over the full beam.
        let none = idx.plan_filter(&Filter::TagsAny(1 << 40), 64);
        assert_eq!((none.strategy, none.widened_ef), (Strategy::BruteForce, 2000));
        // On a tiny index the scan is cheaper even for common filters.
        let small = tagged_index(300);
        assert_eq!(small.plan_filter(&common, 64).strategy, Strategy::BruteForce);
    }

    #[test]
    fn selectivity_falls_back_when_sample_hits_only_tombstones() {
        let mut idx = Hnsw::new(HnswConfig::new(1).with_m(4).with_ef_construction(8)).unwrap();
        for i in 0..1024u64 {
            idx.insert_with_tags(i, &[i as f32], i & 1).unwrap();
        }
        // Stride is 2, so the sample sees only even slots: delete them all.
        for k in (0..1024).step_by(2) {
            idx.delete(k).unwrap();
        }
        assert_eq!(idx.estimate_selectivity(&Filter::TagsAny(1)), 1.0);
        for k in (1..1024).step_by(2) {
            idx.delete(k).unwrap();
        }
        assert_eq!(idx.estimate_selectivity(&Filter::TagsAny(1)), 0.0);
    }
}
