//! Deletion (tombstones + local repair), compaction and graph health checks.
//!
//! Deleting from a proximity graph is the hard part of HNSW operations: simply
//! removing a node can disconnect regions that were only reachable through it.
//! The strategy here has three tiers:
//!
//! 1. **Tombstone** (O(1)): the node stops appearing in results immediately but
//!    keeps its edges, so searches can still route *through* it.
//! 2. **Local repair** (O(deg² · d) per layer): every live out-neighbor `u` that
//!    links back to the deleted node replaces that edge using the deleted node's
//!    own neighbors as candidates, re-running the diversity heuristic.
//! 3. **Vacuum** (O(n · M · d)): drops every remaining edge into a tombstone,
//!    compacts storage (renumbering ids), then verifies reachability from the
//!    entry point on every layer and reconnects any orphaned node.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use super::{Hnsw, Node, Scored};
use crate::error::{Error, Result};

/// Outcome of [`Hnsw::vacuum`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VacuumReport {
    /// Tombstoned slots physically removed.
    pub removed: usize,
    /// Adjacency lists rewritten because they pointed at tombstones.
    pub lists_repaired: usize,
    /// Nodes that were unreachable from the entry point and were re-linked.
    pub reconnected: usize,
}

/// Structural health of the graph, for dashboards and alerts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthReport {
    /// Live points.
    pub live: usize,
    /// Tombstoned slots.
    pub tombstones: usize,
    /// Highest layer in use.
    pub max_level: usize,
    /// Number of nodes per level (index = level).
    pub level_histogram: Vec<usize>,
    /// Mean out-degree of live nodes on layer 0.
    pub mean_degree_l0: f64,
    /// Edges from live nodes into tombstones.
    pub dangling_edges: usize,
    /// Live nodes not reachable from the entry point on layer 0, even when
    /// routing through tombstones. Non-zero means some points are unfindable.
    pub unreachable_l0: usize,
}

impl Hnsw {
    /// Remove `key` from the index (tombstone + local repair).
    pub fn delete(&mut self, key: u64) -> Result<()> {
        let id = *self.key_to_id.get(&key).ok_or(Error::KeyNotFound(key))?;
        self.delete_slot(id);
        Ok(())
    }

    /// Tombstone internal slot `id` and repair its neighborhood.
    pub(crate) fn delete_slot(&mut self, id: u32) {
        let node = &mut self.nodes[id as usize];
        if node.deleted {
            return;
        }
        node.deleted = true;
        let key = node.key;
        if self.key_to_id.get(&key) == Some(&id) {
            self.key_to_id.remove(&key);
        }
        self.live -= 1;
        self.repair_around(id);
        if self.entry_point == Some(id) {
            self.reelect_entry_point();
        }
    }

    /// Local repair after deleting `dead`: each live out-neighbor that points
    /// back at `dead` drops that edge and chooses replacements among `dead`'s
    /// neighbors, so the region stays navigable without a global rebuild.
    fn repair_around(&mut self, dead: u32) {
        let levels = self.nodes[dead as usize].links.len();
        for l in 0..levels {
            let dead_links = self.nodes[dead as usize].links[l].clone();
            for &u in &dead_links {
                let un = &self.nodes[u as usize];
                if un.deleted || !un.links[l].contains(&dead) {
                    continue;
                }
                let original_len = un.links[l].len();
                let mut cand_ids: Vec<u32> =
                    un.links[l].iter().copied().filter(|&x| x != dead).collect();
                for &x in &dead_links {
                    if x != u && !self.nodes[x as usize].deleted && !cand_ids.contains(&x) {
                        cand_ids.push(x);
                    }
                }
                let list = self.rebuild_list(u, &cand_ids, l, original_len);
                self.nodes[u as usize].links[l] = list;
            }
        }
    }

    /// Choose `u`'s adjacency on `level` from `cand_ids`: the diversity
    /// heuristic first, then nearest leftovers until the list is at least
    /// `min_len` long (capped at the layer's max degree). Back-filling keeps
    /// in-degrees from eroding across many deletions.
    fn rebuild_list(&self, u: u32, cand_ids: &[u32], level: usize, min_len: usize) -> Vec<u32> {
        let cap = self.config.max_degree(level);
        let mut cands: Vec<Scored> = cand_ids
            .iter()
            .map(|&x| Scored {
                dist: self.dist_between(u, x),
                id: x,
            })
            .collect();
        cands.sort_unstable();
        let mut kept: Vec<u32> = self
            .select_neighbors(&cands, cap)
            .into_iter()
            .map(|s| s.id)
            .collect();
        let target = min_len.min(cap).min(cands.len());
        for c in &cands {
            if kept.len() >= target {
                break;
            }
            if !kept.contains(&c.id) {
                kept.push(c.id);
            }
        }
        kept
    }

    /// Point the entry at the live node with the highest level (ties: lowest id).
    fn reelect_entry_point(&mut self) {
        let best = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| !n.deleted)
            .max_by(|(ia, a), (ib, b)| a.level().cmp(&b.level()).then(ib.cmp(ia)))
            .map(|(i, n)| (i as u32, n.level()));
        match best {
            Some((id, level)) => {
                self.entry_point = Some(id);
                self.max_level = level;
            }
            None => {
                self.entry_point = None;
                self.max_level = 0;
            }
        }
    }

    /// Physically remove tombstones, renumber ids and guarantee every live node
    /// is reachable from the entry point on every layer it belongs to.
    pub fn vacuum(&mut self) -> VacuumReport {
        let lists_repaired = self.drop_dangling_edges();
        let removed = self.nodes.len() - self.live;
        self.compact();
        let reconnected = self.ensure_connectivity();
        tracing::debug!(removed, lists_repaired, reconnected, "vacuum complete");
        VacuumReport {
            removed,
            lists_repaired,
            reconnected,
        }
    }

    /// Replace every edge into a tombstone with the tombstone's live
    /// neighborhood (following chains of tombstones), re-running the heuristic.
    fn drop_dangling_edges(&mut self) -> usize {
        let mut repaired = 0;
        for u in 0..self.nodes.len() as u32 {
            if self.nodes[u as usize].deleted {
                continue;
            }
            for l in 0..self.nodes[u as usize].links.len() {
                let list = &self.nodes[u as usize].links[l];
                if list.iter().all(|&x| !self.nodes[x as usize].deleted) {
                    continue;
                }
                let original_len = list.len();
                let mut cand_ids: Vec<u32> = Vec::new();
                let mut stack: Vec<u32> = Vec::new();
                let mut seen: Vec<u32> = vec![u];
                for &x in list {
                    if self.nodes[x as usize].deleted {
                        stack.push(x);
                    } else if !cand_ids.contains(&x) {
                        cand_ids.push(x);
                    }
                    seen.push(x);
                }
                // Walk through tombstones (bounded) to find live replacements.
                let mut budget = 4 * self.config.max_degree(l);
                while let Some(d) = stack.pop() {
                    if budget == 0 {
                        break;
                    }
                    budget -= 1;
                    for &y in &self.nodes[d as usize].links[l] {
                        if seen.contains(&y) {
                            continue;
                        }
                        seen.push(y);
                        if self.nodes[y as usize].deleted {
                            stack.push(y);
                        } else {
                            cand_ids.push(y);
                        }
                    }
                }
                let new_list = self.rebuild_list(u, &cand_ids, l, original_len);
                self.nodes[u as usize].links[l] = new_list;
                repaired += 1;
            }
        }
        repaired
    }

    /// Renumber live nodes densely and rebuild vector storage on the heap.
    fn compact(&mut self) {
        if self.live == self.nodes.len() {
            return;
        }
        let keep: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| !self.nodes[i as usize].deleted)
            .collect();
        let mut remap: HashMap<u32, u32> = HashMap::with_capacity(keep.len());
        for (new, &old) in keep.iter().enumerate() {
            remap.insert(old, new as u32);
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(keep.len());
        for &old in &keep {
            let n = &self.nodes[old as usize];
            nodes.push(Node {
                key: n.key,
                tags: n.tags,
                deleted: false,
                links: n
                    .links
                    .iter()
                    .map(|list| list.iter().filter_map(|x| remap.get(x).copied()).collect())
                    .collect(),
            });
        }
        self.vectors = self.vectors.compacted(&keep);
        self.key_to_id = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.key, i as u32))
            .collect();
        self.entry_point = self.entry_point.and_then(|e| remap.get(&e).copied());
        self.nodes = nodes;
        if self.entry_point.is_none() && self.live > 0 {
            self.reelect_entry_point();
        }
    }

    /// Nodes reachable from the entry point on `level`, routing through any node.
    fn reachable(&self, level: usize) -> Vec<bool> {
        let mut seen = vec![false; self.nodes.len()];
        let Some(ep) = self.entry_point else {
            return seen;
        };
        let mut queue = VecDeque::from([ep]);
        seen[ep as usize] = true;
        while let Some(u) = queue.pop_front() {
            for &v in &self.nodes[u as usize].links[level] {
                if !seen[v as usize] {
                    seen[v as usize] = true;
                    queue.push_back(v);
                }
            }
        }
        seen
    }

    /// Re-link live nodes that the entry point cannot reach on some layer and
    /// return how many links were added. Vacuum runs this automatically.
    ///
    /// Even without deletions, the neighbor-selection heuristic occasionally
    /// prunes the last in-edge of a node while shrinking a full adjacency list
    /// (hnswlib and the original paper share this property). Such a node can
    /// never be returned by a search; [`HealthReport::unreachable_l0`] counts
    /// them and this repairs them without a full vacuum. Cost: one BFS per
    /// layer plus one `ef_construction` search per orphan.
    pub fn repair_connectivity(&mut self) -> usize {
        self.ensure_connectivity()
    }

    /// Re-link live nodes that the entry point cannot reach. Returns how many
    /// were reconnected. Each orphan gets fresh out-edges from a search, and an
    /// in-edge from the nearest reachable node with spare capacity (or, if all
    /// are full, by replacing that node's farthest edge).
    fn ensure_connectivity(&mut self) -> usize {
        let mut reconnected = 0;
        for level in (0..=self.max_level).rev() {
            for _pass in 0..8 {
                let seen = self.reachable(level);
                let orphans: Vec<u32> = (0..self.nodes.len() as u32)
                    .filter(|&i| {
                        let n = &self.nodes[i as usize];
                        !n.deleted && n.level() >= level && !seen[i as usize]
                    })
                    .collect();
                if orphans.is_empty() {
                    break;
                }
                for &o in &orphans {
                    self.reconnect(o, level);
                    reconnected += 1;
                }
            }
        }
        reconnected
    }

    fn reconnect(&mut self, orphan: u32, level: usize) {
        let Some(ep) = self.entry_point else { return };
        let q = self.vectors.get(orphan).to_vec();
        let reach = self.reachable(level);
        let mut evals = 0;
        let start = Scored {
            dist: self.dist_to(&q, ep),
            id: ep,
        };
        let mut visited = self.visited_take();
        let found = {
            let nodes = &self.nodes;
            self.search_layer(
                &q,
                &[start],
                self.config.ef_construction,
                level,
                &|n| n != orphan && !nodes[n as usize].deleted && reach[n as usize],
                &mut visited,
                &mut evals,
            )
        };
        self.visited_give(visited);
        if found.is_empty() {
            return;
        }
        // Out-edges: merge existing ones with the new candidates.
        let mut cand_ids: Vec<u32> = self.nodes[orphan as usize].links[level].clone();
        for s in &found {
            if !cand_ids.contains(&s.id) {
                cand_ids.push(s.id);
            }
        }
        let list = self.rebuild_list(orphan, &cand_ids, level, self.config.m);
        self.nodes[orphan as usize].links[level] = list;
        // In-edge from the nearest reachable node, preferring spare capacity.
        let cap = self.config.max_degree(level);
        let target = found
            .iter()
            .find(|s| self.nodes[s.id as usize].links[level].len() < cap)
            .unwrap_or(&found[0])
            .id;
        let tlist = &mut self.nodes[target as usize].links[level];
        if tlist.contains(&orphan) {
            return;
        }
        if tlist.len() < cap {
            tlist.push(orphan);
        } else {
            let far = (0..tlist.len())
                .max_by(|&a, &b| {
                    let da = self.dist_between(target, self.nodes[target as usize].links[level][a]);
                    let db = self.dist_between(target, self.nodes[target as usize].links[level][b]);
                    da.total_cmp(&db)
                })
                .expect("full list is non-empty");
            self.nodes[target as usize].links[level][far] = orphan;
        }
    }

    fn visited_take(&self) -> crate::visited::VisitedList {
        self.visited.take(self.nodes.len())
    }

    fn visited_give(&self, v: crate::visited::VisitedList) {
        self.visited.give(v)
    }

    /// Compute a [`HealthReport`]. O(n · M).
    pub fn health(&self) -> HealthReport {
        let mut level_histogram = vec![0usize; self.max_level + 1];
        let mut deg_sum = 0usize;
        let mut dangling = 0usize;
        for n in self.nodes.iter().filter(|n| !n.deleted) {
            let lvl = n.level().min(self.max_level);
            for slot in level_histogram.iter_mut().take(lvl + 1) {
                *slot += 1;
            }
            deg_sum += n.links[0].len();
            dangling += n
                .links
                .iter()
                .flatten()
                .filter(|&&x| self.nodes[x as usize].deleted)
                .count();
        }
        let seen = self.reachable(0);
        let unreachable_l0 = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| !n.deleted && !seen[*i])
            .count();
        HealthReport {
            live: self.live,
            tombstones: self.tombstones(),
            max_level: self.max_level,
            level_histogram,
            mean_degree_l0: if self.live == 0 {
                0.0
            } else {
                deg_sum as f64 / self.live as f64
            },
            dangling_edges: dangling,
            unreachable_l0,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::HnswConfig;
    use crate::error::Error;
    use crate::hnsw::tests::grid_index;
    use crate::hnsw::Hnsw;

    #[test]
    fn deleted_keys_never_returned() {
        let mut idx = grid_index(100, 6);
        for k in (0..100).step_by(3) {
            idx.delete(k).unwrap();
        }
        assert_eq!(idx.len(), 66);
        assert_eq!(idx.tombstones(), 34);
        for i in 0..100u64 {
            let q = [(i % 10) as f32, (i / 10) as f32];
            for h in idx.search(&q, 10).unwrap() {
                assert!(h.key % 3 != 0, "deleted key {} returned", h.key);
            }
        }
        assert!(matches!(idx.delete(0), Err(Error::KeyNotFound(0))));
    }

    #[test]
    fn deleting_everything_then_reinserting_works() {
        let mut idx = grid_index(30, 4);
        for k in 0..30 {
            idx.delete(k).unwrap();
        }
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0, 0.0], 3).unwrap().is_empty());
        idx.insert(500, &[1.0, 1.0]).unwrap();
        idx.insert(501, &[2.0, 2.0]).unwrap();
        assert_eq!(idx.search(&[0.0, 0.0], 3).unwrap()[0].key, 500);
        let report = idx.vacuum();
        assert_eq!(report.removed, 30);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.tombstones(), 0);
    }

    #[test]
    fn vacuum_compacts_and_keeps_graph_connected() {
        let mut idx = grid_index(200, 4);
        for k in 0..200u64 {
            if k % 2 == 0 || k % 7 == 0 {
                idx.delete(k).unwrap();
            }
        }
        let live_before: Vec<u64> = {
            let mut v: Vec<u64> = idx.keys().collect();
            v.sort_unstable();
            v
        };
        let r = idx.vacuum();
        assert!(r.removed > 0);
        let h = idx.health();
        assert_eq!(h.tombstones, 0);
        assert_eq!(h.dangling_edges, 0);
        assert_eq!(h.unreachable_l0, 0);
        let mut live_after: Vec<u64> = idx.keys().collect();
        live_after.sort_unstable();
        assert_eq!(live_before, live_after);
        // With ef >= n the search must be exact on a connected graph.
        for &k in &live_after {
            let q = [(k % 10) as f32, (k / 10) as f32];
            let got = idx
                .search_with(&q, crate::hnsw::SearchParams::top(1).ef(idx.len()))
                .unwrap();
            assert_eq!(got.hits[0].key, k);
        }
    }

    #[test]
    fn health_reports_dangling_edges_before_vacuum() {
        let mut idx = grid_index(100, 4);
        for k in 0..50 {
            idx.delete(k).unwrap();
        }
        let h = idx.health();
        assert_eq!(h.live, 50);
        assert_eq!(h.tombstones, 50);
        assert!(h.mean_degree_l0 > 0.0);
        assert_eq!(h.level_histogram[0], 50);
        let empty = Hnsw::new(HnswConfig::new(3)).unwrap().health();
        assert_eq!(empty.mean_degree_l0, 0.0);
        assert_eq!(empty.unreachable_l0, 0);
    }

    #[test]
    fn reconnect_relinks_an_orphan() {
        let mut idx = grid_index(50, 4);
        // Sever every in-edge to node 0 on layer 0 by hand.
        for n in idx.nodes.iter_mut() {
            n.links[0].retain(|&x| x != 0);
        }
        if idx.entry_point == Some(0) {
            return; // entry cannot be orphaned; nothing to test for this seed
        }
        assert_eq!(idx.health().unreachable_l0, 1);
        let fixed = idx.repair_connectivity();
        assert!(fixed >= 1);
        assert_eq!(idx.health().unreachable_l0, 0);
    }
}
