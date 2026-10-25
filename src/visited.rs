//! Reusable visited-sets for graph traversal.
//!
//! Each search needs an O(1) "have I seen node i" check. Allocating and zeroing
//! a bitmap per query would cost O(n); instead each list stores an epoch per
//! node and a query bumps the epoch, so reset is O(1) amortized. Lists are
//! pooled so concurrent readers (`&Hnsw` is `Sync`) do not contend on one.

use std::sync::Mutex;

/// Epoch-stamped visited set.
#[derive(Debug, Default)]
pub struct VisitedList {
    marks: Vec<u32>,
    epoch: u32,
}

impl VisitedList {
    /// Prepare for a traversal over `n` nodes.
    pub fn reset(&mut self, n: usize) {
        if self.marks.len() < n {
            self.marks.resize(n, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped around: old stamps could alias the new epoch.
            self.marks.iter_mut().for_each(|m| *m = 0);
            self.epoch = 1;
        }
    }

    /// Mark `i` visited; returns `true` if it was not visited before.
    #[inline]
    pub fn insert(&mut self, i: u32) -> bool {
        let slot = &mut self.marks[i as usize];
        if *slot == self.epoch {
            false
        } else {
            *slot = self.epoch;
            true
        }
    }
}

/// Pool of visited lists shared by concurrent searches.
#[derive(Debug, Default)]
pub struct VisitedPool {
    lists: Mutex<Vec<VisitedList>>,
}

impl VisitedPool {
    /// Take a list sized for `n` nodes (allocating if the pool is empty).
    pub fn take(&self, n: usize) -> VisitedList {
        let mut list = self
            .lists
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
            .unwrap_or_default();
        list.reset(n);
        list
    }

    /// Return a list to the pool.
    pub fn give(&self, list: VisitedList) {
        let mut lists = self.lists.lock().unwrap_or_else(|p| p.into_inner());
        if lists.len() < 64 {
            lists.push(list);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epochs_isolate_traversals() {
        let pool = VisitedPool::default();
        let mut v = pool.take(4);
        assert!(v.insert(1));
        assert!(!v.insert(1));
        pool.give(v);
        let mut v = pool.take(8);
        assert!(v.insert(1), "new epoch forgets old marks");
        assert!(v.insert(7));
    }

    #[test]
    fn epoch_wraparound_clears_marks() {
        let mut v = VisitedList::default();
        v.reset(2);
        v.insert(0);
        v.epoch = u32::MAX;
        v.marks[1] = 1; // stale stamp equal to the post-wrap epoch
        v.reset(2);
        assert_eq!(v.epoch, 1);
        assert!(v.insert(1));
    }
}
