//! Online recall monitoring by sampled brute-force shadow queries.
//!
//! An ANN index silently trades accuracy for speed, and that trade drifts:
//! deletions erode the graph, the data distribution shifts, somebody lowers
//! `efSearch` to hit a latency target. Offline benchmarks cannot see any of
//! that. This module measures recall *on live traffic*:
//!
//! 1. A configurable fraction of served queries is **sampled**: the query, its
//!    parameters, the hits that were returned and the index write-generation
//!    are queued (the serving path pays only a clone and a queue push).
//! 2. A shadow evaluator later runs the **exact** search for each sample and
//!    scores the served hits against it. If the index was modified in between,
//!    the ANN query is replayed first so both sides see the same data.
//! 3. Scores feed a sliding **window**; the window's mean and 10th percentile
//!    drive a `Warming → Healthy ⇄ Degraded` status with hysteresis, and an
//!    optional [`MonitoredIndex::autotune`] step raises or lowers `efSearch`.
//!
//! Recall is *distance-aware*: a served hit counts if its distance is no worse
//! than the k-th exact distance, so ties between equidistant points are not
//! reported as misses.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::hnsw::{Filter, Hit, Hnsw, SearchParams, SearchResult, Strategy};
use crate::rng::SplitMix64;

/// Tuning knobs for [`RecallMonitor`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Fraction of served queries that get a shadow evaluation, in `[0, 1]`.
    pub sample_rate: f64,
    /// Number of most recent samples the statistics are computed over.
    pub window: usize,
    /// Samples needed before the status leaves `Warming`.
    pub min_samples: usize,
    /// Alert when the window's mean recall falls below this.
    pub target_recall: f64,
    /// Hysteresis: a degraded monitor recovers only above `target + margin`.
    pub recover_margin: f64,
    /// Bound on queued, not yet evaluated samples; extra samples are dropped.
    pub max_pending: usize,
    /// Seed of the sampling decision stream.
    pub seed: u64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 0.01,
            window: 512,
            min_samples: 32,
            target_recall: 0.95,
            recover_margin: 0.01,
            max_pending: 1024,
            seed: 0x5A3D_0E0C,
        }
    }
}

impl MonitorConfig {
    /// Reject inconsistent settings.
    pub fn validate(&self) -> Result<()> {
        let unit = 0.0..=1.0;
        if !unit.contains(&self.sample_rate) || !unit.contains(&self.target_recall) {
            return Err(Error::InvalidConfig(
                "sample_rate and target_recall must be within [0, 1]".into(),
            ));
        }
        if !(0.0..=0.5).contains(&self.recover_margin) {
            return Err(Error::InvalidConfig(
                "recover_margin must be within [0, 0.5]".into(),
            ));
        }
        if self.window == 0 || self.min_samples == 0 || self.min_samples > self.window {
            return Err(Error::InvalidConfig(
                "need 1 <= min_samples <= window".into(),
            ));
        }
        if self.max_pending == 0 {
            return Err(Error::InvalidConfig("max_pending must be >= 1".into()));
        }
        Ok(())
    }
}

/// Health of the index as seen by the monitor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecallStatus {
    /// Fewer than `min_samples` samples in the window.
    Warming,
    /// Mean recall at or above target.
    Healthy,
    /// Mean recall below target.
    Degraded,
}

/// One scored shadow query.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RecallSample {
    /// Distance-aware recall@k of the served hits.
    pub recall: f64,
    /// Beam width the query was served with.
    pub ef: usize,
    /// Planner strategy used.
    pub strategy: Strategy,
    /// Whether the ANN query had to be replayed because the index changed.
    pub replayed: bool,
}

/// Snapshot of monitor statistics, ready for a metrics endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallReport {
    /// Current status.
    pub status: RecallStatus,
    /// Samples in the sliding window.
    pub window_samples: usize,
    /// Mean recall over the window (1.0 when empty).
    pub mean_recall: f64,
    /// 10th-percentile recall over the window — the "bad tail".
    pub p10_recall: f64,
    /// Worst recall in the window.
    pub min_recall: f64,
    /// Samples evaluated since start.
    pub evaluated_total: u64,
    /// Samples queued but not yet evaluated.
    pub pending: usize,
    /// Samples dropped because the queue was full.
    pub dropped: u64,
    /// Samples whose ANN query was replayed after an index write.
    pub replayed: u64,
    /// Number of `Healthy → Degraded` transitions (alerts fired).
    pub alerts: u64,
}

#[derive(Debug, Clone)]
struct PendingSample {
    query: Vec<f32>,
    k: usize,
    ef: usize,
    filter: Option<Filter>,
    served: Vec<Hit>,
    strategy: Strategy,
    generation: u64,
}

#[derive(Debug)]
struct WindowState {
    samples: VecDeque<RecallSample>,
    status: RecallStatus,
    evaluated_total: u64,
    dropped: u64,
    replayed: u64,
    alerts: u64,
}

/// Collects shadow samples and turns them into recall statistics.
///
/// The monitor is index-agnostic: [`RecallMonitor::observe`] is called on the
/// serving path, [`RecallMonitor::evaluate`] by whoever owns spare CPU.
#[derive(Debug)]
pub struct RecallMonitor {
    config: MonitorConfig,
    rng: Mutex<SplitMix64>,
    pending: Mutex<VecDeque<PendingSample>>,
    state: Mutex<WindowState>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // A panic while holding a monitor lock cannot leave the data inconsistent
    // in a way that matters for statistics, so recover from poisoning.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl RecallMonitor {
    /// Create a monitor.
    pub fn new(config: MonitorConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            rng: Mutex::new(SplitMix64::new(config.seed)),
            pending: Mutex::new(VecDeque::new()),
            state: Mutex::new(WindowState {
                samples: VecDeque::with_capacity(config.window),
                status: RecallStatus::Warming,
                evaluated_total: 0,
                dropped: 0,
                replayed: 0,
                alerts: 0,
            }),
            config,
        })
    }

    /// Active configuration.
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Decide whether to sample a served query and, if so, queue it.
    /// Returns `true` when the query was queued.
    pub fn observe(
        &self,
        query: &[f32],
        params: &SearchParams<'_>,
        served: &SearchResult,
        default_ef: usize,
        generation: u64,
    ) -> bool {
        if self.config.sample_rate <= 0.0 || params.k == 0 {
            return false;
        }
        let draw = lock(&self.rng).next_f64_open();
        if draw >= self.config.sample_rate {
            return false;
        }
        let mut pending = lock(&self.pending);
        if pending.len() >= self.config.max_pending {
            drop(pending);
            lock(&self.state).dropped += 1;
            return false;
        }
        pending.push_back(PendingSample {
            query: query.to_vec(),
            k: params.k,
            ef: params.ef.unwrap_or(default_ef).max(params.k),
            filter: params.filter.cloned(),
            served: served.hits.clone(),
            strategy: served.stats.strategy,
            generation,
        });
        true
    }

    /// Number of queued samples.
    pub fn pending(&self) -> usize {
        lock(&self.pending).len()
    }

    /// Evaluate up to `budget` queued samples against `index`, whose current
    /// write-generation is `generation`. Returns how many were evaluated.
    pub fn evaluate(&self, index: &Hnsw, generation: u64, budget: usize) -> Result<usize> {
        let mut done = 0;
        while done < budget {
            let Some(sample) = lock(&self.pending).pop_front() else {
                break;
            };
            let scored = score_sample(index, &sample, generation)?;
            if let Some(s) = scored {
                self.record(s);
            }
            done += 1;
        }
        Ok(done)
    }

    /// Add a scored sample to the window and update the status.
    pub fn record(&self, sample: RecallSample) {
        let mut st = lock(&self.state);
        if st.samples.len() == self.config.window {
            st.samples.pop_front();
        }
        st.samples.push_back(sample);
        st.evaluated_total += 1;
        if sample.replayed {
            st.replayed += 1;
        }
        let mean = mean(st.samples.iter().map(|s| s.recall));
        let next = match st.status {
            _ if st.samples.len() < self.config.min_samples => RecallStatus::Warming,
            RecallStatus::Degraded
                if mean < self.config.target_recall + self.config.recover_margin =>
            {
                RecallStatus::Degraded
            }
            _ if mean < self.config.target_recall => RecallStatus::Degraded,
            _ => RecallStatus::Healthy,
        };
        if next == RecallStatus::Degraded && st.status != RecallStatus::Degraded {
            st.alerts += 1;
            tracing::warn!(
                mean_recall = mean,
                target = self.config.target_recall,
                "recall degraded"
            );
        } else if next == RecallStatus::Healthy && st.status == RecallStatus::Degraded {
            tracing::info!(mean_recall = mean, "recall recovered");
        }
        st.status = next;
    }

    /// Forget the window (e.g. after changing `efSearch`, old samples no
    /// longer describe the index). Counters are kept.
    pub fn reset_window(&self) {
        let mut st = lock(&self.state);
        st.samples.clear();
        st.status = RecallStatus::Warming;
    }

    /// Current statistics.
    pub fn report(&self) -> RecallReport {
        let pending = self.pending();
        let st = lock(&self.state);
        let mut recalls: Vec<f64> = st.samples.iter().map(|s| s.recall).collect();
        recalls.sort_by(f64::total_cmp);
        let p10 = if recalls.is_empty() {
            1.0
        } else {
            recalls[(recalls.len() - 1) / 10]
        };
        RecallReport {
            status: st.status,
            window_samples: recalls.len(),
            mean_recall: if recalls.is_empty() {
                1.0
            } else {
                mean(recalls.iter().copied())
            },
            p10_recall: p10,
            min_recall: recalls.first().copied().unwrap_or(1.0),
            evaluated_total: st.evaluated_total,
            pending,
            dropped: st.dropped,
            replayed: st.replayed,
            alerts: st.alerts,
        }
    }
}

fn mean(it: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = it.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 {
        1.0
    } else {
        sum / n as f64
    }
}

/// Distance-aware recall@k of `served` against the exact answer `exact`
/// (sorted ascending). Returns `None` when there is nothing to find.
pub fn distance_aware_recall(served: &[Hit], exact: &[Hit], k: usize) -> Option<f64> {
    let truth = &exact[..k.min(exact.len())];
    let kth = truth.last()?.distance;
    let tolerance = kth.abs() * 1e-5 + 1e-6;
    let good = served
        .iter()
        .take(k)
        .filter(|h| h.distance <= kth + tolerance)
        .count();
    Some((good.min(truth.len())) as f64 / truth.len() as f64)
}

fn score_sample(index: &Hnsw, s: &PendingSample, generation: u64) -> Result<Option<RecallSample>> {
    let exact = index.exact_search(&s.query, s.k, s.filter.as_ref())?;
    let replayed = s.generation != generation;
    let served = if replayed {
        let mut p = SearchParams::top(s.k).ef(s.ef);
        if let Some(f) = &s.filter {
            p = p.filter(f);
        }
        index.search_with(&s.query, p)?.hits
    } else {
        s.served.clone()
    };
    Ok(
        distance_aware_recall(&served, &exact, s.k).map(|recall| RecallSample {
            recall,
            ef: s.ef,
            strategy: s.strategy,
            replayed,
        }),
    )
}

/// Bounds and step sizes for [`MonitoredIndex::autotune`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AutotunePolicy {
    /// Never go below this beam width.
    pub min_ef: usize,
    /// Never go above this beam width.
    pub max_ef: usize,
    /// Lower `ef` only when mean recall exceeds `target + slack`.
    pub slack: f64,
}

impl Default for AutotunePolicy {
    fn default() -> Self {
        Self {
            min_ef: 16,
            max_ef: 1024,
            slack: 0.03,
        }
    }
}

/// Suggest a new `efSearch` from a report: double when degraded, shrink by a
/// quarter when comfortably above target, otherwise keep. `None` = no change.
pub fn suggest_ef(
    report: &RecallReport,
    target: f64,
    current: usize,
    policy: &AutotunePolicy,
) -> Option<usize> {
    let next = match report.status {
        RecallStatus::Warming => return None,
        RecallStatus::Degraded => (current * 2).min(policy.max_ef),
        RecallStatus::Healthy if report.mean_recall > target + policy.slack => {
            (current * 3 / 4).max(policy.min_ef)
        }
        RecallStatus::Healthy => return None,
    };
    (next != current).then_some(next)
}

/// An index behind a reader/writer lock with recall monitoring built in.
///
/// Searches take the read lock; writes take the write lock and bump a
/// generation counter so the shadow evaluator knows when to replay.
#[derive(Debug)]
pub struct MonitoredIndex {
    index: RwLock<Hnsw>,
    generation: AtomicU64,
    monitor: RecallMonitor,
}

impl MonitoredIndex {
    /// Wrap an index.
    pub fn new(index: Hnsw, config: MonitorConfig) -> Result<Self> {
        Ok(Self {
            index: RwLock::new(index),
            generation: AtomicU64::new(0),
            monitor: RecallMonitor::new(config)?,
        })
    }

    fn read(&self) -> RwLockReadGuard<'_, Hnsw> {
        self.index.read().unwrap_or_else(|p| p.into_inner())
    }

    fn write(&self) -> RwLockWriteGuard<'_, Hnsw> {
        self.index.write().unwrap_or_else(|p| p.into_inner())
    }

    /// Serve a query (and maybe sample it).
    pub fn search(&self, query: &[f32], params: SearchParams<'_>) -> Result<SearchResult> {
        let idx = self.read();
        let gen = self.generation.load(Ordering::Acquire);
        let result = idx.search_with(query, params)?;
        self.monitor
            .observe(query, &params, &result, idx.config().ef_search, gen);
        Ok(result)
    }

    /// Run any mutation under the write lock and bump the generation.
    pub fn write_with<T>(&self, f: impl FnOnce(&mut Hnsw) -> Result<T>) -> Result<T> {
        let mut idx = self.write();
        let out = f(&mut idx);
        self.generation.fetch_add(1, Ordering::AcqRel);
        out
    }

    /// Insert or replace a point.
    pub fn insert_with_tags(&self, key: u64, vector: &[f32], tags: u64) -> Result<()> {
        self.write_with(|i| i.insert_with_tags(key, vector, tags))
    }

    /// Delete a point.
    pub fn delete(&self, key: u64) -> Result<()> {
        self.write_with(|i| i.delete(key))
    }

    /// Read access to the wrapped index (e.g. for snapshots).
    pub fn with_index<T>(&self, f: impl FnOnce(&Hnsw) -> T) -> T {
        f(&self.read())
    }

    /// Evaluate up to `budget` pending shadow samples.
    pub fn run_shadow(&self, budget: usize) -> Result<usize> {
        let idx = self.read();
        let gen = self.generation.load(Ordering::Acquire);
        self.monitor.evaluate(&idx, gen, budget)
    }

    /// Current recall statistics.
    pub fn report(&self) -> RecallReport {
        self.monitor.report()
    }

    /// The underlying monitor.
    pub fn monitor(&self) -> &RecallMonitor {
        &self.monitor
    }

    /// Current write generation.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Apply [`suggest_ef`]; on change, update `efSearch` and reset the
    /// window. Returns the new `ef` if it changed.
    pub fn autotune(&self, policy: &AutotunePolicy) -> Option<usize> {
        let report = self.report();
        let current = self.read().config().ef_search;
        let next = suggest_ef(&report, self.monitor.config.target_recall, current, policy)?;
        let mut idx = self.write();
        idx.set_ef_search(next).ok()?;
        drop(idx);
        self.monitor.reset_window();
        tracing::info!(
            from = current,
            to = next,
            mean = report.mean_recall,
            "efSearch autotuned"
        );
        Some(next)
    }

    /// Spawn a background thread that evaluates pending samples every
    /// `interval`. The worker stops when the returned handle is dropped.
    pub fn spawn_shadow_worker(
        self: &Arc<Self>,
        interval: Duration,
        budget: usize,
    ) -> ShadowWorker {
        let stop = Arc::new(AtomicBool::new(false));
        let this = Arc::clone(self);
        let flag = Arc::clone(&stop);
        let handle = std::thread::spawn(move || {
            while !flag.load(Ordering::Acquire) {
                if let Err(e) = this.run_shadow(budget) {
                    tracing::error!(error = %e, "shadow evaluation failed");
                }
                std::thread::sleep(interval);
            }
            // Drain what is left so a clean shutdown loses no samples.
            let _ = this.run_shadow(usize::MAX);
        });
        ShadowWorker {
            stop,
            handle: Some(handle),
        }
    }
}

/// Handle of a background shadow evaluator; stops and joins on drop.
#[derive(Debug)]
pub struct ShadowWorker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ShadowWorker {
    /// Stop the worker and wait for it to drain.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ShadowWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HnswConfig;
    use crate::dataset::gaussian_mixture;

    fn cfg(rate: f64) -> MonitorConfig {
        MonitorConfig {
            sample_rate: rate,
            window: 64,
            min_samples: 8,
            target_recall: 0.9,
            recover_margin: 0.02,
            max_pending: 16,
            seed: 1,
        }
    }

    fn index(n: usize) -> Hnsw {
        let data = gaussian_mixture(n, 16, 8, 3.0, 4);
        let mut idx = Hnsw::new(HnswConfig::new(16).with_m(8).with_ef_construction(64)).unwrap();
        for (i, v) in data.iter().enumerate() {
            idx.insert(i as u64, v).unwrap();
        }
        idx
    }

    fn sample(recall: f64) -> RecallSample {
        RecallSample {
            recall,
            ef: 10,
            strategy: Strategy::Graph,
            replayed: false,
        }
    }

    #[test]
    fn config_validation() {
        MonitorConfig::default().validate().unwrap();
        let bad = [
            MonitorConfig {
                sample_rate: 1.5,
                ..cfg(0.1)
            },
            MonitorConfig {
                recover_margin: 0.9,
                ..cfg(0.1)
            },
            MonitorConfig {
                min_samples: 100,
                ..cfg(0.1)
            },
            MonitorConfig {
                max_pending: 0,
                ..cfg(0.1)
            },
        ];
        for c in bad {
            assert!(RecallMonitor::new(c).is_err());
        }
    }

    #[test]
    fn status_has_hysteresis_and_counts_alerts() {
        let m = RecallMonitor::new(cfg(1.0)).unwrap();
        assert_eq!(m.report().status, RecallStatus::Warming);
        assert_eq!(m.report().mean_recall, 1.0);
        for _ in 0..8 {
            m.record(sample(1.0));
        }
        assert_eq!(m.report().status, RecallStatus::Healthy);
        for _ in 0..16 {
            m.record(sample(0.5));
        }
        let r = m.report();
        assert_eq!(r.status, RecallStatus::Degraded);
        assert_eq!(r.alerts, 1);
        assert_eq!(r.min_recall, 0.5);
        // Climb to just above target but inside the margin: still degraded.
        while m.report().mean_recall < 0.905 {
            m.record(sample(1.0));
        }
        assert_eq!(m.report().status, RecallStatus::Degraded);
        for _ in 0..64 {
            m.record(sample(1.0));
        }
        let r = m.report();
        assert_eq!(r.status, RecallStatus::Healthy);
        assert_eq!(r.window_samples, 64);
        assert_eq!(r.alerts, 1);
        m.reset_window();
        assert_eq!(m.report().status, RecallStatus::Warming);
    }

    #[test]
    fn distance_aware_recall_handles_ties_and_empty_truth() {
        let h = |key, distance| Hit { key, distance };
        let exact = [h(1, 1.0), h(2, 2.0), h(3, 2.0)];
        // Key 4 is a different point at the same distance as the k-th: not a miss.
        assert_eq!(
            distance_aware_recall(&[h(1, 1.0), h(2, 2.0), h(4, 2.0)], &exact, 3),
            Some(1.0)
        );
        assert_eq!(
            distance_aware_recall(&[h(1, 1.0), h(5, 9.0)], &exact, 3),
            Some(1.0 / 3.0)
        );
        assert_eq!(distance_aware_recall(&[], &[], 3), None);
    }

    #[test]
    fn sampling_queue_and_evaluation() {
        let idx = index(400);
        let m = RecallMonitor::new(cfg(1.0)).unwrap();
        let q = idx.vector(7).unwrap().to_vec();
        let p = SearchParams::top(5);
        for _ in 0..20 {
            let r = idx.search_with(&q, p).unwrap();
            m.observe(&q, &p, &r, 64, 0);
        }
        let r = m.report();
        assert_eq!(r.pending, 16);
        assert_eq!(r.dropped, 4);
        assert_eq!(m.evaluate(&idx, 0, 10).unwrap(), 10);
        assert_eq!(m.evaluate(&idx, 0, 100).unwrap(), 6);
        let r = m.report();
        assert_eq!(r.evaluated_total, 16);
        assert_eq!(r.replayed, 0);
        assert!(r.mean_recall > 0.99);
        // Sampling disabled or k == 0: nothing queued.
        let off = RecallMonitor::new(cfg(0.0)).unwrap();
        let r = idx.search_with(&q, p).unwrap();
        assert!(!off.observe(&q, &p, &r, 64, 0));
        assert!(!m.observe(&q, &SearchParams::top(0), &r, 64, 0));
    }

    #[test]
    fn stale_samples_are_replayed_after_writes() {
        let mi = MonitoredIndex::new(index(300), cfg(1.0)).unwrap();
        let q = mi.with_index(|i| i.vector(3).unwrap().to_vec());
        let filter = Filter::KeyRange {
            start: 0,
            end: 1000,
        };
        let served = mi.search(&q, SearchParams::top(5).filter(&filter)).unwrap();
        assert_eq!(served.hits[0].key, 3);
        // Delete the top hit before the shadow runs: replay must not count it.
        mi.delete(3).unwrap();
        assert_eq!(mi.generation(), 1);
        assert_eq!(mi.run_shadow(10).unwrap(), 1);
        let r = mi.report();
        assert_eq!(r.replayed, 1);
        assert!(r.mean_recall > 0.99, "{r:?}");
        mi.insert_with_tags(3, &q, 0).unwrap();
        assert!(mi.write_with(|i| i.delete(99_999)).is_err());
        assert_eq!(mi.generation(), 3);
    }

    #[test]
    fn autotune_raises_ef_when_degraded_and_lowers_when_comfortable() {
        let policy = AutotunePolicy::default();
        let mut r = RecallReport {
            status: RecallStatus::Degraded,
            window_samples: 50,
            mean_recall: 0.8,
            p10_recall: 0.5,
            min_recall: 0.4,
            evaluated_total: 50,
            pending: 0,
            dropped: 0,
            replayed: 0,
            alerts: 1,
        };
        assert_eq!(suggest_ef(&r, 0.9, 32, &policy), Some(64));
        assert_eq!(suggest_ef(&r, 0.9, 1024, &policy), None);
        r.status = RecallStatus::Healthy;
        r.mean_recall = 0.999;
        assert_eq!(suggest_ef(&r, 0.9, 64, &policy), Some(48));
        assert_eq!(suggest_ef(&r, 0.9, 16, &policy), None);
        r.mean_recall = 0.91;
        assert_eq!(suggest_ef(&r, 0.9, 64, &policy), None);
        r.status = RecallStatus::Warming;
        assert_eq!(suggest_ef(&r, 0.9, 64, &policy), None);

        // End to end: a starved beam on clustered data degrades, autotune fixes it.
        let mut idx = index(2000);
        idx.set_ef_search(1).unwrap();
        let mi = MonitoredIndex::new(idx, cfg(1.0)).unwrap();
        let data = gaussian_mixture(64, 16, 8, 3.0, 99);
        let serve = |mi: &MonitoredIndex| {
            for q in data.iter() {
                mi.search(q, SearchParams::top(10)).unwrap();
                mi.run_shadow(4).unwrap();
            }
        };
        serve(&mi);
        assert_eq!(mi.report().status, RecallStatus::Degraded);
        let mut ef = 1;
        while let Some(next) = mi.autotune(&policy) {
            assert!(next > ef);
            ef = next;
            serve(&mi);
            if mi.report().status == RecallStatus::Healthy {
                break;
            }
        }
        assert_eq!(mi.report().status, RecallStatus::Healthy);
        assert!(mi.with_index(|i| i.config().ef_search) > 1);
    }

    #[test]
    fn background_worker_drains_on_stop() {
        let mi = Arc::new(MonitoredIndex::new(index(200), cfg(1.0)).unwrap());
        let worker = mi.spawn_shadow_worker(Duration::from_millis(1), 2);
        let q = mi.with_index(|i| i.vector(1).unwrap().to_vec());
        for _ in 0..12 {
            mi.search(&q, SearchParams::top(3)).unwrap();
        }
        worker.stop();
        let r = mi.report();
        assert_eq!(r.pending, 0);
        assert_eq!(r.evaluated_total + r.dropped, 12);
        assert!(mi.monitor().config().sample_rate > 0.0);
    }
}
