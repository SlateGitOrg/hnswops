//! Zero-downtime re-embedding migration: dual-write, backfill, verify, cut over.
//!
//! Changing the embedding model invalidates every stored vector, and the new
//! vectors may not even have the same dimension. Rebuilding offline means
//! either downtime or serving stale data. This module runs the migration
//! online, as a state machine:
//!
//! ```text
//!            start                backfill done           cutover(report)
//!  (serving) ─────▶ DualWrite ────────────────▶ Backfilled ─────────────▶ CutOver ──finalize──▶ new index
//!                      │  reads: old index           │  reads: old           │  reads: NEW
//!                      │  writes: old + new          │  writes: old + new    │  writes: old + new
//!                      └────────── abort ◀───────────┴────── rollback ◀──────┘
//! ```
//!
//! * **Dual-write.** Every upsert/delete is applied to both indexes (embedded
//!   with each one's model), so the new index never falls behind on fresh data
//!   and the old one stays complete for rollback.
//! * **Backfill.** Existing documents are copied in key order from the
//!   [`Corpus`] through a resumable cursor. Embedding (the slow part) happens
//!   *outside* any lock: [`Migration::plan_backfill`] takes a batch and the
//!   per-key write sequence numbers, [`BackfillBatch::embed`] runs unlocked,
//!   and [`Migration::apply_backfill`] applies a row only if no dual-write
//!   touched that key in between. This compare-and-set closes the classic
//!   race where a backfill overwrites a newer write with stale content.
//! * **Verification gate.** [`Migration::verify`] checks coverage (every
//!   corpus key present, nothing extra) and the new index's own ANN recall on
//!   probe queries; [`Migration::cutover`] refuses a failing report.
//! * **Cutover / rollback.** Cutover flips reads to the new index; because
//!   writes still go to both, rollback is instant and lossless until
//!   [`Migration::finalize`] drops the old index.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::brute::recall_at_k;
use crate::config::HnswConfig;
use crate::embed::{Corpus, Document, Embedder};
use crate::error::{Error, Result};
use crate::hnsw::{Hit, Hnsw};

/// Migration phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Writes go to both indexes; backfill in progress; reads from old.
    DualWrite,
    /// Backfill reached the end of the corpus; reads still from old.
    Backfilled,
    /// Reads served from the new index; writes still go to both.
    CutOver,
}

/// Counters describing a migration's progress.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationStats {
    /// Upserts applied to both indexes.
    pub dual_writes: u64,
    /// Deletes applied to both indexes.
    pub dual_deletes: u64,
    /// Documents copied by the backfill.
    pub backfilled: u64,
    /// Backfill rows skipped because a dual-write changed the key after planning.
    pub skipped_stale: u64,
    /// Backfill rows skipped because a dual-write had already written the key.
    pub skipped_fresh: u64,
    /// Backfill batches planned.
    pub batches: u64,
}

/// A planned backfill batch: documents plus the write sequence observed for
/// each key at planning time.
#[derive(Debug, Clone)]
pub struct BackfillBatch {
    rows: Vec<(Document, u64)>,
}

/// A backfill batch with its new-model vectors computed.
#[derive(Debug, Clone)]
pub struct EmbeddedBatch {
    rows: Vec<(Document, u64, Vec<f32>)>,
}

impl BackfillBatch {
    /// Number of documents in the batch.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when the batch is empty (backfill complete).
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Embed every document with the new model. Holds no lock.
    pub fn embed(self, embedder: &dyn Embedder) -> Result<EmbeddedBatch> {
        let rows = self
            .rows
            .into_iter()
            .map(|(doc, seq)| {
                let v = embedder.embed(&doc.text)?;
                Ok((doc, seq, v))
            })
            .collect::<Result<_>>()?;
        Ok(EmbeddedBatch { rows })
    }
}

/// Result of applying one backfill batch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyReport {
    /// Rows written to the new index.
    pub applied: usize,
    /// Rows skipped by the compare-and-set check.
    pub skipped: usize,
}

/// Outcome of the pre-cutover verification gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifyReport {
    /// Documents in the corpus.
    pub corpus_docs: usize,
    /// Live points in the new index.
    pub new_live: usize,
    /// Live points in the old index.
    pub old_live: usize,
    /// Corpus keys missing from the new index.
    pub missing: usize,
    /// New-index keys absent from the corpus.
    pub extra: usize,
    /// Mean recall@k of the new index's ANN search vs its exact search.
    pub new_index_recall: f64,
    /// Mean overlap of old vs new top-k (informational: a different model is
    /// *expected* to disagree somewhat; a value near 0 suggests a broken model).
    pub old_new_overlap: f64,
    /// Probe queries evaluated.
    pub probes: usize,
    /// Recall threshold the gate was evaluated with.
    pub min_recall: f64,
    /// Whether the gate passed.
    pub passed: bool,
    /// Human-readable reasons for a failure.
    pub failures: Vec<String>,
}

/// An in-flight re-embedding migration. Owns both indexes until it ends via
/// [`Migration::finalize`] (keep new) or [`Migration::abort`] (keep old).
pub struct Migration {
    old: Hnsw,
    old_embedder: Arc<dyn Embedder>,
    new: Hnsw,
    new_embedder: Arc<dyn Embedder>,
    phase: Phase,
    cursor: Option<u64>,
    seq: u64,
    write_seq: HashMap<u64, u64>,
    stats: MigrationStats,
}

impl std::fmt::Debug for Migration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Migration")
            .field("phase", &self.phase)
            .field("old_model", &self.old_embedder.name())
            .field("new_model", &self.new_embedder.name())
            .field("cursor", &self.cursor)
            .field("stats", &self.stats)
            .finish()
    }
}

impl Migration {
    /// Begin migrating `old` (embedded by `old_embedder`) to a fresh index
    /// built with `new_config` and `new_embedder`. Enters `DualWrite`.
    pub fn start(
        old: Hnsw,
        old_embedder: Arc<dyn Embedder>,
        new_config: HnswConfig,
        new_embedder: Arc<dyn Embedder>,
    ) -> Result<Self> {
        if old.config().dim != old_embedder.dim() {
            return Err(Error::Migration(format!(
                "old index has dim {} but old embedder '{}' produces {}",
                old.config().dim,
                old_embedder.name(),
                old_embedder.dim()
            )));
        }
        if new_config.dim != new_embedder.dim() {
            return Err(Error::Migration(format!(
                "new config has dim {} but new embedder '{}' produces {}",
                new_config.dim,
                new_embedder.name(),
                new_embedder.dim()
            )));
        }
        let new = Hnsw::new(new_config)?;
        tracing::info!(
            from = old_embedder.name(),
            to = new_embedder.name(),
            live = old.len(),
            "migration started"
        );
        Ok(Self {
            old,
            old_embedder,
            new,
            new_embedder,
            phase: Phase::DualWrite,
            cursor: None,
            seq: 0,
            write_seq: HashMap::new(),
            stats: MigrationStats::default(),
        })
    }

    /// Current phase.
    pub fn phase(&self) -> Phase {
        self.phase
    }

    /// Progress counters.
    pub fn stats(&self) -> &MigrationStats {
        &self.stats
    }

    /// Last key copied by the backfill (resume point).
    pub fn cursor(&self) -> Option<u64> {
        self.cursor
    }

    /// The index currently answering reads.
    pub fn read_index(&self) -> &Hnsw {
        match self.phase {
            Phase::CutOver => &self.new,
            Phase::DualWrite | Phase::Backfilled => &self.old,
        }
    }

    /// The old index.
    pub fn old_index(&self) -> &Hnsw {
        &self.old
    }

    /// The new index.
    pub fn new_index(&self) -> &Hnsw {
        &self.new
    }

    /// The embedder of the index currently answering reads.
    pub fn read_embedder(&self) -> &dyn Embedder {
        match self.phase {
            Phase::CutOver => self.new_embedder.as_ref(),
            Phase::DualWrite | Phase::Backfilled => self.old_embedder.as_ref(),
        }
    }

    /// The new model.
    pub fn new_embedder(&self) -> Arc<dyn Embedder> {
        Arc::clone(&self.new_embedder)
    }

    /// Serve a text query from whichever index is active.
    pub fn search(&self, text: &str, k: usize) -> Result<Vec<Hit>> {
        let q = self.read_embedder().embed(text)?;
        self.read_index().search(&q, k)
    }

    fn bump(&mut self, key: u64) {
        self.seq += 1;
        self.write_seq.insert(key, self.seq);
    }

    /// Dual-write an upsert. Both embeddings are computed before either index
    /// is touched, so a model failure leaves both indexes unchanged.
    pub fn upsert(&mut self, doc: &Document) -> Result<()> {
        let v_old = self.old_embedder.embed(&doc.text)?;
        let v_new = self.new_embedder.embed(&doc.text)?;
        self.old.insert_with_tags(doc.key, &v_old, doc.tags)?;
        self.new.insert_with_tags(doc.key, &v_new, doc.tags)?;
        self.bump(doc.key);
        self.stats.dual_writes += 1;
        Ok(())
    }

    /// Dual-write a delete. A key not yet backfilled is legitimately absent
    /// from the new index; the delete fails only if neither index has it.
    pub fn delete(&mut self, key: u64) -> Result<()> {
        let in_old = self.old.delete(key).is_ok();
        let in_new = self.new.delete(key).is_ok();
        if !in_old && !in_new {
            return Err(Error::KeyNotFound(key));
        }
        self.bump(key);
        self.stats.dual_deletes += 1;
        Ok(())
    }

    /// Take the next `limit` documents after the cursor and advance it. An
    /// empty batch means the backfill is complete (phase → `Backfilled`).
    pub fn plan_backfill(&mut self, corpus: &dyn Corpus, limit: usize) -> Result<BackfillBatch> {
        if self.phase != Phase::DualWrite {
            return Ok(BackfillBatch { rows: Vec::new() });
        }
        let docs = corpus.scan(self.cursor, limit.max(1));
        if let Some(last) = docs.last() {
            self.cursor = Some(last.key);
            self.stats.batches += 1;
        } else {
            self.phase = Phase::Backfilled;
            tracing::info!(backfilled = self.stats.backfilled, "backfill complete");
        }
        let rows = docs
            .into_iter()
            .map(|d| {
                let seq = self.write_seq.get(&d.key).copied().unwrap_or(0);
                (d, seq)
            })
            .collect();
        Ok(BackfillBatch { rows })
    }

    /// Apply an embedded batch with per-key compare-and-set against the
    /// write sequence captured at planning time.
    pub fn apply_backfill(&mut self, batch: EmbeddedBatch) -> Result<ApplyReport> {
        let mut report = ApplyReport::default();
        for (doc, planned_seq, v) in batch.rows {
            let current = self.write_seq.get(&doc.key).copied().unwrap_or(0);
            if current != planned_seq {
                // A dual-write (upsert or delete) raced this row: it is newer.
                self.stats.skipped_stale += 1;
                report.skipped += 1;
                continue;
            }
            if current != 0 {
                // Dual-written before planning: the new index already has the
                // latest version (or the key was deleted after the scan read it).
                self.stats.skipped_fresh += 1;
                report.skipped += 1;
                continue;
            }
            self.new.insert_with_tags(doc.key, &v, doc.tags)?;
            self.stats.backfilled += 1;
            report.applied += 1;
        }
        Ok(report)
    }

    /// Plan, embed and apply one batch (single-threaded convenience).
    /// Returns `None` once the backfill is complete.
    pub fn backfill_step(
        &mut self,
        corpus: &dyn Corpus,
        limit: usize,
    ) -> Result<Option<ApplyReport>> {
        let batch = self.plan_backfill(corpus, limit)?;
        if batch.is_empty() {
            return Ok(None);
        }
        let embedded = batch.embed(self.new_embedder.as_ref())?;
        self.apply_backfill(embedded).map(Some)
    }

    /// Run the verification gate: coverage against the corpus, the new
    /// index's ANN recall on `probes`, and old/new agreement.
    pub fn verify(
        &self,
        corpus: &dyn Corpus,
        probes: &[String],
        k: usize,
        min_recall: f64,
    ) -> Result<VerifyReport> {
        let mut corpus_docs = 0usize;
        let mut missing = 0usize;
        let mut cursor = None;
        loop {
            let page = corpus.scan(cursor, 1024);
            let Some(last) = page.last() else { break };
            cursor = Some(last.key);
            corpus_docs += page.len();
            missing += page.iter().filter(|d| !self.new.contains(d.key)).count();
        }
        let present = corpus_docs - missing;
        let extra = self.new.len().saturating_sub(present);

        let (mut recall_sum, mut overlap_sum) = (0.0, 0.0);
        for text in probes {
            let qn = self.new_embedder.embed(text)?;
            let ann: Vec<u64> = self.new.search(&qn, k)?.iter().map(|h| h.key).collect();
            let exact: Vec<u64> = self
                .new
                .exact_search(&qn, k, None)?
                .iter()
                .map(|h| h.key)
                .collect();
            recall_sum += recall_at_k(&ann, &exact, k);
            let qo = self.old_embedder.embed(text)?;
            let old: Vec<u64> = self.old.search(&qo, k)?.iter().map(|h| h.key).collect();
            let denom = old.len().max(ann.len()).max(1);
            overlap_sum += ann.iter().filter(|x| old.contains(x)).count() as f64 / denom as f64;
        }
        let n = probes.len().max(1) as f64;
        let new_index_recall = if probes.is_empty() {
            0.0
        } else {
            recall_sum / n
        };
        let mut failures = Vec::new();
        if self.phase == Phase::DualWrite {
            failures.push("backfill has not completed".to_string());
        }
        if missing > 0 {
            failures.push(format!("{missing} corpus keys missing from the new index"));
        }
        if extra > 0 {
            failures.push(format!(
                "{extra} keys in the new index are not in the corpus"
            ));
        }
        if probes.is_empty() {
            failures.push("no probe queries supplied".to_string());
        } else if new_index_recall < min_recall {
            failures.push(format!(
                "new index recall {new_index_recall:.3} below {min_recall:.3}"
            ));
        }
        Ok(VerifyReport {
            corpus_docs,
            new_live: self.new.len(),
            old_live: self.old.len(),
            missing,
            extra,
            new_index_recall,
            old_new_overlap: if probes.is_empty() {
                0.0
            } else {
                overlap_sum / n
            },
            probes: probes.len(),
            min_recall,
            passed: failures.is_empty(),
            failures,
        })
    }

    /// Flip reads to the new index. Requires `Backfilled` and a passing report.
    pub fn cutover(&mut self, report: &VerifyReport) -> Result<()> {
        if self.phase != Phase::Backfilled {
            return Err(Error::Migration(format!(
                "cutover requires phase backfilled, current phase is {:?}",
                self.phase
            )));
        }
        if !report.passed {
            return Err(Error::Migration(format!(
                "verification failed: {}",
                report.failures.join("; ")
            )));
        }
        self.phase = Phase::CutOver;
        tracing::info!(to = self.new_embedder.name(), "reads cut over to new index");
        Ok(())
    }

    /// Flip reads back to the old index (lossless: it kept receiving writes).
    pub fn rollback(&mut self) -> Result<()> {
        if self.phase != Phase::CutOver {
            return Err(Error::Migration(format!(
                "rollback requires phase cut_over, current phase is {:?}",
                self.phase
            )));
        }
        self.phase = Phase::Backfilled;
        tracing::warn!(
            to = self.old_embedder.name(),
            "reads rolled back to old index"
        );
        Ok(())
    }

    /// End the migration keeping the new index. Requires `CutOver`.
    pub fn finalize(self) -> Result<Hnsw> {
        if self.phase != Phase::CutOver {
            return Err(Error::Migration(format!(
                "finalize requires phase cut_over, current phase is {:?}",
                self.phase
            )));
        }
        tracing::info!(stats = ?self.stats, "migration finalized");
        Ok(self.new)
    }

    /// Abandon the migration, keeping the (fully up to date) old index.
    pub fn abort(self) -> Hnsw {
        tracing::warn!(phase = ?self.phase, "migration aborted");
        self.old
    }
}

/// Drive the whole backfill against a shared migration, holding the write
/// lock only to plan and to apply each batch — never while embedding — so
/// concurrent readers and dual-writers are blocked for O(batch) index work
/// at most. Returns the number of batches processed.
pub fn backfill_concurrently(
    migration: &RwLock<Migration>,
    corpus: &dyn Corpus,
    batch_size: usize,
) -> Result<usize> {
    let mut batches = 0;
    loop {
        let (batch, embedder) = {
            let mut m = migration.write().unwrap_or_else(|p| p.into_inner());
            (m.plan_backfill(corpus, batch_size)?, m.new_embedder())
        };
        if batch.is_empty() {
            return Ok(batches);
        }
        let embedded = batch.embed(embedder.as_ref())?;
        migration
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .apply_backfill(embedded)?;
        batches += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Metric;
    use crate::embed::{synthetic_corpus, synthetic_query, HashingEmbedder, InMemoryCorpus};

    fn models() -> (Arc<dyn Embedder>, Arc<dyn Embedder>) {
        (
            Arc::new(HashingEmbedder::new("hash-v1", 32, 1, 1).unwrap()),
            Arc::new(HashingEmbedder::new("hash-v2", 48, 2, 2).unwrap()),
        )
    }

    fn cfg(dim: usize) -> HnswConfig {
        HnswConfig::new(dim)
            .with_metric(Metric::Cosine)
            .with_m(8)
            .with_ef_construction(64)
    }

    fn setup(n: usize) -> (InMemoryCorpus, Migration) {
        let (v1, v2) = models();
        let corpus = InMemoryCorpus::from_docs(synthetic_corpus(n, 6, 3));
        let mut old = Hnsw::new(cfg(32)).unwrap();
        for d in corpus.scan(None, usize::MAX) {
            old.insert_with_tags(d.key, &v1.embed(&d.text).unwrap(), d.tags)
                .unwrap();
        }
        let m = Migration::start(old, v1, cfg(48), v2).unwrap();
        (corpus, m)
    }

    fn probes() -> Vec<String> {
        (0..12).map(|i| synthetic_query(i % 6, i as u64)).collect()
    }

    #[test]
    fn full_lifecycle_with_rollback() {
        let (corpus, mut m) = setup(300);
        assert_eq!(m.phase(), Phase::DualWrite);
        assert!(format!("{m:?}").contains("hash-v2"));
        // Reads come from the old index while the new one is empty.
        assert_eq!(m.search("t1w3 t1w4", 3).unwrap().len(), 3);
        assert_eq!(m.read_embedder().name(), "hash-v1");
        let early = m.verify(&corpus, &probes(), 10, 0.9).unwrap();
        assert!(!early.passed);
        assert!(m.cutover(&early).is_err());

        let mut steps = 0;
        while m.backfill_step(&corpus, 64).unwrap().is_some() {
            steps += 1;
        }
        assert_eq!(steps, 5);
        assert_eq!(m.phase(), Phase::Backfilled);
        assert_eq!(m.cursor(), Some(299));
        assert_eq!(m.stats().backfilled, 300);
        // Planning after completion is a no-op.
        assert!(m.plan_backfill(&corpus, 10).unwrap().is_empty());

        let report = m.verify(&corpus, &probes(), 10, 0.9).unwrap();
        assert!(report.passed, "{report:?}");
        assert_eq!((report.missing, report.extra), (0, 0));
        assert!(report.new_index_recall >= 0.9);
        assert!(m.rollback().is_err());
        m.cutover(&report).unwrap();
        assert_eq!(m.read_embedder().name(), "hash-v2");
        assert_eq!(m.read_index().config().dim, 48);

        // Writes during cutover reach both sides, so rollback loses nothing.
        let doc = Document {
            key: 10_000,
            text: "t5w1 t5w2 brand new".into(),
            tags: 1,
        };
        m.upsert(&doc).unwrap();
        corpus.upsert(doc);
        m.rollback().unwrap();
        assert!(m.old_index().contains(10_000) && m.new_index().contains(10_000));
        m.cutover(&report).unwrap();
        let new = m.finalize().unwrap();
        assert_eq!(new.len(), 301);
    }

    #[test]
    fn racing_writes_are_not_overwritten_by_backfill() {
        let (corpus, mut m) = setup(50);
        let batch = m.plan_backfill(&corpus, 10).unwrap();
        assert_eq!(batch.len(), 10);
        // While the batch is being embedded, key 3 is updated and key 4 deleted.
        let fresh = Document {
            key: 3,
            text: "t0w0 updated content".into(),
            tags: 2,
        };
        m.upsert(&fresh).unwrap();
        corpus.upsert(fresh.clone());
        m.delete(4).unwrap();
        corpus.delete(4);
        let embedded = batch.embed(m.new_embedder().as_ref()).unwrap();
        let r = m.apply_backfill(embedded).unwrap();
        assert_eq!(
            r,
            ApplyReport {
                applied: 8,
                skipped: 2
            }
        );
        assert_eq!(m.stats().skipped_stale, 2);
        // Stored vectors are cosine-normalized, so compare the prepared form.
        let want = Metric::Cosine
            .prepare(&m.new_embedder().embed(&fresh.text).unwrap())
            .unwrap();
        assert_eq!(m.new_index().vector(3).unwrap(), want.as_slice());
        assert_eq!(m.new_index().tags(3), Some(2));
        assert!(!m.new_index().contains(4) && !m.old_index().contains(4));

        // A key dual-written before it is planned is skipped as already fresh.
        let early = Document {
            key: 40,
            text: "t2w2 early write".into(),
            tags: 4,
        };
        m.upsert(&early).unwrap();
        corpus.upsert(early);
        while m.backfill_step(&corpus, 16).unwrap().is_some() {}
        assert_eq!(m.stats().skipped_fresh, 1);
        let report = m.verify(&corpus, &probes(), 5, 0.8).unwrap();
        assert!(report.passed, "{report:?}");
        assert_eq!(report.corpus_docs, 49);
        assert_eq!(m.delete(4).unwrap_err().to_string(), "key 4 not found");
    }

    #[test]
    fn guards_reject_misuse() {
        let (v1, v2) = models();
        let old = Hnsw::new(cfg(32)).unwrap();
        assert!(
            Migration::start(Hnsw::new(cfg(16)).unwrap(), v1.clone(), cfg(48), v2.clone()).is_err()
        );
        assert!(
            Migration::start(Hnsw::new(cfg(32)).unwrap(), v1.clone(), cfg(16), v2.clone()).is_err()
        );
        let m = Migration::start(old, v1, cfg(48), v2).unwrap();
        assert!(Migration::finalize(m).is_err());
        let (corpus, mut m) = setup(20);
        let no_probes = m.verify(&corpus, &[], 5, 0.5).unwrap();
        assert!(no_probes.failures.iter().any(|f| f.contains("probe")));
        // Extra keys in the new index (not in corpus) fail the gate.
        m.upsert(&Document {
            key: 777,
            text: "t1w1 orphan".into(),
            tags: 0,
        })
        .unwrap();
        while m.backfill_step(&corpus, 8).unwrap().is_some() {}
        let r = m.verify(&corpus, &probes(), 5, 0.5).unwrap();
        assert_eq!(r.extra, 1);
        assert!(!r.passed);
        assert!(m.cutover(&r).is_err());
        assert!(m
            .upsert(&Document {
                key: 1,
                text: "   ".into(),
                tags: 0
            })
            .is_err());
        let old = m.abort();
        assert_eq!(old.len(), 21);
    }

    #[test]
    fn concurrent_readers_never_fail_during_backfill() {
        let (corpus, m) = setup(400);
        let m = RwLock::new(m);
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|s| {
            let reader = s.spawn(|| {
                let mut served = 0usize;
                while !stop.load(std::sync::atomic::Ordering::Acquire) {
                    let hits = m.read().unwrap().search("t3w3 t3w7", 5).unwrap();
                    assert_eq!(hits.len(), 5);
                    served += 1;
                }
                served
            });
            let writer = s.spawn(|| {
                for i in 0..50u64 {
                    let d = Document {
                        key: 1000 + i,
                        text: format!("t{}w{} live write", i % 6, i % 16),
                        tags: 1,
                    };
                    m.write().unwrap().upsert(&d).unwrap();
                    corpus.upsert(d);
                }
            });
            let batches = backfill_concurrently(&m, &corpus, 32).unwrap();
            writer.join().unwrap();
            // Writes that landed behind the cursor were dual-written, so the new
            // index has them even though the scan never saw them.
            stop.store(true, std::sync::atomic::Ordering::Release);
            assert!(batches >= 13);
            assert!(reader.join().unwrap() > 0);
        });
        let m = m.into_inner().unwrap();
        assert_eq!(m.phase(), Phase::Backfilled);
        let r = m.verify(&corpus, &probes(), 10, 0.85).unwrap();
        assert!(r.passed, "{r:?}");
        assert_eq!(r.new_live, 450);
    }
}
