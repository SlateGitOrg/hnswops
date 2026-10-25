//! `hnswops demo`: an offline, end-to-end tour of every subsystem against the
//! committed synthetic fixture. Every number printed is measured on the spot.

use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use hnswops::dataset::{self, Dataset};
use hnswops::embed::{synthetic_corpus, synthetic_query, Corpus, Document, Embedder, HashingEmbedder, InMemoryCorpus};
use hnswops::migration::backfill_concurrently;
use hnswops::monitor::AutotunePolicy;
use hnswops::persist::LoadOptions;
use hnswops::{
    Error, Filter, Hnsw, HnswConfig, Metric, Migration, MonitorConfig, MonitoredIndex, Result, SearchParams,
    SnapshotStore,
};

use crate::cli::{build_tagged, mean_recall};

fn section(title: &str) {
    println!("\n== {title} ==");
}

/// Mean recall@k of `index` against exact search over its *current* live set.
fn live_recall(index: &Hnsw, ds: &Dataset, k: usize, ef: usize) -> Result<f64> {
    let mut total = 0.0;
    for q in ds.queries.iter() {
        let got: Vec<u64> = index
            .search_with(q, SearchParams::top(k).ef(ef))?
            .hits
            .iter()
            .map(|h| h.key)
            .collect();
        let truth: Vec<u64> = index.exact_search(q, k, None)?.iter().map(|h| h.key).collect();
        total += hnswops::brute::recall_at_k(&got, &truth, k);
    }
    Ok(total / ds.queries.len().max(1) as f64)
}

/// Run the demo.
pub fn run(fixtures: &Path, workdir: &Path) -> Result<()> {
    if workdir.exists() {
        fs::remove_dir_all(workdir).map_err(|e| Error::io(workdir, e))?;
    }
    fs::create_dir_all(workdir).map_err(|e| Error::io(workdir, e))?;
    let k = 10;

    section("1. dataset");
    let ds = dataset::load_dataset(fixtures, &dataset::FIXTURE)?;
    println!(
        "fixture: {} base x {}d, {} held-out queries, exact top-{} ground truth (sha256 verified)",
        ds.base.len(),
        ds.base.dim(),
        ds.queries.len(),
        ds.groundtruth[0].len()
    );

    section("2. build (M=16, efConstruction=200)");
    let config = HnswConfig::new(ds.base.dim()).with_m(16).with_ef_construction(200);
    let t = Instant::now();
    let mut index = build_tagged(&ds, config)?;
    let h = index.health();
    println!(
        "inserted {} points in {:.2}s; max level {}, level histogram {:?}, mean layer-0 degree {:.1}",
        index.len(),
        t.elapsed().as_secs_f64(),
        h.max_level,
        h.level_histogram,
        h.mean_degree_l0
    );

    if h.unreachable_l0 > 0 {
        let fixed = index.repair_connectivity();
        println!(
            "health: {} node(s) unreachable on layer 0 after build (heuristic pruned their last in-edge); repair_connectivity re-linked {fixed} -> unreachable={}",
            h.unreachable_l0,
            index.health().unreachable_l0
        );
    } else {
        println!("health: every live node reachable from the entry point on layer 0");
    }

    section("3. recall@10 vs ground truth (held-out queries)");
    for ef in [10, 16, 32, 64, 128] {
        let t = Instant::now();
        let r = mean_recall(&index, &ds, k, ef)?;
        let us = t.elapsed().as_secs_f64() * 1e6 / ds.queries.len() as f64;
        println!("efSearch={ef:<4} recall@10={r:.4}  ({us:.0} µs/query incl. scoring)");
    }

    section("4. filtered search (cost-based planner)");
    let q = ds.queries.get(0);
    for (label, filter) in [
        ("tenant 3 (~12.5%)", Filter::TagsAny(1 << 3)),
        ("flagged (~2%)", Filter::TagsAny(1 << 8)),
        ("tenant 3 AND key<5000", Filter::parse("tags_any=0x8,key=0..5000")?),
        ("NOT flagged (~98%)", Filter::parse("!tags_any=0x100")?),
    ] {
        let plan = index.plan_filter(&filter, index.config().ef_search);
        let r = index.search_with(q, SearchParams::top(k).filter(&filter))?;
        let exact: Vec<u64> = index.exact_search(q, k, Some(&filter))?.iter().map(|h| h.key).collect();
        let got: Vec<u64> = r.hits.iter().map(|h| h.key).collect();
        println!(
            "{label:<24} selectivity≈{:.3} est.cost graph={:<6.0} scan={:<6.0} -> {:?} ef_used={} dist_evals={} recall={:.2}",
            plan.selectivity,
            plan.graph_cost,
            plan.scan_cost,
            r.stats.strategy,
            r.stats.ef_used,
            r.stats.distance_evals,
            hnswops::brute::recall_at_k(&got, &exact, k)
        );
    }

    section("5. delete 30% (tombstones + local repair), then vacuum");
    let before = live_recall(&index, &ds, k, 64)?;
    let t = Instant::now();
    let deleted: Vec<u64> = (0..ds.base.len() as u64).filter(|key| key % 10 < 3).collect();
    for &key in &deleted {
        index.delete(key)?;
    }
    println!(
        "deleted {} keys in {:.2}s; live={} tombstones={}",
        deleted.len(),
        t.elapsed().as_secs_f64(),
        index.len(),
        index.tombstones()
    );
    let h = index.health();
    let leaked = ds
        .queries
        .iter()
        .flat_map(|q| index.search(q, k).unwrap_or_default())
        .filter(|hit| hit.key % 10 < 3)
        .count();
    println!(
        "before vacuum: dangling edges={} unreachable(l0)={} deleted keys returned={} recall@10(ef=64, vs live exact)={:.4} (was {before:.4})",
        h.dangling_edges,
        h.unreachable_l0,
        leaked,
        live_recall(&index, &ds, k, 64)?
    );
    let t = Instant::now();
    let v = index.vacuum();
    let h = index.health();
    println!(
        "vacuum in {:.2}s: removed={} lists repaired={} reconnected={} -> dangling={} unreachable={} recall@10={:.4}",
        t.elapsed().as_secs_f64(),
        v.removed,
        v.lists_repaired,
        v.reconnected,
        h.dangling_edges,
        h.unreachable_l0,
        live_recall(&index, &ds, k, 64)?
    );
    for (i, &key) in deleted.iter().enumerate() {
        index.insert_with_tags(key, ds.base.get(key as usize), crate::cli::fixture_tags(key))?;
        if i + 1 == deleted.len() {
            println!(
                "re-inserted the {} keys: live={} recall@10 vs original ground truth={:.4}",
                deleted.len(),
                index.len(),
                mean_recall(&index, &ds, k, 64)?
            );
        }
    }

    section("6. versioned snapshots (atomic publish, mmap'd load, rollback)");
    let store = SnapshotStore::open(workdir.join("index-store"))?;
    let v1 = store.commit(&index, "initial")?;
    for key in (0..500u64).step_by(2) {
        index.delete(key)?;
    }
    let v2 = store.commit(&index, "after-deletes")?;
    let t = Instant::now();
    let (cur, loaded) = store.load_current(LoadOptions { verify_vectors: false })?;
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let same = ds
        .queries
        .iter()
        .all(|q| loaded.search(q, k).ok() == index.search(q, k).ok());
    println!(
        "committed v{} ({} live, {} bytes) and v{} ({} live); CURRENT=v{cur}",
        v1.version, v1.live, v1.bytes, v2.version, v2.live
    );
    println!(
        "loaded CURRENT in {load_ms:.1} ms with {} vectors memory-mapped; results identical to in-memory index: {same}",
        loaded.mapped_vectors()
    );
    store.rollback(v1.version)?;
    let (cur, rolled) = store.load_current(LoadOptions::default())?;
    println!(
        "rollback -> CURRENT=v{cur}, live={} (checksums verified), versions={:?}",
        rolled.len(),
        store.versions()?
    );

    section("7. online recall monitoring (sampled brute-force shadow queries)");
    let mut serving = rolled;
    serving.set_ef_search(10)?;
    let monitor_cfg = MonitorConfig {
        sample_rate: 0.25,
        window: 128,
        min_samples: 32,
        target_recall: 0.97,
        ..MonitorConfig::default()
    };
    let mi = Arc::new(MonitoredIndex::new(serving, monitor_cfg)?);
    let worker = mi.spawn_shadow_worker(std::time::Duration::from_millis(2), 64);
    let serve = |rounds: usize| -> Result<()> {
        for _ in 0..rounds {
            for q in ds.queries.iter() {
                mi.search(q, SearchParams::top(k))?;
            }
        }
        Ok(())
    };
    let policy = AutotunePolicy::default();
    for step in 0..6 {
        serve(2)?;
        while mi.report().pending > 0 {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let r = mi.report();
        let ef = mi.with_index(|i| i.config().ef_search);
        println!(
            "step {step}: efSearch={ef:<4} status={:?} window={} mean={:.4} p10={:.2} alerts={} dropped={}",
            r.status, r.window_samples, r.mean_recall, r.p10_recall, r.alerts, r.dropped
        );
        match mi.autotune(&policy) {
            Some(next) => println!("        autotune: efSearch {ef} -> {next}"),
            None if r.status == hnswops::RecallStatus::Healthy => break,
            None => {}
        }
    }
    worker.stop();

    section("8. zero-downtime re-embedding migration (hash-v1 64d -> hash-v2 128d)");
    let v1_model: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new("hash-v1", 64, 1, 1)?);
    let v2_model: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new("hash-v2", 128, 2, 2)?);
    let corpus = InMemoryCorpus::from_docs(synthetic_corpus(4000, 12, 7));
    let mut old = Hnsw::new(HnswConfig::new(64).with_metric(Metric::Cosine).with_m(12).with_ef_construction(100))?;
    for d in corpus.scan(None, usize::MAX) {
        old.insert_with_tags(d.key, &v1_model.embed(&d.text)?, d.tags)?;
    }
    let docs_store = SnapshotStore::open(workdir.join("docs-store"))?;
    let pre = docs_store.commit(&old, "pre-migration hash-v1")?;
    println!("old index: {} docs, snapshot v{} labelled {:?}", old.len(), pre.version, pre.label);
    let migration = RwLock::new(Migration::start(
        old,
        Arc::clone(&v1_model),
        HnswConfig::new(128).with_metric(Metric::Cosine).with_m(12).with_ef_construction(100),
        Arc::clone(&v2_model),
    )?);
    let reads = AtomicUsize::new(0);
    let read_errors = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let t = Instant::now();
    let batches = std::thread::scope(|s| -> Result<usize> {
        for r in 0..2 {
            let (m, reads, read_errors, stop) = (&migration, &reads, &read_errors, &stop);
            s.spawn(move || {
                let mut i = 0u64;
                while !stop.load(Ordering::Acquire) {
                    let q = synthetic_query((i % 12) as usize, i * 2 + r);
                    match m.read().map(|g| g.search(&q, 5)) {
                        Ok(Ok(h)) if h.len() == 5 => reads.fetch_add(1, Ordering::Relaxed),
                        _ => read_errors.fetch_add(1, Ordering::Relaxed),
                    };
                    i += 1;
                }
            });
        }
        let (m, corpus_ref) = (&migration, &corpus);
        let writer = s.spawn(move || -> Result<()> {
            for i in 0..200u64 {
                let doc = Document {
                    key: 10_000 + i,
                    text: format!("t{}w{} t{}w{} fresh write during migration", i % 12, i % 16, i % 12, (i + 3) % 16),
                    tags: 1,
                };
                m.write().map_err(|_| Error::Migration("lock poisoned".into()))?.upsert(&doc)?;
                corpus_ref.upsert(doc);
                if i % 4 == 0 {
                    m.write().map_err(|_| Error::Migration("lock poisoned".into()))?.delete(i * 7)?;
                    corpus_ref.delete(i * 7);
                }
            }
            Ok(())
        });
        let batches = backfill_concurrently(&migration, &corpus, 128)?;
        writer.join().map_err(|_| Error::Migration("writer panicked".into()))??;
        stop.store(true, Ordering::Release);
        Ok(batches)
    })?;
    let mut m = migration.into_inner().map_err(|_| Error::Migration("lock poisoned".into()))?;
    let st = m.stats().clone();
    println!(
        "backfill: {batches} batches in {:.2}s while serving {} reads ({} failed) and {} dual-writes/{} dual-deletes",
        t.elapsed().as_secs_f64(),
        reads.load(Ordering::Relaxed),
        read_errors.load(Ordering::Relaxed),
        st.dual_writes,
        st.dual_deletes
    );
    println!(
        "backfilled={} skipped(stale CAS)={} skipped(already dual-written)={} phase={:?}",
        st.backfilled,
        st.skipped_stale,
        st.skipped_fresh,
        m.phase()
    );
    let probes: Vec<String> = (0..48).map(|i| synthetic_query(i % 12, 1000 + i as u64)).collect();
    let report = m.verify(&corpus, &probes, k, 0.95)?;
    println!(
        "verify: corpus={} new_live={} missing={} extra={} new-index recall@10={:.4} old/new top-10 overlap={:.2} passed={}",
        report.corpus_docs,
        report.new_live,
        report.missing,
        report.extra,
        report.new_index_recall,
        report.old_new_overlap,
        report.passed
    );
    let probe = synthetic_query(4, 99);
    let before: Vec<u64> = m.search(&probe, 5)?.iter().map(|h| h.key).collect();
    m.cutover(&report)?;
    let after: Vec<u64> = m.search(&probe, 5)?.iter().map(|h| h.key).collect();
    println!("query {probe:?}: old model top-5 {before:?} -> new model top-5 {after:?}");
    m.rollback()?;
    println!("rollback: reads back on {} (lossless; writes kept flowing to both)", m.read_embedder().name());
    m.cutover(&report)?;
    let new_index = m.finalize()?;
    let post = docs_store.commit(&new_index, "post-migration hash-v2")?;
    println!(
        "finalized: new index {} docs x {}d, snapshot v{} labelled {:?}",
        new_index.len(),
        new_index.config().dim,
        post.version,
        post.label
    );
    println!("\ndemo complete; artifacts in {}", workdir.display());
    Ok(())
}
