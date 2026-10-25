//! Recall/QPS benchmark harness.
//!
//! Builds an index over a dataset, then sweeps `efSearch` and for each value
//! measures single-threaded throughput, latency percentiles, mean recall@k
//! against the dataset's ground truth, and the mean number of distance
//! evaluations per query. Results render to JSON, Markdown and an SVG curve.

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::brute::recall_at_k;
use crate::config::HnswConfig;
use crate::dataset::Dataset;
use crate::error::Result;
use crate::hnsw::{Hnsw, SearchParams};

/// Build-phase measurements.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildReport {
    /// Points inserted.
    pub n: usize,
    /// Wall-clock build time in seconds.
    pub seconds: f64,
    /// Inserts per second.
    pub inserts_per_sec: f64,
    /// Highest layer of the resulting graph.
    pub max_level: usize,
    /// Mean layer-0 out-degree.
    pub mean_degree_l0: f64,
}

/// One point of the recall/QPS curve.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepPoint {
    /// Beam width.
    pub ef: usize,
    /// Mean recall@k over all queries.
    pub recall: f64,
    /// Worst single-query recall@k.
    pub min_recall: f64,
    /// Queries per second (single thread).
    pub qps: f64,
    /// Median latency in microseconds.
    pub p50_us: f64,
    /// 99th-percentile latency in microseconds.
    pub p99_us: f64,
    /// Mean distance evaluations per query.
    pub mean_distance_evals: f64,
}

/// Full benchmark result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    /// Dataset name.
    pub dataset: String,
    /// Base vectors.
    pub n: usize,
    /// Dimensionality.
    pub dim: usize,
    /// Number of queries.
    pub queries: usize,
    /// k in recall@k.
    pub k: usize,
    /// Index configuration.
    pub config: HnswConfig,
    /// Build measurements.
    pub build: BuildReport,
    /// Exact linear-scan throughput over the same index (baseline).
    pub brute_force_qps: f64,
    /// The sweep, in the order given.
    pub sweep: Vec<SweepPoint>,
    /// `os/arch` the numbers were measured on.
    pub platform: String,
}

/// Insert every base row (key = row index) and time it.
pub fn build_index(ds: &Dataset, config: HnswConfig) -> Result<(Hnsw, BuildReport)> {
    let mut index = Hnsw::new(config)?;
    let start = Instant::now();
    for (i, v) in ds.base.iter().enumerate() {
        index.insert(i as u64, v)?;
    }
    let seconds = start.elapsed().as_secs_f64();
    let health = index.health();
    let report = BuildReport {
        n: ds.base.len(),
        seconds,
        inserts_per_sec: ds.base.len() as f64 / seconds.max(1e-9),
        max_level: health.max_level,
        mean_degree_l0: health.mean_degree_l0,
    };
    Ok((index, report))
}

/// Measure one `ef` setting. Queries are replayed until at least
/// `min_time` has elapsed so fast settings still get stable throughput numbers.
pub fn measure(
    index: &Hnsw,
    ds: &Dataset,
    k: usize,
    ef: usize,
    min_time: Duration,
) -> Result<SweepPoint> {
    let mut recalls = Vec::with_capacity(ds.queries.len());
    let mut evals = 0usize;
    for (qi, q) in ds.queries.iter().enumerate() {
        let r = index.search_with(q, SearchParams::top(k).ef(ef))?;
        evals += r.stats.distance_evals;
        let found: Vec<u64> = r.hits.iter().map(|h| h.key).collect();
        let truth: Vec<u64> = ds.groundtruth[qi].iter().map(|&x| u64::from(x)).collect();
        recalls.push(recall_at_k(&found, &truth, k));
    }
    let mut latencies: Vec<f64> = Vec::new();
    let start = Instant::now();
    let mut done = 0usize;
    while start.elapsed() < min_time || done == 0 {
        for q in ds.queries.iter() {
            let t = Instant::now();
            let r = index.search_with(q, SearchParams::top(k).ef(ef))?;
            latencies.push(t.elapsed().as_secs_f64() * 1e6);
            std::hint::black_box(r);
            done += 1;
        }
    }
    let total = start.elapsed().as_secs_f64();
    latencies.sort_by(f64::total_cmp);
    let pct = |p: f64| latencies[((latencies.len() - 1) as f64 * p).round() as usize];
    Ok(SweepPoint {
        ef,
        recall: recalls.iter().sum::<f64>() / recalls.len().max(1) as f64,
        min_recall: recalls.iter().copied().fold(1.0, f64::min),
        qps: done as f64 / total.max(1e-9),
        p50_us: pct(0.5),
        p99_us: pct(0.99),
        mean_distance_evals: evals as f64 / ds.queries.len().max(1) as f64,
    })
}

/// Exact-scan throughput baseline.
pub fn brute_force_qps(index: &Hnsw, ds: &Dataset, k: usize, min_time: Duration) -> Result<f64> {
    let start = Instant::now();
    let mut done = 0usize;
    while start.elapsed() < min_time || done == 0 {
        for q in ds.queries.iter() {
            std::hint::black_box(index.exact_search(q, k, None)?);
            done += 1;
        }
    }
    Ok(done as f64 / start.elapsed().as_secs_f64().max(1e-9))
}

/// Build, then sweep every `ef` in `efs`.
pub fn run(
    name: &str,
    ds: &Dataset,
    config: HnswConfig,
    k: usize,
    efs: &[usize],
    min_time: Duration,
) -> Result<BenchReport> {
    let (index, build) = build_index(ds, config.clone())?;
    tracing::info!(n = build.n, seconds = build.seconds, "index built");
    let mut sweep = Vec::with_capacity(efs.len());
    for &ef in efs {
        let p = measure(&index, ds, k, ef, min_time)?;
        tracing::info!(ef, recall = p.recall, qps = p.qps, "sweep point");
        sweep.push(p);
    }
    let brute = brute_force_qps(&index, ds, k, min_time)?;
    Ok(BenchReport {
        dataset: name.to_string(),
        n: ds.base.len(),
        dim: ds.base.dim(),
        queries: ds.queries.len(),
        k,
        config,
        build,
        brute_force_qps: brute,
        sweep,
        platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
    })
}

/// Markdown table of a report.
pub fn render_markdown(r: &BenchReport) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "### {} — n={}, d={}, {} queries, recall@{} (M={}, efConstruction={}, {})\n",
        r.dataset, r.n, r.dim, r.queries, r.k, r.config.m, r.config.ef_construction, r.platform
    );
    let _ = writeln!(
        s,
        "Build: {:.2} s ({:.0} inserts/s), max level {}, mean layer-0 degree {:.1}. Brute-force baseline: {:.0} QPS.\n",
        r.build.seconds, r.build.inserts_per_sec, r.build.max_level, r.build.mean_degree_l0, r.brute_force_qps
    );
    let _ = writeln!(
        s,
        "| efSearch | recall@{k} | min recall | QPS | speed-up vs brute | p50 µs | p99 µs | dist evals/query |",
        k = r.k
    );
    let _ = writeln!(s, "|---:|---:|---:|---:|---:|---:|---:|---:|");
    for p in &r.sweep {
        let _ = writeln!(
            s,
            "| {} | {:.4} | {:.2} | {:.0} | {:.1}x | {:.1} | {:.1} | {:.0} |",
            p.ef,
            p.recall,
            p.min_recall,
            p.qps,
            p.qps / r.brute_force_qps.max(1e-9),
            p.p50_us,
            p.p99_us,
            p.mean_distance_evals
        );
    }
    s
}

/// SVG line chart of QPS (log scale) against recall, one point per `ef`.
pub fn render_svg(r: &BenchReport) -> String {
    let (w, h) = (640.0, 400.0);
    let (left, right, top, bottom) = (70.0, 20.0, 40.0, 50.0);
    let pw = w - left - right;
    let ph = h - top - bottom;
    let rmin = r
        .sweep
        .iter()
        .map(|p| p.recall)
        .fold(1.0, f64::min)
        .min(0.9)
        .max(0.0);
    let rmin = (rmin * 20.0).floor() / 20.0;
    let qmax = r.sweep.iter().map(|p| p.qps).fold(1.0, f64::max);
    let qmin = r
        .sweep
        .iter()
        .map(|p| p.qps)
        .fold(qmax, f64::min)
        .min(r.brute_force_qps)
        .max(1.0);
    let (lo, hi) = (qmin.log10().floor(), qmax.log10().ceil().max(qmin.log10().floor() + 1.0));
    let x = |rec: f64| left + (rec - rmin) / (1.0 - rmin).max(1e-9) * pw;
    let y = |q: f64| top + ph - (q.max(1.0).log10() - lo) / (hi - lo) * ph;
    let mut s = String::new();
    let _ = write!(
        s,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="sans-serif" font-size="12">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{cx}" y="22" text-anchor="middle" font-size="14">{name}: QPS vs recall@{k} (M={m}, efC={efc})</text>
"##,
        cx = w / 2.0,
        name = r.dataset,
        k = r.k,
        m = r.config.m,
        efc = r.config.ef_construction
    );
    let steps = ((1.0 - rmin) / 0.05).round() as usize;
    for i in 0..=steps {
        let rec = rmin + i as f64 * 0.05;
        let _ = writeln!(
            s,
            r##"<line x1="{0:.1}" y1="{1}" x2="{0:.1}" y2="{2}" stroke="#e5e7eb"/><text x="{0:.1}" y="{3}" text-anchor="middle">{4:.2}</text>"##,
            x(rec),
            top,
            top + ph,
            top + ph + 18.0,
            rec
        );
    }
    let mut e = lo;
    while e <= hi {
        let q = 10f64.powf(e);
        let _ = writeln!(
            s,
            r##"<line x1="{0}" y1="{1:.1}" x2="{2}" y2="{1:.1}" stroke="#e5e7eb"/><text x="{3}" y="{4:.1}" text-anchor="end">{5}</text>"##,
            left,
            y(q),
            left + pw,
            left - 6.0,
            y(q) + 4.0,
            q
        );
        e += 1.0;
    }
    let pts: Vec<String> = r
        .sweep
        .iter()
        .map(|p| format!("{:.1},{:.1}", x(p.recall), y(p.qps)))
        .collect();
    let _ = writeln!(
        s,
        r##"<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{}"/>"##,
        pts.join(" ")
    );
    for p in &r.sweep {
        let _ = writeln!(
            s,
            r##"<circle cx="{0:.1}" cy="{1:.1}" r="3.5" fill="#2563eb"/><text x="{0:.1}" y="{2:.1}" text-anchor="middle" fill="#1e3a8a">ef={3}</text>"##,
            x(p.recall),
            y(p.qps),
            y(p.qps) - 8.0,
            p.ef
        );
    }
    let by = y(r.brute_force_qps);
    let _ = writeln!(
        s,
        r##"<line x1="{left}" y1="{by:.1}" x2="{0}" y2="{by:.1}" stroke="#dc2626" stroke-dasharray="6 4"/><text x="{0}" y="{1:.1}" text-anchor="end" fill="#dc2626">brute force {2:.0} QPS (recall 1.0)</text>"##,
        left + pw,
        by - 6.0,
        r.brute_force_qps
    );
    let _ = writeln!(
        s,
        r##"<text x="{0}" y="{1}" text-anchor="middle">recall@{2}</text><text x="16" y="{3}" text-anchor="middle" transform="rotate(-90 16 {3})">queries/second (log, 1 thread)</text></svg>"##,
        left + pw / 2.0,
        h - 10.0,
        r.k,
        top + ph / 2.0
    );
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute::exact_knn;
    use crate::dataset::gaussian_mixture;
    use crate::distance::Metric;

    fn small_dataset() -> Dataset {
        let base = gaussian_mixture(600, 12, 6, 2.0, 11);
        let queries = gaussian_mixture(20, 12, 6, 2.0, 12);
        let groundtruth = exact_knn(&base, &queries, 10, Metric::L2, 2).unwrap();
        Dataset {
            base,
            queries,
            groundtruth,
        }
    }

    #[test]
    fn sweep_produces_monotone_work_and_high_recall() {
        let ds = small_dataset();
        let cfg = HnswConfig::new(12).with_m(8).with_ef_construction(64);
        let r = run("tiny", &ds, cfg, 10, &[10, 50, 200], Duration::from_millis(5)).unwrap();
        assert_eq!(r.sweep.len(), 3);
        assert!(r.sweep[0].mean_distance_evals < r.sweep[2].mean_distance_evals);
        assert!(r.sweep[2].recall > 0.97, "recall {}", r.sweep[2].recall);
        assert!(r.brute_force_qps > 0.0);
        let md = render_markdown(&r);
        assert!(md.contains("| 200 |"));
        let svg = render_svg(&r);
        assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
        assert_eq!(svg.matches("<circle").count(), 3);
        let json = serde_json::to_string(&r).unwrap();
        let back: BenchReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sweep.len(), 3);
    }
}
