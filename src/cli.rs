//! Argument parsing and the non-demo subcommands.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

use hnswops::bench::{self, render_markdown, render_svg};
use hnswops::brute::recall_at_k;
use hnswops::dataset::{self, Dataset, DatasetSpec, FixtureSpec};
use hnswops::persist::LoadOptions;
use hnswops::snapshot::SnapshotStore;
use hnswops::{Error, Filter, Hnsw, HnswConfig, Metric, Result, SearchParams};

/// HNSW vector index with recall monitoring, snapshots and re-embedding migration.
#[derive(Debug, Parser)]
#[command(name = "hnswops", version, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Which pinned dataset layout to read.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum DatasetName {
    /// INRIA ANN_SIFT10K (fetch with `make fetch-data`).
    Siftsmall,
    /// The synthetic fixture committed in `fixtures/`.
    Fixture,
}

impl DatasetName {
    fn spec(self) -> &'static DatasetSpec {
        match self {
            DatasetName::Siftsmall => &dataset::SIFT_SMALL,
            DatasetName::Fixture => &dataset::FIXTURE,
        }
    }
}

#[derive(Debug, clap::Args)]
struct DataArgs {
    /// Dataset layout / checksum set.
    #[arg(long, value_enum, default_value = "fixture")]
    dataset: DatasetName,
    /// Directory holding the dataset files.
    #[arg(long, default_value = "fixtures")]
    data: PathBuf,
}

impl DataArgs {
    fn load(&self) -> Result<Dataset> {
        dataset::load_dataset(&self.data, self.dataset.spec())
    }
}

#[derive(Debug, clap::Args)]
struct IndexArgs {
    /// Out-degree on upper layers (layer 0 uses 2*M).
    #[arg(long, default_value_t = 16)]
    m: usize,
    /// Beam width while building.
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    /// Default beam width while searching.
    #[arg(long, default_value_t = 64)]
    ef_search: usize,
    /// Distance metric: l2, cosine or ip.
    #[arg(long, default_value = "l2")]
    metric: Metric,
    /// Level-assignment seed.
    #[arg(long, default_value_t = 0x005E_ED0F_4A55)]
    seed: u64,
}

impl IndexArgs {
    fn config(&self, dim: usize) -> HnswConfig {
        HnswConfig::new(dim)
            .with_m(self.m)
            .with_ef_construction(self.ef_construction)
            .with_ef_search(self.ef_search)
            .with_metric(self.metric)
            .with_seed(self.seed)
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate the synthetic benchmark fixture (TEXMEX layout, exact ground truth).
    GenFixture {
        /// Output directory.
        #[arg(long, default_value = "fixtures")]
        out: PathBuf,
        /// Base vectors.
        #[arg(long, default_value_t = 10_000)]
        n: usize,
        /// Held-out queries.
        #[arg(long, default_value_t = 200)]
        queries: usize,
        /// Dimensionality.
        #[arg(long, default_value_t = 32)]
        dim: usize,
        /// Mixture components.
        #[arg(long, default_value_t = 32)]
        clusters: usize,
        /// Ground-truth depth.
        #[arg(long, default_value_t = 100)]
        k: usize,
        /// Seed.
        #[arg(long, default_value_t = 41)]
        seed: u64,
    },
    /// Verify a dataset's pinned SHA-256 checksums and shapes.
    VerifyData(DataArgs),
    /// Build an index over a dataset and commit it as a new snapshot version.
    Build {
        #[command(flatten)]
        data: DataArgs,
        #[command(flatten)]
        index: IndexArgs,
        /// Snapshot store directory.
        #[arg(long)]
        store: PathBuf,
        /// Snapshot label.
        #[arg(long, default_value = "build")]
        label: String,
    },
    /// Recall@k of the CURRENT snapshot (vectors mmap'd) against ground truth.
    Eval {
        #[command(flatten)]
        data: DataArgs,
        /// Snapshot store directory.
        #[arg(long)]
        store: PathBuf,
        /// Neighbors per query.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Beam width (defaults to the index's efSearch).
        #[arg(long)]
        ef: Option<usize>,
    },
    /// Search the CURRENT snapshot with one of the dataset's query vectors.
    Search {
        #[command(flatten)]
        data: DataArgs,
        /// Snapshot store directory.
        #[arg(long)]
        store: PathBuf,
        /// Row of the query file to use.
        #[arg(long, default_value_t = 0)]
        query: usize,
        /// Neighbors to return.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Beam width.
        #[arg(long)]
        ef: Option<usize>,
        /// Filter, e.g. `tags_any=0x3,!key=0..100`.
        #[arg(long)]
        filter: Option<String>,
    },
    /// Build, then sweep efSearch and write the recall/QPS curve (JSON, Markdown, SVG).
    Bench {
        #[command(flatten)]
        data: DataArgs,
        #[command(flatten)]
        index: IndexArgs,
        /// recall@k.
        #[arg(long, default_value_t = 10)]
        k: usize,
        /// Comma-separated efSearch values.
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "10,16,24,32,48,64,96,128,256"
        )]
        efs: Vec<usize>,
        /// Minimum measuring time per point, in milliseconds.
        #[arg(long, default_value_t = 500)]
        min_ms: u64,
        /// Output prefix; writes `<out>.json`, `<out>.md` and `<out>.svg`.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Inspect and manage snapshot versions.
    Snapshot {
        /// Snapshot store directory.
        #[arg(long)]
        store: PathBuf,
        #[command(subcommand)]
        action: SnapshotAction,
    },
    /// Structural health report of the CURRENT snapshot.
    Health {
        /// Snapshot store directory.
        #[arg(long)]
        store: PathBuf,
    },
    /// Offline end-to-end demo against the bundled fixture.
    Demo {
        /// Fixture directory.
        #[arg(long, default_value = "fixtures")]
        fixtures: PathBuf,
        /// Scratch directory for snapshot stores (recreated).
        #[arg(long, default_value = "demo-out")]
        workdir: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum SnapshotAction {
    /// List versions.
    List,
    /// Verify checksums and graph invariants of a version.
    Verify {
        /// Version number.
        version: u64,
    },
    /// Point CURRENT at a version.
    Rollback {
        /// Version number.
        version: u64,
    },
    /// Delete all but the newest N versions (CURRENT is always kept).
    Prune {
        /// Versions to keep.
        #[arg(long, default_value_t = 3)]
        keep: usize,
    },
}

/// Tags assigned to fixture/benchmark points: one of 8 "tenants" by key,
/// plus a rare "flagged" bit on every 50th key. Used by filtered-search demos.
pub fn fixture_tags(key: u64) -> u64 {
    let tenant = 1u64 << (key % 8);
    if key % 50 == 0 {
        tenant | 1 << 8
    } else {
        tenant
    }
}

fn print_json<T: Serialize>(value: &T) -> Result<()> {
    let s = serde_json::to_string_pretty(value).map_err(|source| Error::Json {
        path: PathBuf::from("<stdout>"),
        source,
    })?;
    println!("{s}");
    Ok(())
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    fs::write(path, text).map_err(|e| Error::io(path, e))
}

/// Build an index over a dataset's base vectors (key = row, tags = [`fixture_tags`]).
pub fn build_tagged(ds: &Dataset, config: HnswConfig) -> Result<Hnsw> {
    let mut index = Hnsw::new(config)?;
    for (i, v) in ds.base.iter().enumerate() {
        index.insert_with_tags(i as u64, v, fixture_tags(i as u64))?;
    }
    Ok(index)
}

/// Mean recall@k of `index` over the dataset's held-out queries.
pub fn mean_recall(index: &Hnsw, ds: &Dataset, k: usize, ef: usize) -> Result<f64> {
    let mut total = 0.0;
    for (qi, q) in ds.queries.iter().enumerate() {
        let hits = index.search_with(q, SearchParams::top(k).ef(ef))?.hits;
        let found: Vec<u64> = hits.iter().map(|h| h.key).collect();
        let truth: Vec<u64> = ds.groundtruth[qi].iter().map(|&x| u64::from(x)).collect();
        total += recall_at_k(&found, &truth, k);
    }
    Ok(total / ds.queries.len().max(1) as f64)
}

/// Dispatch a parsed command line.
pub fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::GenFixture {
            out,
            n,
            queries,
            dim,
            clusters,
            k,
            seed,
        } => {
            let spec = FixtureSpec {
                n,
                queries,
                dim,
                clusters,
                k,
                seed,
                ..FixtureSpec::default()
            };
            dataset::write_synthetic_fixture(&out, &spec)?;
            for f in [
                dataset::FIXTURE.base,
                dataset::FIXTURE.query,
                dataset::FIXTURE.groundtruth,
            ] {
                let p = out.join(f.name);
                println!("{}  {}", dataset::sha256_file(&p)?, p.display());
            }
            Ok(())
        }
        Command::VerifyData(data) => {
            let ds = data.load()?;
            println!(
                "ok: {} base x {}d, {} queries, ground truth depth {}",
                ds.base.len(),
                ds.base.dim(),
                ds.queries.len(),
                ds.groundtruth.first().map_or(0, Vec::len)
            );
            Ok(())
        }
        Command::Build {
            data,
            index,
            store,
            label,
        } => {
            let ds = data.load()?;
            let start = std::time::Instant::now();
            let idx = build_tagged(&ds, index.config(ds.base.dim()))?;
            let secs = start.elapsed().as_secs_f64();
            let info = SnapshotStore::open(&store)?.commit(&idx, &label)?;
            println!(
                "built {} points in {secs:.2}s; committed snapshot v{} ({} bytes) to {}",
                idx.len(),
                info.version,
                info.bytes,
                store.display()
            );
            Ok(())
        }
        Command::Eval { data, store, k, ef } => {
            let ds = data.load()?;
            let (version, idx) =
                SnapshotStore::open(&store)?.load_current(LoadOptions::default())?;
            let ef = ef.unwrap_or(idx.config().ef_search);
            let recall = mean_recall(&idx, &ds, k, ef)?;
            print_json(&serde_json::json!({
                "version": version,
                "mapped_vectors": idx.mapped_vectors(),
                "k": k,
                "ef": ef,
                "queries": ds.queries.len(),
                "recall": recall,
            }))
        }
        Command::Search {
            data,
            store,
            query,
            k,
            ef,
            filter,
        } => {
            let ds = data.load()?;
            if query >= ds.queries.len() {
                return Err(Error::InvalidConfig(format!(
                    "query row {query} out of range (dataset has {})",
                    ds.queries.len()
                )));
            }
            let (_, idx) = SnapshotStore::open(&store)?.load_current(LoadOptions::default())?;
            let filter = filter.as_deref().map(Filter::parse).transpose()?;
            let mut params = SearchParams::top(k);
            if let Some(ef) = ef {
                params = params.ef(ef);
            }
            if let Some(f) = &filter {
                params = params.filter(f);
            }
            let q = ds.queries.get(query);
            let result = idx.search_with(q, params)?;
            let exact = idx.exact_search(q, k, filter.as_ref())?;
            print_json(&serde_json::json!({
                "strategy": result.stats.strategy,
                "ef_used": result.stats.ef_used,
                "distance_evals": result.stats.distance_evals,
                "hits": result.hits,
                "exact_keys": exact.iter().map(|h| h.key).collect::<Vec<_>>(),
            }))
        }
        Command::Bench {
            data,
            index,
            k,
            efs,
            min_ms,
            out,
        } => {
            let ds = data.load()?;
            let name = format!("{:?}", data.dataset).to_lowercase();
            let report = bench::run(
                &name,
                &ds,
                index.config(ds.base.dim()),
                k,
                &efs,
                Duration::from_millis(min_ms),
            )?;
            let md = render_markdown(&report);
            print!("{md}");
            if let Some(prefix) = out {
                let json = serde_json::to_string_pretty(&report).map_err(|source| Error::Json {
                    path: prefix.clone(),
                    source,
                })?;
                write_text(&prefix.with_extension("json"), &json)?;
                write_text(&prefix.with_extension("md"), &md)?;
                write_text(&prefix.with_extension("svg"), &render_svg(&report))?;
                println!("\nwrote {}.{{json,md,svg}}", prefix.display());
            }
            Ok(())
        }
        Command::Snapshot { store, action } => {
            let store = SnapshotStore::open(&store)?;
            match action {
                SnapshotAction::List => {
                    let current = store.current()?;
                    for info in store.list()? {
                        let mark = if Some(info.version) == current {
                            "*"
                        } else {
                            " "
                        };
                        println!(
                            "{mark} v{:06}  live={:<8} bytes={:<10} parent={:<8} label={}",
                            info.version,
                            info.live,
                            info.bytes,
                            info.parent.map_or("-".to_string(), |p| format!("v{p:06}")),
                            info.label
                        );
                    }
                    Ok(())
                }
                SnapshotAction::Verify { version } => {
                    let m = store.verify(version)?;
                    println!(
                        "v{version:06} ok: {} slots, {} live, vectors crc32 {:08x}, graph crc32 {:08x}",
                        m.slots, m.live, m.vectors.crc32, m.graph.crc32
                    );
                    Ok(())
                }
                SnapshotAction::Rollback { version } => {
                    store.rollback(version)?;
                    println!("CURRENT -> v{version:06}");
                    Ok(())
                }
                SnapshotAction::Prune { keep } => {
                    let removed = store.prune(keep)?;
                    println!("removed {removed:?}");
                    Ok(())
                }
            }
        }
        Command::Health { store } => {
            let (version, idx) =
                SnapshotStore::open(&store)?.load_current(LoadOptions::default())?;
            print_json(&serde_json::json!({ "version": version, "health": idx.health() }))
        }
        Command::Demo { fixtures, workdir } => crate::demo::run(&fixtures, &workdir),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_mark_tenants_and_rare_flag() {
        assert_eq!(fixture_tags(0), 1 | 1 << 8);
        assert_eq!(fixture_tags(3), 1 << 3);
        assert_eq!(fixture_tags(9), 1 << 1);
        assert_eq!(fixture_tags(100), (1 << 4) | (1 << 8));
    }

    #[test]
    fn cli_parses() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        let c = Cli::try_parse_from([
            "hnswops",
            "bench",
            "--efs",
            "8,16",
            "--dataset",
            "siftsmall",
        ])
        .unwrap();
        assert!(matches!(c.command, Command::Bench { ref efs, .. } if efs == &vec![8, 16]));
    }
}
