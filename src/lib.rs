//! # hnswops
//!
//! A Hierarchical Navigable Small World (HNSW) approximate-nearest-neighbor
//! index written from scratch, plus the operational tooling a vector-search
//! service needs around it:
//!
//! * [`Hnsw`] — insert/upsert, delete (tombstones + local repair), vacuum,
//!   filtered search with a selectivity-based planner, health reports.
//! * [`persist`] — checksummed on-disk images whose vectors are memory-mapped.
//! * [`snapshot`] — versioned, atomically published snapshots with rollback
//!   and retention.
//! * [`monitor`] — online recall monitoring via sampled brute-force shadow queries.
//! * [`migration`] — zero-downtime re-embedding (dual-write, backfill,
//!   verification gates, atomic cutover, rollback).
//! * [`dataset`] / [`bench`] — `.fvecs`/`.ivecs` I/O, checksummed benchmark
//!   data, recall/QPS sweeps.
//!
//! ```
//! use hnswops::{Hnsw, HnswConfig};
//!
//! let mut index = Hnsw::new(HnswConfig::new(2).with_m(8)).unwrap();
//! for i in 0..100u64 {
//!     index.insert(i, &[(i % 10) as f32, (i / 10) as f32]).unwrap();
//! }
//! index.delete(43).unwrap();
//! let hits = index.search(&[3.0, 4.0], 3).unwrap();
//! assert!(hits.iter().all(|h| h.key != 43));
//! ```
#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod bench;
pub mod brute;
pub mod config;
pub mod dataset;
pub mod distance;
pub mod embed;
pub mod error;
pub mod hnsw;
pub mod migration;
pub mod monitor;
pub mod persist;
pub mod rng;
pub mod snapshot;
pub mod storage;
pub mod visited;

pub use config::HnswConfig;
pub use distance::Metric;
pub use error::{Error, Result};
pub use hnsw::{Filter, FilterPlan, HealthReport, Hit, Hnsw, SearchParams, SearchResult, SearchStats, Strategy, VacuumReport};
pub use migration::{Migration, Phase};
pub use monitor::{MonitorConfig, MonitoredIndex, RecallReport, RecallStatus};
pub use snapshot::SnapshotStore;
