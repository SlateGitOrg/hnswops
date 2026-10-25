//! Error type shared by every module.

use std::path::PathBuf;

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// All failures surfaced by the public API.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A configuration value is out of range or inconsistent.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// A vector had the wrong number of components for this index.
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch {
        /// Dimension the index was built with.
        expected: usize,
        /// Dimension of the offending vector.
        got: usize,
    },

    /// A vector contained NaN or infinity, or was all-zero under the cosine metric.
    #[error("invalid vector: {0}")]
    InvalidVector(String),

    /// The requested key is not present (or already deleted).
    #[error("key {0} not found")]
    KeyNotFound(u64),

    /// The index ran out of internal slots (more than `u32::MAX - 1` nodes).
    #[error("index is full: at most {0} nodes are supported")]
    Capacity(usize),

    /// An on-disk structure failed validation (bad magic, checksum, truncation).
    #[error("corrupt data in {path}: {reason}")]
    Corrupt {
        /// File that failed validation.
        path: PathBuf,
        /// What was wrong with it.
        reason: String,
    },

    /// A snapshot version that does not exist was requested.
    #[error("snapshot {0} not found")]
    SnapshotNotFound(String),

    /// A migration operation was called in the wrong phase.
    #[error("migration: {0}")]
    Migration(String),

    /// Underlying I/O failure, with the path it happened on.
    #[error("i/o error on {path}: {source}")]
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },

    /// JSON (de)serialization failure for manifests and reports.
    #[error("json error on {path}: {source}")]
    Json {
        /// File being (de)serialized.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: serde_json::Error,
    },
}

impl Error {
    /// Wrap an I/O error with the path it happened on.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    /// Report a validation failure in `path`.
    pub fn corrupt(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Error::Corrupt {
            path: path.into(),
            reason: reason.into(),
        }
    }
}
