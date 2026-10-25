//! Versioned index snapshots with atomic publication, rollback and retention.
//!
//! Layout under a store root:
//!
//! ```text
//! root/
//!   CURRENT                 # "v000003\n" — the version readers should load
//!   versions/
//!     v000001/              # immutable once published
//!       manifest.json  vectors.bin  graph.bin  snapshot.json
//!     v000002/ ...
//! ```
//!
//! **Atomicity.** A commit writes the image into `versions/.tmp-vNNNNNN/`,
//! fsyncs every file, renames the directory to `vNNNNNN` and fsyncs `versions/`,
//! then replaces `CURRENT` via write-to-temp + rename. A crash at any point
//! leaves either the old or the new `CURRENT`, never a half-written image;
//! leftover `.tmp-*` directories are removed on the next [`SnapshotStore::open`].
//!
//! **Concurrency.** One writer process per store (the commit/rollback/prune
//! calls are not cross-process locked). Any number of readers may load
//! published versions concurrently because those files are never modified.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::hnsw::Hnsw;
use crate::persist::{self, LoadOptions, Manifest};

const CURRENT_FILE: &str = "CURRENT";
const VERSIONS_DIR: &str = "versions";
const META_FILE: &str = "snapshot.json";
const TMP_PREFIX: &str = ".tmp-";

/// Metadata about one published snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotInfo {
    /// Monotonically increasing version number.
    pub version: u64,
    /// Free-form label given at commit time (e.g. "nightly", "pre-migration").
    pub label: String,
    /// Version that was `CURRENT` when this one was committed.
    pub parent: Option<u64>,
    /// Commit wall-clock time, seconds since the Unix epoch.
    pub created_unix: u64,
    /// Live points in the image.
    pub live: usize,
    /// Total bytes of the image's data files.
    pub bytes: u64,
}

/// A directory of versioned snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotStore {
    root: PathBuf,
}

/// Directory name of a version.
pub fn version_name(version: u64) -> String {
    format!("v{version:06}")
}

fn parse_version(name: &str) -> Option<u64> {
    let digits = name.strip_prefix('v')?;
    if digits.len() < 6 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

impl SnapshotStore {
    /// Open (creating if needed) a store, discarding interrupted commits.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let versions = root.join(VERSIONS_DIR);
        fs::create_dir_all(&versions).map_err(|e| Error::io(&versions, e))?;
        for entry in fs::read_dir(&versions).map_err(|e| Error::io(&versions, e))? {
            let entry = entry.map_err(|e| Error::io(&versions, e))?;
            if entry.file_name().to_string_lossy().starts_with(TMP_PREFIX) {
                tracing::warn!(dir = %entry.path().display(), "removing interrupted snapshot commit");
                fs::remove_dir_all(entry.path()).map_err(|e| Error::io(entry.path(), e))?;
            }
        }
        Ok(Self { root })
    }

    /// Store root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Directory of a published version.
    pub fn version_dir(&self, version: u64) -> PathBuf {
        self.root.join(VERSIONS_DIR).join(version_name(version))
    }

    /// Published versions in ascending order.
    pub fn versions(&self) -> Result<Vec<u64>> {
        let dir = self.root.join(VERSIONS_DIR);
        let mut out: Vec<u64> = fs::read_dir(&dir)
            .map_err(|e| Error::io(&dir, e))?
            .filter_map(|e| e.ok())
            .filter_map(|e| parse_version(&e.file_name().to_string_lossy()))
            .collect();
        out.sort_unstable();
        Ok(out)
    }

    /// The version `CURRENT` points at, if any.
    pub fn current(&self) -> Result<Option<u64>> {
        let path = self.root.join(CURRENT_FILE);
        match fs::read_to_string(&path) {
            Ok(text) => parse_version(text.trim())
                .map(Some)
                .ok_or_else(|| Error::corrupt(&path, format!("bad CURRENT contents {text:?}"))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::io(&path, e)),
        }
    }

    /// Persist `index` as a new version and make it `CURRENT`.
    pub fn commit(&self, index: &Hnsw, label: &str) -> Result<SnapshotInfo> {
        let version = self.versions()?.last().copied().unwrap_or(0) + 1;
        let parent = self.current()?;
        let name = version_name(version);
        let versions_dir = self.root.join(VERSIONS_DIR);
        let tmp = versions_dir.join(format!("{TMP_PREFIX}{name}"));
        if tmp.exists() {
            fs::remove_dir_all(&tmp).map_err(|e| Error::io(&tmp, e))?;
        }
        let manifest = persist::save(index, &tmp)?;
        let info = SnapshotInfo {
            version,
            label: label.to_string(),
            parent,
            created_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            live: manifest.live,
            bytes: manifest.vectors.bytes + manifest.graph.bytes,
        };
        let meta_path = tmp.join(META_FILE);
        let json = serde_json::to_vec_pretty(&info).map_err(|source| Error::Json {
            path: meta_path.clone(),
            source,
        })?;
        persist::write_synced(&meta_path, &json)?;
        let dest = self.version_dir(version);
        fs::rename(&tmp, &dest).map_err(|e| Error::io(&dest, e))?;
        sync_dir(&versions_dir)?;
        self.set_current(version)?;
        tracing::info!(version, label, live = info.live, "snapshot committed");
        Ok(info)
    }

    /// Metadata of a published version.
    pub fn info(&self, version: u64) -> Result<SnapshotInfo> {
        let dir = self.version_dir(version);
        if !dir.is_dir() {
            return Err(Error::SnapshotNotFound(version_name(version)));
        }
        let path = dir.join(META_FILE);
        let text = fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
        serde_json::from_str(&text).map_err(|source| Error::Json { path, source })
    }

    /// Metadata of every published version, oldest first.
    pub fn list(&self) -> Result<Vec<SnapshotInfo>> {
        self.versions()?.into_iter().map(|v| self.info(v)).collect()
    }

    /// Load a specific version (vectors are memory-mapped).
    pub fn load(&self, version: u64, opts: LoadOptions) -> Result<Hnsw> {
        let dir = self.version_dir(version);
        if !dir.is_dir() {
            return Err(Error::SnapshotNotFound(version_name(version)));
        }
        persist::load(&dir, opts)
    }

    /// Load whatever `CURRENT` points at.
    pub fn load_current(&self, opts: LoadOptions) -> Result<(u64, Hnsw)> {
        let v = self
            .current()?
            .ok_or_else(|| Error::SnapshotNotFound("CURRENT".into()))?;
        Ok((v, self.load(v, opts)?))
    }

    /// Fully verify a version's checksums and graph invariants.
    pub fn verify(&self, version: u64) -> Result<Manifest> {
        let dir = self.version_dir(version);
        if !dir.is_dir() {
            return Err(Error::SnapshotNotFound(version_name(version)));
        }
        persist::verify(&dir)
    }

    /// Point `CURRENT` at an existing, verified version (rollback or roll-forward).
    pub fn rollback(&self, version: u64) -> Result<()> {
        self.verify(version)?;
        self.set_current(version)?;
        tracing::info!(version, "CURRENT moved");
        Ok(())
    }

    /// Delete all but the newest `keep` versions. `CURRENT` is always kept,
    /// even when it is older. Returns the removed versions.
    pub fn prune(&self, keep: usize) -> Result<Vec<u64>> {
        let versions = self.versions()?;
        let current = self.current()?;
        let cutoff = versions.len().saturating_sub(keep);
        let mut removed = Vec::new();
        for &v in &versions[..cutoff] {
            if Some(v) == current {
                continue;
            }
            let dir = self.version_dir(v);
            fs::remove_dir_all(&dir).map_err(|e| Error::io(&dir, e))?;
            removed.push(v);
        }
        Ok(removed)
    }

    fn set_current(&self, version: u64) -> Result<()> {
        let tmp = self.root.join(format!("{CURRENT_FILE}.tmp"));
        persist::write_synced(&tmp, format!("{}\n", version_name(version)).as_bytes())?;
        let dest = self.root.join(CURRENT_FILE);
        fs::rename(&tmp, &dest).map_err(|e| Error::io(&dest, e))?;
        sync_dir(&self.root)
    }
}

/// fsync a directory so a rename inside it is durable (no-op where unsupported).
fn sync_dir(dir: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        let f = fs::File::open(dir).map_err(|e| Error::io(dir, e))?;
        f.sync_all().map_err(|e| Error::io(dir, e))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::tests::grid_index;

    #[test]
    fn commit_list_load_rollback_prune() {
        let dir = tempfile::tempdir().unwrap();
        let store = SnapshotStore::open(dir.path()).unwrap();
        assert_eq!(store.current().unwrap(), None);
        assert!(store.load_current(LoadOptions::default()).is_err());

        let mut idx = grid_index(40, 4);
        let a = store.commit(&idx, "first").unwrap();
        idx.delete(3).unwrap();
        let b = store.commit(&idx, "second").unwrap();
        idx.insert(1000, &[9.5, 9.5]).unwrap();
        let c = store.commit(&idx, "third").unwrap();
        assert_eq!((a.version, b.version, c.version), (1, 2, 3));
        assert_eq!(b.parent, Some(1));
        assert_eq!(store.current().unwrap(), Some(3));
        assert_eq!(store.list().unwrap().len(), 3);

        let (v, loaded) = store.load_current(LoadOptions::default()).unwrap();
        assert_eq!(v, 3);
        assert!(loaded.contains(1000));

        store.rollback(1).unwrap();
        let (_, old) = store.load_current(LoadOptions::default()).unwrap();
        assert!(old.contains(3) && !old.contains(1000));

        // CURRENT (v1) survives pruning even though it is the oldest.
        let removed = store.prune(1).unwrap();
        assert_eq!(removed, vec![2]);
        assert_eq!(store.versions().unwrap(), vec![1, 3]);
        assert!(matches!(store.load(2, LoadOptions::default()), Err(Error::SnapshotNotFound(_))));
        assert!(store.verify(2).is_err());
        assert!(store.info(2).is_err());
        assert!(store.rollback(9).is_err());
        assert_eq!(store.root(), dir.path());
    }

    #[test]
    fn interrupted_commits_are_cleaned_and_bad_current_detected() {
        let dir = tempfile::tempdir().unwrap();
        let stale = dir.path().join(VERSIONS_DIR).join(".tmp-v000001");
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("vectors.bin"), b"partial").unwrap();
        let store = SnapshotStore::open(dir.path()).unwrap();
        assert!(!stale.exists());
        assert!(store.versions().unwrap().is_empty());
        fs::write(dir.path().join(CURRENT_FILE), "garbage").unwrap();
        assert!(matches!(store.current(), Err(Error::Corrupt { .. })));
    }

    #[test]
    fn version_names_parse() {
        assert_eq!(version_name(42), "v000042");
        assert_eq!(parse_version("v000042"), Some(42));
        assert_eq!(parse_version("v1234567"), Some(1_234_567));
        assert_eq!(parse_version("v42"), None);
        assert_eq!(parse_version(".tmp-v000001"), None);
        assert_eq!(parse_version("x000001"), None);
    }
}
