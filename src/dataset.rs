//! Benchmark data: `.fvecs` / `.ivecs` I/O, pinned checksums, synthetic data.
//!
//! The TEXMEX formats (used by SIFT/GIST and ann-benchmarks' originals) store
//! each row as a little-endian `i32` dimension followed by that many `f32`
//! (`.fvecs`) or `i32` (`.ivecs`) values.

use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::rng::SplitMix64;

/// A dense row-major set of equal-length vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct VecSet {
    dim: usize,
    data: Vec<f32>,
}

impl VecSet {
    /// Wrap a flat buffer; `data.len()` must be a multiple of `dim`.
    pub fn new(dim: usize, data: Vec<f32>) -> Result<Self> {
        if dim == 0 || data.len() % dim != 0 {
            return Err(Error::InvalidConfig(format!(
                "buffer of {} floats is not a whole number of {dim}-d rows",
                data.len()
            )));
        }
        Ok(Self { dim, data })
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row `i`.
    pub fn get(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Iterate rows.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> + '_ {
        self.data.chunks_exact(self.dim)
    }

    /// First `n` rows (or all, if fewer).
    pub fn head(&self, n: usize) -> VecSet {
        let n = n.min(self.len());
        VecSet {
            dim: self.dim,
            data: self.data[..n * self.dim].to_vec(),
        }
    }
}

fn read_rows(path: &Path) -> Result<(usize, Vec<[u8; 4]>, usize)> {
    let f = fs::File::open(path).map_err(|e| Error::io(path, e))?;
    let mut r = BufReader::new(f);
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes).map_err(|e| Error::io(path, e))?;
    if bytes.len() < 4 {
        return Err(Error::corrupt(path, "file too short"));
    }
    let dim = i32::from_le_bytes(bytes[..4].try_into().expect("4")) ;
    if dim <= 0 {
        return Err(Error::corrupt(path, format!("bad dimension {dim}")));
    }
    let dim = dim as usize;
    let row = 4 + 4 * dim;
    if bytes.len() % row != 0 {
        return Err(Error::corrupt(
            path,
            format!("{} bytes is not a multiple of the {row}-byte row", bytes.len()),
        ));
    }
    let rows = bytes.len() / row;
    let mut words = Vec::with_capacity(rows * dim);
    for chunk in bytes.chunks_exact(row) {
        let d = i32::from_le_bytes(chunk[..4].try_into().expect("4"));
        if d as usize != dim {
            return Err(Error::corrupt(path, "rows have differing dimensions"));
        }
        for w in chunk[4..].chunks_exact(4) {
            words.push(w.try_into().expect("4"));
        }
    }
    Ok((dim, words, rows))
}

/// Read a `.fvecs` file.
pub fn read_fvecs(path: &Path) -> Result<VecSet> {
    let (dim, words, _) = read_rows(path)?;
    VecSet::new(dim, words.into_iter().map(f32::from_le_bytes).collect())
}

/// Read a `.ivecs` file as one `Vec<u32>` per row.
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<u32>>> {
    let (dim, words, _) = read_rows(path)?;
    let flat: Vec<i32> = words.into_iter().map(i32::from_le_bytes).collect();
    if flat.iter().any(|&x| x < 0) {
        return Err(Error::corrupt(path, "negative id in ivecs"));
    }
    Ok(flat
        .chunks_exact(dim)
        .map(|c| c.iter().map(|&x| x as u32).collect())
        .collect())
}

/// Write a `.fvecs` file.
pub fn write_fvecs(path: &Path, set: &VecSet) -> Result<()> {
    let f = fs::File::create(path).map_err(|e| Error::io(path, e))?;
    let mut w = BufWriter::new(f);
    for row in set.iter() {
        w.write_all(&(set.dim as i32).to_le_bytes())
            .map_err(|e| Error::io(path, e))?;
        for x in row {
            w.write_all(&x.to_le_bytes()).map_err(|e| Error::io(path, e))?;
        }
    }
    w.flush().map_err(|e| Error::io(path, e))
}

/// Write a `.ivecs` file (all rows must share a length).
pub fn write_ivecs(path: &Path, rows: &[Vec<u32>]) -> Result<()> {
    let f = fs::File::create(path).map_err(|e| Error::io(path, e))?;
    let mut w = BufWriter::new(f);
    for row in rows {
        w.write_all(&(row.len() as i32).to_le_bytes())
            .map_err(|e| Error::io(path, e))?;
        for &x in row {
            w.write_all(&(x as i32).to_le_bytes())
                .map_err(|e| Error::io(path, e))?;
        }
    }
    w.flush().map_err(|e| Error::io(path, e))
}

/// Hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|e| Error::io(path, e))?;
    let digest = Sha256::digest(&bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// One pinned file of a benchmark dataset.
#[derive(Debug, Clone, Copy)]
pub struct PinnedFile {
    /// File name inside the dataset directory.
    pub name: &'static str,
    /// Expected SHA-256 (hex).
    pub sha256: &'static str,
}

/// A benchmark dataset pinned by checksum.
#[derive(Debug, Clone, Copy)]
pub struct DatasetSpec {
    /// Short name used on the command line.
    pub name: &'static str,
    /// Base vectors (`.fvecs`).
    pub base: PinnedFile,
    /// Query vectors (`.fvecs`).
    pub query: PinnedFile,
    /// Ground-truth neighbor ids (`.ivecs`).
    pub groundtruth: PinnedFile,
}

/// ANN_SIFT10K ("siftsmall") from the INRIA TEXMEX corpus: 10,000 base and
/// 100 query 128-d SIFT descriptors with 100-NN ground truth. See `DATA.md`.
pub const SIFT_SMALL: DatasetSpec = DatasetSpec {
    name: "siftsmall",
    base: PinnedFile {
        name: "siftsmall_base.fvecs",
        sha256: "1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0",
    },
    query: PinnedFile {
        name: "siftsmall_query.fvecs",
        sha256: "095af38bda2741be721447ba97e9286fa918ece946be92a9ae133276eddc1a10",
    },
    groundtruth: PinnedFile {
        name: "siftsmall_groundtruth.ivecs",
        sha256: "b0717bf276111a166f14dcc2653fd04465b2afd6d34c72be641b622cf99f95ae",
    },
};

/// The synthetic fixture bundled in `fixtures/` (generated by
/// `hnswops gen-fixture`, see `DATA.md`).
pub const FIXTURE: DatasetSpec = DatasetSpec {
    name: "fixture",
    base: PinnedFile {
        name: "base.fvecs",
        sha256: "",
    },
    query: PinnedFile {
        name: "query.fvecs",
        sha256: "",
    },
    groundtruth: PinnedFile {
        name: "groundtruth.ivecs",
        sha256: "",
    },
};

/// A loaded benchmark dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Base vectors.
    pub base: VecSet,
    /// Query vectors.
    pub queries: VecSet,
    /// Ground-truth neighbor ids (indices into `base`), nearest first.
    pub groundtruth: Vec<Vec<u32>>,
}

/// Verify (where a checksum is pinned) and load a dataset from `dir`.
pub fn load_dataset(dir: &Path, spec: &DatasetSpec) -> Result<Dataset> {
    for f in [spec.base, spec.query, spec.groundtruth] {
        let path = dir.join(f.name);
        if f.sha256.is_empty() {
            continue;
        }
        let got = sha256_file(&path)?;
        if got != f.sha256 {
            return Err(Error::corrupt(
                &path,
                format!("sha256 {got} does not match pinned {}", f.sha256),
            ));
        }
    }
    let base = read_fvecs(&dir.join(spec.base.name))?;
    let queries = read_fvecs(&dir.join(spec.query.name))?;
    let groundtruth = read_ivecs(&dir.join(spec.groundtruth.name))?;
    if base.dim() != queries.dim() || groundtruth.len() != queries.len() {
        return Err(Error::corrupt(dir, "base/query/groundtruth shapes disagree"));
    }
    Ok(Dataset {
        base,
        queries,
        groundtruth,
    })
}

/// Gaussian-mixture data: `clusters` centers drawn from N(0, 10²) per axis,
/// points drawn around a uniformly chosen center with standard deviation `spread`.
/// Clustered data is harder for graph indexes than uniform noise (it needs
/// the long-range edges the selection heuristic preserves).
pub fn gaussian_mixture(n: usize, dim: usize, clusters: usize, spread: f32, seed: u64) -> VecSet {
    let mut rng = SplitMix64::new(seed);
    let centers: Vec<Vec<f32>> = (0..clusters.max(1))
        .map(|_| (0..dim).map(|_| rng.next_gaussian() as f32 * 10.0).collect())
        .collect();
    let mut data = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let c = &centers[rng.below(centers.len() as u64) as usize];
        for &x in c {
            data.push(x + rng.next_gaussian() as f32 * spread);
        }
    }
    VecSet { dim, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fvecs_ivecs_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let set = gaussian_mixture(20, 5, 3, 1.0, 9);
        let fp = dir.path().join("a.fvecs");
        write_fvecs(&fp, &set).unwrap();
        assert_eq!(read_fvecs(&fp).unwrap(), set);
        let ip = dir.path().join("a.ivecs");
        let rows = vec![vec![1, 2, 3], vec![4, 5, 6]];
        write_ivecs(&ip, &rows).unwrap();
        assert_eq!(read_ivecs(&ip).unwrap(), rows);
        assert_eq!(sha256_file(&ip).unwrap().len(), 64);
        assert_eq!(set.head(3).len(), 3);
        assert_eq!(set.head(99).len(), 20);
        assert!(!set.is_empty());
    }

    #[test]
    fn malformed_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("bad.fvecs");
        fs::write(&p, [1u8, 0]).unwrap();
        assert!(read_fvecs(&p).is_err());
        fs::write(&p, (-3i32).to_le_bytes()).unwrap();
        assert!(read_fvecs(&p).is_err());
        let mut b = 2i32.to_le_bytes().to_vec();
        b.extend_from_slice(&[0; 8]);
        b.extend_from_slice(&3i32.to_le_bytes());
        b.extend_from_slice(&[0; 8]);
        fs::write(&p, &b).unwrap();
        assert!(read_fvecs(&p).is_err(), "differing dims / ragged");
        let mut neg = 1i32.to_le_bytes().to_vec();
        neg.extend_from_slice(&(-1i32).to_le_bytes());
        fs::write(&p, &neg).unwrap();
        assert!(read_ivecs(&p).is_err());
        assert!(read_fvecs(&dir.path().join("missing")).is_err());
        assert!(VecSet::new(3, vec![0.0; 4]).is_err());
    }

    #[test]
    fn checksum_mismatch_is_detected() {
        let dir = tempfile::tempdir().unwrap();
        let set = gaussian_mixture(4, 2, 1, 1.0, 1);
        write_fvecs(&dir.path().join("siftsmall_base.fvecs"), &set).unwrap();
        let err = load_dataset(dir.path(), &SIFT_SMALL).unwrap_err();
        assert!(matches!(err, Error::Corrupt { .. }));
    }

    #[test]
    fn unpinned_dataset_loads_and_validates_shapes() {
        let dir = tempfile::tempdir().unwrap();
        write_fvecs(&dir.path().join("base.fvecs"), &gaussian_mixture(10, 3, 2, 1.0, 1)).unwrap();
        write_fvecs(&dir.path().join("query.fvecs"), &gaussian_mixture(2, 3, 2, 1.0, 2)).unwrap();
        write_ivecs(&dir.path().join("groundtruth.ivecs"), &[vec![0], vec![1]]).unwrap();
        let ds = load_dataset(dir.path(), &FIXTURE).unwrap();
        assert_eq!(ds.base.len(), 10);
        write_ivecs(&dir.path().join("groundtruth.ivecs"), &[vec![0]]).unwrap();
        assert!(load_dataset(dir.path(), &FIXTURE).is_err());
    }

    #[test]
    fn mixture_is_deterministic() {
        assert_eq!(gaussian_mixture(5, 4, 2, 0.5, 3), gaussian_mixture(5, 4, 2, 0.5, 3));
        assert_ne!(gaussian_mixture(5, 4, 2, 0.5, 3), gaussian_mixture(5, 4, 2, 0.5, 4));
    }
}
