//! Vector storage: an optional read-only memory-mapped base segment plus an
//! in-memory append-only tail.
//!
//! A freshly built index keeps every vector on the heap. An index loaded from
//! a snapshot maps `vectors.bin` instead of reading it, so start-up cost is
//! O(1) in the number of vectors and the OS page cache is shared between
//! processes serving the same snapshot. Vectors inserted after load go to the
//! heap tail; the mapped file is never written to.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};

/// Magic bytes at the start of every vectors file.
pub const VECTORS_MAGIC: &[u8; 8] = b"HNSWVEC1";
/// Fixed header size; the payload starts at this (4-byte aligned) offset.
pub const VECTORS_HEADER_LEN: usize = 64;

/// A read-only mapping of a `vectors.bin` file.
#[derive(Debug)]
pub struct MappedVectors {
    mmap: Mmap,
    path: PathBuf,
    count: usize,
}

impl MappedVectors {
    /// Map `path`, validating the header against the expected shape.
    /// With `expect_crc` set, the payload checksum is verified (touching every page).
    pub fn open(path: &Path, dim: usize, count: usize, expect_crc: Option<u32>) -> Result<Self> {
        if cfg!(target_endian = "big") {
            return Err(Error::corrupt(
                path,
                "vector files are little-endian; big-endian hosts are unsupported",
            ));
        }
        let file = File::open(path).map_err(|e| Error::io(path, e))?;
        // SAFETY: snapshot files are immutable once committed (they are written
        // to a temporary directory and atomically renamed), and this crate never
        // opens them for writing afterwards. Mutation by an external process is
        // outside the supported contract and documented in docs/ARCHITECTURE.md.
        #[allow(unsafe_code)]
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| Error::io(path, e))?;
        let want_len = VECTORS_HEADER_LEN + count * dim * 4;
        if mmap.len() != want_len {
            return Err(Error::corrupt(
                path,
                format!("expected {want_len} bytes, found {}", mmap.len()),
            ));
        }
        if &mmap[..8] != VECTORS_MAGIC {
            return Err(Error::corrupt(path, "bad magic"));
        }
        let file_dim = u32::from_le_bytes(mmap[12..16].try_into().expect("4 bytes")) as usize;
        let file_count = u64::from_le_bytes(mmap[16..24].try_into().expect("8 bytes")) as usize;
        if file_dim != dim || file_count != count {
            return Err(Error::corrupt(
                path,
                format!("header says {file_count}x{file_dim}, manifest says {count}x{dim}"),
            ));
        }
        if let Some(want) = expect_crc {
            let got = crc32fast::hash(&mmap[VECTORS_HEADER_LEN..]);
            if got != want {
                return Err(Error::corrupt(
                    path,
                    format!("payload crc32 {got:08x} != manifest {want:08x}"),
                ));
            }
        }
        let mapped = Self {
            mmap,
            path: path.to_path_buf(),
            count,
        };
        // Validate alignment once so `slice` can never fail later.
        bytemuck::try_cast_slice::<u8, f32>(mapped.payload())
            .map_err(|e| Error::corrupt(path, format!("payload not castable to f32: {e:?}")))?;
        Ok(mapped)
    }

    fn payload(&self) -> &[u8] {
        &self.mmap[VECTORS_HEADER_LEN..]
    }

    /// All mapped components as one flat slice.
    pub fn as_f32(&self) -> &[f32] {
        bytemuck::cast_slice(self.payload())
    }

    /// Number of vectors in the mapping.
    pub fn count(&self) -> usize {
        self.count
    }

    /// File backing this mapping.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Write `count` vectors of `dim` components to `path` in the mmap-able format.
/// Returns the crc32 of the payload.
pub fn write_vectors_file<'a, I>(path: &Path, dim: usize, count: usize, vectors: I) -> Result<u32>
where
    I: IntoIterator<Item = &'a [f32]>,
{
    let file = File::create(path).map_err(|e| Error::io(path, e))?;
    let mut out = BufWriter::with_capacity(1 << 20, file);
    let mut header = [0u8; VECTORS_HEADER_LEN];
    header[..8].copy_from_slice(VECTORS_MAGIC);
    header[8..12].copy_from_slice(&1u32.to_le_bytes());
    header[12..16].copy_from_slice(&(dim as u32).to_le_bytes());
    header[16..24].copy_from_slice(&(count as u64).to_le_bytes());
    out.write_all(&header).map_err(|e| Error::io(path, e))?;
    let mut hasher = crc32fast::Hasher::new();
    let mut written = 0usize;
    for v in vectors {
        debug_assert_eq!(v.len(), dim);
        for x in v {
            let bytes = x.to_le_bytes();
            hasher.update(&bytes);
            out.write_all(&bytes).map_err(|e| Error::io(path, e))?;
        }
        written += 1;
    }
    if written != count {
        return Err(Error::corrupt(
            path,
            format!("writer produced {written} vectors, header promised {count}"),
        ));
    }
    let file = out
        .into_inner()
        .map_err(|e| Error::io(path, e.into_error()))?;
    file.sync_all().map_err(|e| Error::io(path, e))?;
    Ok(hasher.finalize())
}

/// Dense vector storage addressed by internal node id.
#[derive(Debug)]
pub struct VectorStore {
    dim: usize,
    base: Option<MappedVectors>,
    base_count: usize,
    tail: Vec<f32>,
}

impl VectorStore {
    /// Empty heap-backed store.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            base: None,
            base_count: 0,
            tail: Vec::new(),
        }
    }

    /// Store whose first `mapped.count()` vectors come from a mapping.
    pub fn from_mapped(dim: usize, mapped: MappedVectors) -> Self {
        let base_count = mapped.count();
        Self {
            dim,
            base: Some(mapped),
            base_count,
            tail: Vec::new(),
        }
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Total number of stored vectors.
    pub fn len(&self) -> usize {
        self.base_count + self.tail.len() / self.dim
    }

    /// True when no vectors are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many vectors are served from the memory map.
    pub fn mapped_count(&self) -> usize {
        self.base_count
    }

    /// Vector of internal id `id`. Panics on out-of-range ids (an internal bug).
    #[inline]
    pub fn get(&self, id: u32) -> &[f32] {
        let i = id as usize;
        if i < self.base_count {
            let base = self.base.as_ref().expect("base present when base_count > 0");
            &base.as_f32()[i * self.dim..(i + 1) * self.dim]
        } else {
            let j = i - self.base_count;
            &self.tail[j * self.dim..(j + 1) * self.dim]
        }
    }

    /// Append a vector (already validated to have `dim` components).
    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.tail.extend_from_slice(v);
    }

    /// New heap-backed store containing only `keep` (in that order).
    pub fn compacted(&self, keep: &[u32]) -> Self {
        let mut tail = Vec::with_capacity(keep.len() * self.dim);
        for &id in keep {
            tail.extend_from_slice(self.get(id));
        }
        Self {
            dim: self.dim,
            base: None,
            base_count: 0,
            tail,
        }
    }

    /// Iterate vectors in id order.
    pub fn iter(&self) -> impl Iterator<Item = &[f32]> + '_ {
        (0..self.len() as u32).map(move |i| self.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapped_plus_tail_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.bin");
        let data: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, -(i as f32), 0.5]).collect();
        let crc = write_vectors_file(&path, 3, 5, data.iter().map(|v| v.as_slice())).unwrap();
        let mapped = MappedVectors::open(&path, 3, 5, Some(crc)).unwrap();
        assert_eq!(mapped.path(), path.as_path());
        let mut store = VectorStore::from_mapped(3, mapped);
        store.push(&[9.0, 9.0, 9.0]);
        assert_eq!(store.len(), 6);
        assert_eq!(store.mapped_count(), 5);
        assert_eq!(store.get(2), &[2.0, -2.0, 0.5]);
        assert_eq!(store.get(5), &[9.0, 9.0, 9.0]);
        let c = store.compacted(&[5, 0]);
        assert_eq!(c.mapped_count(), 0);
        assert_eq!(c.iter().collect::<Vec<_>>(), vec![&[9.0f32, 9.0, 9.0][..], &[0.0, -0.0, 0.5][..]]);
    }

    #[test]
    fn rejects_mismatched_or_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.bin");
        let data = [vec![1.0f32, 2.0]];
        let crc = write_vectors_file(&path, 2, 1, data.iter().map(|v| v.as_slice())).unwrap();
        assert!(MappedVectors::open(&path, 3, 1, None).is_err());
        assert!(MappedVectors::open(&path, 2, 1, Some(crc ^ 1)).is_err());
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] = b'X';
        std::fs::write(&path, &bytes).unwrap();
        assert!(MappedVectors::open(&path, 2, 1, None).is_err());
        assert!(write_vectors_file(&path, 2, 2, data.iter().map(|v| v.as_slice())).is_err());
    }
}
