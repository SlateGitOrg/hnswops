//! On-disk format for one index image.
//!
//! An image is a directory with three files:
//!
//! | file            | contents                                              | loaded by |
//! |-----------------|-------------------------------------------------------|-----------|
//! | `manifest.json` | format version, config, counts, file sizes + crc32s   | read      |
//! | `vectors.bin`   | 64-byte header + raw little-endian `f32` rows         | **mmap**  |
//! | `graph.bin`     | node keys, tags, tombstone flags, adjacency lists     | read      |
//!
//! `graph.bin` ends with a crc32 of everything before it. Loading validates
//! magic numbers, sizes, checksums and every graph invariant (link targets in
//! range, degrees within limits, unique live keys) so a corrupt or truncated
//! file is reported as [`Error::Corrupt`] instead of causing a panic later.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{HnswConfig, MAX_LEVEL};
use crate::error::{Error, Result};
use crate::hnsw::{Hnsw, Node};
use crate::storage::{write_vectors_file, MappedVectors, VectorStore};

/// Current on-disk format version.
pub const FORMAT_VERSION: u32 = 1;
const GRAPH_MAGIC: &[u8; 8] = b"HNSWGRF1";

/// Name of the manifest inside an image directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Name of the vector file inside an image directory.
pub const VECTORS_FILE: &str = "vectors.bin";
/// Name of the graph file inside an image directory.
pub const GRAPH_FILE: &str = "graph.bin";

/// Size and checksum of one file in an image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileEntry {
    /// File name relative to the image directory.
    pub name: String,
    /// Exact size in bytes.
    pub bytes: u64,
    /// crc32 (IEEE) of the payload (vectors) or of the whole body (graph).
    pub crc32: u32,
}

/// Describes one saved index image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    /// On-disk format version.
    pub format_version: u32,
    /// Index configuration at save time.
    pub config: HnswConfig,
    /// Total slots, including tombstones.
    pub slots: usize,
    /// Live points.
    pub live: usize,
    /// Highest layer.
    pub max_level: usize,
    /// Internal id of the entry point.
    pub entry_point: Option<u32>,
    /// PRNG state so a reloaded index continues the same level sequence.
    pub rng_state: u64,
    /// `vectors.bin` description.
    pub vectors: FileEntry,
    /// `graph.bin` description.
    pub graph: FileEntry,
}

/// Options for [`load`].
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Verify the crc32 of `vectors.bin`. This reads every page of the file,
    /// trading the O(1) mmap start-up for end-to-end integrity checking.
    pub verify_vectors: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            verify_vectors: true,
        }
    }
}

/// Write `index` into `dir` (created if missing). Files are fsynced; atomic
/// publication is the caller's job (see [`crate::snapshot`]).
pub fn save(index: &Hnsw, dir: &Path) -> Result<Manifest> {
    fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    let vpath = dir.join(VECTORS_FILE);
    let n = index.nodes.len();
    let vcrc = write_vectors_file(&vpath, index.config.dim, n, index.vectors.iter())?;
    let vbytes = fs::metadata(&vpath)
        .map_err(|e| Error::io(&vpath, e))?
        .len();

    let gpath = dir.join(GRAPH_FILE);
    let graph = encode_graph(&index.nodes);
    let gcrc = crc32fast::hash(&graph[..graph.len() - 4]);
    write_synced(&gpath, &graph)?;

    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        config: index.config.clone(),
        slots: n,
        live: index.live,
        max_level: index.max_level,
        entry_point: index.entry_point,
        rng_state: index.rng.state(),
        vectors: FileEntry {
            name: VECTORS_FILE.into(),
            bytes: vbytes,
            crc32: vcrc,
        },
        graph: FileEntry {
            name: GRAPH_FILE.into(),
            bytes: graph.len() as u64,
            crc32: gcrc,
        },
    };
    write_manifest(dir, &manifest)?;
    Ok(manifest)
}

/// Read only the manifest of an image.
pub fn read_manifest(dir: &Path) -> Result<Manifest> {
    let path = dir.join(MANIFEST_FILE);
    let text = fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
    let m: Manifest = serde_json::from_str(&text).map_err(|source| Error::Json {
        path: path.clone(),
        source,
    })?;
    if m.format_version != FORMAT_VERSION {
        return Err(Error::corrupt(
            &path,
            format!("unsupported format version {}", m.format_version),
        ));
    }
    Ok(m)
}

/// Load an image: the graph is read and validated, the vectors are mmap'd.
pub fn load(dir: &Path, opts: LoadOptions) -> Result<Hnsw> {
    let manifest = read_manifest(dir)?;
    let gpath = dir.join(&manifest.graph.name);
    let graph = fs::read(&gpath).map_err(|e| Error::io(&gpath, e))?;
    if graph.len() as u64 != manifest.graph.bytes || graph.len() < 4 {
        return Err(Error::corrupt(&gpath, "size differs from manifest"));
    }
    let (body, trailer) = graph.split_at(graph.len() - 4);
    let stored = u32::from_le_bytes(trailer.try_into().expect("4 bytes"));
    let actual = crc32fast::hash(body);
    if stored != actual || actual != manifest.graph.crc32 {
        return Err(Error::corrupt(
            &gpath,
            format!("crc32 mismatch (computed {actual:08x}, stored {stored:08x})"),
        ));
    }
    let nodes = decode_graph(body, manifest.slots).map_err(|r| Error::corrupt(&gpath, r))?;

    let vpath = dir.join(&manifest.vectors.name);
    let expect = opts.verify_vectors.then_some(manifest.vectors.crc32);
    let mapped = MappedVectors::open(&vpath, manifest.config.dim, manifest.slots, expect)?;
    let store = VectorStore::from_mapped(manifest.config.dim, mapped);
    let index = Hnsw::from_parts(
        manifest.config.clone(),
        store,
        nodes,
        manifest.entry_point,
        manifest.rng_state,
    )
    .map_err(|r| Error::corrupt(&gpath, r))?;
    if index.live != manifest.live {
        return Err(Error::corrupt(
            dir.join(MANIFEST_FILE),
            "live count disagrees with graph",
        ));
    }
    tracing::debug!(
        dir = %dir.display(),
        live = index.live,
        mapped = index.mapped_vectors(),
        "index image loaded"
    );
    Ok(index)
}

/// Recompute checksums of an image without constructing an index.
pub fn verify(dir: &Path) -> Result<Manifest> {
    let manifest = read_manifest(dir)?;
    load(
        dir,
        LoadOptions {
            verify_vectors: true,
        },
    )?;
    Ok(manifest)
}

pub(crate) fn write_manifest(dir: &Path, manifest: &Manifest) -> Result<()> {
    let path = dir.join(MANIFEST_FILE);
    let json = serde_json::to_vec_pretty(manifest).map_err(|source| Error::Json {
        path: path.clone(),
        source,
    })?;
    write_synced(&path, &json)
}

pub(crate) fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut f = fs::File::create(path).map_err(|e| Error::io(path, e))?;
    f.write_all(bytes).map_err(|e| Error::io(path, e))?;
    f.sync_all().map_err(|e| Error::io(path, e))
}

fn encode_graph(nodes: &[Node]) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + nodes.len() * 64);
    out.extend_from_slice(GRAPH_MAGIC);
    out.extend_from_slice(&(nodes.len() as u64).to_le_bytes());
    for n in nodes {
        out.extend_from_slice(&n.key.to_le_bytes());
        out.extend_from_slice(&n.tags.to_le_bytes());
        out.push(u8::from(n.deleted));
        out.push(n.level() as u8);
        for list in &n.links {
            out.extend_from_slice(&(list.len() as u16).to_le_bytes());
            for &x in list {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    let crc = crc32fast::hash(&out);
    out.extend_from_slice(&crc.to_le_bytes());
    out
}

/// Bounds-checked little-endian reader.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> std::result::Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| format!("truncated at byte {}", self.pos))?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> std::result::Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> std::result::Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().expect("2")))
    }
    fn u32(&mut self) -> std::result::Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4")))
    }
    fn u64(&mut self) -> std::result::Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
}

fn decode_graph(body: &[u8], expected: usize) -> std::result::Result<Vec<Node>, String> {
    let mut r = Reader { buf: body, pos: 0 };
    if r.take(8)? != GRAPH_MAGIC {
        return Err("bad magic".into());
    }
    let n = r.u64()? as usize;
    if n != expected {
        return Err(format!("graph has {n} nodes, manifest says {expected}"));
    }
    let mut nodes = Vec::with_capacity(n.min(body.len() / 18));
    for _ in 0..n {
        let key = r.u64()?;
        let tags = r.u64()?;
        let deleted = match r.u8()? {
            0 => false,
            1 => true,
            f => return Err(format!("bad tombstone flag {f}")),
        };
        let level = r.u8()? as usize;
        if level > MAX_LEVEL {
            return Err(format!("level {level} exceeds maximum {MAX_LEVEL}"));
        }
        let mut links = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            let deg = r.u16()? as usize;
            let mut list = Vec::with_capacity(deg);
            for _ in 0..deg {
                list.push(r.u32()?);
            }
            links.push(list);
        }
        nodes.push(Node {
            key,
            tags,
            deleted,
            links,
        });
    }
    if r.pos != body.len() {
        return Err(format!("{} trailing bytes", body.len() - r.pos));
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hnsw::tests::grid_index;

    #[test]
    fn save_load_round_trip_is_exact() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = grid_index(120, 6);
        idx.insert_with_tags(7, &[50.0, 50.0], 3).unwrap();
        idx.delete(11).unwrap();
        let m = save(&idx, dir.path()).unwrap();
        assert_eq!(m.live, idx.len());
        let loaded = load(dir.path(), LoadOptions::default()).unwrap();
        assert_eq!(loaded.nodes, idx.nodes);
        assert_eq!(loaded.mapped_vectors(), idx.nodes.len());
        assert_eq!(loaded.tags(7), Some(3));
        for i in 0..30 {
            let q = [i as f32 * 0.3, i as f32 * 0.2];
            assert_eq!(loaded.search(&q, 5).unwrap(), idx.search(&q, 5).unwrap());
        }
        assert_eq!(verify(dir.path()).unwrap(), m);
    }

    #[test]
    fn corrupt_graph_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        save(&grid_index(30, 4), dir.path()).unwrap();
        let g = dir.path().join(GRAPH_FILE);
        let mut bytes = fs::read(&g).unwrap();
        bytes[20] ^= 0xFF;
        fs::write(&g, &bytes).unwrap();
        let err = load(dir.path(), LoadOptions::default()).unwrap_err();
        assert!(matches!(err, Error::Corrupt { .. }), "{err}");
    }

    #[test]
    fn decoder_rejects_malformed_bodies() {
        let nodes = grid_index(5, 4).nodes;
        let enc = encode_graph(&nodes);
        let body = &enc[..enc.len() - 4];
        assert_eq!(decode_graph(body, 5).unwrap(), nodes);
        assert!(decode_graph(body, 6).is_err());
        assert!(decode_graph(&body[..body.len() - 1], 5).is_err());
        let mut extra = body.to_vec();
        extra.push(0);
        assert!(decode_graph(&extra, 5).is_err());
        let mut flag = body.to_vec();
        flag[32] = 7; // first node's tombstone flag
        assert!(decode_graph(&flag, 5).is_err());
        let mut lvl = body.to_vec();
        lvl[33] = 200;
        assert!(decode_graph(&lvl, 5).is_err());
        assert!(decode_graph(b"NOTMAGIC", 0).is_err());
    }

    #[test]
    fn manifest_version_is_checked() {
        let dir = tempfile::tempdir().unwrap();
        let mut m = save(&grid_index(5, 4), dir.path()).unwrap();
        m.format_version = 99;
        write_manifest(dir.path(), &m).unwrap();
        assert!(read_manifest(dir.path()).is_err());
        fs::write(dir.path().join(MANIFEST_FILE), "{").unwrap();
        assert!(matches!(read_manifest(dir.path()), Err(Error::Json { .. })));
        assert!(matches!(
            read_manifest(&dir.path().join("missing")),
            Err(Error::Io { .. })
        ));
    }
}
