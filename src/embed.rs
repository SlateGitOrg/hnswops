//! Text embedders and the document source a re-embedding migration reads from.
//!
//! The [`Embedder`] trait is the seam where a real model (an HTTP embedding
//! service, an ONNX model) plugs in. The crate ships one implementation,
//! [`HashingEmbedder`]: deterministic signed feature hashing of word n-grams.
//! It is not a semantic model, but texts that share words land close together,
//! it needs no network or weights, and two instances with different
//! `dim`/`ngram`/`seed` behave like two incompatible model versions — exactly
//! what a migration test needs.

use std::collections::BTreeMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Turns text into a fixed-length vector.
pub trait Embedder: Send + Sync {
    /// Model identifier recorded in reports (e.g. `"hash-v1"`).
    fn name(&self) -> &str;
    /// Output dimensionality.
    fn dim(&self) -> usize;
    /// Embed one text.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Signed feature-hashing embedder over lowercase word n-grams.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashingEmbedder {
    name: String,
    dim: usize,
    ngram: usize,
    seed: u64,
}

impl HashingEmbedder {
    /// `ngram = 1` hashes words; `ngram = 2` adds adjacent word pairs, etc.
    pub fn new(name: impl Into<String>, dim: usize, ngram: usize, seed: u64) -> Result<Self> {
        if dim == 0 || ngram == 0 {
            return Err(Error::InvalidConfig(
                "embedder dim and ngram must be >= 1".into(),
            ));
        }
        Ok(Self {
            name: name.into(),
            dim,
            ngram,
            seed,
        })
    }
}

/// FNV-1a over the seed followed by the bytes, then a SplitMix finalizer so
/// low bits are well mixed for the modulo.
fn hash_feature(seed: u64, parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for (i, p) in parts.iter().enumerate() {
        if i > 0 {
            h ^= u64::from(b' ');
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        for b in p.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

impl Embedder for HashingEmbedder {
    fn name(&self) -> &str {
        &self.name
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .collect();
        if words.is_empty() {
            return Err(Error::InvalidVector(format!(
                "text {text:?} has no tokens to embed"
            )));
        }
        let mut v = vec![0.0f32; self.dim];
        for n in 1..=self.ngram.min(words.len()) {
            // Longer n-grams are rarer and more specific: weight them up gently.
            let weight = 1.0 + 0.5 * (n - 1) as f32;
            for gram in words.windows(n) {
                let h = hash_feature(self.seed, gram);
                let slot = (h % self.dim as u64) as usize;
                let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
                v[slot] += sign * weight;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            // Every feature cancelled out; nudge deterministically.
            v[0] = 1.0;
        } else {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(v)
    }
}

/// A unit of source data: the system of record a migration re-embeds from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// Primary key, shared with the index.
    pub key: u64,
    /// Text that gets embedded.
    pub text: String,
    /// Filterable tag bitmask.
    pub tags: u64,
}

/// Read access to the system of record, paginated by ascending key so a
/// backfill can resume from a cursor.
pub trait Corpus: Send + Sync {
    /// Up to `limit` documents with key strictly greater than `after`
    /// (or from the start when `None`), in ascending key order.
    fn scan(&self, after: Option<u64>, limit: usize) -> Vec<Document>;
}

/// Thread-safe in-memory corpus used by tests and the demo.
#[derive(Debug, Default)]
pub struct InMemoryCorpus {
    docs: RwLock<BTreeMap<u64, Document>>,
}

impl InMemoryCorpus {
    /// Corpus pre-filled with `docs`.
    pub fn from_docs(docs: impl IntoIterator<Item = Document>) -> Self {
        Self {
            docs: RwLock::new(docs.into_iter().map(|d| (d.key, d)).collect()),
        }
    }

    /// Insert or replace a document.
    pub fn upsert(&self, doc: Document) {
        self.docs
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(doc.key, doc);
    }

    /// Remove a document; returns whether it existed.
    pub fn delete(&self, key: u64) -> bool {
        self.docs
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .remove(&key)
            .is_some()
    }

    /// Fetch one document.
    pub fn get(&self, key: u64) -> Option<Document> {
        self.docs
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&key)
            .cloned()
    }

    /// Number of documents.
    pub fn len(&self) -> usize {
        self.docs.read().unwrap_or_else(|p| p.into_inner()).len()
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Corpus for InMemoryCorpus {
    fn scan(&self, after: Option<u64>, limit: usize) -> Vec<Document> {
        let docs = self.docs.read().unwrap_or_else(|p| p.into_inner());
        let range = match after {
            Some(a) => docs.range(a.saturating_add(1)..),
            None => docs.range(..),
        };
        // `a.saturating_add(1)` would re-yield u64::MAX; exclude it explicitly.
        range
            .filter(|(k, _)| after != Some(**k))
            .take(limit)
            .map(|(_, d)| d.clone())
            .collect()
    }
}

/// Deterministic synthetic corpus: each document mixes words from one of
/// `topics` topic vocabularies with a few shared filler words, so documents
/// on the same topic are near each other under any n-gram embedder.
pub fn synthetic_corpus(n: usize, topics: usize, seed: u64) -> Vec<Document> {
    use crate::rng::SplitMix64;
    const FILLER: [&str; 12] = [
        "the", "a", "report", "update", "notes", "about", "and", "with", "new", "on", "for",
        "summary",
    ];
    let topics = topics.max(1);
    let mut rng = SplitMix64::new(seed);
    (0..n as u64)
        .map(|key| {
            let topic = rng.below(topics as u64) as usize;
            let len = 6 + rng.below(6) as usize;
            let words: Vec<String> = (0..len)
                .map(|_| {
                    if rng.below(4) == 0 {
                        FILLER[rng.below(FILLER.len() as u64) as usize].to_string()
                    } else {
                        format!("t{topic}w{}", rng.below(16))
                    }
                })
                .collect();
            Document {
                key,
                text: words.join(" "),
                tags: 1 << (topic % 64),
            }
        })
        .collect()
}

/// A query in the style of [`synthetic_corpus`] for `topic`.
pub fn synthetic_query(topic: usize, seed: u64) -> String {
    use crate::rng::SplitMix64;
    let mut rng = SplitMix64::new(seed);
    (0..5)
        .map(|_| format!("t{topic}w{}", rng.below(16)))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::dot;

    #[test]
    fn hashing_embedder_is_deterministic_normalized_and_similarity_aware() {
        let e = HashingEmbedder::new("hash-v1", 64, 2, 7).unwrap();
        assert_eq!(e.name(), "hash-v1");
        assert_eq!(e.dim(), 64);
        let a = e.embed("Vector search with HNSW graphs").unwrap();
        assert_eq!(a, e.embed("vector SEARCH, with hnsw graphs!").unwrap());
        assert!((dot(&a, &a) - 1.0).abs() < 1e-5);
        let near = e.embed("hnsw graphs for vector search").unwrap();
        let far = e.embed("chocolate cake recipe with butter").unwrap();
        assert!(dot(&a, &near) > dot(&a, &far));
        assert!(e.embed("  ,,, ").is_err());
        assert!(HashingEmbedder::new("x", 0, 1, 0).is_err());
        let other = HashingEmbedder::new("hash-v2", 64, 2, 8).unwrap();
        assert_ne!(a, other.embed("vector search with hnsw graphs").unwrap());
    }

    #[test]
    fn corpus_scan_paginates_by_key() {
        let c = InMemoryCorpus::from_docs(synthetic_corpus(10, 3, 1));
        assert_eq!(c.len(), 10);
        let first = c.scan(None, 4);
        assert_eq!(
            first.iter().map(|d| d.key).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        let next = c.scan(Some(3), 100);
        assert_eq!(next.len(), 6);
        assert!(c.delete(5));
        assert!(!c.delete(5));
        assert_eq!(c.scan(Some(3), 100).len(), 5);
        c.upsert(Document {
            key: u64::MAX,
            text: "edge".into(),
            tags: 0,
        });
        assert!(c.scan(Some(u64::MAX), 10).is_empty());
        assert_eq!(c.get(u64::MAX).unwrap().text, "edge");
        assert!(!c.is_empty());
    }

    #[test]
    fn synthetic_data_is_deterministic_and_topical() {
        assert_eq!(synthetic_corpus(20, 4, 9), synthetic_corpus(20, 4, 9));
        let q = synthetic_query(2, 3);
        assert!(q.split(' ').all(|w| w.starts_with("t2w")));
        let docs = synthetic_corpus(50, 4, 9);
        assert!(docs.iter().all(|d| d.tags.count_ones() == 1));
    }
}
