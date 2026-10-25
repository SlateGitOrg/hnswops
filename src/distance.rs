//! Distance metrics.
//!
//! All metrics are expressed as a *distance* (smaller is closer) so the graph
//! code can treat them uniformly:
//!
//! | metric        | distance                 | note                                  |
//! |---------------|--------------------------|---------------------------------------|
//! | `L2`          | squared Euclidean        | monotone in true L2, no `sqrt`        |
//! | `Cosine`      | `1 - <a,b>` on unit vectors | vectors are normalized at insert   |
//! | `InnerProduct`| `-<a,b>`                 | maximum inner product search          |

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Which similarity the index is built for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    /// Squared Euclidean distance.
    #[default]
    L2,
    /// Cosine distance; inputs are L2-normalized on insert and on query.
    Cosine,
    /// Negative dot product (maximum inner product search).
    InnerProduct,
}

impl Metric {
    /// Distance between two equal-length vectors already prepared for this metric.
    #[inline]
    pub fn distance(self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Cosine => 1.0 - dot(a, b),
            Metric::InnerProduct => -dot(a, b),
        }
    }

    /// Validate a raw input vector and return the form stored in the index
    /// (normalized for cosine, unchanged otherwise).
    pub fn prepare(self, v: &[f32]) -> Result<Vec<f32>> {
        if let Some(i) = v.iter().position(|x| !x.is_finite()) {
            return Err(Error::InvalidVector(format!("component {i} is not finite")));
        }
        match self {
            Metric::Cosine => {
                let norm = dot(v, v).sqrt();
                if norm == 0.0 {
                    return Err(Error::InvalidVector(
                        "zero vector has no direction under the cosine metric".into(),
                    ));
                }
                Ok(v.iter().map(|x| x / norm).collect())
            }
            Metric::L2 | Metric::InnerProduct => Ok(v.to_vec()),
        }
    }

    /// Stable one-byte code used by the on-disk format.
    pub fn code(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::Cosine => 1,
            Metric::InnerProduct => 2,
        }
    }

    /// Inverse of [`Metric::code`].
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Metric::L2),
            1 => Some(Metric::Cosine),
            2 => Some(Metric::InnerProduct),
            _ => None,
        }
    }
}

impl std::str::FromStr for Metric {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "l2" | "euclidean" => Ok(Metric::L2),
            "cosine" | "angular" => Ok(Metric::Cosine),
            "ip" | "dot" | "inner_product" => Ok(Metric::InnerProduct),
            other => Err(Error::InvalidConfig(format!("unknown metric '{other}'"))),
        }
    }
}

/// Squared Euclidean distance. Written with eight independent accumulators so
/// LLVM vectorizes it without `unsafe` intrinsics.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let (rem_a, rem_b) = (chunks_a.remainder(), chunks_b.remainder());
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..8 {
            let d = ca[i] - cb[i];
            acc[i] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in rem_a.iter().zip(rem_b) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Dot product, vectorization-friendly like [`l2_squared`].
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let (rem_a, rem_b) = (chunks_a.remainder(), chunks_b.remainder());
    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..8 {
            acc[i] += ca[i] * cb[i];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in rem_a.iter().zip(rem_b) {
        sum += x * y;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_l2(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
    }

    #[test]
    fn l2_matches_naive_for_odd_lengths() {
        for len in [1usize, 7, 8, 9, 17, 128, 131] {
            let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.25).collect();
            let got = l2_squared(&a, &b);
            let want = naive_l2(&a, &b);
            assert!((got - want).abs() <= want.abs() * 1e-5 + 1e-5, "len {len}");
            let d = dot(&a, &b);
            let dn: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            assert!((d - dn).abs() <= dn.abs() * 1e-5 + 1e-5);
        }
    }

    #[test]
    fn cosine_prepare_normalizes_and_rejects_bad_input() {
        let v = Metric::Cosine.prepare(&[3.0, 4.0]).unwrap();
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        assert!(Metric::Cosine.distance(&v, &v).abs() < 1e-6);
        assert!(Metric::Cosine.prepare(&[0.0, 0.0]).is_err());
        assert!(Metric::L2.prepare(&[f32::NAN]).is_err());
        assert_eq!(Metric::InnerProduct.distance(&[1.0, 2.0], &[3.0, 4.0]), -11.0);
    }

    #[test]
    fn codes_and_parsing_round_trip() {
        for m in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            assert_eq!(Metric::from_code(m.code()), Some(m));
        }
        assert_eq!(Metric::from_code(9), None);
        assert_eq!("angular".parse::<Metric>().unwrap(), Metric::Cosine);
        assert_eq!("IP".parse::<Metric>().unwrap(), Metric::InnerProduct);
        assert!("manhattan".parse::<Metric>().is_err());
    }
}
