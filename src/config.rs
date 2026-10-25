//! Index construction and query parameters.

use serde::{Deserialize, Serialize};

use crate::distance::Metric;
use crate::error::{Error, Result};

/// Hard cap on the number of layers; with `M >= 2` the probability of a node
/// reaching layer 16 is below `2^-16` per node, so this is never binding in practice.
pub const MAX_LEVEL: usize = 16;

/// Parameters of an HNSW index. See `docs/ARCHITECTURE.md` for how each one
/// trades build time, memory and recall.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Vector dimensionality.
    pub dim: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Target out-degree on layers >= 1; layer 0 allows `2 * m`.
    pub m: usize,
    /// Beam width used while inserting.
    pub ef_construction: usize,
    /// Default beam width used while searching (overridable per query).
    pub ef_search: usize,
    /// Seed for level assignment, making builds reproducible.
    pub seed: u64,
    /// When the estimated fraction of nodes matching a filter is below this,
    /// filtered search scans the matching set exactly instead of walking the graph.
    pub filter_brute_force_below: f64,
}

impl HnswConfig {
    /// Defaults that work well for 10^3..10^6 vectors of 32..1024 dimensions.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            metric: Metric::L2,
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x005E_ED0F_4A55,
            filter_brute_force_below: 0.05,
        }
    }

    /// Builder-style metric override.
    pub fn with_metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Builder-style `M` override.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Builder-style `efConstruction` override.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Builder-style `efSearch` override.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Builder-style seed override.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Maximum out-degree on layer `level`.
    #[inline]
    pub fn max_degree(&self, level: usize) -> usize {
        if level == 0 {
            self.m * 2
        } else {
            self.m
        }
    }

    /// Level-generation normalization factor `mL = 1 / ln(M)` from the paper.
    pub fn level_multiplier(&self) -> f64 {
        1.0 / (self.m as f64).ln()
    }

    /// Reject configurations that would produce a broken or degenerate graph.
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.dim > 65_536 {
            return Err(Error::InvalidConfig(format!(
                "dim must be in 1..=65536, got {}",
                self.dim
            )));
        }
        if !(2..=256).contains(&self.m) {
            return Err(Error::InvalidConfig(format!(
                "m must be in 2..=256, got {}",
                self.m
            )));
        }
        if self.ef_construction < self.m {
            return Err(Error::InvalidConfig(format!(
                "ef_construction ({}) must be >= m ({})",
                self.ef_construction, self.m
            )));
        }
        if self.ef_search == 0 {
            return Err(Error::InvalidConfig("ef_search must be >= 1".into()));
        }
        if !(0.0..=1.0).contains(&self.filter_brute_force_below) {
            return Err(Error::InvalidConfig(
                "filter_brute_force_below must be within [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate_and_builders_apply() {
        let c = HnswConfig::new(8)
            .with_m(8)
            .with_ef_construction(40)
            .with_ef_search(10)
            .with_seed(3)
            .with_metric(Metric::Cosine);
        c.validate().unwrap();
        assert_eq!(c.max_degree(0), 16);
        assert_eq!(c.max_degree(3), 8);
        assert!((c.level_multiplier() - 1.0 / 8f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_values() {
        assert!(HnswConfig::new(0).validate().is_err());
        assert!(HnswConfig::new(4).with_m(1).validate().is_err());
        assert!(HnswConfig::new(4)
            .with_m(16)
            .with_ef_construction(4)
            .validate()
            .is_err());
        assert!(HnswConfig::new(4).with_ef_search(0).validate().is_err());
        let mut c = HnswConfig::new(4);
        c.filter_brute_force_below = 1.5;
        assert!(c.validate().is_err());
    }
}
