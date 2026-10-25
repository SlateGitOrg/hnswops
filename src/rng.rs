//! Small deterministic PRNG (SplitMix64).
//!
//! The index needs randomness only for level assignment and sampling; a
//! dependency-free, seedable generator keeps every build and test reproducible.

/// SplitMix64 generator (Steele, Lea & Flood, 2014). Passes BigCrush and is
/// more than adequate for level sampling and synthetic data.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from a seed. Equal seeds yield equal streams.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Current internal state (persisted so a reloaded index continues the stream).
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Next 64 uniformly distributed bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in the open interval (0, 1); never returns 0 so `ln` is safe.
    pub fn next_f64_open(&mut self) -> f64 {
        // 53 random mantissa bits, shifted by half an ulp to exclude 0.
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// Uniform integer in `0..bound` (`bound > 0`), via Lemire's multiply-shift.
    pub fn below(&mut self, bound: u64) -> u64 {
        debug_assert!(bound > 0);
        ((u128::from(self.next_u64()) * u128::from(bound)) >> 64) as u64
    }

    /// Standard normal sample (Box–Muller).
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64_open();
        let u2 = self.next_f64_open();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_seed_sensitive() {
        let a: Vec<u64> = {
            let mut r = SplitMix64::new(7);
            (0..4).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = SplitMix64::new(7);
            (0..4).map(|_| r.next_u64()).collect()
        };
        let c = SplitMix64::new(8).next_u64();
        assert_eq!(a, b);
        assert_ne!(a[0], c);
    }

    #[test]
    fn ranges_hold() {
        let mut r = SplitMix64::new(1);
        for _ in 0..10_000 {
            let f = r.next_f64_open();
            assert!(f > 0.0 && f < 1.0);
            assert!(r.below(10) < 10);
        }
        let mean: f64 = (0..20_000).map(|_| r.next_gaussian()).sum::<f64>() / 20_000.0;
        assert!(mean.abs() < 0.05, "gaussian mean {mean}");
    }
}
