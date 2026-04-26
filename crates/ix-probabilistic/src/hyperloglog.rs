//! HyperLogLog: cardinality estimation with ~1.6KB memory.
//!
//! Use case for agents: estimate unique queries, unique skills used,
//! unique error types — without storing all values.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// HyperLogLog cardinality estimator.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    precision: usize,   // p: number of bits for bucket index
    num_buckets: usize, // m = 2^p
}

impl HyperLogLog {
    /// Create with precision p (4 <= p <= 18). Higher p = more memory, better accuracy.
    /// Memory: 2^p bytes. Standard error: 1.04 / sqrt(2^p).
    pub fn new(precision: usize) -> Self {
        let precision = precision.clamp(4, 18);
        let num_buckets = 1 << precision;
        Self {
            registers: vec![0u8; num_buckets],
            precision,
            num_buckets,
        }
    }

    /// Standard precision (p=14): ~16KB, ~0.81% error.
    pub fn standard() -> Self {
        Self::new(14)
    }

    /// Add an item.
    pub fn add<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();

        let bucket = (hash as usize) & (self.num_buckets - 1);
        let remaining = hash >> self.precision;
        let leading_zeros = if remaining == 0 {
            (64 - self.precision) as u8
        } else {
            remaining.leading_zeros() as u8 - self.precision as u8 + 1
        };

        self.registers[bucket] = self.registers[bucket].max(leading_zeros);
    }

    /// Estimate the number of distinct items.
    pub fn count(&self) -> f64 {
        let m = self.num_buckets as f64;
        let alpha = match self.num_buckets {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };

        // Harmonic mean
        let raw_estimate = alpha * m * m
            / self
                .registers
                .iter()
                .map(|&r| 2.0_f64.powi(-(r as i32)))
                .sum::<f64>();

        // Small range correction
        if raw_estimate <= 2.5 * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return m * (m / zeros as f64).ln();
            }
        }

        raw_estimate
    }

    /// Merge another HLL (must have same precision).
    pub fn merge(&mut self, other: &Self) -> Result<(), &'static str> {
        if self.precision != other.precision {
            return Err("HyperLogLog instances must have same precision");
        }
        for (i, &r) in other.registers.iter().enumerate() {
            self.registers[i] = self.registers[i].max(r);
        }
        Ok(())
    }

    /// Estimated error rate.
    pub fn error_rate(&self) -> f64 {
        1.04 / (self.num_buckets as f64).sqrt()
    }

    /// Memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.num_buckets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hll_basic() {
        let mut hll = HyperLogLog::new(10); // ~1KB
        for i in 0..1000 {
            hll.add(&i);
        }

        let estimate = hll.count();
        // Should be within ~10% for p=10
        assert!(
            (estimate - 1000.0).abs() / 1000.0 < 0.15,
            "Estimate {} too far from 1000",
            estimate
        );
    }

    #[test]
    fn test_hll_duplicates_dont_increase() {
        let mut hll = HyperLogLog::new(12);
        for _ in 0..1000 {
            hll.add(&"same_item");
        }
        let estimate = hll.count();
        assert!(
            estimate < 5.0,
            "Duplicates should not increase count, got {}",
            estimate
        );
    }

    #[test]
    fn test_hll_merge() {
        let mut hll1 = HyperLogLog::new(10);
        let mut hll2 = HyperLogLog::new(10);

        for i in 0..500 {
            hll1.add(&i);
        }
        for i in 500..1000 {
            hll2.add(&i);
        }

        hll1.merge(&hll2).unwrap();
        let estimate = hll1.count();
        assert!(
            (estimate - 1000.0).abs() / 1000.0 < 0.15,
            "Merged estimate {} too far from 1000",
            estimate
        );
    }

    #[test]
    fn test_hll_memory() {
        let hll = HyperLogLog::standard();
        assert_eq!(hll.memory_bytes(), 16384); // 2^14
    }
}
