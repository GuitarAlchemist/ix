//! Count-Min Sketch: frequency estimation with bounded overcount.
//!
//! Use case for agents: track how often each skill is invoked,
//! estimate query frequencies for caching decisions — all in O(1) space.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Count-Min Sketch for frequency estimation.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    table: Vec<Vec<u64>>,
    width: usize,
    depth: usize,
}

impl CountMinSketch {
    /// Create with given width (columns) and depth (rows/hash functions).
    /// Error <= total_count / width with probability >= 1 - (1/e)^depth.
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            table: vec![vec![0u64; width]; depth],
            width,
            depth,
        }
    }

    /// Create with desired error rate and confidence.
    /// epsilon: error factor (smaller = more accurate, larger table)
    /// delta: failure probability (smaller = more confident, more hashes)
    pub fn with_error(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width.max(1), depth.max(1))
    }

    /// Increment the count for an item.
    pub fn add<T: Hash>(&mut self, item: &T) {
        self.add_count(item, 1);
    }

    /// Add a specific count for an item.
    pub fn add_count<T: Hash>(&mut self, item: &T, count: u64) {
        for row in 0..self.depth {
            let col = self.hash_index(item, row);
            self.table[row][col] += count;
        }
    }

    /// Estimate the frequency of an item (always >= true count).
    pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
        (0..self.depth)
            .map(|row| {
                let col = self.hash_index(item, row);
                self.table[row][col]
            })
            .min()
            .unwrap_or(0)
    }

    /// Total count of all items.
    pub fn total_count(&self) -> u64 {
        // Sum of any single row equals total count
        self.table[0].iter().sum()
    }

    /// Merge another sketch (must have same dimensions).
    pub fn merge(&mut self, other: &Self) -> Result<(), &'static str> {
        if self.width != other.width || self.depth != other.depth {
            return Err("Sketches must have same dimensions");
        }
        for row in 0..self.depth {
            for col in 0..self.width {
                self.table[row][col] += other.table[row][col];
            }
        }
        Ok(())
    }

    fn hash_index<T: Hash>(&self, item: &T, seed: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        seed.hash(&mut hasher);
        (hasher.finish() as usize) % self.width
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_min_basic() {
        let mut cms = CountMinSketch::new(100, 5);
        for _ in 0..10 {
            cms.add(&"hello");
        }
        for _ in 0..3 {
            cms.add(&"world");
        }

        assert!(cms.estimate(&"hello") >= 10);
        assert!(cms.estimate(&"world") >= 3);
        assert_eq!(cms.total_count(), 13);
    }

    #[test]
    fn test_count_min_accuracy() {
        let mut cms = CountMinSketch::with_error(0.01, 0.01);

        // Insert known frequencies
        for _ in 0..1000 {
            cms.add(&"frequent");
        }
        for _ in 0..10 {
            cms.add(&"rare");
        }

        let freq_est = cms.estimate(&"frequent");
        let rare_est = cms.estimate(&"rare");

        // Should be close to true values (overcount possible)
        assert!(freq_est >= 1000);
        assert!(rare_est >= 10);
        // Overcount should be bounded
        assert!(freq_est < 1050, "Overcount too high: {}", freq_est);
    }

    #[test]
    fn test_count_min_merge() {
        let mut cms1 = CountMinSketch::new(100, 5);
        let mut cms2 = CountMinSketch::new(100, 5);

        cms1.add(&"a");
        cms2.add(&"a");

        cms1.merge(&cms2).unwrap();
        assert!(cms1.estimate(&"a") >= 2);
    }
}
