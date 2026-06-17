//! Bloom Filter: space-efficient set membership test.
//!
//! - insert(item): add an item
//! - contains(item): check if item *might* be in the set
//! - False positives possible, false negatives impossible
//!
//! Use case for agents: quickly check "has this query been seen before?"
//! or "does this skill exist?" without loading full data.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A Bloom filter.
///
/// Serialization (`serde`) lets the filter round-trip to a portable blob — e.g.
/// the `ix-duck` `ix_bloom_*` SQL UDFs persist it as a column value. Hashing uses
/// `DefaultHasher` (fixed seed), so a serialized filter probes identically on any
/// machine running the same Rust std version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
    size: usize,
    count: usize,
}

impl BloomFilter {
    /// Create a Bloom filter with optimal parameters for `capacity` items
    /// and desired false positive rate `fp_rate`.
    pub fn new(capacity: usize, fp_rate: f64) -> Self {
        let size = optimal_bits(capacity, fp_rate);
        let num_hashes = optimal_hashes(size, capacity);
        Self {
            bits: vec![false; size],
            num_hashes,
            size,
            count: 0,
        }
    }

    /// Create with explicit size and hash count.
    pub fn with_params(size: usize, num_hashes: usize) -> Self {
        Self {
            bits: vec![false; size],
            num_hashes,
            size,
            count: 0,
        }
    }

    /// Insert an item.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(item, i);
            self.bits[idx] = true;
        }
        self.count += 1;
    }

    /// Whether the internal arrays are self-consistent. A filter built via the
    /// constructors always is, but one *deserialized* from an untrusted blob may
    /// not be — guarding keeps query methods total (no `% 0` / out-of-bounds panic).
    fn consistent(&self) -> bool {
        self.size > 0 && self.num_hashes > 0 && self.bits.len() == self.size
    }

    /// Check if an item *might* be in the set.
    /// Returns false = definitely not in set, true = probably in set.
    /// A structurally-degenerate filter (e.g. from a malformed blob) returns false
    /// rather than panicking.
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        if !self.consistent() {
            return false;
        }
        (0..self.num_hashes).all(|i| {
            let idx = self.hash_index(item, i);
            self.bits[idx]
        })
    }

    /// Estimated false positive rate at current fill level.
    pub fn estimated_fp_rate(&self) -> f64 {
        let ones = self.bits.iter().filter(|&&b| b).count() as f64;
        let ratio = ones / self.size as f64;
        ratio.powi(self.num_hashes as i32)
    }

    /// Number of items inserted.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Bit array size.
    pub fn bit_size(&self) -> usize {
        self.size
    }

    /// Clear all bits.
    pub fn clear(&mut self) {
        self.bits.fill(false);
        self.count = 0;
    }

    /// Union of two Bloom filters (must have same parameters).
    pub fn union(&self, other: &Self) -> Option<Self> {
        if self.size != other.size || self.num_hashes != other.num_hashes {
            return None;
        }
        let bits = self
            .bits
            .iter()
            .zip(other.bits.iter())
            .map(|(&a, &b)| a || b)
            .collect();
        Some(Self {
            bits,
            num_hashes: self.num_hashes,
            size: self.size,
            count: self.count + other.count, // Approximate
        })
    }

    fn hash_index<T: Hash>(&self, item: &T, seed: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        seed.hash(&mut hasher);
        (hasher.finish() as usize) % self.size
    }
}

/// Optimal number of bits for given capacity and false positive rate.
fn optimal_bits(capacity: usize, fp_rate: f64) -> usize {
    let ln2_sq = std::f64::consts::LN_2 * std::f64::consts::LN_2;
    let m = -(capacity as f64 * fp_rate.ln()) / ln2_sq;
    m.ceil() as usize
}

/// Optimal number of hash functions.
fn optimal_hashes(num_bits: usize, capacity: usize) -> usize {
    let k = (num_bits as f64 / capacity as f64) * std::f64::consts::LN_2;
    k.ceil().max(1.0) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_basic() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.insert(&"hello");
        bf.insert(&"world");

        assert!(bf.contains(&"hello"));
        assert!(bf.contains(&"world"));
        // "foobar" should (almost certainly) not be there
        // Can't guarantee due to probabilistic nature, but with 0.01 fp rate it's very unlikely
    }

    #[test]
    fn test_bloom_false_positive_rate() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..1000 {
            bf.insert(&i);
        }

        // Check items NOT inserted
        let mut false_positives = 0;
        let test_count = 10000;
        for i in 1000..(1000 + test_count) {
            if bf.contains(&i) {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f64 / test_count as f64;
        assert!(fp_rate < 0.05, "FP rate {} is too high", fp_rate);
    }

    #[test]
    fn test_bloom_no_false_negatives() {
        let mut bf = BloomFilter::new(100, 0.01);
        for i in 0..100 {
            bf.insert(&i);
        }
        for i in 0..100 {
            assert!(bf.contains(&i), "False negative for {}", i);
        }
    }

    #[test]
    fn test_bloom_degenerate_blob_no_panic() {
        // A malformed blob (size 0, empty bits) must not panic on probe (`% 0`).
        let bf: BloomFilter =
            serde_json::from_str(r#"{"bits":[],"num_hashes":1,"size":0,"count":0}"#).unwrap();
        assert!(!bf.contains(&"anything"));
        // size > 0 but bits shorter than size would index out of bounds.
        let bf2: BloomFilter =
            serde_json::from_str(r#"{"bits":[true],"num_hashes":2,"size":8,"count":1}"#).unwrap();
        assert!(!bf2.contains(&42));
    }

    #[test]
    fn test_bloom_union() {
        let mut bf1 = BloomFilter::with_params(100, 3);
        let mut bf2 = BloomFilter::with_params(100, 3);

        bf1.insert(&"a");
        bf2.insert(&"b");

        let union = bf1.union(&bf2).unwrap();
        assert!(union.contains(&"a"));
        assert!(union.contains(&"b"));
    }
}
