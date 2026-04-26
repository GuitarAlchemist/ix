//! Cuckoo Filter: like Bloom filter but supports deletion.
//!
//! Use case for agents: track active sessions, skill availability,
//! or active tool registrations with the ability to remove entries.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A Cuckoo Filter supporting insert, lookup, and delete.
#[derive(Debug, Clone)]
pub struct CuckooFilter {
    buckets: Vec<Vec<u16>>, // Each bucket holds up to BUCKET_SIZE fingerprints
    num_buckets: usize,
    max_kicks: usize,
    count: usize,
}

const BUCKET_SIZE: usize = 4;

impl CuckooFilter {
    /// Create with given capacity (approximate number of items).
    pub fn new(capacity: usize) -> Self {
        let num_buckets = (capacity / BUCKET_SIZE).next_power_of_two().max(1);
        Self {
            buckets: vec![Vec::with_capacity(BUCKET_SIZE); num_buckets],
            num_buckets,
            max_kicks: 500,
            count: 0,
        }
    }

    /// Insert an item. Returns false if the filter is full.
    pub fn insert<T: Hash>(&mut self, item: &T) -> bool {
        let (fp, i1, i2) = self.indices(item);

        // Try bucket i1
        if self.buckets[i1].len() < BUCKET_SIZE {
            self.buckets[i1].push(fp);
            self.count += 1;
            return true;
        }

        // Try bucket i2
        if self.buckets[i2].len() < BUCKET_SIZE {
            self.buckets[i2].push(fp);
            self.count += 1;
            return true;
        }

        // Both full: kick existing entries
        let mut idx = i1;
        let mut fingerprint = fp;

        for _ in 0..self.max_kicks {
            // Swap with random entry in bucket
            let pos = fingerprint as usize % self.buckets[idx].len().max(1);
            if pos < self.buckets[idx].len() {
                std::mem::swap(&mut fingerprint, &mut self.buckets[idx][pos]);
            }

            // Find alternate bucket
            idx = self.alt_index(idx, fingerprint);

            if self.buckets[idx].len() < BUCKET_SIZE {
                self.buckets[idx].push(fingerprint);
                self.count += 1;
                return true;
            }
        }

        false // Filter is too full
    }

    /// Check if an item might be in the filter.
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let (fp, i1, i2) = self.indices(item);
        self.buckets[i1].contains(&fp) || self.buckets[i2].contains(&fp)
    }

    /// Remove an item. Returns true if found and removed.
    pub fn remove<T: Hash>(&mut self, item: &T) -> bool {
        let (fp, i1, i2) = self.indices(item);

        if let Some(pos) = self.buckets[i1].iter().position(|&f| f == fp) {
            self.buckets[i1].swap_remove(pos);
            self.count -= 1;
            return true;
        }
        if let Some(pos) = self.buckets[i2].iter().position(|&f| f == fp) {
            self.buckets[i2].swap_remove(pos);
            self.count -= 1;
            return true;
        }

        false
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Load factor (0.0 to 1.0).
    pub fn load_factor(&self) -> f64 {
        self.count as f64 / (self.num_buckets * BUCKET_SIZE) as f64
    }

    fn fingerprint<T: Hash>(item: &T) -> u16 {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let h = hasher.finish();
        // Non-zero fingerprint
        ((h >> 32) as u16).max(1)
    }

    fn index<T: Hash>(&self, item: &T) -> usize {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        (hasher.finish() as usize) % self.num_buckets
    }

    fn alt_index(&self, index: usize, fingerprint: u16) -> usize {
        let mut hasher = DefaultHasher::new();
        fingerprint.hash(&mut hasher);
        (index ^ (hasher.finish() as usize)) % self.num_buckets
    }

    fn indices<T: Hash>(&self, item: &T) -> (u16, usize, usize) {
        let fp = Self::fingerprint(item);
        let i1 = self.index(item);
        let i2 = self.alt_index(i1, fp);
        (fp, i1, i2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuckoo_insert_contains() {
        let mut cf = CuckooFilter::new(100);
        assert!(cf.insert(&"hello"));
        assert!(cf.insert(&"world"));

        assert!(cf.contains(&"hello"));
        assert!(cf.contains(&"world"));
        assert_eq!(cf.len(), 2);
    }

    #[test]
    fn test_cuckoo_delete() {
        let mut cf = CuckooFilter::new(100);
        cf.insert(&"hello");
        assert!(cf.contains(&"hello"));

        assert!(cf.remove(&"hello"));
        assert!(!cf.contains(&"hello"));
        assert_eq!(cf.len(), 0);
    }

    #[test]
    fn test_cuckoo_capacity() {
        let mut cf = CuckooFilter::new(100);
        let mut inserted = 0;
        for i in 0..100 {
            if cf.insert(&i) {
                inserted += 1;
            }
        }
        assert!(inserted >= 80, "Should insert most items, got {}", inserted);
    }
}
