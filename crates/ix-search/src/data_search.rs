//! Data structure search algorithms.
//!
//! Linear, binary, interpolation, jump, exponential search, and hashing.

/// Linear search — O(n). Works on unsorted data.
pub fn linear_search<T: PartialEq>(data: &[T], target: &T) -> Option<usize> {
    data.iter().position(|item| item == target)
}

/// Binary search — O(log n). Requires sorted data.
pub fn binary_search(data: &[f64], target: f64) -> Option<usize> {
    let mut low = 0usize;
    let mut high = data.len();

    while low < high {
        let mid = low + (high - low) / 2;
        if (data[mid] - target).abs() < 1e-10 {
            return Some(mid);
        } else if data[mid] < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    None
}

/// Interpolation search — O(log log n) for uniformly distributed sorted data.
pub fn interpolation_search(data: &[f64], target: f64) -> Option<usize> {
    if data.is_empty() {
        return None;
    }

    let mut low = 0i64;
    let mut high = (data.len() - 1) as i64;

    while low <= high && target >= data[low as usize] && target <= data[high as usize] {
        let range = data[high as usize] - data[low as usize];
        if range.abs() < 1e-15 {
            if (data[low as usize] - target).abs() < 1e-10 {
                return Some(low as usize);
            }
            return None;
        }

        let pos = low + ((target - data[low as usize]) / range * (high - low) as f64) as i64;

        if pos < low || pos > high {
            return None;
        }

        let pos = pos as usize;
        if (data[pos] - target).abs() < 1e-10 {
            return Some(pos);
        } else if data[pos] < target {
            low = pos as i64 + 1;
        } else {
            high = pos as i64 - 1;
        }
    }

    None
}

/// Jump search (block search) — O(√n). Sorted data.
pub fn jump_search(data: &[f64], target: f64) -> Option<usize> {
    let n = data.len();
    if n == 0 {
        return None;
    }

    let jump = (n as f64).sqrt() as usize;
    let mut prev = 0;
    let mut curr = 0;

    // Jump forward
    while curr < n && data[curr] < target {
        prev = curr;
        curr += jump;
    }

    // Linear search within block
    for (i, &val) in data.iter().enumerate().take(n.min(curr + 1)).skip(prev) {
        if (val - target).abs() < 1e-10 {
            return Some(i);
        }
        if val > target {
            return None;
        }
    }

    None
}

/// Exponential search — O(log n). Good when target is near the beginning.
pub fn exponential_search(data: &[f64], target: f64) -> Option<usize> {
    let n = data.len();
    if n == 0 {
        return None;
    }

    if (data[0] - target).abs() < 1e-10 {
        return Some(0);
    }

    // Find range
    let mut bound = 1;
    while bound < n && data[bound] <= target {
        bound *= 2;
    }

    // Binary search within range
    let low = bound / 2;
    let high = n.min(bound + 1);
    binary_search_range(data, target, low, high)
}

fn binary_search_range(
    data: &[f64],
    target: f64,
    mut low: usize,
    mut high: usize,
) -> Option<usize> {
    while low < high {
        let mid = low + (high - low) / 2;
        if (data[mid] - target).abs() < 1e-10 {
            return Some(mid);
        } else if data[mid] < target {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    None
}

/// Simple hash table for string-keyed search.
pub struct HashTable {
    buckets: Vec<Vec<(String, String)>>,
    size: usize,
    count: usize,
}

impl HashTable {
    pub fn new(size: usize) -> Self {
        Self {
            buckets: vec![Vec::new(); size],
            size,
            count: 0,
        }
    }

    fn hash(&self, key: &str) -> usize {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in key.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        (hash as usize) % self.size
    }

    pub fn insert(&mut self, key: String, value: String) {
        let idx = self.hash(&key);
        // Update if exists
        for entry in &mut self.buckets[idx] {
            if entry.0 == key {
                entry.1 = value;
                return;
            }
        }
        self.buckets[idx].push((key, value));
        self.count += 1;
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        let idx = self.hash(key);
        self.buckets[idx]
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn remove(&mut self, key: &str) -> Option<String> {
        let idx = self.hash(key);
        let pos = self.buckets[idx].iter().position(|(k, _)| k == key);
        pos.map(|p| {
            self.count -= 1;
            self.buckets[idx].remove(p).1
        })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Load factor.
    pub fn load_factor(&self) -> f64 {
        self.count as f64 / self.size as f64
    }
}

/// Ternary search — find max of unimodal function on [lo, hi].
pub fn ternary_search_max<F: Fn(f64) -> f64>(
    f: F,
    mut lo: f64,
    mut hi: f64,
    iterations: usize,
) -> f64 {
    for _ in 0..iterations {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if f(m1) < f(m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_search() {
        let data = vec![5, 3, 8, 1, 9];
        assert_eq!(linear_search(&data, &8), Some(2));
        assert_eq!(linear_search(&data, &7), None);
    }

    #[test]
    fn test_binary_search() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(binary_search(&data, 42.0), Some(42));
        assert_eq!(binary_search(&data, 42.5), None);
    }

    #[test]
    fn test_interpolation_search() {
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 2.0).collect();
        assert_eq!(interpolation_search(&data, 42.0), Some(21));
    }

    #[test]
    fn test_jump_search() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(jump_search(&data, 67.0), Some(67));
    }

    #[test]
    fn test_exponential_search() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        assert_eq!(exponential_search(&data, 3.0), Some(3));
        assert_eq!(exponential_search(&data, 999.0), Some(999));
    }

    #[test]
    fn test_hash_table() {
        let mut ht = HashTable::new(64);
        ht.insert("hello".into(), "world".into());
        ht.insert("foo".into(), "bar".into());

        assert_eq!(ht.get("hello"), Some("world"));
        assert_eq!(ht.get("foo"), Some("bar"));
        assert_eq!(ht.get("missing"), None);
        assert_eq!(ht.len(), 2);
    }

    #[test]
    fn test_ternary_search() {
        // Find max of -(x-3)^2 + 10 => x = 3
        let x = ternary_search_max(|x| -(x - 3.0).powi(2) + 10.0, 0.0, 10.0, 100);
        assert!((x - 3.0).abs() < 1e-6);
    }
}
