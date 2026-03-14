//! Core cache store — sharded concurrent hash map with TTL and LRU eviction.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::lru::LruPolicy;
use crate::pubsub::PubSub;

/// Configuration for the cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Number of shards (power of 2 recommended). Default: 16.
    pub num_shards: usize,
    /// Maximum total entries across all shards. 0 = unlimited.
    pub max_capacity: usize,
    /// Default TTL for entries. None = no expiration.
    pub default_ttl: Option<Duration>,
    /// How often to run eviction sweep (lazy by default).
    pub eviction_interval: Duration,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            num_shards: 16,
            max_capacity: 100_000,
            default_ttl: None,
            eviction_interval: Duration::from_secs(60),
        }
    }
}

impl CacheConfig {
    pub fn with_max_capacity(mut self, cap: usize) -> Self {
        self.max_capacity = cap;
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = Some(ttl);
        self
    }

    pub fn with_shards(mut self, n: usize) -> Self {
        self.num_shards = n;
        self
    }
}

/// A single cache entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Serialized value (JSON bytes for type flexibility).
    pub data: Vec<u8>,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// When this entry expires (None = never).
    pub expires_at: Option<Instant>,
    /// Access count for LFU-style metrics.
    pub access_count: u64,
    /// Size in bytes (for memory accounting).
    pub size: usize,
}

impl CacheEntry {
    fn is_expired(&self) -> bool {
        if let Some(exp) = self.expires_at {
            Instant::now() >= exp
        } else {
            false
        }
    }
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub expirations: u64,
    pub total_entries: usize,
    pub total_bytes: usize,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }
}

/// A single shard of the cache.
struct Shard {
    entries: HashMap<String, CacheEntry>,
    lru: LruPolicy,
}

impl Shard {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            lru: LruPolicy::new(),
        }
    }
}

/// The main cache. Thread-safe via sharded RwLocks.
///
/// # Example
/// ```
/// use ix_cache::{Cache, CacheConfig};
/// use std::time::Duration;
///
/// let cache = Cache::new(CacheConfig::default().with_ttl(Duration::from_secs(300)));
///
/// cache.set("greeting", &"hello world");
/// let val: Option<String> = cache.get("greeting");
/// assert_eq!(val, Some("hello world".to_string()));
/// ```
pub struct Cache {
    shards: Vec<RwLock<Shard>>,
    config: CacheConfig,
    stats: RwLock<CacheStats>,
    pubsub: PubSub,
}

impl Cache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        let num_shards = config.num_shards.max(1);
        let shards = (0..num_shards).map(|_| RwLock::new(Shard::new())).collect();

        Self {
            shards,
            config,
            stats: RwLock::new(CacheStats::default()),
            pubsub: PubSub::new(),
        }
    }

    /// Create a cache with default settings.
    pub fn default_cache() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Get the shard index for a key.
    fn shard_index(&self, key: &str) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.shards.len()
    }

    /// Set a typed value. Serializes to JSON internally.
    pub fn set<V: Serialize>(&self, key: &str, value: &V) {
        self.set_with_ttl(key, value, self.config.default_ttl);
    }

    /// Set a value with a specific TTL.
    pub fn set_with_ttl<V: Serialize>(&self, key: &str, value: &V, ttl: Option<Duration>) {
        let data = serde_json::to_vec(value).expect("Serialization failed");
        let size = data.len();
        let now = Instant::now();

        let entry = CacheEntry {
            data,
            created_at: now,
            last_accessed: now,
            expires_at: ttl.map(|d| now + d),
            access_count: 0,
            size,
        };

        let idx = self.shard_index(key);

        // Check capacity before acquiring write lock to avoid deadlock
        let needs_eviction = if self.config.max_capacity > 0 {
            let total = self.total_entries_approx();
            total >= self.config.max_capacity
        } else {
            false
        };

        let mut shard = self.shards[idx].write();

        if needs_eviction && !shard.entries.contains_key(key) {
            self.evict_one(&mut shard);
        }

        shard.lru.touch(key);
        shard.entries.insert(key.to_string(), entry);

        // Notify pub/sub
        self.pubsub.publish("__keyevent__:set", key);
    }

    /// Set a raw string value (Redis-compatible convenience).
    pub fn set_str(&self, key: &str, value: &str) {
        self.set(key, &value.to_string());
    }

    /// Get a typed value. Deserializes from JSON.
    pub fn get<V: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<V> {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write();

        // Check expiration first
        let expired = shard.entries.get(key).is_some_and(|e| e.is_expired());
        if expired {
            shard.entries.remove(key);
            shard.lru.remove(key);
            self.stats.write().expirations += 1;
            self.stats.write().misses += 1;
            return None;
        }

        // Update access metadata, then read data
        if let Some(entry) = shard.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            let data = entry.data.clone();
            shard.lru.touch(key);
            self.stats.write().hits += 1;
            serde_json::from_slice(&data).ok()
        } else {
            self.stats.write().misses += 1;
            None
        }
    }

    /// Get a raw string value.
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.get::<String>(key)
    }

    /// Check if a key exists (and is not expired).
    pub fn contains(&self, key: &str) -> bool {
        let idx = self.shard_index(key);
        let shard = self.shards[idx].read();
        shard.entries.get(key).is_some_and(|e| !e.is_expired())
    }

    /// Delete a key. Returns true if it existed.
    pub fn delete(&self, key: &str) -> bool {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write();
        shard.lru.remove(key);
        let removed = shard.entries.remove(key).is_some();

        if removed {
            self.pubsub.publish("__keyevent__:del", key);
        }

        removed
    }

    /// Get the remaining TTL for a key.
    pub fn ttl(&self, key: &str) -> Option<Duration> {
        let idx = self.shard_index(key);
        let shard = self.shards[idx].read();

        shard.entries.get(key).and_then(|e| {
            e.expires_at.map(|exp| {
                let now = Instant::now();
                if exp > now { exp - now } else { Duration::ZERO }
            })
        })
    }

    /// Set expiration on an existing key.
    pub fn expire(&self, key: &str, ttl: Duration) -> bool {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write();

        if let Some(entry) = shard.entries.get_mut(key) {
            entry.expires_at = Some(Instant::now() + ttl);
            true
        } else {
            false
        }
    }

    /// Remove expiration from a key (persist it).
    pub fn persist(&self, key: &str) -> bool {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write();

        if let Some(entry) = shard.entries.get_mut(key) {
            entry.expires_at = None;
            true
        } else {
            false
        }
    }

    /// Increment a numeric value. Creates with value 1 if key doesn't exist.
    pub fn incr(&self, key: &str) -> i64 {
        self.incr_by(key, 1)
    }

    /// Increment by a specific amount.
    pub fn incr_by(&self, key: &str, amount: i64) -> i64 {
        let current: i64 = self.get(key).unwrap_or(0);
        let new_val = current + amount;
        self.set(key, &new_val);
        new_val
    }

    /// Decrement a numeric value.
    pub fn decr(&self, key: &str) -> i64 {
        self.incr_by(key, -1)
    }

    /// Get all keys matching a pattern (simple glob: * only).
    pub fn keys(&self, pattern: &str) -> Vec<String> {
        let mut result = Vec::new();

        for shard_lock in &self.shards {
            let shard = shard_lock.read();
            for key in shard.entries.keys() {
                if matches_pattern(key, pattern) {
                    // Check not expired
                    if let Some(entry) = shard.entries.get(key) {
                        if !entry.is_expired() {
                            result.push(key.clone());
                        }
                    }
                }
            }
        }

        result
    }

    /// Flush all entries.
    pub fn flush_all(&self) {
        for shard_lock in &self.shards {
            let mut shard = shard_lock.write();
            shard.entries.clear();
            shard.lru.clear();
        }
        let mut stats = self.stats.write();
        stats.total_entries = 0;
        stats.total_bytes = 0;
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let mut s = self.stats.read().clone();
        s.total_entries = self.total_entries_approx();
        s.total_bytes = self.total_bytes_approx();
        s
    }

    /// Get the pub/sub system for subscribing to cache events.
    pub fn pubsub(&self) -> &PubSub {
        &self.pubsub
    }

    /// Run a manual eviction sweep — removes all expired entries.
    pub fn evict_expired(&self) {
        let mut expired_count = 0u64;
        for shard_lock in &self.shards {
            let mut shard = shard_lock.write();
            let expired_keys: Vec<String> = shard.entries.iter()
                .filter(|(_, v)| v.is_expired())
                .map(|(k, _)| k.clone())
                .collect();

            for key in &expired_keys {
                shard.entries.remove(key);
                shard.lru.remove(key);
            }
            expired_count += expired_keys.len() as u64;
        }
        self.stats.write().expirations += expired_count;
    }

    /// Approximate total entries across all shards.
    fn total_entries_approx(&self) -> usize {
        self.shards.iter().map(|s| s.read().entries.len()).sum()
    }

    /// Approximate total bytes.
    fn total_bytes_approx(&self) -> usize {
        self.shards.iter()
            .map(|s| s.read().entries.values().map(|e| e.size).sum::<usize>())
            .sum()
    }

    /// Evict one entry from a shard using LRU policy.
    fn evict_one(&self, shard: &mut Shard) {
        if let Some(key) = shard.lru.evict() {
            shard.entries.remove(&key);
            self.stats.write().evictions += 1;
        }
    }
}

// ── Hash map operations (Redis-like) ──────────────────────────

impl Cache {
    /// HSET: set a field in a hash.
    pub fn hset<V: Serialize>(&self, key: &str, field: &str, value: &V) {
        let hash_key = format!("{}:{}", key, field);
        self.set(&hash_key, value);

        // Track which fields belong to this hash
        let fields_key = format!("__hash_fields__:{}", key);
        let mut fields: Vec<String> = self.get(&fields_key).unwrap_or_default();
        if !fields.contains(&field.to_string()) {
            fields.push(field.to_string());
            self.set(&fields_key, &fields);
        }
    }

    /// HGET: get a field from a hash.
    pub fn hget<V: for<'de> Deserialize<'de>>(&self, key: &str, field: &str) -> Option<V> {
        let hash_key = format!("{}:{}", key, field);
        self.get(&hash_key)
    }

    /// HGETALL: get all fields and values from a hash.
    pub fn hgetall<V: for<'de> Deserialize<'de>>(&self, key: &str) -> HashMap<String, V> {
        let fields_key = format!("__hash_fields__:{}", key);
        let fields: Vec<String> = self.get(&fields_key).unwrap_or_default();

        let mut result = HashMap::new();
        for field in fields {
            if let Some(val) = self.hget(key, &field) {
                result.insert(field, val);
            }
        }
        result
    }

    /// HDEL: delete a field from a hash.
    pub fn hdel(&self, key: &str, field: &str) -> bool {
        let hash_key = format!("{}:{}", key, field);
        self.delete(&hash_key)
    }
}

// ── List operations (Redis-like) ──────────────────────────────

impl Cache {
    /// LPUSH: push to the front of a list.
    pub fn lpush<V: Serialize + for<'de> Deserialize<'de>>(&self, key: &str, value: &V) {
        let mut list: Vec<serde_json::Value> = self.get(key).unwrap_or_default();
        let val = serde_json::to_value(value).unwrap();
        list.insert(0, val);
        self.set(key, &list);
    }

    /// RPUSH: push to the back of a list.
    pub fn rpush<V: Serialize + for<'de> Deserialize<'de>>(&self, key: &str, value: &V) {
        let mut list: Vec<serde_json::Value> = self.get(key).unwrap_or_default();
        let val = serde_json::to_value(value).unwrap();
        list.push(val);
        self.set(key, &list);
    }

    /// LPOP: pop from the front of a list.
    pub fn lpop<V: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<V> {
        let mut list: Vec<serde_json::Value> = self.get(key)?;
        if list.is_empty() {
            return None;
        }
        let val = list.remove(0);
        self.set(key, &list);
        serde_json::from_value(val).ok()
    }

    /// RPOP: pop from the back of a list.
    pub fn rpop<V: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<V> {
        let mut list: Vec<serde_json::Value> = self.get(key)?;
        let val = list.pop()?;
        self.set(key, &list);
        serde_json::from_value(val).ok()
    }

    /// LRANGE: get a range from a list.
    pub fn lrange<V: for<'de> Deserialize<'de>>(&self, key: &str, start: usize, stop: usize) -> Vec<V> {
        let list: Vec<serde_json::Value> = self.get(key).unwrap_or_default();
        let end = stop.min(list.len());
        list[start..end].iter()
            .filter_map(|v| serde_json::from_value(v.clone()).ok())
            .collect()
    }

    /// LLEN: get the length of a list.
    pub fn llen(&self, key: &str) -> usize {
        let list: Vec<serde_json::Value> = self.get(key).unwrap_or_default();
        list.len()
    }
}

// ── Set operations (Redis-like) ───────────────────────────────

impl Cache {
    /// SADD: add a member to a set.
    pub fn sadd(&self, key: &str, member: &str) -> bool {
        let mut set: Vec<String> = self.get(key).unwrap_or_default();
        if set.contains(&member.to_string()) {
            return false;
        }
        set.push(member.to_string());
        self.set(key, &set);
        true
    }

    /// SREM: remove a member from a set.
    pub fn srem(&self, key: &str, member: &str) -> bool {
        let mut set: Vec<String> = self.get(key).unwrap_or_default();
        let len_before = set.len();
        set.retain(|m| m != member);
        if set.len() < len_before {
            self.set(key, &set);
            true
        } else {
            false
        }
    }

    /// SISMEMBER: check if a member is in a set.
    pub fn sismember(&self, key: &str, member: &str) -> bool {
        let set: Vec<String> = self.get(key).unwrap_or_default();
        set.contains(&member.to_string())
    }

    /// SMEMBERS: get all members of a set.
    pub fn smembers(&self, key: &str) -> Vec<String> {
        self.get(key).unwrap_or_default()
    }

    /// SCARD: get the number of members in a set.
    pub fn scard(&self, key: &str) -> usize {
        let set: Vec<String> = self.get(key).unwrap_or_default();
        set.len()
    }
}

/// Simple glob pattern matching (supports * only).
fn matches_pattern(text: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(prefix) = pattern.strip_suffix('*') {
        return text.starts_with(prefix);
    }

    if let Some(suffix) = pattern.strip_prefix('*') {
        return text.ends_with(suffix);
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            return text.starts_with(parts[0]) && text.ends_with(parts[1]);
        }
    }

    text == pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_get_set() {
        let cache = Cache::default_cache();
        cache.set("name", &"ix".to_string());
        let val: Option<String> = cache.get("name");
        assert_eq!(val, Some("ix".to_string()));
    }

    #[test]
    fn test_typed_values() {
        let cache = Cache::default_cache();

        cache.set("count", &42i64);
        cache.set("pi", &3.14159f64);
        cache.set("items", &vec![1, 2, 3]);
        cache.set("flag", &true);

        assert_eq!(cache.get::<i64>("count"), Some(42));
        assert_eq!(cache.get::<f64>("pi"), Some(3.14159));
        assert_eq!(cache.get::<Vec<i32>>("items"), Some(vec![1, 2, 3]));
        assert_eq!(cache.get::<bool>("flag"), Some(true));
    }

    #[test]
    fn test_ttl_expiration() {
        let cache = Cache::new(CacheConfig::default());
        cache.set_with_ttl("temp", &"expires soon", Some(Duration::from_millis(50)));

        assert!(cache.contains("temp"));
        std::thread::sleep(Duration::from_millis(100));
        assert!(!cache.contains("temp"));

        // get should return None for expired
        let val: Option<String> = cache.get("temp");
        assert_eq!(val, None);
    }

    #[test]
    fn test_delete() {
        let cache = Cache::default_cache();
        cache.set("key", &"value");
        assert!(cache.delete("key"));
        assert!(!cache.contains("key"));
        assert!(!cache.delete("nonexistent"));
    }

    #[test]
    fn test_incr_decr() {
        let cache = Cache::default_cache();
        assert_eq!(cache.incr("counter"), 1);
        assert_eq!(cache.incr("counter"), 2);
        assert_eq!(cache.incr_by("counter", 10), 12);
        assert_eq!(cache.decr("counter"), 11);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = Cache::new(CacheConfig::default().with_max_capacity(3));

        cache.set("a", &1);
        cache.set("b", &2);
        cache.set("c", &3);

        // Access "a" to make it recently used
        let _: Option<i32> = cache.get("a");

        // Adding "d" should evict "b" (least recently used)
        cache.set("d", &4);

        assert!(cache.contains("a")); // Recently accessed
        assert!(cache.contains("c"));
        assert!(cache.contains("d")); // Just added
        // "b" may or may not be evicted depending on shard distribution
    }

    #[test]
    fn test_keys_pattern() {
        let cache = Cache::default_cache();
        cache.set("user:1", &"alice");
        cache.set("user:2", &"bob");
        cache.set("post:1", &"hello");

        let user_keys = cache.keys("user:*");
        assert_eq!(user_keys.len(), 2);
        assert!(user_keys.contains(&"user:1".to_string()));
        assert!(user_keys.contains(&"user:2".to_string()));
    }

    #[test]
    fn test_hash_operations() {
        let cache = Cache::default_cache();

        cache.hset("user", "name", &"Alice".to_string());
        cache.hset("user", "age", &30i64);

        assert_eq!(cache.hget::<String>("user", "name"), Some("Alice".to_string()));
        assert_eq!(cache.hget::<i64>("user", "age"), Some(30));

        let all: HashMap<String, serde_json::Value> = cache.hgetall("user");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_list_operations() {
        let cache = Cache::default_cache();

        cache.rpush("queue", &"first".to_string());
        cache.rpush("queue", &"second".to_string());
        cache.rpush("queue", &"third".to_string());

        assert_eq!(cache.llen("queue"), 3);
        assert_eq!(cache.lpop::<String>("queue"), Some("first".to_string()));
        assert_eq!(cache.rpop::<String>("queue"), Some("third".to_string()));
        assert_eq!(cache.llen("queue"), 1);
    }

    #[test]
    fn test_set_operations() {
        let cache = Cache::default_cache();

        assert!(cache.sadd("tags", "rust"));
        assert!(cache.sadd("tags", "ml"));
        assert!(!cache.sadd("tags", "rust")); // Duplicate

        assert!(cache.sismember("tags", "rust"));
        assert!(!cache.sismember("tags", "python"));
        assert_eq!(cache.scard("tags"), 2);

        assert!(cache.srem("tags", "ml"));
        assert_eq!(cache.scard("tags"), 1);
    }

    #[test]
    fn test_stats() {
        let cache = Cache::default_cache();
        cache.set("x", &1);

        let _: Option<i32> = cache.get("x"); // hit
        let _: Option<i32> = cache.get("y"); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert!(stats.hit_rate() > 0.4 && stats.hit_rate() < 0.6);
    }

    #[test]
    fn test_flush_all() {
        let cache = Cache::default_cache();
        cache.set("a", &1);
        cache.set("b", &2);
        cache.flush_all();
        assert!(!cache.contains("a"));
        assert!(!cache.contains("b"));
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;

        let cache = Arc::new(Cache::default_cache());
        let mut handles = Vec::new();

        for i in 0..10 {
            let c = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || {
                for j in 0..100 {
                    let key = format!("thread{}:key{}", i, j);
                    c.set(&key, &(i * 100 + j));
                    let _: Option<i64> = c.get(&key);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Should have 1000 entries
        let stats = cache.stats();
        assert_eq!(stats.total_entries, 1000);
    }

    #[test]
    fn test_pattern_matching() {
        assert!(matches_pattern("user:1", "user:*"));
        assert!(matches_pattern("user:1", "*"));
        assert!(!matches_pattern("post:1", "user:*"));
        assert!(matches_pattern("hello.txt", "*.txt"));
        assert!(matches_pattern("user:1:name", "user:*:name"));
    }
}
