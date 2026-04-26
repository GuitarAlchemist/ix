//! Content-addressed cache for [`crate::model::ContextBundle`]s.
//!
//! Keyed by the tuple
//! `(git_head_sha, file_hash, manifest_hash, strategy, root)` so that any
//! change to the file's content, the workspace manifest, or the git HEAD
//! invalidates cached bundles.
//!
//! # Scope (MVP)
//!
//! - Pure in-process `Mutex<HashMap>` — no shared-memory or RESP server
//! - Content hashing via `std::hash::DefaultHasher` (SipHash 1-3) — fine
//!   for cache fingerprints, not cryptographic
//! - Manual [`ContextCache::invalidate_path`] for consumer-driven eviction
//! - **No `notify` file watcher integration in MVP** — the brainstorm's
//!   pub/sub invalidation flow is deferred. Rationale: file-system events
//!   on Windows are timing-flaky in tests, and the walker functions
//!   correctly without automatic invalidation; users can call
//!   `invalidate_path` explicitly when they know a file has changed
//! - **No `ix-cache` integration in MVP** — that crate's API targets a
//!   larger concurrent-store use case; the context cache is single-
//!   process and can upgrade later

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::model::ContextBundle;
use crate::walk::WalkStrategy;

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// The fingerprint under which a [`ContextBundle`] is filed.
///
/// The tuple is wide on purpose — every component answers a distinct
/// "has the world changed?" question, and missing any one produces false
/// hits in replay scenarios. The governance-instrument contract requires
/// that cached results be bit-exact replayable, so being conservative about
/// cache keys is the right default.
///
/// # Why strategy parameters AND budget bounds are both in the key
///
/// Both affect the output bundle. `CallersTransitive { max_depth: 3 }`
/// produces a strict subset of `{ max_depth: 5 }` — serving the deeper
/// walk from the shallower cache would return an incomplete bundle and
/// break the replayability contract. Budget bounds are the same story:
/// a walk with `max_nodes: 1000` served from a `max_nodes: 100` cache
/// hit would return a truncated bundle masquerading as complete.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Git HEAD SHA of the workspace at the time the bundle was produced.
    /// Empty string when the workspace is not a git working tree.
    pub git_head_sha: String,
    /// Content hash of the root's file, from
    /// [`hash_file_content`].
    pub file_hash: u64,
    /// Content hash of the crate's `Cargo.toml` at the time of the walk.
    pub manifest_hash: u64,
    /// The strategy name (e.g. `"callers_transitive"`).
    pub strategy: String,
    /// Strategy parameters serialized into the key so different
    /// `max_depth` or `min_commits_shared` values don't false-hit.
    /// `0` for strategies where the field doesn't apply.
    pub strategy_max_depth: u8,
    pub strategy_min_commits_shared: u32,
    /// Budget bounds that affect the bundle's completeness. A walk with
    /// higher bounds cannot safely reuse a bundle produced with lower
    /// bounds (the lower-bound walk might have been truncated).
    pub budget_max_nodes: usize,
    pub budget_max_edges: usize,
    /// The stable root ID the walk started from.
    pub root: String,
    /// Workspace-relative path of the root's file. Used for targeted
    /// invalidation via [`ContextCache::invalidate_path`].
    pub root_file: String,
}

impl CacheKey {
    /// Construct a cache key from the inputs that affect a walk's output.
    pub fn new(
        git_head_sha: impl Into<String>,
        file_hash: u64,
        manifest_hash: u64,
        strategy: WalkStrategy,
        budget: &crate::walk::WalkBudget,
        root: impl Into<String>,
        root_file: impl Into<String>,
    ) -> Self {
        let (max_depth, min_commits_shared) = strategy_params(strategy);
        Self {
            git_head_sha: git_head_sha.into(),
            file_hash,
            manifest_hash,
            strategy: strategy_name(strategy),
            strategy_max_depth: max_depth,
            strategy_min_commits_shared: min_commits_shared,
            budget_max_nodes: budget.max_nodes,
            budget_max_edges: budget.max_edges,
            root: root.into(),
            root_file: root_file.into(),
        }
    }
}

fn strategy_name(strategy: WalkStrategy) -> String {
    match strategy {
        WalkStrategy::CallersTransitive { .. } => "callers_transitive".to_string(),
        WalkStrategy::CalleesTransitive { .. } => "callees_transitive".to_string(),
        WalkStrategy::ModuleSiblings => "module_siblings".to_string(),
        WalkStrategy::GitCochange { .. } => "git_cochange".to_string(),
    }
}

fn strategy_params(strategy: WalkStrategy) -> (u8, u32) {
    match strategy {
        WalkStrategy::CallersTransitive { max_depth }
        | WalkStrategy::CalleesTransitive { max_depth } => (max_depth, 0),
        WalkStrategy::ModuleSiblings => (0, 0),
        WalkStrategy::GitCochange { min_commits_shared } => (0, min_commits_shared),
    }
}

/// Hash a byte slice into a stable `u64` fingerprint.
///
/// Uses the Rust standard library's `DefaultHasher` (SipHash 1-3). Fine
/// for cache keys — we need distinct inputs to produce distinct outputs
/// with high probability, not cryptographic resistance.
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    data.hash(&mut h);
    h.finish()
}

/// Read a file's contents and hash them. Returns `0` if the file cannot
/// be read (missing, permission denied, etc.) — the hash of "file
/// unreadable" is deterministic and distinct from any real file's hash.
pub fn hash_file_content(path: impl AsRef<Path>) -> u64 {
    match std::fs::read(path.as_ref()) {
        Ok(bytes) => hash_bytes(&bytes),
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// ContextCache
// ---------------------------------------------------------------------------

/// In-process cache mapping [`CacheKey`] → [`ContextBundle`].
///
/// Thread-safe via an internal `Mutex`. The Mutex is the right tool here
/// because context cache accesses are rare and coarse (one lock per walk),
/// not hot-path.
pub struct ContextCache {
    inner: Mutex<HashMap<CacheKey, ContextBundle>>,
    stats: Mutex<CacheStats>,
}

/// Observability counters for cache hit-rate tracking.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub inserts: u64,
    pub invalidations: u64,
}

impl ContextCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
            stats: Mutex::new(CacheStats::default()),
        }
    }

    /// Insert a bundle under `key`. Any existing entry at the same key is
    /// overwritten.
    pub fn put(&self, key: CacheKey, bundle: ContextBundle) {
        let mut guard = self.inner.lock().expect("cache mutex poisoned");
        guard.insert(key, bundle);
        self.stats.lock().expect("stats mutex poisoned").inserts += 1;
    }

    /// Retrieve a bundle by key. Returns a clone so downstream consumers
    /// can mutate without affecting the cache.
    pub fn get(&self, key: &CacheKey) -> Option<ContextBundle> {
        let guard = self.inner.lock().expect("cache mutex poisoned");
        let result = guard.get(key).cloned();
        let mut stats = self.stats.lock().expect("stats mutex poisoned");
        if result.is_some() {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        result
    }

    /// Remove every cache entry whose `root_file` equals `path`.
    /// Returns the number of entries evicted.
    ///
    /// Consumers call this when they know a file has changed and want
    /// to pre-emptively drop stale bundles without waiting for a new
    /// cache miss. In MVP this is the primary invalidation mechanism;
    /// automatic `notify` file watcher integration is a documented
    /// follow-up.
    pub fn invalidate_path(&self, path: &str) -> usize {
        let mut guard = self.inner.lock().expect("cache mutex poisoned");
        let before = guard.len();
        guard.retain(|k, _| k.root_file != path);
        let evicted = before - guard.len();
        self.stats
            .lock()
            .expect("stats mutex poisoned")
            .invalidations += evicted as u64;
        evicted
    }

    /// Drop every cached bundle.
    pub fn clear(&self) {
        let mut guard = self.inner.lock().expect("cache mutex poisoned");
        let cleared = guard.len();
        guard.clear();
        self.stats
            .lock()
            .expect("stats mutex poisoned")
            .invalidations += cleared as u64;
    }

    /// Current number of cached bundles.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("cache mutex poisoned").len()
    }

    /// `true` iff the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot the hit / miss / insert counters.
    pub fn stats(&self) -> CacheStats {
        *self.stats.lock().expect("stats mutex poisoned")
    }
}

impl Default for ContextCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{WalkAction, WalkStep};

    fn sample_bundle(root: &str) -> ContextBundle {
        ContextBundle {
            root: root.to_string(),
            strategy: "module_siblings".to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
            unresolved_count: 0,
            walk_trace: vec![WalkStep {
                step: 0,
                node_id: root.to_string(),
                action: WalkAction::Start,
            }],
            truncated: false,
        }
    }

    fn key(root: &str, file: &str, fh: u64) -> CacheKey {
        CacheKey::new(
            "abcdef0",
            fh,
            0x1234,
            WalkStrategy::ModuleSiblings,
            &crate::walk::WalkBudget::default_generous(),
            root,
            file,
        )
    }

    // ── hash helpers ───────────────────────────────────────────────────

    #[test]
    fn hash_bytes_is_deterministic() {
        assert_eq!(hash_bytes(b"hello"), hash_bytes(b"hello"));
        assert_ne!(hash_bytes(b"hello"), hash_bytes(b"world"));
    }

    #[test]
    fn hash_file_content_missing_file_returns_zero() {
        assert_eq!(hash_file_content("/nonexistent/path/xyz"), 0);
    }

    #[test]
    fn hash_file_content_reads_and_hashes() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("a.rs");
        std::fs::write(&path, "fn main() {}").expect("write");
        let h1 = hash_file_content(&path);
        let h2 = hash_file_content(&path);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }

    // ── CacheKey ────────────────────────────────────────────────────────

    #[test]
    fn cache_key_with_different_file_hash_is_distinct() {
        let k1 = key("fn:foo", "a.rs", 1);
        let k2 = key("fn:foo", "a.rs", 2);
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_strategy_name_matches() {
        let k = key("fn:foo", "a.rs", 1);
        assert_eq!(k.strategy, "module_siblings");
        let k2 = CacheKey::new(
            "sha",
            0,
            0,
            WalkStrategy::GitCochange {
                min_commits_shared: 2,
            },
            &crate::walk::WalkBudget::default_generous(),
            "fn:foo",
            "a.rs",
        );
        assert_eq!(k2.strategy, "git_cochange");
        assert_eq!(k2.strategy_min_commits_shared, 2);
    }

    #[test]
    fn cache_key_distinguishes_max_depth() {
        // Two CallersTransitive walks with different max_depth must
        // produce different cache keys — otherwise a shallow cache hit
        // serves an incomplete bundle to a deeper walk.
        let budget = crate::walk::WalkBudget::default_generous();
        let k3 = CacheKey::new(
            "sha",
            0,
            0,
            WalkStrategy::CallersTransitive { max_depth: 3 },
            &budget,
            "fn:foo",
            "a.rs",
        );
        let k5 = CacheKey::new(
            "sha",
            0,
            0,
            WalkStrategy::CallersTransitive { max_depth: 5 },
            &budget,
            "fn:foo",
            "a.rs",
        );
        assert_ne!(k3, k5);
        assert_eq!(k3.strategy_max_depth, 3);
        assert_eq!(k5.strategy_max_depth, 5);
    }

    #[test]
    fn cache_key_distinguishes_budget_bounds() {
        // A walk with max_nodes=1000 cannot safely reuse a cached bundle
        // produced with max_nodes=100 — the cached bundle may be
        // silently truncated.
        let strategy = WalkStrategy::ModuleSiblings;
        let budget_small = crate::walk::WalkBudget {
            max_nodes: 100,
            max_edges: 500,
            timeout: std::time::Duration::from_secs(10),
        };
        let budget_large = crate::walk::WalkBudget {
            max_nodes: 1000,
            max_edges: 5000,
            timeout: std::time::Duration::from_secs(10),
        };
        let ks = CacheKey::new("sha", 0, 0, strategy, &budget_small, "fn:foo", "a.rs");
        let kl = CacheKey::new("sha", 0, 0, strategy, &budget_large, "fn:foo", "a.rs");
        assert_ne!(ks, kl);
        assert_eq!(ks.budget_max_nodes, 100);
        assert_eq!(kl.budget_max_nodes, 1000);
    }

    // ── ContextCache basic operations ──────────────────────────────────

    #[test]
    fn put_then_get_returns_bundle() {
        let cache = ContextCache::new();
        let k = key("fn:foo", "a.rs", 1);
        cache.put(k.clone(), sample_bundle("fn:foo"));
        assert_eq!(cache.len(), 1);
        let got = cache.get(&k).expect("hit");
        assert_eq!(got.root, "fn:foo");
    }

    #[test]
    fn get_missing_key_returns_none_and_increments_misses() {
        let cache = ContextCache::new();
        let k = key("fn:bar", "a.rs", 1);
        assert!(cache.get(&k).is_none());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn put_and_get_updates_stats() {
        let cache = ContextCache::new();
        let k = key("fn:foo", "a.rs", 1);
        cache.put(k.clone(), sample_bundle("fn:foo"));
        cache.get(&k);
        cache.get(&k);
        let stats = cache.stats();
        assert_eq!(stats.inserts, 1);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 0);
    }

    // ── Invalidation ───────────────────────────────────────────────────

    #[test]
    fn invalidate_path_removes_matching_entries_only() {
        let cache = ContextCache::new();
        cache.put(key("fn:a", "a.rs", 1), sample_bundle("fn:a"));
        cache.put(key("fn:b", "a.rs", 1), sample_bundle("fn:b"));
        cache.put(key("fn:c", "b.rs", 1), sample_bundle("fn:c"));
        assert_eq!(cache.len(), 3);

        let evicted = cache.invalidate_path("a.rs");
        assert_eq!(evicted, 2);
        assert_eq!(cache.len(), 1);

        // Remaining entry is the one from b.rs
        let remaining = cache.get(&key("fn:c", "b.rs", 1));
        assert!(remaining.is_some(), "b.rs entry should still be cached");

        let stats = cache.stats();
        assert_eq!(stats.invalidations, 2);
    }

    #[test]
    fn invalidate_path_with_no_matches_is_noop() {
        let cache = ContextCache::new();
        cache.put(key("fn:a", "a.rs", 1), sample_bundle("fn:a"));
        let evicted = cache.invalidate_path("no-such-file.rs");
        assert_eq!(evicted, 0);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn clear_empties_cache() {
        let cache = ContextCache::new();
        cache.put(key("fn:a", "a.rs", 1), sample_bundle("fn:a"));
        cache.put(key("fn:b", "b.rs", 1), sample_bundle("fn:b"));
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        let stats = cache.stats();
        assert_eq!(stats.invalidations, 2);
    }

    // ── File-change simulation: populate, edit on disk, invalidate ─────

    #[test]
    fn file_edit_simulation_evicts_via_manual_invalidate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let file_path = tmp.path().join("target.rs");
        std::fs::write(&file_path, "fn target() {}").expect("write v1");

        let cache = ContextCache::new();
        let v1_hash = hash_file_content(&file_path);
        let k1 = CacheKey::new(
            "sha",
            v1_hash,
            0,
            WalkStrategy::ModuleSiblings,
            &crate::walk::WalkBudget::default_generous(),
            "fn:target",
            "target.rs",
        );
        cache.put(k1.clone(), sample_bundle("fn:target"));

        // Simulate a file edit.
        std::fs::write(&file_path, "fn target() { let _ = 1; }").expect("write v2");
        let v2_hash = hash_file_content(&file_path);
        assert_ne!(v1_hash, v2_hash, "file content hash should change");

        // The v1 key is now stale; invalidate_path drops it.
        let evicted = cache.invalidate_path("target.rs");
        assert_eq!(evicted, 1);
        assert!(cache.get(&k1).is_none());
    }
}
