//! Determinism-aware result cache.
//!
//! Wraps `ix_cache::Cache` keyed by `blake3(serde_json(config) ++ salt)`.
//! Targets opt in by returning `Some(salt)` from `Experiment::cache_salt`;
//! returning `None` disables caching entirely. The salt allows a target
//! to invalidate stale entries without bumping the cache crate.
//!
//! ## Determinism contract
//!
//! - Adapters that include wall-clock or external state in their eval
//!   MUST return `cache_salt = None` (or salt with the external state's
//!   hash so changes invalidate).
//! - `Config` types MUST serialize deterministically. No `HashMap` —
//!   use `BTreeMap` if ordered map needed. Plain structs already
//!   serialize in declaration order.
//!
//! ## TTL
//!
//! 24-hour default, configurable via [`CacheBridge::with_ttl`]. TTL is
//! the secondary safety net; the primary safety is the salt.

use std::sync::OnceLock;
use std::time::Duration;

use serde::{de::DeserializeOwned, Serialize};

use ix_cache::{Cache, CacheConfig};

use crate::error::AutoresearchError;

const DEFAULT_CACHE_TTL: Duration = Duration::from_secs(60 * 60 * 24); // 24 h

/// Process-global cache shared by all `CacheBridge` instances.
fn global_cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Cache::new(CacheConfig::default()))
}

/// Adapter wrapping `ix_cache::Cache` with autoresearch-specific keying.
#[derive(Debug, Clone)]
pub struct CacheBridge {
    salt: Option<String>,
    ttl: Option<Duration>,
}

impl CacheBridge {
    /// `salt = None` disables caching (every `get` returns None and
    /// `set` is a no-op). `salt = Some(s)` opts the target into caching.
    pub fn new(salt: Option<String>) -> Self {
        Self {
            salt,
            ttl: Some(DEFAULT_CACHE_TTL),
        }
    }

    pub fn with_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.ttl = ttl;
        self
    }

    /// `true` if caching is active for this bridge.
    pub fn is_enabled(&self) -> bool {
        self.salt.is_some()
    }

    /// Compute the cache key for a config. Returns `None` when caching
    /// is disabled. Hash is `blake3(serde_json(config) || salt)`.
    pub fn key_for<C: Serialize>(&self, config: &C) -> Result<Option<String>, AutoresearchError> {
        let Some(salt) = &self.salt else { return Ok(None) };
        let mut buf = serde_json::to_vec(config)?;
        buf.extend_from_slice(salt.as_bytes());
        let hash = blake3::hash(&buf);
        Ok(Some(format!("autoresearch:{}", hash.to_hex())))
    }

    /// Look up a cached score. Returns `None` if caching is disabled or
    /// the key is absent.
    pub fn get<C, S>(&self, config: &C) -> Result<Option<S>, AutoresearchError>
    where
        C: Serialize,
        S: DeserializeOwned,
    {
        let Some(key) = self.key_for(config)? else { return Ok(None) };
        Ok(global_cache().get::<S>(&key))
    }

    /// Store a score. No-op if caching is disabled.
    pub fn set<C, S>(&self, config: &C, score: &S) -> Result<(), AutoresearchError>
    where
        C: Serialize,
        S: Serialize,
    {
        let Some(key) = self.key_for(config)? else { return Ok(()) };
        global_cache().set_with_ttl(&key, score, self.ttl);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct Cfg(f64, f64);
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct Score(f64);

    #[test]
    fn disabled_bridge_never_caches() {
        let b = CacheBridge::new(None);
        assert!(!b.is_enabled());
        let key = b.key_for(&Cfg(1.0, 2.0)).unwrap();
        assert!(key.is_none());
        b.set(&Cfg(1.0, 2.0), &Score(0.5)).unwrap();
        let got: Option<Score> = b.get(&Cfg(1.0, 2.0)).unwrap();
        assert!(got.is_none(), "disabled cache must not return values");
    }

    #[test]
    fn enabled_bridge_round_trips_score() {
        let b = CacheBridge::new(Some("test-salt-roundtrip-unique".to_string()));
        let cfg = Cfg(1.5, 2.5);
        let score = Score(0.42);
        b.set(&cfg, &score).unwrap();
        let got: Option<Score> = b.get(&cfg).unwrap();
        assert_eq!(got, Some(score));
    }

    #[test]
    fn different_salts_produce_different_keys() {
        let cfg = Cfg(1.0, 2.0);
        let a = CacheBridge::new(Some("v1".to_string()))
            .key_for(&cfg)
            .unwrap()
            .unwrap();
        let b = CacheBridge::new(Some("v2".to_string()))
            .key_for(&cfg)
            .unwrap()
            .unwrap();
        assert_ne!(a, b, "salt change must invalidate keys");
    }

    #[test]
    fn same_config_produces_stable_key() {
        let cfg = Cfg(1.0, 2.0);
        let b = CacheBridge::new(Some("stable-test".to_string()));
        let k1 = b.key_for(&cfg).unwrap().unwrap();
        let k2 = b.key_for(&cfg).unwrap().unwrap();
        assert_eq!(k1, k2);
    }
}
