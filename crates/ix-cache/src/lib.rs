//! Embedded in-process cache — blazing fast, zero external dependencies.
//!
//! Features:
//! - Concurrent sharded hash map (lock-free reads, sharded writes)
//! - TTL-based expiration
//! - LRU eviction with configurable max capacity
//! - Typed values via serde
//! - Pub/Sub channels for cache invalidation
//! - Optional disk persistence (snapshots)
//! - Optional RESP protocol server (feature = "resp-server")

pub mod lru;
pub mod persist;
pub mod pubsub;
#[cfg(feature = "resp-server")]
pub mod resp;
pub mod store;

pub use lru::LruPolicy;
pub use pubsub::{PubSub, Subscription};
pub use store::{Cache, CacheConfig, CacheEntry, CacheStats};
