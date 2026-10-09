---
name: ix-cache
description: Embedded Redis-like cache with TTL, LRU, pub/sub, and RESP protocol
disable-model-invocation: true
---

# Cache

In-memory key-value cache with Redis-compatible features.

## When to Use
When the user needs in-process caching, TTL-based expiry, LRU eviction, pub/sub messaging, or a lightweight Redis-compatible RESP server.

## Capabilities
- **Key-Value Store** — Concurrent sharded HashMap with `set`, `get`, `delete`, `keys`
- **TTL** — Time-to-live expiry on keys
- **LRU Eviction** — Least-recently-used eviction when capacity is reached
- **Pub/Sub** — In-process publish/subscribe messaging channels
- **RESP Server** — Redis-compatible protocol server (connect with redis-cli)
- **Persistence** — Snapshot to disk, restore on startup

## Programmatic Usage
```rust
use ix_cache::{Cache, CacheConfig};

let cache = Cache::new(CacheConfig::default());
cache.set("key", &json!("value"));
let val: Option<Value> = cache.get("key");
cache.delete("key");
```

## MCP Tool Reference
Tool: `ix_cache` — Operations: `set`, `get`, `delete`, `keys`
