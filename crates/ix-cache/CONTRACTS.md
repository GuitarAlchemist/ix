# ix-cache Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `Cache::set` / `set_with_ttl` PANICS on serialization failure (`expect("Serialization failed")`). Non-Serialize-safe types must be filtered at the caller.
- `Cache::get<V>` returns `None` if (a) key absent, (b) entry expired, OR (c) deserialization into `V` fails. Callers cannot distinguish "missing" from "wrong type" via `Option`.
- TTL = `None` means "never expires" — `is_expired()` short-circuits to `false`. Default config has `default_ttl = None`.
- `Cache::set` overwrites without a "set if absent" semantic. Use `incr`/`incr_by` for atomic counter semantics (single shard lock).
- `CacheStats::hit_rate()` returns 0.0 (not NaN) when `hits + misses == 0`. Forward-compat contract for dashboards.
- `Cache::keys(pattern)` matches GLOB style (`*`, `?`), NOT regex. Iteration order is NOT sorted — depends on shard count and `HashMap` iteration.
- `Cache::flush_all` is atomic per-shard but NOT atomic across shards. A reader during flush may see a partial state.
- `incr` / `incr_by` / `decr` on a key holding a non-integer value return the new value as if the prior was 0; they do NOT error. (Redis-compatible semantics: missing key = 0.)
- `delete` returns `true` iff the key was present at the moment of delete. `expire` returns `true` iff the key exists at call time.
- LRU eviction triggers ONLY when `max_capacity > 0` AND total entries would exceed it after insert. `max_capacity = 0` is "unlimited" (default = 100_000).
- TTL expiration is LAZY by default — entries past `expires_at` remain in memory until next access or `evict_expired()` call. Stats `total_bytes` may include expired entries.

## Concurrency contracts
- `Cache` is `Send + Sync`. Sharded `RwLock`s (default 16 shards) — concurrent readers in different shards do not block each other.
- Writes within a single shard serialize. Hot keys hashing to the same shard contend.
- `pubsub()` returns `&PubSub`; subscriptions are independent of cache locks — publishing during cache mutation is supported but the publish itself happens AFTER the cache mutation visibility window.
- `stats()` returns a CLONE of `CacheStats` under a read lock; not a live view.

## Failure contracts
- No `Result` return on the public surface. Serialization failures PANIC (treat as a contract violation by the caller).
- `Cache::new(config)` with `num_shards = 0` is silently bumped to 1 (`max(1)`).
- Persist/restore (`persist::*`) does return `io::Result`; that's the only place I/O failures surface.

## Determinism contracts
- Shard assignment uses `DefaultHasher` (FNV/SipHash) — NOT deterministic across Rust versions (DefaultHasher implementation may change). Cross-version persisted snapshots may rebalance shards on load.
- TTL is `Instant::now() + duration` — wall-clock-free, but means snapshot/restore must adjust TTLs based on `ttl_ms` rather than `expires_at` (see `persist::SnapshotEntry`).
- `keys(pattern)` order is NON-deterministic across runs even with identical insert order (HashMap iteration).

## Memory contracts
- Each entry stores serialized JSON bytes (`Vec<u8>`), not the original `V`. Memory cost = `key.len() + serde_json::to_vec(value).len() + ~64B entry overhead`.
- LRU eviction triggers BEFORE insert when capacity would be exceeded; it does not "burst" then evict.
- `flush_all` does not shrink shard `HashMap` capacity — memory is reclaimed on drop, not on flush. Long-lived caches with churn may exhibit elevated steady-state memory.
- The `resp-server` feature pulls in a TCP listener; with the feature off, the cache is in-process only with zero network surface.
