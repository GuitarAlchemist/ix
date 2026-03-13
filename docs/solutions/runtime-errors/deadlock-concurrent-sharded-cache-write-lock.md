---
title: "Cache deadlock: set_with_ttl holds write lock while total_entries_approx reads all shards"
category: runtime-errors
date: 2026-03-13
severity: critical
component: crates/machin-cache/src/store.rs
tags:
  - deadlock
  - parking_lot
  - RwLock
  - sharded-cache
  - lock-ordering
---

# Cache Deadlock: Write Lock Held During Cross-Shard Read

## Problem

`CacheStore::set_with_ttl()` deadlocks under concurrent access. The process hangs indefinitely with no error output — a classic deadlock symptom.

**Symptom:** Any call to `set_with_ttl()` that triggers the capacity-check path blocks forever when another thread holds a read lock on any shard.

**Error:** None — the process simply hangs. No panic, no timeout, no log output.

## Root Cause

**Lock ordering violation** with non-reentrant `parking_lot::RwLock`.

The call chain:

```
set_with_ttl()
  → acquires WRITE lock on shard N
  → calls total_entries_approx()
    → tries to acquire READ lock on ALL shards (0..num_shards)
    → blocks on shard N (already write-locked by the same thread)
    → DEADLOCK
```

`parking_lot::RwLock` is non-reentrant — a thread holding a write lock on shard N cannot acquire a read lock on shard N, even from the same thread. The write lock blocks the read lock, and the read lock waits for the write lock to release. Neither can proceed.

**Before (deadlocking code):**

```rust
pub fn set_with_ttl(&self, key: &str, value: CacheValue, ttl: Option<Duration>) {
    let shard_idx = self.shard_index(key);
    let mut shard = self.shards[shard_idx].write(); // WRITE lock on shard N

    // ... insert the entry ...

    // Capacity check — DEADLOCK: reads ALL shards while holding write lock on N
    if self.max_capacity > 0 && self.total_entries_approx() > self.max_capacity {
        self.evict_lru(&mut shard);
    }
}

fn total_entries_approx(&self) -> usize {
    self.shards.iter()
        .map(|s| s.read().len()) // Tries to READ lock every shard, including N
        .sum()
}
```

## Solution

Move the capacity check **before** acquiring the write lock. This ensures `total_entries_approx()` runs without any locks held, avoiding the nested lock violation.

**After (fixed code):**

```rust
pub fn set_with_ttl(&self, key: &str, value: CacheValue, ttl: Option<Duration>) {
    let shard_idx = self.shard_index(key);

    // Capacity check BEFORE acquiring the write lock — no locks held here
    let needs_eviction = self.max_capacity > 0
        && self.total_entries_approx() >= self.max_capacity;

    let mut shard = self.shards[shard_idx].write(); // WRITE lock on shard N

    // ... insert the entry ...

    if needs_eviction {
        self.evict_lru(&mut shard);
    }
}
```

**TOCTOU note:** This introduces a time-of-check-to-time-of-use gap — between the capacity check and the write lock, another thread could insert entries. This is acceptable for a cache:
- Eviction is best-effort, not a correctness invariant
- Over-capacity by a few entries is harmless
- The alternative (deadlock) is catastrophic

## Investigation Steps

1. **Identified the hang:** `cargo test` hung indefinitely on cache tests with no output
2. **Narrowed to `set_with_ttl`:** Bisected to the capacity-check code path
3. **Traced lock acquisition:** Found `total_entries_approx()` iterates all shards while a write lock is held
4. **Confirmed `parking_lot` non-reentrancy:** `parking_lot::RwLock` does not support reentrant locking — write lock blocks same-thread read lock
5. **Applied fix:** Moved capacity check before write lock acquisition
6. **Verified:** All 348 tests pass, no hangs

## Prevention

### Rules

1. **Never call cross-shard methods while holding a shard lock.** Any method that iterates `self.shards` must be called with zero locks held.
2. **Treat `parking_lot::RwLock` as strictly non-reentrant.** Do not assume a thread holding a write lock can acquire any other lock on the same resource.
3. **Prefer lock-free approximate counts.** Use `AtomicUsize` for entry counts instead of iterating shards under locks.
4. **Minimize lock scope.** Acquire locks as late as possible, release as early as possible. Never call methods with unknown lock behavior while holding a lock.

### Code Review Checklist

- [ ] Does the function acquire a lock and then call another method?
- [ ] Does that called method acquire its own locks?
- [ ] Could any of those locks overlap with the already-held lock?
- [ ] Is the lock scope minimized (acquired late, released early)?
- [ ] Are cross-shard iterations done without holding any shard lock?
- [ ] Are approximate counts acceptable vs. exact counts under locks?
- [ ] Has concurrent access been tested with multiple threads?

### Testing Strategy

- **Timeout-based deadlock tests:** Wrap concurrent cache operations in `std::thread::spawn` with a timeout — if the test doesn't complete in 5 seconds, it's likely deadlocked
- **parking_lot `deadlock_detection` feature:** Enable `parking_lot`'s built-in deadlock detector in test builds for automatic detection
- **Stress tests:** Run many threads doing concurrent `set`/`get`/`delete` operations to surface lock ordering issues

### Rust-Specific Lock Guidance

| Pattern | Safe? | Notes |
|---------|-------|-------|
| Read lock A → Read lock B | Usually | Unless A and B are the same lock (reentrant read is OK for parking_lot but risky pattern) |
| Write lock A → Read lock A | **DEADLOCK** | parking_lot is non-reentrant |
| Write lock A → Read lock B | Risky | Safe only if no other thread does Write B → Read A |
| Write lock A → method call → ??? | **Audit** | Must trace all transitive lock acquisitions |

## Additional Risk Areas

The `get()` method in `store.rs` has a similar pattern worth auditing — it acquires a shard read lock and then accesses stats, which may involve additional locking. While read-read nesting is less dangerous, it should be verified for consistency.

## Related

- `crates/machin-cache/src/store.rs` — Primary fix location
- `crates/machin-cache/src/pubsub.rs` — Uses similar shard locking patterns
- `crates/machin-pipeline/src/executor.rs` — Uses `parking_lot::RwLock` for pipeline state
- [parking_lot documentation](https://docs.rs/parking_lot) — Non-reentrancy is documented but easy to miss
