# ix-probabilistic Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `BloomFilter::contains` returns `true` for "possibly in set" and `false` for "definitely not in set". FALSE POSITIVES are expected; FALSE NEGATIVES are a contract violation.
- `BloomFilter::new(capacity, fp_rate)` sizes `bits` and `num_hashes` for the GIVEN capacity. Inserting more than `capacity` items inflates fp_rate beyond `fp_rate` silently — no warning, no error.
- `BloomFilter::estimated_fp_rate()` is computed from observed fill ratio, not from the insert count. Two filters at the same fill report the same rate even if one has more inserts.
- `BloomFilter::len()` is the count of `insert` calls, NOT unique items. Inserting the same item twice increments `count`.
- `CountMinSketch::estimate` returns a value GREATER THAN OR EQUAL TO the true count — never underestimates (one-sided error).
- `CountMinSketch::with_error(epsilon, delta)` ceils width to `e/epsilon` and depth to `ln(1/delta)`; the actual error guarantee holds with probability `>= 1 - delta`. Choosing `delta = 0` would compute `ln(infinity)` — guarded by `max(1)`.
- `HyperLogLog::new(precision)` CLAMPS precision to `[4, 18]` silently. Standard error: `1.04 / sqrt(2^p)`. p=14 gives ~0.81%.
- `HyperLogLog::count()` returns `f64` (estimate), not `u64`. May be non-integer. For low cardinalities (< ~2.5 * m), small-range correction is applied; for high cardinalities (> 2^32 / 30), large-range correction is applied.
- `CuckooFilter::insert` returns `false` when the filter is FULL (all kick attempts exhausted) — NOT when the item is already present. Distinguishing "already in" vs "rejected" requires a prior `contains` check.
- `CuckooFilter::delete` returns `true` iff a matching fingerprint was found and removed. Because fingerprints are 16-bit, deleting an item never inserted may by chance succeed (collision) and corrupt another entry.
- `Bloom::contains` of unhashable types is not possible by signature; `T: Hash` is the only constraint — `Hash` implementations MUST be consistent between insert and contains.

## Concurrency contracts
- All four structures are `Send + Sync` but require `&mut self` for `insert` / `add` / `delete`. Concurrent inserts require external locking.
- `contains` / `estimate` / `count` are `&self` and safe to share across threads.
- `DefaultHasher` is used internally — hash output is NOT guaranteed stable across Rust versions. Persisting filter state across toolchain upgrades may invalidate `contains` results.

## Failure contracts
- No `Result` return anywhere. `CuckooFilter::insert` returns `bool` for failure (full). `BloomFilter::insert` always succeeds (returns `()`).
- `BloomFilter::with_params(size, num_hashes)` with `size = 0` panics on first insert (index out of bounds). No constructor-time validation.
- `CountMinSketch::new(0, 0)` is silently bumped to `(1, 1)` in `with_error`; via `new` directly it allows zero and panics on first add.

## Determinism contracts
- All four structures are deterministic given identical insertion order and identical `Hash` impls. No internal RNG on Bloom/Count-Min/HLL.
- `CuckooFilter::insert` uses internal RNG (`rand::random()`) for victim selection during kicks — NOT deterministic. This is the only non-deterministic path; expect different bucket layouts across runs even with identical input order.
- `HyperLogLog::merge` is commutative and associative — order-independent for union estimation.

## Memory contracts
- `BloomFilter` stores `Vec<bool>` (not bit-packed) — 1 byte per bit. For tight memory budgets, callers should consider an external bit-vec wrapper.
- `HyperLogLog` memory is `2^precision` bytes (one `u8` per register). p=14 = 16KB.
- `CountMinSketch` memory is `width * depth * 8 bytes` (u64 counters).
- `CuckooFilter` memory is `num_buckets * BUCKET_SIZE * 2 bytes` plus `Vec` overhead per bucket. `num_buckets` is rounded UP to next power of two.
