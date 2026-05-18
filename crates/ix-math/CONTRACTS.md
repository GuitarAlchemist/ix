# ix-math Behavioral Contracts

> Guarantees the public API makes that the stable-surface guard (API hash) can't enforce.
> Downstream consumers (ga, tars, Demerzel, agent-blackbox, hari) may rely on these.
> Breaking any of these requires a major version bump regardless of whether the API shape changed.

## Functional contracts
- `distance::*` (euclidean, manhattan, minkowski, chebyshev, cosine_*) — returns `Err(MathError::DimensionMismatch)` on length mismatch BEFORE allocating, never panics on input shape.
- `distance::cosine_similarity` — returns `Ok(0.0)` (not `Err`) when either vector has norm < 1e-12. Callers depending on "0 means undefined" rely on this.
- `distance::minkowski` — returns `Err(MathError::InvalidParameter)` for `p < 1.0`; `p = 1` is L1, `p = 2` is L2 (verified against `euclidean`).
- `distance::cosine_distance` returns `1.0 - cosine_similarity`, i.e. in `[0, 2]`, NOT in `[0, 1]`.
- `stats::mean` / `variance` / `std_dev` / `median` / `min_max` — return `Err(MathError::EmptyInput)` on empty arrays, never `Ok(NaN)`.
- `stats::sample_variance` requires n >= 2; returns `Err(InvalidParameter)` for n < 2 (Bessel correction would divide by zero).
- `stats::variance` is POPULATION variance (divides by N), not sample (N-1). `sample_variance` is the N-1 form.
- `stats::correlation_matrix` returns 0.0 (not NaN) for any pair whose covariance denominator is < 1e-12.
- `linalg::matmul` / `matvec` — dimension check BEFORE delegating to ndarray; returns `Err(DimensionMismatch)` instead of letting ndarray panic.
- `linalg::determinant` is recursive cofactor expansion — O(n!) — intended for small (n <= ~10) square matrices. Do NOT call on large matrices in a hot path.
- `activation::sigmoid` returns finite values for any finite f64 (no overflow path); `sigmoid_derivative` recomputes sigmoid internally (not a cached pass-through).
- `activation::relu(x)` for negative `x` returns exactly `0.0`, not `-0.0`.

## Concurrency contracts
- All functions are pure (`fn`, no shared state). Safe to call from multiple threads concurrently.
- No types in this crate hold `Mutex`/`RwLock`. `BSPTree`, `KMeans`-style state structs are `Send + Sync` iff their generic parameters are.

## Failure contracts
- Errors flow through `MathError` (`thiserror`). Variants: `DimensionMismatch { expected, got }`, `EmptyInput`, `InvalidParameter(String)`, `NotSquare { rows, cols }`.
- No public function panics on user input shape. Internal `unwrap()` only on infallible ndarray operations (e.g. `mean()` after explicit emptiness check).
- `bsp::BSPTree::nearest_neighbor` returns `Option`, never panics on an empty tree — returns `None`.

## Determinism contracts
- All non-randomized functions are bit-deterministic across runs on the same platform/toolchain.
- Functions taking `seed: u64` (`random::*`) produce identical output for identical seeds (uses `StdRng` / `ChaCha8Rng`).
- `stats::correlation_matrix` iterates `(i, j)` in nested `for` loops — order is deterministic; output `Array2` is symmetric within f64 rounding.

## Memory contracts
- Distance and stats functions allocate only the output (typically scalar or `Array1` / `Array2` of declared dims). They do NOT allocate intermediate buffers proportional to input.
- `transpose` clones to a new owned `Array2`; the input view is left untouched.
- `Complex` (re-exported via `ix-signal::fft::Complex`) is `Copy`; no allocation for arithmetic ops.
