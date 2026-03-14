---
title: "feat: Add quaternions, dual quaternions, Plücker coordinates, primes, and fractal curves"
type: feat
status: completed
date: 2026-03-13
origin: docs/brainstorms/2026-03-13-tars-math-concepts-brainstorm.md
---

# feat: Add Quaternions, Dual Quaternions, Plücker Coordinates, Primes, and Fractal Curves

## Enhancement Summary

**Deepened on:** 2026-03-13
**Sections enhanced:** 8
**Research agents used:** best-practices-researcher, framework-docs-researcher, architecture-strategist, performance-oracle, pattern-recognition-specialist, code-simplicity-reviewer, security-sentinel, spec-flow-analyzer

### Key Improvements
1. PluckerLine uses `[f64; 3]` instead of `Array1<f64>` — enables Copy, eliminates heap allocation
2. Comprehensive numeric stability: Taylor expansion for exp/ln, NLERP fallback in SLERP, clamp all acos arguments
3. Sieve returns `Result` with hard cap + bit-packing; `is_prime` adds deterministic Miller-Rabin for large inputs
4. De Rham uses iterative generation (no stack overflow) with depth cap at 20
5. Added `rotate_vector`, `nlerp`, `inverse` for DualQuaternion, operator traits (Add/Sub/Neg/Mul<f64>)
6. Renamed `derham.rs` → `de_rham.rs` for naming consistency with `poincare_map.rs`

### New Considerations Discovered
- machin-chaos has NO error types — Takagi/de Rham should return plain values with sensible defaults, not `Result`
- Hamilton convention must be documented in module-level `//!` comment (vs JPL)
- Dual quaternion normalization must enforce `real · dual = 0` orthogonality constraint
- ScLERP has a degenerate pure-translation case requiring special handling
- Cross product is NOT provided by ndarray — must implement manually
- Architecture note: primes.rs is a domain mismatch for machin-math (number theory vs continuous math) but acceptable for Phase 1 scope

### YAGNI Analysis (Noted, User Override)
The simplicity reviewer recommended deferring dual_quaternion.rs and plucker.rs (zero current consumers). However, the user explicitly requested "all of them, add dual quaternions too (Include Plücker notation)." All 6 modules are retained per user intent.

---

## Overview

Phase 1 of TARS mathematical concepts: extend `machin-math` with algebraic types (quaternions, dual quaternions, Plücker coordinates) and prime utilities; extend `machin-chaos` with fractal curves (Takagi, de Rham). These are foundational types that later phases (Lie groups, TDA, Neural ODEs) build upon.

(see brainstorm: docs/brainstorms/2026-03-13-tars-math-concepts-brainstorm.md)

## Problem Statement / Motivation

MachinDeOuf lacks 3D rotation/rigid-body primitives, line geometry, number-theoretic utilities, and fractal curve generators. These are needed for:
- 3D rotation interpolation (quaternion SLERP) — robotics, animation, spatial ML
- Rigid-body transforms (dual quaternions) — 6-DOF pose representation
- Line geometry (Plücker) — collision detection, screw theory bridge
- Prime patterns — sparse hashing, number-theoretic features
- Fractal curves — non-smooth optimization landscapes, fractal noise injection

## Proposed Solution

Add 6 new modules across 2 existing crates, following established conventions.

### Extend `machin-math` (4 new modules)

#### 1. `math/quaternion.rs` — Unit Quaternion for 3D Rotation

```rust
//! Quaternion algebra for 3D rotations.
//!
//! Uses **Hamilton convention**: ijk = -1, scalar-first storage.
//! Rotation application: `q * v * q.conjugate()` rotates vector v.
//! Composition: `q1 * q2` applies q2 first, then q1 (right-to-left, matching matrix convention).

/// Threshold for near-zero norm checks (normalize, inverse, exp, ln).
const NORM_EPSILON: f64 = 1e-12;

/// SLERP falls back to NLERP when |dot| exceeds this (avoids sin(~0)/sin(~0)).
const SLERP_DOT_THRESHOLD: f64 = 1.0 - 1e-6;

/// Unit quaternion for 3D rotation (Hamilton convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self;
    pub fn identity() -> Self;                          // (1, 0, 0, 0)
    pub fn from_axis_angle(axis: &[f64; 3], angle: f64) -> Result<Self, MathError>;
    pub fn to_rotation_matrix(&self) -> Array2<f64>;    // 3×3, re-normalizes first
    pub fn rotate_vector(&self, v: &[f64; 3]) -> [f64; 3];  // q * v * q⁻¹ (O(1), no matrix)
    pub fn conjugate(&self) -> Self;
    pub fn inverse(&self) -> Result<Self, MathError>;   // conjugate / norm², fails if norm ≈ 0
    pub fn norm(&self) -> f64;
    pub fn norm_squared(&self) -> f64;
    pub fn normalize(&self) -> Result<Self, MathError>; // Err if norm < NORM_EPSILON
    pub fn dot(&self, other: &Self) -> f64;
    pub fn is_unit(&self, tolerance: f64) -> bool;
    pub fn scale(&self, s: f64) -> Self;                // scalar multiplication
    pub fn exp(&self) -> Self;                          // exponential map (Taylor near ||v||→0)
    pub fn ln(&self) -> Result<Self, MathError>;        // logarithmic map (clamp acos)
    pub fn nlerp(&self, other: &Self, t: f64) -> Quaternion; // normalized lerp
}

impl Default for Quaternion { fn default() -> Self { Self::identity() } }
impl std::ops::Mul for Quaternion { ... }               // Hamilton product
impl std::ops::Mul<f64> for Quaternion { ... }          // scalar multiplication
impl std::ops::Add for Quaternion { ... }               // component-wise
impl std::ops::Sub for Quaternion { ... }               // component-wise
impl std::ops::Neg for Quaternion { ... }               // negate all components
impl std::fmt::Display for Quaternion { ... }           // "w + xi + yj + zk"
impl From<Quaternion> for Array1<f64> { ... }           // [w, x, y, z] — infallible
impl TryFrom<Array1<f64>> for Quaternion { ... }        // fallible (checks len == 4)

/// Spherical linear interpolation. Returns None for near-antipodal quaternions.
pub fn try_slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Option<Quaternion>;

/// SLERP that falls back to NLERP for degenerate cases.
pub fn slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Quaternion;
```

### Research Insights — Quaternion

**Numeric Stability (Critical):**
- `exp()`: When `||v|| → 0`, use Taylor expansion for `sinc(x) = sin(x)/x ≈ 1 - x²/6`. Avoids 0/0 singularity.
- `ln()`: Always `clamp(-1.0, 1.0)` before `acos()` — floating-point values like `1.0000000000000002` produce NaN without clamping.
- `normalize()`: Return `Err(MathError::InvalidParameter("zero-norm quaternion"))` when norm < 1e-12.
- `inverse()`: Use `norm_squared()` to avoid unnecessary sqrt. Check `ns < NORM_EPSILON²`.
- **Normalization drift**: After 10-20 successive multiplications, re-normalize. Do NOT assume unit quaternions stay unit.

**SLERP Three-Tier Pattern (from nalgebra):**
1. `|dot| > SLERP_DOT_THRESHOLD` → fall back to NLERP (avoids sin(~0)/sin(~0))
2. `dot < 0` → negate one quaternion (shortest path / double cover)
3. Standard path → `sin((1-t)*θ)/sin(θ) * q1 + sin(t*θ)/sin(θ) * q2`, re-normalize result

**Hamilton Product**: 16 multiplies + 12 add/sub. Write as four direct expressions on raw f64 fields — the compiler will auto-vectorize. Do NOT use ndarray for this.

**`from_axis_angle`**: Validate axis norm > NORM_EPSILON. Internally normalize the axis (divide by its length) rather than requiring pre-normalized input — more robust.

**`to_rotation_matrix`**: Re-normalize quaternion before building matrix. Use `array!` macro for readable 3×3 construction. Use `2*(y+y)` style (double, then multiply) to minimize operations.

**References:**
- nalgebra UnitQuaternion: https://docs.rs/nalgebra/latest/nalgebra/geometry/type.UnitQuaternion.html
- Exponential Rotations: https://thenumb.at/Exponential-Rotations/
- CMU Exp Map: https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf

---

#### 2. `math/dual_quaternion.rs` — 6-DOF Rigid Transform

```rust
//! Dual quaternion algebra for rigid-body transforms.
//!
//! A dual quaternion `dq = q_r + ε * q_d` encodes rotation + translation
//! in 8 elements. Unit constraint: `||q_r|| = 1` AND `q_r · q_d = 0`.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualQuaternion {
    pub real: Quaternion,
    pub dual: Quaternion,
}

impl DualQuaternion {
    pub fn identity() -> Self;
    pub fn from_rotation_translation(rotation: &Quaternion, translation: &[f64; 3]) -> Result<Self, MathError>;
    pub fn to_rotation_translation(&self) -> (Quaternion, [f64; 3]);
    pub fn conjugate(&self) -> Self;
    pub fn norm(&self) -> (f64, f64);                   // (real_norm, dual_norm)
    pub fn normalize(&self) -> Result<Self, MathError>; // enforces real·dual = 0
    pub fn inverse(&self) -> Result<Self, MathError>;   // for undo/relative transforms
    pub fn transform_point(&self, point: &[f64; 3]) -> Result<[f64; 3], MathError>;
}

impl Default for DualQuaternion { fn default() -> Self { Self::identity() } }
impl std::ops::Mul for DualQuaternion { ... }
impl std::ops::Neg for DualQuaternion { ... }

/// Screw linear interpolation via screw decomposition.
pub fn sclerp(dq1: &DualQuaternion, dq2: &DualQuaternion, t: f64) -> Result<DualQuaternion, MathError>;
```

### Research Insights — Dual Quaternion

**Normalization Constraint (Critical):**
After normalizing by real-part norm, enforce orthogonality by projecting out the parallel component:
```
dual = dual - (real · dual) * real
```
Without this, the dual quaternion gains a "stretching" component and no longer represents a pure rigid transform.

**ScLERP via Screw Parameters:**
1. Compute relative: `diff = self⁻¹ * other`
2. Ensure shortest path: negate if `diff.real.w < 0`
3. Extract screw parameters: `(axis, angle, moment, pitch)`
4. **Degenerate case**: When `sin(half_angle) < ε` → pure translation. Extract translation direction from dual part instead.
5. Scale parameters by `t`, reconstruct, multiply: `result = self * powered`
6. Re-normalize result.

**`inverse()`**: For unit dual quaternions, `inverse = conjugate`. But after floating-point drift, use full inverse: divide by norm squared.

**References:**
- Kavan et al., Dual Quaternions for Rigid Transformation Blending
- Kenwright, Dual Quaternion Interpolation (arXiv:2303.13395)

---

#### 3. `math/plucker.rs` — 6D Line Representation

```rust
//! Plücker coordinates for lines in 3D space.
//!
//! Convention: direction-first `(l, m)` where `l` is unit direction,
//! `m = p × l` is the moment. Follows Clifford/Study screw theory convention.

/// Plücker coordinates for a line in 3D space.
/// Uses [f64; 3] for stack allocation and Copy semantics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PluckerLine {
    pub direction: [f64; 3],  // unit direction vector
    pub moment: [f64; 3],     // moment = point × direction
}

impl PluckerLine {
    pub fn from_two_points(p1: &[f64; 3], p2: &[f64; 3]) -> Result<Self, MathError>;
    pub fn from_point_direction(point: &[f64; 3], direction: &[f64; 3]) -> Result<Self, MathError>;
    pub fn reciprocal_product(&self, other: &PluckerLine) -> f64;
    pub fn intersects(&self, other: &PluckerLine, tolerance: f64) -> bool;
    pub fn distance_between(&self, other: &PluckerLine) -> f64; // perpendicular distance
    pub fn closest_point_to_origin(&self) -> [f64; 3];
}
```

### Research Insights — Plücker

**`[f64; 3]` instead of `Array1<f64>` (Critical Pattern Fix):**
- `Array1<f64>` does NOT implement `Copy` → `#[derive(Copy)]` fails to compile
- 48 bytes total (fits in one cache line), no heap allocation
- Consistent with Quaternion's stack-allocated design
- Provide `From<PluckerLine> for Array1<f64>` at API boundary if ndarray interop is needed

**Construction Validation:**
- `from_two_points`: Validate `||p2 - p1|| > NORM_EPSILON` (coincident points = degenerate line)
- `from_point_direction`: Validate `||direction|| > NORM_EPSILON` (zero direction = no line)
- Both normalize direction internally

**Cross Product**: ndarray does NOT provide a cross product. Implement a `cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3]` helper in this module (or in a shared `geometry_helpers` section).

**Plücker↔DualQuaternion Bridge — Deferred:**
The conversion from a line to a dual quaternion requires specifying rotation angle and translation distance (a line is not a transform). Per architecture and spec-flow reviewers, defer `to_dual_quaternion` / `from_dual_quaternion` to Phase 2 when screw theory semantics are fully defined. Keep the module focused on line geometry for now.

---

#### 4. `math/primes.rs` — Prime Utilities

```rust
//! Number-theoretic utilities: prime generation, testing, and patterns.

const MAX_SIEVE_LIMIT: u64 = 100_000_000; // 100M — ~12.5 MB with bit-packing

pub fn sieve_of_eratosthenes(limit: u64) -> Result<Vec<u64>, MathError>;
pub fn is_prime(n: u64) -> bool;
pub fn nth_prime(n: usize) -> Result<u64, MathError>;
pub fn prime_triplets(limit: u64) -> Result<Vec<(u64, u64, u64)>, MathError>;
pub fn prime_factors(n: u64) -> Vec<(u64, u32)>;  // (prime, exponent) pairs
```

### Research Insights — Primes

**Sieve Memory (Critical Security Fix):**
- `Vec<bool>` of 10^9 = ~1 GB. Use **bit-packing** (`Vec<u64>` with manual bit ops) for 8× reduction.
- Hard cap at `MAX_SIEVE_LIMIT = 100_000_000` (100M, ~12.5 MB bit-packed). Return `MathError::InvalidParameter` above.
- Never panic in library code — always `Result`.

**`is_prime` — Two-Tier Strategy:**
- `n < 1_000_000`: trial division with sqrt optimization (fast, simple)
- `n >= 1_000_000`: **deterministic Miller-Rabin** with witnesses `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` — exact for all `n < 2^64`, runs in O(k · log²n)
- Use `u128` intermediate arithmetic for modular exponentiation to avoid `u64` overflow:
  ```rust
  fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
      ((a as u128 * b as u128) % m as u128) as u64
  }
  ```

**`nth_prime`:** Cap at `n <= 10_000_000`. Use prime-counting estimate `p_n ≈ n * (ln(n) + ln(ln(n))) * 1.3` to size the sieve upfront. Return `Result`.

**`prime_memory_hash` — Removed:** No clear consumer or spec. Removed per simplicity review.

**`prime_factors` — Added:** Most common number-theory operation after `is_prime`. Trial division up to sqrt(n), return `Vec<(u64, u32)>` of (prime, exponent) pairs.

**Architecture Note:** primes.rs is a domain mismatch for machin-math (number theory vs continuous math). Acceptable for Phase 1 scope, but consider relocating to machin-probabilistic in a future refactor if the module grows.

---

### Extend `machin-chaos` (2 new modules)

**Important Pattern:** machin-chaos has NO error types. No `Result`, no `thiserror`, no dependency on machin-math. Functions handle edge cases with early returns of sensible defaults or empty collections. Follow this convention.

#### 5. `chaos/takagi.rs` — Blancmange / Takagi Curve

```rust
//! Takagi (Blancmange) curve — continuous but nowhere differentiable.
//!
//! T(t) = Σ_{k=0}^{N} dist(2^k·t, nearest_int) / 2^k
//! The standard Blancmange curve has fractal dimension 1.5.

/// Evaluate the Takagi function at point t.
/// Uses periodic extension: t is mapped to [0, 1] via t - floor(t).
/// `terms` is capped at 53 (beyond this, 2^k overflows f64 exact integer range).
pub fn takagi(t: f64, terms: usize) -> f64;

/// Sample the Takagi curve at n_points evenly spaced in [0, 1].
/// Returns Array1 of length n_points. For n_points=0, returns empty array.
pub fn takagi_series(n_points: usize, terms: usize) -> Array1<f64>;
```

### Research Insights — Takagi

**`terms` cap at 53:** Beyond `k=53`, `2^k` is not exactly representable as `f64`, causing silent precision loss. Cap silently with `terms.min(53)`.

**Periodic extension:** For `t` outside [0, 1], use `t = t - t.floor()`. This is the standard mathematical definition and prevents surprising behavior.

**Vectorization:** Structure `takagi_series` as outer loop over `k` (terms), inner loop over points. The inner operation `(scale * t).fract()` mapped through `|x| x.min(1.0 - x)` is pure arithmetic — highly SIMD-friendly, will auto-vectorize.

**`takagi_perturbation` — Deferred:** Simplicity review flagged this as speculative (no current consumer for fractal noise injection). Defer to when an optimizer or sampler actually needs it.

**Integration with existing `machin-chaos::fractal`:** The Blancmange curve has known fractal dimension 1.5. Add a test computing `box_counting_dimension_2d` on Takagi samples as a cross-module validation.

---

#### 6. `chaos/de_rham.rs` — De Rham Fractal Curves

**Renamed from `derham.rs`** to preserve the word boundary in "de Rham", consistent with `poincare_map.rs` using underscores for multi-word names.

```rust
//! De Rham fractal curves — IFS-based interpolation.
//!
//! Roughness decays 0.5× per recursion level. Seeded RNG for reproducibility.

const MAX_DEPTH: usize = 20; // 2^20 ≈ 1M points

/// Generate a de Rham fractal curve by IFS interpolation.
/// Depth is capped at 20. Returns empty vec for depth=0.
pub fn de_rham_interpolate(
    p0: &Array1<f64>, p1: &Array1<f64>,
    depth: usize, roughness: f64,
    rng: &mut impl rand::Rng,
) -> Vec<Array1<f64>>;

/// Generate a 1D de Rham fractal signal.
/// Depth is capped at 20.
pub fn de_rham_curve_1d(
    depth: usize, roughness: f64,
    rng: &mut impl rand::Rng,
) -> Array1<f64>;
```

### Research Insights — De Rham

**Stack Overflow Prevention (Critical):**
- Naive recursion at depth=30 risks stack overflow (Rust default 8MB stack, each frame carries `Array1<f64>` allocations)
- **Use iterative implementation** with explicit worklist (`Vec` as stack). Eliminates stack overflow risk entirely.
- Pre-allocate output: `Vec::with_capacity(1 << depth.min(MAX_DEPTH))`

**Depth Cap at 20:** `2^20 ≈ 1M` points is sufficient for visualization and numerical analysis. `2^30 ≈ 1B` would exhaust memory. Cap silently with `depth.min(MAX_DEPTH)` (chaos crate convention: no error types).

**Edge Cases:**
- `depth=0` → return `vec![p0.clone(), p1.clone()]` (just endpoints)
- `roughness=0.0` → straight line interpolation (no fractal structure)
- `roughness=1.0` → maximum displacement

---

## Technical Considerations

### Dependencies
- `machin-math`: no new dependencies (uses existing ndarray, thiserror)
- `machin-chaos`: no new dependencies (uses existing ndarray, rand, rand_distr). Does NOT depend on machin-math.
- Dual quaternion and Plücker modules depend on quaternion module (same crate, no circular deps)

### Error Handling

**machin-math modules** follow the crate's `Result<T, MathError>` pattern for fallible operations. Extend `MathError` with:

```rust
// In crates/machin-math/src/error.rs — no new variants needed.
// Existing InvalidParameter(String) covers all new cases:
//   "zero-norm quaternion", "coincident points", "sieve limit exceeds maximum", etc.
// Existing DimensionMismatch covers Array1 length checks.
// Existing Singular covers quaternion inverse of zero.
```

**machin-chaos modules** return plain values. No `Result`, no errors. Handle edge cases with:
- Early returns of empty/default values
- Silent capping of parameters (`terms.min(53)`, `depth.min(20)`)

### Shared Helpers

Add a `fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3]` helper (used by Plücker and quaternion's `rotate_vector`). Place in `quaternion.rs` as `pub(crate)` or in a small internal helpers section.

### Testing Strategy
- Unit tests in each module (tolerance `1e-10` per workspace convention, `1e-6` for accumulated-error operations like exp→ln round-trip)
- **Algebraic property tests:**
  - `q * q.conjugate() ≈ |q|² * identity`
  - `(q1 * q2) * q3 == q1 * (q2 * q3)` (associativity)
  - `q * q.inverse() ≈ identity` for all non-zero q
  - `slerp(q, q, t) ≈ q` for all t
  - `slerp(q1, q2, 0) ≈ q1`, `slerp(q1, q2, 1) ≈ q2`
  - `dq.transform_point(p)` then `dq.inverse().transform_point(result) ≈ p`
- **Known-value tests:** rotation matrix from 90° around Z-axis, π(10^6) = 78498 primes, T(0.5) = 0.5 for standard Blancmange
- **Round-trip tests:** axis-angle → quaternion → rotation matrix → verify determinant = 1; rotation+translation → DualQuaternion → extract back
- **Edge cases:** zero quaternion, coincident points in Plücker, `slerp(q, -q, 0.5)`, `is_prime(0)`, `is_prime(1)`, `is_prime(2)`, `nth_prime(1) = 2`
- **Cross-module test:** compute fractal dimension of Takagi curve ≈ 1.5 using `machin_chaos::fractal::box_counting_dimension_2d`
- **Implement `AbsDiffEq` for Quaternion** (approx crate already in dev-deps) for cleaner test assertions

### Performance
- Quaternion ops: O(1), 32 bytes, stack-allocated, auto-vectorizable
- DualQuaternion: O(1), 64 bytes, Copy semantics, fits in two AVX registers
- PluckerLine: O(1), 48 bytes, stack-allocated, Copy
- Sieve: O(n log log n) time, ~12.5 MB max (bit-packed, capped at 100M)
- is_prime: O(1) effective for all u64 (Miller-Rabin with fixed witness set)
- Takagi series: O(n_points × terms), auto-vectorizable inner loop
- De Rham: O(2^depth) points, iterative, capped at 2^20 ≈ 1M

## Acceptance Criteria

### `machin-math` — Quaternions (`crates/machin-math/src/quaternion.rs`)
- [x] `Quaternion` struct with `#[derive(Debug, Clone, Copy, PartialEq)]`
- [x] `new`, `identity`, `from_axis_angle` (validates axis), `to_rotation_matrix` (re-normalizes)
- [x] `rotate_vector` — direct q*v*q⁻¹ without building rotation matrix
- [x] `conjugate`, `inverse` (Result), `norm`, `norm_squared`, `normalize` (Result), `dot`, `is_unit`
- [x] `exp` (Taylor expansion near zero), `ln` (clamp acos) for continuous rotation paths
- [x] `nlerp` — normalized linear interpolation
- [x] `Mul`, `Add`, `Sub`, `Neg`, `Mul<f64>` trait implementations
- [x] `Default` (identity), `Display` ("w + xi + yj + zk")
- [x] `From<Quaternion> for Array1<f64>` (infallible, `[w,x,y,z]`), `TryFrom<Array1<f64>>` (fallible)
- [x] `try_slerp` → `Option<Quaternion>`, `slerp` with NLERP fallback
- [x] `//!` module doc specifying Hamilton convention and rotation semantics
- [x] Tests: identity, rotation, slerp interpolation, exp/ln round-trip, edge cases, algebraic properties

### `machin-math` — Dual Quaternions (`crates/machin-math/src/dual_quaternion.rs`)
- [x] `DualQuaternion` struct composed of two `Quaternion`s
- [x] `identity`, `from_rotation_translation`, `to_rotation_translation`
- [x] `conjugate`, `norm`, `normalize` (enforces `real · dual = 0`), `inverse` (Result)
- [x] `transform_point` for rigid-body transform
- [x] `Mul`, `Neg` trait implementations, `Default` (identity)
- [x] `sclerp(dq1, dq2, t)` — screw linear interpolation with pure-translation fallback
- [x] Tests: identity transform, rotation-only, translation-only, combined, inverse round-trip

### `machin-math` — Plücker Coordinates (`crates/machin-math/src/plucker.rs`)
- [x] `PluckerLine` struct with `direction: [f64; 3]`, `moment: [f64; 3]`, derives Copy
- [x] `from_two_points` (validates non-coincident), `from_point_direction` (validates non-zero direction)
- [x] `reciprocal_product`, `intersects`
- [x] `distance_between` — perpendicular distance between skew lines
- [x] `closest_point_to_origin`
- [x] Tests: construction, intersection detection, known distances, degenerate inputs

### `machin-math` — Primes (`crates/machin-math/src/primes.rs`)
- [x] `sieve_of_eratosthenes(limit)` → `Result<Vec<u64>, MathError>`, bit-packed, capped at 100M
- [x] `is_prime(n)` — trial division for small n, deterministic Miller-Rabin for large n
- [x] `nth_prime(n)` → `Result<u64, MathError>`, capped at 10M
- [x] `prime_triplets(limit)` → `Result<Vec<(u64, u64, u64)>, MathError>`
- [x] `prime_factors(n)` → `Vec<(u64, u32)>` — (prime, exponent) pairs
- [x] Tests: known primes, π(10^6) = 78498, edge cases (0, 1, 2), factorization, Miller-Rabin correctness

### `machin-chaos` — Takagi Curve (`crates/machin-chaos/src/takagi.rs`)
- [x] `takagi(t, terms)` — Blancmange evaluation with periodic extension, terms capped at 53
- [x] `takagi_series(n_points, terms)` — sampled curve, vectorized inner loop
- [x] Tests: known values (T(0)=0, T(0.5)≈0.5, T(1)=0), symmetry T(t)=T(1-t), series length
- [ ] Cross-module test: fractal dimension ≈ 1.5 via box_counting_dimension_2d (deferred — requires integration test)

### `machin-chaos` — De Rham Curves (`crates/machin-chaos/src/de_rham.rs`)
- [x] `de_rham_interpolate(p0, p1, depth, roughness, rng)` — iterative IFS, depth capped at 20
- [x] `de_rham_curve_1d(depth, roughness, rng)` — 1D fractal signal
- [x] Tests: endpoint preservation, point count = 2^depth + 1, seeded reproducibility, depth=0 edge case

### Integration
- [x] Register new modules in `machin-math/src/lib.rs` and `machin-chaos/src/lib.rs`
- [x] Add `#![forbid(unsafe_code)]` to machin-math crate root
- [x] `cargo test --workspace` passes
- [x] `cargo clippy --workspace` clean

## Implementation Order

1. **Quaternion** — foundational, no internal deps. Includes cross3 helper.
2. **Dual Quaternion** — depends on Quaternion
3. **Plücker Coordinates** — depends on Quaternion (uses cross3). DQ bridge deferred.
4. **Primes** — independent, includes Miller-Rabin
5. **Takagi Curve** — independent
6. **De Rham Curves** — independent, iterative implementation
7. **Integration** — register modules, forbid unsafe, workspace-wide test + clippy pass

## Sources & References

- **Origin brainstorm:** [docs/brainstorms/2026-03-13-tars-math-concepts-brainstorm.md](docs/brainstorms/2026-03-13-tars-math-concepts-brainstorm.md) — key decisions: struct-based quaternion, f64 only, seeded RNG
- **Existing patterns:** `crates/machin-math/src/hyperbolic.rs` (Poincaré geometry — similar math-heavy module with validation)
- **Existing patterns:** `crates/machin-chaos/src/attractors.rs` (`State3D` struct, `LorenzParams` with Default)
- **Existing patterns:** `crates/machin-math/src/distance.rs` (`check_same_len` helper, `Result<f64, MathError>`)
- **nalgebra UnitQuaternion:** https://docs.rs/nalgebra/latest/nalgebra/geometry/type.UnitQuaternion.html
- **Quaternion Exp Map (CMU):** https://www.cs.cmu.edu/~spiff/moedit99/expmap.pdf
- **Exponentially Better Rotations:** https://thenumb.at/Exponential-Rotations/
- **Dual Quaternion Blending (Kavan et al.):** https://users.cs.utah.edu/~ladislav/kavan06dual/kavan06dual.pdf
- **Dual Quaternion Interpolation (Kenwright):** https://arxiv.org/pdf/2303.13395
- **ndarray cross product discussion:** https://github.com/rust-ndarray/ndarray/discussions/1140
- **Jonathan Blow, "Understanding Slerp Then Not Using It":** http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
