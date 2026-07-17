---
title: "feat: Expose ix-fractal Takagi and de Rham primitives through DuckDB and skills"
type: plan
status: proposed
date: 2026-07-10
issue_meta:
  level: task
  parent: "GuitarAlchemist/ix#202"
  area: research
  priority: P1
  complexity: M
  risk: medium
---

# Design & Exposure Plan: Exposing Takagi and de Rham Primitives

This document outlines the design and exposure plan to bring existing `ix-fractal` primitives (specifically Takagi/Blancmange curves and de Rham curves) to analyst-friendly surfaces in the IX ecosystem.

These mathematical curves were identified in TARS V1 as highly valuable for:
- **Multi-scale signals**: Simulating multi-frequency noise or signal structures.
- **Roughness modeling**: Generating varying degrees of fractal roughness for synthetic datasets or perturbation scenarios.
- **Path interpolation**: Non-linear, fractal interpolation between coordinate endpoints.
- **Evolutionary perturbations**: Smooth yet continuous nowhere-differentiable landscapes for optimization benchmarks.

---

## 1. Exposure & Architecture Plan

Our strategy is to expose the mature, stable implementations already present in `crates/ix-fractal` without any algorithmic reimplementation. We will surface them via two major integration boundaries:

1. **In-Process DuckDB Analyst's Bench (`ix-duck`)**
   - Direct SQL querying allows rapid exploration, data joining, and inline dataset generation.
   - We will expose scalar UDFs for point evaluation and table functions for array/series generation.
   - Table functions return `TABLE(row BIGINT, val DOUBLE)` to integrate seamlessly with the DuckDB engine's streaming paradigm.

2. **Claude Code ML Skills (`ix-skill` / `ix-agent`)**
   - High-level, action-oriented, JSON-in/JSON-out capability tools exposed to the LLM agent.
   - Ideal for multi-step ML pipelines, path generation, and data-bound roughness profiling.

### Guardrails and Parameter Bounding

To protect the sandbox and local-first execution environments from out-of-memory or stack-overflow conditions, all exposed boundaries will strictly enforce the following safety caps:
- **Takagi Terms Cap**: Capped at `53` terms. Beyond this, $2^k$ is not exactly representable as an `f64`, leading to silent precision loss.
- **De Rham Depth Cap**: Capped at `20` levels ($2^{20} + 1 \approx 1,048,577$ points). This fits comfortably within standard memory footprints (~16MB raw array).
- **De Rham Path Dimensionality Cap**: Interpolation points are restricted to a maximum of `10` dimensions to avoid processing arbitrarily large vectors.
- **Output Sample Cap**: Table-generating UDFs and skills will limit the maximum returned points to `1,000,000` samples.

---

## 2. DuckDB UDF Candidate List

The proposed DuckDB functions will reside inside `crates/ix-duck/src/udf.rs` and `crates/ix-duck/src/tablefn.rs`, registering themselves automatically in `register_all(&conn)`.

### 1. `ix_takagi` (Scalar UDF)
Evaluates the continuous Blancmange function at a single parameter point `t`.
- **SQL Signature**: `ix_takagi(t DOUBLE, terms BIGINT) -> DOUBLE`
- **Underlying Rust**: `ix_fractal::takagi::takagi(t, terms)`
- **Behavior**: Periodically extends `t` to `[0, 1]` via `t - floor(t)`. Caps `terms` at `53` silently.

### 2. `ix_takagi_series` (Table Function)
Samples the Takagi curve at `n_points` evenly spaced in `[0, 1]`.
- **SQL Signature**: `ix_takagi_series(n_points BIGINT, terms BIGINT) -> TABLE(row BIGINT, val DOUBLE)`
- **Underlying Rust**: `ix_fractal::takagi::takagi_series(n_points, terms)`
- **Behavior**: Returns a table of points. `n_points` is capped at `1,000,000` to prevent memory exhaustion.

### 3. `ix_de_rham_curve_1d` (Table Function)
Generates a 1D de Rham fractal curve (midpoint displacement) on the interval `[0.0, 1.0]`.
- **SQL Signature**: `ix_de_rham_curve_1d(depth BIGINT, roughness DOUBLE, seed BIGINT) -> TABLE(row BIGINT, val DOUBLE)`
- **Underlying Rust**: `ix_fractal::de_rham::de_rham_curve_1d(depth, roughness, &mut rng)`
- **Behavior**: Initializes a seeded generator with `StdRng::seed_from_u64(seed)` to guarantee mathematical reproducibility. Caps `depth` at `20`.

### 4. `ix_de_rham_path_json` (Scalar UDF)
Generates a multi-dimensional de Rham fractal path between two JSON-specified coordinate points `p0` and `p1`.
- **SQL Signature**: `ix_de_rham_path_json(p0_json VARCHAR, p1_json VARCHAR, depth BIGINT, roughness DOUBLE, seed BIGINT) -> VARCHAR` (JSON)
- **Underlying Rust**: `ix_fractal::de_rham::de_rham_interpolate(&p0, &p1, depth, roughness, &mut rng)`
- **Behavior**: Takes stringified JSON arrays like `[0.0, 0.0]` and `[1.0, 1.0]`. Runs iterative midpoint displacement and returns a stringified JSON 2D array of interpolated path coordinates (e.g. `[[0.0,0.0], [0.3,0.4], ..., [1.0,1.0]]`).

---

## 3. Skill Candidate List

These skills will be registered under the `fractal` domain in `crates/ix-agent/src/skills/` (using the `#[ix_skill]` macro).

### 1. `fractal_signal_generate`
Generates a bounded 1D fractal noise or signal sequence for time series injection or roughness exploration.

#### Input Schema
```json
{
  "type": "object",
  "properties": {
    "method": {
      "type": "string",
      "enum": ["takagi", "de_rham"]
    },
    "length": {
      "type": "integer",
      "minimum": 2,
      "maximum": 100000,
      "description": "Number of samples to generate"
    },
    "roughness": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 2.0,
      "description": "Fractal roughness scaling (applies to de_rham / terms mapping)"
    },
    "seed": {
      "type": "integer",
      "default": 42,
      "description": "Seeded RNG for reproducible signals"
    }
  },
  "required": ["method", "length"]
}
```

#### Output Schema
```json
{
  "type": "object",
  "properties": {
    "signal": {
      "type": "array",
      "items": { "type": "number" },
      "description": "Generated 1D signal values"
    },
    "method": { "type": "string" },
    "length": { "type": "integer" }
  }
}
```

---

### 2. `fractal_path_interpolate`
Performs midpoint-displacement fractal interpolation between multi-dimensional endpoints.

#### Input Schema
```json
{
  "type": "object",
  "properties": {
    "p0": {
      "type": "array",
      "items": { "type": "number" },
      "description": "Starting multi-dimensional coordinate vector"
    },
    "p1": {
      "type": "array",
      "items": { "type": "number" },
      "description": "Ending multi-dimensional coordinate vector"
    },
    "depth": {
      "type": "integer",
      "minimum": 0,
      "maximum": 12,
      "default": 5,
      "description": "Recursion depth (yields 2^depth + 1 points)"
    },
    "roughness": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.3,
      "description": "Midpoint displacement scaling coefficient"
    },
    "seed": {
      "type": "integer",
      "default": 42
    }
  },
  "required": ["p0", "p1"]
}
```

#### Output Schema
```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "number" }
      },
      "description": "Array of interpolated coordinates"
    },
    "num_points": { "type": "integer" },
    "dimension": { "type": "integer" }
  }
}
```

---

### 3. `fractal_roughness_probe`
Measures the estimated fractal dimension or roughness of a given sequence using box-counting or Hurst exponent estimation.

#### Input Schema
```json
{
  "type": "object",
  "properties": {
    "signal": {
      "type": "array",
      "items": { "type": "number" },
      "description": "1D sequence or 2D coordinate pairs to analyze"
    },
    "metric": {
      "type": "string",
      "enum": ["hurst", "box_counting"],
      "default": "hurst"
    }
  },
  "required": ["signal"]
}
```

#### Output Schema
```json
{
  "type": "object",
  "properties": {
    "roughness_score": {
      "type": "number",
      "description": "Hurst exponent (H in [0,1]) or box-counting dimension (D in [1,2])"
    },
    "metric": { "type": "string" },
    "interpreted_type": {
      "type": "string",
      "description": "Interpretation of the signal (e.g. 'trending', 'mean-reverting', 'anti-persistent')"
    }
  }
}
```

---

## 4. Smoke Test & Validation Plan

### Part A: DuckDB UDF Verification

The following SQL test cases should be evaluated in a test file (e.g., `crates/ix-duck/tests/fractal_udf_tests.rs`) using an in-memory DuckDB connection:

1. **Takagi Point Check**:
   ```sql
   SELECT ix_takagi(0.0, 20), ix_takagi(0.5, 20), ix_takagi(1.0, 20)
   ```
   - *Expected*: `0.0`, `0.5`, `0.0` (endpoints are 0.0, and standard Takagi reaches maximum at 0.5).

2. **Takagi Symmetry Check**:
   ```sql
   SELECT abs(ix_takagi(0.37, 20) - ix_takagi(0.63, 20)) < 1e-9
   ```
   - *Expected*: `true` (since $T(t) = T(1-t)$).

3. **Takagi Series Count**:
   ```sql
   SELECT count(*) FROM ix_takagi_series(101, 10)
   ```
   - *Expected*: `101` rows returned.

4. **De Rham Curve length & seeding**:
   ```sql
   SELECT count(*), avg(val) FROM ix_de_rham_curve_1d(6, 0.3, 42)
   ```
   - *Expected*: `65` rows ($2^6 + 1$). The average `val` must be identical across multiple query executions with seed `42` (reproducibility test).

5. **De Rham Zero Roughness Check**:
   ```sql
   SELECT count(distinct val) FROM ix_de_rham_curve_1d(4, 0.0, 42)
   ```
   - *Expected*: `17` distinct values representing a perfect straight line from `0.0` to `1.0`.

6. **De Rham Path Endpoint Preservation**:
   - Query: `SELECT * FROM ix_de_rham_path_json('[0.0, 0.0]', '[10.0, 20.0]', 4, 0.3, 123)`
   - *Expected*: Stringified JSON containing 17 points, where the first element is `[0.0, 0.0]` and the last element is `[10.0, 20.0]`.

---

### Part B: Skill Verification

These tests will run in `crates/ix-agent/tests/skill_fractal_tests.rs`:

1. **`fractal_signal_generate` (Takagi)**:
   - Call with: `{"method": "takagi", "length": 51, "terms": 20}`
   - Validate that `"signal"` has length 51, first and last element are 0.0, and the peak is at index 25 (value 0.5).

2. **`fractal_path_interpolate` (de Rham)**:
   - Call with: `{"p0": [0.0, 0.0], "p1": [1.0, 2.0], "depth": 5, "roughness": 0.2, "seed": 99}`
   - Validate that output `"num_points"` is 33, `"dimension"` is 2, first point is exactly `[0.0, 0.0]`, and last point is `[1.0, 2.0]`.

3. **`fractal_roughness_probe` (Hurst)**:
   - Create a generated trending series, call `fractal_roughness_probe` with `{"signal": ..., "metric": "hurst"}`.
   - Verify that the resulting `roughness_score` is greater than `0.5` (persistent trend).

---

## 5. Cost Notes

- **Execution Tier**: Free-local tier.
- **CPU Resource Bounds**:
  - `ix_takagi_series`: $O(\text{n\_points} \times \text{terms})$ operations. Highly SIMD-vectorized in ndarray. For the max cap ($10^6$ points, $53$ terms), execution takes $< 15\text{ms}$ on a single core.
  - `de_rham_interpolate`: $O(2^{\text{depth}} \times \text{dimension})$ operations. For max depth $20$, we perform $1,048,577$ vector operations, taking $< 50\text{ms}$ in Rust.
- **Memory Footprint**:
  - Max heap usage at extreme cap (depth 20) is limited to $\approx 32\text{MB}$ for array storage, which is perfectly safe for local sandbox environments.

---

## 6. Follow-up Implementation Issues

Once this plan is approved, developers can pick up the following tickets:

### Issue 1: Expose Takagi & de Rham via `ix-duck` UDFs
**Area**: Research, DuckDB
**Complexity**: S
**Tasks**:
- Implement `ix_takagi` in `crates/ix-duck/src/udf.rs`.
- Implement `ix_takagi_series`, `ix_de_rham_curve_1d`, and `ix_de_rham_path_json` in `crates/ix-duck/src/tablefn.rs`.
- Register the new functions in `crates/ix-duck/src/udf::register_all`.
- Add unit and integration tests verifying performance, capping, and mathematical reproducibility.

### Issue 2: Register Capability Skills in `ix-agent`
**Area**: AI Infrastructure
**Complexity**: S
**Tasks**:
- Register `fractal_signal_generate`, `fractal_path_interpolate`, and `fractal_roughness_probe` as new skills in `crates/ix-agent/src/skills/`.
- Wire them to invoke the appropriate `ix_fractal` and `ix_chaos::fractal` (for box dimension / Hurst exponent) modules.
- Complete parity testing so they are discovered by the capability registry.

### Issue 3: Add Walkthrough and Analyst Guides
**Area**: Documentation
**Complexity**: XS
**Tasks**:
- Write a short guide/example (e.g. under `docs/duck/`) on using these new SQL functions to generate rough surfaces and sample them into tables.
