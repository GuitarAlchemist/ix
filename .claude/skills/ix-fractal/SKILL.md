---
name: ix-fractal
description: Fractal generation — Takagi curves, space-filling curves, Morton codes
disable-model-invocation: true
---

# Fractals

Generate fractal curves and space-filling curve data.

## When to Use
When the user needs fractal curve data, space-filling curve coordinates, or Morton Z-order encoding/decoding.

## Capabilities
- **Takagi (Blancmange) curve** — Nowhere-differentiable continuous fractal curve
- **Hilbert curve** — Space-filling curve mapping 1D → 2D, preserves locality
- **Peano curve** — Original space-filling curve with ternary structure
- **Morton encoding** — Z-order curve encode/decode for spatial hashing
- **IFS chaos game** — Sierpinski, Barnsley fern, Koch snowflake
- **L-systems** — Dragon, Sierpinski arrowhead, Koch curve

## Programmatic Usage
```rust
use ix_fractal::takagi::takagi_series;
use ix_fractal::space_filling::{hilbert_curve, peano_curve, morton_encode, morton_decode};
use ix_fractal::ifs::{ifs_iterate, sierpinski_maps};
use ix_fractal::lsystem::{dragon_curve, interpret};
```

## MCP Tool
Tool name: `ix_fractal`
Operations: `takagi`, `de_rham_1d`, `hilbert`, `peano`, `morton_encode`, `morton_decode`

### `de_rham_1d` (ix#203 — **draft proposal, not approved surface**)
Seeded, bounded de Rham (midpoint-displacement) 1-D signal generator, wrapping
`ix_fractal::de_rham::de_rham_curve_1d`.

> The operation name, its params and its response keys are a *proposal* under ix#203 and
> need explicit sign-off before merge — see the **Implementation status** block in
> `docs/plans/2026-07-20-ix-fractal-takagi-derham-exposure.md` for the full table of what
> is awaiting approval. Do not build a consumer against these names yet.

```json
{ "operation": "de_rham_1d", "depth": 8, "roughness": 0.3, "seed": 42 }
```

- Returns `{ points: [[t, value]], n_samples, depth, roughness, seed, max_depth, max_roughness }`
  with `n_samples = 2^depth + 1` and `t` spanning `[0, 1]`.
- `depth` is capped at **12** (4097 samples) and depth > 12 is a loud error — the
  callee's own cap is a *silent* 20 (1,048,577 samples), which an MCP payload must
  never materialize. The cap is tighter than the depth ≤ 16 the plan proposes for a
  (still unshipped) DuckDB table function, because this payload lands in a context
  window rather than in a scrollable warehouse table.
- `roughness` must be finite and in `[0, 1e6]`; `seed` is required. The upper bound is
  numerical, not aesthetic: midpoint displacement compounds over `depth` levels, so
  `max|value| ≈ roughness^depth`, and past ≈1.6e27 (at depth 12) samples overflow `f64` —
  which `serde_json` would emit as JSON `null`. A response is therefore either an explicit
  error or entirely finite points; it is never a success containing nulls. Every generated
  sample is re-checked for finiteness before serialization, so that holds even if the
  overflow estimate is wrong.
- Same `(depth, roughness, seed)` yields identical samples **within a build**:
  `StdRng`/`rand_distr` sampling is not stable across `rand` major versions, so do not
  freeze these values into a golden file without pinning those crate versions.
- Pair with the `ix_hurst` UDF below to measure the roughness back out — probe the
  *first differences*, not the raw values.

## DuckDB UDFs (ix-duck, `udf`/`duck` feature; ix#203 — **draft proposal, not approved surface**)

> These two UDF names and signatures are the one-way door
> `docs/plans/2026-07-20-ix-fractal-takagi-derham-exposure.md` §7 flags. They are registered
> here so they can be reviewed against running code, **not** because they are approved.
> Explicit sign-off on the names, the arities and the `ix_hurst` NULL contract is required
> before merge — see that plan's **Implementation status** block. Do not reference them from
> a notebook or a `ga/state/quality/` query yet: that is precisely what makes a rename
> breaking.

Two scalar wraps, no math reimplemented — see `crates/ix-duck/src/fractal.rs`:
- `ix_takagi(t DOUBLE, terms BIGINT) -> DOUBLE` — the Blancmange (Takagi) function,
  wraps `ix_fractal::takagi::takagi`. NULL in -> NULL out; `terms < 0` errors.
  ```sql
  SELECT ix_takagi(i / 100.0, 20) AS t FROM range(101) r(i);
  ```
- `ix_hurst(x DOUBLE[]) -> DOUBLE` — Hurst exponent (uncorrected R/S) over a
  `LIST<DOUBLE>`, wraps `ix_chaos::fractal::hurst_exponent`. Order-sensitive:
  materialize with `list(value ORDER BY i)`, never bare `list(value)`. Below
  ~256 samples the estimate is noisy, not a validated pass/fail signal; and
  feeding raw curve *values* (rather than first differences) pushes the
  estimate toward 1 and saturates the smooth/rough contrast.
  ```sql
  SELECT ix_hurst(list(value ORDER BY i)) FROM my_series;
  ```
  **Returns SQL NULL, never a number, for input R/S cannot score**: fewer than 8 samples
  (the callee answers those with a hard-coded `0.5` that reads exactly like a real
  "Brownian" estimate), an empty list, any non-finite sample (which would otherwise come
  back as a `NaN` DOUBLE), or a list containing a NULL *element* — element nullity is read
  from DuckDB's child validity mask, since the raw child buffer still holds a stale double
  at a NULL slot. A NULL *list* and a partly-NULL list are independent checks. Bad rows
  NULL only themselves; they never abort the query. Filter with `WHERE h IS NOT NULL`.

**De Rham curve generation is intentionally NOT exposed as a new DuckDB
table function in this slice.** `de_rham_curve_1d` is reachable from MCP via
`ix_fractal { operation: "de_rham_1d" }` (documented above); `de_rham_interpolate`
(the arbitrary-dimension path form) stays Rust-only. This is a deliberate,
un-self-approved scope cut, not an oversight — see
`docs/plans/2026-07-20-ix-fractal-takagi-derham-exposure.md` §7 ("Research
Insights" simplicity review) for the reasoning (no named SQL consumer yet,
output-amplification risk, UDF-name one-way door) and the sign-off this needs
before a `ix_de_rham_1d`/`ix_de_rham_path` table fn ships.
