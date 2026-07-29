---
title: Exposing ix-fractal Takagi + de Rham through DuckDB, skills, and MCP
type: feat
status: draft
date: 2026-07-20
issue: GuitarAlchemist/ix#203 (parent story #202)
reversibility: two-way (new UDF/skill surfaces over an `experimental`-tier crate) — one one-way sub-decision flagged below
revisit-trigger: first real consumer of a fractal UDF outside a smoke test, OR ix-fractal promoted out of `experimental` in crate-maturity.toml
---

# Exposing ix-fractal Takagi + de Rham

Design-only response to ix#203. No Takagi or de Rham math is reimplemented here; every
proposed surface is a thin wrap over the existing `crates/ix-fractal` functions.

## 0. What is ALREADY exposed (checked, not assumed)

This matters because the issue's candidate list partially duplicates live surface.

| Surface | Takagi | de Rham |
|---|---|---|
| MCP tool `ix_fractal` (`crates/ix-agent/src/handlers.rs:1627`) | **YES** — `operation:"takagi"` with `n_points`+`terms`, returns `{points:[[t,y]],…}` via `takagi_series` | **NO** |
| Claude skill `.claude/skills/ix-fractal/SKILL.md` | listed as a capability | **NO** — de Rham is not mentioned at all |
| DuckDB (`crates/ix-duck/`) | **NO** — `ix-fractal` is not even a dependency in `crates/ix-duck/Cargo.toml` | **NO** |
| CLI `ix run <skill>` (`ix-registry`) | **NO** — no fractal skill registered | **NO** |
| `ix-demo` egui app (`crates/ix-demo/src/demos/fractal.rs`) | yes (interactive only) | **NO** |

Also relevant and **unexposed anywhere**: `ix_chaos::fractal::hurst_exponent(data: &[f64]) -> f64`,
`box_counting_dimension_2d`, `correlation_dimension` (`crates/ix-chaos/src/fractal.rs`). The
issue's `fractal_roughness_probe` skill has no measurement primitive behind it today — but the
primitive exists, one crate over. That is the actual gap, and it is what makes the smoke tests
below falsifiable rather than "it returned some numbers."

Net: **de Rham is entirely unexposed. Takagi is exposed on MCP only. Neither is in SQL.**

## 1. Real public signatures (source of truth)

```rust
// crates/ix-fractal/src/takagi.rs   — MAX_TERMS = 53 (silently capped)
pub fn takagi(t: f64, terms: usize) -> f64;
pub fn takagi_series(n_points: usize, terms: usize) -> ndarray::Array1<f64>;

// crates/ix-fractal/src/de_rham.rs  — MAX_DEPTH = 20 (silently capped)
pub fn de_rham_interpolate(
    p0: &Array1<f64>, p1: &Array1<f64>,
    depth: usize, roughness: f64, rng: &mut impl Rng,
) -> Vec<Array1<f64>>;                       // len = 2^depth + 1  (2 when depth == 0)
pub fn de_rham_curve_1d(depth: usize, roughness: f64, rng: &mut impl Rng) -> Array1<f64>;

// crates/ix-chaos/src/fractal.rs
pub fn hurst_exponent(data: &[f64]) -> f64;
pub fn box_counting_dimension_2d(points: &[(f64, f64)], num_scales: usize) -> f64;
```

Two facts drive every design choice below:

1. **`takagi` is a pure scalar `f64 -> f64`.** It is the only function here that maps cleanly
   onto a DuckDB *scalar* UDF applied across an existing column.
2. **de Rham takes `&mut impl Rng`.** SQL has no RNG handle, so the UDF layer must own seeding:
   `StdRng::seed_from_u64(seed)`. Determinism is not a nice-to-have — it is the only way a de
   Rham UDF can be a legal SQL function at all.

## 2. exposure_plan — per function, which surface and why

| Function | DuckDB UDF | Skill | CLI (`ix run`) | Pipeline | Rationale |
|---|---|---|---|---|---|
| `takagi(t, terms)` | **YES — scalar** | via skill's SQL recipe | no | no | The one true column-wise case: `SELECT ix_takagi(t, 20) FROM series`. Cannot be replaced by existing SQL. |
| `takagi_series(n, terms)` | **NO** | already on MCP | no | no | Strictly dominated: `SELECT i/100.0 AS t, ix_takagi(i/100.0, 20) FROM range(101) r(i)` gives the same curve with idiomatic control of the grid. Adding a table fn would be a second way to do one thing. Already reachable via MCP `ix_fractal{operation:"takagi"}`. |
| `de_rham_curve_1d(depth, roughness, rng)` | **YES — table fn** | **YES** (`fractal_signal_generate`) | no | no | A synthetic multi-scale signal generator has no SQL equivalent; a table fn is the natural shape (one row per sample). |
| `de_rham_interpolate(p0, p1, …)` | **YES — table fn, long format** | **YES** (`fractal_path_interpolate`) | no | no | Arbitrary-dim path; long `(i, dim, value)` pivots in SQL. See §3 for why NOT the issue's `_path_json` shape. |
| `hurst_exponent` (ix-chaos) | **YES — scalar over `LIST<DOUBLE>`** | **YES** (`fractal_roughness_probe`) | no | no | Closes the loop: without it, generated roughness is unverifiable. Follows the exact `ix_skewness`-style `LIST<DOUBLE> -> DOUBLE` pattern. |
| `box_counting_dimension_2d` | **NO (defer)** | no | no | no | Needs a scale-selection contract (`num_scales`) and a bounded-support argument we have not designed. Out of scope for #203. |
| `ifs_iterate`, L-systems, `hilbert/peano/morton` | **NO** | already covered | no | no | Explicitly out of scope of #203; space-filling is already on MCP. `morton_encode/decode` would be a defensible future SQL scalar for spatial hashing — noted, not proposed here. |

**Surfaces deliberately left empty.** Nothing here belongs on the `ix run` CLI or in a
`ix-pipeline` DAG stage. `ix-registry` today registers algorithm skills that transform *user
data*; these are *generators* with no input dataset. Adding CLI verbs would be surface for its
own sake with no caller. If a pipeline ever needs synthetic fractal noise (e.g. an
`ix-evolution` perturbation operator), that is a separate issue with a real consumer attached.

## 3. udf_candidate_list

Grounded in the real seam: `crates/ix-duck/src/inference.rs` is the closest analogue (scalars
via `VScalar` + one table fn via `VTab`, both wired through a module-private
`pub(crate) fn register(conn: &Connection) -> duckdb::Result<()>`, itself called from
`udf::register_all` at `crates/ix-duck/src/udf.rs:279`). The plan is a new
`crates/ix-duck/src/fractal.rs` following that file line for line.

**Wiring (three mechanical steps, matching how `inference` was added):**

1. `crates/ix-duck/Cargo.toml`: add `ix-fractal` + `ix-chaos` + `rand` as `optional = true`
   deps and append `"dep:ix-fractal", "dep:ix-chaos", "dep:rand"` to the `udf` feature list.
   Both crates are `experimental` in `crate-maturity.toml`, so this does **not** trip the
   stable-surface hash gate.
2. `crates/ix-duck/src/lib.rs`: `#[cfg(feature = "udf")] mod fractal;` with the doc comment
   style used by the sibling modules.
3. `crates/ix-duck/src/udf.rs::register_all`: add `crate::fractal::register(conn)?;`.

### Scalars (`VScalar`)

```text
ix_takagi(t DOUBLE, terms BIGINT) -> DOUBLE
    -> ix_fractal::takagi::takagi(t, terms as usize)
    Signature: ScalarFunctionSignature::exact(vec![double(), bigint()], double())
    NULL on either arg -> SQL NULL (the invoke_binary convention in inference.rs).
    terms < 0 -> SQL error; terms > 53 is silently capped by the callee (documented, not re-checked).

ix_hurst(x DOUBLE[]) -> DOUBLE
    -> ix_chaos::fractal::hurst_exponent(&x)
    Signature: exact(vec![list_double()], double())  — identical to ix_skewness/ix_mad.
    Materialize a column with list(col), same as the rest of the inference scalars.
```

### Table functions (`VTab`)

All params are `Varchar` (the `ix_two_sample` / `ix_viterbi` convention: JSON-in-a-string,
parsed in `bind`), rows precomputed in `bind` and drained by a `Cursor` in `func`.

```text
ix_de_rham_1d(depth VARCHAR, roughness VARCHAR, seed VARCHAR)
    -> TABLE(i BIGINT, t DOUBLE, value DOUBLE)
    bind: parse depth/roughness/seed; reject depth > 16 (see cost_notes);
          let mut rng = StdRng::seed_from_u64(seed);
          de_rham_curve_1d(depth, roughness, &mut rng)
    rows: 2^depth + 1; t = i / (n-1).

ix_de_rham_path(p0_json VARCHAR, p1_json VARCHAR, depth VARCHAR, roughness VARCHAR, seed VARCHAR)
    -> TABLE(i BIGINT, dim BIGINT, value DOUBLE)
    bind: p0/p1 are JSON number arrays (serde_json::from_str::<Vec<f64>>, same as
          ix_two_sample's sample parsing); error if lengths differ or either is empty;
          reject depth > 12 and dim > 8 (see cost_notes);
          de_rham_interpolate(&p0, &p1, depth, roughness, &mut rng)
    rows: (2^depth + 1) * dim, long format.
```

**Rejected: `ix_de_rham_path_json(...) -> JSON` from the issue.** A scalar returning a whole
path as a JSON blob is un-SQL-like — you cannot filter, aggregate, or join it without
`json_extract` gymnastics, and it hides the row-count blow-up behind a single opaque cell.
The long-format table fn is the same information in a shape DuckDB can actually work with
(`PIVOT` recovers wide form when needed). This is the one substantive departure from the
issue's candidate list.

**Naming note (one-way-ish):** `ix_de_rham_1d` / `ix_de_rham_path` differ from the issue's
`ix_de_rham_curve_1d` / `ix_de_rham_path_json`. Once a UDF name ships and an analyst notebook
references it, renaming is a breaking change. Worth 30 seconds of sign-off before implementation,
not after.

## 4. skill_candidate_list

Skills here mean `.claude/skills/<name>/SKILL.md` (the markdown surface), not `ix-skill` verbs —
`crates/ix-skill/src/verbs/` holds only generic verbs (`run`, `list`, `describe`, `demo`,
`compile`, `pipeline`), and `ix-registry` has no fractal entry.

**Recommendation: extend the existing `.claude/skills/ix-fractal/SKILL.md` rather than create
three new skills.** It already exists, already covers Takagi, and today omits de Rham entirely.
Three sibling skills for one small crate is exactly the artifact bloat `demerzel-lolli-remediate`
exists to clean up. The three named workflows become three documented *sections* with worked
examples. Note the file carries `disable-model-invocation: true`; leave that as-is unless the
operator wants these auto-invocable.

### `fractal_signal_generate`
```yaml
input:  { depth: int 1..16, roughness: float >= 0, seed: uint64 }
output: { n_samples: 2^depth + 1, samples: [ {i, t, value} ], seed_echoed: uint64 }
route:  DuckDB  SELECT * FROM ix_de_rham_1d('8','0.3','42')
        or MCP  ix_fractal { operation: "de_rham_1d", depth, roughness, seed }   [new op]
contract: same (depth, roughness, seed) MUST yield byte-identical samples.
```

### `fractal_path_interpolate`
```yaml
input:  { p0: [float], p1: [float] (same length, 1..8), depth: int 1..12,
          roughness: float >= 0, seed: uint64 }
output: { n_points: 2^depth + 1, dim: int, path: [[float]] (wide, i-major) }
route:  DuckDB  SELECT * FROM ix_de_rham_path('[0,0]','[1,1]','6','0.3','42')
contract: path[0] == p0 and path[n-1] == p1 EXACTLY (endpoints are never displaced —
          holds by construction in de_rham_interpolate, covered by test_endpoints_preserved).
```

### `fractal_roughness_probe`
```yaml
input:  { samples: [float] (>= 32 values) }
output: { hurst: float, reading: "smooth (H>0.5, persistent)" |
                                 "brownian (H~0.5)" |
                                 "rough (H<0.5, anti-persistent)" }
route:  DuckDB  SELECT ix_hurst(list(value)) FROM ...
contract: descriptive only. H is an ESTIMATE with real variance on short series — the skill
          must state that and must NOT be used as a pass/fail gate without a calibrated
          threshold. (@ai:assumption the smooth/brownian/rough banding is a reading aid,
          not a validated classifier [U:uncertain])
```

## 5. smoke_test_plan

Bounded, seeded, property-based. Placed as `#[cfg(all(test, feature = "duck"))] mod tests` in
`crates/ix-duck/src/fractal.rs`, using `crate::open_bench()` exactly like
`ix_duck::inference::tests`. Every case asserts a *property*, never a golden float dump.

| # | Test | SQL / setup | Expected property |
|---|---|---|---|
| S1 | `takagi_known_values` | `SELECT ix_takagi(0.0,20), ix_takagi(0.5,20), ix_takagi(1.0,20)` | `0.0`, `0.5`, `0.0` within 1e-10. Mirrors the crate's own doc-tested values; catches an arg-order or cast bug. |
| S2 | `takagi_symmetry` | `ix_takagi(0.37,20)` vs `ix_takagi(0.63,20)` | equal within 1e-10 (T(t) = T(1−t)). |
| S3 | `takagi_null_propagates` | `SELECT ix_takagi(NULL, 20)` | SQL NULL, not an error — the `inference.rs` NULL convention. |
| S4 | `takagi_over_a_column` | `SELECT count(*) FROM (SELECT ix_takagi(i/100.0,20) v FROM range(101) r(i)) WHERE v BETWEEN 0 AND 0.6667` | 101. Proves vectorized invoke over a real chunk (the bug class a single-row test misses) and the known 2/3 bound. |
| S5 | `de_rham_1d_row_count` | `SELECT count(*) FROM ix_de_rham_1d('8','0.3','42')` | exactly 257 (= 2^8 + 1). |
| S6 | `de_rham_1d_determinism` | run S5's query twice, full-join on `i` | zero rows differ. This is *the* load-bearing test: a SQL function that isn't deterministic is a correctness bug, not a quality issue. |
| S7 | `de_rham_1d_seed_sensitivity` | seed `'42'` vs `'43'`, same depth/roughness | at least one interior sample differs by > 1e-12 (guards against a silently-ignored seed param). |
| S8 | `de_rham_zero_roughness_is_a_line` | `ix_de_rham_1d('6','0.0','42')` | every `value` equals `t` within 1e-10 (the crate's `test_zero_roughness_is_straight_line`, re-asserted through SQL). |
| S9 | `de_rham_path_endpoints_exact` | `ix_de_rham_path('[0,0]','[1,1]','6','0.3','42')`, rows `i=0` and `i=64` | exactly `[0,0]` and `[1,1]` — bitwise, no tolerance. |
| S10 | `de_rham_path_shape` | same query | `count(*) = 130` (= 65 × 2), `dim ∈ {0,1}`. |
| S11 | `de_rham_path_rejects_ragged` | `ix_de_rham_path('[0,0]','[1]',...)` | SQL error mentioning dimension mismatch. No panic. |
| S12 | `de_rham_depth_cap` | `ix_de_rham_1d('17','0.3','42')` | SQL error naming the 16 cap. Explicit, **not** the callee's silent cap-at-20 — a silent cap here would emit 1M rows the analyst did not ask for. |
| S13 | `hurst_round_trip` | generate `ix_de_rham_1d('10', r, '42')` for `r ∈ {0.05, 0.9}`, feed each through `ix_hurst(list(value))` | `hurst(r=0.05) > hurst(r=0.9)`. The end-to-end tracer bullet: generator and estimator agree on the *direction* of roughness. Deliberately an ordering assertion, not an absolute-value one — H on 1025 samples has real variance. |
| S14 | `hurst_null_propagates` | `SELECT ix_hurst(NULL)` | SQL NULL. |

S13 is the tracer bullet the CLAUDE.md discipline asks for: one thin slice through
generate → SQL → estimate → assert, before any of it is scaled or made pretty. If S13 cannot be
made to pass reliably at a fixed seed, the whole roughness-probe skill should be dropped rather
than shipped as decoration.

## 6. cost_notes

**Output-size bounds. This is the main risk of the whole issue** — `de_rham_interpolate`'s own
cap is `MAX_DEPTH = 20`, i.e. **1,048,577 points**, applied *silently*. Passing that cap through
to SQL means one typo (`depth=20` instead of `2`) materializes a million rows inside `bind`,
in memory, before a single row is emitted.

| Surface | Limit | Worst case | Reasoning |
|---|---|---|---|
| `ix_takagi` scalar | none needed | O(53) ops/row | `terms` is capped at 53 by the callee and the work is per-row constant. No output amplification: one input row → one output row. |
| `ix_de_rham_1d` | **depth ≤ 16** | 65,537 rows × 3 cols ≈ 1.5 MB | 65k rows is a plottable, scrollable, in-a-notebook quantity. Depth 17–20 (131k–1M rows) has no analyst use case that a `TABLESAMPLE` over depth 16 doesn't serve. Rejected loudly (S12), never silently. |
| `ix_de_rham_path` | **depth ≤ 12 AND dim ≤ 8** | 4,097 × 8 = 32,776 rows | Long format multiplies by `dim`, so the depth budget must be lower than the 1-D case to land in the same order of magnitude. dim ≤ 8 because the primitive is for paths in low-dimensional parameter/latent spaces; nobody interpolates a 512-d embedding this way. |
| `ix_hurst` | inherits `LIST<DOUBLE>` | O(n log n)-ish over one materialized list | Same profile as `ix_skewness` et al. Recommend the skill warn below ~32 samples where the estimate is noise. |

**Memory shape.** The `VTab` pattern precomputes all rows in `bind` and drains them via a
cursor (`RowsF64` + `Cursor` in `graphsig.rs`). There is no streaming. So the caps above are
peak-RSS caps, not throughput caps — which is exactly why they must be enforced at the UDF
boundary rather than inherited from the callee.

**Build cost.** Zero for the default workspace build: everything lands behind the existing
`udf` feature, which `cargo build --workspace` never compiles. `ix-fractal`, `ix-chaos`, and
`rand` are all already workspace members with no new third-party deps introduced.

**Parity cost.** If the MCP `de_rham_1d` operation in §4 is added to `ix_fractal`, note that
`crates/ix-agent/tests/parity.rs` asserts `EXPECTED.len() == 94` — but adding an *operation* to
the existing `ix_fractal` tool does not change the tool count, so no bump is needed. Adding a
new tool would.

## 7. Reversibility and revisit trigger

**Two-way door, with one exception.**

- Adding UDFs behind the opt-in `udf` feature over two `experimental`-tier crates is reversible:
  delete the module, drop the `register` call, drop the Cargo entries. Nothing in the default
  build or CI path depends on it, and no on-disk contract or schema hash is involved.
- **The one-way component is the UDF *names* and the `ix_de_rham_path` column contract**
  (`i, dim, value`). Once an analyst notebook or a `ga/state/quality/` query references them,
  renaming breaks callers silently. Cheap to get right now, expensive later — hence the naming
  note in §3. Sign-off needed on names only, before implementation.
- The `ix_de_rham_path_json` rejection is itself reversible: if a JSON-shaped consumer shows up,
  it can be added alongside the table fn without removing anything.

**Revisit trigger:** the first consumer of any fractal UDF outside of the smoke tests. Until
then this stays a small, cheap, unloved surface — and if no consumer appears within one quarter
(by 2026-10-20), the honest move is to delete the UDFs rather than let them become
green-but-dead surface. Second trigger: `ix-fractal` promoted out of `experimental` in
`crate-maturity.toml`, which would put these signatures under the stable-surface hash gate.

## 8. Recommended build order (vertical slices)

Each step is end-to-end and independently mergeable.

1. **Slice 1 — `ix_takagi` scalar.** One function, one file, tests S1–S4. Proves the
   Cargo/feature/`register_all` wiring with the simplest possible payload.
2. **Slice 2 — `ix_de_rham_1d` table fn.** Tests S5–S8, S12. Introduces seeded determinism.
3. **Slice 3 — `ix_hurst` + S13.** The tracer bullet closes; roughness becomes measurable.
4. **Slice 4 — `ix_de_rham_path` table fn.** Tests S9–S11. Deliberately last: highest output
   amplification, lowest confidence in the consumer.
5. **Slice 5 — SKILL.md update + optional MCP `de_rham_1d` op.** Documentation follows working
   surface, never precedes it.

Slices 4 and 5 are legitimately droppable if slices 1–3 find no consumer.

---

## Research Insights (deepened 2026-07-21)

_Parallel domain research + simplicity/YAGNI review (Fable-5 agents; all numerical facts verified against source) + learnings scan. Every asserted Takagi/de Rham value checks out; the deltas are a scope cut and several correctness caveats._

### Enhancement summary
- **Simplicity:** cut `ix_de_rham_path` entirely (deferred deletion with design cost paid up front); strongly consider cutting `ix_de_rham_1d` too — the MCP `de_rham_1d` op + a Rust-side S13 cover the real gap. Recommended minimum: `ix_takagi` + `ix_hurst` scalars + MCP op; 14 tests → ~5.
- **Correctness:** the `ix_hurst(list(value))` recipe has an **ordering hazard** (Hurst is order-sensitive); add `ORDER BY i`. Plus Hurst-estimator caveats, per-build-only determinism, NaN-roughness rejection, and an S9 wording fix.

### Simplicity / YAGNI (agent: simplicity review)
- **Cut `ix_de_rham_path` from the plan** — demote to the §2 deferred list next to `box_counting`, condition "a named SQL consumer." The plan already convicts it (Slice 4 "lowest confidence in the consumer," droppable), yet designs its full `(i,dim,value)` contract, `dim≤8` cap, ragged-input error, S9–S11, a skill section, *and* half the one-way naming sign-off. That's deferred deletion with the design cost paid now. Cutting it removes the sharpest one-way-door risk and S9–S11.
- **Consider cutting `ix_de_rham_1d` too.** Its only consumers are the skill section and S13 — and **S13 doesn't need the generator to be SQL** (the test can call `de_rham_curve_1d` in Rust and feed samples through `ix_hurst(list(value))`; the tracer bullet survives intact). §6 confirms the MCP `de_rham_1d` op costs zero parity bump. Apply the plan's *own* rules symmetrically: it rejected `takagi_series` as "a second way to do one thing" already on MCP (§2) and says "delete if no consumer" (§7) — both de Rham table fns fall out under those rules. Cutting both deletes §6's self-declared "main risk of the whole issue" (output-size bounds) rather than mitigating it.
- **Recommended minimum:** `ix_takagi` + `ix_hurst` (two scalars following `ix_skewness`/`invoke_binary` line-for-line, no caps, no RNG-in-SQL, no amplification) + MCP `de_rham_1d` promoted from "optional slice 5" to *the* de Rham exposure — on the surface that actually has users (agent/MCP), not the warehouse. Base rate for a fractal table fn finding a SQL consumer in a quarter is low (ecosystem history of green-but-dead ix-duck UDFs, `project_epistemic_sql_ixduck`); the plan's own delete-by-2026-10-20 trigger is effectively that prediction.
- **Skill:** drop the `smooth/brownian/rough` banding (the plan itself marks it `@ai:assumption … not a validated classifier`) — a banded enum invites downstream branching that turns a reading aid into a load-bearing unvalidated gate. Return `hurst: float` + one sentence. `fractal_path_interpolate` dies with `ix_de_rham_path`. Resolve the dual DuckDB-or-MCP route (same "two ways to do one thing").
- **Tests:** cut S2 (symmetry) and S8 (zero-roughness line) as *UDF* tests — they re-verify callee math the wrap can't selectively break (S1+S4 catch arg-order/cast/vectorization; S6/S7/S12 are the real wrap responsibilities). Full plan 14 → ~10; recommended-minimum plan → **5** (S1, S3, S4, Rust-side S13, S14).

### Hurst-estimator caveats (agent: domain research) — the roughness probe understates the risk
- **ix-chaos implements uncorrected R/S** (`crates/ix-chaos/src/fractal.rs:97-145`): windows 4,8,16,…, log-log OLS slope, **no Anis–Lloyd correction**, and **n<8 returns a hard-coded 0.5 sentinel** (not an estimate). Its own test accepts H ∈ (0.1, 1.5) on 1024 samples.
- Uncorrected R/S is **positively biased** on short series (worst at the size-4 windows this code includes; +0.05–0.1 near H=0.5 at ~1000 samples). So S13's "ordering not absolute" is exactly right; any absolute-value assertion would flake. "≥32 samples" is a floor, not sufficiency — estimates are noise below ~256 samples; raise the skill's warning floor to ~256.
- Two failure modes to name in the skill: **(1) profile-vs-increments** — feeding de Rham *values* (an fBm-like profile) pushes H→1 and saturates; probe *first differences* for a calibrated reading (S13's direction still survives). **(2) out-of-range output** — OLS slope is unclamped (H<0 or H>1 occur on strong trends; degenerate → 0.0; n<8 → exactly 0.5, indistinguishable from Brownian). Add an `unreliable` band for H∉[0,1] or n<256 and document the 0.5 sentinel.

### Determinism — correct stance, three gotchas (agent: domain research)
- **The `ix_hurst(list(value))` recipe is the real determinism hole.** `list()` element order follows input order, which is only deterministic for a single-threaded source; over a parallel-scanned table it is **not guaranteed**, and Hurst is order-sensitive. **Every recipe and S13 must use `ix_hurst(list(value ORDER BY i))`** (DuckDB supports ORDER BY inside aggregates). This is the single most important correctness fix.
- Phrase the §4 de Rham contracts as "same (depth,roughness,seed) ⇒ identical **(i→value) mapping**" (not "byte-identical samples" in sequence order — result order without `ORDER BY` is not a SQL guarantee, and DuckDB's `preserve_insertion_order` default is under pressure to flip). S6's full-join-on-`i` design is already right (keys, not order). Skill SQL should append `ORDER BY i`.
- **Determinism is per-build, not eternal:** `StdRng` (ChaCha12) and `rand_distr::Normal` ziggurat sampling are *not* stable across `rand`/`rand_distr` major versions. Same-seed ⇒ same-bytes holds within a compiled binary (all S6 needs), but any frozen `ga/state/` golden file silently pins those crate versions — add a one-line §7 note (or switch to an explicitly-named `ChaCha12Rng`).

### Numerical facts — verified, with small fixes (agent: domain research)
- All asserted values confirmed: `T(0)=0, T(0.5)=0.5, T(1)=0`, symmetry `T(t)=T(1−t)`, max `2/3` (S4's `0.6667` bound is safe); de Rham zero-roughness ⇒ line (keep S8 at 1e-10, don't tighten — midpoint averaging accumulates rounding); endpoints preserved by construction.
- **S9: say "exact equality (`=`), no tolerance" — not "bitwise."** `x*0.0 + y*1.0` can normalize `-0.0`→`+0.0`; invisible under f64 `==`/SQL equality, but a literal bit-compare would (spuriously) notice.
- **Takagi 53-term cap is lossless** because f64 inputs are dyadic rationals (terms past the cap are *exactly* 0 for f64; truncation vs ideal-real ≤ 2⁻⁵³ ≈ 1.1e-16) — not because "2^k is unrepresentable" (it's exact far beyond 2⁵³). Fix the code-comment rationale if touched.
- **Add NaN/negative-roughness rejection in `bind`** (+ a smoke test) — a NaN roughness silently produces an all-NaN column that nothing in S1–S14 currently catches.

### DuckDB VTab notes (agent: domain research)
- duckdb-rs `VTab` is **single-threaded by default** (`max_threads=1`); the `graphsig.rs` cursor's non-atomic load-then-store would double-emit if anyone enabled `max_threads>1`. Copy the pattern as-is but **don't advertise parallel-safety**.
- Precompute-in-`bind` ⇒ peak-RSS cap (correct framing): depth 16 ≈ 1 MB (fine), the callee's silent 20 ≈ 16 MB — exactly why §6's "enforce at the UDF boundary" is right. True streaming isn't worth it (midpoint displacement is inherently coarse-to-fine over the whole buffer). Caps (depth≤16; depth≤12 & dim≤8) are comfortably inside DuckDB's happy zone — no change.

### Institutional notes (agent: learnings)
- Experimental-tier crates (`ix-fractal`, `ix-chaos`) don't trip the stable-surface hash gate; promotion later *would*. Adding an *operation* to `ix_fractal` doesn't change the tool count → no `parity.rs` bump (a new *tool* would). UDF names are one-way; keep ix-duck terse (no `cargo fmt`). No-UDAF constraint stands (use table fn + GROUP BY if aggregation is ever needed).

### References
Weron 2002 (arXiv cond-mat/0103510); Kristoufek 2010 (R/S vs DFA finite-sample); Hamed 2007 (bias-corrected R/S); DuckDB C-API table functions + order-preservation docs; duckdb-rs VTab architecture.
