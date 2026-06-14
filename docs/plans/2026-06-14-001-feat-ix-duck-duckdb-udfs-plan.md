---
title: "ix-duck — in-process DuckDB analyst bench + IX UDFs (Tier 1)"
type: feat
status: active
date: 2026-06-14
origin: docs/DUCKDB.md
---

# ✨ ix-duck — in-process DuckDB analyst bench + IX UDFs (Tier 1)

## Overview

Add a new **`crates/ix-duck`** (library only) that embeds DuckDB in-process via `duckdb-rs`,
behind an **optional `duck` cargo feature** (mirroring the `fastembed`/`embeddings` pattern in
`ix-skill`). It gives IX an in-memory OLAP bench over the JSONL/Parquet IX already emits, and
registers IX algorithms as SQL UDFs so SQL can call into the real IX crates. **Not** a production
engine, **not** a source of truth — the analyst's bench (see `docs/DUCKDB.md`).

This plan covers **Tier 1 only**. Tier 0 (install `duckdb/duckdb-skills`) is **done + proven**
(2026-06-14, see `docs/DUCKDB.md`). Tier 2 (GA→Parquet export) and Tier 3 (`ix_optick` loadable
extension, a one-way door) are **out of scope**.

## Problem Statement / Motivation

IX emits rich JSONL telemetry (`state/thinking-machine/{hits,gaps,provenance}.jsonl`, etc.) but has
no ergonomic way to *analyze across it*. The documented `hits.jsonl` yield-split footgun
(`reference_dogfood_yield_measurement_gotcha`: cumulative mean blended pre/post → read 0.43 when the
post-fix value was 0.66) is a direct symptom: ad-hoc analysis is done with bespoke code that's easy
to get wrong. DuckDB answers these in one SQL statement, in-memory, zero-server. The differentiated
value over raw DuckDB is calling **IX algorithms** (vector/ML ops DuckDB lacks) from SQL.

## Key Technical Finding (from the API cross-check — read before scoping)

`duckdb-rs`'s scalar UDF interface is **row-wise**:

```rust
pub trait VScalar: Sized {
    type State: Sized + Send + Sync + 'static;
    fn invoke(state: &Self::State, input: &mut DataChunkHandle,
              output: &mut dyn WritableVector) -> Result<(), Box<dyn std::error::Error>>;
    fn signatures() -> Vec<ScalarFunctionSignature>;
    fn volatile() -> bool { false }
}
// Connection::register_scalar_function::<S: VScalar>(&self, name: &str) -> Result<()>
//   where S::State: Default;        // gated behind the `vscalar` feature
```

A scalar UDF maps **one row → one value**. Of the 4 UDFs locked in the brainstorm, only **`ix_cosine`
(two vectors → scalar)** is naturally row-wise. The other three are **set-relative** and cannot be
honest scalar UDFs:

| Brainstorm UDF | Reality | Correct DuckDB mechanism |
|---|---|---|
| `ix_cosine(a, b)` | row-wise (2 vecs → f64) | **scalar UDF** (`VScalar`) ✅ |
| `ix_kdist(v, k)` | needs a *reference set* (OOD = dist to k-NN in a corpus) | scalar `ix_euclidean`/`ix_cosine` + **SQL kNN recipe** (`ORDER BY dist LIMIT k`), or a LIST-typed scalar later |
| `ix_pca_project(v, k)` | needs a *fitted* model (components from the whole set) | **table function** (`VTab`): input relation → projected relation |
| `ix_silhouette(points, labels)` | *aggregate* over all points+labels → one score | **table/aggregate function** |

This is a "surface, don't guess" moment: the brainstorm picked 4 "UDFs" loosely; the API says 3 are
not row-scalar. **Recommendation:** stage them by mechanism rather than force-fit. Crucially, the
**MVP yield-split needs no custom UDF at all** — it's pure SQL over `hits.jsonl` — so v1 can prove the
whole harness + the one genuine scalar UDF without blocking on the table-function machinery.

## Proposed Solution (phased)

### Phase 1 — Crate skeleton + harness (the walking skeleton)
- New workspace member `crates/ix-duck` (library only).
- `duckdb = { version = "1", features = ["bundled", "vscalar"], optional = true }` + `[features] duck = ["dep:duckdb"]`. **Not** in default features (CI's `cargo build --workspace` must never pull the native build).
- `pub fn open_bench() -> Result<Connection>` → `Connection::open_in_memory()` with IX UDFs pre-registered (Phase 2). Behind `#[cfg(feature = "duck")]`.
- Smoke test: open in-memory conn, `SELECT 1`, and `read_json_auto('state/thinking-machine/hits.jsonl')` returns rows.

### Phase 2 — `ix_cosine` scalar UDF (proves the mechanism)
- Implement `VScalar` for an `IxCosine` type; `invoke` reads two DOUBLE[] (LIST) columns, calls **`ix_math::distance::cosine_similarity`** (pinned: `(&Array1<f64>, &Array1<f64>) -> Result<f64, MathError>`), writes one DOUBLE.
- Register via `register_scalar_function::<IxCosine>("ix_cosine")`.
- Test: `SELECT ix_cosine([1,0,0]::DOUBLE[], [1,0,0]::DOUBLE[])` ≈ 1.0; orthogonal ≈ 0.0; dimension-mismatch → SQL error (not panic).

### Phase 3 — MVP yield-split example + the kNN/OOD SQL recipe
- `examples/yield_analysis.rs` (gated `#[cfg(feature = "duck")]`): open bench, run the proven yield-split over `hits.jsonl` (by `outcome`, and pre/post a `--fix-ts`), print a table. This is the **value proof** and needs **no UDF**.
- Add `ix_euclidean(a, b)` scalar (same shape as `ix_cosine`, wraps `ix_math::distance::euclidean`) so the **kNN-distance/OOD** question is expressible as a documented SQL recipe (`ORDER BY ix_euclidean(q, r) LIMIT k`) — this is the honest replacement for the row-scalar `ix_kdist`.

### Phase 4 (flagged, may slip to v1.1) — set-relative ops as table functions
- `ix_pca_project` and `ix_silhouette` as **table functions** (`VTab`), wrapping **`ix_unsupervised::pca::PCA`** and a **centralized `silhouette_score`**. See "Open Questions". Only build in this plan if Phase 1–3 land cleanly with budget; otherwise split to a follow-up plan.

## Architecture

- **Crate:** `crates/ix-duck`, `[lib]` only, workspace member. Deps: `ix-math` (cosine/euclidean),
  `ix-unsupervised` (PCA, + silhouette if centralized), `ndarray` (workspace), `duckdb` (optional).
  **No `ix-agent` dependency** (keeps it lib-light, off the stable/MCP surface).
- **Feature gating:** all DuckDB code behind `#[cfg(feature = "duck")]`. Without the feature the crate
  compiles to ~nothing (or a stub returning a "feature disabled" error), so the CI workspace build stays clean.
- **No new MCP tool / no parity.rs change** in v1 — ix-duck is a dev/analysis library + example, not an
  agent surface. (Avoids the parity-cascade entirely; see `reference_ix_agent_parity_cascade`.)

### Wiring (pinned, do NOT reimplement)
- `ix_cosine` → `ix_math::distance::cosine_similarity` (`crates/ix-math/src/distance.rs:60`)
- `ix_euclidean` → `ix_math::distance::euclidean` (`crates/ix-math/src/distance.rs:18`)
- `ix_pca_project` → `ix_unsupervised::pca::PCA::new(k).fit(..).transform(..)` (`crates/ix-unsupervised/src/pca.rs:26`)
- `ix_silhouette` → `silhouette_score(&Array2<f64>, &[usize]) -> f64` — currently duplicated in
  `crates/ix-agent/src/eval/silhouette.rs:26` and `crates/ix-voicings/src/lib.rs:611`. **Sub-decision:**
  centralize into `ix-unsupervised` (its natural home) and have both existing callers + ix-duck use it,
  rather than depend on ix-voicings (a music crate) from ix-duck.

## MVP (verified value — already proven via raw CLI, now through ix-duck)

```rust
// crates/ix-duck/examples/yield_analysis.rs   (cfg(feature = "duck"))
let conn = ix_duck::open_bench()?;                 // in-memory, IX UDFs registered
let rows = conn.prepare(
    "SELECT outcome, count(*) AS n, round(avg(coverage_max),4) AS yield
     FROM read_json_auto('state/thinking-machine/hits.jsonl')
     GROUP BY outcome ORDER BY n DESC")?
    .query_map([], /* ... */)?;
// proven output (2026-06-14): out_of_domain 0.3883 | compiled 0.4495 | governance_rejected 0.6433
```

## Acceptance Criteria

- [x] `crates/ix-duck` exists as a workspace member; `cargo build --workspace` (no features) does **not** pull `duckdb`/native build. *(no-feature build 0.24s)*
- [x] `cargo build -p ix-duck --features duck` builds (bundled DuckDB compiles). *(1m03s on Windows; build.rs links `rstrtmgr`)*
- [x] `open_bench()` returns an in-memory connection that can `read_json_auto` the IX JSONL.
- [x] `ix_cosine` registered + correct: identical→1.0, orthogonal→0.0, dim-mismatch→SQL error (no panic). Each case a test.
- [x] `ix_euclidean` registered + tested; kNN/OOD SQL recipe documented in the crate doc-comment.
- [x] `examples/yield_analysis.rs` runs under `--features duck` and reproduces the proven yield-split numbers (out_of_domain 0.3883 / compiled 0.4495 / blended 0.4215); smoke test covers `read_json_auto`.
- [x] Every UDF invariant carries an `@ai:` annotation with truth_value + certainty bound to its test.
- [x] `cargo clippy -p ix-duck --all-targets -- -D warnings` clean **and** `... --features duck ...` clean (CI-exact invocation).
- [x] `docs/DUCKDB.md` Tier-1 row flipped to done; plan linked.

> **Phase 4 (`ix_pca_project`/`ix_silhouette` table functions) deferred to a follow-up plan**, per the "may slip to v1.1" gate. Phases 1–3 shipped.

## System-Wide Impact

- **Build graph:** the `duck` feature pulls a **bundled C++ DuckDB build** (slow first compile, needs a
  C++ toolchain). Isolated behind the optional feature exactly like `fastembed`, so the default CI path is unaffected.
- **No runtime/agent surface change:** no MCP tool, no `tools.rs`/`parity.rs`/`classify.rs` edits → no parity cascade.
- **Stable tier:** ix-duck is explicitly **not** stable-surface; keep it out of the stable-surface hash set.
- **Error propagation:** UDF errors must surface as DuckDB SQL errors (`Box<dyn Error>` from `invoke`), never panic across the FFI boundary.

## Risks & Mitigations

- **Bundled build time / Windows C++ toolchain** → optional feature; document `--features duck`; CI never builds it by default. Confirm it builds on this Windows box once.
- **duckdb-rs API drift** (the `VScalar`/`register_scalar_function` shape moved across versions) → **pin an exact `duckdb` version**; the signatures in this plan are from current `main` and must be re-confirmed against the pinned version at impl time.
- **Vector/LIST marshalling** (reading `DOUBLE[]` into `Array1<f64>` in `invoke`) is the fiddly part → cover with the dim-mismatch + happy-path tests first.
- **Set-relative UDFs** (`pca_project`/`silhouette`/`kdist`) tempt a wrong row-scalar impl → explicitly staged to table functions / SQL recipes (Phase 4 / v1.1).

## One-way / two-way door log

- **Two-way:** `duckdb` as an *optional* dep behind `duck`. Reversible (delete crate/feature). Logged per CLAUDE.md.
- **Would-be one-way (NOT in this plan):** a published `ix_optick` DuckDB extension (loadable artifact, per-platform builds/signing, public API) — Tier 3, needs sign-off.

## Open Questions (resolve during impl)

1. Centralize `silhouette_score` into `ix-unsupervised` (recommended) vs depend on `ix-voicings`? (Recommend: centralize.)
2. Build Phase 4 (table functions) in this plan, or split to `2026-06-14-002-...` once Phase 1–3 land? (Recommend: split unless budget is ample.)
3. Pinned `duckdb` crate version + exact feature set (`bundled`, `vscalar`, need `vtab` only if Phase 4 here).

## Sources & References

### Origin
- **Brainstorm/hub:** [docs/DUCKDB.md](../DUCKDB.md) — decisions carried forward: ix-duck crate + `duck` feature; UDF set (cosine/kdist/pca/silhouette); IX-local yield-split MVP; optick.index stays production; GA stays source of truth; skip MotherDuck; `acp` parked.

### Internal references
- `crates/ix-skill/Cargo.toml:61-71` — the optional-dep + feature pattern to mirror.
- `crates/ix-math/src/distance.rs:18,60` — euclidean / cosine_similarity.
- `crates/ix-unsupervised/src/pca.rs:14,26` — `PcaState` / `PCA`.
- `crates/ix-agent/src/eval/silhouette.rs:26`, `crates/ix-voicings/src/lib.rs:611` — silhouette_score (to centralize).
- `reference_ix_agent_parity_cascade` (memory) — CI-exact clippy; no-MCP-tool keeps us off the parity cascade.
- `reference_dogfood_yield_measurement_gotcha` (memory) — the footgun the MVP kills.

### External references
- `duckdb-rs` `VScalar` trait + `register_scalar_function` (verified against repo `main`, 2026-06-14). DuckDB CLI v1.5.3 used for the Tier-0 proof.

## Next steps
→ `/ce-work docs/plans/2026-06-14-001-feat-ix-duck-duckdb-udfs-plan.md` to implement Phase 1–3.
