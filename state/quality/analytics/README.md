# Quality analytics (DuckDB)

A zero-server SQL layer over ix's file-based quality / eval / metrics artifacts
under `state/quality/` (plus the cross-repo SAE contract output and the
`router-spike` eval). DuckDB reads the JSON in place and materializes it into a
portable, self-contained `quality.duckdb` so trends are queryable across sessions
and from Rust — without bespoke loaders that silently skip off-pattern files.

Mirrors GA's design at `ga/state/quality/analytics/`.

## Files

| File | Tracked | Purpose |
|---|---|---|
| `build-views.sql` | ✅ | Materializes the artifacts into tables + the `quality_latest` view. |
| `quality.duckdb` | ❌ (gitignored) | Generated binary; rebuild any time from the script. |

## Rebuild / refresh

Run from `state/quality/` so the relative globs resolve:

```bash
duckdb analytics/quality.duckdb < analytics/build-views.sql
```

(`duckdb` CLI: `winget install DuckDB.cli`.)

## Tables

- **`optick_sae`** — the cross-repo CONTRACT OUTPUT ix produces for GA's consumer
  side (`state/quality/optick-sae/<date>/optick-sae-artifact.json`, per
  `ga/docs/contracts/2026-05-02-optick-sae-artifact.contract.md`). Columns are the
  flattened contract shape (`input_*`, `model_*`, `metrics`, `features_*`,
  `links_supersedes`, `narrative`) so producer (ix) and consumer (GA) have SQL
  parity over the same fields. **0 rows today** — no artifact is emitted on disk
  yet, so the table is defined with the explicit contract schema (mirroring GA's
  `pr_grades` pattern). When a `<date>/optick-sae-artifact.json` lands, swap the
  table body for the commented `read_json_auto` query in `build-views.sql` and
  re-run.
- **`ix_harness`** — repo-level Agent Blackbox readiness (`harness_ready`) from
  `ix-harness/last.json`.
- **`quality_health`** — daily regression-scan roll-up (`quality-health-<date>.json`).
- **`router_eval`** — semantic intent router accuracy / macro-F1 / OOS decline from
  `../router-spike/head-eval.json`.
- **`quality_latest`** (view) — latest value per source.

## Query from Rust

`crates/ix-duck` (the in-process DuckDB bench) ships an `ix_quality_lens` example
— the ix-native analog of GA's `Tools/QualityLens`. It opens the DB **read-only**
and runs a query:

```bash
# default = quality_latest rollup over state/quality/analytics/quality.duckdb
cargo run -p ix-duck --features duck --example ix_quality_lens

# custom query (+ optional explicit DB path)
cargo run -p ix-duck --features duck --example ix_quality_lens -- "SELECT * FROM router_eval"
cargo run -p ix-duck --features duck --example ix_quality_lens -- "SELECT * FROM ix_harness" state/quality/analytics/quality.duckdb
```

The `duck` feature pulls a bundled (C++) DuckDB build and is opt-in, so the default
`cargo build --workspace` / CI path never compiles it.
