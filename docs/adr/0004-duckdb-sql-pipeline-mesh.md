# DuckDB-SQL + IX UDFs is the IX pipeline-mesh correlation substrate (complementary to IXQL)

Status: accepted (2026-06-21)

Runtime status updated: 2026-10-13. IXQL's governance dialect now has an executor
(`crates/ix-ixql`); [ADR-0001](0001-ixql-duckdb-integration-via-mcp-seam.md) records what
is and is not wired between it and DuckDB. The decision below is unchanged: DuckDB SQL is
the mesh's analytical engine.

## Context

We want to **declare many IX "pipelines" and compose them into a mesh** that correlates
real-world data scenarios (sensors, market ticks, log-rates, ecosystem telemetry) — on the
order of 100+ pipelines — to answer "which streams move together, and which one leads?".

The instinct is to reach for **IXQL** (Demerzel's governance pipeline DSL). The
governance dialect now has a bounded executor in `crates/ix-ixql`, but it neither parses
the separate ML-pipeline dialect nor supplies DuckDB source/sink nodes. Per
[ADR-0001](0001-ixql-duckdb-integration-via-mcp-seam.md), a new grammar node needs a
demonstrated use case and Galactic-Protocol sign-off. IXQL can govern a mesh run through
registered capabilities and file artifacts; it is not the mesh's analytical engine.

Meanwhile `ix-duck` (the analyst bench, `docs/DUCKDB.md`) now exposes ~80 IX algorithm UDFs,
including the exact correlation primitives a mesh needs: `ix_pearson`, `ix_two_sample`,
`ix_kl`/`ix_js`, `ix_connected_components`, `ix_centrality`, and the per-stream conditioners
`ix_wavelet_denoise`/`ix_kalman_smooth`.

## Decision

**The pipeline-mesh correlation substrate is DuckDB SQL as the composition language with IX
UDFs as the stages — *not* IXQL, and *not* a new execution engine.** Concretely:

1. **A "pipeline" is a named DuckDB `VIEW` / table `MACRO`** over a JSON-on-disk stream,
   composing IX UDFs. The "catalog" is a set of such macros; the "mesh" is the N×N
   correlation across their outputs. This is pure analyst-bench analysis (read-only,
   no source-of-truth), so it crosses **no** governed boundary — ADR-0001 stands untouched
   (we add no IXQL grammar nodes).
2. **The canonical mesh pipeline shape** is:
   `N streams → ix_wavelet_denoise/ix_kalman_smooth (condition) → ix_pearson / ix_two_sample
   (pairwise) → threshold → edge list → ix_connected_components (incident clusters) →
   ix_centrality (lead/hub indicator)`.
3. **A thin reusable driver** (`ix-duck::mesh`) orchestrates that shape over an arbitrary set
   of named streams and returns a structured `MeshResult` (correlation matrix + clusters +
   centrality ranking + lead). It also installs a small **pipeline catalog** of reusable
   table macros (`ix_smooth`, `ix_zsmooth`) as the SQL-native declaration layer.

## Why (the trade-off)

- **Runnable analysis, zero new governed syntax.** SQL views/macros + existing UDFs run on
  the bundled-DuckDB (`duck`) feature now. The IXQL governance executor can coordinate a
  run only after a concrete capability adapter or file handoff is wired; it does not
  replace DuckDB's analytical operations (ADR-0001 path #1).
- **SQL *is* a declarative composition language.** CTEs, views, and table macros already give
  naming, parameterization, and composition — re-deriving that in a bespoke engine is the
  "build the whole thing" failure mode the repo's tracer-bullet discipline warns against.
- **The primitives already exist and are tested.** The mesh is *assembly*, not new math.

## Consequences

- `ix-duck::mesh` is **`duck`-feature-gated** (it runs SQL on a bundled engine), like the other
  lenses; the default/`--workspace` build never compiles it.
- The mesh is **advisory analysis**, never a binding gate or source of truth (`docs/DUCKDB.md`).
  Binding verdicts still go through the governed `maintain-gate` (ADR-0002), not raw mesh SQL.
- **Centrality choice is load-bearing.** On a hub-and-spoke (near-**bipartite**) correlation
  graph, `eigenvector` centrality is still the wrong lens — it spreads weight toward dense
  mutually-correlated clusters and under-weights a structural hub whose spokes don't correlate
  with each other. (The earlier power-iteration *non-convergence* on exactly-bipartite graphs is
  fixed in `ix-graph` as of #165 via the `A + I` self-loop shift, so eigenvector now returns a
  stable vector — it's just not the hub lens we want here.) `betweenness` (or `degree`) is the
  correct lead/hub lens for hub-and-spoke; `eigenvector` suits *dense, mutually-correlated*
  clusters. The driver defaults to `betweenness`.
- Pearson over **constant** streams is undefined → a SQL error; the driver treats such a pair as
  uncorrelated (no edge) rather than aborting the whole mesh.

## Use-case catalog (good scenarios)

1. **Ecosystem regression radar** — one drift pipeline per `state/quality/*.json` metric;
   `ix_two_sample` vs each metric's 30-day baseline; `ix_centrality` ranks which regressing
   metric is *causally central* vs a downstream symptom.
2. **Cross-repo correlation** — ix/tars/ga emit JSONL; the mesh joins them to test "does a tars
   grammar-weight change correlate with a ga routing-F1 drop?".
3. **Sensor / IoT / market correlation** — N raw streams → `ix_kalman_smooth` → pairwise
   `ix_pearson` → `ix_connected_components` (incident clusters) → `ix_centrality` (lead indicator).
4. **Telemetry anomaly mesh** — per-intent query streams; `ix_kdist`/`ix_dbscan` flag OOD,
   `ix_connected_components` groups co-failing intents into incident clusters.
5. **A/B & experiment fleet** — 100 arms as macros; `ix_two_sample` (distribution-free) gates each.

## Revisit trigger

When a measured mesh use case needs more than the capability/file seam, evaluate a
first-class `duckdb(…)` source inside the supported IXQL dialect, with an implemented
adapter, tests, and grammar sign-off per ADR-0001. Until then the analytical mesh lives
in ix-duck; IXQL may only coordinate it through an explicit integration.
