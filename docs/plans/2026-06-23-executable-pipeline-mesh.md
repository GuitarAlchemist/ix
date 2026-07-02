# Plan: an *executable* 100+ node pipeline mesh (ix-pipeline over DuckDB+IX)

**Status:** Option C built — phases 1–5 done (`crates/ix-agent/examples/ix_pipeline_mesh.rs`,
walkthrough `docs/walkthroughs/executable-pipeline-mesh.md`). 136 real skill nodes execute
through `ix_pipeline::executor`; the mesh recovers a planted outlier. Open follow-up: the
in-pipeline weighted/thresholded centrality (Option A's `mesh_correlate` skill) + swapping
synthetic streams for the real voicing fret-profiles.
**Date:** 2026-06-23
**Relates to:** [ADR-0004](../adr/0004-duckdb-sql-pipeline-mesh.md) (the SQL mesh substrate),
[ADR-0001](../adr/0001-ixql-duckdb-integration-via-mcp-seam.md) (IXQL ↔ DuckDB+IX seam)

## Why this plan

The original goal was *"DuckDB+IX … complex IXQL data pipelines, a mesh of 100+ nodes,
for a real-world analysis problem."* What shipped so far:

- `ix_voicing_mesh` (#171/#172) delivered the **100+ node mesh** — but its operator DAG was
  **reified for display** (`ix_pipeline::dag::Dag` built only to print stats); the actual
  work ran as inline DuckDB SQL + Rust. The pipelines did not *execute through* IX's
  pipeline framework.
- **IXQL itself is not runnable.** ADR-0001 is explicit: *"the IXQL executor is spec-only."*
  IXQL (Demerzel's governance DSL) and DuckDB+IX are complementary layers joined by
  JSON-on-disk. The executable IX pipeline DSL is **`ix-pipeline`** (`PipelineSpec` YAML →
  `lower()` → `Dag<PipelineNode>` → `executor::execute`), where each stage invokes a
  registered **skill** (`ix-registry`).

This plan closes the gap: realize the mesh as a **PipelineSpec that actually executes**
through `ix-pipeline`, so the 100+ nodes are *run*, not drawn.

## What is already proven (tracer, 2026-06-23)

A throwaway tracer ran a real spec end-to-end:

```text
spec: 3 stages → lower() → 3-node / 1-edge / 2-level Dag → execute()
  ac_a, ac_b (autocorrelation) ran in parallel (level 0)
  ac_chain consumed ac_a's output via {from: "ac_a.autocorrelation"} (level 1)
```

Confirmed: `PipelineSpec::from_yaml_str` → `lower` → `execute` works; `from:`-ref field
extraction wires stage outputs to downstream inputs; parallel levels execute. Host = an
**`ix-agent` example** (must `use ix_agent as _;` so the `linkme` skill slice links in).

## The gap

The available skills (`ix-agent` batches) cover per-stream work — `autocorrelation`,
`distance` (pairwise), `fft`, `fir_filter`, `kmeans`, `pca`, `graph`, `markov`,
`nn.forward`, `evolution`, … — but there is **no single "correlate-N / aggregate" skill**
for the mesh fan-in (N streams → N×N correlation → clusters → centrality). That fan-in is
the one missing piece.

## Options (the shape decision)

| Option | Executable DAG shape | Nodes | New surface | Faithfulness |
|---|---|---|---|---|
| **A. `mesh_correlate` skill** | N per-stream transform stages → **1** fan-in stage (wraps `ix_duck::mesh::correlate`) | ~N+1 (N≈114 → ~115) | new `#[ix_skill]` in ix-agent + handler + schema + **parity.rs bump** | mesh hidden inside one node; DAG is a fan-in *star*, not a mesh |
| **B. pure pairwise** | N transform stages + **O(N²)** pairwise `distance` stages + a merge + `graph` centrality | N=18 → ~170 | a small *merge* skill (collect pairwise → edge list) | the N×N mesh **is** the DAG — most literal "mesh of 100+ nodes"; cosine-distance, not Pearson |
| **C. hybrid** | N **multi-stage** stream pipelines (source→condition→feature) + pairwise edges among a capped subset + `graph` | tunable to ~120 | merge skill (as B) | "100+ pipelines forming a mesh"; balances faithfulness vs node blow-up |

**Reversibility.** Option A registers a public skill (`ix-registry` entry + parity test) —
a **one-way-ish door** (others may depend on the skill name/contract). Options B/C add only
example code + possibly a tiny internal merge skill. Revisit trigger: if a second mesh
consumer appears, promote the chosen fan-in into a first-class skill regardless.

## Recommendation

**Option C, leaning on B's machinery**, because it is the most faithful to the literal ask
("a mesh of 100+ *pipeline* nodes") and avoids prematurely freezing a public `mesh_correlate`
skill contract:

- **Streams:** reuse the real voicing fret-profiles (per set-class, computed once via the
  bench, embedded as stage data) — same data the validated `ix_voicing_mesh` used.
- **Per-stream pipeline (3 stages):** `source` (inline series) → `fir_filter`/`autocorrelation`
  (condition) → feature. ~20 streams × 3 = ~60 pipeline nodes.
- **Mesh edges:** pairwise `distance` (cosine) among the 20 stream features = 190 nodes →
  capped/thresholded to keep the DAG legible; this is where the node count crosses 100+.
- **Aggregate:** one small internal merge (pairwise → edge list) → `graph` centrality →
  hub. The merge is the only new code on the skill side; keep it example-local first, and
  only promote to a registered skill (ADR) if a second consumer appears.
- **Validation:** carry the discipline forward — the executed mesh's hub must clear the
  same null model the SQL mesh used (`ix_voicing_mesh_nullcheck`), so "executable" doesn't
  silently drop the rigor.

If the operator prefers the cleaner code and is fine logging the public-skill one-way door,
**Option A** is the smaller build (one skill + a fan-in star) — but the DAG is then a star,
not a mesh.

## Phased build

1. ✅ **Tracer** — executable spec of real skills (done).
2. **Minimal executable mesh** — 4 streams, full shape (source→condition→feature→pairwise→
   merge→graph), lowered + executed; prove the fan-in/merge composes. *(tracer-bullet)*
3. **Scale to 100+ nodes** — bump stream count + pairwise edges to cross 100 executed nodes;
   report real `execute()` DAG stats (nodes/edges/levels/wall-clock).
4. **Validate** — run the null model on the executed mesh's hub; report honestly.
5. **Docs** — EN + FR walkthrough; note this *executes* where `ix_voicing_mesh` *reified*.

## Open decision

Pick the shape (A / B / C) and the node budget. Default if unspecified: **C**, ~20 streams,
pairwise-capped to land ~120 executed nodes.
