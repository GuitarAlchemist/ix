# `ix_pipeline_mesh` — an *executable* 100+ node pipeline mesh

Where [`ix_voicing_mesh`](voicing-mesh.md) **reified** an operator DAG for *display* (the
real work ran as DuckDB SQL), this builds a real `ix_pipeline::PipelineSpec`, `lower()`s it
to a `Dag`, and **`execute()`s** it through the ix-pipeline executor — every node is a real
`ix-agent` skill invocation. See the plan: `docs/plans/2026-06-23-executable-pipeline-mesh.md`.

> _Version française : [`docs/fr/pas-a-pas/maillage-pipeline-executable.md`](../fr/pas-a-pas/maillage-pipeline-executable.md)._

```bash
cargo run -p ix-agent --example ix_pipeline_mesh
```

## What "executable" means here

The IX pipeline stack is `PipelineSpec` (YAML or built in code) → `lower()` →
`Dag<PipelineNode>` → `executor::execute`, where each stage invokes a **registered skill**
(`ix-registry`). (IXQL — Demerzel's governance DSL — is *spec-only*; its executor doesn't
ship, per ADR-0001. `ix-pipeline` is the executable IX pipeline layer.)

```text
executable pipeline mesh — 136 stages (N=16 streams)
lowered → 136 nodes, 240 edges, 2 levels (widest 120 — the pairwise tier)
✓ executed 136 real skill nodes through ix_pipeline::executor (not reified)
```

The shape (Option C of the plan):

```text
cond_i   = autocorrelation(series_i)          [16 nodes, level 0]
dist_ij  = distance(cond_i, cond_j, cosine)   [120 nodes, level 1]   ← the mesh tier
```

Each `dist_ij` stage is wired to two `cond` stages by `{from: "cond_i.autocorrelation"}`
references, which `lower()` resolves into dependencies and the executor substitutes at run
time. No glue code — the wiring *is* the spec.

## Validation (ground-truth)

The 16 streams are **15 mutually-similar tones + 1 planted outlier** (a very different
frequency). The hub is read back from the executed pairwise outputs (the stream with the
largest mean cosine distance to the rest). The executed mesh must recover the planted
outlier:

```text
mean cosine distance to the rest (top 5):
   stream 15: 0.8899  ← planted outlier
   stream 12: 0.0597
   stream  4: 0.0595
validation — hub = stream 15; planted outlier = 15  →  ✓ RECOVERED
```

## A real constraint, surfaced by building

The first attempt aggregated with a `graph` **pagerank** fan-in node (edge weights as
`{from: "dist_ij.distance"}` refs — which *do* resolve inside arrays). It executed, but
returned **uniform** rank: pagerank on a *complete* graph is uniform regardless of weight,
and a spec **cannot threshold edges at build time** because the weights are run-time values.

So the executable mesh is the **pairwise tier** (the 100+ nodes that genuinely run), and
the thresholded/centrality aggregation is computed example-local from the executed outputs.
A weighted/thresholded centrality *inside* the pipeline would need a dedicated fan-in skill
(Option A in the plan) — deferred, since that freezes a public skill contract.

## Scope and caveats

- The streams here are synthetic-but-structured (planted outlier) to give a checkable
  ground truth; swapping in the real voicing fret-profiles is the obvious next step (it
  needs the DuckDB bench as a data source — a heavier dep for an `ix-agent` example).
- Node count is O(N²) in the pairwise tier by design — that *is* the mesh; N = 16 → 136.
- Advisory/illustrative, like the other `ix_duck` / mesh demos.
