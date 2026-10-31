# `mesh_correlate` — the in-pipeline correlation-mesh fan-in tool

A registered `ix-agent` tool (`#[ix_tool] name = "mesh_correlate"`) that turns N series
into a correlation graph **inside an executable pipeline**, computing centrality where a
spec otherwise couldn't.

> _Version française : [`docs/fr/pas-a-pas/mesh-correlate-skill.md`](../fr/pas-a-pas/mesh-correlate-skill.md)._

## Why it exists

The executable-mesh demo ([`ix_pipeline_mesh`](executable-pipeline-mesh.md), plan
`docs/plans/2026-06-23-executable-pipeline-mesh.md`) found a real limit: a `PipelineSpec`
**can't threshold edges at build time** (the weights are run-time values), so the mesh's
clustering/centrality had to be computed back in the harness. `mesh_correlate` closes that
gap — it does the `|Pearson| ≥ τ` thresholding **and** betweenness centrality at run time,
in one fan-in node (Option A of the plan).

```
mesh_correlate({ series: [[…],[…],…], threshold: τ })
  → { correlation: N×N, centrality: [betweenness…], components: […], hub, n_streams }
```

It wraps `ix_math::inference::pearson` + `ix_graph::graph::Graph` (`betweenness_centrality`,
`connected_components`) — no new algorithm, just the fan-in shape as a tool.

## In a pipeline

`crates/ix-agent/examples/ix_pipeline_mesh_hub.rs`:

```text
mesh    = mesh_correlate(streams, τ=0.4)   → {centrality, hub, …}
summary = stats({from: mesh.centrality})   → spread of the scores
✓ executed 2 nodes through ix_pipeline::executor
```

Validation — a planted **hub-and-spoke** (one hub = mean of 3 orthogonal spokes, plus pure
-noise distractors). The executed mesh returns the hub:

```text
stream 0: 3.00  ← planted hub        stream 1: 0.00 (spoke)   …distractors: 0.00
hub = 0; planted hub = 0  →  ✓ RECOVERED (centrality computed inside the executed pipeline)
```

The downstream `stats` stage consumes `mesh.centrality`, so the mesh result flows on
through the pipeline like any other stage output.

## Scope and caveats

- This is the **Option A** fan-in (mesh in one node), complementing `ix_pipeline_mesh`'s
  Option C (the explicit 100+ node pairwise mesh). Together they show the tradeoff: an
  explicit mesh DAG vs. in-pipeline centrality.
- The example passes streams to `mesh_correlate` directly rather than through N per-stream
  cond stages, because the obvious cond (`autocorrelation`) distorts correlation (every ACF
  shares the lag-0 = 1 spike). A correlation-preserving *identity* cond tool would restore
  the N-per-stream-pipeline shape — a follow-up.
- A betweenness hub needs **hub-and-spoke** structure; a set of mutually-correlated streams
  is a clique with no hub (the same lesson as the voicing mesh).
- Advisory/illustrative, like the other mesh demos.
