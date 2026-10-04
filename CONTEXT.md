# CONTEXT — ix domain glossary

> The shared language of this repo. `/grill-with-docs` grows this file lazily as
> terms get resolved during planning; `/improve-codebase-architecture`,
> `/diagnose`, and `/tdd` read it so their output uses **our** words, not synonyms.
> This is a **seed** — add terms when a real ambiguity is resolved, not speculatively.

## What ix is

A Rust workspace (77 crates) implementing foundational **ML/math algorithms** and
**AI governance** as composable crates, exposed via an MCP server (`ix-agent`) and
CLI (`ix-skill`). Part of the GuitarAlchemist ecosystem (**ix** + **tars** + **ga**
+ **Demerzel**). Source-of-truth for cross-repo collaboration is JSON-on-disk
contracts (see `docs/contracts/`), not runtime coupling.

## Core terms

- **Crate** — a unit of capability (`ix-<domain>`). Each defines traits
  (`Regressor`, `Classifier`, `Clusterer`, `Optimizer`, …) and uses the builder
  pattern + seeded RNG for reproducibility. CPU = `f64` + `ndarray`; GPU = `f32`
  via WGPU shaders. See `README.md` for the full crate map.
- **Skill** — a capability exposed to Claude Code / agents (a `.claude/skills/<name>/SKILL.md`).
  Distinct from a **crate** (Rust library) and a **tool** (MCP-callable function).
- **Tool** — an MCP-callable function (registered in `ix-agent`; the count is
  asserted by `crates/ix-agent/tests/parity.rs` — every tool-adding PR bumps it).
- **Governance / Demerzel constitution** — all agent actions are subject to the
  Demerzel constitution (`governance/demerzel`). The **Galactic Protocol** is the
  cross-repo contract layer; **Prime Radiant** is the 3D governance-graph viz.
- **Hexavalent logic** — truth values are **T / P / U / D / F / C** (not just
  true/false/unknown/contradictory; `D` is **Doubtful** — evidence *leans false*,
  the symmetric mirror of `P` Probable — never "Disputed"). Used in `@ai:`
  annotations and belief state. The **canonical deep type** is
  `ix_types::Hexavalent`: it owns the *one* truth-table algebra
  (`and`/`or`/`not`/`implies`/`xor`/`equiv`, with `or` De Morgan-derived from
  `and` so the two can't drift) plus `polarity`/`conflicts`/`weighted`. The two
  wire-contract enums — `ix_ai_annotations::TruthValue` (the `T/P/U/D/F/C`
  annotation-schema form) and `ix_governance::TruthValue` (the governance/MCP
  `"True"`-named form) — are **adapters** that delegate their algebra to it; they
  exist only because their serialized wire formats genuinely differ.
- **`@ai:` annotation** — an inline claim marker (`@ai:invariant`/`assumption`/…)
  with a truth_value + certainty, where **`certainty := strength of live binding`**
  (test-bound → `T:test`; human-only → cap at `P:assumed`). Drift-gated in CI.
- **Federation / registrar** — the JSON-on-disk pattern where each repo declares a
  per-repo source (frontmatter / manifest) that an ix generator federates into a
  `state/<x>/catalog.jsonl`, DuckDB-queryable + drift-gated. Instances: **Streeling**
  (learnings), and the in-flight **business-value scorecard**.
- **OPTIC-K** — the mmap voicing index schema (`ix-voicings`/`ix-optick`) consumed
  by `ga`. **Voicing** = a fingered chord shape (music-domain, lives in `ga`).
- **Analyst's bench** — the in-process, in-memory DuckDB layer (`ix-duck`) over the
  JSONL/Parquet IX and `ga` already emit. Not a production engine, not a source of
  truth (see `docs/DUCKDB.md`).
- **Coordination shape** — a content-addressed, read-only advisory artifact derived
  from typed event-graph windows: balance residuals, queue pressure, tail latency,
  graph gradients or Laplacian energy, and cycle exposure. It is not a continuum
  stress tensor, routing decision, authority grant, safety verdict, or control
  command. Current execution belongs to DuckDB and IX modules; IXQL may describe or
  verify the plan but remains spec-only and non-executable.
- **Lens** — a read-only analyst module on the bench (`ix_duck::{chatbot, routing,
  loops, ood, maintain}`) that turns a GA artifact set into a queryable signal. A lens
  owns *analytics*, not ingest.
- **Topological fingerprint** — the *static* per-instrument Betti numbers (β₀ = connected
  components, β₁ = loops) of an embedding sample at a fixed filtration radius, computed daily
  by `ix-embedding-diagnostics` (via `ix_topo::pointcloud::betti_at_radius`) and retained in
  `state/quality-snapshots/embeddings/*.json`. Answers *"what shape does the embedding space
  have today?"* — one snapshot, no time axis.
- **Topological drift** — the *change* in an embedding space's shape between two points in time,
  measured as the `bottleneck_distance` / `wasserstein_distance` between their persistence
  diagrams (both in `ix-topo`, neither yet called by the diagnostic). Distinct from a
  **topological fingerprint** (static) and from **distributional drift** (the OOD lens /
  `ix_two_sample`, which sees a moment/quantile shift but not a shape change — a hole closing or
  clusters merging). An unproven signal: whether β₀-drift carries information the existing
  `leak_detection` metric misses is a measure-first question against the retained β₀ history.
- **Pipeline mesh** (`ix_duck::mesh`) — composing **N IX "pipelines"** (each a named
  SQL view/macro over a stream, built from IX UDFs) and correlating their outputs N×N
  to find which streams move together (clusters) and which one leads (centrality). The
  canonical shape is `condition → ix_pearson → ix_connected_components → ix_centrality`.
  DuckDB SQL is the composition language, **not** IXQL (which is spec-only and stays
  complementary — see `docs/adr/0004-duckdb-sql-pipeline-mesh.md` + ADR-0001). Advisory
  analysis only; betweenness/degree (not eigenvector) is the hub lens on bipartite meshes.
- **Artifact source** (`ix_duck::source`) — the deep module a **lens** reads through:
  given a file selector + a flat **column spec**, it materializes a GA-emitted JSON
  artifact set into a bench table, owning file selection, the `read_json_auto` flags,
  the empty-fallback (typed schema, 0 rows), and the **safe projection**
  (`json_extract(to_json(obj),'$.f')` / `TRY_CAST`, never struct-field access, never
  `coalesce(...,0)`). The seam that makes the absence-as-zero / struct-bind-crash
  defect class non-recurring (see `docs/solutions/.../2026-06-19-duckdb-absence-as-zero-and-struct-bind-crash.md`).
- **Activation coverage** (`activations_coverage` in `optick-sae-artifact.json`) — the
  declared share of the OPTIC-K corpus present in `feature_activations.parquet`. The
  parquet holds the **train split only** (measured 2026-09-07: 297,395 of 313,047
  voicings, **95.0%**), each row keyed by `optick_row` — its position in the *full*
  index, not in the split. So a full-corpus join legitimately misses 15,652 rows, and
  before this block existed it missed them **silently**: no error, no field, and a
  plausible-looking row count (ix#248). Coverage is *declared* (producer-side counts),
  *validated* (`validate_coverage` — additivity, staleness, a 90% floor) and
  *reconciled* (`optick_coverage.reconcile` — the declaration against the parquet's
  actual key column). Only the third layer can see the bytes; the first two compare the
  producer's numbers to themselves. Contract:
  `docs/contracts/2026-09-07-optick-sae-activations-coverage.contract.md`.
- **Capability (IXQL)** (`ix_ixql::capability`) — an adapter registered under the name
  a pipeline calls a peer operation by (`tars.research`, `alert`) or a named governance
  check by (`→ explanation_requirement`). The evaluator dispatches to it instead of
  matching on peer names, and never decides what a check *means* — that is Demerzel's.
  An unregistered name fails the run; the language's own built-ins (`ix.io.write`, …)
  cannot be registered over, so no adapter can route around the schema gate. Distinct
  from a **tool** (MCP-callable) and a **skill**: a capability is how one IXQL run
  reaches either, or something else entirely.
- **Verdict gate** — consecutive `→ when T >= 0.8: …` / `→ when C: …` steps, read as
  **one** match over the verdict (hexavalent truth + confidence) the previous
  capability attached — not as successive filters. The verdict travels *beside* the
  JSON value, never inside it. Semantics are provisional until Demerzel's spec states
  them: no verdict is an error, no matching arm stops the pipeline and is recorded in
  `RunOutcome::gates`, an unhandled `C` fails the run.
- **Rope diagram** (`ix_knot::RopeDiagram`) — a knot as a person ties it: up to 8
  ropes, each a list of 2D control points (open with two ends, or closed), plus which
  rope passes over at each crossing (`O`/`U` letters per passage, `alternating`, or by
  height). IX smooths the points, finds the crossings, and lifts the rope where it
  passes in front. Distinct from a **braid word** (`ix_braid`), which is strands
  between two bars — the drawing of a plait, not of a tied knot.
- **Closure** (of a rope diagram) — the mathematical knot a tied knot becomes once each
  open rope's two ends are joined by an arc drawn outside and above everything else.
  A practical knot is not a mathematical knot until closed; the overhand closes into
  the trefoil `3_1`, the figure-eight into `4_1`. The **catalogue** (`ix_knot::catalog`)
  records each entry's closure, and a test checks the drawing's Jones polynomial against
  that of the closure's KnotInfo braid (writhe depends on the drawing, so it is not
  compared).
- **Gauss code** (`ix_knot::GaussCode`) — a knot spelled by its crossings. Walk along
  each rope and write down every crossing as you pass it: `O` if this rope is in front,
  `U` if it is behind, and a number the crossing's two passages share. `U1 O2 U3 O1 U2
  O3` is the overhand; `(…)` marks a closed rope, and `|` separates ropes. The code says
  in which order the crossings come, not where they are, so `ix_knot::gauss::draw`
  finds a drawing (a **rope diagram**) from it. The letters do not fix handedness: a
  drawing and its mirror image share a code, and so do the granny and the reef.
  `closure` names the knot the drawing must close into. Distinct from a **braid word**,
  which fixes the strands' positions, not only the crossings' order.
- **Tying mistake** (`ix_knot::Mistake`) — one crossing of a rope diagram passed the
  wrong way: over where the drawing goes under, or the reverse. `RopeDiagram::mistakes`
  makes each in turn and names what the closure becomes: `same`, `untied` (one rope,
  now the unknot), `apart` (several ropes, now lying separate) or `other`. Topology
  only: a bend still caught may yet slip, which is friction's business.
- **Connected sum** (`a#b`, `ix_knot::catalog::closure_jones`) — two knots cut open
  and joined end to end; its Jones polynomial is the product of theirs. Two overhands
  in a row close into `3_1#3_1` (the granny) or `3_1#m3_1` (the reef). A `closure` or
  `expect jones` may name one; in a `.knot` file a `#` starts a comment only at the
  start of a line or after a space.
- **Linking number** (`RopeDiagram::linking_numbers`) — for two ropes of a closed
  diagram, half the signed sum of the crossings between them. Not zero proves the two
  are caught; zero proves nothing (the Whitehead link). A slip counts as `apart` only
  when every linking number is zero.
- **Knot mechanics** (`ix_knot::Mechanics`, Patil et al., Science 2020) — three counts
  on a drawing whose ropes are oriented toward the ends they are **pulled** from: the
  crossings N, the **twist fluctuation** τ = 1 − (Wr/N)² (1 when as many crossings turn
  each way as the other), and the **circulation** Γ (around each bounded face, edges
  running anticlockwise minus clockwise, over the face's edges). Bends with higher τ,
  then higher Γ, held better in their experiments; no formula combines the three.
  Drawing-dependent, not invariants: the same knot drawn otherwise counts otherwise.
- **`.knot` file** (`ix_knot::knot_file`) — a knot written down to be checked: its
  name, the knot itself (ropes and their points, or a Gauss code) and `expect` lines
  stating what it must be. IX owns the grammar and checks each expectation; a writer,
  human or TARS, only proposes (ADR-0008). An **expectation** is a claim IX can check
  (`expect jones 6_3`: the polynomial is 6_3's), not a name it can prove.

## Conventions

See `CLAUDE.md` for the authoritative build/convention/discipline rules
(Karpathy 4 Rules, Cherny loops, tracer-bullets, the certainty-binding rule).
