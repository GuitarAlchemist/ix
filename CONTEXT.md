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
- **Skill** — a capability exposed to Claude Code / agents (a `.claude/skills/<name>/SKILL.md`, or a
  plugin's `skills/<name>/SKILL.md`, invoked as `<plugin>:<name>`).
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
  verify the plan but does not execute it.
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
  DuckDB SQL is the composition language, **not** IXQL (whose executor runs governance
  pipelines, not analytics, and stays complementary — see
  `docs/adr/0004-duckdb-sql-pipeline-mesh.md` + ADR-0001). Advisory
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

- **Stated folded state** (`ix-origami`) — a FOLD crease pattern with one folded frame whose
  `faceOrders` state the layer order. `ix-origami` checks it and never searches for one; the only
  one it derives is the **replay** of a folding sequence, which is checked like any other. The checks
  treat the sheet as having no thickness. A **thickness `t`** is the sheet's full thickness, in
  crease-pattern units. **Ply** is the most faces stacked at one point, so any stack is at least
  `ply · t` thick. A **panel** is a set of faces joined by flat joints. A **rigid stack** gives
  each panel one height, parallel to the sheet, and exists only if the stated order over
  overlapping panels has no cycle. A rigid-stack refusal means "no stack of parallel panels", not
  "cannot be made" (ADR-0007).
- **Folding sequence** — a sheet and the steps that fold it, in the order a diagram gives them
  (Yoshizawa–Randlett). It is the high level of the origami grammar: replaying it gives a stated
  folded state, the low level, which is what gets checked. A **step** is one operation of the
  sequence, named in the folder's vocabulary. _Avoid_: instructions, recipe, crease pattern (which
  is the result, not the sequence).
- **Landmark** — a named point of the paper (a corner, an edge's midpoint, the centre), used where
  it lies at the moment a step names it, not where it started. A step's **fold line** is built from
  landmarks by a Huzita–Justin axiom (through two points, one point onto another, one line onto
  another); coordinates are only a fallback. _Avoid_: reference point, vertex (a FOLD index).
- **Simple fold** — a fold that carries every layer on one side of its fold line onto the other
  side: in front for a valley, behind for a mountain. It is the only fold whose result needs no
  choice of layers.
- **Moving side** — the side of the fold line that a step carries over; the other side stays where
  it is. It is the side of the landmark or line the step names first (`A to C`: A's side;
  `AB to DC`: AB's side). A fold line through two landmarks names nothing that moves, so that step must name a
  landmark on the moving side, or it is refused. There is no default side.
- **Line** (of a step) — named by two points taken where they lie at that step: two corners (`AC`),
  or any two points joined by `-` (`mAB-mCD`). Its letter order is its direction: a fold that carries
  one line onto another lands the first direction on the second (`AB to AC` is the bisector at A).
  _Avoid_: edge, for a line that is not a side of the paper (an edge is a FOLD index).
- **`.origami` file** — a folding sequence written one statement per line, with `expect` lines.
  IX reads it, replays it and checks the result: a file that does not parse is refused with its
  line, and a file whose expectations fail is answered, not refused (ADR-0010).

## Conventions

See `CLAUDE.md` for the authoritative build/convention/discipline rules
(Karpathy 4 Rules, Cherny loops, tracer-bullets, the certainty-binding rule).
