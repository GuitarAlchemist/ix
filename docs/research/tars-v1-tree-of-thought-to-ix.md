# Research: TARS V1 Tree-of-Thought → IX

**Issue:** [GuitarAlchemist/ix#192](https://github.com/GuitarAlchemist/ix/issues/192) (parent epic #189, depends on #190)
**Status:** Research / extraction only. No production Tree-of-Thought (ToT) engine is proposed or built by this document.
**Companion inventory:** [`tars-v1-exploration-inventory.md`](tars-v1-exploration-inventory.md)
([GuitarAlchemist/ix#190](https://github.com/GuitarAlchemist/ix/issues/190)). Its rebuilt evidence pass removed the earlier bare-filename claims for `ToTReasoner.fs` and
`WoTDerivation.fs`: neither file exists anywhere in the TARS repository. This document therefore audits the
validated V1 Tree-of-Thought and workflow-pipeline sources rather than treating those names as implementations.

---

## 0. Evidence base

Every source claim below was read in the sibling TARS checkout at
`../tars` (branch `refactor/reason-feedback-seam`, commit `9490f73e`) on 2026-08-03.
Paths are relative to the TARS repo root. Nothing here is inferred from file names alone.

Two caveats that shape the whole extraction:

1. **The V1 ToT tree is duplicated and divergent.** `TarsEngine.TreeOfThought/` exists both at the TARS repo
   root and under `v1/parked_legacy/`, and the two copies of `ThoughtNode.fs` define *different* record types
   (root: non-generic `Thought: string`; parked: generic `ThoughtNode<'T>` with `Id`/`Value`). Where the copies
   differ, this document cites the `v1/` path, matching the #190 corpus definition, and flags the divergence.
2. **Most V1 ToT "implementations" are scaffolds, not algorithms.** `BasicTreeOfThought.fs` carries
   `// TODO: Implement real functionality` at lines 52, 77 and 102, and its "analysis" produces hardcoded
   literal scores. The reusable content is the *shape* of the search record, not the reasoning logic.
3. **V1 has no separate Workflow-of-Thought implementation.** A full-history, all-ref filename search found
   neither `ToTReasoner.fs` nor `WoTDerivation.fs`. The Workflow-of-Thought evidence is the staged metascript
   pipeline in §1.4, not a missing engine that IX should wait to recover.

---

## 1. Source list (validated)

### 1.1 F# / C# implementations

| Source | Lines | What it actually contains | Verdict |
|---|---|---|---|
| `v1/parked_legacy/TarsEngine.TreeOfThought/ThoughtNode.fs` | 59 | Generic `ThoughtNode<'T>` = `Id`, `Value`, `Children`, `Score: float`, `Pruned: bool`, `Metadata`. Pure constructors (`create`, `addChild`, `evaluateNode`, `pruneNode`). | Reuse shape |
| `v1/parked_legacy/TarsEngine.TreeOfThought/ThoughtTree.fs` | 118 | Tree container with an id→node `Dictionary`, `findNode`, `getLeafNodes`, `traverseDepthFirst`, `getPathToNode`, `getDepth`, `getNodesAtLevel`, `prune`. **Defective:** `addNode` discards the result of the immutable `addChild` (`|> ignore`, lines 32 and 34), so children never reach the root; the dictionary and the root disagree. | Reuse concepts, not code |
| `v1/parked_legacy/TarsEngine.TreeOfThought/SimpleTreeOfThought.fs` | 249 | Node type without `Id`; `toJson` via `sprintf` (lines 38–52) with **no string escaping**; `Analysis` / `FixGeneration` / `FixApplication` with hardcoded scores; `Selection.selectBestApproach` (lines 229–249). | Reuse concepts |
| `v1/parked_legacy/TarsEngine.TreeOfThought/BasicTreeOfThought.fs` | 125 | Same again, minus `Pruned` and `Metadata`; three `TODO: Implement real functionality` markers. | Non-reusable |
| `v1/parked_legacy/TarsEngine.FSharp/MetascriptToT.fs` | 161 | The most complete V1 version: `MetascriptEvaluationMetrics` = correctness / efficiency / robustness / maintainability / overall (unweighted mean, despite the "weighted average" doc comment), `MetascriptThoughtNode` with `Evaluation: … option`, `depth`, `breadth`. | Reuse shape |
| `v1/parked_legacy/TarsEngine.TreeOfThought/MetascriptToTAlias.fs`, `v1/parked_legacy/TreeOfThought.fs` | — | Alias/compat layers over the above. | Non-reusable |
| `v1/parked_csharp/TarsEngine.CSharp.Adapters/TreeOfThought/*.cs`, `v1/parked_csharp/TarsEngine.FSharp.Adapters/TreeOfThought/*.cs` | — | `ThoughtNodeAdapter`, `ThoughtTreeAdapter`, `ThoughtNodeWrapper`, `FSharpTreeOfThoughtService`, `ITreeOfThoughtService`: F#↔C# marshalling only. | Non-reusable |
| `v1/parked_csharp/Legacy_CSharp_Projects/TarsEngine/Services/TreeOfThought/*.cs` (6 services) | — | `Basic`/`Simple`/`Demo`/`Enhanced`/`Metascript`/`TreeOfThoughtService`: DI-registered service wrappers. | Non-reusable |
| `v1/parked_csharp/Legacy_CSharp_Projects/TarsCli/Commands/*TreeOfThought*.cs` (6 commands) | — | CLI surface (`tars metascript-tot generate|validate|execute|analyze|pipeline`). | TARS runtime |

### 1.2 Tests

| Source | Status |
|---|---|
| `v1/parked_legacy/TarsEngine.FSharp.Tests/TreeOfThought/ThoughtNodeTests.cs` | xUnit facts over `createNode` / `addChild` / `evaluateNode`. **Orphaned:** they `using TarsEngine.FSharp.Core.TreeOfThought`, and no file in the checkout declares that namespace (verified by grep across `**/*.fs`). They cannot be run as-is. |
| `v1/parked_legacy/TarsEngine.FSharp.Tests/TreeOfThought/ThoughtTreeTests.cs` | Same namespace, same orphan status. |
| `v1/parked_csharp/Legacy_CSharp_Projects/TarsEngine.Tests/TreeOfThought/{Simple,Metascript}TreeOfThoughtTests.cs` | Legacy-parked; assert on the hardcoded-score scaffolds, so they pin fixtures rather than behaviour. |

The test *intent* (constructor contracts, immutability of `addChild`) is worth reproducing. The test *code* is not portable.

### 1.3 Specifications and plans

| Source | Lines | Why it matters |
|---|---|---|
| `v1/Implementation-Plan-F#-Tree-of-Thought.md` | 525 | The only V1 artifact that specifies the algorithms: `EvaluationMetrics` (line 101) with three metric quads, `Branching.generateBranches` with a `branchingFactor`, `Pruning.beamSearch root beamWidth scoreNodeFn`, threshold pruning, pruned-node tracking, and five `Selection` strategies (best-first, diversity, confidence, hybrid, top-N). Largely aspirational — several bodies are `// Implementation`. |
| `v1/docs/MetascriptTreeOfThought.md` | 187 | Component map of the metascript ToT pipeline (generation → validation → execution → result analysis) and the CLI contract. |
| `v1/TODOs/TODOs-Tree-of-Thought-Auto-Improvement-Pipeline.md` | 158 | Assigns a distinct metric quad per pipeline stage: analysis = relevance/precision/impact/confidence; generation = correctness/robustness/elegance/maintainability; application = safety/reliability/traceability/reversibility. Each stage repeats "pruning strategy using beam search". |
| `v1/TODOs/TODOs-Tree-of-Thought-Auto-Improvement-Pipeline-Detailed.md` | 475 | Task-level expansion of the same; visualization appears only as unchecked TODOs (lines 97, 127, 192, 222, 282, 312, 451). |
| `v1/TODOs/TODOs-Tree-of-Thought-F#-Implementation.md`, `v1/TODOs/TODOs-Tree-of-Thought-Implementation.md` | 389 / 126 | Sequencing plans; no algorithm content beyond the above. |
| `v1/README-Tree-of-Thought.md`, `v1/README-Tree-of-Thought-Demo.md`, `v1/README-Demo-Tree-of-Thought.md` | 104 / 124 / 75 | Demo walkthroughs of the CLI. |

### 1.4 Metascripts, scripts and reports

| Source | Why it matters |
|---|---|
| `v1/parked_legacy/Metascripts/TreeOfThought/{CodeImprovement,GenerateImprovements,ApplyImprovements,AutoImprovement}.tars` (+ `_unified.trsx` twins) | The pipeline shape as executable-ish DSL: `DESCRIBE` / `VARIABLE` / `FUNCTION` with embedded `CSHARP` blocks that build the thought tree as dynamic JSON and emit `{correctness, efficiency, robustness, maintainability, overall}` (e.g. `CodeImprovement.tars:69-70`). |
| `v1/parked_csharp/Legacy_CSharp_Projects/TarsCli/Metascripts/Core/tree_of_thought_generator.tars` | Metascript that *generates* ToT metascripts. |
| `v1/src/{demo,run}_tree_of_thought.ps1`, `v1/src/update_tree_of_thought_extensions*.ps1` | Demo drivers and codegen patchers. |
| `v1/tree_of_thought_generation_report.md` | The closest thing V1 has to a dataset: a run report with an explicit parameter block (branching factor 3, max depth 3, beam width 2, metrics *relevance/feasibility/impact/novelty*, pruning `beam_search`, lines 11–15) and embedded JSON thought trees. Timestamps and scores are round, hand-authored values — a narrative artifact, not a measurement. |
| `v1/tree_of_thought_implementation_summary.md` | 56-line status summary. |

### 1.5 Explicitly out of scope

`v1/parked_csharp/Legacy_CSharp_Projects/TarsEngine/Consciousness/**` (`ThoughtProcess.cs`, `SpontaneousThought.cs`, `ThoughtModel.cs`, `RandomThoughtGeneration.cs`) and
`v1/parked_legacy/TarsEngine.FSharp.Reasoning/ChainOfThoughtEngine.fs` match on the word "thought" but model
free-running introspection, not bounded search. They are also the artifacts closest to storing private model
reasoning, which §6 forbids. Excluded.

---

## 2. What the sources actually establish

**Finding 1 — the node schema is unstable across V1.** Five incompatible `ThoughtNode` shapes coexist:
generic `Id`/`Value` (parked `ThoughtNode.fs`); non-generic `Thought` + `Score` + `Metadata` (root copy);
`Thought` + `Score` + `Pruned` without metadata (`SimpleTreeOfThought.fs`); `Thought` + `Score` only
(`BasicTreeOfThought.fs`); and `Thought` + `Evaluation: EvaluationMetrics option` (`MetascriptToT.fs`, and the
plan at line 101). *Any* IX extraction must pick one and make the others convert into it.

**Finding 2 — the metric vocabulary is unstable too.** Three different quads appear: analysis/generation/application
quads in the plan and pipeline TODOs; correctness/efficiency/robustness/maintainability in `MetascriptToT.fs` and
the `.tars` metascripts; relevance/feasibility/impact/novelty in the generation report. IX should own an
*open* metric map with one reserved aggregate, not a closed enum of TARS's stage vocabularies.

**Finding 3 — the same name means two things.** `breadth` is *immediate child count* in
`MetascriptToT.fs:70` and *total node count* in the plan at line 78. A named, tested IX metric ends this.

**Finding 4 — selection is greedy, not argmax.** `SimpleTreeOfThought.fs:229-249` descends only into the
single highest-scoring child and stops as soon as that child does not beat its parent
(`| Some child when child.Score > node.Score ->`, line 239). A high-scoring grandchild behind a mediocre parent
is unreachable. This is defensible as *greedy best-path*, but V1 documents it as "selects the best approach".
Two different, individually useful primitives are conflated.

**Finding 5 — there is no benchmark dataset.** No V1 ToT TODO or plan document mentions the word "benchmark"
(verified by grep). The generation report's trees are hand-authored. **IX must author its fixtures fresh**;
there is nothing to port.

**Finding 6 — the search record is the durable idea.** Across every V1 variant, the surviving concept is:
*record what was proposed, what it scored, what was kept, what was pruned, and why*. That is a data-and-algorithm
artifact — exactly IX's half of the boundary in §3.

### 2.1 Workflow-of-Thought disposition

The validated V1 workflow sources add no distinct reasoning algorithm beyond the ToT search record. They define
an ordered pipeline — generate, validate, execute, analyse — with dependencies, per-stage evaluation, and a
record of stage completion. Those structural operations already belong to `ix-pipeline::dag::Dag<N>`, which
provides cycle rejection, topological ordering, parallel execution levels, and critical-path analysis. TARS owns
the stage implementations, model and tool calls, retries, side effects, and rollback; IX owns only the generic
deterministic DAG operations.

**Disposition:** do not create an IX Workflow-of-Thought engine, grammar, or second workflow graph. A future TARS
interop story may serialize its declared workflow steps into an `ix-pipeline` DAG for validation or analysis.
That artifact remains separate from `ThoughtTrace`: the DAG records *what executes and in what dependency order*;
the thought trace records *which bounded search candidates were proposed, scored, kept, or pruned*. This is the
reusable Workflow-of-Thought result required by #192: reuse the existing IX DAG primitives and extract no new
V1-specific primitive.

---

## 3. Boundary: TARS runtime vs IX algorithms

This is the load-bearing split. It is not negotiable per-feature; it is the reason IX can own any of this at all.

| Concern | Owner | Rationale |
|---|---|---|
| Producing candidate thoughts (LLM calls, prompts, model/provider choice, temperature, retries) | **TARS** | Requires a model runtime and a budget. IX has neither and must not acquire them here. |
| Domain judgement — deciding *that* a fix is correct or a hypothesis plausible | **TARS** | Semantic, model- or human-sourced. IX cannot verify it. |
| Metascript execution, F# compilation, file mutation, rollback | **TARS** | Runtime side-effects, governed on the TARS side. |
| Node/edge/tree schema and its validation | **IX** | Pure data. Deterministic, testable, versionable. |
| Expansion, pruning and selection *strategies* as pure functions over an injected candidate generator | **IX** | Algorithms; already IX's domain (`ix-search`). |
| Deterministic tie-breaking, seeded ordering, reproducibility | **IX** | IX's stated convention (seeded RNG, `f64`). |
| Graph metrics over a recorded trace (depth, branching, kept/pruned ratio, path extraction) | **IX** | Analysis over a finished artifact. |
| Persistence, replay and diff of a trace | **IX** | JSON-on-disk, the ecosystem's canonical handoff. |

**Operational form of the boundary:** IX exposes a `Scorer`/`Expander` seam. TARS implements it (by calling a
model); IX's tests implement it with a fixture table. IX code never opens a socket and never calls an LLM.
An IX ToT run with a fixture expander must be reproducible byte-for-byte offline — that is the acceptance test
for the boundary holding.

---

## 4. Candidate IX data model

Refined from the #223 sketch to satisfy Findings 1–3 and 6. Field names are proposals, not a frozen contract;
no `docs/contracts/` entry is created by this story.

### `ThoughtNode`

| Field | Type | Notes |
|---|---|---|
| `id` | `String` | Unique within a trace. Required — the missing `Id` is what makes `SimpleTreeOfThought.fs` untraceable. |
| `parent` | `Option<String>` | Single parent. Roots carry `None`. Makes the parent/child relation representable exactly once (Finding 1's divergence came from storing it twice). |
| `kind` | `String` | Open vocabulary: `observation`, `hypothesis`, `action`, `candidate`, `evaluation`. Open, because TARS stages differ. |
| `content` | `String` | The **declared** candidate — a proposed action, hypothesis or artifact. Not a model's private reasoning (§6). |
| `metrics` | `Map<String, f64>` | Open metric map (Finding 2). Producers write their own quad. |
| `score` | `Option<f64>` | The single aggregate used by search. Explicitly *not* derived by IX: if a producer supplies both, IX validates but does not recompute. |
| `disposition` | `Enum` | `Open` \| `Kept` \| `Pruned`. Supersedes V1's bare `Pruned: bool`; a node dropped by beam width and a node dropped by threshold are both "pruned" but distinguishable via `prune_reason`. |
| `prune_reason` | `Option<String>` | `beam_width` \| `threshold`. Required exactly when `disposition = Pruned`; this is the kept/dropped manifest V1 never wrote down. |
| `expansion` | `Enum` | `Expandable` \| `Terminal`. Independent from selection disposition, so a kept leaf is representable as `Kept + Terminal`. |
| `terminal_reason` | `Option<String>` | `depth_limit` \| `expander_exhausted`. Required exactly when `expansion = Terminal`; termination is not pruning. |
| `depth` | `u32` | Denormalized for cheap level queries; validated against `parent` chain. |
| `provenance` | `Map<String, Value>` | Producer id, timestamps, model name, token counts. Free-form; IX does not interpret it. |

Edges are implicit in `parent` for the tree case. A separate `ThoughtEdge { source, target, edge_type, weight }`
remains available for cross-links (`supports`, `refutes`, `refines`) — but a v1 IX prototype should ship the
tree only and add cross-links when a real consumer needs them.

### `ThoughtTrace`

| Field | Type | Notes |
|---|---|---|
| `trace_id` | `String` | |
| `schema_version` | `u32` | `1`. |
| `roots` | `Vec<String>` | Forest permitted; single root is the normal case. |
| `nodes` | `Vec<ThoughtNode>` | |
| `params` | `SearchParams` | `branching_factor`, `max_depth`, `beam_width`, `score_threshold`, `seed`, `strategy`. Directly recovered from `tree_of_thought_generation_report.md:11-15`. |
| `status` | `Enum` | `Running` \| `Completed` \| `Halted`. |

**Relation to existing IX artifacts.** This is *not* a second traceability tree.
`docs/contracts/2026-05-24-traceability-tree.contract.md` describes a static, content-addressed, multi-resolution
summary of code/product/process that is expected to be stable and re-derivable. A thought trace is the opposite:
an episodic record of one bounded search, valid only for the run that produced it. They may cross-reference;
they must not be merged.

---

## 5. Testable IX primitives

Each row is a primitive with an executable oracle. "Oracle" here means a test that fails loudly when the
behaviour is wrong — not an LLM judgement.

| # | Primitive | Proposed signature (sketch) | Oracle | Testability |
|---|---|---|---|---|
| P1 | Trace validation | `validate(&ThoughtTrace) -> Result<(), Vec<TraceError>>` | Property tests: every `parent` resolves; no cycles; `depth` matches the parent chain; roots have no parent; ids are unique; prune and terminal reasons match their respective enums. Fixture set of malformed traces must each produce the expected error. | High |
| P2 | Deterministic beam step | `beam_step(frontier, beam_width, tie_break) -> (kept, pruned)` | Golden test: equal-score candidates resolve by an explicit documented tie-break (id-lexicographic), so two runs on the same input are byte-identical. V1 specifies no tie-break. | High |
| P3 | Threshold pruning | `prune_below(nodes, threshold) -> Vec<Node>` | Monotonicity property: raising the threshold never increases the kept set; the best surviving score never increases after pruning. | High |
| P4 | `select_best_global` | `argmax` over non-pruned nodes | Differential test against P5 on a fixture where they disagree by construction (root 0.0 → child 0.4 → grandchild 0.95, sibling 0.5). | High |
| P5 | `select_best_greedy_path` | Faithful port of `SimpleTreeOfThought.fs:229-249` semantics | Same fixture; asserts it returns the 0.5 sibling, documenting the V1 behaviour instead of silently "fixing" it. | High |
| P6 | Trace metrics | `depth`, `size`, `max_branching`, `kept_ratio`, `pruned_by_reason` | Hand-computed values on the fixtures. Resolves Finding 3 by naming `size` and `max_branching` separately. | High |
| P7 | JSON round-trip | `serde` in/out | Round-trip property, including content with quotes, newlines and non-ASCII — the exact class of input `SimpleTreeOfThought.fs:38-52` corrupts. | High |
| P8 | Path extraction | `path_to(&trace, id) -> Option<Vec<&Node>>` | Fixture assertions incl. missing-id and multi-root cases. | High |
| P9 | Expander seam | `trait Expander { fn expand(&self, node) -> Vec<Candidate> }` | A fixture-table expander makes a whole run deterministic and offline; the test *is* the boundary proof from §3. | High |

Not proposed: any primitive that scores content semantically, any LLM-backed evaluator, any MCP tool. Those are
TARS-side or a later story.

**Cost tier: `free-local` ($0, no network, no model).** Every oracle above is a `cargo test` over in-repo JSON.
This matches issue #192's declared budget (`tier: free-local`, `max_cost_usd: 0`, `max_runner_minutes: 30`) and
is only achievable *because* of the §3 boundary. If a future story needs a real expander, the model cost lands
on the TARS side of the seam and IX CI stays free.

---

## 6. Chain-of-thought discipline (hard constraint)

Issue #192's non-goals are explicit: *"No hidden chain-of-thought storage requirement; store structured
reasoning artifacts, not private model reasoning."* Concretely, for any IX ToT work:

1. **What may be stored:** the search record — declared candidate content (a proposed hypothesis, action or
   patch), its metrics, disposition, expansion status and reasons, and the search parameters. These are outputs
   a producer deliberately emits as artifacts.
2. **What may not be stored:** raw model reasoning traces, provider "thinking" blocks, or any text a model
   produced for itself rather than as a declared candidate. No IX field exists to receive them, and none should
   be added: `content` is specified as the declared candidate, and `provenance` is metadata only.
3. **Enforcement is producer-side.** IX validates structure, not semantics — it cannot detect that a producer
   stuffed a reasoning dump into `content`. So the rule belongs in the producer contract (TARS side), and any
   future IX trace-ingest surface must state it at the boundary rather than pretend to police it.
4. **Consequence for the V1 corpus:** the `Consciousness/**` and `ChainOfThoughtEngine.fs` artifacts (§1.5) are
   excluded on this ground as well as on relevance.

---

## 7. Deferred and non-reusable

| Item | Disposition | Why |
|---|---|---|
| F# compiler service / script execution (`TODOs-…-Pipeline.md` §1) | **Non-reusable** | .NET runtime concern. IX is pure Rust. |
| Metascript DSL (`.tars` / `.trsx`, `DESCRIBE`/`VARIABLE`/`FUNCTION`, embedded `CSHARP`) | **Non-reusable in IX** | TARS's execution substrate. IX's equivalent seam is `ix-pipeline`, which already exists. |
| C#↔F# adapters, wrappers, DI services (§1.1, 10+ files) | **Non-reusable** | Pure marshalling for a language boundary IX does not have. |
| `TarsCli` ToT commands (6 files) and `tars metascript-tot` CLI | **TARS runtime** | Belongs to TARS by §3. |
| Auto-apply fix pipeline (analyze → generate → **apply** → verify) | **Deferred, gated** | #192 non-goal: no generated fix is accepted without tests. IX may score and rank fix candidates; applying them is a governed TARS action. |
| Hardcoded-score analysis/fix/apply scaffolds (`Basic`/`Simple`ToT) | **Non-reusable** | No algorithm inside; three `TODO: Implement real functionality`. |
| `Consciousness/**`, `ChainOfThoughtEngine.fs` | **Excluded** | §1.5 and §6. |
| Visualization / HTML demos | **Deferred** | Only unchecked TODOs exist upstream; a JSON trace + `ix-graph` metrics is a better substrate than porting a demo. Revisit when a consumer (ga dashboard) asks. |
| `tree_of_thought_generation_report.md` as a dataset | **Non-reusable as data** | Hand-authored values (Finding 5). Reusable only as a *parameter vocabulary* (§4 `SearchParams`). |
| Orphaned xUnit ToT tests | **Non-reusable as code** | Reference a namespace absent from the checkout. Their intent is re-expressed in §5. |
| Cross-link edges (`supports` / `refutes` / `refines`) | **Deferred** | No consumer yet. IX already has `ix-assumption-graph` for argumentation-shaped claims; adding a second one here would duplicate it. |

---

## 8. Prototype task suggestions

Ordered smallest-first. Only Phase 0 is proposed for the next story; Phases 1 and 2 are sequels, listed so the
sequencing is visible.

### Phase 0 — tracer bullet: `ix-thought`, offline, one vertical slice (recommended next story)

The smallest slice that touches every layer: **fixture → parse → validate → search (fixture expander) → trace →
serialize → assert**. No LLM, no MCP tool, no network.

Scope:
- New crate `crates/ix-thought`, **`internal` tier** in `crate-maturity.toml`.
- Types from §4 (`ThoughtNode`, `ThoughtTrace`, `SearchParams`) with `serde`.
- Primitives P1–P7 and P9 from §5 (validation, beam step, threshold prune, global/greedy selection,
  metrics, round-trip, and the expander seam). P8 path extraction remains a later additive utility.
- Five fixtures authored fresh (§9), plus golden expected traces.
- No MCP tool, so `crates/ix-agent/tests/parity.rs` stays untouched.

**Why a new crate rather than extending `ix-search`.** `ix-search` already has the search algorithms
(`local.rs::beam_search` at line 98, `mcts.rs`, `astar.rs`, `qstar.rs`), and reusing them is right — but two
things block putting ToT *in* it. First, `ix-search` is `stable` in `crate-maturity.toml:22`, and
`.github/workflows/stable-surface.yml` diffs a per-crate public-API hash between base and PR, treating a
changed hash on a `stable`-tier crate as blocking — so adding a public ToT surface there forces a version
decision on a stable crate for research code. Second, `beam_search` deliberately returns only the winning state
(`local.rs:98-124`) — it discards the very thing ToT exists to keep, the record of what was expanded and
pruned. `ix-thought` should therefore *depend on* `ix-search`'s `LocalSearchState` shape conceptually while
owning the recorded-trace variant. Whether it literally reuses `beam_search` or reimplements a recording
variant is a Phase 0 implementation decision, to be justified in the PR.

Success criteria (verifiable): `cargo test -p ix-thought` green offline; two runs of the same fixture produce
byte-identical JSON; the P4/P5 differential test demonstrably distinguishes global argmax from V1 greedy descent.

Estimated size: one crate, ~400–600 lines including tests. Cost tier `free-local`.

### Phase 1 — TARS interop check (only after Phase 0)

Have TARS emit one real trace against the §4 schema and have IX ingest it unchanged. This is the honest test of
whether the schema survives contact with a producer; it should precede any contract freeze. Requires a TARS-side
change, so it is cross-repo and needs coordination per CLAUDE.md.

### Phase 2 — analysis surface (only after Phase 1)

`ix-graph` metrics over accumulated traces (branching distribution, prune-reason histogram, score calibration).
Deferred until traces actually accumulate — building the dashboard before the data exists is the failure mode
this repo already names.

**Not proposed:** an MCP `ix_thought_*` tool, a `docs/contracts/` entry, or a ToT-driven auto-fix loop. Each
needs a real consumer first.

---

## 9. Benchmark fixture plan

Because V1 ships no dataset (Finding 5), fixtures are authored in-repo. They live with the crate
(`crates/ix-thought/tests/fixtures/*.json`), version with the code, and are small enough to read in a diff.
Run artifacts, if a later story produces any, follow the `state/` convention
(`state/thought-traces/{date}-{short-description}.trace.json`).

| Fixture | Shape | What it pins |
|---|---|---|
| `analysis-tree.json` | Root observation → 3 hypotheses (0.8 / 0.4 / 0.9) → 2 actions under the best | The V1 code-analysis tree, re-authored with the §4 schema. Pins P1 and P6. |
| `fix-selection.json` | One problem node → 3 candidate fixes with distinct scores, `beam_width: 2` | Beam keep/prune with a recorded manifest. Pins P2, P3. |
| `greedy-vs-global.json` | Root 0.0 → child A 0.4 → grandchild 0.95; sibling B 0.5 | The P4/P5 disagreement. Pins the Finding-4 distinction. |
| `degenerate.json` | Empty node list; a cycle; a dangling `parent`; duplicate ids; invalid reason/enumeration pairs | Each must produce a specific `TraceError`. Pins P1's negative path. |
| `escaping.json` | Content with `"`, `\n`, `é`, and a `{` | Round-trip fidelity — the class `SimpleTreeOfThought.fs:38-52` corrupts. Pins P7. |

Fixture rules: hand-authored and reviewed (no generated data); every score is a fixed literal (no RNG); the
expected outputs are golden files, so a behaviour change shows up as a reviewable diff; total fixture bytes stay
under a few KB so a human can audit them.

---

## 10. Acceptance-criteria trace

| #192 acceptance criterion | Where satisfied |
|---|---|
| Reusable Tree/Workflow-of-Thought primitives are identified | §2 (findings and WoT disposition), §4 (data model), §5 (P1–P9) |
| At least one minimal IX prototype is proposed | §8 Phase 0 (`ix-thought` tracer bullet) |
| Testability and cost tier are documented | §5 (per-primitive oracles, `free-local`), §9 (fixtures) |
| TARS runtime responsibilities remain separate from IX algorithm responsibilities | §3, restated operationally in §5 P9 and §7 |

Evidence artifacts named by the issue: `tot_extraction_doc` = this document; `candidate_data_model` = §4;
`benchmark_fixture_plan` = §9. The missing-filename correction is independently recorded by the rebuilt
GuitarAlchemist/ix#190 inventory; §2.1 is the resulting evidence-backed Workflow-of-Thought disposition.
