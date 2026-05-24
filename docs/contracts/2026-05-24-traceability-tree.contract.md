# Traceability Tree — Cross-Repo Contract

**Version:** 0.1.0 (draft, Phase 0 of 6)
**Schema version:** 1
**Status:** Draft (Phase 0 of `traceability-tree` campaign, 2026-05-24)
**Producers:** future `ix-traceability-tree` crate, future ga BACKLOG.md parser exporter
**Consumers:** future `ix_tree_query` / `ix_tree_zoom` / `ix_tree_drift` / `ix_tree_rebuild` MCP tools, future ga dashboard zoom UI, `grade-last-pr` skill, post-mortem ingestion
**Schema file:** `docs/contracts/traceability-tree.schema.json`
**Related contracts:** `docs/contracts/2026-05-24-ai-annotation.contract.md` (PR #54)

---

## 1. Why This Contract Exists

The ecosystem now produces three roughly parallel hierarchies that no one can hold in their head at the same time:

1. The **product hierarchy** — `BACKLOG.md` in ga: epic → epic-subsection → task → subtask, parsed by a TypeScript reader into `BacklogEpic` / `EpicSubSection` / `BacklogItem`.
2. The **code hierarchy** — workspace → crate → module → file → symbol, with line-local `@ai:` annotations (PR #54, just shipped: `2026-05-24-ai-annotation.contract.md`) attaching invariants and contracts at the leaf.
3. The **process hierarchy** — test plans, QA sessions, demos, post-mortems, dated `docs/plans/` and `docs/solutions/` — currently a flat set of files connected only by date prefix and grep.

Agents (and operators) re-derive the mapping between these three on every task. A bug report cites a symbol; the agent has to find the file, walk to the module README, climb to the crate-level CONTEXT.md, locate the matching task in BACKLOG, find the test plan, find the post-mortem, find the `@ai:invariant` that was supposed to prevent this. Every climb costs context and silently drops detail. Every drop is a future hallucination.

The **traceability tree** is one Merkle-style summary tree that unifies all three hierarchies — code, product, process — under a single semantic-zoom contract. The leaves are raw (a symbol's source span, an annotation, a single task line, a single test). Each internal node is a level-appropriate summary of its children, with an explicit `kept` / `dropped` manifest so an operator (or the next agent) can see what was compressed away without having to drill down to find out. AI annotations are pinned: every `@ai:invariant` and `@ai:contract` at any leaf is hoisted into every ancestor's `kept` list — they are the *low-frequency coefficients* of the wavelet decomposition and survive every zoom-out. This PR ships the contract; a Phase-1 `ix-traceability-tree` crate will materialize trees against it.

---

## 2. Mental Model

Three converging metaphors. None alone is right; the contract sits at their intersection.

### 2.1 Wavelet-style multi-resolution

A traceability tree is a discrete wavelet decomposition over the workspace. Each level is a coarser approximation of the level below, and every level can be **reconstructed exactly** by walking back down to the leaves. The `kept` field carries the low-frequency coefficients that survive zoom-out (invariants, contracts, public API shapes, named architectural decisions). The `dropped` field is the explicit accounting of the high-frequency detail compressed away. *Lossy at this level, lossless from the root* — that is the trick.

### 2.2 Merkle-tree provenance

Every node is content-addressed: `id = blake3(canonical(content || sorted(children_hashes)))`. A child re-summarization changes the child's hash, which changes the parent's hash, which propagates to the root. This makes drift cheap to detect (one hash compare per parent) and gives every summary cryptographic provenance — the hash is the receipt that "this summary is what you got from these children at this moment."

### 2.3 Hierarchical GraphRAG / Tree Index

Microsoft's GraphRAG (<https://microsoft.github.io/graphrag/>) shows that hierarchical summaries — community → super-community — make global queries cheap on graphs that would otherwise require flat retrieval over the whole corpus. LlamaIndex's Tree Index (<https://docs.llamaindex.ai/en/stable/api_reference/indices/tree/>) demonstrates the same pattern for documents: query goes to the root, the root routes to the relevant child, the child routes further down. The traceability tree is the GraphRAG / Tree Index pattern applied to the code-plus-process-plus-product graph, with the additional discipline of *provenance* (Merkle) and *signal preservation* (kept/dropped manifests).

### 2.4 Related work and prior art

- **Microsoft GraphRAG** — <https://github.com/microsoft/graphrag> — community summarization, hierarchical query routing.
- **LlamaIndex Tree Index** — <https://docs.llamaindex.ai/en/stable/api_reference/indices/tree/> — recursive document summaries for question-answering.
- **Aider repo-map** — <https://aider.chat/docs/repomap.html> — code-graph extraction for LLM context, no hierarchy / no provenance.
- **OpenTelemetry tree-of-spans** — <https://opentelemetry.io/docs/specs/otel/trace/api/> — runtime traces as trees; we borrow the parent/child + cross-link shape, applied statically.
- **ix-bracelet `grothendieck_delta`** — `crates/ix-bracelet/src/grothendieck.rs` — signed Z⁶ delta between musical PC-sets via interval-class vectors. The *idea* (cheap signed delta over a structured fingerprint) is what we want for code-content drift; the *implementation* is music-theory-specific and not directly reusable. A future general `ix-grothendieck-delta` crate would close this gap.
- **ix-context::cache** — `crates/ix-context/src/cache.rs` — content-hash-keyed cache with path-based invalidation; the existing primitive most analogous to the Merkle layer.
- **ix-io::watcher** — `crates/ix-io/src/watcher.rs` — file-mtime watcher that emits change events; the existing trigger most analogous to the drift-alarm layer.

---

## 3. Node Format

The full JSON Schema is in `docs/contracts/traceability-tree.schema.json`. Field-by-field commentary:

| field | required | meaning |
|---|---|---|
| `id` | yes | Content-addressed blake3 hash over canonical(`summary` ⊕ `kind` ⊕ `level` ⊕ `provenance.source_path` ⊕ sorted(`children`)). Determines node identity. |
| `schema_version` | yes | `const: 1`. Bumped only on breaking field rename/removal. |
| `level` | yes | Integer `0..=6`. Section 4 defines the semantics. |
| `kind` | yes | One of the 13-entry enum (Section 4). Aligns with ga's `BacklogEpic` / `EpicSubSection` / `BacklogItem` as a *superset* — the contract owns more kinds because it also covers code and process artifacts. |
| `summary` | yes | Self-contained markdown body at this level. Must be readable without the children. |
| `children` | no | Ordered list of child node `id`s. Empty for leaves. |
| `parent` | no | Single parent `id` or `null` for root. Trees are forests-of-one in v1. |
| `cross_refs` | no | Non-tree edges. Example: a `task` node cross-references the `test_unit` that proves it; a `file` cross-references the `post_mortem` whose root cause lives in this file. Cross-refs are typed by the kinds at both ends — schema does not enforce a typed-edge enum in v1. |
| `provenance.source_path` | yes | Repo-relative path. For aggregate nodes, the directory or BACKLOG section path. |
| `provenance.source_line_range` | no | `[start, end]` for leaf nodes that point at a span. |
| `provenance.source_mtime` | no | Filesystem mtime at summarization time — feeds the drift detector. |
| `provenance.summarizer_model` | no | `claude-opus-4-7@2026-05-24` style — captures both model and date so a model swap shows up as drift. |
| `provenance.summarized_at` | no | ISO-8601 timestamp of the summarization call. |
| `fidelity` | no | 0.0 – 1.0 self-report from the summarizer. Useful as a soft signal; not load-bearing. Leaves are 1.0 by definition. |
| `kept` | no | Explicit list of facts preserved at this level. AI annotations always go here (Section 5). |
| `dropped` | no | Explicit list of facts compressed away. Each entry should be a one-line summary readable on its own. |
| `stale` | no | Set by the sync engine when the underlying children have changed but this node has not yet been re-summarized. |
| `stale_reason` | no | Free text: which child diverged, by what hash delta, when. |
| `ai_claims` | no | List of `@ai:` annotation ids (per `ai-annotation.schema.json`) that live in the subtree rooted here. Promoted upward by the build. |
| `contract_invariants` | no | Free-text invariants this node guarantees to its parent. The parent's `kept` list must include every entry here. |

Two non-obvious rules:

1. **`id` is recursive**: it depends on `children`, which are themselves hashes. Building bottom-up is required.
2. **`kept` is a contract with the parent**, not a hint to the reader. The sync engine treats a divergence between a child's `kept` and the parent's `kept` as an alarm condition (Section 6.3).

---

## 4. Level Semantics

Seven levels, L0 (raw) to L6 (epic). The compression rate at each level is a target, not a hard constraint — fidelity matters more than ratio.

| level | name | kind examples | compression target | what KIND of summary lives here |
|---|---|---|---|---|
| **L0** | Raw leaf | `symbol`, `test_unit`, `@ai:` annotation row, single BACKLOG line | 1.0× (no compression) | Verbatim source span or annotation. No LLM. |
| **L1** | Local | `symbol` (function-with-docstring), small `test_integration` | ~3–5× | One-paragraph LLM summary of the leaf in context. Names, signatures, and `@ai:invariant` claims are preserved. |
| **L2** | Aggregate | `file`, `test_plan`, `qa_session` | ~5–10× | Per-file or per-session summary. Lists the L1 children's invariants. |
| **L3** | Composite | `module`, `subsystem` | ~10–20× | Cross-file summary. The boundary at which a human reviewer can still hold the entire summary in working memory. |
| **L4** | Subsystem | `subsystem` (crate-level), `demo` | ~20–50× | Crate-level overview. Architectural shape, public surface, named invariants, integration points. |
| **L5** | Story | `story`, `post_mortem` | ~50–100× | A unit of delivered or planned work spanning multiple subsystems. Mirrors ga `EpicSubSection` granularity. |
| **L6** | Epic | `epic` | ~100×+ | Top-level objective; mirrors ga `BacklogEpic`. The root of a workspace tree. |

Levels do not have to be fully populated — a small workspace may have empty L5/L6. A node's `level` is **declarative**: it tells the consumer what compression contract to expect, not how the node was built.

A node MUST NOT have children at a strictly higher level than itself, and MUST NOT have children at a level more than two below itself (so an L4 can have L2 or L3 children, never L1 directly — force the intermediate aggregation).

---

## 5. Signal Preservation Contract

This is the load-bearing idea of the contract.

A naive recursive summary loses information silently. A reader at L4 sees a polished paragraph and has no way to know that "thread-safety of `Foo::bar` under MIRI" was dropped two levels below. The traceability tree forbids this. Every summary above L0 MUST:

1. Carry an explicit `kept` array of facts preserved at this level. Each entry is a short string the reader can take to the bank.
2. Carry an explicit `dropped` array of facts compressed away. Each entry must be drilled-down-reachable — the reader can follow `children` until a descendant's `summary` covers that fact in full.
3. Promote **every** `@ai:invariant` and `@ai:contract` annotation in the subtree into its own `kept` list, verbatim or as a one-line rephrase that preserves the truth-value tag. AI annotations are the wavelet's low-frequency coefficients: they are short, structured, named, and operator-relevant. They survive every zoom-out.

The contract guarantees: **a fact named in any descendant's `kept` is named in every ancestor's `kept` up to the root.** A fact in `dropped` at level N is reachable in `summary` or `kept` at some level < N. Nothing disappears.

The reconciler (Phase 1) will enforce this with a structural check. A summarizer that drops an `@ai:invariant` without recording it explicitly is a build failure, not a soft warning.

---

## 6. Sync Protocol

Three layers, ordered cheapest-first.

### 6.1 Merkle hash propagation (free)

On any file change, walk up from the affected leaf to the root, recomputing `id`s. Any node whose recomputed `id` differs from its stored `id` is marked `stale`. This is O(depth) per file change, runs on every git commit hook, and costs no LLM tokens.

### 6.2 Selective re-summarization (LLM cost)

A `stale` node is re-summarized by the L-appropriate prompt. Re-summarization is **bottom-up and lazy** — only when a `stale` node is queried (or on a scheduled rebuild) does the LLM call happen. The Phase-1 builder will emit a cost telemetry line per re-summarization so the operator can see "$3.40 spent re-summarizing 17 L1 nodes after PR #56."

### 6.3 Drift alarm (algedonic signal)

The most interesting layer. When a re-summarization changes a node's `kept` set in a way that violates the parent's `contract_invariants` (i.e., a fact the parent promised has been silently dropped by the new child summary), the sync engine raises a *drift alarm*: a structured event written to `state/quality/traceability-drift.jsonl` and surfaced to the operator via the ga dashboard. This is the algedonic channel — pain signal back to the operator when the tree's invariants stop holding. The parent is not auto-rebuilt; the operator (or a follow-up agent) decides whether the drift is acceptable (update the parent's contract) or whether the child has regressed (revert or fix the underlying code).

---

## 7. LLM Orchestration

For v1, the Phase-1 builder shells out to the `claude` CLI per summary call. Rationale: zero new dependencies, easy to swap models, no API key handling in the crate, easy to cap per-invocation cost via the CLI's own controls. The summarization prompt is checked into the crate's `prompts/` directory so the prompt history is git-tracked alongside the schema.

Future options, documented but not adopted in v1:

- **`async-anthropic`** — <https://docs.rs/async-anthropic/> — direct API client. Justified once the cost telemetry shows per-rebuild costs that warrant a tighter feedback loop than `claude` CLI provides.
- **Hari MCP cache** — `hari_record_observation` + `hari_query_belief` — store every summary as a Hari observation and let Hari's belief-state machinery answer "has this content already been summarized at this level?" before each LLM call. Big potential cost savings on repeated builds; only worthwhile after Phase 3.
- **Local models for L0→L1** — L1 summaries are short and structured; a 7B local model probably suffices. Investigated when local-model infra exists in the workspace.

Cost discipline (v1):

- Hard cap per build invocation: `--max-cost 5.00` USD (configurable). Builds that would exceed it stop with a checkpoint and surface `state/quality/traceability-cost-budget.json` to the operator.
- L1 batches: at most 32 leaves per `claude` call (context window discipline).
- L4+ summaries cache aggressively — they should change only when L3 children's `kept` lists materially change.

---

## 8. Cross-References

How this contract composes with the rest of the workspace:

- **`@ai:` annotations** — see `2026-05-24-ai-annotation.contract.md`. Every annotation id is a candidate for hoisting into `ai_claims` and `kept` on every ancestor. The traceability tree is the index *over* the annotation corpus; the annotation contract owns the leaf format.
- **ga BACKLOG.md parser** — the TypeScript reader emitting `BacklogEpic` / `EpicSubSection` / `BacklogItem` is the L6 / L5 / L4 producer on the product side. The traceability tree's `kind` enum is a superset of those three so a BACKLOG entry maps to a node with no schema bending.
- **`ix-bracelet::grothendieck_delta`** — signed Z⁶ delta on musical PC-sets. The *shape* of the idea — cheap signed delta over a fingerprint — is the model for a future general code-content delta primitive. Not used directly in v1; cited as inspiration.
- **`ix-context::cache`** — the existing content-hash-keyed cache. The traceability tree's `id` field uses the same blake3 family of primitives; integration will likely share a hash helper.
- **`ix-category`** — composable transformations and categorical primitives. The summarization functor `L_n → L_{n+1}` is a candidate for expression as a category arrow once the data shape stabilizes; not pursued in v1.
- **`ix-graph`** — graph data structures. `children` + `cross_refs` form a DAG that can be loaded into `ix-graph` for whole-tree queries (ancestor / descendant / shortest path).

---

## 9. MCP Tools (specification only; no implementation in this PR)

Four tools to be implemented by the Phase-5 `ix-traceability-tree` MCP surface. Schemas are provisional; Phase 1 will lock them.

### 9.1 `ix_tree_query`

Find nodes by content or structure.

```json
{
  "input": {
    "workspace": "string (repo root path)",
    "query": "string (free-text, matched against summary + kept + ai_claims)",
    "level_min": "integer 0..=6 (optional, default 0)",
    "level_max": "integer 0..=6 (optional, default 6)",
    "kind": "string (optional, restrict to one kind)",
    "limit": "integer (optional, default 10)"
  },
  "output": {
    "matches": [
      {
        "id": "string (node id)",
        "level": "integer",
        "kind": "string",
        "summary": "string",
        "path_from_root": ["string (id chain)"],
        "score": "number 0.0..=1.0"
      }
    ]
  }
}
```

### 9.2 `ix_tree_zoom`

Navigate the tree by id. The primary zoom-in / zoom-out operator.

```json
{
  "input": {
    "node_id": "string",
    "direction": "string enum: in | out | siblings",
    "depth": "integer (optional, default 1, max 3)"
  },
  "output": {
    "focus": "node object (full schema)",
    "neighbors": ["node objects per direction + depth"]
  }
}
```

### 9.3 `ix_tree_drift`

List drift alarms.

```json
{
  "input": {
    "workspace": "string",
    "since": "ISO-8601 timestamp (optional)",
    "level_min": "integer (optional)"
  },
  "output": {
    "alarms": [
      {
        "node_id": "string (the parent whose contract was violated)",
        "child_id": "string",
        "violated_invariant": "string",
        "first_seen": "ISO-8601 timestamp",
        "current_status": "string enum: open | acknowledged | resolved"
      }
    ]
  }
}
```

### 9.4 `ix_tree_rebuild`

Trigger selective re-summarization.

```json
{
  "input": {
    "workspace": "string",
    "scope": "string enum: stale | subtree | full",
    "root_id": "string (required if scope=subtree)",
    "max_cost_usd": "number (optional, default 5.00)",
    "dry_run": "boolean (optional, default false)"
  },
  "output": {
    "rebuilt_count": "integer",
    "skipped_count": "integer",
    "cost_usd_estimated": "number",
    "cost_usd_actual": "number (omitted on dry_run)",
    "stopped_reason": "string enum: complete | cost_cap | error"
  }
}
```

---

## 10. Out of Scope for v1

Captured here so they're not silently re-litigated in Phase 1.

- **Source-line-accurate test coverage mapping.** The schema reserves `cross_refs` for code↔test edges, but populating them from `cargo-llvm-cov` source-line data is a separate workstream. Phase 1 emits coverage cross-refs at file granularity only.
- **UI rendering.** The zoom UX lives in ga (separate PR). This contract owns the data shape; the renderer is downstream.
- **Cross-repo trees.** v1 is single-workspace. Federation (one tree spanning ix + ga + tars + Demerzel) requires resolving repo-relative paths across siblings and likely a dedicated cross-repo node kind. Deferred.
- **Auto-update of `kept` contracts.** When `contract_invariants` evolves, the operator must reconcile manually. Auto-derivation from the L0 layer is interesting but moves the contract from "human-readable" toward "LLM-readable" and is not free.
- **Translation / multilingual summaries.** Per `feedback_french_docs.md` and `user_french.md`, the workspace cares about French. v1 stores English summaries only; a `lang` field is a candidate v2 addition.

---

## 11. Validation Plan

A one-week proof-of-concept against `ix-code` (specifically its `catalog.rs` module — small, meta, hand-verifiable). See `docs/plans/2026-05-24-traceability-tree-validation-plan.md` for the day-by-day breakdown.

Success criteria (operator-visible):

- Clean L0–L3 build on the `ix-code` crate with < 2 % staleness false positives over 10 random file edits.
- Zoom UX (MCP only, no UI) answers 5 representative queries correctly (e.g., "what invariants does `ix-code::analyze` promise?" → returns the L3 node with `kept` covering the relevant `@ai:invariant`s).
- Total LLM cost for the full one-week PoC < USD 50.
- Drift alarm fires on at least one synthetic regression where a child's invariant is silently dropped.

Failure on any single criterion is a go-back-to-the-drawing-board signal, not a soft regress.

---

## 12. Open Questions

Surfaced rather than papered over. Phase 1 must resolve these before locking.

1. **Multi-language workspaces.** ix is Rust; ga is C# + TypeScript + React; tars is F#. A workspace-rooted tree spanning all three has to handle three symbol-extraction toolchains. v1 assumes single-language per workspace. Is that sustainable, or do we need a per-language adapter trait in Phase 1?
2. **L4+ rebuild cadence.** Cron vs on-demand vs hybrid. Cron risks expensive rebuilds when no one will read the result; on-demand risks stale roots when an operator zooms in cold. Phase 1 will probably start on-demand-only and add cron later, but the right answer depends on usage telemetry.
3. **Contract versioning.** When the schema itself evolves (v1 → v2), every existing tree's `schema_version: 1` either gets migrated or pinned. Which? Lock-and-migrate is safer; lock-and-pin is cheaper. The `ai-annotation` contract took lock-and-migrate; should this one match?
4. **`cross_refs` type system.** v1 leaves edge types implicit (the kinds at both ends imply the type). Phase 1 may need an explicit `edge_kind` enum for the `task ↔ test`, `code ↔ doc`, `demo ↔ post_mortem` cases to make MCP queries tractable.
5. **Fidelity self-report calibration.** `fidelity: number` is currently free-form. Is "0.8" comparable across summarizers? Probably not without a calibration pass. Either drop the field or define a rubric in Phase 1.
6. **Algedonic channel integration.** Drift alarms (Section 6.3) feel like they want to flow into the Demerzel algedonic-channel substrate once that contract lands. Wait for it, or define our own channel and migrate later?
7. **Backlog parser ownership.** The BACKLOG.md → tree producer naturally lives in ga (where the parser already is). But the schema lives in ix. Phase 1 needs a clear ownership boundary — likely: ga emits node JSON conforming to this schema; ix consumes and aggregates.

---

## 13. Phase Plan (this contract = Phase 0)

- **Phase 0 (this PR):** schema + contract + validation plan.
- **Phase 1:** `ix-traceability-tree` crate scaffold; L0–L3 builder; cost telemetry; one-week PoC against `ix-code`.
- **Phase 2:** L4–L6 builder; drift alarm wired to `state/quality/traceability-drift.jsonl`.
- **Phase 3:** MCP surface (`ix_tree_query`, `ix_tree_zoom`, `ix_tree_drift`, `ix_tree_rebuild`).
- **Phase 4:** ga BACKLOG.md producer emits compliant node JSON; ix consumes.
- **Phase 5:** ga dashboard zoom UI.
- **Phase 6:** Cross-repo federation; contract freeze at end of Phase 6.

---

## 14. Versioning

The contract is `v0.1.0` (draft). Per the ix cross-repo contracts pattern, schema_version bumps are reserved for breaking field rename / removal. New optional fields are additive. Freeze milestone: end of Phase 6 (full pipeline shipped + ≥ 3 weeks of drift-alarm telemetry showing < 1 false-positive-per-day on the ix workspace).
