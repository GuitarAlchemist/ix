---
date: 2026-05-24
reversibility: two-way door (PoC scope; no schema freeze, no public-API commitment, no production data path)
revisit-trigger: PoC LLM cost projection > USD 50 per workspace-rebuild; OR drift-alarm false-positive rate > 2 % over 10 controlled edits; OR schema field churn during Phase 1 build exceeds 3 changes
status: design — depends on `docs/contracts/2026-05-24-traceability-tree.contract.md` (this PR)
governance: out-of-scope for this PR; Phase-1 implementation will run `ix_governance_check` against Demerzel constitution before shipping
---

# Traceability Tree — One-Week Proof-of-Concept Validation Plan

## Why this plan exists

The contract in `docs/contracts/2026-05-24-traceability-tree.contract.md` is a hypothesis: that a Merkle-style summary tree with a `kept`/`dropped` manifest is the right shape for the workspace's traceability problem. A one-week PoC against a deliberately small target proves or refutes the hypothesis cheaply, before we commit to the six-phase roadmap. Karpathy R4 (goal-driven execution) applied to a contract: turn the schema into verifiable success criteria; loop until each holds.

## Target

The `catalog` module of the `ix-code` crate (`crates/ix-code/src/catalog.rs` and any L1/L2 neighbors needed to ground the build):

- **Small enough to walk by hand.** ~300 lines of curated data plus query helpers; the operator can verify an L2 summary against the source in under five minutes.
- **Meta.** The module is itself a catalog of code-analysis tools, which is conceptually adjacent to what the traceability tree does — a useful test of self-description.
- **Stable.** The module changes infrequently, which makes the staleness signal less noisy than picking a target under active development.
- **Owned by ix.** No cross-repo handoff to coordinate during the PoC.

Stretch target if Day 3 finishes early: extend to the whole `ix-code` crate (~12 files). Out-of-scope: the entire workspace, anything in ga, anything cross-repo.

## Day-by-day breakdown

### Day 1 — Schema settled; CLI scaffold

- Merge this PR (contract + schema + plan).
- `cargo new --lib crates/ix-traceability-tree`; wire into the workspace `Cargo.toml`; add `blake3`, `serde`, `serde_json`, `jsonschema` (validation), `chrono`, `clap` (CLI surface).
- Implement the L0 ingestor: take the existing `ix-code` analysis output (file listing + symbol spans) and emit one L0 JSON-Schema-compliant node per symbol. No LLM yet.
- `cargo test` proves the L0 nodes validate against `traceability-tree.schema.json`.
- **Exit gate:** every L0 node round-trips through the schema validator without error; blake3 ids are deterministic across two builds.

### Day 2 — L1 builder; caching; cost telemetry

- L1 prompt template lives in `crates/ix-traceability-tree/prompts/l0-to-l1.md`. Shells out to `claude` CLI per leaf with `--max-cost 0.10` and the symbol's source span as input.
- Cache: blake3 of (prompt-template-hash || symbol-source-hash) keyed against a local sled or json-file store. Cache hit skips the LLM call entirely.
- Cost telemetry: every LLM call appends one JSONL line to `state/quality/traceability-cost.jsonl` (timestamp, model, input-tokens, output-tokens, USD-estimated). Sum is the operator-visible spend.
- **Exit gate:** building L1 over `catalog.rs` costs less than USD 1.00 cold, less than USD 0.05 warm. Every L1 summary contains the symbol's `@ai:` annotations verbatim in its `kept` array.

### Day 3 — L2 and L3 builders; Merkle chain

- L2 prompt aggregates per-file L1 children into one file summary. L3 aggregates per-module L2 children. Templates in `prompts/l1-to-l2.md` and `prompts/l2-to-l3.md`.
- Implement the canonical hash: `blake3(canonical(summary || kind || level || provenance.source_path || sorted(children)))`. The hash chain MUST be deterministic across builds with the same inputs.
- Implement the `kept`/`dropped` promotion check: any `@ai:invariant` or `@ai:contract` in a child's `kept` that does not appear in the parent's `kept` is a build failure. Phase-1 enforcement, not a soft warning.
- **Exit gate:** full L0–L3 build over the `ix-code::catalog` subtree completes; cold cost less than USD 5.00; structural check passes; every annotation is hoisted to L3.

### Day 4 — Sync engine; selective re-summarization

- Stale detection: on `cargo run --bin ix-tree -- check`, walk the tree, recompute every node's hash from its children, mark mismatches `stale: true` with a `stale_reason` populated.
- Re-summarization: on `cargo run --bin ix-tree -- rebuild --stale-only`, re-summarize every `stale` node bottom-up.
- Source-side change detection: use `ix-context::cache::hash_file_content` plus `ix-io::watcher` primitives (both already exist; no new general-purpose `ix-grothendieck-delta` crate required in Phase 1). The watcher emits change events; the cache hash confirms material change.
- Drift alarm: when a re-summarized child's `kept` no longer covers an entry in the parent's `contract_invariants`, write one JSONL line to `state/quality/traceability-drift.jsonl` and surface to stderr.
- **Exit gate:** on 10 controlled edits to `catalog.rs`, the stale-detection false-positive rate is below 2 %; at least one synthetic regression (delete an `@ai:invariant`) fires a drift alarm correctly.

### Day 5 — `ix_tree_query` MCP surface; minimal ga consumer

- Implement `ix_tree_query` against `ix-agent` per the contract's section 9.1. Input: free-text query + level / kind filters. Output: ranked matches with `path_from_root`.
- Ranking in v1: substring match on `summary` + `kept` + `ai_claims`, weighted toward higher `kept` matches. No embedding model yet.
- Minimal ga consumer: a 50-line PowerShell script that hits `ix_tree_query` over MCP and prints the result as plain JSON. No UI in this PoC — confirms the data path end-to-end.
- **Exit gate:** 5 representative queries (see "Success criteria" below) return correct top-1 results.

### Day 6 — Cost analysis; drift detection rigor

- Run a controlled cost analysis: cold rebuild of the entire `ix-code` crate (not just `catalog.rs`), warm rebuild after no changes, warm rebuild after one localized change. Report each as USD and as fraction of the cold-rebuild cost.
- Run the drift detector across 10 synthetic regressions (each: delete one `@ai:invariant`, re-summarize, expect alarm; restore, expect alarm clears). Measure true-positive rate, false-positive rate, latency between source edit and alarm.
- **Exit gate:** cold rebuild of `ix-code` < USD 15.00; warm-no-change rebuild < USD 0.10; drift detector at 100 % true-positive rate, less than 10 % false-positive rate.

### Day 7 — Writeup; go / no-go

- One markdown writeup in `docs/solutions/traceability-tree/2026-05-31-poc-writeup.md` covering: what worked, what broke, total LLM spend, where the schema bent under reality, and a recommendation: GO (proceed to Phase 1 build), HOLD (schema needs revision; recycle), or NO-GO (the wavelet model is not the right shape; rethink).
- Update the contract's section 12 (Open Questions) with the resolutions PoC produced.
- File the writeup under `state/digests/` as well so the next session's `sessionstart-digest.sh` hook picks it up.

## Explicit success criteria

Operator-visible. Failure on any one is a no-go signal, not a soft regress.

1. **Schema validates the build.** Every node emitted by Days 1–4 passes `traceability-tree.schema.json` under Ajv 2020 (or equivalent).
2. **Clean L0–L3 build on `ix-code`.** All four levels populated, no orphan nodes, no broken `parent`/`children` references.
3. **Staleness false-positive rate below 2 %** over 10 controlled file edits (edit non-load-bearing whitespace, expect no stale; edit load-bearing logic, expect stale).
4. **Five representative queries answer correctly.** Defined for Day 5:
   - "What invariants does `ix-code::catalog::all` promise?" → top-1 hit is the L2 node for `catalog.rs` with the invariants in `kept`.
   - "Which tools in the catalog cover Rust?" → top-1 hit is the L1 node for `by_language`.
   - "Show me the module summary for `ix-code::catalog`." → top-1 hit is the L3 module node.
   - "What `@ai:contract` claims live under `ix-code`?" → returns all ancestors whose `ai_claims` include `@ai:contract`-typed annotations.
   - "Are there any drift alarms open on the catalog subtree?" → returns the open alarms from `traceability-drift.jsonl`, or an empty list if none.
5. **Drift detector fires on at least one synthetic regression** where a child's `@ai:invariant` is deleted and the parent's `contract_invariants` is silently violated.
6. **Total LLM spend less than USD 50.00** across the full week. Per-build costs surface in `state/quality/traceability-cost.jsonl` and are summable by `jq`.

## Out of scope for the PoC

- UI rendering (lives in ga, separate PR after Phase 5).
- Cross-repo trees (single workspace for the PoC; federation deferred to Phase 6).
- Source-line-accurate test coverage cross-refs (Phase 1 uses file-granularity only).
- Local-model substitution for `claude` CLI (investigate after the cost telemetry is in hand).
- Embedding-based query ranking (substring matching is enough for the 5 test queries; embeddings come with Phase 3 MCP polish).

## Open risks

Captured here so they don't surprise us mid-week.

- **`claude` CLI rate-limiting on cold rebuild.** Mitigation: throttle the L0→L1 batch to one call per second; if it blocks the cold build under 30 minutes, look at the `--max-cost` cap rather than parallelism.
- **`ix-code` analysis output not symbol-accurate enough for L0.** Mitigation: fall back to a regex-based symbol splitter for the PoC; do not block on a richer parser.
- **`@ai:` annotation extractor (PR #54) not yet merged on main.** Mitigation: cherry-pick or vendor the extractor binary for the PoC; do not block on the PR landing first.
- **Hash determinism across platforms.** Mitigation: pin blake3 version; canonicalize JSON via `serde_json::to_string_pretty` with a sorted-keys helper; add a Linux + Windows CI smoke test on the determinism property before Day 4.
- **Cache invalidation on prompt-template edits.** Mitigation: include the prompt-template-hash in every cache key from Day 2. Tested by editing the L1 prompt and confirming a full rebuild.

## Phase gate

If Days 1–7 all pass their exit gates and the success criteria above hold, the PoC is GO and Phase 1 proceeds: full `ix-traceability-tree` build over the entire ix workspace, drift telemetry surfaced to the ga dashboard, and the `ix_tree_query` MCP tool promoted from PoC to stable. If any exit gate fails, the writeup names the failure and either schedules a revision (HOLD) or kills the line (NO-GO).
