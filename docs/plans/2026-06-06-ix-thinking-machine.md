# Plan / decision log — IX "thinking machine" (NL ↔ dynamic IX pipelines)

**Date:** 2026-06-06
**Branch:** `feat/ix-thinking-machine-skeleton`
**Brainstorm + panel:** `docs/brainstorms/2026-06-06-ix-thinking-machine-architecture.md`
**Status:** walking skeleton + MCP wrapper shipped; passed a 5-dimension
adversarial self-review (1 P1 + 3 P2 confirmed, all addressed — see findings
below). Next increments listed at the end.

## Problem

Make IX an agent that translates natural language ↔ dynamic IX pipelines,
bidirectionally, executes them under governance, and is **dogfooded to improve
IX itself** — failed translations and out-of-domain requests become a logged
backlog of concrete IX gaps (`state/thinking-machine/gaps.jsonl`).

## What shipped (walking skeleton)

`ix pipeline compile "<NL>" [--run]`: direct-LLM proposer → canonical
`PipelineSpec` → coverage gate (lexical + LLM relevance) → `lower()` validate
(≤3-round repair) → fail-closed governance gate → execute → NL narration.
Plus transport-agnostic introspection: `ix pipeline schema`,
`ix list skills --schemas`. Fail-closed governance gate also added to
`ix pipeline run`.

## Decision log (reversibility + revisit trigger)

| # | Decision | Door | Revisit trigger |
|---|---|---|---|
| D1 | **Transport = direct Anthropic Messages API**, NOT MCP `sampling/createMessage` | two-way | MCP sampling un-deprecated AND a real host implements the provider callback (see `reference_mcp_sampling_deprecated`) |
| D2 | **Canonical IR = `PipelineSpec`** (`ix.yaml`), deprecating System A's `{steps:[…]}` | **ONE-WAY** (needs sign-off to delete `{steps}`) | zero non-test callers of `ix_pipeline_run`'s `{steps}` executor for 14 days |
| D3 | **Proposer home = `ix-skill` CLI verb** (not an MCP tool yet) | two-way | agent-native parity needed → add MCP-tool wrapper |
| D4 | **Coverage = two-tier**: free lexical TF-IDF pre-filter + fail-closed LLM relevance (`NO_COVERAGE`) | two-way | lexical false-positives matter → replace pre-filter with real embeddings (ix-gpu cosine) |
| D5 | **Governance gate in the CLI verbs** (`run`+`compile`), template-time only, not yet inside `ix-pipeline::execute()` | two-way | **NOW TRIGGERED** by the P1 below — a `{from}`-ref can hide a destructive op from the template-time scan; the durable fix is gating the *resolved* args in `ix-pipeline::execute()` (add an `ix-governance` dep to `ix-pipeline`; also covers MCP `ix_pipeline_run`) |

**One-way doors needing explicit sign-off before they harden:**
- Publishing the `PipelineSpec` JSON Schema (`ix pipeline schema`) to a stable
  `$id` (currently a draft `urn:`). Keep internal + `{schema,registry}`-hashed.
- The governance-tag → constitutional-article contract once a structured
  `check_pipeline_spec` maps them (currently uses free-text `check_action`).
- Deleting the `{steps:[…]}` path (D2).

## Instrumentation (baseline + direction + guardrail)

- **Headline:** translation-success-rate (% NL → spec that validates+runs).
- **Value:** dogfood-fixes-merged / cycle (gaps.jsonl rows → IX improvements).
- **Guardrail:** no net increase in clippy/test failures.
- **Oracle for "did this improve IX?" is EXECUTABLE** (`cargo test` + sentrux +
  `ix-quality-trend`), never an LLM judge panel (≈96% TPR / <25% TNR).

## Dogfood findings so far (the loop working)

1. Out-of-domain requests confabulate structurally-valid-but-irrelevant specs →
   added coverage gate.
2. Lexical TF-IDF coverage misses partial content-word collisions (measured
   0.32 for a scrape/email request) → added fail-closed LLM relevance; logged
   "replace lexical with embeddings" as the next-level fix.
3. **(Adversarial self-review, P1)** The governance gate is *template-time*: a
   `{"from": "upstream"}` ref can supply a destructive operation that resolves
   only at execution time, slipping past the substring scan. Bounded
   exploitability, but structurally the gate never sees the resolved value.
   Addressed this pass (removed the "unbypassable" overclaim; gate now surfaces
   `unvetted_runtime_inputs`); **durable fix = the `unify` increment** (gate
   resolved args in the executor). The loop found a real gap in the gate it
   built. Also confirmed 3 P2s (UTF-8 panic in narration, missing HTTP timeout,
   a fail-open comment labeled fail-closed) — all fixed.

## Next increments (priority order)

1. ✅ **Repair-loop proof** — deliberate-error test confirming the ≤3-round
   structural repair fires live. *(shipped)*
2. ✅ **MCP-tool wrapper** — `ix_nl_to_pipeline` exposes `compile` as a tool
   (agent-native parity, live-verified). *(shipped)*
3. **IR unification (`unify`)** — retarget + deprecate-with-shim
   `ix_pipeline_compile` (`{steps}`); **gate the resolved args inside
   `ix-pipeline::execute()`** (D5) — now carries the P1 security driver above,
   not just coverage of the MCP path. ONE-WAY door (new `ix-governance` dep on
   foundational `ix-pipeline`; deleting `{steps}`) → needs sign-off.
4. **Embeddings coverage** — replace lexical pre-filter (D4).
