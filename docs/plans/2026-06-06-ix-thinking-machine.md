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
| D5 | **Governance now gates RESOLVED args at execution time** via a trait-injected `ix_pipeline::gate::StageGate` (template-time `governance_gate` kept as a fail-fast pre-flight) | two-way | ✅ **RESOLVED for the canonical `PipelineSpec` path** — the P1 `{from}`-ref bypass is closed: `lower_with_gate` consults the gate on post-resolution args before each skill runs. Design improvement vs. the original plan: a **trait seam** keeps `ix-pipeline` governance-agnostic (no `ix-governance` dep on the foundational crate; `ix-skill` injects `ConstitutionGate`). A caller is gated **iff it lowers with a gate** — `ix pipeline run` + `compile --run` do. ⚠️ The legacy System A MCP `ix_pipeline_run` (`{steps}` substitution in `ix-agent/src/tools.rs`) does NOT lower with a gate and stays ungated until `{steps}` retirement (D2) |

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
3. **(Adversarial self-review, P1)** The governance gate was *template-time*: a
   `{"from": "upstream"}` ref could supply a destructive operation that resolves
   only at execution time, slipping past the substring scan. First-pass
   mitigation (#77): removed the "unbypassable" overclaim; gate surfaces
   `unvetted_runtime_inputs`. **Durable fix now SHIPPED** (the `unify` PR): a
   trait-injected `ix_pipeline::gate::StageGate` consulted on each stage's
   *resolved* args before its skill runs (`lower_with_gate` + `ConstitutionGate`).
   The loop found a real gap in the gate it built — and closed it. Also confirmed
   3 P2s (UTF-8 panic in narration, missing HTTP timeout, a fail-open comment
   labeled fail-closed) — all fixed in #77.

## Next increments (priority order)

1. ✅ **Repair-loop proof** — deliberate-error test confirming the ≤3-round
   structural repair fires live. *(shipped)*
2. ✅ **MCP-tool wrapper** — `ix_nl_to_pipeline` exposes `compile` as a tool
   (agent-native parity, live-verified). *(shipped)*
3. **IR unification (`unify`)** — split into two halves:
   - ✅ **Resolved-args governance** (the P1 security half) — *shipped* via the
     `ix_pipeline::gate::StageGate` seam + `lower_with_gate` + `ConstitutionGate`
     (D5) for the canonical `PipelineSpec` path (`ix pipeline run`, `compile
     --run`). Trait injection avoided the foundational-crate dep entirely.
   - ⏳ **Retire System A `{steps}`** (the cleanup half) — deferred to its **D2
     trigger** (14-day zero-non-test-caller window) before deleting; premature
     to delete now. ONE-WAY door → still needs sign-off when the window clears.
     **Until then the legacy MCP `ix_pipeline_run` (`{steps}`) is NOT
     resolved-args-gated** — it executes `$step.field` substitution directly,
     bypassing `lower_with_gate`. Gating-or-retiring it is the open security
     follow-up.
4. **Embeddings coverage** — replace lexical pre-filter (D4).
