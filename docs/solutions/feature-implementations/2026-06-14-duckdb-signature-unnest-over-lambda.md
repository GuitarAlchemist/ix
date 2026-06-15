---
title: "DuckDB over chatbot golden-traces: UNNEST a flat _signature.json beats a nested list-lambda"
category: feature-implementations
date: 2026-06-14
tags: [duckdb, ix-duck, chatbot, json, unnest, read_json_auto, regression-gate]
symptom: "Extracting the routed agent from GA golden-traces looked like it needed a fragile DuckDB list-lambda over canonicalSteps[]; response_length diffs false-flagged correct answers."
root_cause: "GA emits a flat _signature.json projection; the nested _canonical.json path and char-length comparison were the wrong tools."
---

# DuckDB over chatbot golden-traces (ix-duck flight recorder)

Building the chatbot canonical-diff gate (`crates/ix-duck/src/chatbot.rs`) surfaced three
non-obvious facts worth reusing for any DuckDB-over-GA-JSON work.

## 1. Prefer the flat `_signature.json` + `UNNEST` over a nested list-lambda

The "expected routed agent" looked like it lived only in `_canonical.json` at
`canonicalSteps[].invariantAttributes["agent.id"]`, needing a DuckDB list-lambda
(`list_filter(canonicalSteps, lambda s : s.name='orchestration.answer')[1]...`). Two traps:
the `->` lambda arrow is **deprecated** (removed in DuckDB v2.0), and `invariantAttributes` may
infer as STRUCT *or* MAP, changing the access syntax.

GA also ships **`_signature.json`** — a flat `steps[]` of `{name, status, agentId}`. So:

```sql
CREATE OR REPLACE TABLE chatbot_signatures AS
SELECT t.promptId AS prompt_id, step.agentId AS expected_agent
FROM read_json_auto([...], filename=true, union_by_name=true, sample_size=-1) t,
     UNNEST(t.steps) AS u(step)
WHERE step.name = 'orchestration.answer';
```

`UNNEST` is version-stable across all DuckDB releases (no lambda-syntax gamble), and a prompt
with no matching step simply produces no row → `LEFT JOIN` yields `expected_agent = NULL` →
treated as a degraded/missing-canonical signal. **Check what flat projections the producer
already emits before writing nested extraction.**

## 2. `response_length` drifts on *correct* answers — don't hard-gate on it

`_canonical.json` files the answer's `response.length` under `invariantAttributes` (implying
exact-match), but it is **not** invariant: e.g. `explain-the-circle-of-fifths` canonical 3688 vs
live 3882 (+5.3%), `notes-in-c-major` canonical 86 vs actual `naturalLanguageAnswer` 98 — both
correct answers whose generated prose length wanders. An exact length diff false-flags day one.
The stable signal is the **routed `agent_id`**; length is soft-only (a band, never a hard fail).
(All `rangeAttributes` are empty across the corpus — the band mechanism exists but is unused.)

## 3. Pass `read_json_auto` an explicit file LIST, not a glob, for graceful-degrade

`read_json_auto('dir/*.json')` errors when zero files match — bad for the absent-sibling
(`../ga` missing) path. Enumerate the files in Rust and pass a list literal
`read_json_auto(['p1','p2'], …)`; an empty list means "create an empty table, return 0" (skip,
not error). Bonus: explicit enumeration bounds untrusted input (security review). Pair with
`union_by_name=true` + `TRY_CAST` + `sample_size=-1` so ragged files (missing optional fields,
late type drift) NULL-fill instead of aborting.

**Files:** `crates/ix-duck/src/chatbot.rs`, fixtures under `crates/ix-duck/tests/fixtures/`,
plan `docs/plans/2026-06-14-004-feat-chatbot-duckdb-flight-recorder-plan.md`.
