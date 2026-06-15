---
date: 2026-06-14
topic: chatbot-duckdb-leverage
---

# Leveraging DuckDB to develop & maintain the GA chatbot

## What We're Building

An **ix-side, cross-repo DuckDB analysis module** (in the existing `ix-duck` crate)
that turns the GA chatbot's write-only JSON exhaust into a queryable flight recorder.
It reads GA's on-disk chatbot artifacts (`../ga/state/quality/...`,
`../ga/state/chatbot-reviews/`) with graceful-degrade when the sibling is absent, and
exposes a portable set of DuckDB tables/views for both ad-hoc dev queries and a CI
regression gate. GA's existing `build-views.sql` / `quality.duckdb` remains the
.NET-facing surface; this is the Rust/agent-facing reader over the **same files**.

## Why This Approach

GA already materializes **daily aggregates** (`chatbot_qa`, `routing_eval`,
`voicing_analysis`) in `state/quality/analytics/build-views.sql` (commit 4c3dcb1c).
That's the trend layer — and it's thin (latest `chatbot_qa` row is
`pass_pct: null, degraded: backend_unavailable`). The real leverage is one level
deeper, in artifacts nobody queries:

- **Golden traces** (45 prompts) — `chatbot-qa/golden-traces/*/run-*.json` +
  `_canonical.json`: full OTel/genAI trace per prompt — `agent.id` (skill that
  answered), `routing.method`, `routing.confidence`, `candidate.count`,
  `response.length`, `elapsedMs`, `grounding`, `traceId`.
- **AFK loop** — `chatbot-qa/afk-runs/*.jsonl`: autonomous fix-loop events
  (`target_selected`, det/sem scores, deferred reasons).
- **Review ledger** — `state/chatbot-reviews/*.json`: per-PR reviewer verdicts,
  `findingsCount`, `blockingCount`, persona.
- **Cost** — `state/quality/ai-costs/pricing.json`: token pricing, joinable to
  trace latency.

Recurring dev questions we currently answer from memory (weak intents
scaleinfo 0.73 / progressioncompletion 0.60; `modes` 64s latency; ungrounded
answers) all live in these files. DuckDB makes them queries, not memory.

**ix-side fit:** CLAUDE.md's realtime/offline boundary assigns offline
large-corpus analysis to ix (sentrux owns the live view). An offline analyzer of
GA's trace corpus is squarely ix's job.

## Key Decisions

- **v1 scope = full sweep A–E** (user choice), built as five vertical slices so each
  table ships end-to-end (test + one real query) before the next — honors the
  tracer-bullet discipline despite the broad scope.
- **Home = `ix-duck`** (user choice), cross-repo reader of `../ga`, graceful-degrade
  when GA is absent (cf. `ix-streeling`, the `governance/demerzel` submodule).
- **Contract aligns on SQL/table schema, not runtime coupling** — GA keeps its
  `quality.duckdb`; ix reads the same JSON. No second store, no service.
- **A + B are the tracer bullet:** `chatbot_traces` warehouse + canonical-diff gate.
  A per-prompt trace diff is a far better green-but-dead guard than the single flat
  `pass_pct`.

### Leverage map (Develop vs Maintain)
- **A. Trace warehouse** (`chatbot_traces`): weak-intent, latency-outlier,
  ungrounded-answer (`grounding_present = false`), routing-method-drift queries. *(Develop)*
- **B. Regression gate**: SQL-diff each run vs `_canonical.json`; flag `agent_id` /
  `response_length` drift; wire as CI gate. *(Maintain)*
- **C. AFK telemetry** (`chatbot_afk_events`): what the loop tried/deferred. *(Maintain)*
- **D. Review ledger** (`chatbot_reviews`): recurring finding classes, QA-rigor trend. *(Maintain)*
- **E. Cost lens**: pricing × trace latency → per-skill/per-prompt cost. *(Maintain)*
- **F. Cross-repo (stretch)**: join chatbot voicing queries against OPTIC-K corpus
  health — same SQL surface spans ix algorithms ↔ ga chatbot traces.

## Open Questions

- Trace-flattening: JSON_TABLE/UNNEST over the nested `trace.steps[]` vs a thin Rust
  pre-flatten pass — which is cleaner for the OTel step array? (resolve in plan)
- Where does the CI gate run — ix CI reading `../ga` won't have GA checked out;
  does the gate live in GA CI consuming an ix-emitted contract, or does ix CI vendor
  a trace fixture? (cross-repo coordination — log as one-way-ish door)
- Does this feed the existing `learnings-researcher` / Streeling registrar (traces as
  a queryable learning source)?

## Next Steps
→ `/ce-plan` for implementation details (v1 = A–E sliced, A+B first).
