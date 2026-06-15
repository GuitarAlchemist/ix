# Contract: `chatbot-trace-regression` (v0.1 — draft)

**Status:** v0.1 **draft** (documented JSON shape — *no formal `.schema.json` yet*).
**Producer:** `ix-duck` `ix_chatbot_lens check` (the chatbot flight recorder, Slice B).
**Consumer:** none yet (intended: GA `Tools/QualityLens` / dashboard, read-only).
**Location:** `state/quality/analytics/chatbot-trace-regressions.json` (**ix-side**, gitignored —
regenerable). Never written into the GA tree.

This contract is the machine-readable verdict of the canonical-diff gate: did any chatbot
prompt's **routed agent** (`response.agentId`) diverge from its canonical expectation
(`_signature.json`'s `orchestration.answer` step)? Per CLAUDE.md, v0.1.x is a draft — only
freeze at the named Phase-4 milestone, and changing a locked field needs GA coordination.

## Why a documented shape, not a frozen schema (yet)

No consumer reads it today. A formal `.schema.json` (`$schema`/`$id`/`additionalProperties`/
`const`) plus a `provenance{}` block and the human-ack re-baseline machinery land in the **same
PR that wires the first reader** — formalizing earlier would freeze a surface that is still
moving (e.g. `response_length` migrated from hard to soft). Consumers MUST ignore unknown keys.

## Shape

```jsonc
{
  "schema_version": "chatbot-trace-regression.v0.1",
  "run_at": "2026-06-14T18:22:05+00:00",   // RFC3339
  "run_selection": "single",               // v1 only; corpus is 100% runCount:1.
                                            // reserved for a future majority/median reducer
  "status": "pass",                         // pass | regression | degraded | skipped
  "prompts_checked": 45,
  "baseline_ref": "fnv1a64:9f3c…",          // deterministic hash of the sorted
                                            // (prompt_id, expected_agent) canonical set
  "baseline_changed": false,                // v1: always false (no history store yet)
  "regressions": [
    {
      "prompt_id": "diatonic-chords-in-g-major",
      "category": "case-variants",
      "signal": "agent_id",                 // hard signal in v1
      "severity": "hard",                   // hard (fails) | soft (advisory)
      "expected": "skill.diatonicchords",
      "actual": "skill.WRONG"
    }
  ],
  "degraded_reason": "…"                     // present only when status == degraded
}
```

## Status semantics (exit-code map)

| `status` | meaning | exit | ledger `decision` projection |
|----------|---------|------|------------------------------|
| `pass` | no routed-agent regression | 0 | `pass` |
| `regression` | ≥1 clean heterogeneous `agent_id` flip | **1** | `fail` |
| `degraded` | corpus absent-canonical-heavy, or homogeneous collapse (backend/env symptom) | 0 | `warn` |
| `skipped` | corpus absent/empty (e.g. GA sibling not present) | 0 | `skip` |

**Fingerprint rule:** a single clean heterogeneous flip ⇒ `regression` (never masked by count); a
homogeneous collapse (≥2 prompts → same agent, ≥50% of corpus) ⇒ `degraded` (warn, not fail).
**Fail-closed:** a read error when the corpus *is* present is a failure, distinct from
`skipped` (corpus absent).

## Locked surface (when formalized at Phase 4)

`status` enum + exit-code map · `regressions[]` element shape · `baseline_ref` semantics
(content hash of the compared canonical set). These are the GA-readable surface — a change needs
GA ack, logged as a one-way-door entry. Everything else is additive.

## Deferred to first-consumer / future

`provenance{ga_commit,ix_commit,corpus_size,tool_version}` · the formal `.schema.json` ·
`baseline_changed` enforcement (human-ack re-baseline + prior-hash tamper-evidence) ·
`warn` status + `soft_flags[]` for the `response_length` soft band · the multi-run reducer.

## Reference
- Plan: [`docs/plans/2026-06-14-004-feat-chatbot-duckdb-flight-recorder-plan.md`](../plans/2026-06-14-004-feat-chatbot-duckdb-flight-recorder-plan.md)
- Producer: `crates/ix-duck/src/chatbot.rs` (`check_regressions`, `GateReport`).
