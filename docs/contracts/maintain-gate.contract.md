# Contract: `maintain-gate` (v0.1 — draft)

**Status:** v0.1 **draft** — documented shape + a companion `maintain-gate.schema.json`
(draft-07, `additionalProperties: true`). Per CLAUDE.md, v0.1.x is a draft; the
schema + exit-code semantics **freeze only at the named Phase-4 milestone**
(`docs/plans/2026-06-16-002-feat-maintain-gate-rsi-oracle-plan.md`).
**Producer:** `ix-duck` — `ix_duck::maintain::evaluate` (CLI: `ix_maintain_gate`).
**Consumer:** none yet (intended: Demerzel governance + the bounded-RSI loop, read-only).
**Location:** `state/thinking-machine/maintain-gate.jsonl` (**ix-side**, append-only,
gitignored / regenerable). One verdict per line. Never written into a sibling tree.

The verdict answers one question per self-improvement iteration: **accept, reject, or
escalate?** — by fusing the DuckDB+IX maintain lenses into a single hexavalent value.
DuckDB+IX is the *referee, never the player*: ground truth stays executable
(compile/test/proof); the gate only aggregates evidence about it.

## Why a documented shape, not yet a frozen schema

The schema ships now (the plan calls for it in Phase 2) but stays **draft** until a
consumer reads it. The `provenance{}`/freeze machinery lands in the **same PR that
wires the first reader** (Phase 3 — Demerzel), matching the `chatbot-trace-regression`
precedent — formalizing earlier freezes a surface still moving (e.g. soft-lens
iteration-scoping is a known Phase-3 change). **Consumers MUST ignore unknown keys.**

## The rule (conjunction, never an average)

The two **hard** lenses decide; the two **soft** lenses only downgrade an accept:

```
metric↑ ∧ guardrail held ∧ converging ∧ in-distribution → T  accept
metric↑ ∧ guardrail held ∧ converging ∧ drifting        → P  accept w/ flag
metric↑ ∧ guardrail held ∧ oscillating                  → D  escalate (disputed)
metric↑ ∧ guardrail broke                               → C  reject + ALARM (reward-hack)
metric not↑                                             → F  reject
metric or guardrail unknown, or a *required* dormant lens → U  escalate
```

**C (contradiction)** is the load-bearing case: a metric rising *while* a held
capability breaks is the reward-hack signature — it hard-fails, never averages out.
**Fail-closed:** missing metric/guardrail evidence, or a *required-but-dormant* lens,
escalates to **U** — never a silent ACCEPT. A supplied-but-empty soft lens is
*unknown* (advisory no-op), not a green light.

## Status → decision → exit code

| `status` | meaning | `decision` | exit |
|---|---|---|---|
| `T` | accept | `accept` | 0 |
| `P` | accept with soft flag (drift) | `accept` | 0 |
| `U` | unknown — escalate to human | `escalate` | 2 |
| `D` | disputed (lenses disagree) — escalate | `escalate` | 2 |
| `F` | reject (no improvement / guardrail broke) | `reject` | 1 |
| `C` | contradiction (reward-hack) — reject + alarm | `reject` | 1 |

Hexavalent per ecosystem T/P/U/D/F/C logic. Loop integration (Phase 3): exit 0 →
propose-merge, 1 → reject + revert, 2 → human escalate.

## Shape

```jsonc
{
  "schema_version": "maintain-gate.v0.1",
  "run_at": "2026-06-18T12:00:00+00:00",     // RFC3339
  "status": "T",                              // T | P | U | D | F | C
  "decision": "accept",                       // accept | reject | escalate
  "metric_delta": 0.0184,                     // yield(later half) - yield(earlier half), omitted if unknown
  "metric_up": true,                          // metric_delta >= min_metric_delta; omitted if unknown
  "guardrail_held": true,                     // chatbot gate pass=true, regression=false; omitted if inconclusive
  "converging": true,                         // loops not oscillating; omitted if lens not consulted / dormant
  "drifting": false,                          // queries out-of-distribution; omitted if not consulted / <2 queries
  "signals": [                                // one per lens consulted; ok=null means "no signal"
    { "lens": "metric",      "ok": true,  "detail": "yield delta +0.0184" },
    { "lens": "guardrail",   "ok": true,  "detail": "chatbot gate: pass (0 regression(s))" },
    { "lens": "convergence", "ok": true,  "detail": "loops converging (no oscillation)" },
    { "lens": "drift",       "ok": true,  "detail": "queries in-distribution" }
  ],
  "evidence": [                               // input provenance — hash the EVIDENCE, not just the verdict
    { "kind": "metric",             "source": "state/thinking-machine/hits.jsonl", "hash": "fnv1a64:4e9e8d269a877fa7" },
    { "kind": "guardrail-baseline", "source": ".../chatbot-qa",                    "hash": "fnv1a64:32a3940840062373" }
  ],
  "reason": "metric improved and guardrail held"
}
```

### Field semantics
- **`metric_delta` is externally derived** — yield over a harness-written `hits.jsonl`
  split by `ts_ms` (NOT the cumulative mean). A self-declared delta from the proposing
  agent is forbidden (panel review 2026-06-16): that would be the player scoring its
  own game. Lens metrics carrying a recomputable-evidence pointer are acceptable.
- **`signals[].ok`**: `true` = healthy, `false` = failed, `null` = no signal (a
  required null caps the verdict at `U`).
- **`evidence[].hash`**: FNV-1a-64 content hash of the input. Input provenance >
  verdict provenance — it lets an audit detect a swapped/forged input. **Open
  (Phase 3):** hashing metric + guardrail evidence is here; constitution-checksum is
  v2; write-isolation of the ledgers from the proposing agent is a Phase-3 precondition
  for the gate going *authoritative*.

## Iteration correlation + provenance (Phase 3a)
When the gate is given an iteration scope (`loop_id` + `commit_sha` + the loop's repo):
- **Convergence is scoped** to that `loop_id` (no longer whole-ledger).
- **`commit_sha` is externally verified** against real git state — hex-validated, then
  `git cat-file -e <sha>^{commit}` (exists) + `git status --porcelain` (clean) — **and**
  a recorded loop row must exist for the exact `(loop_id, commit_sha)` pair (verifying
  the commit alone is not enough: a clean-but-unrelated commit for a loop with earlier
  improving rows would otherwise be scored as that loop). A forged/absent sha, a dirty
  worktree, **or no matching loop row** fail-closes to **U** *before any lens verdict*
  (the key is minted by the judged loop, so it gets the same external-derivation
  discipline as the metric). Surfaced as a `provenance` signal + an
  `iteration-commit:<loop_id>` evidence entry (`hash: "git:<sha>"`).

**Still whole-history (deferred):** drift scoping per-iteration awaits a query→loop tag
in Contract B; until then drift stays corpus-level advisory. **Still Phase 3b:** the
verdict is *advisory* — write-isolation of the ledgers from the proposing agent (and
therefore any authoritative auto-revert) is not yet enforced.

## Locked-field discipline
Changing `status`/`decision` value sets or the exit-code mapping after Phase-4 freeze
needs cross-consumer (Demerzel) coordination. Until then, additive changes only;
consumers ignore unknown keys.
