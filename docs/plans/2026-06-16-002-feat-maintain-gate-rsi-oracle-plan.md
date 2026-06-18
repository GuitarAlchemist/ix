---
title: "maintain-gate — a fail-closed RSI evaluation oracle over the four DuckDB+IX lenses"
type: feat
status: active
date: 2026-06-16
---

# maintain-gate — a fail-closed RSI evaluation oracle

## Overview

A single, deterministic, fail-closed verdict — *should this self-improvement
iteration be accepted, rejected, or escalated?* — computed by fusing the four
DuckDB+IX maintain lenses (`chatbot` regression, `routing`/yield, `loops`
convergence, `ood` drift) into one hexavalent contract that a bounded
recursive-improvement (RSI) loop and Demerzel both read.

DuckDB+IX is the **referee, never the player**: ground truth stays executable
(compile/test/proof); the gate only *aggregates evidence about it* into a verdict
that is hard to fake by construction (it reads append-only ledgers it does not own).

## Problem statement / motivation

IX already encodes the bounded-RSI loop the 2025–26 field converged on
(propose → executable oracle → govern → human-merge; validated, see
`~/.claude` memory `reference_safe_rsi_loop_validated`). The systems that got
caught reward-hacking (DGM faked its own test logs) removed the human gate and
trusted a **weak oracle**. The leverage is therefore the oracle + observability,
not the "improve" step.

Today IX has the *sensors* but no *oracle*:

- `ix_duck::chatbot::check_regressions` — the held-capability guardrail (Slice B). **Live.**
- `ix_duck::routing` + `state/thinking-machine/hits.jsonl` yield — the improvement metric. **Live.**
- `ix_duck::loops` — convergence / oscillation / reward-hack thrash detector. **Dormant** (Contract A: GA/IX loop dir is seed-only until a real loop writes rows).
- `ix_duck::ood` — distribution-drift detector. **Dormant** (Contract B: query vectors not persisted yet).

They are queried ad hoc. Nobody fuses them into one stop/continue/merge verdict, so:

- **The RSI loop** can't safely decide when to stop or whether a change is *real*
  improvement vs metric-gaming.
- **Demerzel** has no single evidence artifact to govern on.
- **The human reviewer** must hand-cross-read four lenses per iteration.

### Who is in pain
- *Validated:* the AFK/auto-optimize loop (no convergence/stop oracle).
- *Validated:* the human merge-reviewer (manual 4-lens cross-read).
- *Assumed:* Demerzel governance wants a structured verdict rather than raw lens output.

## Proposed solution

A new `duck`-gated module `ix_duck::maintain` that mirrors the proven Slice-B
gate shape (`evaluate` → `MaintainVerdict` → `write_contract` → `exit_code`), plus
an `ix_maintain_gate` example CLI and a frozen-at-Phase-4 contract.

The verdict is a **conjunction, not a single metric** — the core anti-Goodhart move:

```
ACCEPT  iff  metric ↑   AND  guardrail held   AND  not oscillating   AND  not drifting
```

…rendered as a **hexavalent** status (T/P/U/D/F/C — per ecosystem logic,
`feedback_hexavalent_logic`), fail-closed: any lens that errors, is absent, or
returns UNKNOWN caps the verdict at U/D — it can never silently read as ACCEPT.

### Verdict mapping (T/P/U/D/F/C)
| Status | Meaning | Trigger |
|---|---|---|
| **T** (true) | accept | metric↑ ∧ guardrail held ∧ converging ∧ in-distribution |
| **P** (probable) | accept w/ soft flags | all hard checks pass, ≥1 soft advisory |
| **U** (unknown) | escalate to human | a required lens absent/dormant or returns no signal |
| **D** (disputed) | escalate to human | lenses disagree (e.g. metric↑ but oscillating) |
| **F** (false) | reject | guardrail broke, or drift/regression detected |
| **C** (contradiction) | reject + alarm | metric↑ **and** guardrail broke = classic reward-hack signature |

The **C** case is the one that matters most: a number going up *while* a held
capability breaks is the Goodhart signature, and it must hard-fail + alarm, never
average out to "net positive".

## Technical approach

### Architecture

```
crates/ix-duck/src/maintain.rs        (new, #[cfg(feature="duck")])
  pub struct MaintainConfig   { yield_floor, guardrail_max_fail, ood_k, ood_threshold,
                                max_oscillation, require_loops, require_ood, ... }
  pub struct MaintainVerdict  { schema_version, run_at, status (hexavalent),
                                metric_delta, guardrail_held, converging, drifting,
                                signals: Vec<Signal>, sources: SourceProvenance }
  pub struct Signal           { lens, severity (hard|soft), value, threshold, verdict }
  pub fn evaluate(conn, cfg, sources) -> Result<MaintainVerdict, MaintainError>
  pub fn write_contract(&MaintainVerdict, path) -> io::Result<()>     // append-only ledger
  pub fn exit_code(status) -> i32                                     // 0=T/P, 2=U/D, 1=F/C

crates/ix-duck/examples/ix_maintain_gate.rs   (new)   gate | explain
docs/contracts/maintain-gate.contract.md      (new)   + maintain-gate.schema.json
crates/ix-duck/tests/fixtures/maintain/       (new)   pass / reward-hack / dormant scenarios
```

`evaluate()` composes the existing lens functions — **no lens logic is
reimplemented**:
- guardrail ← `chatbot::check_regressions(...).status`
- metric  ← yield over `hits.jsonl` split by `ts_ms` (NOT the cumulative mean —
  see `reference_dogfood_yield_measurement_gotcha`; before/after must be split)
- converging ← `loops::oscillating_loops` empty ∧ `loops::loop_summary` net_delta ≥ 0
- drifting ← `ood::flag_ood(...)` non-empty over the iteration's queries

### Tamper-evidence (necessary, not optional — the DGM lesson)
- The verdict is appended to `state/thinking-machine/maintain-gate.jsonl`
  (append-only; the gate writes, never rewrites).
- v1: record the `baseline_ref` FNV hash already emitted by the chatbot gate so a
  silent baseline swap is detectable.
- v2 (deferred, flagged below): constitution checksum fail-close
  (`reference_safe_rsi_loop_validated` gap #2).

### What the gate must NOT do
- **Not** be the ground truth. It reads test/ledger evidence; it never replaces
  `cargo test`/compile/proof.
- **Not** use an LLM panel as the decision — ~96% TPR / <25% TNR rubber-stamps
  drift (`feedback_llm_judge_panel_failclosed`). LLM allowed only as a fail-closed
  *advisory* soft signal, never a hard gate.
- **Not** mutate any source of truth — DuckDB stays in-memory analyst bench.
- **Not** average across the conjunction — a broken guardrail is not offset by a
  metric gain (that's the **C** case).

## Implementation phases

### Phase 0 — Tracer bullet (the vertical slice)
The smallest end-to-end that touches every layer, per aihero discipline:
`evaluate()` fusing **only two** lenses (yield metric ∧ chatbot guardrail) →
`MaintainVerdict` → `write_contract` → `exit_code`, tested against:
- [ ] a **pass** fixture (metric↑, guardrail held) → **T**
- [ ] a **reward-hack** fixture (metric↑, guardrail broke) → **C**, exit 1
This proves the anti-Goodhart conjunction before scaling.

### Phase 1 — Add convergence + drift
- [ ] Fold in `loops` (converging) and `ood` (drifting) signals.
- [ ] Graceful-degrade: dormant lens (absent data) → that signal is **U**, which
      caps the overall verdict at **U** (escalate), never silently ACCEPT.
- [ ] `require_loops`/`require_ood` config: when false, a dormant lens is advisory
      not blocking (so the gate is usable today over the live lenses only).

### Phase 2 — Contract + tamper-evidence
- [ ] `docs/contracts/maintain-gate.contract.md` + JSON schema (v0.1 draft).
- [ ] Append-only ledger write + `baseline_ref` capture.
- [ ] `ix_maintain_gate` example: `gate` (verdict + exit code) and `explain`
      (per-signal breakdown).

### Phase 3 — Wire consumers
- [ ] Demerzel governance reads the verdict (existing governance-check path or a
      new MCP tool — **open question**, see below).
- [ ] Document the loop integration: an RSI loop calls `gate` after each iteration;
      exit 1 → reject + revert, exit 2 → human escalate, exit 0 → propose-merge.

### Phase 4 — Freeze (one-way door)
- [ ] Freeze contract schema + exit-code semantics only here. Log sign-off.

## Instrument before you ship

- **Baseline:** none today (no fused verdict exists).
- **Primary metric:** the gate's **true-negative rate on the reward-hack fixtures**
  — it MUST reject every metric-up-guardrail-broke case. (The LLM-panel lesson
  generalizes: measure a gate by its TNR, not by confirming it passes good cases.)
- **Guardrail:** false-reject rate on the pass fixtures must stay 0 (a gate that
  cries wolf gets disabled).
- **Direction:** TNR → 1.0 on the adversarial fixture set; FRR = 0 on pass set.

## Reversibility (log one-way doors)

| Decision | Door | Revisit trigger |
|---|---|---|
| Contract schema + exit-code semantics | **one-way** (Demerzel + loop depend) | freeze at Phase 4; change needs sign-off |
| Conjunction (no averaging) anti-Goodhart rule | **one-way** (the safety property) | only if a formal proof shows it over-rejects |
| Which lenses compose the verdict | two-way | add/remove a lens freely pre-freeze |
| Hexavalent vs boolean status | two-way | — |

## Open questions
1. **Yield source in-loop:** ~~lean: use the ledger's own metric~~ **REVERSED by panel
   review (2026-06-16, Codex + Gemini, unanimous).** A self-declared `metric_delta`
   written by the proposing agent is *the player scoring its own game* — accepting it
   recreates the reward-hacking path the gate exists to catch. **Decision: the metric
   must be externally derived** (`hits.jsonl` split-by-`ts_ms`, harness-written) OR a
   ledger metric that carries a *source pointer to independently recomputable evidence
   + a content hash*. Never trust a bare agent-declared delta.
2. **Demerzel consumption:** new MCP tool (`ix_maintain_gate`) vs feeding the
   contract through the existing governance-check. *Assumed: start with the JSON
   contract on disk, add a tool only if demanded.*
3. **Constitution checksum:** v1 or deferred to v2? *Lean: v2 — keep Phase 0–2 to
   the lens fusion; checksum is an independent guardrail.*
4. **Whose loop:** v1 targets IX's own bounded-RSI ledger; GA's loop feeds the same
   gate once Contract A/B emit (the GA-session brief covers that emission).

## Panel review consensus (2026-06-16, Codex gpt-5.5 + Gemini)

Both independent reviewers: **`build-phase0-now`**. Both named the same
highest-value angle (conjunction + treat metric↑∧guardrail↓ as **C**, never an
average). Their convergent flip-condition — *the gate is only as trustworthy as the
integrity of the inputs it reads* — drives three plan amendments:

1. **Input integrity (input provenance > verdict provenance).** The load-bearing
   tamper-evidence isn't hashing the *verdict* — it's that the *evidence* be
   non-forgeable. Metric must be externally derived (Q1, reversed). Hash the metric
   evidence + guardrail evidence + policy context, not just the verdict line.
2. **Write-isolation (Gemini's strongest catch).** If the proposing agent's sandbox
   can write the telemetry files (`hits.jsonl`, guardrail logs, the ledger) before
   DuckDB reads them, it will learn to forge inputs and the referee is neutralised.
   → **Go condition:** the harness must enforce that the proposer cannot write the
   ledgers the gate ingests. This is a *prerequisite for the gate going authoritative*
   (Phase 3), **not** a Phase-0 blocker — Phase 0 runs over static vendored fixtures.
3. **Iteration correlation, not wall-clock.** Joining disparate ledgers by timestamp
   risks evaluating a guardrail drop against the wrong iteration's metric → false or
   missed **C** verdicts. → Join lenses by an explicit `iteration_id` / `commit_sha`
   key, never by `ts`.

Both also flagged a **fail-closed deadlock**: "any dormant lens → U" would make
Phase 0 permanently U while `loops`/`ood` have no data. Resolved by the existing
`require_loops`/`require_ood` config — **Phase 0 sets both false** (fuses only the two
live lenses); dormant lenses are advisory, not blocking, until their data emits.

Q2 (disk contract first) and Q3 (constitution checksum → v2) **confirmed** — with
Codex's caveat that v2 is acceptable *only while Phase 0 is explicitly
non-authoritative/tracer-only*; the checksum must precede the gate gaining authority.

## Acceptance criteria
- [ ] `evaluate()` returns **C** (reject, exit 1) for the reward-hack fixture and
      **T** (accept, exit 0) for the pass fixture.
- [ ] A dormant/absent lens yields **U** (escalate), never silent ACCEPT; Phase 0
      sets `require_loops=require_ood=false` so it does not deadlock on U.
- [ ] Metric is externally derived (not a bare agent-declared `metric_delta`); the
      verdict records the metric/guardrail **evidence** provenance (source pointer +
      content hash), not just the verdict line.
- [ ] Lenses are correlated by `iteration_id`/`commit_sha`, never wall-clock `ts`.
- [ ] No lens logic reimplemented — `maintain` only composes existing functions.
- [ ] Verdict appended to an append-only ledger with `baseline_ref`.
- [ ] All behind `duck`; default workspace build unaffected; clippy + fmt clean.
- [ ] Contract doc + schema; `ix_maintain_gate` example runs over live `../ga` and IX state with graceful degrade.

## Sources & references
- Bounded-RSI validation + the two gaps (hits.jsonl pair, constitution checksum):
  `~/.claude` memory `reference_safe_rsi_loop_validated`.
- Yield measurement gotcha (split by ts, not cumulative mean):
  `reference_dogfood_yield_measurement_gotcha`.
- LLM panels are fail-closed gates, not oracles: `feedback_llm_judge_panel_failclosed`.
- Hexavalent logic: `feedback_hexavalent_logic`.
- Gate pattern mirrored from: `crates/ix-duck/src/chatbot.rs` (`GateReport` /
  `check_regressions` / `write_contract` / `exit_code`, Slice B) and
  `docs/contracts/chatbot-trace-regression.contract.md`.
- Lenses fused: `crates/ix-duck/src/{chatbot,routing,loops,ood}.rs`;
  inventory in `docs/DUCKDB.md`.
- This session's lens build: commit `ddff2bb` (loops + ood) on
  `feat/ix-duck-loop-ood-lenses`.
