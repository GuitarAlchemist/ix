# LEARNING.md — the Learn → Ship → Steer loop

> A personal operating system for learning this project (IX + GA + the ecosystem),
> its languages and methodologies, and contributing to steering GA-chatbot development.
> It invents nothing new — it *composes* skills and disciplines that already exist in
> this repo. The one rule: **every learning turn produces a real, shipped artifact.**

## The loop

```
        ┌─────────────┐     ┌──────────────────────────┐     ┌─────────────────────┐
        │   LEARN     │ ──▶ │          SHIP            │ ──▶ │       STEER         │
        │  /teach     │     │ brainstorm→/ce-plan→     │     │ telemetry → weakest │
        │ (one slice) │     │ /ce-work (tests+plan)    │     │ → fix → measure     │
        └─────────────┘     └──────────────────────────┘     └─────────────────────┘
              ▲                                                          │
              └──────────────── /digest · /learnings · /correct ◀───────┘
                                (compound across sessions)
```

### 1. Learn — `/teach <slice>`
The `/teach` skill builds a MISSION, curates resources, runs lessons with self-checks,
and keeps **progress-tracked learning-records that persist across sessions**.
- Learn **artifact-first**, never abstractly. Don't "learn Rust" — `/teach` the exact
  slice you need to touch a crate (`/teach rust-for-ix-crates`, `/teach ga-chatbot-fsharp`).
- One slice at a time. A slice is small enough to apply the same day.

### 2. Ship — `brainstorming → /ce-plan → /ce-work`
The standard contribution pipeline (the same one used to build `ix-duck`):
- **brainstorming** — explore the decision space; surface options, don't guess.
- **/ce-plan** — write a plan in `docs/plans/` with reversibility + revisit trigger
  (log one-way doors). Cross-check external APIs before finalizing.
- **/ce-work** — implement following existing patterns, with tests; doc/examples get tests.
- Honor the **Karpathy 4 rules** (think before coding, simplicity first, surgical changes,
  goal-driven) and **IDSD** (declare Intent + Expectations, get approval, before coding).

### 3. Steer — telemetry-driven prioritization (the GA-chatbot focus)
Don't guess what to improve; let the data choose. The cycle: **read telemetry → find the
weakest behavior → drive a fix through step 2 → measure before/after** (the "instrument
before you ship" rule: baseline + expected direction + guardrail).
- Steering instruments that already exist:
  - GA chatbot eval harnesses: the `adversarial-qa` skill, the AFK harness, the router
    eval harness (`project_ga_router_eval_tune_loop`).
  - Telemetry JSONL: voicing search telemetry, `chatbot-qa` quality snapshots
    (`ga/state/quality/`), IX thinking-machine `state/thinking-machine/{hits,gaps}.jsonl`.
  - Live inspection: GA Prime Radiant is MCP→SignalR remote-controllable (not Playwright).
- **DuckDB is the read-tool for this step** (see `docs/DUCKDB.md`): query the telemetry
  in-memory, join across datasets, surface the weakest intent/voicing/behavior.

### Wrap every session — the Cherny continuity loops
- `/digest` — capture cursor/in-flight/hypotheses to `state/digests/latest.md` (survives compaction).
- `/learnings` — capture surprises into `docs/solutions/<category>/`.
- `/correct` — turn a correction into a permanent rule in `CLAUDE.md`.
These make the loop **compound** instead of evaporate.

## Languages — learned on demand, not as courses
| Stack | Where | First slice |
|-------|-------|-------------|
| **Rust** | IX (this repo, 70+ crates) | `/teach rust-ownership-via-ix-math` then read one crate you'll change |
| **C# / F# / React** | GA (`../ga/`) — music theory + chatbot | `/teach ga-chatbot-architecture` |
| **F#** | tars (`../tars/`) — grammar + metacognition | only when a cross-repo contract needs it |
Always tie the language slice to the artifact you're about to ship.

## How to run a cycle (the checklist)
1. Pick the **weakest behavior** from telemetry (Steer) — or the next dependency a goal needs.
2. `/teach` the smallest slice that unblocks it. Do the self-checks.
3. `brainstorming` the change → `/ce-plan` → `/ce-work` (with a baseline metric declared).
4. Measure before/after. If it moved the metric the right way without tripping the guardrail, ship.
5. `/digest` + `/learnings` (+ `/correct` if you corrected the agent).

## First instance: DuckDB for GA + IX
DuckDB is both the topic to learn *and* the instrument for Steer. Course arc
(`/teach DuckDB for GA + IX`): L1 query `hits.jsonl` → L2 joins over IX telemetry →
L3 read GA-chatbot telemetry → L4 the `ix-duck` `ix_cosine` UDF over voicing embeddings →
L5 capstone: a steering query that surfaces the chatbot's weakest intent. By the end of
learning DuckDB, you can steer the chatbot — learning and contribution are the same activity.

## See also
- `docs/DUCKDB.md` — the DuckDB hub (tools, decisions, the analyst bench).
- `CLAUDE.md` — Collaboration discipline, Karpathy 4 rules, Cherny loops, session-learned rules.
- `README.md` — the 70+ crate map by domain.
