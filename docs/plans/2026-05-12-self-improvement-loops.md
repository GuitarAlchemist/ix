---
date: 2026-05-12
reversibility: two-way door (each loop is independently revertible; cross-repo additions surface as separate PRs)
revisit-trigger: any one of the proposed loops reaches a self-modification rate that would breach the 3-mod/session guardrail in `self-modification-policy v1.0.0`, OR Demerzel constitution moves past v2.2.0, OR a new repo joins the ecosystem
status: design — IX side is already wired (audit correction below); cross-repo gaps surfaced for owners
governance: ix_governance_check returned `compliant: true` on both proposed action classes against Demerzel constitution v2.2.0 (2026-05-12)
---

# Self-improvement loops across the GuitarAlchemist ecosystem

## Problem

Boris Cherny's framing (Anthropic, 2026) names three loop patterns. The ecosystem (ix, ga, tars, Demerzel) carries pieces of all three but has uneven coverage. This plan audits the gap, classifies each candidate addition against `self-modification-policy v1.0.0`, and authorizes only the safest IX-side wiring for this session.

## The three patterns

1. **Scheduled developer automation** — recurring agents that produce artifacts on a cron, no real-time human prompts.
2. **Repository guard / institutional memory** — `CLAUDE.md` + `MEMORY.md` + `docs/solutions/` capture every steering correction and replay it at session start.
3. **Execution verification** — closed loop where the agent runs tests/lints, reads its own errors, fixes, retests.

## Current state (audited 2026-05-12)

| Pattern | IX | GA | TARS | Demerzel |
|---|---|---|---|---|
| 1 (scheduled) | `ga-nightly-quality.yml` daily 05:15 UTC | `gemini-scheduled-triage.yml` hourly + `quality-snapshot.yml` on push | **none** | `demerzel-self-improvement.yml` weekly Sun 15:17 UTC |
| 2 (memory) | CLAUDE.md + `docs/solutions/` + 54-entry user auto-memory | CLAUDE.md + 7 solutions + dated plans | CLAUDE.md (sparse) | CLAUDE.md + AGENTS.md + 44 policies |
| 3 (verification) | 4 unwired hooks in `.claude/hooks/` | `.githooks/pre-commit` (dotnet format + build) | `dotnet test` on push/PR only | 22 workflows, IXQL pipelines, schema validation |

## Audit correction (2026-05-12)

Initial subagent audit asserted IX's `.claude/hooks/*.sh` were "unwired" and `.claude/scheduled_tasks.lock` was "empty." Both wrong — verified by reading `.claude/settings.json`:

- `governance-check.sh` is wired to PreToolUse Bash matcher.
- `pipeline-validate.sh` is wired to PreToolUse Write|Edit matcher.
- `rust-check.sh` is wired to PostToolUse Write|Edit matcher.
- `cache-lifecycle.sh` is **not** wired to SessionStart/SessionEnd — but it only emits stderr log lines, no state change. Cosmetic.
- `scheduled_tasks.lock` is a runtime process lock (`{"sessionId":"…","pid":…,"acquiredAt":…}`), not a config file.

**IX Pattern 3 verification is therefore complete.** Three of the four hooks fire on every relevant Claude Code tool invocation. The cache-lifecycle wiring is a no-op gap not worth implementing.

This correction is logged here per Article 7 (Auditability) — a wrong audit becomes a real bug if it drives unnecessary plumbing.

## Gap matrix and classification

Classification against `self-modification-policy v1.0.0`:

| Gap | Repo | Cherny pattern | Policy bucket | Confidence | Owner |
|---|---|---|---|---|---|
| ~~G1: `.claude/hooks/*.sh` unwired~~ | IX | 3 | **already done** — audit correction above | n/a | n/a |
| G2: No IX-side recurring routines (telemetry sweep, invariant drift, branch hygiene) | IX | 1 | **with approval** — creates new agent routines | 0.7 | surface, do not implement |
| G3: TARS has no scheduled automation | TARS | 1 | **with approval** — new tools | 0.7 | surface for TARS owner |
| G4: TARS lacks `docs/solutions/` | TARS | 2 | **without approval** — directory + README | 0.85 | surface for TARS owner |
| G5: GA `qa_score_chatbot_drift` TODO | GA | 1 + 3 | already designed in `2026-05-04-chatbot-autonomy-action-layer.md` | n/a | GA owner (existing plan) |
| G6: Demerzel `qa-architect-cycle.ixql` Phase 0 only | Demerzel | 1 + 3 | already designed in same chatbot-autonomy plan | n/a | Demerzel owner (existing plan) |
| G7: GA `AGENTS.md` duplicates `CLAUDE.md` | GA | 2 | **without approval** — hygiene | 0.9 | surface for GA owner |
| G8: No cross-repo trace feedback loop (IX↔TARS, GA→IX) | all | 1 | **with approval** — new tools + new contract | 0.5 | surface; needs contract design |

**Net IX-side work for this session: zero.** All remaining gaps are owned by sibling repos or require explicit human approval per the self-modification policy.

## G2–G8 — surfaced for owners (NOT implemented here)

Each one would benefit from its own dated plan + governance check. Specifically:

- **G2 (IX scheduled routines)** — Three candidate routines, all draft-only, never auto-merge:
  - `hourly /loop` reading `state/telemetry/voicing-search/*.jsonl` → flag concrete bugs to `docs/solutions/` as advisories.
  - `daily /loop` running `ix-invariant-coverage` + catalog drift detector → write to `state/quality/` snapshot.
  - `weekly /loop` scanning stale branches → emit `docs/parked/branch-pruning-YYYY-WW.md`.
  - **Blocker:** these create agent routines that fall under "with human approval." Need explicit go-ahead per routine.

- **G3, G4 (TARS gaps)** — Need TARS owner. Recommend: `dotnet.yml` extension with weekly grammar-promotion summary cron; new `docs/solutions/` directory seeded from `~/.tars/promotion/index.json` deltas.

- **G5, G6** — Already covered by `2026-05-04-chatbot-autonomy-action-layer.md`. No new plan needed; that plan estimates 2.5 days of GA + Demerzel work.

- **G7 (GA AGENTS.md)** — One-PR hygiene fix in GA.

- **G8 (cross-repo trace feedback)** — Biggest design work. Would require a new contract `trace-feedback-v0.1` analogous to `optick-sae-artifact.contract.md`. Recommend deferring until G5/G6 ship and produce telemetry that justifies the contract shape.

## Sequencing

1. **This session:** this plan (with audit correction).
2. **Next session, after explicit approval:** G2 (IX scheduled routines), one routine at a time, governance-check each.
3. **GA-owned (existing plan):** G5, G6, G7 — 2.5 days per the chatbot-autonomy plan.
4. **TARS-owned:** G3, G4 — separate PR in tars/.
5. **Deferred:** G8 — wait for G5/G6 telemetry.

## What NOT to build (deliberate)

- **No auto-merge anywhere.** Per the chatbot-autonomy plan's explicit decision and Article 6 (Escalation).
- **No indefinite cloud loops.** Cherny mentions "indefinitely in the cloud" but local execution + GitHub Actions cron is sufficient and respects the 3-mod/session guardrail.
- **No self-modifying skill definitions.** Skill/tool schema changes require human approval per policy 1.0.0.
- **No new MCP servers.** Existing tools (ix_governance_check, ix-quality-trend, ix-invariant-coverage, sentrux) are sufficient.

## Memory entries to write after this plan lands

- `feedback_audit_before_building.md` — "Always read settings.json before claiming hooks are unwired. The subagent audit asserted `.claude/hooks/*.sh` were unwired; reading `.claude/settings.json` showed 3 of 4 were already wired. Lesson: file existence ≠ wiring; check configuration before proposing plumbing."
- `project_self_improvement_loops_status.md` — "IX Pattern 1/2/3 all wired. G2–G8 remaining gaps owned by sibling repos. See `docs/plans/2026-05-12-self-improvement-loops.md`."
