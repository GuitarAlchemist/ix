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

## Audit corrections (2026-05-12)

The initial subagent audit made three factual errors. All caught before implementation. Logged here per Article 7 (Auditability) — a wrong audit becomes a real bug if it drives unnecessary plumbing.

### IX — hooks already wired

Verified by reading `.claude/settings.json`:

- `governance-check.sh` is wired to PreToolUse Bash matcher.
- `pipeline-validate.sh` is wired to PreToolUse Write|Edit matcher.
- `rust-check.sh` is wired to PostToolUse Write|Edit matcher.
- `cache-lifecycle.sh` is **not** wired to SessionStart/SessionEnd — but it only emits stderr log lines, no state change. Cosmetic.
- `scheduled_tasks.lock` is a runtime process lock (`{"sessionId":"…","pid":…,"acquiredAt":…}`), not a config file.

**IX Pattern 3 verification is therefore complete.**

### TARS — Pattern 2 (memory) is rich, not sparse

Subagent audit said TARS "lacks `docs/solutions/`" and called memory "moderate." Wrong. Verified by directory listing `../tars/v2/docs/`:

- 30+ markdown files including `COMPOUND_EVOLUTION.md`, `RELEASE_NOTES_v2.0-alpha.md`, `RTX5080_Optimization_Report.md`, session summaries.
- Numbered topic sections `0_Vision`, `1_Getting_Started`, `2_Architecture`, `3_Roadmap`, `4_Integration`/`Research`, `5_Quality`, `6_Maintenance`, `7_Reference`, plus `QA/`, `architecture/`, `conversations/`, `demos/`, `research/`.
- `v2/_archive/session_notes/` has timestamped session logs.
- `v1/docs/post-mortem/` exists too.

**TARS Pattern 2 is strong** — different convention from IX (numbered sections instead of flat `docs/solutions/`) but functionally equivalent. G4 retracted.

### TARS — Pattern 1 has 6-hourly submodule sync

Subagent said "✗ No scheduled automation." Wrong. Verified by reading `../tars/.github/workflows/submodule-auto-update.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours UTC
```

This workflow auto-commits Demerzel submodule updates (≤10 commits) or auto-opens a PR (>10 commits). It's a real scheduled loop. G3 is narrowed: TARS has scheduled *sync* but no scheduled *quality/promotion-trend* loop.

## Gap matrix and classification

Classification against `self-modification-policy v1.0.0`:

| Gap | Repo | Cherny pattern | Policy bucket | Confidence | Owner |
|---|---|---|---|---|---|
| ~~G1: `.claude/hooks/*.sh` unwired~~ | IX | 3 | **already done** — audit correction above | n/a | n/a |
| G2: No IX-side recurring routines (telemetry sweep, invariant drift, branch hygiene) | IX | 1 | **with approval** — creates new agent routines | 0.7 | surface, do not implement |
| G3 (narrowed): TARS has 6-hourly submodule sync but no scheduled quality/promotion-trend loop | TARS | 1 | **with approval** — new tools | 0.6 | surface for TARS owner |
| ~~G4: TARS lacks `docs/solutions/`~~ | TARS | 2 | **retracted** — `v2/docs/` already rich | n/a | n/a |
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

## Cherny ↔ Demerzel mapping

Demerzel is a **superset** of Cherny's three-pattern framework, not merely compliant with it. Cherny is a tactical 3-step pattern (loop, memory, verify); Demerzel embeds those patterns inside a constitutional framework with confidence-gated autonomy, hexavalent logic, and append-only audit trails.

### Pattern 1 (scheduled /loop) — 8 scheduled workflows

In `../Demerzel/.github/workflows/`:

| Workflow | Cherny role |
|---|---|
| `streeling-daily.yml` | daily knowledge sweep |
| `demerzel-self-improvement.yml` (weekly Sun 15:17 UTC) | self-improvement cycle |
| `demerzel-capability-expansion.yml` | autonomous capability discovery |
| `demerzel-discussion-lifecycle.yml` | GH Discussions triage |
| `demerzel-ideation.yml` | idea generation |
| `demerzel-showcase.yml` | output publishing |
| `ga-chatbot-discussions.yml` | cross-repo chatbot triage |
| `seldon-plan.yml` | strategic planning loop |

Plus a Phase-4 cron stub inside `qa-architect-cycle.ixql` (`0 6 * * *`) waiting to be enabled in Phase 4.

### Pattern 2 (institutional memory / CLAUDE.md) — `LOG.md` + driver cycle

`../Demerzel/LOG.md` is the explicit Cherny analogue — auto-appended after every governance action, with a **COMPOUND** phase in the driver cycle:

```
WAKE → RECON → PLAN → EXECUTE → VERIFY → COMPOUND → PERSIST → SLEEP
```

This is structurally richer than Cherny's "log the mistake and write a prevention rule." COMPOUND distills durable insights; PERSIST writes them to `LOG.md` + constitutions (append-only) + policies. Sample LOG.md entry from cycle 001 (2026-03-20):

> Submodule staleness is #1 health drag — submodule-notify should create triggers
> ga has 50+ untracked files needing cleanup
> tars has 26 dependabot alerts (next critical priority)
> Demerzel CI failures are missing API keys, not code bugs

Each insight becomes input to future plan stages. Plus `CLAUDE.md`, `AGENTS.md`, personas, contracts, and `state/`.

### Pattern 3 (execution verification) — 20 IXQL pipelines

In `../Demerzel/pipelines/`:

| Pipeline | LOC | Cherny analogue |
|---|---|---|
| `shake-metafix-loop.ixql` | 240 | **literal Pattern 3** — inject → detect → fix → re-measure, Netflix Chaos Monkey + 5-level MetaFix |
| `ml-feedback-loop.ixql` | 188 | ML training feedback closed loop |
| `conscience-cycle.ixql` | 147 | ethics-check loop |
| `qa-architect-cycle.ixql` | 109 | verdict pipeline (**Phase 0 skeleton — G6 gap**) |
| `chaos-test.ixql` | — | chaos-injection catalog (11 injection points) |
| `governance-shake-test.ixql` | — | shake-testing |
| `algedonic-belief-monitor.ixql` | — | pain/pleasure signal monitor |
| `driver-cycle.ixql` | — | top-level WAKE→…→SLEEP orchestrator |
| `scheduled-research.ixql` | — | autonomous research loop |
| (10 more) | — | constitutional-evolution, governance-audit, governance-markov, content-intelligence, cross-pipeline-deps, hyperlight-orchestrator, lolli-lint, metasync, render-critic, resilience-dashboard |

### Where Cherny < Demerzel (structural deltas)

| Dimension | Cherny | Demerzel |
|---|---|---|
| Logic | binary pass/fail | hexavalent T/P/U/D/F/C |
| Autonomy | implicit (agent decides) | confidence-gated (≥0.9 auto, ≥0.7 note, ≥0.5 confirm, ≥0.3 escalate) |
| Reversibility | not mandated | Article 3 mandate |
| Audit trail | optional | Article 7 mandate, append-only constitutions |
| Stop condition | "until tests pass" | guardrails: max 3 mods/session, 5-min cooldown, test-pass before permanent |
| Memory substrate | `CLAUDE.md` (flat) | `LOG.md` + constitutions + policies + personas + state/ + driver-cycle COMPOUND phase |
| Verification primitive | run tests, read errors | 20 IXQL pipelines spanning chaos-injection, ML feedback, ethics, governance audit, conscience |

### Net verdict

**Demerzel is Cherny-compliant by construction, with one known stub.** `qa-architect-cycle.ixql` remains Phase 0 (hardcoded skeleton verdicts) — this is G6 in the gap matrix. Phases 1-4 are designed in `../ga/docs/plans/2026-05-04-chatbot-autonomy-action-layer.md` and shipping them closes the last fragment of Cherny Pattern 3 that's currently a stub.

The mapping is preserved here so that "is Demerzel Cherny-compliant?" doesn't need re-deriving next time. Short answer: yes, and then some.

## Appendix — copy-pasteable issue templates

When ready to push these to siblings, the bodies below are pre-shaped against the gap matrix.

### GA — `AGENTS.md` duplicates `CLAUDE.md`

```
Title: AGENTS.md is a verbatim copy of CLAUDE.md

The two files share an identical header and structure (verified 2026-05-12 by
`diff -q CLAUDE.md AGENTS.md` showing they differ only by recent edits in
progress). Cherny Pattern 2 expects AGENTS.md to define agent roster +
capabilities, not duplicate the human-oriented CLAUDE.md.

Suggested content for AGENTS.md:
- Available agent personas (chatbot, qa-architect, blast-radius classifier)
- MCP tools exposed
- Cross-repo handoff contracts (qa-verdict, voicings.payload, optick-sae-artifact)
- Confidence thresholds for autonomous vs advisory actions

Cross-ref: ix/docs/plans/2026-05-12-self-improvement-loops.md (G7)
```

### TARS — scheduled promotion-trend snapshot

```
Title: Add weekly scheduled job emitting promotion-index trend snapshot

TARS has 6-hourly submodule sync but no scheduled quality/learning loop.
The promotion index at ~/.tars/promotion/index.json carries scores, occurrence
counts, contexts — perfect input for a weekly trend artifact.

Proposed: .github/workflows/promotion-trend.yml running Sunday 06:00 UTC:
1. dotnet run --project src/Tars.Interface.Cli -- promote --snapshot
   (or equivalent — depends on CLI surface; verify `tars promote --help`)
2. Write artifact to v2/_archive/promotion_snapshots/YYYY-WW.json
3. Append one-line delta summary to v2/docs/5_Quality/PROMOTION_TREND.md

Mirrors ix/.github/workflows/ga-nightly-quality.yml pattern.

Governance: needs ix_governance_check before merge — new scheduled tool
falls under self-modification-policy v1.0.0 "with human approval."

Cross-ref: ix/docs/plans/2026-05-12-self-improvement-loops.md (G3 narrowed)
```

### TARS — `_archive/session_notes/` retention policy

```
Title: Define retention/promotion policy for _archive/session_notes/

The directory accumulates timestamped session logs but has no documented
retention policy or promotion path. Cherny Pattern 2 (institutional memory)
needs notes to be discoverable, not just stored.

Suggested:
- Quarterly review: promote durable lessons to v2/docs/5_Quality/Lessons-Learned.md
- Archive notes older than 6 months to v2/_archive/session_notes/archive/
- Add v2/docs/5_Quality/README.md describing the promotion path

Zero-risk doc-only change; no code touched.

Cross-ref: ix/docs/plans/2026-05-12-self-improvement-loops.md (G4 retracted but follow-up here)
```
