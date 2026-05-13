---
date: 2026-05-13
purpose: Manual triage of 15 open GA issues — the work Seldon should have done
reversibility: two-way door (read-only — no closes, no edits to GitHub state)
revisit-trigger: Seldon Plan resumes producing real artifacts (kill.switch in Demerzel removed) OR backlog passes 20 open issues OR user re-prioritizes
status: triage complete; recommendations only — user decides what to close/park/ship
---

# GA issue triage — 2026-05-13

Pulled from `gh issue list -R GuitarAlchemist/ga --state open` on 2026-05-13. 15 issues open. Read-only triage; no closes, comments, or edits performed against GitHub.

## Summary by category

| Category | Count | Age range | Recommended action |
|---|---|---|---|
| **A. Active decisions/bugs** | 3 | -1 to 6 days | decide / park / fix this week |
| **B. Prime Radiant panel buildouts** (Portfolio G1-G10 batch) | 8 | 36 days, all same day | sequence, ship one as Demerzel-proof |
| **C. ix integration carry-over** | 2 | 37 days | ix side ready, schedule |
| **D. Frontend perf wins** | 2 | 41 days | small, self-contained — ship |

## A. Active decisions/bugs (3)

### #204 — Decide support status for legacy chatbot REST/SSE routes
- Age: -1 day (freshest)
- Type: **decision needed**, not implementation
- Body: `/api/chatbot/chat` and `/api/chatbot/chat/stream` exist in parallel with AG-UI + SignalR public paths.
- **Recommended:** decide deprecation policy + sunset date this week, close with a one-line ADR. 5-minute action.

### #145 — Bug #6: ChatbotTestBase selectors don't match chatbot-demo.html
- Age: 6 days · related to #134
- Type: test infra mismatch after PR #140
- **Recommended:** either fix selectors in one PR (likely 1-2 hours) OR park to `ga/docs/parked/backlog.md` per `feedback_park_broken_then_rework.md` if chatbot-demo.html is being deprecated alongside #204.

### #134 — Playwright tests need React frontend (ga-client) running
- Age: 6 days · CI test-job precondition gap
- Type: infrastructure
- **Recommended:** decide if Playwright-in-CI is critical. Per `reference_ga_two_react_apps.md`, `ga-react-components` (port 5176) is the canonical app, not `ga-client` (5173). The issue may be obsolete — check whether tests should target ga-react-components instead.

## B. Prime Radiant panel buildouts — Portfolio G1-G10 batch (8)

All created within the same window (36 days old, labeled `enhancement`, all derived from "Portfolio improvement backlog G1-G10"). These are buildout items, not bugs:

| # | Backlog ref | Component | Data source | Estimated complexity |
|---|---|---|---|---|
| #48 | G1 | Unpark AdaptiveAI/AdvancedAI controllers | parked controllers | medium |
| #49 | G2 | GovernanceCompliancePanel | ix MCP tools | small-medium |
| #50 | G3 | Live GovernanceMetricsDashboard | Demerzel schemas | medium |
| #51 | G4 | WebSocket bridge Discord → Prime Radiant | demerzel-bot | medium-large |
| #52 | G5 | AgentSpectralPanel | /api/spectral/agent-loop | small-medium |
| #53 | G6 | KnowledgeGraphPanel | TARS MCP | medium |
| #54 | G7 | GodotSceneInspectorPanel | Godot MCP (24+ tools) | medium-large |
| #56 | G10 | Wire AdminInbox to Demerzel governance decisions | AdminInbox.tsx | small |

**Pattern:** bulk-imported as a batch 36 days ago; sat untouched. The auto-triage Seldon was supposed to do never happened because Seldon was stub.

**Recommended sequencing:**

1. **Ship #50 first** — Live GovernanceMetricsDashboard. Reason: directly counters the "Demerzel is dead" finding in `2026-05-12-self-improvement-loops.md`. Renders conscience digests, regrets, patterns. Visual proof of governance health (or absence). Small-medium, high signal.
2. **Then #49** — GovernanceCompliancePanel from ix MCP tools. Reuses `ix_governance_check` already proven working this session.
3. **Then #56** — AdminInbox to Demerzel governance, smallest of the batch, completes the governance UI triad.
4. **Defer #48, #51, #53, #54** — depend on out-of-repo systems (parked controllers, Discord bot, TARS, Godot). Re-evaluate after #50/#49/#56 prove the panel pattern.
5. **#52 AgentSpectralPanel** — orthogonal. Schedule independently when SpectralAnalyticsController is the priority.

## C. ix integration carry-over (2)

### #47 — Prime Radiant integrate ix governance.graph pipeline
- Age: 37 days
- ix side: **ready** (per `project_prime_radiant_integration.md`, governance.graph feeds 196-node graph to Prime Radiant 3D viz)
- **Recommended:** unblocked — schedule. Owner: GA-side.

### #46 — Port ix capability registry pattern to C#
- Age: 37 days
- ix side: **ready** (v0.2.0 shipped registry + 43 tools + 150 tests per `project_phase1_delivery.md`)
- **Recommended:** unblocked — schedule. Cross-ref `project_ecosystem_mirror.md` (~2-3 days for GA mirror).

## D. Frontend perf wins (2)

Both self-contained, no dependencies, multi-LLM-reviewed:

### #42 — InstancedMesh for governance nodes (+6-12 FPS)
- Identified as #1 perf bottleneck (~540 draw calls collapsible to ~7).
- **Recommended:** ship in next perf PR. Highest FPS gain per LOC.

### #43 — Bake skybox shader to static cubemap (+2-4 FPS)
- Procedural fBm renders every frame; output never changes.
- **Recommended:** ship alongside #42.

Combined: +8-16 FPS, ~1 day of work.

## Cross-cutting observations

1. **All "P1 backlog" issues are the same age (36 days)** because they were imported as a batch, not because they actively pile up. The user's "issues are piling up" framing is correct in aggregate but the *flow rate* is closer to 0 — issues land in batches and sit. Seldon being stub means no continuous triage pressure.
2. **Three issues (#204, #145, #134) are about the chatbot REST surface.** Solving #204 (decide deprecation) likely makes #145 and #134 trivial or moot. One decision unblocks three issues.
3. **ix-side dependencies for #47 and #46 are already shipped** — the bottleneck is GA-side scheduling, not cross-repo coordination.
4. **The 8 portfolio items (#48-#56) are buildout, not maintenance.** Conflating them with bugs (#145, #134) inflates the "issues piling up" perception.

## What this triage does NOT do

- Does not close, comment, label, or assign anything on GitHub. Recommendations only.
- Does not write to any GA-repo file (GA has uncommitted edits to AGENTS.md + CLAUDE.md by another session as of 2026-05-12).
- Does not block on user response — recommendations stand as-is; user picks which to action.

## Recommended next actions (in order of effort/impact)

1. **5 min:** decide #204 (REST/SSE deprecation policy).
2. **30 min:** check if #134's `ga-client` reference is obsolete (should target `ga-react-components` per memory).
3. **1 day:** ship #42 + #43 perf wins.
4. **2-3 days:** ship #50 GovernanceMetricsDashboard as the visible-proof-of-Demerzel item.
5. **1 day:** #49 GovernanceCompliancePanel from ix MCP.
6. **1 day:** #56 AdminInbox wiring.

After (1) through (6): backlog drops from 15 → 9, and the remaining 9 are deferable buildouts (B-tier) or perf scheduled (D-tier).

Cross-ref:
- `ix/docs/plans/2026-05-12-self-improvement-loops.md` — green-but-dead finding that motivated this manual triage.
- `Demerzel/state/seldon-plan/kill.switch` — Seldon paused 2026-05-13 pending wire-up of the commented-out Claude invocation.
