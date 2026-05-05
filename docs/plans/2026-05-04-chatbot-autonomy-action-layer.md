---
date: 2026-05-04
reversibility: two-way-door (Phase 3 PR/issue plumbing); one-way-door (daily snapshot retention policy + drift baseline format once consumers exist)
revisit-trigger: qa-architect-cycle reaches Phase 4 (algedonic monitor consuming derived signals), OR drift artifacts grow past 90 days and retention policy becomes load-bearing, OR a second producer (besides ga-chatbot baseline) wants to write to state/quality/chatbot-baseline-*.json
status: design — Phase 1 blast-radius analyzer shipped (commit pending). Phase 3 + daily snapshot are the next two action-layer pieces.
---

# Chatbot autonomy — close the eval→action loop

## Problem

The GA chatbot has a strong **detection layer**: 83-prompt adversarial corpus, 3-layer deterministic eval (sanitize / corpus-grounding / confidence), Octopus judge panel, Playwright suite, qa-architect-cycle pipeline, baseline JSONs in `ga/state/quality/chatbot-baseline-*.json`. None of it autonomously **acts** on what it finds.

Concrete proof point: `ga/state/quality/chatbot-baseline-2026-05-03.json` contains 3 priority-ordered fix recommendations including the one-line `appsettings.Development.json` Ollama-model swap that would unwedge GaApi. Detected, written to disk, sat unactioned.

Two missing pieces, sized for separate landings:

1. **Phase 3 action layer** — turn structured `qa-verdict.followups` into draft PRs / GitHub issues (cross-repo).
2. **Daily snapshot cadence + drift detection** — extend the one-shot 2026-05-03 chatbot baseline into a daily timeseries with `qa_score_chatbot_drift` analogous to the existing SAE drift tool.

## Phase 3 — verdict-to-PR action layer

### Producers (already shipped or in flight)

| Producer | Output | Status |
|---|---|---|
| `qa-architect-cycle.ixql` Phase 0 | hardcoded skeleton verdict | shipped |
| `ix-blast-radius` | `blast_radius` field of qa-verdict | shipped (commit pending this PR) |
| Tribunal aggregation (Phase 2) | filled `reviewer_chain` | TODO, not in scope here |

### Consumer to build: `qa-verdict-actor`

Repo: **GA** (it's a GH-API-heavy task; GA already owns `gh` CLI flow + authentication). New project: `Apps/GaQaVerdictActor` (Worker service).

**Input:** `state/quality/verdicts/<repo>/<ref>/<verdict_id>.json` matching `docs/contracts/qa-verdict.schema.json`.

**Decision tree per `followup` entry:**

```
followup.severity ∈ {must_fix, should_fix}  AND
followup.proposed_test is not null          AND
followup.location is not null                AND
classify(followup) ∈ {config_edit, single_file_patch}
  → open a draft PR with the proposed change

followup.severity ∈ {must_fix, should_fix}  AND
NOT eligible for auto-PR
  → open a GitHub issue, link verdict_id, assign owner=null

followup.severity == nice_to_have
  → append to ga/docs/chatbot/Chatbot_Backlog.md (no GH activity)

followup.severity == info
  → log only
```

**Classifier `classify(followup)`:**

- `config_edit` — `location` matches `appsettings*.json`, `*.toml`, `*.yaml` env files, AND rationale contains pattern matchable to one of: model-name swap, env var setting, threshold bump
- `single_file_patch` — `proposed_test` describes a localized assertion; location is one file; rationale describes a deterministic before/after

Anything else → `complex` (issue, not PR).

### Why GA owns this, not IX

- GA already authenticates `gh` for cross-repo PR creation
- The chatbot-baseline producer that emits these followups runs in GA (`Apps/GaChatbot.Api`)
- Demerzel `qa-architect-cycle.ixql` shells out — it can call GA's actor binary the same way it calls `ix-blast-radius`

### Reversibility

Two-way door at the actor level (delete the actor + revert one ixql line). The PRs/issues it creates are visible artifacts, but agents and humans can close them. The classifier rules are reversible code.

The **one-way door** inside this work: any followup we auto-merge (not draft) without human review changes the production code state. **Decision: never auto-merge.** v1 actor opens drafts only; humans flip to mergeable. Revisit when actor has shown 30 days of correct draft proposals.

## Daily snapshot + drift detection

### Producer pattern (mirror SAE)

`ix-optick-sae` writes daily artifacts to `ga/state/quality/optick-sae/YYYY-MM-DD/optick-sae-artifact.json` and `ga/qa_score_quality_drift` (Phase 2, GA commit `56384f2b`) consumes consecutive artifacts for drift.

Same shape for chatbot:

```
ga/state/quality/chatbot-baseline/YYYY-MM-DD/chatbot-baseline.json
ga/docs/contracts/chatbot-baseline.schema.json   ← new
```

Contract additions vs. the 2026-05-03 one-shot:

- `schema_version: "0.1.0"` (was `"0.1.0"` already — keep)
- `baseline_window: { start_iso, end_iso, prompt_count }` so drift can aggregate
- Per-prompt block stays as-is (already structured well)
- New top-level `aggregates`: `{ success_rate, p50_latency_ms, p95_latency_ms, fail_count_by_category }`

### Consumer to build: `qa_score_chatbot_drift` MCP tool

Repo: **GA** (matches sibling `qa_score_quality_drift` in `Apps/GaQaMcp/Tools/`).

Drift tolerances (calibrated from one prior data point — refine after 7 days of real snapshots):

| Metric | Tolerance | Outcome on breach |
|---|---|---|
| `success_rate` | absolute drop > 0.10 | concern |
| `p95_latency_ms` | relative increase > 50% | concern |
| `fail_count_by_category` | new category appears OR doubling | concern |

### Cadence

Daily at 06:00 UTC, mirroring `qa-architect-cycle.ixql` Phase 4 cron placeholder. Snapshot writer runs in GA's existing scheduled-jobs harness.

### Reversibility

Schema is a one-way door once consumers exist. **Decision: schema_version 0.1.x is draft until Phase 3 actor consumes it; bump to 1.0.0 only when a second consumer arrives.** Until then, additive-only changes (new optional fields), no breaking renames.

## Sequencing

1. Land this plan (commit pending)
2. GA: scaffold `Apps/GaQaVerdictActor` + classifier + draft-PR creator (~1 day, GA-side)
3. GA: extend `Apps/GaChatbot.Api` snapshot endpoint to write daily artifacts + add `chatbot-baseline.schema.json` (~0.5 day)
4. GA: `qa_score_chatbot_drift` MCP tool consuming the daily artifacts (~0.5 day)
5. Demerzel: wire `qa-architect-cycle.ixql` Phase 1 to call `ix-blast-radius` and Phase 3 to call the actor (~0.5 day, ixql-side)

Total ~2.5 days of focused work, all in GA + Demerzel; IX is unblocked by the Phase 1 producer that already shipped.

## What NOT to build (deliberate)

- Auto-merge for actor-proposed PRs — human stays in the loop on draft → mergeable until 30 days of evidence
- Drift on individual prompts (just on aggregates) — per-prompt drift is noisy without a much larger snapshot history
- Rich-graph reasoning over followups — the classifier is rule-based until telemetry shows the rules are insufficient
- A new MCP server for the actor — it's a worker service, not an agent endpoint
