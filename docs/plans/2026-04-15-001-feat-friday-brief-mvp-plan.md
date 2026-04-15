# Friday Brief — MVP Plan

**Status:** Draft
**Date:** 2026-04-15
**Owner:** spareilleux
**Depends on:** R2 Phase 2 (4ebe402), R3 (8c8b0d9), claude-mem ≥0.x, Octopus plugin, Workspace Enterprise/Edu Google identity

## Goal

Ship a weekly auto-generated NotebookLM podcast ("Friday Brief") that turns a team's claude-mem session episodes + IX algorithmic verdicts + Octopus multi-LLM dissent into an 8-minute audio overview consumable by PMs who never open a terminal.

The composite proves itself when run #2 of the brief recalls the prior week's Betti-number / Lyapunov / governance verdicts as part of the new podcast — a behavior no single tool produces alone.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  weekly cron (Friday 16:00 local)                               │
│      │                                                          │
│      ▼                                                          │
│  /friday-brief  ── Octopus slash command                        │
│      │                                                          │
│      ▼                                                          │
│  ix_pipeline DAG                                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. claude-mem export (last 7 days of session episodes)   │   │
│  │    via :37777 HTTP API                                   │   │
│  │ 2. source_sanitizer  ── G2: strip injection patterns     │   │
│  │ 3. ix_code_analyze   ── cyclomatic + call graph deltas   │   │
│  │ 4. ix_topo           ── persistent homology, Betti delta │   │
│  │ 5. ix_chaos_lyapunov ── velocity/PR-throughput drift     │   │
│  │ 6. ix_governance_check ── T/P/U/D/F/C verdict per item   │   │
│  │ 7. octopus orchestrate ── 3-LLM dissent on each verdict  │   │
│  │ 8. brief_compiler    ── render Markdown source bundle    │   │
│  │ 9. tier_gate         ── G1: refuse on non-Workspace acct │   │
│  │ 10. anything-to-notebooklm ── upload sources             │   │
│  │ 11. notebooklm-mcp ── trigger audio overview generation  │   │
│  │ 12. audio_blob_scrape ── pull podcast .m4a               │   │
│  │ 13. claude-mem hook ── persist run trace                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│      │                                                          │
│      ▼                                                          │
│  Slack/email delivery: podcast + 1-page citation-linked brief   │
└─────────────────────────────────────────────────────────────────┘
```

## Wiring

### `.mcp.json` (additions)

```jsonc
{
  "mcpServers": {
    "ix": {
      "command": "ix-mcp",
      "args": [],
      "env": { "IX_PIPELINE_TRACE": "1" }
    },
    "notebooklm-read": {
      "command": "node",
      "args": ["./vendor/notebooklm-mcp/dist/server.js", "--profile", "minimal"],
      "env": {
        "NLM_USER_DATA_DIR": "${SANDBOX_DIR}/chrome-nlm",
        "NLM_PINNED_COMMIT": "REPLACE_WITH_SHA"
      }
    },
    "notebooklm-write": {
      "command": "python",
      "args": ["-m", "anything_to_notebooklm.server"],
      "env": {
        "ATN_USER_DATA_DIR": "${SANDBOX_DIR}/chrome-nlm",
        "ATN_PINNED_COMMIT": "REPLACE_WITH_SHA",
        "ATN_TIER_GATE": "workspace-only"
      }
    }
  }
}
```

Both NotebookLM MCPs share **one** sandboxed Chrome profile under `${SANDBOX_DIR}` (G3) and are pinned to specific commit SHAs (not tags).

### Octopus skill — `.claude/skills/friday-brief/SKILL.md`

```markdown
---
name: friday-brief
version: 0.1.0
description: Weekly team brief — ix pipeline + claude-mem export + NotebookLM audio overview
affordances:
  - mcp:ix:ix_pipeline
  - mcp:ix:ix_code_analyze
  - mcp:ix:ix_topo
  - mcp:ix:ix_chaos_lyapunov
  - mcp:ix:ix_governance_check
  - mcp:notebooklm-read:list_notebooks
  - mcp:notebooklm-read:select_notebook
  - mcp:notebooklm-write:add_source
  - mcp:notebooklm-write:trigger_audio_overview
  - http:localhost:37777/export
goal_directedness: task-scoped
estimator_pairing: skeptical-auditor
---

## Trigger
- Slash command: `/friday-brief [--week last|current]`
- Cron: Fridays 16:00 local

## Pre-flight (must pass before any upload)
1. Confirm Chrome profile is on a Google Workspace tier (Business/Enterprise/Edu)
2. Confirm `IX_PIPELINE_TRACE=1` so claude-mem can record DAG nodes
3. Confirm pinned commit SHAs match `.mcp.json`

## Pipeline
Run `ix_pipeline` with the DAG defined in `.claude/skills/friday-brief/dag.json`.
Each node emits `notifications/progress` so claude-mem ToolUse hooks see the full chain.

## Output
- Markdown brief at `state/briefs/{date}-friday-brief.md` with citations
- Audio at `state/briefs/{date}-friday-brief.m4a`
- Belief snapshot at `state/snapshots/{date}-friday-brief.snapshot.json`
```

### `dag.json` (sketch)

```json
{
  "nodes": [
    {"id": "export", "tool": "http_get", "args": {"url": "http://localhost:37777/export?days=7"}},
    {"id": "sanitize", "tool": "source_sanitizer", "deps": ["export"]},
    {"id": "complexity", "tool": "ix_code_analyze", "args": {"op": "complexity"}, "deps": ["sanitize"]},
    {"id": "topology", "tool": "ix_topo", "deps": ["sanitize"]},
    {"id": "chaos", "tool": "ix_chaos_lyapunov", "deps": ["sanitize"]},
    {"id": "verdict", "tool": "ix_governance_check", "deps": ["complexity", "topology", "chaos"]},
    {"id": "dissent", "tool": "octopus_orchestrate", "args": {"phase": "tangle"}, "deps": ["verdict"]},
    {"id": "compile", "tool": "brief_compiler", "deps": ["verdict", "dissent"]},
    {"id": "tier_check", "tool": "tier_gate", "deps": ["compile"]},
    {"id": "upload", "tool": "add_source", "deps": ["tier_check"]},
    {"id": "audio", "tool": "trigger_audio_overview", "deps": ["upload"]},
    {"id": "scrape", "tool": "audio_blob_scrape", "deps": ["audio"]}
  ]
}
```

## Guardrails (from security review)

### G1 — Tier gate + sacrificial identity
- Pre-flight probes account-type endpoint; refuses non-Workspace.
- Onboarding doc instructs creating dedicated `friday-brief-bot@customer.com` Workspace identity, **no Drive/Gmail/admin scopes**.
- Lives in: `tier_gate` DAG node.

### G2 — Source sanitizer
- Strips imperative second-person text matching injection regex set.
- Wraps remaining content in `<source author="claude-mem" trust="observed">` envelope.
- Runs `ix_governance_check` with Confidential classifier; **refuses upload on Confidential or Unknown verdicts**.
- Lives in: `sanitize` + `verdict` DAG nodes.

### G3 — Pinned, sandboxed MCP execution
- Both NotebookLM MCPs pinned to specific commit SHAs in `.mcp.json`.
- Run inside a Windows sandbox / rootless container with:
  - No access to developer's real Chrome profile
  - Dedicated `user-data-dir` on an encrypted volume
  - Egress firewall allowlist: `*.google.com` only
- Lives in: MCP launcher wrapper in `ix-agent` + host-level sandbox config.

## Known holes (documented, not fixed in MVP)

1. **Audio scrape is fragile.** Audio-overview blob URL has minute-scale lifetime and DOM selectors will rev. Expect 2-6 week MTBF; budget for monthly maintenance.
2. **NotebookLM round-trip is non-deterministic.** Demerzel verdicts on the audio output are inferential, not empirical — tagged accordingly per scientific-objectivity policy.
3. **Sampling/createMessage inside `ix_pipeline` LLM-in-the-loop steps** doesn't traverse claude-mem hooks. IX must POST directly to `:37777` to record those turns.
4. **`ix_pipeline` recursion visibility** — depends on R4 progress notifications. Not a hard blocker for MVP if `IX_PIPELINE_TRACE=1` flushes per-node summaries.

## MVP scope (2 weeks, 1 engineer)

**Ship:**
- Single hardcoded DAG (above) wired through `ix_pipeline`
- Tier gate + source sanitizer + sandboxed Chrome profile (the three guardrails)
- One real run on this repo's last-7-days history
- Manual audio-button click acceptable for demo (audio scrape is week-3 work)

**Cut:**
- Slack/email delivery (manual file handoff for demo)
- `audio_blob_scrape` (manual download)
- Multi-tenant onboarding flow
- Federation to TARS/GA enrichment

## Kill criteria

- Pre-MVP: if Workspace tier gate can't be implemented reliably from the Chrome profile, abandon NotebookLM path and pivot to Gemini 2.5 + Kokoro-82M TTS (already in Demerzel multi-model orchestration policy v1.1.0).
- Post-MVP: if no design partner listens to brief #1 within 48 hours of delivery, the audio format is wrong — fall back to written brief and reassess.

## Open questions

1. Where does the source sanitizer live — new `crates/ix-sanitize` crate or extension to `ix-governance`?
2. Does `IX_PIPELINE_TRACE=1` already exist or does the R4 progress-notification work need to ship first?
3. Is there a Workspace-tier check endpoint that doesn't require admin SDK access?
4. Belief snapshot schema for non-deterministic third-party round trips — does the existing `state/snapshots/*.snapshot.json` schema cover inferred-verdict provenance?

## References

- Security review: octopus security-auditor persona, 2026-04-15 (in-conversation)
- Strategy review: octopus strategy-analyst + ai-engineer + ux-researcher + business-analyst, 2026-04-14/15 (in-conversation)
- Related plan: `docs/plans/2026-04-14-001-feat-r4-meta-mcp-gateway-plan.md` (R4 progress notifications would unblock claude-mem's view into `ix_pipeline` recursion)
- Octopus integration memory: `~/.claude/projects/.../memory/project_octopus_integration.md`

## Phase 1 shipped

Landed on branch `worktree-agent-a8ac5719` (see final commit for the SHA
at merge time):

- **`crates/ix-sanitize`** — new crate. Baseline injection regex set,
  `Sanitizer::sanitize` with stripped-count tracking, CDATA-safe
  `wrap_envelope`, hexavalent `verdict_gate` (T/P -> Allow, F/D ->
  RefuseConfidential, U/C/other -> RefuseUnknown). 6 unit tests.
- **`crates/ix-friday-brief`** — new crate (library + binary). Builds
  the 12-node pipeline via `ix_pipeline::builder::PipelineBuilder` (not
  the JSON DAG from the plan sketch — the real API is Rust code).
  Hard-coded 12-episode fixture anchored at 2026-04-06. Writes
  `state/briefs/{date}-friday-brief.md` and
  `state/snapshots/{date}-friday-brief.snapshot.json` (tagged
  `trust: "inferred"`). Smoke test redirects state via
  `IX_FRIDAY_BRIEF_STATE_DIR` and asserts both artifacts + the full
  12-node trace in declared order.
- **`.claude/skills/friday-brief/SKILL.md`** — natural-language skill
  doc: when to use, pre-flight (all stubbed in MVP), invocation,
  output locations, known limitations, guardrail notes.
- **`.mcp.friday-brief.example.json`** — skeleton entries for the two
  NotebookLM MCPs with `REPLACE_WITH_SHA` placeholders and an
  explanatory `_comment`. Intentionally NOT merged into `.mcp.json`
  (would break MCP startup).
- **`docs/runbooks/friday-brief-vendoring.md`** — draft runbook for
  creating the sacrificial Workspace identity, cloning + pinning the
  two NotebookLM MCPs, sandbox profile directory setup, egress
  firewall allowlist, and five-step validation. Marked `DRAFT — phase
  2, not production.`

Remaining phase-2 TODOs (not in this branch):

- Real NotebookLM MCPs: clone, pin SHAs, merge example JSON into
  `.mcp.json` after vendoring runbook has actually been executed.
- `tier_gate` probe: replace the stub with a real Chrome-profile-based
  Workspace tier detection.
- `upload` / `audio` / `scrape` nodes: wire to the pinned MCP tools
  (`add_source`, `trigger_audio_overview`, blob scraper).
- Replace `complexity`/`topology`/`chaos` stubs with real calls to
  `ix_code_analyze`, `ix_topo`, `ix_chaos_lyapunov` from inside the
  pipeline compute closures.
- Replace fixed `T` verdict with a real `ix_governance_check` call that
  consumes the combined structural outputs.
- Replace fake 3-LLM dissent with a real Octopus orchestrator call.
- Slack/email delivery of the compiled brief + audio blob.
- Federation hook to TARS/GA enrichment (grammar patterns, music
  theory, etc.).
- `IX_PIPELINE_TRACE=1` progress-notification wiring (depends on R4).
- MCP tool form of the binary (`mcp__ix__ix_friday_brief` or similar).
