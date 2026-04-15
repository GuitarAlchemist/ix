---
name: friday-brief
description: Weekly team brief — ix-friday-brief binary + NotebookLM audio overview (phase 2)
---

# Friday Brief

Generates the weekly Friday Brief: a Markdown summary of the last 7 days
of session activity, topological/chaos/governance verdicts, and
multi-LLM dissent — delivered today as a local file, eventually as a
NotebookLM audio overview.

## When to Use

- Weekly, Friday 16:00 local, to produce a PM-readable digest of what
  the team shipped that week.
- Ad-hoc, when a human asks "what happened in ix this week?" and you
  want the short, structured answer rather than scrolling git log.
- When demonstrating the end-to-end pipeline that chains sanitation,
  structural analysis, governance verdict, and multi-AI dissent.

## Pre-flight

All of the following are **stubbed in the MVP** and will be enforced in
phase 2 — document them in your run notes, don't pretend they ran:

1. **Tier gate** — in phase 2, the pipeline will probe the Chrome
   profile for Google Workspace tier (Business/Enterprise/Edu) and
   refuse to proceed on personal accounts. Currently logs a warning
   and continues.
2. **`IX_PIPELINE_TRACE=1`** — in phase 2, this will cause each pipeline
   node to emit `notifications/progress` so claude-mem can record the
   DAG walk. Currently ignored; trace is still pinned via
   `PIPELINE_NODE_ORDER` inside the library.
3. **Pinned MCP SHAs** — in phase 2, the NotebookLM MCPs in
   `.mcp.json` will be pinned to specific commits. Currently the
   entries live in `.mcp.friday-brief.example.json` and are not
   wired into `.mcp.json` at all.

## Invocation

Direct binary:

```
cargo run -p ix-friday-brief -- run
```

Output is two paths, the Markdown brief and the belief snapshot.

MCP tool form (`mcp__ix__ix_friday_brief` or similar) is **phase 2** —
the MVP is intentionally shipped as a standalone binary so the plumbing
can be exercised without MCP round-trips.

## Output Locations

- `state/briefs/{YYYY-MM-DD}-friday-brief.md` — the Markdown brief
- `state/snapshots/{YYYY-MM-DD}-friday-brief.snapshot.json` — the
  belief snapshot, tagged `trust: "inferred"` per the
  scientific-objectivity policy (because the source data is
  synthetic-fixture in the MVP).

The state root honors `IX_FRIDAY_BRIEF_STATE_DIR` for tests and for
running against a throwaway directory.

## Known MVP Limitations

The following four nodes are **stubs** in phase 1. They log a warning
and return synthetic JSON:

1. `tier_gate` — does not actually probe the Google account tier.
2. `upload` — does not call NotebookLM `add_source`.
3. `audio` — does not trigger NotebookLM audio overview generation.
4. `scrape` — does not download the audio blob.

Additionally, `complexity`, `topology`, and `chaos` currently emit
synthetic invariant numbers instead of calling `ix_code_analyze`,
`ix_topo`, and `ix_chaos_lyapunov`. Wiring those in is a trivial
follow-up once the MCP tools are callable from inside the pipeline
compute closure (see `docs/plans/2026-04-15-001-feat-friday-brief-mvp-plan.md`
phase 2 TODO list).

## Guardrails

The one guardrail that **is** active in MVP is the source sanitizer:
every episode's `raw` text is run through `ix_sanitize::Sanitizer`,
which strips a curated set of injection regex patterns (imperative
override, instruction override, credential leak, tool-use tag
injection) before the content reaches any downstream node. Episodes
that would normally flow into a compile step thus cannot carry prompt
injection payloads into the brief body.

The verdict gate (`ix_sanitize::verdict_gate`) is also wired in: the
`verdict` node runs its emitted letter through the hexavalent gate and
includes the result in the compile input. Today the letter is
hard-coded to `T`, so the gate always returns `allow`; phase 2 will
replace that with a real `ix_governance_check` call.
