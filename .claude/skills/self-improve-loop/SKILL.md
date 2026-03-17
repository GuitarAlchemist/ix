---
name: self-improve-loop
description: Full feedback cycle across GA, ix, and TARS — trace export, analysis, pattern promotion, grammar evolution
---

# Self-Improving Feedback Loop

Closed-loop skill that drives continuous improvement across the GA/ix/TARS federation. Execution traces flow from GA through ix analysis into TARS pattern promotion, then back into ix grammar weight evolution.

## When to Use

- When you want to **close the feedback loop** between GA, ix, and TARS after a batch of work
- When accumulated traces in `~/.ga/traces/` need to be analyzed and fed forward
- When grammar rule weights feel stale and should be updated from real execution data
- After a session where multiple federation tools were used and you want the system to learn from it

## Prerequisites

- GA MCP server running (provides `ExportTraces`, `AskChatbot`, etc.)
- ix MCP server running (provides `ix_trace_ingest`, `ix_tars_bridge`, `ix_grammar_evolve`)
- TARS MCP server running (provides `ingest_ga_traces`, `run_promotion_pipeline`)
- Trace directory exists: `~/.ga/traces/`

## The Loop (5 Steps)

```
+---------------------------------------------+
|                                               |
|   1. GA     -->  2. ix      -->  3. TARS      |
|   Export         Analyze         Promote       |
|                                   |            |
|        <---  4. ix Evolve  <------+            |
|                                               |
+---------------------------------------------+
```

### Step 1: GA Exports Execution Traces

**GA Tool**: `ExportTraces`
**Input**: `{ "Count": 50 }` (or `"SinceIso": "2026-03-14T00:00:00Z"`)
**Check**: Confirm at least 1 trace file in `~/.ga/traces/`. If 0, stop.

### Step 2: ix Analyzes Traces

**ix Tool**: `ix_tars_bridge` action=prepare_traces
This loads traces, computes stats, and formats a payload for TARS.

**Follow-up**: `ix_trace_ingest` for detailed stats (latency, anomalies, event distribution).

**Check**: If `success_rate < 0.5`, trigger recovery. If anomaly count > 20%, investigate.

### Step 3: TARS Promotes Patterns

**TARS Tool**: `ingest_ga_traces`
**Input**: `{ "Count": 100, "MinOccurrences": 3 }`
**Output**: Discovered patterns promoted through 7-step pipeline.

**Follow-up**: `promotion_index` to view ranked results.

**Check**: Verify promoted patterns have confidence >= 0.7.

### Step 4: ix Updates Grammar Weights

**ix Tool**: `ix_grammar_evolve`
**Input**: Promoted pattern weights from Step 3 output.

**Pre-flight**: Run `ix_governance_check` (Articles 3, 4, 7) before modifying weights.

**Check**: Confirm `rules_updated <= 3` (max modifications per session).

### Step 5: Loop Back

Return to Step 1 in the next session. Updated grammar weights influence which tool sequences GA favors.

## Governance Checks

### Self-modification policy
- **Max 3 grammar modifications per session**
- **Max weight delta of 0.2 per rule per session**
- **No rule deletion** — only weight adjustments

### Pre-flight governance check (before Step 4)
**Tool**: `ix_governance_check`
```json
{
  "action": "Update grammar weights based on trace analysis",
  "context": "Self-improving loop iteration",
  "articles": [3, 4, 7]
}
```

### Relevant constitutional articles
- **Article 3 (Reversibility)**: Weight changes must be reversible
- **Article 4 (Proportionality)**: Changes proportional to evidence
- **Article 7 (Auditability)**: All changes logged with before/after values

### Abort conditions
- Governance check returns non-compliant
- More than 3 modifications required
- Success rate drops below 0.3
- Any anomaly classified as `security` type

## Failure Handling

1. **Step 1 fails** (no traces): Check GA server. Retry once. Skip if still failing.
2. **Step 2 fails** (analysis error): Use `ix_stats` on raw values as fallback.
3. **Step 3 fails** (TARS unavailable): Cache at `~/.ix/pending_promotions.json`. Retry next iteration.
4. **Step 4 fails** (grammar error): Do NOT retry. Preserve current version. Escalate (Article 6).

For 3+ consecutive failures, escalate to user with summary.
