---
name: self-improve-loop
description: Full feedback cycle across GA, ix, and TARS — trace export, analysis, pattern promotion, grammar evolution
---

# Self-Improving Feedback Loop

Closed-loop skill that drives continuous improvement across the GA/ix/TARS federation. Execution traces flow from GA through ix analysis into TARS pattern promotion, then back into ix grammar weight evolution — completing the cycle.

## When to Use

- When you want to **close the feedback loop** between GA, ix, and TARS after a batch of work
- When accumulated traces in `~/.ga/traces/` need to be analyzed and fed forward
- When grammar rule weights feel stale and should be updated from real execution data
- When you suspect certain tool-call patterns are suboptimal and want data-driven refinement
- After a session where multiple federation tools were used and you want the system to learn from it

## Prerequisites

- GA MCP server running (provides `ga_export_traces`)
- ix MCP server running (provides `ix_trace_ingest`, `ix_stats`, `ix_grammar_evolve`)
- TARS MCP server running (provides `tars_trace_ingest`)
- Trace directory exists: `~/.ga/traces/`

## The Loop (5 Steps)

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │  1. GA    │───▶│  2. ix   │───▶│  3. TARS │         │
│   │  Export   │    │  Analyze │    │  Promote │         │
│   └──────────┘    └──────────┘    └──────────┘         │
│        ▲                               │                │
│        │          ┌──────────┐         │                │
│        │          │  4. ix   │◀────────┘                │
│        └──────────│  Evolve  │                          │
│                   └──────────┘                          │
└─────────────────────────────────────────────────────────┘
```

### Step 1: GA Exports Execution Traces

**Tool**: `ga_export_traces`
**Input**:
```json
{
  "output_dir": "~/.ga/traces/",
  "format": "json",
  "since": "last_ingest"
}
```
**Output**: Trace files written to `~/.ga/traces/*.json`, each containing:
```json
{
  "trace_id": "tr_20260314_001",
  "timestamp": "2026-03-14T10:30:00Z",
  "events": [
    { "type": "tool_call", "tool": "ga_chord", "duration_ms": 12, "metadata": { "chord": "Cmaj7" } },
    { "type": "tool_call", "tool": "ix_kmeans", "duration_ms": 45, "metadata": { "k": 3 } }
  ],
  "outcome": "success",
  "session_id": "sess_abc123"
}
```
**What to check**: Confirm at least 1 trace file was written. If 0 traces, the loop has nothing to learn from — stop here.

### Step 2: ix Analyzes Traces

**Tool**: `ix_trace_ingest`
**Input**:
```json
{
  "trace_dir": "~/.ga/traces/",
  "analysis": ["latency_stats", "anomaly_detection", "pattern_extraction"],
  "anomaly_method": "dbscan",
  "min_samples": 3
}
```
**Output**:
```json
{
  "summary": {
    "total_traces": 47,
    "success_rate": 0.89,
    "mean_latency_ms": 34.2,
    "p95_latency_ms": 120.0
  },
  "anomalies": [
    { "trace_id": "tr_20260314_012", "reason": "latency_spike", "value_ms": 890 }
  ],
  "patterns": [
    { "sequence": ["ga_chord", "ix_fft", "ix_kmeans"], "frequency": 12, "avg_duration_ms": 95 },
    { "sequence": ["ga_scale", "ix_supervised"], "frequency": 8, "avg_duration_ms": 60 }
  ]
}
```

**Follow-up — get detailed stats on patterns**:
**Tool**: `ix_stats`
**Input**:
```json
{
  "data": [95, 60, 120, 45, 88, 73, 55, 110, 42, 67],
  "compute": ["mean", "median", "std", "skewness", "kurtosis"]
}
```
**What to check**: If `success_rate < 0.5`, trigger recovery (see Failure Handling below). If anomaly count > 20% of traces, investigate before proceeding.

### Step 3: TARS Promotes Patterns

**Tool**: `tars_trace_ingest`
**Input**:
```json
{
  "patterns": [
    { "sequence": ["ga_chord", "ix_fft", "ix_kmeans"], "frequency": 12, "avg_duration_ms": 95 },
    { "sequence": ["ga_scale", "ix_supervised"], "frequency": 8, "avg_duration_ms": 60 }
  ],
  "anomalies": [
    { "trace_id": "tr_20260314_012", "reason": "latency_spike", "value_ms": 890 }
  ],
  "promotion_threshold": 5
}
```
**Output**:
```json
{
  "promoted": [
    {
      "pattern": ["ga_chord", "ix_fft", "ix_kmeans"],
      "weight": 0.82,
      "confidence": 0.91,
      "label": "chord_spectral_cluster"
    }
  ],
  "demoted": [],
  "unchanged": [
    { "pattern": ["ga_scale", "ix_supervised"], "weight": 0.55, "reason": "below_confidence_threshold" }
  ]
}
```
**What to check**: Verify that promoted patterns have `confidence >= 0.7`. Patterns below this threshold should remain unchanged, not promoted.

### Step 4: ix Updates Grammar Weights

**Tool**: `ix_grammar_evolve`
**Input**:
```json
{
  "promoted_patterns": [
    {
      "pattern": ["ga_chord", "ix_fft", "ix_kmeans"],
      "weight": 0.82,
      "label": "chord_spectral_cluster"
    }
  ],
  "evolution_strategy": "weight_adjustment",
  "learning_rate": 0.1,
  "max_weight_delta": 0.2
}
```
**Output**:
```json
{
  "rules_updated": 3,
  "weight_changes": [
    { "rule": "chord_analysis → spectral_extract cluster", "old_weight": 0.5, "new_weight": 0.62 },
    { "rule": "scale_analysis → classify", "old_weight": 0.5, "new_weight": 0.45 }
  ],
  "grammar_version": "v2.3.1"
}
```
**What to check**: Confirm `rules_updated <= max_modifications` (see governance below). Verify no weight went below 0.05 or above 0.95 (degenerate grammar risk).

### Step 5: Loop Back

Return to Step 1 in the next session. The updated grammar weights will influence which tool sequences GA favors, generating new traces that reflect the learned patterns.

## Governance Checks

**Self-modification policy**: This loop modifies grammar weights, which influence future behavior. The following guardrails are mandatory.

### Modification Limits

- **Max 3 grammar modifications per session**. If `ix_grammar_evolve` returns `rules_updated > 3`, reject the update and re-run with a higher `max_weight_delta` constraint or fewer promoted patterns.
- **Max weight delta of 0.2 per rule per session**. No single rule weight should change by more than 0.2 in one iteration.
- **No rule deletion**. The loop adjusts weights only; it never removes grammar rules.

### Pre-flight Governance Check

Before Step 4, run a governance check:

**Tool**: `ix_governance_check`
**Input**:
```json
{
  "action": "Update grammar weights based on trace analysis",
  "context": "Self-improving loop iteration: updating weights for promoted patterns from TARS. Rules affected: N. Max delta: 0.2.",
  "articles": [3, 4, 7]
}
```

Relevant constitutional articles:
- **Article 3 (Reversibility)**: Grammar weight changes must be reversible. The previous `grammar_version` must be retained.
- **Article 4 (Proportionality)**: Weight changes should be proportional to the evidence (trace count, confidence score).
- **Article 7 (Auditability)**: All weight changes must be logged with before/after values and the traces that motivated them.

### Abort Conditions

Stop the loop immediately if:
- Governance check returns non-compliant
- More than 3 modifications would be required
- Success rate in Step 2 drops below 0.3 (system is degrading)
- Any anomaly is classified as `security` type

## Failure Handling

If any step fails, invoke the **recovery-agent** persona:

1. **Step 1 fails** (no traces exported): Check GA server connectivity. Retry once. If still failing, log and skip this iteration.
2. **Step 2 fails** (analysis error): Likely malformed trace data. Use `ix_stats` on raw latency values as a fallback for basic metrics. Skip anomaly detection.
3. **Step 3 fails** (TARS unavailable): Cache the analysis output locally at `~/.ix/pending_promotions.json`. Retry TARS ingestion next iteration.
4. **Step 4 fails** (grammar update error): Do NOT retry automatically. Log the failure, preserve current grammar version, and escalate (Article 6).

For persistent failures across 3+ iterations, escalate to the user with a summary of what has been failing and the accumulated pending data.

## Session Log Format

After each loop iteration, produce a summary:

```
=== Self-Improve Loop — 2026-03-14T10:45:00Z ===
Traces ingested: 47
Success rate: 89%
Patterns found: 5 (2 promoted, 0 demoted, 3 unchanged)
Grammar updates: 2 of 3 max
  - chord_analysis → spectral_extract cluster: 0.50 → 0.62
  - scale_analysis → classify: 0.50 → 0.45
Grammar version: v2.3.0 → v2.3.1
Governance: COMPLIANT (Articles 3, 4, 7)
Next iteration: ready
================================================
```
