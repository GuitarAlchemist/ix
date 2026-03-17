---
name: federation-traces
description: Ingest GA trace artifacts, analyze with ix stats, feed results to TARS pattern promotion
---

# Federation Trace Pipeline

Cross-repo trace pipeline: GA exports, ix analyzes, TARS promotes patterns.

## When to Use
When trace data from GA needs statistical analysis before being fed into TARS's pattern promotion pipeline.

## Pipeline

### Step 1: Prepare traces from GA
**Tool**: `ix_tars_bridge` with action=prepare_traces
- Loads traces from `~/.ga/traces/`
- Computes stats (latency, event distribution, anomalies)
- Returns payload formatted for TARS ingestion

### Step 2: Analyze traces
**Tool**: `ix_trace_ingest`
- Loads trace directory, computes detailed stats
- Mean/median/p95 latency per event type
- Success/failure rates
- Event frequency distribution

### Step 3: Ingest into TARS
**Tool**: TARS `ingest_ga_traces`
- Input: `{"Count": N, "MinOccurrences": 3}`
- Runs pattern discovery on trace artifacts
- Promotes recurring patterns up the staircase

### Step 4: View promotion results
**Tool**: TARS `promotion_index`
- Returns ranked patterns sorted by level, score, weight

### Step 5: Export insights
**Tool**: TARS `export_insights`
- Writes `~/.tars/insights/latest.json` with pattern scores, gaps, recommendations

## Trace Format (GA)
```json
{
  "trace_id": "...",
  "timestamp": "...",
  "events": [{ "type": "...", "duration_ms": 0, "metadata": {} }],
  "outcome": "success|failure"
}
```

## Analysis Output (ix)
- Mean/median/p95 latency per event type
- Event frequency distribution
- Success/failure correlation analysis
