---
name: federation-traces
description: Ingest GA trace artifacts, analyze with ix stats, feed results to TARS pattern promotion
---

# Federation Trace Pipeline

Cross-repo trace pipeline: GA exports → ix analyzes → TARS promotes patterns.

## When to Use
When trace data from GA needs statistical analysis before being fed into TARS's pattern promotion pipeline.

## Pipeline
1. **Export**: GA `ga_export_traces` writes trace JSON to `~/.ga/traces/`
2. **Analyze**: ix `ix_trace_ingest` loads traces, computes stats (latency, event distribution, anomaly detection)
3. **Promote**: TARS `tars_trace_ingest` receives analyzed traces for pattern promotion

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
- Anomaly detection via DBSCAN clustering
- Success/failure correlation analysis
