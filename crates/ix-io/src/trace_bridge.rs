//! Trace bridge — reads GA (Guitar Alchemist) trace files for ix analysis.
//!
//! Traces are JSON files exported by GA tools, each containing a list of events
//! with timing and metadata. This module loads them and computes statistics
//! suitable for feeding into ix ML pipelines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::IoError;

// ── Types ────────────────────────────────────────────────────────

/// A single GA trace, representing one execution run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    pub trace_id: String,
    pub timestamp: String,
    pub events: Vec<TraceEvent>,
    pub outcome: String, // "success" or "failure"
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// A single event within a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    pub event_type: String,
    pub duration_ms: f64,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Aggregate statistics computed over a collection of traces.
#[derive(Debug, Clone)]
pub struct TraceStats {
    pub total_traces: usize,
    pub success_count: usize,
    pub failure_count: usize,
    pub avg_duration_ms: f64,
    pub event_type_counts: HashMap<String, usize>,
    pub p50_duration_ms: f64,
    pub p95_duration_ms: f64,
}

// ── Loading ──────────────────────────────────────────────────────

/// Load a single trace from a JSON file.
pub fn load_trace(path: &Path) -> Result<Trace, IoError> {
    let content = std::fs::read_to_string(path)?;
    let trace: Trace = serde_json::from_str(&content)?;
    Ok(trace)
}

/// Load all traces from a directory (reads every `*.json` file).
pub fn load_traces(dir: &Path) -> Result<Vec<Trace>, IoError> {
    let mut traces = Vec::new();
    let entries = std::fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            match load_trace(&path) {
                Ok(trace) => traces.push(trace),
                Err(_) => {
                    // Skip files that aren't valid traces
                    continue;
                }
            }
        }
    }

    Ok(traces)
}

/// Default trace directory: `~/.ga/traces/`
pub fn default_trace_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".ga").join("traces")
}

// ── Statistics ───────────────────────────────────────────────────

/// Total duration of a trace (sum of all event durations).
fn trace_duration(trace: &Trace) -> f64 {
    trace.events.iter().map(|e| e.duration_ms).sum()
}

/// Compute aggregate statistics over a set of traces.
pub fn compute_stats(traces: &[Trace]) -> TraceStats {
    let total_traces = traces.len();
    let success_count = traces.iter().filter(|t| t.outcome == "success").count();
    let failure_count = total_traces - success_count;

    let mut event_type_counts: HashMap<String, usize> = HashMap::new();
    for trace in traces {
        for event in &trace.events {
            *event_type_counts
                .entry(event.event_type.clone())
                .or_default() += 1;
        }
    }

    let mut durations: Vec<f64> = traces.iter().map(trace_duration).collect();
    durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let avg_duration_ms = if durations.is_empty() {
        0.0
    } else {
        durations.iter().sum::<f64>() / durations.len() as f64
    };

    let p50_duration_ms = percentile(&durations, 50.0);
    let p95_duration_ms = percentile(&durations, 95.0);

    TraceStats {
        total_traces,
        success_count,
        failure_count,
        avg_duration_ms,
        event_type_counts,
        p50_duration_ms,
        p95_duration_ms,
    }
}

/// Compute a percentile from a sorted slice. Returns 0.0 for empty input.
fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct / 100.0 * (sorted.len() as f64 - 1.0)).round() as usize;
    let idx = idx.min(sorted.len() - 1);
    sorted[idx]
}

// ── CSV export ───────────────────────────────────────────────────

/// Export traces as CSV rows (one row per event).
///
/// Returns a header row followed by data rows with columns:
/// `trace_id, timestamp, outcome, event_type, duration_ms`.
pub fn traces_to_csv_rows(traces: &[Trace]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    rows.push(vec![
        "trace_id".into(),
        "timestamp".into(),
        "outcome".into(),
        "event_type".into(),
        "duration_ms".into(),
    ]);

    for trace in traces {
        for event in &trace.events {
            rows.push(vec![
                trace.trace_id.clone(),
                trace.timestamp.clone(),
                trace.outcome.clone(),
                event.event_type.clone(),
                event.duration_ms.to_string(),
            ]);
        }
    }

    rows
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_traces() -> Vec<Trace> {
        vec![
            Trace {
                trace_id: "t1".into(),
                timestamp: "2026-01-01T00:00:00Z".into(),
                events: vec![
                    TraceEvent {
                        event_type: "parse".into(),
                        duration_ms: 10.0,
                        metadata: serde_json::Value::Null,
                    },
                    TraceEvent {
                        event_type: "eval".into(),
                        duration_ms: 20.0,
                        metadata: serde_json::Value::Null,
                    },
                ],
                outcome: "success".into(),
                metadata: serde_json::Value::Null,
            },
            Trace {
                trace_id: "t2".into(),
                timestamp: "2026-01-01T00:01:00Z".into(),
                events: vec![TraceEvent {
                    event_type: "parse".into(),
                    duration_ms: 50.0,
                    metadata: serde_json::Value::Null,
                }],
                outcome: "failure".into(),
                metadata: serde_json::Value::Null,
            },
            Trace {
                trace_id: "t3".into(),
                timestamp: "2026-01-01T00:02:00Z".into(),
                events: vec![
                    TraceEvent {
                        event_type: "parse".into(),
                        duration_ms: 5.0,
                        metadata: serde_json::Value::Null,
                    },
                    TraceEvent {
                        event_type: "eval".into(),
                        duration_ms: 15.0,
                        metadata: serde_json::Value::Null,
                    },
                    TraceEvent {
                        event_type: "render".into(),
                        duration_ms: 30.0,
                        metadata: serde_json::Value::Null,
                    },
                ],
                outcome: "success".into(),
                metadata: serde_json::Value::Null,
            },
        ]
    }

    #[test]
    fn test_load_trace_from_json_string() {
        let json = r#"{
            "trace_id": "abc-123",
            "timestamp": "2026-03-14T10:00:00Z",
            "events": [
                {"event_type": "init", "duration_ms": 5.0},
                {"event_type": "run",  "duration_ms": 42.0}
            ],
            "outcome": "success"
        }"#;
        let trace: Trace = serde_json::from_str(json).unwrap();
        assert_eq!(trace.trace_id, "abc-123");
        assert_eq!(trace.events.len(), 2);
        assert_eq!(trace.outcome, "success");
        assert_eq!(trace.events[1].duration_ms, 42.0);
        // metadata should default to null
        assert!(trace.metadata.is_null());
    }

    #[test]
    fn test_compute_stats() {
        let traces = sample_traces();
        let stats = compute_stats(&traces);

        assert_eq!(stats.total_traces, 3);
        assert_eq!(stats.success_count, 2);
        assert_eq!(stats.failure_count, 1);

        // durations: t1=30, t2=50, t3=50 => avg = 130/3 ≈ 43.33
        let expected_avg = (30.0 + 50.0 + 50.0) / 3.0;
        assert!((stats.avg_duration_ms - expected_avg).abs() < 0.01);

        // event_type_counts: parse=3, eval=2, render=1
        assert_eq!(stats.event_type_counts["parse"], 3);
        assert_eq!(stats.event_type_counts["eval"], 2);
        assert_eq!(stats.event_type_counts["render"], 1);

        // sorted durations: [30, 50, 50]
        // p50 = 50, p95 = 50
        assert!((stats.p50_duration_ms - 50.0).abs() < 0.01);
        assert!((stats.p95_duration_ms - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert_eq!(stats.total_traces, 0);
        assert_eq!(stats.success_count, 0);
        assert_eq!(stats.failure_count, 0);
        assert_eq!(stats.avg_duration_ms, 0.0);
        assert_eq!(stats.p50_duration_ms, 0.0);
        assert_eq!(stats.p95_duration_ms, 0.0);
        assert!(stats.event_type_counts.is_empty());
    }

    #[test]
    fn test_traces_to_csv_rows() {
        let traces = sample_traces();
        let rows = traces_to_csv_rows(&traces);

        // header + 2 events (t1) + 1 event (t2) + 3 events (t3) = 7
        assert_eq!(rows.len(), 7);

        // Check header
        assert_eq!(rows[0], vec!["trace_id", "timestamp", "outcome", "event_type", "duration_ms"]);

        // Check first data row
        assert_eq!(rows[1][0], "t1");
        assert_eq!(rows[1][3], "parse");
        assert_eq!(rows[1][4], "10");

        // Check last data row
        assert_eq!(rows[6][0], "t3");
        assert_eq!(rows[6][3], "render");
        assert_eq!(rows[6][4], "30");
    }

    #[test]
    fn test_default_trace_dir() {
        let dir = default_trace_dir();
        // Should end with .ga/traces (or .ga\traces on Windows)
        let dir_str = dir.to_string_lossy();
        assert!(
            dir_str.contains(".ga") && (dir_str.contains("traces")),
            "Expected path to contain .ga/traces, got: {}",
            dir_str
        );
    }
}
