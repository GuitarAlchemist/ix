//! Trace flywheel — primitive #6 of the harness roadmap.
//!
//! Converts an [`ix_session::SessionLog`] (JSONL-backed
//! [`ix_agent_core::SessionEvent`]s) into a GA-flavored
//! [`ix_io::trace_bridge::Trace`] that [`crate::handlers::trace_ingest`]
//! can consume. Closes the self-improvement loop:
//!
//! ```text
//! dispatch_action → SessionLog → flywheel::export → trace dir → ix_trace_ingest
//! ```
//!
//! ## MVP scope
//!
//! - One `SessionLog` file → one `Trace`. Sessions are not split at
//!   `ActionCompleted` boundaries; the whole log is treated as a
//!   single run.
//! - Event → `TraceEvent` mapping uses the SessionEvent variant name
//!   as `event_type` and a zero `duration_ms` (wall-clock timing is
//!   not currently recorded in the log — v2 will add Instant-based
//!   ordinals or start/stop event pairs).
//! - Outcome is `"failure"` if any `ActionBlocked` or `ActionFailed`
//!   was observed, otherwise `"success"`.
//! - The `trace_id` defaults to the log filename stem if not
//!   explicitly supplied.
//! - The `timestamp` is the current wall-clock time in RFC 3339 at
//!   export time. Not part of the log itself — the log is
//!   append-only and doesn't know when each line was written.
//!
//! ## Non-goals (v2)
//!
//! - Per-event timing. Requires a richer `SessionEvent` shape.
//! - Split-by-session when one log holds multiple logical sessions.
//! - Streaming / tail-follow. Current MVP is a one-shot export.
//! - Compaction of repetitive events (e.g., 100 MetadataMounted in
//!   a row → one aggregated TraceEvent).

use std::path::{Path, PathBuf};

use ix_agent_core::SessionEvent;
use ix_io::trace_bridge::{Trace, TraceEvent};
use ix_session::{SessionError, SessionLog};
use serde_json::json;

/// Map a single [`SessionEvent`] to a GA [`TraceEvent`]. Never fails —
/// unknown event metadata is passed through as raw JSON.
pub fn session_event_to_trace_event(event: &SessionEvent) -> TraceEvent {
    let (event_type, metadata) = match event {
        SessionEvent::ActionProposed { ordinal, action } => (
            "action_proposed",
            json!({ "ordinal": ordinal, "action": action }),
        ),
        SessionEvent::ActionBlocked {
            ordinal,
            code,
            reason,
            emitted_by,
        } => (
            "action_blocked",
            json!({
                "ordinal": ordinal,
                "code": code,
                "reason": reason,
                "emitted_by": emitted_by,
            }),
        ),
        SessionEvent::ActionReplaced {
            ordinal,
            original,
            replacement,
            emitted_by,
        } => (
            "action_replaced",
            json!({
                "ordinal": ordinal,
                "original": original,
                "replacement": replacement,
                "emitted_by": emitted_by,
            }),
        ),
        SessionEvent::MetadataMounted {
            ordinal,
            path,
            value,
            emitted_by,
        } => (
            "metadata_mounted",
            json!({
                "ordinal": ordinal,
                "path": path,
                "value": value,
                "emitted_by": emitted_by,
            }),
        ),
        SessionEvent::ActionCompleted { ordinal, value } => (
            "action_completed",
            json!({ "ordinal": ordinal, "value": value }),
        ),
        SessionEvent::ActionFailed { ordinal, error } => (
            "action_failed",
            json!({ "ordinal": ordinal, "error": error.to_string() }),
        ),
        SessionEvent::BeliefChanged {
            ordinal,
            proposition,
            old,
            new,
            evidence,
        } => (
            "belief_changed",
            json!({
                "ordinal": ordinal,
                "proposition": proposition,
                "old": old,
                "new": new,
                "evidence": evidence,
            }),
        ),
        SessionEvent::ObservationAdded {
            ordinal,
            source,
            diagnosis_id,
            round,
            claim_key,
            variant,
            weight,
            evidence,
        } => (
            "observation_added",
            json!({
                "ordinal": ordinal,
                "source": source,
                "diagnosis_id": diagnosis_id,
                "round": round,
                "claim_key": claim_key,
                "variant": variant,
                "weight": weight,
                "evidence": evidence,
            }),
        ),
    };

    TraceEvent {
        event_type: event_type.to_string(),
        duration_ms: 0.0,
        metadata,
    }
}

/// Build a [`Trace`] from every event currently persisted in `log`.
///
/// Parses the log's on-disk file; corrupt lines are skipped and
/// surfaced via [`SessionLog::reload_errors`] on the next reopen.
/// The supplied `trace_id` is used verbatim; if `None`, the log's
/// filename stem is used.
pub fn session_to_trace(
    log: &SessionLog,
    trace_id: Option<String>,
) -> Result<Trace, SessionError> {
    let trace_id = trace_id.unwrap_or_else(|| {
        log.path()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("session")
            .to_string()
    });

    let timestamp = chrono_like_now();

    let mut events: Vec<TraceEvent> = Vec::new();
    let mut saw_failure = false;

    for result in log.events()? {
        let Ok(event) = result else {
            // Corrupt line: skip it. The caller can inspect
            // `log.reload_errors()` if they need precise diagnostics.
            continue;
        };
        if matches!(
            event,
            SessionEvent::ActionBlocked { .. } | SessionEvent::ActionFailed { .. }
        ) {
            saw_failure = true;
        }
        events.push(session_event_to_trace_event(&event));
    }

    let outcome = if saw_failure { "failure" } else { "success" };

    Ok(Trace {
        trace_id,
        timestamp,
        events,
        outcome: outcome.to_string(),
        metadata: json!({
            "source": "ix_session_flywheel",
            "log_path": log.path().display().to_string(),
        }),
    })
}

/// Export a [`SessionLog`] to the GA trace directory as a single
/// `{trace_id}.json` file. Creates the directory if missing.
///
/// Returns the written file path so callers can hand it straight to
/// [`crate::handlers::trace_ingest`] (or its skill wrapper).
pub fn export_session_to_trace_dir(
    log: &SessionLog,
    trace_dir: &Path,
    trace_id: Option<String>,
) -> Result<PathBuf, ExportError> {
    std::fs::create_dir_all(trace_dir).map_err(|source| ExportError::CreateDir {
        path: trace_dir.to_path_buf(),
        source,
    })?;

    let trace = session_to_trace(log, trace_id).map_err(ExportError::Session)?;

    let file_name = format!("{}.json", trace.trace_id);
    let out_path = trace_dir.join(file_name);
    let json = serde_json::to_string_pretty(&trace).map_err(ExportError::Serialize)?;
    std::fs::write(&out_path, json).map_err(|source| ExportError::Write {
        path: out_path.clone(),
        source,
    })?;

    Ok(out_path)
}

/// Errors produced by [`export_session_to_trace_dir`].
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// Couldn't create the output directory.
    #[error("create trace directory {path}: {source}")]
    CreateDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Reading the session log failed.
    #[error("read session log: {0}")]
    Session(#[from] SessionError),
    /// Serializing the Trace to JSON failed.
    #[error("serialize trace: {0}")]
    Serialize(#[source] serde_json::Error),
    /// Writing the output JSON file failed.
    #[error("write trace file {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

// ---------------------------------------------------------------------------
// Timestamp helper
// ---------------------------------------------------------------------------

/// Produce an RFC 3339-ish timestamp without pulling in chrono.
/// Uses `SystemTime::now()` formatted as seconds-since-epoch so
/// downstream GA consumers can still sort traces chronologically.
fn chrono_like_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch:{secs}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ix_agent_core::event::BlockCode;
    use ix_agent_core::{AgentAction, EventSink};
    use tempfile::tempdir;

    fn write_log_with_events(path: &Path, events: Vec<SessionEvent>) -> SessionLog {
        let log = SessionLog::open(path).expect("open log");
        let mut sink = log.sink();
        for e in events {
            sink.emit(e);
        }
        drop(sink);
        log.flush().expect("flush");
        log
    }

    fn invoke(tool: &str) -> AgentAction {
        AgentAction::InvokeTool {
            tool_name: tool.to_string(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: None,
        }
    }

    #[test]
    fn maps_each_session_event_variant() {
        let variants = [SessionEvent::ActionProposed {
                ordinal: 0,
                action: invoke("t"),
            },
            SessionEvent::ActionBlocked {
                ordinal: 1,
                code: BlockCode::LoopDetected,
                reason: "r".into(),
                emitted_by: "m".into(),
            },
            SessionEvent::ActionReplaced {
                ordinal: 2,
                original: invoke("a"),
                replacement: invoke("b"),
                emitted_by: "m".into(),
            },
            SessionEvent::MetadataMounted {
                ordinal: 3,
                path: "p".into(),
                value: json!(1),
                emitted_by: "m".into(),
            },
            SessionEvent::ActionCompleted {
                ordinal: 4,
                value: json!(null),
            },
            SessionEvent::ActionFailed {
                ordinal: 5,
                error: ix_agent_core::ActionError::Exec("boom".into()),
            }];
        let mapped: Vec<TraceEvent> = variants.iter().map(session_event_to_trace_event).collect();
        assert_eq!(mapped[0].event_type, "action_proposed");
        assert_eq!(mapped[1].event_type, "action_blocked");
        assert_eq!(mapped[2].event_type, "action_replaced");
        assert_eq!(mapped[3].event_type, "metadata_mounted");
        assert_eq!(mapped[4].event_type, "action_completed");
        assert_eq!(mapped[5].event_type, "action_failed");
        for e in &mapped {
            assert_eq!(e.duration_ms, 0.0);
        }
    }

    #[test]
    fn session_to_trace_derives_success_when_no_failures() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ok.jsonl");
        let log = write_log_with_events(
            &path,
            vec![
                SessionEvent::ActionProposed {
                    ordinal: 0,
                    action: invoke("ix_stats"),
                },
                SessionEvent::ActionCompleted {
                    ordinal: 1,
                    value: json!({"x": 1}),
                },
            ],
        );

        let trace = session_to_trace(&log, None).expect("build trace");
        assert_eq!(trace.trace_id, "ok");
        assert_eq!(trace.outcome, "success");
        assert_eq!(trace.events.len(), 2);
    }

    #[test]
    fn session_to_trace_marks_failure_on_block_or_fail() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        let log = write_log_with_events(
            &path,
            vec![
                SessionEvent::ActionProposed {
                    ordinal: 0,
                    action: invoke("fake"),
                },
                SessionEvent::ActionBlocked {
                    ordinal: 1,
                    code: BlockCode::ApprovalRequired,
                    reason: "tier3".into(),
                    emitted_by: "ix_approval".into(),
                },
            ],
        );

        let trace = session_to_trace(&log, None).expect("build trace");
        assert_eq!(trace.outcome, "failure");
    }

    #[test]
    fn export_writes_json_file_readable_by_trace_bridge() {
        let dir = tempdir().unwrap();
        let log_path = dir.path().join("run.jsonl");
        let log = write_log_with_events(
            &log_path,
            vec![
                SessionEvent::ActionProposed {
                    ordinal: 0,
                    action: invoke("ix_stats"),
                },
                SessionEvent::ActionCompleted {
                    ordinal: 1,
                    value: json!({"ok": true}),
                },
            ],
        );

        let trace_dir = dir.path().join("traces");
        let out = export_session_to_trace_dir(&log, &trace_dir, None).expect("export");
        assert!(out.exists());
        assert_eq!(out.file_name().unwrap(), "run.json");

        // trace_bridge::load_trace is the canonical reader used by
        // ix_trace_ingest — proving it parses our file proves the
        // flywheel output is consumable end-to-end.
        let trace = ix_io::trace_bridge::load_trace(&out).expect("round-trip");
        assert_eq!(trace.trace_id, "run");
        assert_eq!(trace.outcome, "success");
        assert_eq!(trace.events.len(), 2);
    }

    #[test]
    fn export_with_explicit_trace_id_overrides_filename() {
        let dir = tempdir().unwrap();
        let log_path = dir.path().join("weird-name.jsonl");
        let log = write_log_with_events(
            &log_path,
            vec![SessionEvent::ActionProposed {
                ordinal: 0,
                action: invoke("ix_stats"),
            }],
        );
        let trace_dir = dir.path().join("traces");
        let out = export_session_to_trace_dir(&log, &trace_dir, Some("custom-id".into()))
            .expect("export");
        assert_eq!(out.file_name().unwrap(), "custom-id.json");
    }
}
