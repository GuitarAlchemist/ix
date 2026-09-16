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
pub fn session_to_trace(log: &SessionLog, trace_id: Option<String>) -> Result<Trace, SessionError> {
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
/// The trace id must be a plain file name (see [`check_trace_id`]), so the
/// file always lands directly in `trace_dir`. Whatever already sits at the
/// destination is removed and the file is created afresh, so a symlink or
/// hard link planted there is replaced rather than written through. A link
/// swapped in between the removal and the create makes the create fail; it is
/// not followed.
///
/// Returns the written file path so callers can hand it straight to
/// [`crate::handlers::trace_ingest`] (or its skill wrapper).
pub fn export_session_to_trace_dir(
    log: &SessionLog,
    trace_dir: &Path,
    trace_id: Option<String>,
) -> Result<PathBuf, ExportError> {
    let trace = session_to_trace(log, trace_id).map_err(ExportError::Session)?;
    check_trace_id(&trace.trace_id)?;
    let out_path = trace_dir.join(format!("{}.json", trace.trace_id));
    if out_path.parent() != Some(trace_dir) {
        return Err(ExportError::InvalidTraceId(trace.trace_id));
    }

    std::fs::create_dir_all(trace_dir).map_err(|source| ExportError::CreateDir {
        path: trace_dir.to_path_buf(),
        source,
    })?;

    let json = serde_json::to_string_pretty(&trace).map_err(ExportError::Serialize)?;
    let write_err = |source| ExportError::Write {
        path: out_path.clone(),
        source,
    };
    match std::fs::symlink_metadata(&out_path) {
        Ok(_) => std::fs::remove_file(&out_path).map_err(write_err)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(write_err(e)),
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out_path)
        .map_err(write_err)?;
    std::io::Write::write_all(&mut file, json.as_bytes()).map_err(write_err)?;

    Ok(out_path)
}

/// A trace id becomes the file name `{trace_id}.json`, so it must be a single
/// plain name: not empty, no leading dot or trailing dot or space, no path
/// separator, drive or stream colon, wildcard or control character (NUL
/// included), and not a Windows reserved device name such as `CON` or `COM1`.
/// The same rule applies on every platform so a trace id that exports on one
/// exports on all.
pub fn check_trace_id(trace_id: &str) -> Result<(), ExportError> {
    let bad_char = |c: char| {
        c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
    };
    let device = trace_id
        .split('.')
        .next()
        .unwrap_or_default()
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    let reserved = matches!(
        device.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ((device.starts_with("COM") || device.starts_with("LPT"))
        && device.chars().count() == 4
        && device[3..]
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '¹' | '²' | '³')));
    if trace_id.is_empty()
        || trace_id.starts_with('.')
        || trace_id.ends_with(['.', ' '])
        || trace_id.chars().any(bad_char)
        || reserved
    {
        return Err(ExportError::InvalidTraceId(trace_id.to_string()));
    }
    Ok(())
}

/// Errors produced by [`export_session_to_trace_dir`].
#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// The trace id is not a plain file name (see [`check_trace_id`]).
    #[error("trace_id {0:?} is not a plain file name (no path separators, `:`, leading dot, trailing dot or space, control characters or reserved device names)")]
    InvalidTraceId(String),
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
        let variants = [
            SessionEvent::ActionProposed {
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
            },
        ];
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

    #[test]
    fn trace_ids_that_are_not_plain_file_names_are_refused() {
        let dir = tempdir().unwrap();
        let elsewhere = tempdir().unwrap();
        let log = write_log_with_events(&dir.path().join("run.jsonl"), vec![]);
        let trace_dir = dir.path().join("traces");
        let absolute = elsewhere.path().join("settings");
        let refused = [
            absolute.to_str().unwrap(),
            "../settings",
            "a/b",
            "a\\b",
            "C:x",
            "x:stream",
            "",
            ".",
            "..",
            ".hidden",
            "trailing.",
            "trailing ",
            "nul\0byte",
            "CON",
            "con.backup",
            "COM1",
            "lpt9",
            "wild*",
        ];
        for id in refused {
            let err = export_session_to_trace_dir(&log, &trace_dir, Some(id.to_string()))
                .expect_err(id);
            assert!(matches!(err, ExportError::InvalidTraceId(_)), "{id:?}: {err}");
        }
        assert!(!trace_dir.exists(), "a refused export must not create the dir");
        assert!(std::fs::read_dir(elsewhere.path()).unwrap().next().is_none());

        for id in ["run-2026.09.16", "session_1", "CONSOLE", "COM10", "a b"] {
            assert!(check_trace_id(id).is_ok(), "{id:?}");
        }
    }

    #[test]
    fn export_replaces_a_link_at_the_destination_instead_of_writing_through_it() {
        let dir = tempdir().unwrap();
        let log = write_log_with_events(&dir.path().join("run.jsonl"), vec![]);
        let trace_dir = dir.path().join("traces");
        std::fs::create_dir_all(&trace_dir).unwrap();
        let victim = dir.path().join("victim.json");
        std::fs::write(&victim, "KEEP").unwrap();
        // A hard link needs no privileges on NTFS or Unix and redirects a
        // truncating write the same way a symlink does.
        std::fs::hard_link(&victim, trace_dir.join("run.json")).unwrap();

        let out = export_session_to_trace_dir(&log, &trace_dir, None).expect("export");
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "KEEP");
        assert!(ix_io::trace_bridge::load_trace(&out).is_ok());
    }
}
