//! Integration test for primitive #4 wiring: `dispatch_action` must
//! route its emitted [`ix_agent_core::SessionEvent`]s through an
//! installed [`ix_session::SessionLog`] so events survive the process.
//!
//! Uses its own test binary so the global `session_log_slot` in
//! `registry_bridge` doesn't leak across unrelated tests in
//! `parity.rs`. Each `tests/*.rs` file compiles to a separate binary.

use ix_agent::registry_bridge::{
    clear_session_log, dispatch_action, install_session_log, shared_loop_detector,
};
use ix_agent_core::event::BlockCode;
use ix_agent_core::{ActionError, AgentAction, ReadContext, SessionEvent};
use ix_session::SessionLog;
use tempfile::tempdir;

/// Tier 1 read-only tool: events from an auto-approving dispatch
/// should land on disk and be replayable via `SessionLog::events`.
#[test]
fn dispatch_action_writes_events_to_installed_session_log() {
    let dir = tempdir().expect("create tempdir");
    let path = dir.path().join("session.jsonl");
    let log = SessionLog::open(&path).expect("open session log");
    install_session_log(log);

    // Prevent the shared loop detector from tripping if earlier runs
    // of this test binary left residue. Paranoia: we only make one
    // call, so the detector wouldn't trip anyway, but this keeps the
    // test robust if it grows later.
    shared_loop_detector().clear_key("ix_stats");

    let cx = ReadContext::synthetic_for_legacy();
    let action = AgentAction::InvokeTool {
        tool_name: "ix_stats".to_string(),
        params: serde_json::json!({ "data": [1.0, 2.0, 3.0, 4.0, 5.0] }),
        ordinal: 0,
        target_hint: None,
    };

    let outcome =
        dispatch_action(&cx, action).expect("ix_stats is Tier 1 Read and should auto-approve");
    assert!(outcome.value.is_object());

    // Reopen the log from disk to prove events were flushed, not
    // just buffered in memory.
    let replay = SessionLog::open(&path).expect("reopen log");
    let events: Vec<SessionEvent> = replay
        .events()
        .expect("iter events")
        .collect::<Result<_, _>>()
        .expect("all lines valid JSON");

    assert!(
        !events.is_empty(),
        "expected approval middleware to emit at least one event, got none"
    );
    // ApprovalMiddleware mounts its classification as
    // `approval/verdict` — proves the full chain ran and the event
    // made it onto disk.
    let saw_approval_verdict = events.iter().any(|e| {
        matches!(
            e,
            SessionEvent::MetadataMounted { path, .. }
                if path == "approval/verdict"
        )
    });
    assert!(
        saw_approval_verdict,
        "expected MetadataMounted(approval/verdict), got {events:?}"
    );

    clear_session_log();
}

/// Tier 3 block path: an `ApprovalRequired` verdict should still
/// persist its event (the middleware emits `MetadataMounted` before
/// returning `Block`).
#[test]
fn blocked_dispatch_still_persists_approval_metadata() {
    let dir = tempdir().expect("create tempdir");
    let path = dir.path().join("blocked.jsonl");
    let log = SessionLog::open(&path).expect("open session log");
    install_session_log(log);

    let cx = ReadContext::synthetic_for_legacy();
    let action = AgentAction::InvokeTool {
        tool_name: "totally_fake_tool_xyz".to_string(),
        params: serde_json::json!({}),
        ordinal: 0,
        target_hint: None,
    };

    match dispatch_action(&cx, action) {
        Err(ActionError::Blocked { code, .. }) => {
            assert_eq!(code, BlockCode::ApprovalRequired);
        }
        other => panic!("expected Blocked, got {other:?}"),
    }

    let replay = SessionLog::open(&path).expect("reopen log");
    let events: Vec<SessionEvent> = replay
        .events()
        .expect("iter events")
        .collect::<Result<_, _>>()
        .expect("all lines valid JSON");
    assert!(
        events
            .iter()
            .any(|e| matches!(e, SessionEvent::MetadataMounted { path, .. }
                if path == "approval/verdict")),
        "expected MetadataMounted from ix_approval before block, got {events:?}"
    );

    clear_session_log();
}

/// End-to-end trace flywheel (primitive #6): persist events to a
/// session log via dispatch_action, export it to a GA trace file,
/// then feed that file back to `ix_trace_ingest` — closing the
/// self-improvement loop without any hand-edited data.
#[test]
fn flywheel_round_trip_session_log_to_trace_ingest() {
    let dir = tempdir().expect("create tempdir");
    let log_path = dir.path().join("flywheel.jsonl");
    let log = SessionLog::open(&log_path).expect("open session log");
    install_session_log(log);

    shared_loop_detector().clear_key("ix_stats");

    // Drive a couple of real dispatches so the session log has
    // content to convert.
    let cx = ReadContext::synthetic_for_legacy();
    for _ in 0..2 {
        let action = AgentAction::InvokeTool {
            tool_name: "ix_stats".to_string(),
            params: serde_json::json!({ "data": [1.0, 2.0, 3.0] }),
            ordinal: 0,
            target_hint: None,
        };
        dispatch_action(&cx, action).expect("ix_stats auto-approves");
    }

    // Flush the sink by clearing the install so the log's Drop
    // path finalises. The file is already flush-on-emit, so this
    // is belt-and-suspenders.
    clear_session_log();

    // Reopen a fresh handle and export via the flywheel.
    let log = SessionLog::open(&log_path).expect("reopen");
    let trace_dir = dir.path().join("traces");
    let written = ix_agent::flywheel::export_session_to_trace_dir(&log, &trace_dir, None)
        .expect("flywheel export");
    assert!(written.exists(), "trace file should exist on disk");

    // Now feed the trace directory to ix_trace_ingest via the
    // registry-backed dispatch path — same code path an agent
    // would use to close its own loop.
    let stats = ix_agent::registry_bridge::dispatch(
        "ix_trace_ingest",
        serde_json::json!({ "dir": trace_dir.display().to_string() }),
    )
    .expect("ix_trace_ingest should succeed");
    let total = stats["total_traces"].as_u64().unwrap_or(0);
    assert_eq!(total, 1, "expected one ingested trace, got {stats}");
}

/// When no log is installed, dispatch falls back to the in-memory
/// sink and produces no filesystem side effects — protects the
/// existing `parity.rs` tests which run without a log.
#[test]
fn dispatch_without_installed_log_is_in_memory_only() {
    clear_session_log();
    shared_loop_detector().clear_key("ix_stats");

    let cx = ReadContext::synthetic_for_legacy();
    let action = AgentAction::InvokeTool {
        tool_name: "ix_stats".to_string(),
        params: serde_json::json!({ "data": [1.0, 2.0, 3.0] }),
        ordinal: 0,
        target_hint: None,
    };

    let outcome = dispatch_action(&cx, action).expect("auto-approve");
    assert!(outcome.value.is_object());
    // No assertion on filesystem — the absence of a log path means
    // there is nothing to observe. The test's purpose is to prove
    // dispatch doesn't panic or error when no sink is installed.
}
