//! Integration tests for [`ix_session::SessionLog`].
//!
//! Uses `tempfile::TempDir` to get a real on-disk log per test and
//! verifies the full open → emit → close → reopen → read cycle.

use std::path::PathBuf;

use ix_agent_core::{AgentAction, EventSink, SessionEvent};
use ix_session::{SessionLog, SessionSink};
use serde_json::json;

fn invoke(tool: &str, ordinal: u64) -> AgentAction {
    AgentAction::InvokeTool {
        tool_name: tool.to_string(),
        params: json!({}),
        ordinal,
        target_hint: None,
    }
}

fn action_completed(ordinal: u64, value: serde_json::Value) -> SessionEvent {
    SessionEvent::ActionCompleted { ordinal, value }
}

fn action_proposed(ordinal: u64, tool: &str) -> SessionEvent {
    SessionEvent::ActionProposed {
        ordinal,
        action: invoke(tool, ordinal),
    }
}

// ---------------------------------------------------------------------------
// Open / create
// ---------------------------------------------------------------------------

#[test]
fn open_new_file_starts_at_zero() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("session.jsonl");
    assert!(!path.exists());

    let log = SessionLog::open(&path).expect("open");
    assert_eq!(log.next_ordinal(), 0);
    assert!(log.reload_errors().is_empty());
    // The file should exist now.
    assert!(path.exists());
}

#[test]
fn open_creates_missing_parent_directories() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("state/events/deep/nested/session.jsonl");
    assert!(!path.parent().unwrap().exists());

    let log = SessionLog::open(&path).expect("open with parent creation");
    assert!(path.exists());
    assert_eq!(log.next_ordinal(), 0);
}

// ---------------------------------------------------------------------------
// Single-emit round-trip
// ---------------------------------------------------------------------------

#[test]
fn emit_then_reload_preserves_event() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("session.jsonl");

    let log = SessionLog::open(&path).expect("open");
    let mut sink = log.sink();
    sink.emit(action_completed(0, json!({"value": 42})));
    assert!(sink.last_error().is_none());

    // Reopen and iterate.
    drop(sink);
    drop(log);
    let reopened = SessionLog::open(&path).expect("reopen");
    assert_eq!(reopened.next_ordinal(), 1);

    let events: Vec<_> = reopened
        .events()
        .expect("events iter")
        .collect::<Result<Vec<_>, _>>()
        .expect("all lines valid");
    assert_eq!(events.len(), 1);
    match &events[0] {
        SessionEvent::ActionCompleted { ordinal, value } => {
            assert_eq!(*ordinal, 0);
            assert_eq!(value, &json!({"value": 42}));
        }
        other => panic!("expected ActionCompleted, got {other:?}"),
    }
}

#[test]
fn emit_advances_next_ordinal() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("session.jsonl");

    let log = SessionLog::open(&path).expect("open");
    assert_eq!(log.next_ordinal(), 0);

    let mut sink = log.sink();
    sink.emit(action_completed(0, json!("first")));
    assert_eq!(sink.next_ordinal(), 1);

    sink.emit(action_completed(1, json!("second")));
    assert_eq!(sink.next_ordinal(), 2);

    sink.emit(action_completed(2, json!("third")));
    assert_eq!(sink.next_ordinal(), 3);
}

// ---------------------------------------------------------------------------
// Multi-event reload
// ---------------------------------------------------------------------------

#[test]
fn reload_after_multiple_emits_restores_ordinal() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path: PathBuf = tmp.path().join("multi.jsonl");

    {
        let log = SessionLog::open(&path).expect("open");
        let mut sink = log.sink();
        for i in 0..5u64 {
            sink.emit(action_completed(i, json!(i)));
        }
        assert_eq!(sink.next_ordinal(), 5);
    } // log + sink dropped here, file closed

    // Reopen and check ordinal.
    let reopened = SessionLog::open(&path).expect("reopen");
    assert_eq!(reopened.next_ordinal(), 5);
    assert!(reopened.reload_errors().is_empty());

    let events: Vec<_> = reopened
        .events()
        .expect("events iter")
        .collect::<Result<Vec<_>, _>>()
        .expect("all valid");
    assert_eq!(events.len(), 5);
    for (i, event) in events.iter().enumerate() {
        match event {
            SessionEvent::ActionCompleted { ordinal, .. } => {
                assert_eq!(*ordinal, i as u64);
            }
            _ => panic!("wrong variant at index {i}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Corrupted line handling
// ---------------------------------------------------------------------------

#[test]
fn reload_skips_blank_lines() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("blanks.jsonl");

    // Write a file with blank lines between events.
    let event_json = serde_json::to_string(&action_completed(0, json!("ok"))).expect("serialize");
    let contents = format!("{event_json}\n\n{event_json}\n\n");
    std::fs::write(&path, contents).expect("write fixture");

    let log = SessionLog::open(&path).expect("open");
    // Two real events, blank lines skipped.
    assert_eq!(log.next_ordinal(), 2);
    assert!(log.reload_errors().is_empty());
}

#[test]
fn reload_records_bad_json_in_reload_errors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("corrupt.jsonl");

    // Write a file with one good event, one garbage line, one good event.
    let event_json = serde_json::to_string(&action_completed(0, json!("ok"))).expect("serialize");
    let contents = format!("{event_json}\n{{not valid json}}\n{event_json}\n");
    std::fs::write(&path, contents).expect("write fixture");

    let log = SessionLog::open(&path).expect("open");
    // Two valid events counted.
    assert_eq!(log.next_ordinal(), 2);
    // One reload error for the bad line.
    assert_eq!(log.reload_errors().len(), 1);
}

// ---------------------------------------------------------------------------
// Iterator behavior
// ---------------------------------------------------------------------------

#[test]
fn events_iterator_propagates_bad_json_as_err() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("mixed.jsonl");

    let good = serde_json::to_string(&action_completed(0, json!("a"))).unwrap();
    let contents = format!("{good}\nGARBAGE\n{good}\n");
    std::fs::write(&path, contents).expect("write fixture");

    let log = SessionLog::open(&path).expect("open");
    let results: Vec<_> = log.events().expect("events iter").collect();
    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok());
    assert!(results[1].is_err());
    assert!(results[2].is_ok());
}

// ---------------------------------------------------------------------------
// Different SessionEvent variants all round-trip
// ---------------------------------------------------------------------------

#[test]
fn all_session_event_variants_round_trip() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("variants.jsonl");

    let log = SessionLog::open(&path).expect("open");
    let mut sink = log.sink();

    sink.emit(action_proposed(0, "ix_stats"));
    sink.emit(SessionEvent::ActionCompleted {
        ordinal: 1,
        value: json!({"mean": 2.5}),
    });
    sink.emit(SessionEvent::MetadataMounted {
        ordinal: 2,
        path: "approval/verdict".into(),
        value: json!({"tier": "tier_one"}),
        emitted_by: "ix_approval".into(),
    });

    assert_eq!(sink.next_ordinal(), 3);

    drop(sink);
    drop(log);

    let reopened = SessionLog::open(&path).expect("reopen");
    let events: Vec<_> = reopened
        .events()
        .expect("events iter")
        .collect::<Result<Vec<_>, _>>()
        .expect("all valid");
    assert_eq!(events.len(), 3);
    assert!(matches!(&events[0], SessionEvent::ActionProposed { .. }));
    assert!(matches!(&events[1], SessionEvent::ActionCompleted { .. }));
    assert!(matches!(&events[2], SessionEvent::MetadataMounted { .. }));
}

// ---------------------------------------------------------------------------
// Flush semantics
// ---------------------------------------------------------------------------

#[test]
fn emit_flushes_by_default_so_concurrent_reader_sees_events() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("flush.jsonl");

    let log = SessionLog::open(&path).expect("open");
    let mut sink = log.sink();
    sink.emit(action_completed(0, json!("visible")));

    // Immediately read (without dropping the writer) — the flush
    // inside try_emit should have made the event visible.
    let events: Vec<_> = log.events().expect("iter").collect();
    assert_eq!(events.len(), 1);
    assert!(events[0].is_ok());
}

// ---------------------------------------------------------------------------
// EventSink trait integration
// ---------------------------------------------------------------------------

#[test]
fn sink_satisfies_event_sink_trait_bound() {
    // Compile-time: a SessionSink can be passed as &mut dyn EventSink.
    fn takes_dyn_sink(sink: &mut dyn EventSink) {
        sink.emit(action_completed(0, json!("dyn")));
    }

    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("dyn.jsonl");
    let log = SessionLog::open(&path).expect("open");
    let mut sink: SessionSink<'_> = log.sink();
    takes_dyn_sink(&mut sink);
    assert!(sink.last_error().is_none());
    assert_eq!(sink.next_ordinal(), 1);
}

// ---------------------------------------------------------------------------
// Concurrent writes (two sinks, one log)
// ---------------------------------------------------------------------------

#[test]
fn two_sinks_on_one_log_share_writer_mutex() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("concurrent.jsonl");
    let log = SessionLog::open(&path).expect("open");

    let mut sink_a = log.sink();
    let mut sink_b = log.sink();

    sink_a.emit(action_completed(0, json!("a1")));
    sink_b.emit(action_completed(1, json!("b1")));
    sink_a.emit(action_completed(2, json!("a2")));
    sink_b.emit(action_completed(3, json!("b2")));

    assert_eq!(log.next_ordinal(), 4);

    drop(sink_a);
    drop(sink_b);
    drop(log);

    let reopened = SessionLog::open(&path).expect("reopen");
    let events: Vec<_> = reopened
        .events()
        .expect("iter")
        .collect::<Result<Vec<_>, _>>()
        .expect("all valid");
    assert_eq!(events.len(), 4);
}
