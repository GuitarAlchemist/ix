//! End-to-end integration test for `ix_triage_session`.
//!
//! Drives the full harness in one test binary:
//!
//!   1. Installs a real [`SessionLog`] at a temp path and pre-seeds
//!      it with a handful of events so the triage prompt has history.
//!   2. Stands up a [`ServerContext`] and spawns a stub sampling
//!      thread that intercepts `sampling/createMessage` outbound
//!      envelopes and responds with a canned plan JSON.
//!   3. Calls [`ix_agent::handlers::triage_session_with_ctx`] and
//!      asserts the full result shape: plan parsed, distribution
//!      built, dispatched actions recorded, escalation flag, and
//!      (for the happy path) a trace file written to the log's
//!      sibling traces directory.
//!   4. Covers three scenarios:
//!      - happy path: valid plan → dispatched + flywheel export
//!      - escalation: all-`C` plan → escalated before dispatch
//!      - recursion guard: plan proposes ix_triage_session → parse
//!        fails with Recursion, handler returns parse_failed status

use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Mutex, OnceLock};
use std::thread;
use std::time::Duration;

/// Serialize all tests in this binary — they share process-wide state
/// (session log slot, loop detector) and would race under cargo's
/// default multi-threaded test runner. Each test grabs this mutex
/// for its whole duration; held-guard ordering is uncontended because
/// tests run sequentially through this one point.
fn test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

use ix_agent::handlers::triage_session_with_ctx;
use ix_agent::registry_bridge::{clear_session_log, install_session_log, shared_loop_detector};
use ix_agent::server_context::ServerContext;
use ix_agent_core::{AgentAction, EventSink, SessionEvent};
use ix_session::SessionLog;
use serde_json::{json, Value};
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------------

/// Spawn a thread that pretends to be a sampling-capable MCP client:
/// it reads outbound envelopes from `rx`, parses the `id` + method,
/// and calls `ctx.deliver_response` with a response whose `content.text`
/// field is `canned_response`. Panics on unexpected method names so
/// mistakes surface loudly in test output.
fn spawn_stub_client(ctx: ServerContext, rx: Receiver<String>, canned_response: String) {
    thread::spawn(move || {
        // Wait up to 10s for an outbound message. If nothing arrives,
        // the test handler probably didn't call `sample()` and will
        // fail for its own reasons.
        let line = match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(l) => l,
            Err(RecvTimeoutError::Timeout) => return,
            Err(RecvTimeoutError::Disconnected) => return,
        };
        let envelope: Value = serde_json::from_str(&line).expect("outbound is JSON");
        let id = envelope
            .get("id")
            .and_then(|v| v.as_i64())
            .expect("outbound envelope has id");
        let method = envelope
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert_eq!(
            method, "sampling/createMessage",
            "stub only handles sampling/createMessage, got {method}"
        );

        let response = json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "role": "assistant",
                "model": "stub-model-v0",
                "stopReason": "endTurn",
                "content": {
                    "type": "text",
                    "text": canned_response
                }
            }
        });
        ctx.deliver_response(id, response);
    });
}

/// Build a seeded SessionLog at a temp path with a few plausible
/// events so the triage summary isn't empty.
fn seed_session_log(dir: &std::path::Path) -> SessionLog {
    let log_path = dir.join("session.jsonl");
    let log = SessionLog::open(&log_path).expect("open log");
    let mut sink = log.sink();
    sink.emit(SessionEvent::ActionProposed {
        ordinal: 0,
        action: AgentAction::InvokeTool {
            tool_name: "ix_stats".into(),
            params: json!({"data": [1.0, 2.0, 3.0]}),
            ordinal: 0,
            target_hint: None,
        },
    });
    sink.emit(SessionEvent::ActionCompleted {
        ordinal: 0,
        value: json!({"mean": 2.0, "std": 1.0}),
    });
    drop(sink);
    log.flush().expect("flush seed events");
    log
}

/// Reset state that leaks between tests in the same binary.
fn reset() {
    clear_session_log();
    // Paranoia: the shared detector is process-wide. Wipe known keys
    // our tests touch so repeated runs don't trip stale counts.
    let detector = shared_loop_detector();
    for key in [
        "ix_stats",
        "ix_distance",
        "ix_fft",
        "ix_optimize",
        "ix_triage_session",
    ] {
        detector.clear_key(key);
    }
}

// ---------------------------------------------------------------------------
// Happy path: valid plan → dispatched + flywheel export
// ---------------------------------------------------------------------------

#[test]
fn happy_path_dispatches_plan_and_exports_trace() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    reset();
    let dir = tempdir().expect("tempdir");
    let log = seed_session_log(dir.path());
    install_session_log(log);

    // Canned plan: two Tier-1 ix_stats calls with different params so
    // the loop detector doesn't trip (same tool_name+target).
    let canned = json!([
        {
            "tool_name": "ix_stats",
            "params": {"data": [10.0, 20.0, 30.0]},
            "confidence": "T",
            "reason": "baseline re-stats with new window"
        },
        {
            "tool_name": "ix_distance",
            "params": {"a": [1.0, 2.0], "b": [4.0, 6.0], "metric": "euclidean"},
            "confidence": "probable",
            "reason": "distance check"
        }
    ])
    .to_string();

    let (ctx, rx) = ServerContext::new();
    spawn_stub_client(ctx.clone(), rx, canned);

    let params = json!({
        "focus": "verify the stats investigation",
        "max_actions": 3,
        "learn": true,
    });
    let result = triage_session_with_ctx(params, &ctx).expect("triage returns Ok");

    // ── Result shape ─────────────────────────────────────────────
    assert_eq!(result["status"], "dispatched");
    assert_eq!(result["focus"], "verify the stats investigation");
    assert_eq!(result["max_actions"], 3);
    assert_eq!(result["escalated"], false);

    // ── Plan content ─────────────────────────────────────────────
    let plan = result["plan"].as_array().expect("plan is array");
    assert_eq!(plan.len(), 2);
    // Sorted by priority: P beats T in tiebreak order, so the
    // distance item (P=probable) comes first.
    assert_eq!(plan[0]["tool_name"], "ix_distance");
    assert_eq!(plan[0]["confidence"], "P");
    assert_eq!(plan[1]["tool_name"], "ix_stats");
    assert_eq!(plan[1]["confidence"], "T");

    // ── Dispatched outcomes ──────────────────────────────────────
    let dispatched = result["dispatched"].as_array().expect("dispatched array");
    assert_eq!(dispatched.len(), 2);
    for entry in dispatched {
        assert_eq!(
            entry["ok"], true,
            "expected Tier-1 ix tools to dispatch cleanly, got {entry}"
        );
        assert!(entry["value"].is_object());
    }

    // ── Distribution surfaces all six variants ───────────────────
    let dist = &result["distribution"];
    assert!(dist["T"].as_f64().unwrap() >= 0.499); // 1/2 of the plan
    assert!(dist["P"].as_f64().unwrap() >= 0.499);

    // ── Flywheel trace exported and ingested ─────────────────────
    assert!(
        !result["trace_dir"].is_null(),
        "learn=true should have exported a trace; full result: {result}"
    );
    let ingest = &result["trace_ingest"];
    assert_eq!(ingest["ok"], true, "trace_ingest should succeed: {ingest}");
    let stats = &ingest["stats"];
    // Even if no traces were valid, the handler returns a structured
    // response — assert the field exists so the wire path is proven.
    assert!(stats.is_object(), "stats should be an object: {ingest}");

    reset();
}

// ---------------------------------------------------------------------------
// Escalation path: all-`C` plan → escalated before dispatch
// ---------------------------------------------------------------------------

#[test]
fn escalation_blocks_dispatch_when_contradiction_dominates() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    reset();
    let dir = tempdir().expect("tempdir");
    let log = seed_session_log(dir.path());
    install_session_log(log);

    // All Contradictory → plan-level C mass = 1.0 >> 0.3 threshold
    let canned = json!([
        {"tool_name": "ix_stats", "params": {"data": [1.0]}, "confidence": "C", "reason": "conflicting"},
        {"tool_name": "ix_fft", "params": {}, "confidence": "C", "reason": "conflicting"},
        {"tool_name": "ix_distance", "params": {}, "confidence": "contradictory", "reason": "conflicting"}
    ])
    .to_string();

    let (ctx, rx) = ServerContext::new();
    spawn_stub_client(ctx.clone(), rx, canned);

    let params = json!({"focus": "escalation check", "learn": false});
    let result = triage_session_with_ctx(params, &ctx).expect("triage returns Ok");

    assert_eq!(result["status"], "escalated");
    assert!(result["reason"].as_str().unwrap().contains("contradiction"));
    // No dispatched field on escalation path.
    assert!(result.get("dispatched").is_none());

    let dist = &result["distribution"];
    let c_mass = dist["C"].as_f64().unwrap();
    assert!(c_mass > 0.3, "expected C mass > 0.3, got {c_mass}");

    reset();
}

// ---------------------------------------------------------------------------
// Recursion guard: LLM tries to propose ix_triage_session
// ---------------------------------------------------------------------------

#[test]
fn recursion_guard_surfaces_parse_failure() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    reset();
    let dir = tempdir().expect("tempdir");
    let log = seed_session_log(dir.path());
    install_session_log(log);

    // Malicious / buggy LLM: proposes the triage tool itself.
    let canned = json!([
        {"tool_name": "ix_triage_session", "params": {}, "confidence": "T", "reason": "infinite recursion"}
    ])
    .to_string();

    let (ctx, rx) = ServerContext::new();
    spawn_stub_client(ctx.clone(), rx, canned);

    let params = json!({"learn": false});
    let result = triage_session_with_ctx(params, &ctx).expect("triage returns Ok");

    assert_eq!(result["status"], "parse_failed");
    let error = result["error"].as_str().unwrap();
    assert!(
        error.contains("recursion")
            || error.contains("Recursion")
            || error.contains("ix_triage_session"),
        "expected recursion error, got {error}"
    );
    // Raw response is echoed back for debugging.
    assert!(result["raw_response"]
        .as_str()
        .unwrap()
        .contains("ix_triage_session"));

    reset();
}

// ---------------------------------------------------------------------------
// Missing session log: triage refuses to run
// ---------------------------------------------------------------------------

#[test]
fn refuses_without_installed_session_log() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    reset(); // ensures no log is installed

    let (ctx, _rx) = ServerContext::new();
    // No stub needed — we never reach sample().

    let params = json!({});
    let err =
        triage_session_with_ctx(params, &ctx).expect_err("expected error when no log installed");
    assert!(
        err.contains("SessionLog"),
        "expected log requirement error, got {err}"
    );
}

// ---------------------------------------------------------------------------
// Prior observations: cross-source contradiction triggers escalation
// ---------------------------------------------------------------------------

/// Verifies Step 5 integration: a plan that would normally dispatch
/// cleanly (T-confident ix_stats call) gets escalated when a
/// prior_observations payload carries a contradicting F observation
/// on the same claim_key. This is the whole point of the Path C
/// merge — cross-source disagreement visible to the escalation gate.
#[test]
fn prior_observations_escalate_on_cross_source_contradiction() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    reset();
    let dir = tempdir().expect("tempdir");
    let log = seed_session_log(dir.path());
    install_session_log(log);

    // Plan: LLM is confident ix_stats is valuable.
    let canned = json!([
        {
            "tool_name": "ix_stats",
            "params": {"data": [1.0, 2.0, 3.0]},
            "confidence": "T",
            "reason": "baseline"
        }
    ])
    .to_string();

    let (ctx, rx) = ServerContext::new();
    spawn_stub_client(ctx.clone(), rx, canned);

    // Prior observation: tars (or any other source) says the same
    // action was refuted in a previous round.
    let prior = json!([
        {
            "source": "tars",
            "diagnosis_id": "previous-round-diagnosis",
            "round": 0,
            "ordinal": 0,
            "claim_key": "ix_stats::valuable",
            "variant": "F",
            "weight": 1.0,
            "evidence": "tars saw this fail last round"
        }
    ]);

    let params = json!({
        "focus": "cross-source test",
        "max_actions": 3,
        "learn": false,
        "round": 1,
        "prior_observations": prior,
    });
    let result = triage_session_with_ctx(params, &ctx).expect("triage returns Ok");

    // Expected: the merge synthesizes a C observation from T (plan)
    // + F (prior), the merged distribution's C mass exceeds the
    // escalation threshold, and the handler returns escalated
    // status instead of dispatching.
    assert_eq!(
        result["status"], "escalated",
        "expected escalation due to cross-source contradiction, got {result:#}"
    );
    let merged_dist = &result["merged_distribution"];
    let c_mass = merged_dist["C"].as_f64().unwrap();
    assert!(
        c_mass > 0.3,
        "expected merged C mass > 0.3, got {c_mass}: {result:#}"
    );

    // The response should surface the specific contradiction so the
    // caller can see what disagreed with what.
    let contradictions = result["contradictions"].as_array().unwrap();
    assert!(
        !contradictions.is_empty(),
        "expected at least one synthesized contradiction"
    );
    assert!(
        contradictions[0]["claim_key"]
            .as_str()
            .unwrap()
            .contains("ix_stats::valuable"),
        "expected contradiction on ix_stats::valuable"
    );

    // Observation counts should reflect all three sources.
    let counts = &result["observation_counts"];
    assert_eq!(counts["plan"], 1, "one plan observation");
    assert_eq!(counts["prior"], 1, "one prior observation");
    assert!(
        counts["synthesized"].as_u64().unwrap() >= 1,
        "at least one synthesized contradiction"
    );

    reset();
}

/// Verifies that an EMPTY prior_observations payload doesn't break
/// the happy path — backward-compat for callers that haven't been
/// updated to emit observations yet.
#[test]
fn empty_prior_observations_preserves_happy_path() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    reset();
    let dir = tempdir().expect("tempdir");
    let log = seed_session_log(dir.path());
    install_session_log(log);

    let canned = json!([
        {
            "tool_name": "ix_stats",
            "params": {"data": [1.0]},
            "confidence": "T",
            "reason": "baseline"
        }
    ])
    .to_string();

    let (ctx, rx) = ServerContext::new();
    spawn_stub_client(ctx.clone(), rx, canned);

    let params = json!({
        "focus": "empty prior test",
        "learn": false,
        "prior_observations": [],
    });
    let result = triage_session_with_ctx(params, &ctx).expect("triage returns Ok");

    // Should dispatch normally — no prior observations, no
    // contradictions, not escalated.
    assert_eq!(result["status"], "dispatched");
    assert_eq!(result["escalated"], false);
    let counts = &result["observation_counts"];
    assert_eq!(counts["prior"], 0, "zero prior observations");

    reset();
}
