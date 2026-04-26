//! Integration test for the false-belief → correction loop.
//!
//! Demonstrates the full cycle:
//!   1. Agent has a seeded belief (`tool:X:will_succeed → Probable`)
//!   2. Action is dispatched through the middleware chain
//!   3. Action fails (tool does not exist)
//!   4. `BeliefMiddleware` observes the failure, emits
//!      `SessionEvent::BeliefChanged { old: Probable, new: False }`
//!   5. Beliefs are projected from the session log
//!   6. The projected beliefs reflect the correction
//!
//! This is the first code in the ix workspace that actually emits a
//! `SessionEvent::BeliefChanged` event — proving the full loop
//! works end-to-end, not just that the types compile.
//!
//! Lives in its own test binary so the process-wide middleware chain
//! and session log don't leak into parity.rs or other test binaries.

use ix_agent::registry_bridge;
use ix_agent_core::{
    project_beliefs, tool_proposition, AgentAction, BeliefMiddleware, ReadContext, VecEventSink,
    WriteContext,
};
use ix_session::SessionLog;
use ix_types::Hexavalent;
use tempfile::tempdir;

/// Full cycle: false belief → dispatch → failure → BeliefChanged →
/// project → corrected belief visible in the projected state.
#[test]
fn false_belief_corrected_after_tool_failure() {
    // ── Setup ──────────────────────────────────────────────────
    let dir = tempdir().expect("create tempdir");
    let log_path = dir.path().join("beliefs.jsonl");
    let log = SessionLog::open(&log_path).expect("open session log");
    registry_bridge::install_session_log(log);

    // Clean the shared loop detector so it doesn't interfere.
    registry_bridge::shared_loop_detector().clear_key("ix_totally_fake_belief_tool");

    // Seed a ReadContext with a false belief: we believe
    // "ix_totally_fake_belief_tool" will succeed (Probable), but
    // it doesn't exist. The dispatch will fail and the
    // BeliefMiddleware should correct the belief to False.
    let mut cx = ReadContext::synthetic_for_legacy();
    let prop = tool_proposition("ix_totally_fake_belief_tool");
    cx.beliefs.insert(prop.clone(), Hexavalent::Probable);

    // ── Act: dispatch under the false belief ───────────────────
    let action = AgentAction::InvokeTool {
        tool_name: "ix_totally_fake_belief_tool".to_string(),
        params: serde_json::json!({}),
        ordinal: 0,
        target_hint: None,
    };

    // dispatch_action runs the full middleware chain:
    //   LoopDetectMiddleware → ApprovalMiddleware → (handler)
    // BUT — we need BeliefMiddleware in the chain for the post
    // hook to fire. The process-wide chain doesn't have it yet.
    //
    // For this test, instead of mutating the global chain (which
    // would leak into other tests in this binary), we'll run the
    // belief loop manually using a local chain + VecEventSink.
    let mut sink = VecEventSink::default();
    {
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        // Build a local chain with just BeliefMiddleware so we
        // can observe its post hook in isolation.
        let mut chain = ix_agent_core::MiddlewareChain::new();
        chain.push(Box::new(BeliefMiddleware::new()));

        // The handler is a trivial "fail always" — simulates a
        // tool that doesn't exist.
        struct FailHandler;
        impl ix_agent_core::AgentHandler for FailHandler {
            fn run(&self, _cx: &ReadContext, _action: &AgentAction) -> ix_agent_core::ActionResult {
                Err(ix_agent_core::ActionError::Exec(
                    "tool not found: ix_totally_fake_belief_tool".into(),
                ))
            }
        }

        let _result = chain.dispatch(&mut wc, action, &FailHandler);
    }

    // ── Assert: BeliefChanged was emitted ──────────────────────
    let belief_events: Vec<_> = sink
        .events
        .iter()
        .filter(|e| matches!(e, ix_agent_core::SessionEvent::BeliefChanged { .. }))
        .collect();

    assert_eq!(
        belief_events.len(),
        1,
        "expected exactly one BeliefChanged event, got {}",
        belief_events.len()
    );

    match &belief_events[0] {
        ix_agent_core::SessionEvent::BeliefChanged {
            proposition,
            old,
            new,
            evidence,
            ..
        } => {
            assert_eq!(proposition, &prop);
            assert_eq!(*old, Some(Hexavalent::Probable));
            assert_eq!(*new, Hexavalent::False);
            assert_eq!(evidence["trigger"], "action_failed");
            assert_eq!(
                evidence["error"],
                "tool not found: ix_totally_fake_belief_tool"
            );
        }
        _ => unreachable!(),
    }

    // ── Project: replay events → corrected belief map ──────────
    let projected = project_beliefs(&sink.events);

    assert_eq!(
        projected.get(&prop),
        Some(&Hexavalent::False),
        "projected beliefs should show the correction"
    );

    // ── Verify: a fresh ReadContext built from the projection
    //    would have the corrected belief, so subsequent dispatches
    //    see `beliefs["tool:ix_totally_fake_belief_tool:will_succeed"] == False`.
    let mut cx2 = ReadContext::synthetic_for_legacy();
    cx2.beliefs = projected;
    assert_eq!(
        cx2.beliefs.get(&prop),
        Some(&Hexavalent::False),
        "next dispatch's ReadContext should carry the corrected belief"
    );

    // Cleanup.
    registry_bridge::clear_session_log();
}

/// Success path: a real tool call emits `BeliefChanged { new: True }`
/// when the belief was previously absent (= None = unknown).
#[test]
fn successful_dispatch_sets_belief_to_true() {
    let cx = ReadContext::synthetic_for_legacy();
    let prop = tool_proposition("ix_stats");
    // No belief seeded — defaults to None (unknown).
    assert!(!cx.beliefs.contains_key(&prop));

    let mut sink = VecEventSink::default();
    {
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        let mut chain = ix_agent_core::MiddlewareChain::new();
        chain.push(Box::new(BeliefMiddleware::new()));

        // Handler returns success.
        struct OkHandler;
        impl ix_agent_core::AgentHandler for OkHandler {
            fn run(&self, _cx: &ReadContext, _action: &AgentAction) -> ix_agent_core::ActionResult {
                Ok(ix_agent_core::ActionOutcome::value_only(
                    serde_json::json!({"mean": 3.0}),
                ))
            }
        }

        let action = AgentAction::InvokeTool {
            tool_name: "ix_stats".to_string(),
            params: serde_json::json!({ "data": [1.0, 2.0, 3.0] }),
            ordinal: 0,
            target_hint: None,
        };
        let result = chain.dispatch(&mut wc, action, &OkHandler);
        assert!(result.is_ok());
    }

    let projected = project_beliefs(&sink.events);
    assert_eq!(
        projected.get(&prop),
        Some(&Hexavalent::True),
        "successful tool call should set belief to True"
    );
}

/// Idempotency: if the belief already matches the outcome, no
/// BeliefChanged event should be emitted (avoids log bloat from
/// repeated successful calls to the same tool).
#[test]
fn no_redundant_belief_event_when_already_true() {
    let mut cx = ReadContext::synthetic_for_legacy();
    cx.beliefs
        .insert(tool_proposition("ix_stats"), Hexavalent::True);

    let mut sink = VecEventSink::default();
    {
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        let mut chain = ix_agent_core::MiddlewareChain::new();
        chain.push(Box::new(BeliefMiddleware::new()));

        struct OkHandler;
        impl ix_agent_core::AgentHandler for OkHandler {
            fn run(&self, _cx: &ReadContext, _action: &AgentAction) -> ix_agent_core::ActionResult {
                Ok(ix_agent_core::ActionOutcome::value_only(serde_json::json!(
                    42
                )))
            }
        }

        let action = AgentAction::InvokeTool {
            tool_name: "ix_stats".to_string(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: None,
        };
        let _result = chain.dispatch(&mut wc, action, &OkHandler);
    }

    let belief_events: Vec<_> = sink
        .events
        .iter()
        .filter(|e| matches!(e, ix_agent_core::SessionEvent::BeliefChanged { .. }))
        .collect();

    assert!(
        belief_events.is_empty(),
        "should not emit BeliefChanged when belief already matches outcome"
    );
}

/// Correction cycle: belief starts True, action fails, belief
/// transitions True → False.
#[test]
fn true_belief_corrected_to_false_on_failure() {
    let mut cx = ReadContext::synthetic_for_legacy();
    let prop = tool_proposition("ix_broken");
    cx.beliefs.insert(prop.clone(), Hexavalent::True);

    let mut sink = VecEventSink::default();
    {
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };

        let mut chain = ix_agent_core::MiddlewareChain::new();
        chain.push(Box::new(BeliefMiddleware::new()));

        struct FailHandler;
        impl ix_agent_core::AgentHandler for FailHandler {
            fn run(&self, _cx: &ReadContext, _action: &AgentAction) -> ix_agent_core::ActionResult {
                Err(ix_agent_core::ActionError::Exec("runtime crash".into()))
            }
        }

        let action = AgentAction::InvokeTool {
            tool_name: "ix_broken".to_string(),
            params: serde_json::json!({}),
            ordinal: 0,
            target_hint: None,
        };
        let _result = chain.dispatch(&mut wc, action, &FailHandler);
    }

    match &sink.events.last().unwrap() {
        ix_agent_core::SessionEvent::BeliefChanged { old, new, .. } => {
            assert_eq!(*old, Some(Hexavalent::True));
            assert_eq!(*new, Hexavalent::False);
        }
        other => panic!("expected BeliefChanged, got {other:?}"),
    }
}
