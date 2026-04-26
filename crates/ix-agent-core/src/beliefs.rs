//! [`BeliefMiddleware`] — observes action outcomes and emits
//! [`SessionEvent::BeliefChanged`] into the session log.
//!
//! This is the first **producer** of `BeliefChanged` events in the
//! harness. Before this module, the `SessionEvent::BeliefChanged`
//! variant existed (wire-compatible, serializable) but nothing in
//! the system emitted it. The belief state in
//! [`ReadContext::beliefs`] was always empty.
//!
//! ## How it works
//!
//! The middleware has no `pre` hook (it doesn't intercept actions).
//! Its `post` hook observes the handler result:
//!
//! - **`Ok(ActionOutcome)`** — the action succeeded. If the
//!   current belief for the tool was anything other than
//!   [`Hexavalent::True`], emit a `BeliefChanged` transitioning
//!   the tool's proposition to `True`.
//!
//! - **`Err(ActionError)`** — the action failed. If the current
//!   belief was anything other than [`Hexavalent::False`], emit a
//!   `BeliefChanged` transitioning to `False` with the error as
//!   evidence.
//!
//! ## Propositions
//!
//! The middleware keys beliefs by a convention:
//! `"tool:{tool_name}:will_succeed"`. This is purely a naming
//! convention — nothing in the system interprets proposition
//! strings semantically. The consumer of `ReadContext::beliefs`
//! is whoever reads the beliefs map and decides whether to
//! attempt a tool call, which is currently the agent/LLM itself
//! or a future planning middleware.
//!
//! ## Projection
//!
//! [`project_beliefs`] is a pure function that scans a sequence
//! of `SessionEvent`s and builds a `BTreeMap<String, Hexavalent>`
//! by replaying every `BeliefChanged` in order. This is the
//! "replay pure function" from the substrate design doc thesis:
//!
//! > Every `ReadContext` is `f(EventLog, ordinal) → ReadContext`.
//!
//! Callers can use this to build a fresh `ReadContext` whose
//! `beliefs` field reflects all corrections observed so far.

use std::collections::BTreeMap;

use ix_types::Hexavalent;

use crate::action::AgentAction;
use crate::context::WriteContext;
use crate::error::{ActionError, ActionResult};
use crate::event::{MiddlewareVerdict, SessionEvent};
use crate::middleware::Middleware;

/// Middleware that observes action outcomes and emits
/// [`SessionEvent::BeliefChanged`] to track tool reliability.
///
/// Stateless — all state lives in the session event log. The
/// middleware reads the current belief from
/// [`WriteContext::read::beliefs`] and compares it against the
/// observed outcome. If they disagree, a `BeliefChanged` event
/// is emitted to record the correction.
pub struct BeliefMiddleware;

impl BeliefMiddleware {
    /// Construct the middleware. No configuration — the
    /// proposition naming convention and the
    /// success/failure→True/False mapping are hardcoded in the
    /// `post` hook.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BeliefMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// Build the proposition key for a tool's "will succeed" belief.
///
/// Convention: `"tool:{tool_name}:will_succeed"`. This is a
/// stable naming convention that consumers can reconstruct
/// from a tool name without importing this module.
pub fn tool_proposition(tool_name: &str) -> String {
    format!("tool:{tool_name}:will_succeed")
}

impl Middleware for BeliefMiddleware {
    fn name(&self) -> &str {
        "ix_beliefs"
    }

    fn pre(&self, _cx: &mut WriteContext<'_>, _action: &AgentAction) -> MiddlewareVerdict {
        MiddlewareVerdict::Continue
    }

    fn post(&self, cx: &mut WriteContext<'_>, action: &AgentAction, result: &ActionResult) {
        // Only InvokeTool actions generate tool-availability beliefs.
        let tool_name = match action {
            AgentAction::InvokeTool { tool_name, .. } => tool_name,
            _ => return,
        };

        let proposition = tool_proposition(tool_name);
        let current_belief = cx.read.beliefs.get(&proposition).copied();

        match result {
            Ok(_) => {
                // Action succeeded. If the current belief was NOT
                // already True, record the correction.
                if current_belief != Some(Hexavalent::True) {
                    cx.sink.emit(SessionEvent::BeliefChanged {
                        ordinal: cx.sink.next_ordinal(),
                        proposition,
                        old: current_belief,
                        new: Hexavalent::True,
                        evidence: serde_json::json!({
                            "source": "ix_beliefs",
                            "trigger": "action_completed",
                            "tool": tool_name,
                        }),
                    });
                }
            }
            Err(e) => {
                // Action failed. If the current belief was NOT
                // already False, record the correction.
                //
                // We do NOT distinguish between Blocked (approval
                // denied, loop detected) and Exec (tool not found,
                // runtime error). Both mean "it didn't work."
                // A richer version could map Blocked → Doubtful
                // (the tool might succeed if the block reason is
                // resolved) vs Exec → False (the tool is genuinely
                // broken). For the MVP, False is correct enough.
                if current_belief != Some(Hexavalent::False) {
                    let error_evidence = match e {
                        ActionError::Blocked {
                            code,
                            reason,
                            blocker,
                        } => {
                            serde_json::json!({
                                "source": "ix_beliefs",
                                "trigger": "action_blocked",
                                "tool": tool_name,
                                "code": format!("{code:?}"),
                                "reason": reason,
                                "blocker": blocker,
                            })
                        }
                        ActionError::Exec(msg) => {
                            serde_json::json!({
                                "source": "ix_beliefs",
                                "trigger": "action_failed",
                                "tool": tool_name,
                                "error": msg,
                            })
                        }
                        ActionError::InvalidResult(msg) => {
                            serde_json::json!({
                                "source": "ix_beliefs",
                                "trigger": "invalid_result",
                                "tool": tool_name,
                                "error": msg,
                            })
                        }
                    };
                    cx.sink.emit(SessionEvent::BeliefChanged {
                        ordinal: cx.sink.next_ordinal(),
                        proposition,
                        old: current_belief,
                        new: Hexavalent::False,
                        evidence: error_evidence,
                    });
                }
            }
        }
    }
}

/// Replay a sequence of [`SessionEvent`]s and build a belief
/// state by applying every [`SessionEvent::BeliefChanged`] in
/// order.
///
/// This is a pure function: same events in → same beliefs out,
/// across processes, because `BTreeMap` iteration is
/// deterministic. Matches the substrate design doc thesis:
///
/// > Every `ReadContext` is `f(EventLog, ordinal) → ReadContext`.
///
/// Consumers use this to build a fresh `ReadContext::beliefs`
/// from a session log (either in-memory `VecEventSink::events`
/// or on-disk `SessionLog::events()`).
pub fn project_beliefs<'a>(
    events: impl IntoIterator<Item = &'a SessionEvent>,
) -> BTreeMap<String, Hexavalent> {
    let mut beliefs = BTreeMap::new();
    for event in events {
        if let SessionEvent::BeliefChanged {
            proposition, new, ..
        } = event
        {
            beliefs.insert(proposition.clone(), *new);
        }
    }
    beliefs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{ReadContext, VecEventSink};
    use serde_json::json;

    fn invoke(tool: &str) -> AgentAction {
        AgentAction::InvokeTool {
            tool_name: tool.to_string(),
            params: json!({}),
            ordinal: 0,
            target_hint: None,
        }
    }

    #[test]
    fn post_emits_belief_changed_on_success() {
        let mw = BeliefMiddleware::new();
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };
        let action = invoke("ix_stats");
        let result: ActionResult = Ok(crate::event::ActionOutcome::value_only(json!(42)));

        mw.post(&mut wc, &action, &result);

        assert_eq!(sink.events.len(), 1);
        match &sink.events[0] {
            SessionEvent::BeliefChanged {
                proposition,
                old,
                new,
                ..
            } => {
                assert_eq!(proposition, "tool:ix_stats:will_succeed");
                assert_eq!(*old, None);
                assert_eq!(*new, Hexavalent::True);
            }
            other => panic!("expected BeliefChanged, got {other:?}"),
        }
    }

    #[test]
    fn post_emits_belief_changed_on_failure() {
        let mw = BeliefMiddleware::new();
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };
        let action = invoke("ix_nonexistent");
        let result: ActionResult = Err(ActionError::Exec("tool not found".into()));

        mw.post(&mut wc, &action, &result);

        assert_eq!(sink.events.len(), 1);
        match &sink.events[0] {
            SessionEvent::BeliefChanged {
                proposition,
                old,
                new,
                evidence,
                ..
            } => {
                assert_eq!(proposition, "tool:ix_nonexistent:will_succeed");
                assert_eq!(*old, None);
                assert_eq!(*new, Hexavalent::False);
                assert_eq!(evidence["trigger"], "action_failed");
                assert_eq!(evidence["error"], "tool not found");
            }
            other => panic!("expected BeliefChanged, got {other:?}"),
        }
    }

    #[test]
    fn post_is_no_op_when_belief_already_matches() {
        let mw = BeliefMiddleware::new();

        // Seed beliefs with True for ix_stats — success should NOT
        // emit a redundant BeliefChanged.
        let mut cx = ReadContext::synthetic_for_legacy();
        cx.beliefs
            .insert("tool:ix_stats:will_succeed".into(), Hexavalent::True);

        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };
        let action = invoke("ix_stats");
        let result: ActionResult = Ok(crate::event::ActionOutcome::value_only(json!(42)));

        mw.post(&mut wc, &action, &result);
        assert!(
            sink.events.is_empty(),
            "should not emit when belief matches"
        );
    }

    #[test]
    fn post_ignores_non_invoke_actions() {
        let mw = BeliefMiddleware::new();
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        let mut wc = WriteContext {
            read: &cx,
            sink: &mut sink,
        };
        let action = AgentAction::Return {
            ordinal: 0,
            payload: json!(null),
        };
        let result: ActionResult = Ok(crate::event::ActionOutcome::value_only(json!(null)));

        mw.post(&mut wc, &action, &result);
        assert!(sink.events.is_empty());
    }

    #[test]
    fn project_beliefs_replays_belief_changed_events() {
        let events = vec![
            SessionEvent::BeliefChanged {
                ordinal: 0,
                proposition: "tool:a:will_succeed".into(),
                old: None,
                new: Hexavalent::True,
                evidence: json!({}),
            },
            // Some non-belief event interleaved.
            SessionEvent::ActionCompleted {
                ordinal: 1,
                value: json!(42),
            },
            // Correction: a → False.
            SessionEvent::BeliefChanged {
                ordinal: 2,
                proposition: "tool:a:will_succeed".into(),
                old: Some(Hexavalent::True),
                new: Hexavalent::False,
                evidence: json!({"reason": "tool removed"}),
            },
            // Different proposition.
            SessionEvent::BeliefChanged {
                ordinal: 3,
                proposition: "tool:b:will_succeed".into(),
                old: None,
                new: Hexavalent::Probable,
                evidence: json!({}),
            },
        ];

        let beliefs = project_beliefs(&events);

        // 'a' was True then corrected to False — last write wins.
        assert_eq!(beliefs["tool:a:will_succeed"], Hexavalent::False);
        // 'b' was set once to Probable.
        assert_eq!(beliefs["tool:b:will_succeed"], Hexavalent::Probable);
        // Nothing else in the map.
        assert_eq!(beliefs.len(), 2);
    }

    #[test]
    fn project_beliefs_deterministic_across_insert_order() {
        // Two identical event lists inserted in different temporal
        // order must produce bit-identical BTreeMap serialization.
        let events_a = vec![
            SessionEvent::BeliefChanged {
                ordinal: 0,
                proposition: "z:prop".into(),
                old: None,
                new: Hexavalent::True,
                evidence: json!({}),
            },
            SessionEvent::BeliefChanged {
                ordinal: 1,
                proposition: "a:prop".into(),
                old: None,
                new: Hexavalent::False,
                evidence: json!({}),
            },
        ];
        let events_b = vec![
            SessionEvent::BeliefChanged {
                ordinal: 0,
                proposition: "a:prop".into(),
                old: None,
                new: Hexavalent::False,
                evidence: json!({}),
            },
            SessionEvent::BeliefChanged {
                ordinal: 1,
                proposition: "z:prop".into(),
                old: None,
                new: Hexavalent::True,
                evidence: json!({}),
            },
        ];
        let a = project_beliefs(&events_a);
        let b = project_beliefs(&events_b);
        assert_eq!(
            serde_json::to_string(&a).unwrap(),
            serde_json::to_string(&b).unwrap(),
        );
    }
}
