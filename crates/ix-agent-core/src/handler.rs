//! Handler trait + legacy adapter.
//!
//! [`AgentHandler`] is the unified contract for an action-shaped tool
//! call. [`LegacyAdapter`] wraps the existing
//! `fn(serde_json::Value) -> Result<serde_json::Value, String>` skill
//! surface so the 48 `#[ix_skill]`-annotated tools in ix-agent continue
//! to work without signature changes.

use serde_json::Value as JsonValue;

use crate::action::AgentAction;
use crate::context::ReadContext;
use crate::error::{ActionError, ActionResult};
use crate::event::ActionOutcome;

/// The contract for an agent-native handler.
///
/// Handlers receive an immutable `&ReadContext` (the projection of
/// session state at the current ordinal) and a `&AgentAction` (the
/// specific action to handle). They return an [`ActionResult`] whose
/// [`ActionOutcome::events`] field lets them append to the session log
/// without having mutable access to it.
///
/// **Object-safe by design.** The trait deliberately avoids generics
/// so that trait-object dispatch (`Box<dyn AgentHandler>`) works in
/// the registry. No `'static` bound: a dispatcher may pass a handler
/// that borrows its caller (e.g. a manual MCP tool closing over the
/// server context) as `&dyn AgentHandler`.
pub trait AgentHandler: Send + Sync {
    /// Execute the action against the current read context. MUST be
    /// pure in terms of `cx` — any state changes flow via
    /// `ActionOutcome::events`, never via interior mutability.
    fn run(&self, cx: &ReadContext, action: &AgentAction) -> ActionResult;
}

/// Adapter that wraps a `fn(Value) -> Result<Value, String>` legacy
/// skill as an [`AgentHandler`].
///
/// All 48 existing `#[ix_skill]`-annotated functions in ix-agent use
/// this shape. The adapter lets the new dispatch path accept them
/// without rewrites.
///
/// Handles only [`AgentAction::InvokeTool`] — other action variants
/// return [`ActionError::Exec`] because legacy skills have no concept
/// of observations, approvals, or returns.
#[derive(Debug, Clone, Copy)]
pub struct LegacyAdapter {
    /// The wrapped legacy function pointer.
    pub inner: fn(JsonValue) -> Result<JsonValue, String>,
}

impl LegacyAdapter {
    /// Wrap a legacy function.
    pub const fn new(inner: fn(JsonValue) -> Result<JsonValue, String>) -> Self {
        Self { inner }
    }
}

impl AgentHandler for LegacyAdapter {
    fn run(&self, _cx: &ReadContext, action: &AgentAction) -> ActionResult {
        match action {
            AgentAction::InvokeTool { params, .. } => {
                let value = (self.inner)(params.clone()).map_err(ActionError::Exec)?;
                Ok(ActionOutcome::value_only(value))
            }
            _ => Err(ActionError::Exec(
                "legacy adapter only supports InvokeTool actions".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // A no-op handler. If this compiles, `AgentHandler` is object-safe
    // (we box it as `Box<dyn AgentHandler>` below).
    struct EchoHandler;
    impl AgentHandler for EchoHandler {
        fn run(&self, _cx: &ReadContext, action: &AgentAction) -> ActionResult {
            match action {
                AgentAction::InvokeTool { params, .. } => {
                    Ok(ActionOutcome::value_only(params.clone()))
                }
                _ => Err(ActionError::Exec("echo only supports InvokeTool".into())),
            }
        }
    }

    #[test]
    fn agent_handler_is_object_safe() {
        let handler: Box<dyn AgentHandler> = Box::new(EchoHandler);
        let cx = ReadContext::synthetic_for_legacy();
        let action = AgentAction::InvokeTool {
            tool_name: "echo".into(),
            params: json!({"hi": "there"}),
            ordinal: 0,
            target_hint: None,
        };
        let outcome = handler.run(&cx, &action).expect("echo handler runs");
        assert_eq!(outcome.value, json!({"hi": "there"}));
    }

    // ── LegacyAdapter ──────────────────────────────────────────────────

    fn legacy_echo(v: JsonValue) -> Result<JsonValue, String> {
        Ok(v)
    }

    fn legacy_fail(_: JsonValue) -> Result<JsonValue, String> {
        Err("boom".into())
    }

    #[test]
    fn legacy_adapter_wraps_invoke_tool() {
        let adapter = LegacyAdapter::new(legacy_echo);
        let cx = ReadContext::synthetic_for_legacy();
        let action = AgentAction::InvokeTool {
            tool_name: "legacy_echo".into(),
            params: json!({"x": 1}),
            ordinal: 0,
            target_hint: None,
        };
        let outcome = adapter.run(&cx, &action).expect("ok");
        assert_eq!(outcome.value, json!({"x": 1}));
        assert!(outcome.events.is_empty());
    }

    #[test]
    fn legacy_adapter_propagates_string_error() {
        let adapter = LegacyAdapter::new(legacy_fail);
        let cx = ReadContext::synthetic_for_legacy();
        let action = AgentAction::InvokeTool {
            tool_name: "legacy_fail".into(),
            params: json!({}),
            ordinal: 0,
            target_hint: None,
        };
        match adapter.run(&cx, &action) {
            Err(ActionError::Exec(msg)) => assert_eq!(msg, "boom"),
            other => panic!("expected ActionError::Exec, got {other:?}"),
        }
    }

    #[test]
    fn legacy_adapter_rejects_non_invoke_actions() {
        let adapter = LegacyAdapter::new(legacy_echo);
        let cx = ReadContext::synthetic_for_legacy();
        let action = AgentAction::Return {
            payload: json!("done"),
            ordinal: 0,
        };
        match adapter.run(&cx, &action) {
            Err(ActionError::Exec(msg)) => assert!(msg.contains("InvokeTool")),
            other => panic!("expected Exec error, got {other:?}"),
        }
    }
}
