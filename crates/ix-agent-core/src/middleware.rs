//! Middleware trait + MiddlewareChain fold.
//!
//! This is the execution model that lets ix-loop-detect, ix-approval,
//! and future harness primitives layer around the tool dispatcher
//! without each one reaching into `registry_bridge` directly.
//!
//! # Execution shape
//!
//! Middleware runs as a linear chain. For each action:
//!
//! 1. The chain iterates middlewares in order, calling `pre(cx, action)`.
//! 2. Each middleware returns a [`MiddlewareVerdict`]:
//!    - [`MiddlewareVerdict::Continue`] → proceed to the next middleware.
//!    - [`MiddlewareVerdict::Block`] → stop the chain; return
//!      [`crate::error::ActionError::Blocked`].
//!    - [`MiddlewareVerdict::Replace`] → swap the pending action and
//!      restart the chain from the start so the new action is visible
//!      to earlier middlewares.
//! 3. If every middleware returns `Continue`, the chain calls the final
//!    handler with the (possibly replaced) action.
//!
//! # No algebraic composition
//!
//! Middleware chains compose by **sequence**, not by algebraic OR/AND
//! over verdicts. This is a deliberate choice — the `Hexavalent::or`
//! bug (see `docs/brainstorms/2026-04-10-hexavalent-or-discrepancy.md`)
//! makes algebraic composition unsafe, and sequence-based chains are
//! easier to audit for determinism.

use crate::action::AgentAction;
use crate::context::WriteContext;
use crate::error::{ActionError, ActionResult};
use crate::event::{MiddlewareVerdict, SessionEvent};
use crate::handler::AgentHandler;

/// A hook that inspects or rewrites an action before the handler runs.
///
/// Middleware implementations should be stateless or hold their state
/// behind an internal `Mutex` — the chain executes middlewares in
/// sequence without exclusive borrows, so any interior mutability must
/// handle concurrent access itself.
pub trait Middleware: Send + Sync + 'static {
    /// A short identifier used for event attribution — e.g.,
    /// `"ix_loop_detect"`, `"ix_approval"`. Appears in the
    /// `emitted_by` field of `ActionBlocked` and `ActionReplaced`
    /// session events.
    fn name(&self) -> &str;

    /// Inspect the current action and produce a verdict. MAY emit
    /// events via `cx.sink` — these are appended to the session log
    /// regardless of the verdict.
    fn pre(&self, cx: &mut WriteContext<'_>, action: &AgentAction) -> MiddlewareVerdict;

    /// Observe the outcome of an action AFTER the handler has run
    /// (or after a Block prevented the handler from running). The
    /// `post` hook cannot change the result — it can only emit
    /// events into the session log via `cx.sink`.
    ///
    /// Primary consumer: [`crate::beliefs::BeliefMiddleware`],
    /// which emits [`SessionEvent::BeliefChanged`] when an action
    /// succeeds or fails, updating the agent's belief state about
    /// tool availability or parameter correctness.
    ///
    /// Default implementation is a no-op so existing middleware
    /// impls (LoopDetectMiddleware, ApprovalMiddleware) don't need
    /// to change.
    fn post(&self, _cx: &mut WriteContext<'_>, _action: &AgentAction, _result: &ActionResult) {
        // no-op by default
    }
}

/// A linear chain of middlewares that wraps a terminal [`AgentHandler`].
///
/// The chain is built once at dispatcher initialization and reused for
/// every action. It borrows middlewares as `Box<dyn Middleware>` so
/// consumers can mix and match middleware crates without generic
/// parameter explosion.
pub struct MiddlewareChain {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewareChain {
    /// Construct an empty chain. Middlewares are added via
    /// [`MiddlewareChain::push`].
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Append a middleware to the end of the chain. The chain executes
    /// middlewares in the order they were pushed.
    pub fn push(&mut self, mw: Box<dyn Middleware>) -> &mut Self {
        self.middlewares.push(mw);
        self
    }

    /// How many middlewares are in the chain.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// `true` iff the chain has no middlewares.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Execute the chain against an action, finally calling `handler`
    /// if every middleware returned `Continue`.
    ///
    /// The dispatcher owns the [`WriteContext`] and is responsible for
    /// draining any events middlewares emitted into the session log.
    pub fn dispatch(
        &self,
        cx: &mut WriteContext<'_>,
        action: AgentAction,
        handler: &dyn AgentHandler,
    ) -> ActionResult {
        let mut current = action;

        // Outer loop: if a middleware replaces the action, we restart
        // the chain from the beginning so earlier middlewares see the
        // new action. Bounded by `max_replay` to prevent accidental
        // infinite replacement loops.
        let max_replay: usize = 16;
        let mut replay_count: usize = 0;

        'outer: loop {
            for mw in &self.middlewares {
                let verdict = mw.pre(cx, &current);
                match verdict {
                    MiddlewareVerdict::Continue => continue,
                    MiddlewareVerdict::Block { code, reason } => {
                        // Dispatcher emits the ActionBlocked event.
                        cx.sink.emit(SessionEvent::ActionBlocked {
                            ordinal: current.ordinal(),
                            code,
                            reason: reason.clone(),
                            emitted_by: mw.name().to_string(),
                        });
                        return Err(ActionError::Blocked {
                            code,
                            reason,
                            blocker: mw.name().to_string(),
                        });
                    }
                    MiddlewareVerdict::Replace(new_action) => {
                        cx.sink.emit(SessionEvent::ActionReplaced {
                            ordinal: current.ordinal(),
                            original: current.clone(),
                            replacement: new_action.clone(),
                            emitted_by: mw.name().to_string(),
                        });
                        current = new_action;
                        replay_count += 1;
                        if replay_count > max_replay {
                            return Err(ActionError::Exec(format!(
                                "middleware chain exceeded {max_replay} action replacements — possible rewrite loop"
                            )));
                        }
                        continue 'outer;
                    }
                }
            }
            break 'outer;
        }

        // Every middleware returned Continue. Invoke the handler.
        let result = handler.run(cx.read, &current);

        // Post-dispatch observation pass — gives every middleware a
        // chance to observe the outcome and emit events. The result
        // is immutable at this point; post hooks can only observe
        // and emit, not modify. Runs in the same order as pre so
        // middleware authors can pair pre/post logic by position.
        for mw in &self.middlewares {
            mw.post(cx, &current, &result);
        }

        result
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{ReadContext, VecEventSink};
    use crate::event::ActionOutcome;
    use serde_json::json;

    // ── Test fixtures ──────────────────────────────────────────────────

    struct AllowAllMiddleware;
    impl Middleware for AllowAllMiddleware {
        fn name(&self) -> &str {
            "allow_all"
        }
        fn pre(&self, _cx: &mut WriteContext<'_>, _action: &AgentAction) -> MiddlewareVerdict {
            MiddlewareVerdict::Continue
        }
    }

    struct BlockingMiddleware {
        code: crate::event::BlockCode,
        reason: String,
    }
    impl Middleware for BlockingMiddleware {
        fn name(&self) -> &str {
            "blocker"
        }
        fn pre(&self, _cx: &mut WriteContext<'_>, _action: &AgentAction) -> MiddlewareVerdict {
            MiddlewareVerdict::Block {
                code: self.code,
                reason: self.reason.clone(),
            }
        }
    }

    struct ReplaceOnceMiddleware {
        replacement: AgentAction,
        fired: std::sync::Mutex<bool>,
    }
    impl Middleware for ReplaceOnceMiddleware {
        fn name(&self) -> &str {
            "replacer"
        }
        fn pre(&self, _cx: &mut WriteContext<'_>, _action: &AgentAction) -> MiddlewareVerdict {
            let mut guard = self.fired.lock().unwrap();
            if *guard {
                MiddlewareVerdict::Continue
            } else {
                *guard = true;
                MiddlewareVerdict::Replace(self.replacement.clone())
            }
        }
    }

    struct EchoHandler;
    impl AgentHandler for EchoHandler {
        fn run(&self, _cx: &ReadContext, action: &AgentAction) -> ActionResult {
            match action {
                AgentAction::InvokeTool { params, .. } => {
                    Ok(ActionOutcome::value_only(params.clone()))
                }
                AgentAction::Return { payload, .. } => {
                    Ok(ActionOutcome::value_only(payload.clone()))
                }
                _ => Err(ActionError::Exec("unsupported".into())),
            }
        }
    }

    fn invoke(tool: &str) -> AgentAction {
        AgentAction::InvokeTool {
            tool_name: tool.to_string(),
            params: json!({"val": 42}),
            ordinal: 0,
            target_hint: None,
        }
    }

    fn run_chain(chain: &MiddlewareChain, action: AgentAction) -> (ActionResult, VecEventSink) {
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        let result = {
            let mut wc = WriteContext {
                read: &cx,
                sink: &mut sink,
            };
            chain.dispatch(&mut wc, action, &EchoHandler)
        };
        (result, sink)
    }

    // ── Chain basics ───────────────────────────────────────────────────

    #[test]
    fn empty_chain_invokes_handler_directly() {
        let chain = MiddlewareChain::new();
        assert!(chain.is_empty());
        let (result, sink) = run_chain(&chain, invoke("echo"));
        let outcome = result.expect("handler runs");
        assert_eq!(outcome.value, json!({"val": 42}));
        assert!(sink.events.is_empty());
    }

    #[test]
    fn chain_with_allow_all_passes_through() {
        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(AllowAllMiddleware));
        chain.push(Box::new(AllowAllMiddleware));
        assert_eq!(chain.len(), 2);

        let (result, sink) = run_chain(&chain, invoke("echo"));
        assert!(result.is_ok());
        assert!(sink.events.is_empty());
    }

    // ── Blocking ───────────────────────────────────────────────────────

    #[test]
    fn blocking_middleware_returns_error_and_logs_event() {
        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(BlockingMiddleware {
            code: crate::event::BlockCode::LoopDetected,
            reason: "too many".into(),
        }));

        let (result, sink) = run_chain(&chain, invoke("echo"));
        match result {
            Err(ActionError::Blocked { code, blocker, .. }) => {
                assert_eq!(code, crate::event::BlockCode::LoopDetected);
                assert_eq!(blocker, "blocker");
            }
            other => panic!("expected Blocked, got {other:?}"),
        }

        // One ActionBlocked event should be logged.
        assert_eq!(sink.events.len(), 1);
        assert!(matches!(
            &sink.events[0],
            SessionEvent::ActionBlocked { code, emitted_by, .. }
                if *code == crate::event::BlockCode::LoopDetected && emitted_by == "blocker"
        ));
    }

    #[test]
    fn block_stops_chain_before_downstream_middlewares() {
        // If the first middleware blocks, the second should NEVER run.
        // Use a middleware that would panic if invoked to prove it.
        struct PanicIfCalled;
        impl Middleware for PanicIfCalled {
            fn name(&self) -> &str {
                "panicker"
            }
            fn pre(&self, _: &mut WriteContext<'_>, _: &AgentAction) -> MiddlewareVerdict {
                panic!("downstream middleware must not run after a block");
            }
        }

        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(BlockingMiddleware {
            code: crate::event::BlockCode::PolicyDenied,
            reason: "denied".into(),
        }));
        chain.push(Box::new(PanicIfCalled));

        let (result, _) = run_chain(&chain, invoke("echo"));
        assert!(matches!(result, Err(ActionError::Blocked { .. })));
    }

    // ── Replacement ────────────────────────────────────────────────────

    #[test]
    fn replace_middleware_swaps_action_and_continues() {
        let replacement = AgentAction::Return {
            payload: json!("overridden"),
            ordinal: 0,
        };
        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(ReplaceOnceMiddleware {
            replacement: replacement.clone(),
            fired: std::sync::Mutex::new(false),
        }));

        let (result, sink) = run_chain(&chain, invoke("echo"));
        let outcome = result.expect("replaced action runs");
        assert_eq!(outcome.value, json!("overridden"));

        // One ActionReplaced event should be logged.
        assert_eq!(sink.events.len(), 1);
        assert!(matches!(
            &sink.events[0],
            SessionEvent::ActionReplaced { emitted_by, .. } if emitted_by == "replacer"
        ));
    }

    #[test]
    fn replacement_loop_aborts_after_max_replay() {
        // Middleware that always replaces with a fresh action — would
        // loop forever without the max_replay guard.
        struct AlwaysReplace;
        impl Middleware for AlwaysReplace {
            fn name(&self) -> &str {
                "always_replace"
            }
            fn pre(&self, _: &mut WriteContext<'_>, action: &AgentAction) -> MiddlewareVerdict {
                // Bump the ordinal so each replacement is distinct.
                let next = match action {
                    AgentAction::InvokeTool {
                        tool_name,
                        params,
                        ordinal,
                        target_hint,
                    } => AgentAction::InvokeTool {
                        tool_name: tool_name.clone(),
                        params: params.clone(),
                        ordinal: ordinal + 1,
                        target_hint: target_hint.clone(),
                    },
                    other => other.clone(),
                };
                MiddlewareVerdict::Replace(next)
            }
        }

        let mut chain = MiddlewareChain::new();
        chain.push(Box::new(AlwaysReplace));

        let (result, _) = run_chain(&chain, invoke("echo"));
        match result {
            Err(ActionError::Exec(msg)) => {
                assert!(
                    msg.contains("rewrite loop"),
                    "expected rewrite-loop abort, got: {msg}"
                );
            }
            other => panic!("expected Exec error, got {other:?}"),
        }
    }
}
