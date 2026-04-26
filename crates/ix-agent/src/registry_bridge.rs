//! Bridge between `ix-registry`'s capability registry and the MCP `Tool`
//! surface exposed by `ix-agent`.
//!
//! Skills registered via `#[ix_skill]` are adapted to MCP tools by:
//!   1. Mapping the dotted skill name (`supervised.linear_regression.fit`) to
//!      an underscore MCP name (`ix_supervised_linear_regression_fit`).
//!   2. Using the skill's hand-written `json_schema` (via `schema_fn = ...`)
//!      as the MCP input-schema.
//!   3. Wrapping the handler so the MCP caller passes one JSON blob in and
//!      gets one JSON blob out.
//!
//! For composite MCP handlers we annotate a single-arg wrapper fn of shape
//! `fn(params: serde_json::Value) -> Result<serde_json::Value, String>` with
//! `#[ix_skill]`. The registry carries a single `Json → Json` socket pair,
//! but the MCP schema returned to clients is the original hand-written one.

use crate::tools::Tool;
use ix_agent_core::event::BlockCode;
use ix_agent_core::{
    ActionError, ActionOutcome, ActionResult, AgentAction, AgentHandler, BeliefMiddleware,
    EventSink, MiddlewareChain, ReadContext, VecEventSink, WriteContext,
};
use ix_approval::ApprovalMiddleware;
use ix_loop_detect::{LoopDetectMiddleware, LoopDetector};
use ix_registry::SkillDescriptor;
use ix_session::SessionLog;
use ix_types::Value as IxValue;
use serde_json::Value as JsonValue;
use std::sync::{Arc, Mutex, OnceLock};

/// Process-wide loop detector for MCP tool dispatch.
///
/// Shared by **both** the legacy [`dispatch`] path (which records
/// inline) and the new [`dispatch_action`] path (via
/// [`LoopDetectMiddleware`]) so repeated calls through either API
/// contribute to one unified sliding window. Without this sharing,
/// an agent could sidestep the circuit breaker by bouncing between
/// the two dispatch entry points.
///
/// Uses the brainstorm's default configuration (10 calls / 5 minutes
/// per tool) and is lazily initialized.
fn loop_detector() -> Arc<LoopDetector> {
    static DETECTOR: OnceLock<Arc<LoopDetector>> = OnceLock::new();
    Arc::clone(DETECTOR.get_or_init(|| Arc::new(LoopDetector::with_defaults())))
}

/// Expose a handle to the shared detector so the `ix-mcp` binary (or tests)
/// can reset state between sessions. Normal skill callers should not touch
/// the detector directly — `dispatch` manages it.
pub fn shared_loop_detector() -> Arc<LoopDetector> {
    loop_detector()
}

// ---------------------------------------------------------------------------
// Middleware chain — the new dispatch_action path
// ---------------------------------------------------------------------------

/// Process-wide middleware chain used by [`dispatch_action`]. Lazy-
/// initialized with a default chain containing
/// [`ApprovalMiddleware`] with its default config.
///
/// The chain is wrapped in a `Mutex` so tests can push additional
/// middlewares or replace the chain entirely. Production callers
/// should treat it as read-only.
fn middleware_chain() -> &'static Mutex<MiddlewareChain> {
    static CHAIN: OnceLock<Mutex<MiddlewareChain>> = OnceLock::new();
    CHAIN.get_or_init(|| {
        let mut chain = MiddlewareChain::new();
        // Loop detection runs first — a runaway agent should be
        // short-circuited before any classification or handler work.
        // Shares the process-wide detector with the legacy `dispatch`
        // path so both entry points count against one sliding window.
        chain.push(Box::new(LoopDetectMiddleware::from_shared(loop_detector())));
        chain.push(Box::new(ApprovalMiddleware::with_defaults()));
        // Belief revision runs last — observes action outcomes via
        // the post hook and emits BeliefChanged events into the
        // session log so the agent's beliefs track real tool
        // reliability. Must be after approval (which may block) so
        // the belief update reflects the actual outcome, not a
        // pre-flight classification.
        chain.push(Box::new(BeliefMiddleware::new()));
        Mutex::new(chain)
    })
}

/// Expose a handle to the shared middleware chain so tests can push
/// additional middlewares or reset the chain between cases. Production
/// callers should not mutate the chain at runtime.
pub fn shared_middleware_chain() -> &'static Mutex<MiddlewareChain> {
    middleware_chain()
}

// ---------------------------------------------------------------------------
// Optional persistent session log for dispatch_action
// ---------------------------------------------------------------------------

/// Process-wide session log slot used by [`dispatch_action`].
///
/// When populated, every dispatched action routes its emitted
/// [`ix_agent_core::SessionEvent`]s through an [`ix_session::SessionSink`]
/// so middleware verdicts and handler outcomes survive the process.
/// When empty, dispatch falls back to an in-memory [`VecEventSink`] and
/// events are discarded after the call — preserving the existing test
/// behavior.
///
/// Bootstrap order:
///   1. On first access, read the `IX_SESSION_LOG` environment variable.
///      If set and the file can be opened, install that log.
///   2. Otherwise start empty; callers (typically the `ix-mcp` main or
///      tests) can explicitly install a log via [`install_session_log`].
fn session_log_slot() -> &'static Mutex<Option<Arc<SessionLog>>> {
    static SLOT: OnceLock<Mutex<Option<Arc<SessionLog>>>> = OnceLock::new();
    SLOT.get_or_init(|| {
        let initial = std::env::var("IX_SESSION_LOG")
            .ok()
            .and_then(|path| SessionLog::open(&path).ok())
            .map(Arc::new);
        Mutex::new(initial)
    })
}

/// Install a [`SessionLog`] as the process-wide event sink target for
/// [`dispatch_action`]. Replaces any previously installed log.
///
/// Typical callers:
/// - `ix-mcp` main, after parsing config, to route production events
///   to a durable JSONL file.
/// - Integration tests that want to assert events survive dispatch.
///
/// Wrapping in an [`Arc`] lets dispatches clone a handle out of the
/// slot without holding its mutex across the handler call.
pub fn install_session_log(log: SessionLog) {
    let mut slot = session_log_slot()
        .lock()
        .expect("session log mutex poisoned");
    *slot = Some(Arc::new(log));
}

/// Clear any installed session log, reverting [`dispatch_action`] to
/// its in-memory [`VecEventSink`] fallback. Primarily useful for tests
/// that need to tear down state between cases.
pub fn clear_session_log() {
    let mut slot = session_log_slot()
        .lock()
        .expect("session log mutex poisoned");
    *slot = None;
}

/// Snapshot the currently installed log (if any) as a cloned [`Arc`].
/// Used internally by [`dispatch_action`] to release the slot mutex
/// before running the handler, and publicly by tools (e.g.
/// `ix_triage_session`) that need to *read* the session history of
/// the current run to make decisions.
///
/// Cloning the `Arc` is cheap and releases the slot mutex immediately,
/// so callers can iterate `events()` on the returned handle without
/// blocking concurrent dispatches.
pub fn current_session_log() -> Option<Arc<SessionLog>> {
    session_log_slot()
        .lock()
        .expect("session log mutex poisoned")
        .as_ref()
        .map(Arc::clone)
}

/// Terminal handler that looks up the tool by name in the registry at
/// run time. Acts as the "rightmost" node of the middleware chain in
/// [`dispatch_action`] — after every middleware has returned Continue,
/// this handler performs the actual registry lookup + skill invocation.
///
/// Unlike [`ix_agent_core::LegacyAdapter`], which wraps a single
/// known function pointer, this handler defers the lookup to invocation
/// time because the dispatcher only knows the tool name, not the
/// specific `fn_ptr`.
struct RegistryLookupHandler;

impl AgentHandler for RegistryLookupHandler {
    fn run(&self, _cx: &ReadContext, action: &AgentAction) -> ActionResult {
        match action {
            AgentAction::InvokeTool {
                tool_name, params, ..
            } => {
                let mcp_name = tool_name.as_str();
                let skill_name = mcp_to_skill_name(mcp_name).ok_or_else(|| {
                    ActionError::Exec(format!("not a registry-backed tool: {mcp_name}"))
                })?;
                let descriptor = ix_registry::by_name(&skill_name).ok_or_else(|| {
                    ActionError::Exec(format!("skill not in registry: {skill_name}"))
                })?;

                let args = [IxValue::Json(params.clone())];
                let out =
                    (descriptor.fn_ptr)(&args).map_err(|e| ActionError::Exec(e.to_string()))?;
                match out {
                    IxValue::Json(j) => Ok(ActionOutcome::value_only(j)),
                    other => {
                        let value = serde_json::to_value(other)
                            .map_err(|e| ActionError::InvalidResult(e.to_string()))?;
                        Ok(ActionOutcome::value_only(value))
                    }
                }
            }
            _ => Err(ActionError::Exec(
                "RegistryLookupHandler only supports InvokeTool actions".into(),
            )),
        }
    }
}

/// New action-shaped dispatch path that runs the full middleware chain
/// before looking up and invoking the terminal handler.
///
/// This is additive relative to [`dispatch`] — the legacy path stays
/// unchanged for the 48 existing tool callers. Consumers that want
/// the full governance-instrument contract (middleware verdicts,
/// session event emission, action-level typing) call this function
/// instead.
///
/// Returns the handler's output value on success, or an `ActionError`
/// on any failure from the middleware chain or the handler.
pub fn dispatch_action(cx: &ReadContext, action: AgentAction) -> ActionResult {
    // Grab an `Arc` handle to the installed log (if any) before
    // locking the chain. The session log has its own internal mutex,
    // so holding the `Arc` alone does not block concurrent dispatches.
    let log = current_session_log();
    let mut session_sink = log.as_ref().map(|l| l.sink());
    let mut vec_sink = VecEventSink::default();

    // Narrow to a `&mut dyn EventSink` so WriteContext sees one
    // uniform type regardless of which backing sink is active.
    let sink: &mut dyn EventSink = match &mut session_sink {
        Some(s) => s,
        None => &mut vec_sink,
    };

    let chain_guard = middleware_chain()
        .lock()
        .expect("middleware chain mutex poisoned");

    let result = {
        let mut wc = WriteContext { read: cx, sink };
        chain_guard.dispatch(&mut wc, action, &RegistryLookupHandler)
    };

    drop(chain_guard);
    drop(session_sink);
    drop(vec_sink);

    result
}

/// Translate a dotted registry name into the MCP tool name convention.
///
/// `supervised.linear_regression.fit` → `ix_supervised_linear_regression_fit`
pub fn mcp_name(skill_name: &str) -> String {
    format!("ix_{}", skill_name.replace('.', "_"))
}

/// Dispatch a registered skill by MCP name — legacy JSON-in/JSON-out
/// entry point. Internally routes through [`dispatch_action`] so the
/// full middleware chain (loop detection + approval, plus any future
/// middlewares) fires uniformly across both entry points.
///
/// Error shape is preserved so existing callers (MCP JSON-RPC
/// handlers, tests that match on the circuit-breaker string) keep
/// working. Specifically, a `Blocked(LoopDetected)` verdict is
/// translated back into the historical
/// `"ix_loop_detect: circuit breaker tripped on tool '…'"` string
/// that runaway-agent detectors downstream still match against.
pub fn dispatch(mcp_tool_name: &str, params: JsonValue) -> Result<JsonValue, String> {
    let cx = ReadContext::synthetic_for_legacy();
    let action = AgentAction::InvokeTool {
        tool_name: mcp_tool_name.to_string(),
        params,
        ordinal: 0,
        target_hint: None,
    };

    match dispatch_action(&cx, action) {
        Ok(outcome) => Ok(outcome.value),
        Err(ActionError::Blocked {
            code: BlockCode::LoopDetected,
            reason,
            ..
        }) => Err(format!(
            "ix_loop_detect: circuit breaker tripped on tool '{mcp_tool_name}' — {reason}. \
             The agent should stop calling this tool and reconsider its approach. \
             This is a governance-instrument safety check, not a transient error."
        )),
        Err(ActionError::Blocked {
            code,
            reason,
            blocker,
        }) => Err(format!("{blocker}: action blocked ({code:?}) — {reason}")),
        Err(ActionError::Exec(e)) => Err(e),
        Err(ActionError::InvalidResult(e)) => Err(e),
    }
}

/// Undo `mcp_name`: `ix_supervised_linear_regression_fit` → dotted skill name
/// matched against the registry by progressively collapsing underscores.
fn mcp_to_skill_name(mcp_tool_name: &str) -> Option<String> {
    let rest = mcp_tool_name.strip_prefix("ix_")?;
    // The registry stores names with dots. The MCP name replaced dots with
    // underscores, so we must scan the registry for a name whose MCP form
    // matches. O(n) on total skill count — trivial.
    for desc in ix_registry::all() {
        if mcp_name(desc.name) == mcp_tool_name {
            return Some(desc.name.to_string());
        }
    }
    // Fallback: heuristic dot-splitting (works for skills with no dots in
    // segment names, which is the convention we enforce).
    Some(rest.replace('_', "."))
}

/// Produce an MCP [`Tool`] definition for every skill in the registry. Used
/// by `ToolRegistry::register_all` to merge registry-sourced tools with any
/// remaining manual entries.
pub fn all_registry_tools() -> Vec<Tool> {
    ix_registry::all()
        .map(|desc| Tool {
            name: mcp_name_static(desc),
            description: desc.doc,
            input_schema: (desc.json_schema)(),
            handler: registry_handler_for(desc),
        })
        .collect()
}

/// Leak the MCP-name `String` to satisfy `Tool`'s `&'static str` contract.
/// The leak is bounded — one `String` per skill, once, at process startup.
fn mcp_name_static(desc: &'static SkillDescriptor) -> &'static str {
    // Interning via `OnceLock` keyed on the skill pointer identity.
    use std::collections::HashMap;
    use std::sync::OnceLock;
    static INTERN: OnceLock<std::sync::Mutex<HashMap<&'static str, &'static str>>> =
        OnceLock::new();
    let map = INTERN.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap();
    if let Some(&s) = guard.get(desc.name) {
        return s;
    }
    let leaked: &'static str = Box::leak(mcp_name(desc.name).into_boxed_str());
    guard.insert(desc.name, leaked);
    leaked
}

/// Per-descriptor handler: routes back through the registry by the stored
/// descriptor's name. Returned fn-pointer has static lifetime by using a
/// per-skill closure lifted into a stored `Box<dyn Fn>` via `OnceLock`.
///
/// MCP tool handlers must be `fn(Value) -> Result<Value, String>`, a plain
/// function pointer with no captures. To carry the skill name, we generate
/// one thunk per tool through dynamic dispatch stored in an interning table
/// — but since we can't lift a `Box<dyn Fn>` to `fn`, we instead use the
/// MCP name inside a generic dispatch path. `ToolRegistry::call` knows the
/// tool's name and calls [`dispatch`] directly for registry-backed tools.
fn registry_handler_for(_: &'static SkillDescriptor) -> fn(JsonValue) -> Result<JsonValue, String> {
    // Stored as a marker; `ToolRegistry::call` intercepts before invoking.
    registry_handler_marker
}

/// Sentinel handler — never executed directly. `ToolRegistry::call` sees this
/// pointer and routes to [`dispatch`] with the tool name.
pub fn registry_handler_marker(_: JsonValue) -> Result<JsonValue, String> {
    Err("registry_handler_marker should be intercepted by ToolRegistry::call".into())
}

/// True iff `handler` is the registry marker — i.e. the tool is backed by a
/// registry skill, not a manual handler.
pub fn is_registry_backed(handler: fn(JsonValue) -> Result<JsonValue, String>) -> bool {
    (handler as *const ()) == (registry_handler_marker as *const ())
}
