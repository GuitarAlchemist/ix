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
use ix_agent_core::{
    ActionError, ActionOutcome, ActionResult, AgentAction, AgentHandler, MiddlewareChain,
    ReadContext, VecEventSink, WriteContext,
};
use ix_approval::ApprovalMiddleware;
use ix_loop_detect::{LoopDetector, LoopVerdict};
use ix_registry::SkillDescriptor;
use ix_types::Value as IxValue;
use serde_json::Value as JsonValue;
use std::sync::{Mutex, OnceLock};

/// Process-wide loop detector for MCP tool dispatch.
///
/// Every registry-backed dispatch records the MCP tool name here before
/// invoking the skill. If the same tool is called more than the configured
/// threshold within the sliding window, the dispatch returns a structured
/// error instead of calling the tool — giving a runaway agent a hard
/// circuit-break signal.
///
/// The detector uses the brainstorm's default configuration
/// (10 calls / 5 minutes per tool) and is lazily initialized.
fn loop_detector() -> &'static LoopDetector {
    static DETECTOR: OnceLock<LoopDetector> = OnceLock::new();
    DETECTOR.get_or_init(LoopDetector::with_defaults)
}

/// Expose a handle to the shared detector so the `ix-mcp` binary (or tests)
/// can reset state between sessions. Normal skill callers should not touch
/// the detector directly — `dispatch` manages it.
pub fn shared_loop_detector() -> &'static LoopDetector {
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
        chain.push(Box::new(ApprovalMiddleware::with_defaults()));
        Mutex::new(chain)
    })
}

/// Expose a handle to the shared middleware chain so tests can push
/// additional middlewares or reset the chain between cases. Production
/// callers should not mutate the chain at runtime.
pub fn shared_middleware_chain() -> &'static Mutex<MiddlewareChain> {
    middleware_chain()
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
                let skill_name = mcp_to_skill_name(mcp_name)
                    .ok_or_else(|| ActionError::Exec(format!(
                        "not a registry-backed tool: {mcp_name}"
                    )))?;
                let descriptor = ix_registry::by_name(&skill_name)
                    .ok_or_else(|| ActionError::Exec(format!(
                        "skill not in registry: {skill_name}"
                    )))?;

                let args = [IxValue::Json(params.clone())];
                let out = (descriptor.fn_ptr)(&args)
                    .map_err(|e| ActionError::Exec(e.to_string()))?;
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
    let mut sink = VecEventSink::default();
    let chain_guard = middleware_chain()
        .lock()
        .expect("middleware chain mutex poisoned");

    let result = {
        let mut wc = WriteContext {
            read: cx,
            sink: &mut sink,
        };
        chain_guard.dispatch(&mut wc, action, &RegistryLookupHandler)
    };

    // Drop the chain lock before returning. Events emitted during
    // dispatch are currently discarded — ix-session (primitive #4)
    // will replace VecEventSink with a JSONL-backed sink and
    // preserve them.
    drop(chain_guard);
    drop(sink);

    result
}

/// Translate a dotted registry name into the MCP tool name convention.
///
/// `supervised.linear_regression.fit` → `ix_supervised_linear_regression_fit`
pub fn mcp_name(skill_name: &str) -> String {
    format!("ix_{}", skill_name.replace('.', "_"))
}

/// Dispatch a registered skill by MCP name. Wraps the JSON blob into a
/// single-element `[IxValue::Json]` slice, invokes the registry, and unwraps.
///
/// Runs through the process-wide [`LoopDetector`] before dispatching. If
/// the same `mcp_tool_name` has been called more than 10 times in the last
/// 5 minutes, returns a circuit-breaker error without invoking the skill.
/// The error message is structured so a runaway agent sees a clear halt
/// signal and can escalate rather than burn more tokens on the loop.
pub fn dispatch(mcp_tool_name: &str, params: JsonValue) -> Result<JsonValue, String> {
    // Synthesize an AgentAction::InvokeTool for the loop detector. We
    // deliberately use target_hint=None here — the dispatcher sees only
    // the raw params blob and cannot parse per-tool target semantics.
    // Downstream consumers that want finer-grained loop keying should
    // construct actions with target_hint populated and use the new
    // action-shaped API directly.
    let action = AgentAction::InvokeTool {
        tool_name: mcp_tool_name.to_string(),
        params: params.clone(),
        ordinal: 0,
        target_hint: None,
    };

    // Circuit breaker — record this call and check the verdict before
    // doing any work. See `loop_detector()` for configuration.
    let verdict = loop_detector().record(&action);
    if let LoopVerdict::TooManyEdits {
        count,
        window,
        threshold,
    } = verdict
    {
        return Err(format!(
            "ix_loop_detect: circuit breaker tripped on tool '{mcp_tool_name}' — \
             {count} calls in the last {window:?} exceeds threshold {threshold}. \
             The agent should stop calling this tool and reconsider its approach. \
             This is a governance-instrument safety check, not a transient error."
        ));
    }

    // Undo the `ix_` prefix + underscore replacement to find the dotted name.
    let skill_name = mcp_to_skill_name(mcp_tool_name)
        .ok_or_else(|| format!("not a registry-backed tool: {mcp_tool_name}"))?;
    let descriptor = ix_registry::by_name(&skill_name)
        .ok_or_else(|| format!("skill not in registry: {skill_name}"))?;

    let args = [IxValue::Json(params)];
    let out = (descriptor.fn_ptr)(&args).map_err(|e| e.to_string())?;
    match out {
        IxValue::Json(j) => Ok(j),
        // If a skill ever returns a non-Json value from the composite-handler
        // path, surface it losslessly through serde.
        other => serde_json::to_value(other).map_err(|e| e.to_string()),
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
