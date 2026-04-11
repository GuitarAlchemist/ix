# AgentContext / AgentAction — shared substrate for the harness primitives roadmap

**Status:** brainstorm → ready for `/octo:develop` on `ix-agent-core`
**Date:** 2026-04-10
**Session format:** Multi-AI brainstorm (Codex + Gemini, Claude subagent deferred)
**Unblocks:** `ix-approval` (#3), `ix-middleware` full trait (#2), `ix-session` (#4), trace flywheel (#6)
**Related:**
- `docs/brainstorms/2026-04-10-ix-harness-primitives.md` — the seven-primitive roadmap
- `docs/brainstorms/2026-04-10-context-dag.md` — shipped primitive #1
- `docs/brainstorms/2026-04-10-hexavalent-or-discrepancy.md` — the OR bug that forbids algebraic verdict composition
- `crates/ix-context/` — the replayability contract this design inherits
- `crates/ix-loop-detect/` — the first consumer that will migrate to the new shape

---

## Opening frame

Every primitive past #1 has been blocked at the point where it wants to know *what an agent is doing* and *what state the agent is in*. ix-context dodged the question by being a standalone walker with explicit inputs. ix-loop-detect dodged it by keying on raw strings. ix-approval, ix-middleware, and ix-session cannot dodge — they ARE the primitives that need structured agent-level types.

This doc resolves those two types at the brainstorm level so the next three primitives can ship without re-litigating the substrate.

The brainstorm's load-bearing insight, from Gemini: **"Don't build a Context object. Build a Timeline Manager that projects a Context View. Transforms ARE Events."** Combined with Codex's concrete Rust layout and the user's decisions in the challenge round, the design is:

- Two context types (`ReadContext` for handlers, `WriteContext` for middleware) — per user decision to split rather than combine
- A new `ix-agent-core` crate — clean dep graph, no cycles
- First-class `replay_seed: u64` in `ReadContext` — closes Gemini's "if RNG state isn't in context, replay is an illusion" critique
- `AgentAction` as a tagged enum with ordinal, correlation ID, and a `loop_key()` helper
- `ix-loop-detect` rewritten to consume `&AgentAction` — breaking change, explicitly approved
- Backwards compat for the 48 existing `fn(Value) → Result<Value, String>` skills via `LegacyAdapter`

---

## The architectural thesis

> **Transforms ARE Events.** Middleware does not mutate actions. It emits events that project the effective state during replay.

This is the resolution to the Anthropic-read-only vs LangChain-mutable tension. Both frameworks are right about their piece. IX takes Anthropic's read-only constraint on the *handler path* (tools cannot mutate context) and LangChain's transformation semantics on the *middleware path* (hooks can emit events that affect downstream handlers). The reconciliation: those events are written to an append-only log, and the next turn's `ReadContext` is a projection of the log up to that ordinal.

Under this model:

- `ReadContext` is **pure data**, deterministic, serializable, bit-exact reconstructible from the event log.
- `WriteContext` is the **append cursor** — a middleware can emit a `Block` event, a `Replace` event, an `Observation` event, an `Approval` event. It cannot mutate prior events, only append new ones.
- The dispatcher **projects** `ReadContext` from the log before each handler call. Projection is a pure function `f(EventLog, ordinal) → ReadContext`.
- Replay is **trivial**: reset the ordinal, re-project, re-run the handlers. Same inputs produce bit-identical outputs because `ReadContext` is deterministic and handlers are pure over it.

The RNG seed is in `ReadContext` because IX ships ML tools that all use seeded randomness — K-means, transformers, adversarial attacks, evolutionary algorithms. If `replay_seed` is NOT in the context, "replayable walk_trace" is a lie the moment a tool calls `rand::thread_rng()`.

---

## Type layouts — concrete Rust

All types live in a new crate `crates/ix-agent-core/`.

### `ReadContext` — passed to tool handlers

```rust
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use serde_json::Value as JsonValue;

/// Read-only snapshot of agent state visible to a tool handler.
///
/// This is a **projection** of the session event log at a specific ordinal.
/// It is pure data — no interior mutability, no runtime handles, no
/// references into the dispatcher. Handlers receive `&ReadContext` and
/// cannot modify it. State changes flow through [`WriteContext`] and
/// become new events in the log.
///
/// `BTreeMap` is deliberate: the governance-instrument contract requires
/// bit-exact cross-process replayability, and `HashMap` iteration is not
/// stable across processes even though it's stable within one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadContext {
    /// Unique session identifier. Stable across the lifetime of one
    /// agent conversation; changes on every new /clear or restart.
    pub session_id: String,

    /// Monotonic turn counter within the session. Incremented once per
    /// handler invocation. `0` is the session's first turn.
    pub ordinal: u64,

    /// Deterministic RNG seed for the current turn. **This is the
    /// load-bearing field for replay identity when ML tools run.**
    /// Tools that use randomness MUST seed from this value rather than
    /// from `rand::thread_rng()` or timestamps.
    ///
    /// The seed is a pure function of `(session_id, ordinal)` by default,
    /// but middleware can override it via an event — e.g., for A/B
    /// testing or stochastic-exploration escape hatches.
    pub replay_seed: u64,

    /// The caller chain for the current handler. `["user", "ix_ml_pipeline"]`
    /// means the user invoked `ix_ml_pipeline`, which internally called
    /// whatever tool is currently executing. Used by loop detection and
    /// cycle diagnostics.
    pub tool_stack: Vec<String>,

    /// Labels attached by policy middleware — e.g.,
    /// `["approval:tier-two", "governance:proceed-with-note"]`.
    /// Consumers read these to adapt behavior without coupling to specific
    /// middleware crates.
    pub policy_labels: Vec<String>,

    /// VFS-style extensible metadata. Middleware mounts data at a path:
    ///
    /// - `"approval/blast_radius"` → `{"nodes": 42, "edges": 115}`
    /// - `"loop_detect/count"`     → `7`
    /// - `"budget/time_remaining_ms"` → `4200`
    ///
    /// `BTreeMap` preserves key order for replay; values are
    /// `serde_json::Value` so ix-session can log them verbatim without
    /// type coupling to every middleware crate.
    pub metadata: BTreeMap<String, JsonValue>,

    /// Cached hexavalent belief states relevant to the current turn.
    /// Read-only projection from Demerzel's belief files. Keys are
    /// proposition strings; values are hexavalent snapshots.
    pub beliefs: BTreeMap<String, Hexavalent>,

    /// Capability bits that describe which harness primitives are active.
    /// Tools use this to gate behavior (e.g., "only emit this event if
    /// the session has a writer").
    pub capabilities: AgentCapabilities,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub middleware: bool,
    pub approval: bool,
    pub session_log: bool,
    pub context_dag: bool,
}

impl ReadContext {
    /// The top of the tool_stack — the current tool's caller.
    pub fn caller(&self) -> Option<&str> {
        self.tool_stack.last().map(String::as_str)
    }

    /// Derive a child RNG seed for a sub-call within the current turn.
    /// Uses ordinal + sub-index for deterministic substreaming.
    pub fn child_seed(&self, sub_index: u32) -> u64 {
        // Splittable seeding: xor the ordinal and sub_index into the
        // parent seed. Cheap, deterministic, adequate for non-crypto use.
        self.replay_seed
            ^ (self.ordinal.rotate_left(17))
            ^ (sub_index as u64).rotate_left(31)
    }
}
```

### `WriteContext` — passed to middleware

```rust
/// The append cursor middleware uses to emit events.
///
/// Wraps a `&ReadContext` (so middleware can see the current state)
/// and a `&mut EventSink` (so middleware can emit new events). The
/// dispatcher drains the sink into the session log after each middleware
/// runs.
///
/// **Lifetime discipline:** `WriteContext` never escapes a single
/// middleware invocation. It's a stack-scoped handle, not stored in
/// long-lived structures. This prevents middleware from holding the
/// sink across turns and pretending it's not mutating state.
pub struct WriteContext<'a> {
    pub read: &'a ReadContext,
    pub sink: &'a mut dyn EventSink,
}

/// Append-only event emission interface.
///
/// `ix-session` (primitive #4) will provide the concrete implementation
/// backed by a JSONL log. For unit tests and ephemeral use, a
/// `VecEventSink` collects events into a `Vec<SessionEvent>` in-memory.
pub trait EventSink: Send + Sync {
    /// Append one event to the log. MUST be idempotent on equal inputs
    /// (same event payload → same appended state modulo the ordinal
    /// assigned by the sink).
    fn emit(&mut self, event: SessionEvent);

    /// Current ordinal cursor — what the next emitted event's ordinal
    /// will be. Used by middleware that wants to cite the upcoming
    /// event in a block message.
    fn next_ordinal(&self) -> u64;
}

/// In-memory sink for unit tests and ephemeral dispatches. Not for
/// production.
#[derive(Debug, Default, Clone)]
pub struct VecEventSink {
    pub events: Vec<SessionEvent>,
}

impl EventSink for VecEventSink {
    fn emit(&mut self, event: SessionEvent) {
        self.events.push(event);
    }

    fn next_ordinal(&self) -> u64 {
        self.events.len() as u64
    }
}
```

### `AgentAction` — what the agent wants to do

```rust
/// A structured representation of an agent's intended action.
///
/// Actions are immutable after creation. Middleware cannot mutate an
/// `AgentAction` — instead, it emits a `Replace` event that the
/// dispatcher consults before invoking the handler, producing a new
/// action without breaking the original's identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentAction {
    /// Invoke a registered MCP tool.
    InvokeTool {
        tool_name: String,
        params: JsonValue,
        /// Correlation ID — the ordinal of this action in the session
        /// log. Used to link an action to its outcome and any
        /// middleware annotations.
        ordinal: u64,
        /// Optional target argument hint for fine-grained loop
        /// detection. e.g., for `ix_context_walk`, this is the
        /// target function path. `None` means loop detection keys on
        /// tool_name alone.
        target_hint: Option<String>,
    },

    /// Emit a non-tool observation — e.g., a summary, a progress
    /// update, a belief assertion. These do not invoke the dispatcher
    /// but are logged for replay.
    EmitObservation {
        stream: String,
        payload: JsonValue,
        ordinal: u64,
    },

    /// Request approval for a high-risk action. The approval middleware
    /// projects the subject, computes a verdict, and either blocks
    /// (via a Block event) or lets the dispatcher proceed.
    RequestApproval {
        classifier: String,
        subject: JsonValue,
        ordinal: u64,
    },

    /// Return a final result to the caller. Terminates the current
    /// turn for this agent.
    Return {
        payload: JsonValue,
        ordinal: u64,
    },
}

impl AgentAction {
    /// A stable key for loop detection. `ix-loop-detect` uses this
    /// rather than raw strings so future per-target granularity
    /// upgrades are transparent.
    ///
    /// For `InvokeTool`, combines `tool_name` with `target_hint` if
    /// present: `"ix_context_walk:mini::eigen::jacobi"` vs bare
    /// `"ix_context_walk"`.
    ///
    /// For non-tool actions, returns `None` — loop detection only
    /// applies to repeated tool invocations.
    pub fn loop_key(&self) -> Option<String> {
        match self {
            AgentAction::InvokeTool {
                tool_name,
                target_hint,
                ..
            } => Some(match target_hint {
                Some(t) => format!("{tool_name}:{t}"),
                None => tool_name.clone(),
            }),
            _ => None,
        }
    }

    /// Monotonic ordinal within the session. Never None — every action
    /// has a position in the log.
    pub fn ordinal(&self) -> u64 {
        match self {
            AgentAction::InvokeTool { ordinal, .. }
            | AgentAction::EmitObservation { ordinal, .. }
            | AgentAction::RequestApproval { ordinal, .. }
            | AgentAction::Return { ordinal, .. } => *ordinal,
        }
    }
}
```

### `SessionEvent` — the append-only log primitive

```rust
/// An entry in the session event log. ix-session owns the actual
/// persistence and ordinal assignment; ix-agent-core defines the shape.
///
/// **"Transforms ARE Events."** When middleware wants to rewrite an
/// action, it emits `ActionReplaced` rather than mutating. When it
/// wants to block, it emits `ActionBlocked` rather than throwing.
/// When it wants to annotate, it emits `MetadataMounted`. The dispatcher
/// projects `ReadContext` from the event log, so downstream handlers
/// see the effect of the rewrite without caring how it happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionEvent {
    /// The agent proposed an action. First event of every turn.
    ActionProposed {
        ordinal: u64,
        action: AgentAction,
    },

    /// Middleware blocked the action. Handler is NOT called. The block
    /// event carries the reason and the code so downstream middleware
    /// and audit tools can react.
    ActionBlocked {
        ordinal: u64,
        code: BlockCode,
        reason: String,
        emitted_by: String, // middleware name
    },

    /// Middleware rewrote the action. Downstream sees the new action in
    /// place of the old one when projecting ReadContext.
    ActionReplaced {
        ordinal: u64,
        original: AgentAction,
        replacement: AgentAction,
        emitted_by: String,
    },

    /// Middleware or a handler mounted data at a metadata path. This is
    /// Gemini's VFS-namespace idea made concrete.
    MetadataMounted {
        ordinal: u64,
        path: String,
        value: JsonValue,
        emitted_by: String,
    },

    /// A handler completed successfully with a value.
    ActionCompleted {
        ordinal: u64,
        value: JsonValue,
    },

    /// A handler failed with a typed error.
    ActionFailed {
        ordinal: u64,
        error: ActionError,
    },

    /// A hexavalent belief changed. Projected into
    /// `ReadContext::beliefs` on the next turn.
    BeliefChanged {
        ordinal: u64,
        proposition: String,
        old: Option<Hexavalent>,
        new: Hexavalent,
        evidence: JsonValue,
    },
}
```

### `MiddlewareVerdict` and error types

```rust
/// The return shape of a middleware invocation. Middleware produces
/// a verdict AND may emit events through its `WriteContext.sink`. The
/// two are complementary: the verdict tells the dispatcher what to do
/// next (continue / block / replace), and the events tell the session
/// log what happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum MiddlewareVerdict {
    /// Let the dispatcher proceed to the next middleware or, if this
    /// is the last, to the tool handler.
    Continue,

    /// Stop the chain and return an `ActionError::Blocked`. The
    /// dispatcher emits an `ActionBlocked` event into the session log
    /// automatically. The middleware itself does NOT need to emit the
    /// event manually — the dispatcher owns that responsibility to
    /// guarantee it happens.
    Block {
        code: BlockCode,
        reason: String,
    },

    /// Replace the pending action with a new one. The dispatcher emits
    /// an `ActionReplaced` event and re-projects `ReadContext` for
    /// downstream middleware. This is the "Transforms ARE Events"
    /// principle made operational.
    Replace(AgentAction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockCode {
    LoopDetected,
    BudgetExceeded,
    ApprovalRequired,
    PolicyDenied,
    BlastRadiusTooLarge,
    GovernanceViolation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
pub enum ActionError {
    #[error("blocked by {blocker}: {reason} (code: {code:?})")]
    Blocked {
        code: BlockCode,
        reason: String,
        blocker: String, // middleware name
    },
    #[error("execution failed: {0}")]
    Exec(String),
    #[error("handler returned invalid value: {0}")]
    InvalidResult(String),
}

pub type ActionResult = Result<ActionOutcome, ActionError>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcome {
    pub value: JsonValue,
    /// Events the handler (not middleware — middleware uses the sink
    /// directly) wants to append to the session log. Handlers can emit
    /// their own events via this field even though they only see a
    /// `&ReadContext`, because the outcome flows back through the
    /// dispatcher which has write access.
    pub events: Vec<SessionEvent>,
}
```

### The handler trait

```rust
/// The contract for an agent-native handler. Contrast with the legacy
/// `fn(Value) → Result<Value, String>` shape in ix-agent's current
/// dispatcher.
///
/// Handlers receive `&ReadContext` (immutable projection of current
/// state) and `&AgentAction` (the specific action to handle). They
/// return an `ActionResult` whose `ActionOutcome::events` field
/// lets them append to the log without having mutable access to it.
pub trait AgentHandler: Send + Sync + 'static {
    fn run(&self, cx: &ReadContext, action: &AgentAction) -> ActionResult;
}
```

---

## The dispatcher flow

```text
    +-----------------------------+
    |  Agent proposes AgentAction |
    +-------------+---------------+
                  |
                  v
    +-----------------------------+
    |  Dispatcher emits           |
    |  ActionProposed event       |
    +-------------+---------------+
                  |
                  v
    +-----------------------------+     Middleware 1 (e.g., ix-loop-detect)
    |  Project ReadContext from   |     Gets (&WriteContext { read, sink })
    |  event log @ current ordinal| --> Returns MiddlewareVerdict
    +-------------+---------------+
                  |
        Continue  | Block/Replace
                  v
    +-----------------------------+
    |  Re-project ReadContext     |     Middleware 2 (e.g., ix-approval)
    |  (in case Replace fired)    | --> Returns MiddlewareVerdict
    +-------------+---------------+
                  |
        Continue  | Block/Replace
                  v
          ... chain continues ...
                  |
                  v
    +-----------------------------+
    |  All middleware passed.     |
    |  Project final ReadContext. |
    |  Invoke AgentHandler::run   |
    |  with &ReadContext + action |
    +-------------+---------------+
                  |
                  v
    +-----------------------------+
    |  Drain ActionOutcome.events |
    |  into session log.          |
    |  Emit ActionCompleted/Failed|
    +-----------------------------+
```

Every arrow is an event in the log. Replay traverses the same flow, but the middleware invocations are skipped (their Verdicts are cached in the log as `ActionBlocked`/`ActionReplaced` events) — only the handler re-runs with the projected `ReadContext`.

---

## Backwards compatibility

**Non-negotiable:** the 48 existing `#[ix_skill]`-annotated tools must keep working with zero signature changes.

### The `LegacyAdapter`

```rust
/// Adapts a `fn(Value) → Result<Value, String>` legacy skill to the
/// new `AgentHandler` trait.
pub struct LegacyAdapter {
    pub inner: fn(JsonValue) -> Result<JsonValue, String>,
}

impl AgentHandler for LegacyAdapter {
    fn run(&self, _cx: &ReadContext, action: &AgentAction) -> ActionResult {
        match action {
            AgentAction::InvokeTool { params, .. } => {
                let value = (self.inner)(params.clone())
                    .map_err(ActionError::Exec)?;
                Ok(ActionOutcome {
                    value,
                    events: Vec::new(),
                })
            }
            _ => Err(ActionError::Exec(
                "legacy adapter only supports InvokeTool actions".into(),
            )),
        }
    }
}
```

### `registry_bridge::dispatch` after the change

```rust
// crates/ix-agent/src/registry_bridge.rs

pub fn dispatch(mcp_tool_name: &str, params: JsonValue) -> Result<JsonValue, String> {
    // New: synthesize an AgentAction + minimal ReadContext and delegate
    // to the new path. The legacy API returns the bare JsonValue; the
    // middleware chain, session events, etc. are handled transparently.
    let action = AgentAction::InvokeTool {
        tool_name: mcp_tool_name.to_string(),
        params,
        ordinal: 0, // synthetic — legacy callers don't have a session
        target_hint: None,
    };
    let cx = ReadContext::synthetic_for_legacy();
    match dispatch_action(&cx, &action) {
        Ok(outcome) => Ok(outcome.value),
        Err(ActionError::Blocked { reason, .. }) => Err(reason),
        Err(ActionError::Exec(msg)) => Err(msg),
        Err(ActionError::InvalidResult(msg)) => Err(msg),
    }
}

pub fn dispatch_action(cx: &ReadContext, action: &AgentAction) -> ActionResult {
    // Full middleware chain + handler invocation.
    // ...
}
```

The 48 existing parity test entries pass through `dispatch()` unchanged. The new `dispatch_action` path is strictly additive.

### `ix-loop-detect` migration

Per the user's decision, this is a **breaking rewrite**: `LoopDetector` will consume `&AgentAction` directly via `action.loop_key()`, not raw strings.

The migration plan:

1. Bump `ix-loop-detect` to depend on `ix-agent-core` for the `AgentAction` type
2. Rewrite `LoopDetector::record(&str)` → `LoopDetector::record(&AgentAction)`
3. Internally, the detector still keys on `String` — it just calls `action.loop_key()` to extract the key
4. Actions whose `loop_key()` is `None` (observations, returns, approvals) are not counted — they can't loop
5. Update the 8 parity tests and the loop_detector_trips_on_repeated_dispatch integration test to construct `AgentAction::InvokeTool` values instead of passing tool-name strings
6. Update `registry_bridge::dispatch` to synthesize the action before calling `detector.record(&action)`

The breaking change is bounded: only the 8 tests + 1 dispatcher call site need updating. Zero external consumers today.

---

## What this unblocks

| Primitive | Now implementable because | Estimated scope |
|---|---|---|
| **`ix-middleware` full trait** (#2) | `AgentHandler`, `MiddlewareVerdict`, `WriteContext`, `SessionEvent` all defined here | 1–2 days |
| **`ix-approval`** (#3) | `classify_action(cx: &ReadContext, action: &AgentAction, dag: Option<&ContextBundle>)` has real types | 1 day |
| **`ix-session`** (#4) | `SessionEvent` is defined; `EventSink` trait is defined; JSONL persistence is implementation detail | 2–3 days |
| **`ix-loop-detect` v2** | `AgentAction::loop_key()` helper + breaking rewrite of the record API | 0.5 day |
| **Trace flywheel** (#6) | Full event log via `SessionEvent` → `ix_trace_ingest` is mechanical | 0.5 day |

**Total unblocked: ~5–7 days of implementation** after this doc lands.

---

## What this defers

- **Merkle parent hashing** on ReadContext — Gemini's "Git for Thoughts" idea. Deferred until ix-session proves the event log is the right primitive for causal proof. The `ordinal: u64` already gives total ordering without cryptographic hashing.
- **Inverse Tool Pattern** (environment acts on agent) — user-initiated interrupts and external pauses don't exist yet. Revisit when there's a concrete use case.
- **Cross-language federation of AgentAction** (JSON-LD voucher) — defer to when TARS or GA actually need to consume ix actions. Today they consume MCP JSON envelopes, which round-trip fine.
- **Session-scoped LoopDetector instances** — the current `OnceLock<LoopDetector>` is process-global. When ix-session lands, each session should get its own detector. Defer the plumbing until there's a concrete session type.
- **`Hexavalent::or`-style verdict composition** — forbidden per the OR decision doc. Middleware chains compose via sequence (any Block stops the chain), not algebraically. Documented, not implemented.
- **Stochastic exploration escape hatch** — Claude's "map without terra incognita" insight from the Context DAG brainstorm. Would need a middleware that perturbs `replay_seed` or injects random actions. Out of scope for v1 of the substrate.

---

## Open questions for `/octo:develop`

1. **Should `metadata: BTreeMap<String, JsonValue>` enforce path namespacing?** e.g., reject keys that don't match `^[a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)*$`. Helps prevent middleware from polluting the top level, but adds runtime validation overhead.

2. **Does `ReadContext::synthetic_for_legacy()` carry meaningful defaults or just zeros?** If zeros, legacy skills see `session_id = ""`, `ordinal = 0`, `replay_seed = 0` — which means ML tools calling from legacy paths get the same seed every time. Is that the right default, or should we use a process-startup random seed that stays stable for the process lifetime?

3. **Where does the belief projection come from?** `ReadContext::beliefs: BTreeMap<String, Hexavalent>`. Populated by what? ix-session reading Demerzel's `state/beliefs/*.yaml`? A separate `ix-belief-proj` crate? For v1 this can be empty, but the hook should exist.

4. **EventSink trait object vs. generic parameter?** `pub struct WriteContext<'a>` has `sink: &'a mut dyn EventSink`. Alternative: `WriteContext<'a, S: EventSink>`. Trait object is simpler for registry tables; generic is faster and avoids vtable dispatch. **Recommendation: trait object for v1** — performance can be measured later.

5. **Should `AgentAction::InvokeTool::params` be `JsonValue` or a typed socket?** Current ix-agent skills use `JsonValue`. Typed sockets would require per-tool schemas at the action level. **Recommendation: `JsonValue` for v1** — matches the existing skill surface and keeps ix-agent-core dependency-free of ix-registry.

6. **Does `ReadContext` need a `persona: Option<String>`?** Demerzel personas are a thing. For v1, `persona` can live in `metadata["demerzel/persona"]` and avoid a hard coupling to ix-governance.

---

## Multi-perspective analysis

### Provider contributions

| Provider | Key contribution | Unique insight |
|---|---|---|
| 🔴 Codex | Concrete Rust type layout with `BTreeMap`, `BlockCode`, `LegacyAdapter`, `dispatch_action` signature, `ix-agent-core` crate placement with explicit cycle analysis, the gotchas list (Ord on Value, orphan rules, generic poisoning) | *"Don't put trait objects inside AgentContext. Put them in an AgentRuntime passed separately."* Clean separation of data from runtime handles. |
| 🟡 Gemini | The *Timeline Manager* reframe: "Don't build a Context object. Build a Timeline Manager that projects a Context View." The "Transforms ARE Events" paradox resolution. Intent-Voucher and Cognitive Horizon naming. RNG seed as first-class replay concern. VFS-namespace metadata pattern. | *"If the LLM's temperature isn't in the context, replay is an illusion."* Single line that forced `replay_seed: u64` from "nice to have" to "load-bearing." |
| 🔵 Claude subagent | Deferred — output did not arrive before synthesis. Expected contribution was paradox-hunting and pattern-naming, which Gemini substantially covered. | — |

### Cross-provider patterns

- **Convergence on read-only context + event emission.** Both providers independently argued for immutable handler view + append-only log. The `WriteContext` split that the user chose in the challenge round is the Rust-idiomatic encoding of Gemini's "Transforms ARE Events."
- **Convergence on deterministic ordering.** Codex said `BTreeMap` explicitly; Gemini said "topological coordinate" and "Merkle path." Both reject HashMap iteration for cross-process replay.
- **Convergence on federation via wire format, not shared types.** Codex's owned `String` for tool names + serde; Gemini's JSON-LD voucher. Both reject cross-language shared Rust types.
- **Divergence on primary abstraction.** Codex treats `AgentContext` as a concrete struct with fields. Gemini treats it as a "coordinate in execution space." The design reconciles: the struct IS the projection, the coordinate is `(session_id, ordinal)` which fully identifies the projection.

---

## Handoff — three implementation paths, user picks one

After this brainstorm lands, the natural next `/octo:develop` targets in increasing scope:

| Path | Ship | Scope |
|---|---|---|
| **A. Substrate only** | `crates/ix-agent-core/` with all the types from this doc, zero behavior, full serde round-trip tests, no dispatcher changes yet | ~2 hours |
| **B. Substrate + `ix-loop-detect` migration** | Same as A, plus the breaking rewrite of `LoopDetector::record` to consume `&AgentAction` + update the 8 parity tests | ~3 hours |
| **C. Substrate + middleware trait + full `ix-loop-detect` rewrite + `dispatch_action` path** | A+B plus the new `AgentHandler` trait, `LegacyAdapter`, updated `registry_bridge::dispatch` with the middleware chain fold, backwards-compat integration tests | ~1 day |

**Path B** is the recommended first ship — it validates the substrate against a real consumer (ix-loop-detect) without taking on the full middleware architecture. If Path B succeeds, Path C is the obvious follow-up.

---

## References read during this brainstorm

- Codex response: Rust feasibility analysis with concrete type layouts, backwards compat plan, crate placement recommendation, three non-obvious Rust gotchas
- Gemini response: Timeline Manager reframe, three real-world analogies (Temporal, ROS 2, React Fiber), RNG-seed insight, VFS namespace pattern, Merkle path causal proof
- `docs/brainstorms/2026-04-10-ix-harness-primitives.md` — the seven-primitive roadmap this doc unblocks
- `docs/brainstorms/2026-04-10-context-dag.md` — the replayability contract this design inherits
- `docs/brainstorms/2026-04-10-hexavalent-or-discrepancy.md` — the OR bug that forbids algebraic verdict composition
- `crates/ix-loop-detect/src/lib.rs` — the first consumer that will migrate
- `crates/ix-context/src/model.rs` — the `walk_trace` pattern that `SessionEvent` inherits
- `crates/ix-agent/src/registry_bridge.rs` — the current dispatch shape

---

## Decision record (from the challenge round)

| Question | Decision | Rationale |
|---|---|---|
| Primary mutability model? | **Split `ReadContext` / `WriteContext`** | User choice — clearer intent than a single type with a CommandBuffer. Rust encoding: `WriteContext<'a>` borrows a `&'a ReadContext` and holds a `&'a mut dyn EventSink`. |
| Crate placement? | **New `crates/ix-agent-core/`** | Codex recommendation. Clean dep graph. No cycles. |
| RNG seed in `ReadContext`? | **Yes, first-class field** | Gemini's "replay is an illusion" critique. ML tools dominate IX — seeded RNG is already a convention. |
| `ix-loop-detect` migration? | **Breaking rewrite** | User choice — cleaner API justifies the 8-test update. `AgentAction::loop_key()` is the migration helper. |

---

## Next step

Hand this doc to `/octo:develop` on **Path B** (substrate + ix-loop-detect migration) as the next session's implementation target. The design is locked enough that the coding is mechanical — all the type shapes are specified, the compat story is explicit, the test impact is bounded.

When Path B ships, primitive #3 (`ix-approval`) becomes a pure additive extension that consumes the substrate without modifying it.
