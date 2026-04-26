//! Read-only context + event-emission writer for the harness primitives.
//!
//! See `docs/brainstorms/2026-04-10-agent-context-action.md` for the full
//! design rationale. In short: [`ReadContext`] is a pure data projection
//! of the session event log at a specific ordinal; [`WriteContext`] wraps
//! a `&ReadContext` with a `&mut dyn EventSink` so middleware can emit
//! new events without mutating prior state.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

use ix_types::Hexavalent;
use serde_json::Value as JsonValue;

use crate::event::SessionEvent;

// ---------------------------------------------------------------------------
// ReadContext
// ---------------------------------------------------------------------------

/// Read-only snapshot of agent state visible to a tool handler.
///
/// This is a **projection** of the session event log at a specific
/// ordinal. It is pure data — no interior mutability, no runtime handles,
/// no references into the dispatcher. Handlers receive `&ReadContext`
/// and cannot modify it. State changes flow through [`WriteContext`] and
/// become new events in the log.
///
/// `BTreeMap` is deliberate: the governance-instrument contract requires
/// bit-exact cross-process replayability, and `HashMap` iteration is not
/// stable across processes even though it's stable within one.
///
/// `Eq` is deliberately **not** derived because `serde_json::Value` does
/// not implement `Eq` (floating-point `NaN` breaks the `Eq` contract).
/// `PartialEq` is sufficient for test comparisons and round-trips.
///
/// # Example
///
/// ```
/// use ix_agent_core::ReadContext;
///
/// let cx = ReadContext::synthetic_for_legacy();
/// assert_eq!(cx.session_id, "legacy");
/// // The seed is non-zero and stable for the lifetime of the process.
/// let seed1 = cx.replay_seed;
/// let seed2 = ReadContext::synthetic_for_legacy().replay_seed;
/// assert_eq!(seed1, seed2);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadContext {
    /// Unique session identifier. Stable across the lifetime of one
    /// agent conversation; changes on every new `/clear` or restart.
    pub session_id: String,

    /// Monotonic turn counter within the session. Incremented once per
    /// handler invocation. `0` is the session's first turn.
    pub ordinal: u64,

    /// Deterministic RNG seed for the current turn. This is the
    /// load-bearing field for replay identity when ML tools run. Tools
    /// that use randomness MUST seed from this value rather than from
    /// `rand::thread_rng()` or timestamps.
    ///
    /// The seed is a pure function of `(session_id, ordinal)` by
    /// default, but middleware can override it via an event — e.g.,
    /// for A/B testing or stochastic-exploration escape hatches.
    pub replay_seed: u64,

    /// The caller chain for the current handler. `["user",
    /// "ix_ml_pipeline"]` means the user invoked `ix_ml_pipeline`,
    /// which internally called whatever tool is currently executing.
    /// Used by loop detection and cycle diagnostics.
    pub tool_stack: Vec<String>,

    /// Labels attached by policy middleware — e.g.,
    /// `["approval:tier-two", "governance:proceed-with-note"]`.
    /// Consumers read these to adapt behavior without coupling to
    /// specific middleware crates.
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

    /// Capability bits that describe which harness primitives are
    /// active. Tools use this to gate behavior (e.g., "only emit this
    /// event if the session has a writer").
    pub capabilities: AgentCapabilities,
}

/// Feature bits describing which harness primitives are wired into the
/// current dispatcher. Lets consumers check capabilities at runtime
/// rather than via compile-time feature flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct AgentCapabilities {
    /// `true` when the dispatcher has a middleware chain attached.
    pub middleware: bool,
    /// `true` when an approval classifier is wired.
    pub approval: bool,
    /// `true` when a session event log is attached.
    pub session_log: bool,
    /// `true` when the Context DAG walker is reachable.
    pub context_dag: bool,
}

impl ReadContext {
    /// The top of the `tool_stack` — the current tool's direct caller.
    /// Returns `None` for top-level user-initiated calls.
    pub fn caller(&self) -> Option<&str> {
        self.tool_stack.last().map(String::as_str)
    }

    /// Derive a child RNG seed for a sub-call within the current turn.
    /// Uses the parent seed, the ordinal, and a sub-index to produce a
    /// deterministic substream.
    ///
    /// # Example
    ///
    /// ```
    /// use ix_agent_core::ReadContext;
    /// let cx = ReadContext::synthetic_for_legacy();
    /// let a = cx.child_seed(0);
    /// let b = cx.child_seed(1);
    /// assert_ne!(a, b);
    /// // Stable: same inputs -> same output.
    /// assert_eq!(a, cx.child_seed(0));
    /// ```
    pub fn child_seed(&self, sub_index: u32) -> u64 {
        // Splittable seeding: xor the ordinal and sub_index into the
        // parent seed. Cheap, deterministic, adequate for non-crypto use.
        self.replay_seed ^ self.ordinal.rotate_left(17) ^ (sub_index as u64).rotate_left(31)
    }

    /// Build a synthetic context for legacy callers of
    /// `registry_bridge::dispatch(mcp_tool_name, params)` that don't
    /// have a real session yet.
    ///
    /// Uses a process-stable `replay_seed` so repeated legacy calls
    /// within one process get the same seed (otherwise ML tools calling
    /// from the legacy path would lose determinism on retries). The
    /// seed is generated once per process on first access and cached.
    pub fn synthetic_for_legacy() -> Self {
        Self {
            session_id: "legacy".to_string(),
            ordinal: 0,
            replay_seed: process_legacy_seed(),
            tool_stack: Vec::new(),
            policy_labels: Vec::new(),
            metadata: BTreeMap::new(),
            beliefs: BTreeMap::new(),
            capabilities: AgentCapabilities::default(),
        }
    }
}

/// Process-stable legacy seed. Lazily initialized once, reused forever.
fn process_legacy_seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| {
        // Mix the pointer of a local variable (ASLR noise) with a fixed
        // salt. Good enough for non-crypto determinism within a process.
        let local: u64 = 0;
        let ptr = &local as *const u64 as u64;
        // Splittable-ish mix.
        ptr.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xDEAD_BEEF_CAFE_F00D
    })
}

// ---------------------------------------------------------------------------
// WriteContext + EventSink
// ---------------------------------------------------------------------------

/// The append cursor middleware uses to emit events.
///
/// Wraps a `&ReadContext` (so middleware can see the current state) and
/// a `&mut EventSink` (so middleware can emit new events). The
/// dispatcher drains the sink into the session log after each middleware
/// runs.
///
/// **Lifetime discipline:** `WriteContext` never escapes a single
/// middleware invocation. It's a stack-scoped handle, not stored in
/// long-lived structures. This prevents middleware from holding the
/// sink across turns and pretending it's not mutating state.
pub struct WriteContext<'a> {
    /// Immutable view of the current read context.
    pub read: &'a ReadContext,
    /// Append cursor for new events.
    pub sink: &'a mut dyn EventSink,
}

/// Append-only event emission interface.
///
/// `ix-session` (primitive #4) will provide the concrete implementation
/// backed by a JSONL log. For unit tests and ephemeral use, a
/// [`VecEventSink`] collects events into a `Vec<SessionEvent>` in-memory.
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

/// In-memory sink for unit tests and ephemeral dispatches. **Not for
/// production** — it grows unbounded.
#[derive(Debug, Default, Clone)]
pub struct VecEventSink {
    /// All emitted events in append order.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::AgentAction;
    use serde_json::json;

    // ── ReadContext basics ──────────────────────────────────────────────

    #[test]
    fn synthetic_for_legacy_has_non_zero_seed() {
        let cx = ReadContext::synthetic_for_legacy();
        assert_ne!(cx.replay_seed, 0);
        assert_eq!(cx.session_id, "legacy");
        assert_eq!(cx.ordinal, 0);
    }

    #[test]
    fn synthetic_for_legacy_seed_is_process_stable() {
        let a = ReadContext::synthetic_for_legacy().replay_seed;
        let b = ReadContext::synthetic_for_legacy().replay_seed;
        assert_eq!(
            a, b,
            "legacy seed should be stable across calls within a process"
        );
    }

    #[test]
    fn caller_returns_last_tool_stack_entry() {
        let mut cx = ReadContext::synthetic_for_legacy();
        cx.tool_stack.push("user".to_string());
        cx.tool_stack.push("ix_ml_pipeline".to_string());
        assert_eq!(cx.caller(), Some("ix_ml_pipeline"));
    }

    #[test]
    fn caller_on_empty_stack_is_none() {
        let cx = ReadContext::synthetic_for_legacy();
        assert_eq!(cx.caller(), None);
    }

    // ── child_seed determinism ─────────────────────────────────────────

    #[test]
    fn child_seed_is_stable_for_same_inputs() {
        let cx = ReadContext::synthetic_for_legacy();
        assert_eq!(cx.child_seed(0), cx.child_seed(0));
        assert_eq!(cx.child_seed(42), cx.child_seed(42));
    }

    #[test]
    fn child_seed_differs_for_different_sub_indices() {
        let cx = ReadContext::synthetic_for_legacy();
        let a = cx.child_seed(0);
        let b = cx.child_seed(1);
        let c = cx.child_seed(2);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn child_seed_differs_across_ordinals() {
        let mut cx = ReadContext::synthetic_for_legacy();
        let a = cx.child_seed(0);
        cx.ordinal = 1;
        let b = cx.child_seed(0);
        assert_ne!(a, b);
    }

    // ── Serde round-trip ───────────────────────────────────────────────

    fn sample_context() -> ReadContext {
        let mut cx = ReadContext::synthetic_for_legacy();
        cx.session_id = "session-abc".to_string();
        cx.ordinal = 7;
        cx.tool_stack = vec!["user".into(), "ix_ml_pipeline".into()];
        cx.policy_labels = vec!["approval:tier-two".into()];
        cx.metadata
            .insert("approval/blast_radius".into(), json!({"nodes": 42}));
        cx.metadata
            .insert("budget/remaining_ms".into(), json!(4200));
        cx.beliefs.insert("api_stable".into(), Hexavalent::Probable);
        cx.capabilities.middleware = true;
        cx.capabilities.approval = true;
        cx
    }

    #[test]
    fn read_context_serde_round_trip() {
        let cx = sample_context();
        let json = serde_json::to_string(&cx).expect("serialize");
        let back: ReadContext = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, cx);
    }

    // ── BTreeMap ordering is deterministic ─────────────────────────────

    #[test]
    fn metadata_btreemap_order_is_insertion_invariant() {
        // Insert in one order in cx1, the reverse order in cx2. Serialize
        // both. JSON output MUST be identical because BTreeMap serializes
        // in sorted key order regardless of insertion order.
        let mut cx1 = ReadContext::synthetic_for_legacy();
        cx1.metadata.insert("zeta".into(), json!(1));
        cx1.metadata.insert("alpha".into(), json!(2));
        cx1.metadata.insert("mu".into(), json!(3));

        let mut cx2 = ReadContext::synthetic_for_legacy();
        cx2.metadata.insert("mu".into(), json!(3));
        cx2.metadata.insert("alpha".into(), json!(2));
        cx2.metadata.insert("zeta".into(), json!(1));

        let j1 = serde_json::to_string(&cx1).unwrap();
        let j2 = serde_json::to_string(&cx2).unwrap();
        assert_eq!(j1, j2, "BTreeMap must serialize deterministically");
    }

    #[test]
    fn beliefs_btreemap_order_is_insertion_invariant() {
        let mut cx1 = ReadContext::synthetic_for_legacy();
        cx1.beliefs.insert("b".into(), Hexavalent::True);
        cx1.beliefs.insert("a".into(), Hexavalent::False);

        let mut cx2 = ReadContext::synthetic_for_legacy();
        cx2.beliefs.insert("a".into(), Hexavalent::False);
        cx2.beliefs.insert("b".into(), Hexavalent::True);

        assert_eq!(
            serde_json::to_string(&cx1).unwrap(),
            serde_json::to_string(&cx2).unwrap()
        );
    }

    // ── VecEventSink ───────────────────────────────────────────────────

    #[test]
    fn vec_sink_emit_appends_and_tracks_ordinal() {
        let mut sink = VecEventSink::default();
        assert_eq!(sink.next_ordinal(), 0);

        sink.emit(SessionEvent::ActionProposed {
            ordinal: 0,
            action: AgentAction::Return {
                payload: json!({}),
                ordinal: 0,
            },
        });
        assert_eq!(sink.events.len(), 1);
        assert_eq!(sink.next_ordinal(), 1);

        sink.emit(SessionEvent::ActionCompleted {
            ordinal: 0,
            value: json!("ok"),
        });
        assert_eq!(sink.events.len(), 2);
        assert_eq!(sink.next_ordinal(), 2);
    }

    #[test]
    fn write_context_scope_bound_to_borrow() {
        // Compile-time check that WriteContext can't outlive its
        // borrows. If this compiles, the lifetime discipline is
        // enforced by the type system.
        let cx = ReadContext::synthetic_for_legacy();
        let mut sink = VecEventSink::default();
        {
            let wc = WriteContext {
                read: &cx,
                sink: &mut sink,
            };
            wc.sink.emit(SessionEvent::ActionCompleted {
                ordinal: 0,
                value: json!(()),
            });
        }
        assert_eq!(sink.events.len(), 1);
    }
}
