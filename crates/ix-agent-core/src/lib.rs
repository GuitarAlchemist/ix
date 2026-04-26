//! # ix-agent-core — shared substrate for the harness primitives roadmap
//!
//! This crate defines the types that ix-middleware, ix-approval,
//! ix-session, the trace flywheel, and the migrated ix-loop-detect all
//! consume. It exists so the consumers don't each reinvent incompatible
//! versions of the same concepts.
//!
//! ## Governing thesis
//!
//! > **Transforms ARE Events.** Middleware does not mutate actions. It
//! > emits events that project the effective state during replay.
//!
//! Resolving the Anthropic-read-only vs LangChain-mutable tension: tools
//! receive an immutable `&ReadContext` and return an [`ActionOutcome`]
//! whose `events` field lets them append to the log without having
//! mutable access to it. Middleware receives a `WriteContext` that wraps
//! a `&ReadContext` and a `&mut dyn EventSink` — it can emit events but
//! cannot mutate prior state.
//!
//! Every [`ReadContext`] is a pure projection of the session event log
//! at a specific ordinal. Replay is `f(EventLog, ordinal) -> ReadContext`.
//! Same inputs → bit-identical outputs, across processes, because every
//! map in the context uses `BTreeMap` rather than `HashMap`.
//!
//! ## Modules
//!
//! - [`context`] — [`ReadContext`], [`WriteContext`], [`EventSink`],
//!   [`VecEventSink`], [`AgentCapabilities`]
//! - [`action`] — [`AgentAction`] tagged enum with the `loop_key` helper
//! - [`event`] — [`SessionEvent`], [`ActionOutcome`], [`MiddlewareVerdict`],
//!   [`BlockCode`]
//! - [`error`] — [`ActionError`] + the [`ActionResult`] typedef
//! - [`handler`] — [`AgentHandler`] trait + [`LegacyAdapter`] for the
//!   existing `fn(Value) -> Result<Value, String>` skill surface
//! - [`middleware`] — [`Middleware`] trait + [`MiddlewareChain`] fold
//!
//! ## Scope boundaries
//!
//! - **No session persistence.** Primitive #4 (`ix-session`) owns the
//!   JSONL-backed [`EventSink`] implementation. This crate ships a
//!   [`VecEventSink`] for tests only.
//! - **No belief projection pipeline.** `ReadContext::beliefs` is a
//!   passive field; consumers populate it from Demerzel state via a
//!   separate reader (not yet built).
//! - **No approval classifier.** Primitive #3 (`ix-approval`) consumes
//!   [`MiddlewareVerdict`] but lives in its own crate.
//! - **No algebraic verdict composition.** The `Hexavalent::or` bug is
//!   documented and unresolved. Consumers compose middleware chains via
//!   sequence (any Block stops the chain), not algebraically.
//!
//! ## Backwards compatibility
//!
//! The existing `fn(serde_json::Value) -> Result<serde_json::Value, String>`
//! skill signature is preserved via [`handler::LegacyAdapter`], which
//! wraps such functions as an [`handler::AgentHandler`] impl. The 48
//! `#[ix_skill]`-annotated tools in ix-agent continue to work without
//! signature changes.

pub mod action;
pub mod beliefs;
pub mod context;
pub mod error;
pub mod event;
pub mod handler;
pub mod middleware;

pub use action::AgentAction;
pub use beliefs::{project_beliefs, tool_proposition, BeliefMiddleware};
pub use context::{AgentCapabilities, EventSink, ReadContext, VecEventSink, WriteContext};
pub use error::{ActionError, ActionResult};
pub use event::{ActionOutcome, BlockCode, MiddlewareVerdict, SessionEvent};
pub use handler::{AgentHandler, LegacyAdapter};
pub use middleware::{Middleware, MiddlewareChain};
