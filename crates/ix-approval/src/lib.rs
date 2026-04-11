//! # ix-approval — deterministic blast-radius approval policy
//!
//! Primitive #3 of the harness primitives roadmap (see
//! `docs/brainstorms/2026-04-10-ix-harness-primitives.md`). Implements
//! Claude Code auto-mode's Tier 1 / Tier 2 / Tier 3 classification as a
//! pure, hand-coded policy — no LLM classifier, no fuzzy scoring, no
//! algebraic composition.
//!
//! ## Thesis
//!
//! > Auto-mode's 17% false-negative rate on blast-radius judgment is
//! > caused by an LLM classifier guessing whether user consent covers
//! > the proposed scope. IX can do better deterministically by
//! > computing blast radius from the Context DAG and comparing against
//! > a fixed threshold.
//!
//! This crate ships the classification logic as an
//! [`ix_agent_core::Middleware`] impl that inspects
//! [`ix_agent_core::AgentAction`]s at dispatch time, computes an
//! [`ApprovalVerdict`], emits a
//! `SessionEvent::MetadataMounted` event under
//! `approval/verdict`, and blocks Tier 3 actions via
//! `MiddlewareVerdict::Block` with `BlockCode::ApprovalRequired`.
//!
//! ## Scope (MVP)
//!
//! - Pattern-based `ActionKind` classification by tool name prefix
//! - Hand-coded tier rules with early returns (no
//!   `Hexavalent::or`-style composition)
//! - Emits an observable verdict on every action for audit replay
//! - Blocks Tier 3 with a structured rationale
//! - **Not yet integrated** with `ix_context::ContextBundle` for real
//!   blast-radius computation — MVP uses a constant "in-project" score.
//!   The integration lives behind a `from_context` constructor that
//!   future consumers will use when they have a bundle available.
//!
//! ## Not yet
//!
//! - Real `ContextBundle`-backed blast radius (v2)
//! - `ix-code::trajectory` volatility risk priors (v2)
//! - `ix-code::gates` hexavalent verdict integration (v2)
//! - Wiring into `ix-agent::registry_bridge::dispatch` — requires
//!   `dispatch_action` which is a separate step

pub mod classify;
pub mod middleware;
pub mod verdict;

pub use classify::{classify_action_kind, ActionKind};
pub use middleware::{ApprovalConfig, ApprovalMiddleware};
pub use verdict::{ApprovalVerdict, BlastRadius, Evidence, Tier};
