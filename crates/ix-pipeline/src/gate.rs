//! Execution-time gate over **resolved** stage args.
//!
//! `lower()` resolves `{"from": "stage[.key]"}` references to upstream outputs
//! only at execution time, so any check that inspects the spec *as written*
//! cannot see the value a stage will actually run with. A [`StageGate`] is
//! consulted inside each stage's compute closure — **after** reference
//! resolution and **before** the skill runs — closing the gap where a ref
//! could smuggle an unreviewed operation past a template-time check.
//!
//! `ix-pipeline` stays governance-agnostic: it defines the seam but ships no
//! policy. Callers inject whatever gate they want (e.g. `ix-skill` supplies a
//! Demerzel-constitution-backed gate); the default [`AllowAll`] is a no-op so
//! existing `lower()` callers are unaffected.

use serde_json::Value;
use std::sync::Arc;

/// Vets a stage's resolved args immediately before its skill executes.
///
/// Implementors return `Err(reason)` to abort the stage; the executor surfaces
/// it as a `PipelineError::ComputeError`. Must be `Send + Sync` because the
/// gate handle is cloned into every node's compute closure and the executor
/// runs independent levels in parallel.
pub trait StageGate: Send + Sync {
    /// Inspect `resolved_args` (post-`{"from"}`-resolution) for `skill` running
    /// as pipeline stage `stage_id`. `Ok(())` allows the stage; `Err(reason)`
    /// aborts it before the skill is invoked.
    fn check(&self, stage_id: &str, skill: &str, resolved_args: &Value) -> Result<(), String>;
}

/// The default no-op gate: every stage is allowed. Used by plain `lower()` so
/// pipelines that opt out of governance behave exactly as before.
pub struct AllowAll;

impl StageGate for AllowAll {
    fn check(&self, _stage_id: &str, _skill: &str, _resolved_args: &Value) -> Result<(), String> {
        Ok(())
    }
}

/// A shared, thread-safe gate handle baked into each node's compute closure.
pub type SharedGate = Arc<dyn StageGate>;

/// The default shared gate (`AllowAll`).
pub fn allow_all() -> SharedGate {
    Arc::new(AllowAll)
}
