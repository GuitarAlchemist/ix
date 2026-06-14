//! Evaluation metrics surfaced as catalog skills.
//!
//! These are pure functions with their own unit tests, kept in the agent (beta)
//! layer rather than a stable library crate: the workspace shares one `0.1.0`
//! version, so adding a `pub fn` to a stable crate (ix-unsupervised / ix-ensemble)
//! trips the stable-surface gate. Housing them here lets the thinking-machine
//! catalog expose clustering-evaluation and model-explainability skills without
//! a multi-PR version dance. Each is promotable to a library crate if a second
//! (non-agent) consumer appears.

pub mod permutation_importance;
pub mod silhouette;
