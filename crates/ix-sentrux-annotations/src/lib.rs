//! Bridge: sentrux structural-rule findings → AI annotation contract.
//!
//! Sentrux is the GuitarAlchemist ecosystem's realtime structural-quality
//! sensor. It runs deterministic architectural rules over a workspace
//! (`max_fn_lines`, cycle detection, import-graph redundancy, …) and reports
//! violations through an MCP server.
//!
//! This crate converts each sentrux violation into an
//! [`ix_ai_annotations::Annotation`] conforming to the
//! `ai-annotation-v1` schema. The bridge is the **machine-verifier**
//! counterpart to AI/human-authored annotations: every sentrux violation
//! becomes an `F` (False) annotation with `certainty = detected-by-sentrux`
//! and `source.author = sentrux`. That closes the
//! claim → verify → promote-or-demote loop.
//!
//! Two emit modes:
//! - `Sidecar` — JSONL stream at `state/quality/ai-annotations-sentrux.jsonl`
//!   (the reconciler picks these up alongside other authors).
//! - `Inline` — patches the source file with an `// @ai:smell …` comment
//!   above the violating function (skip if the same sentrux annotation
//!   already exists at that location).
//!
//! Transport: the bridge spawns `sentrux.exe mcp` as a one-shot stdio
//! child and runs `initialize → notifications/initialized → scan →
//! check_rules`, parsing JSON-RPC line-delimited frames. The wire protocol
//! mirrors the ga-react-components Vite dev-server bridge.

pub mod convert;
pub mod emit;
pub mod mcp_bridge;
pub mod rules_response;

pub use convert::violation_to_annotation;
pub use emit::{emit_inline, emit_sidecar, EmitOutcome};
pub use mcp_bridge::{run_sentrux_check, SentruxConfig};
pub use rules_response::{parse_check_rules_response, RuleViolation, RulesReport};

/// Sentinel `source.author` value used for every annotation this bridge
/// emits.
pub const SENTRUX_AUTHOR: &str = "sentrux";

/// Default sidecar JSONL output path (relative to a workspace root).
pub const DEFAULT_SIDECAR_PATH: &str = "state/quality/ai-annotations-sentrux.jsonl";

/// Default sentrux executable path on the developer workstation.
pub const DEFAULT_SENTRUX_EXE: &str = "C:/Users/spare/bin/sentrux.exe";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("sentrux binary not found at {0}")]
    SentruxMissing(String),
    #[error("sentrux exited (code {code:?}) before responding; stderr tail: {stderr}")]
    SentruxExitedEarly { code: Option<i32>, stderr: String },
    #[error("sentrux call timed out after {0}ms")]
    Timeout(u64),
    #[error("sentrux returned JSON-RPC error: {0}")]
    RpcError(String),
    #[error("sentrux response was not understood: {0}")]
    BadResponse(String),
}
