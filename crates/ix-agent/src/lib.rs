//! Library interface of `ix-agent` — the MCP server that exposes ix
//! algorithms and governance primitives as Claude Code tools.
//!
//! The binary entry point lives in `main.rs`; tests and external consumers
//! reach into these modules to inspect the tool registry, drive the bridge
//! between `ix-registry` and MCP, and call handlers directly.

pub mod acoustic_tune;
pub mod demo;
pub mod eval;
pub mod flywheel;
pub mod handlers;
/// `ix_maintain_gate` MCP tool (governance verdict via ix-duck). Feature-gated — pulls
/// bundled DuckDB; off by default. See docs/adr/0001-ixql-duckdb-integration-via-mcp-seam.md.
#[cfg(feature = "maintain-gate")]
pub mod maintain_gate;
pub mod ml_pipeline;
pub mod projection;
pub mod registry_bridge;
pub mod scopes;
pub mod server_context;
pub mod skills;
pub mod tools;
pub mod triage;
