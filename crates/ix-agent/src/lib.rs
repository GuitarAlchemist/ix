//! Library interface of `ix-agent` — the MCP server that exposes ix
//! algorithms and governance primitives as Claude Code tools.
//!
//! The binary entry point lives in `main.rs`; tests and external consumers
//! reach into these modules to inspect the tool registry, drive the bridge
//! between `ix-registry` and MCP, and call handlers directly.

pub mod demo;
pub mod flywheel;
pub mod handlers;
pub mod ml_pipeline;
pub mod registry_bridge;
pub mod server_context;
pub mod skills;
pub mod tools;
pub mod triage;
