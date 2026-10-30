//! Registry-backed skill wrappers.
//!
//! Each sub-module annotates composite MCP handlers with `#[ix_skill]` so
//! they appear in the capability registry. The wrappers have shape
//! `fn(serde_json::Value) -> Result<serde_json::Value, String>` — the same
//! as the original MCP handlers — and carry the hand-written JSON schema
//! via the macro's `schema_fn = ...` argument.
//!
//! Migration plan: each batch moves a set of related tools from the manual
//! `ToolRegistry::register_all` list into a module here, then removes the
//! matching entries from `tools.rs`. The 43-tool parity test enforces that
//! every tool remains reachable during the transition.

pub mod acoustic_tune;
pub mod assumption_graph;
pub mod batch1;
pub mod batch2;
pub mod batch3;
pub mod duckdb;
pub mod knot;
pub mod origami;
pub mod petri;
pub mod prime_radiant;
pub mod wmux;
/// `stream.run`, registered in ix-stream, beside the graphs it runs. Nothing in ix-agent calls
/// it, so this re-export is what keeps its registration in every binary that links ix-agent.
pub use ix_stream::skill as stream;
