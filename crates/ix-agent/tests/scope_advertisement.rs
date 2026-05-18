//! POC contract: `agent-blackbox` scope advertises a small, well-known
//! subset of the full tool surface.
//!
//! Two invariants are pinned by this test:
//!
//! 1. **Size ceiling**: `agent-blackbox` advertises ≤ 10 tools. Adding
//!    a new tool to the scope is a deliberate decision — if you're
//!    growing the surface, document why in the PR.
//! 2. **Subset of default**: every tool in `agent-blackbox` also exists
//!    in the default (full) namespace. This catches the regression
//!    where someone renames or removes a tool from default and forgets
//!    to update the scope table.
//!
//! Wave-2 scopes (Demerzel, Hari) will get their own size ceiling +
//! subset assertion in the same file. The 10-tool cap is per-scope, not
//! shared.

use ix_agent::scopes::Scope;
use ix_agent::tools::ToolRegistry;
use std::collections::HashSet;

const AGENT_BLACKBOX_CEILING: usize = 10;

#[test]
fn agent_blackbox_scope_is_small() {
    let reg = ToolRegistry::new();
    let listed = reg.list_scoped(Scope::AgentBlackbox);
    let tools = listed["tools"]
        .as_array()
        .expect("list_scoped returns { tools: [...] }");
    let count = tools.len();
    assert!(
        count > 0,
        "agent-blackbox scope advertised 0 tools — the scope table is probably empty or every \
         listed tool has been renamed in default. Check src/scopes.rs::AGENT_BLACKBOX_TOOLS."
    );
    assert!(
        count <= AGENT_BLACKBOX_CEILING,
        "agent-blackbox scope advertised {} tools but the ceiling is {}. \
         Either drop a tool from the scope or bump the ceiling with a \
         justification in the commit.",
        count,
        AGENT_BLACKBOX_CEILING
    );
}

#[test]
fn agent_blackbox_scope_is_strict_subset_of_default() {
    let reg = ToolRegistry::new();

    let default_names: HashSet<&'static str> = reg.tool_names().collect();

    let scoped = reg.list_scoped(Scope::AgentBlackbox);
    let scoped_tools = scoped["tools"].as_array().unwrap();

    let mut missing = Vec::new();
    for tool in scoped_tools {
        let name = tool["name"].as_str().unwrap();
        if !default_names.contains(name) {
            missing.push(name.to_string());
        }
    }

    assert!(
        missing.is_empty(),
        "agent-blackbox scope advertises tools that no longer exist in the default surface: {:?}. \
         Either restore the tool in src/tools.rs::register_* or remove it from \
         src/scopes.rs::AGENT_BLACKBOX_TOOLS.",
        missing
    );
}

#[test]
fn default_scope_matches_legacy_list() {
    // `list()` is the legacy entry point — assert it advertises the same
    // shape as `list_scoped(Default)` so we don't accidentally drift the
    // two surfaces apart.
    let reg = ToolRegistry::new();
    let legacy = reg.list();
    let scoped = reg.list_scoped(Scope::Default);
    assert_eq!(legacy, scoped);
}

#[test]
fn default_scope_is_strictly_larger_than_agent_blackbox() {
    let reg = ToolRegistry::new();
    let default_count = reg.list_scoped(Scope::Default)["tools"]
        .as_array()
        .unwrap()
        .len();
    let bb_count = reg.list_scoped(Scope::AgentBlackbox)["tools"]
        .as_array()
        .unwrap()
        .len();
    assert!(
        default_count > bb_count,
        "default scope ({} tools) should be strictly larger than agent-blackbox ({} tools); \
         otherwise the POC isn't actually narrowing the surface.",
        default_count,
        bb_count
    );
    // Visible when run with --nocapture; useful for the PR demo line.
    println!(
        "scope sizes: default={} tools, agent-blackbox={} tools",
        default_count, bb_count
    );
}
