//! Per-consumer scope filter for the MCP tool surface.
//!
//! `ix-agent` advertises ~68 tools by default. Real downstream consumers
//! (agent-blackbox, Demerzel, Hari) only depend on a tiny subset each.
//! Without scopes, every consumer sees the full surface, which:
//!
//! 1. Makes the per-consumer contract impossible to reason about
//!    (which tools is each consumer actually relying on?).
//! 2. Lets a tool rename or removal silently break a consumer that
//!    nobody knew was depending on it.
//! 3. Bloats each consumer's tool-list cognitive surface.
//!
//! # Mechanism
//!
//! Server-side filter. The client picks a scope by either:
//!
//! - Setting the `IX_MCP_SCOPE` environment variable before launching
//!   `ix-mcp` (most common — stdio MCP runs one process per session).
//! - Passing a `scope` field inside the `initialize` request's
//!   `clientInfo` (e.g. `{"clientInfo": {"name": "...", "scope":
//!   "agent-blackbox"}}`).
//!
//! `tools/list` then advertises only the subset declared in [`SCOPES`]
//! for the active scope. `tools/call` is **not** gated — once a client
//! discovers a tool name (out-of-band or by switching scope), it can
//! always invoke it. The scope is an *advertisement filter*, not a
//! capability boundary. Layering real auth on top is wave 2.
//!
//! # Backward compatibility
//!
//! Connecting **without** a scope (env unset, no `clientInfo.scope`)
//! yields the full tool surface — the same behavior every existing
//! consumer sees today. This is the [`Scope::Default`] case.
//!
//! # Adding a new scope
//!
//! 1. Add a variant to [`Scope`].
//! 2. Add a `(name, &[tool_names])` row to [`SCOPES`].
//! 3. The unit test `scope_subset_of_default` will assert that every
//!    tool in the new scope also exists in the default namespace.
//!
//! See `crates/ix-agent/README.md` for the consumer-facing docs.

/// Named scopes a client may request.
///
/// `Default` is the catch-all: every registered tool is advertised.
/// Named variants advertise only the tools listed for that scope in
/// [`SCOPES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Full tool surface — backward-compat for existing consumers.
    Default,
    /// Curated subset used by `agent-blackbox`.
    AgentBlackbox,
}

impl Scope {
    /// Parse a scope name (case-insensitive). Returns `None` for
    /// unrecognized values so callers can fall back to `Default` with
    /// a warning instead of silently filtering tools.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "" | "default" | "full" | "all" => Some(Self::Default),
            "agent-blackbox" | "agent_blackbox" | "blackbox" => Some(Self::AgentBlackbox),
            _ => None,
        }
    }

    /// Resolve the active scope from environment + optional client hint.
    ///
    /// Precedence (highest first):
    /// 1. `client_hint` — value from `initialize`'s
    ///    `params.clientInfo.scope` (if any).
    /// 2. `IX_MCP_SCOPE` env var.
    /// 3. [`Scope::Default`].
    ///
    /// Unrecognized names log a warning to stderr and fall back to
    /// `Default` (fail-open: never silently hide tools from a confused
    /// caller).
    pub fn resolve(client_hint: Option<&str>) -> Self {
        let raw = client_hint
            .map(str::to_string)
            .or_else(|| std::env::var("IX_MCP_SCOPE").ok())
            .unwrap_or_default();

        if raw.is_empty() {
            return Self::Default;
        }

        match Self::from_name(&raw) {
            Some(s) => s,
            None => {
                eprintln!(
                    "[ix-mcp] WARN unknown scope '{}' — advertising full tool surface (Default)",
                    raw
                );
                Self::Default
            }
        }
    }

    /// True iff this scope advertises the tool with the given MCP name.
    /// `Default` returns true for every tool.
    pub fn allows(&self, tool_name: &str) -> bool {
        match self {
            Self::Default => true,
            other => SCOPES
                .iter()
                .find(|(s, _)| *s == *other)
                .map(|(_, allowed)| allowed.contains(&tool_name))
                .unwrap_or(false),
        }
    }
}

/// The agent-blackbox tool subset.
///
/// Identified by grepping `cli/agent_blackbox.py`,
/// `templates/workflows/agent-blackbox.yml`, and
/// `docs/ix-real-problems-plan.md` in the `agent-blackbox` repo. The
/// repo today shells out to standalone `ix-*` CLI binaries (e.g.
/// `ix-blast-radius`), but the workflow plan calls out the MCP-tool
/// equivalents it will move to:
///
/// | Plan reference          | MCP tool on ix-agent     |
/// |-------------------------|--------------------------|
/// | `ix_blast_radius` field | `ix_code_analyze`        |
/// | `ix_code_metrics` field | `ix_code_smells`         |
/// | code catalog browsing   | `ix_code_catalog`        |
/// | autoresearch run        | `ix_autoresearch_run`    |
/// | autoresearch path nav   | `ix_grothendieck_delta`  |
/// | autoresearch path nav   | `ix_grothendieck_nearby` |
/// | autoresearch path nav   | `ix_grothendieck_path`   |
/// | quality-trend reader    | `ix_catalog_list`        |
/// | dep audit               | `ix_cargo_deps`          |
/// | history audit           | `ix_git_log`             |
///
/// Total: 10 tools. Kept at or below the unit-test ceiling of 10.
const AGENT_BLACKBOX_TOOLS: &[&str] = &[
    "ix_code_analyze",
    "ix_code_smells",
    "ix_code_catalog",
    "ix_autoresearch_run",
    "ix_grothendieck_delta",
    "ix_grothendieck_nearby",
    "ix_grothendieck_path",
    "ix_catalog_list",
    "ix_cargo_deps",
    "ix_git_log",
];

/// Static scope → allowed-tool-names table. Adding a scope is a
/// one-line change here plus a `Scope` variant.
pub const SCOPES: &[(Scope, &[&str])] = &[(Scope::AgentBlackbox, AGENT_BLACKBOX_TOOLS)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_allows_everything() {
        let s = Scope::Default;
        assert!(s.allows("ix_stats"));
        assert!(s.allows("ix_made_up_tool_name"));
    }

    #[test]
    fn agent_blackbox_subset() {
        let s = Scope::AgentBlackbox;
        assert!(s.allows("ix_code_analyze"));
        assert!(s.allows("ix_git_log"));
        assert!(!s.allows("ix_stats"));
        assert!(!s.allows("ix_fft"));
    }

    #[test]
    fn from_name_aliases() {
        assert_eq!(Scope::from_name("default"), Some(Scope::Default));
        assert_eq!(Scope::from_name(""), Some(Scope::Default));
        assert_eq!(
            Scope::from_name("AGENT-BLACKBOX"),
            Some(Scope::AgentBlackbox)
        );
        assert_eq!(
            Scope::from_name("agent_blackbox"),
            Some(Scope::AgentBlackbox)
        );
        assert_eq!(Scope::from_name("blackbox"), Some(Scope::AgentBlackbox));
        assert_eq!(Scope::from_name("hari"), None);
    }

    #[test]
    fn resolve_precedence() {
        // Save/restore env state. We can't run in parallel with other
        // tests that touch IX_MCP_SCOPE, but this is the only one.
        let prev = std::env::var("IX_MCP_SCOPE").ok();
        std::env::remove_var("IX_MCP_SCOPE");

        assert_eq!(Scope::resolve(None), Scope::Default);
        assert_eq!(Scope::resolve(Some("agent-blackbox")), Scope::AgentBlackbox);

        std::env::set_var("IX_MCP_SCOPE", "agent-blackbox");
        assert_eq!(Scope::resolve(None), Scope::AgentBlackbox);
        // client_hint wins over env.
        assert_eq!(Scope::resolve(Some("default")), Scope::Default);

        // Unknown name falls back to Default with a warning.
        std::env::set_var("IX_MCP_SCOPE", "not-a-real-scope");
        assert_eq!(Scope::resolve(None), Scope::Default);

        // Restore.
        match prev {
            Some(v) => std::env::set_var("IX_MCP_SCOPE", v),
            None => std::env::remove_var("IX_MCP_SCOPE"),
        }
    }
}
