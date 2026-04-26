//! Deterministic classification of [`ix_agent_core::AgentAction`] into
//! high-level action kinds, via pattern-matching on the tool name.
//!
//! **Hand-coded, not LLM-inferred.** The brainstorm thesis: auto-mode's
//! 17% false-negative rate on blast-radius judgment comes from LLM
//! classification uncertainty. Replacing that with a fixed lookup
//! eliminates the uncertainty at the cost of requiring manual
//! registration of new tool kinds.

use serde::{Deserialize, Serialize};

/// Coarse-grained classification of what an action *does* to the
/// system. Drives tier assignment downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    /// Pure read — no side effects. Maps to Tier 1.
    Read,
    /// In-project edit within the workspace. Git provides the safety
    /// net. Maps to Tier 2 if the blast radius is small.
    EditInProject,
    /// Shell command. Maps to Tier 3 (requires approval) regardless of
    /// blast radius because shell arguments can escape the workspace.
    ShellCommand,
    /// Web fetch. Maps to Tier 3 (can exfiltrate data, can be used for
    /// prompt injection via fetched content).
    WebFetch,
    /// Edit targeting a path outside the workspace. Maps to Tier 3.
    EditOutOfProject,
    /// Unknown — the tool isn't in the classification table. Maps to
    /// Tier 3 conservatively. Add the tool to `classify_action_kind`
    /// to opt it into a lower tier.
    Unknown,
}

impl ActionKind {
    /// Short canonical name used in verdict rationale and session
    /// events.
    pub const fn name(self) -> &'static str {
        match self {
            ActionKind::Read => "read",
            ActionKind::EditInProject => "edit_in_project",
            ActionKind::ShellCommand => "shell_command",
            ActionKind::WebFetch => "web_fetch",
            ActionKind::EditOutOfProject => "edit_out_of_project",
            ActionKind::Unknown => "unknown",
        }
    }
}

/// Classify an MCP tool name into an [`ActionKind`] by pattern
/// matching. Returns [`ActionKind::Unknown`] for tools not in the
/// hand-coded table.
///
/// The table reflects the current ix-agent tool surface at 2026-04-11.
/// Adding a new tool requires editing this function (and the
/// corresponding test) — that's the point. Unknown tools default to
/// the most conservative tier.
///
/// # Example
///
/// ```
/// use ix_approval::{classify_action_kind, ActionKind};
/// assert_eq!(classify_action_kind("ix_stats"), ActionKind::Read);
/// assert_eq!(classify_action_kind("ix_context_walk"), ActionKind::Read);
/// assert_eq!(classify_action_kind("ix_trace_ingest"), ActionKind::EditInProject);
/// assert_eq!(classify_action_kind("ix_nope_not_a_tool"), ActionKind::Unknown);
/// ```
pub fn classify_action_kind(tool_name: &str) -> ActionKind {
    // Reads — pure analysis / query tools with no side effects.
    const READ_TOOLS: &[&str] = &[
        // Math & stats
        "ix_stats",
        "ix_distance",
        "ix_fft",
        "ix_linear_regression",
        "ix_kmeans",
        "ix_optimize",
        "ix_search",
        "ix_markov",
        "ix_viterbi",
        "ix_number_theory",
        "ix_rotation",
        "ix_sedenion",
        "ix_fractal",
        "ix_topo",
        "ix_category",
        "ix_graph",
        "ix_hyperloglog",
        "ix_bloom_filter",
        "ix_chaos_lyapunov",
        "ix_game_nash",
        "ix_grammar_weights",
        "ix_grammar_search",
        "ix_grammar_evolve",
        "ix_ml_pipeline",
        "ix_ml_predict",
        "ix_nn_forward",
        "ix_adversarial_fgsm",
        "ix_random_forest",
        "ix_gradient_boosting",
        "ix_supervised",
        "ix_bandit",
        "ix_evolution",
        // Code + context analysis
        "ix_code_analyze",
        "ix_context_walk",
        // Fuzzy distribution eval (deterministic, no side effects)
        "ix_fuzzy_eval",
        // Governance reads
        "ix_governance_check",
        "ix_governance_persona",
        "ix_governance_policy",
        "ix_governance_graph",
        "ix_explain_algorithm",
        "ix_federation_discover",
        // Federation bridges (data shape conversion, no side effects)
        "ix_ga_bridge",
        "ix_tars_bridge",
        // Pipeline info (no execution)
        "ix_pipeline",
        "ix_cache",
    ];

    // In-project edits — tools that write to state/ or emit trace data
    // inside the workspace.
    const EDIT_IN_PROJECT_TOOLS: &[&str] = &[
        "ix_demo",                    // writes demo outputs (images, videos) to project dirs
        "ix_trace_ingest",            // writes trace data
        "ix_governance_belief",       // writes to state/beliefs/
        "ix_governance_graph_rescan", // rebuilds graph state
        "ix_session_flywheel_export", // writes a GA Trace JSON file
    ];

    if READ_TOOLS.contains(&tool_name) {
        ActionKind::Read
    } else if EDIT_IN_PROJECT_TOOLS.contains(&tool_name) {
        ActionKind::EditInProject
    } else {
        ActionKind::Unknown
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_tools_classify_as_read() {
        for tool in &["ix_stats", "ix_context_walk", "ix_code_analyze", "ix_fft"] {
            assert_eq!(
                classify_action_kind(tool),
                ActionKind::Read,
                "expected {tool} to be Read"
            );
        }
    }

    #[test]
    fn edit_tools_classify_as_edit_in_project() {
        for tool in &[
            "ix_demo",
            "ix_trace_ingest",
            "ix_governance_belief",
            "ix_governance_graph_rescan",
        ] {
            assert_eq!(
                classify_action_kind(tool),
                ActionKind::EditInProject,
                "expected {tool} to be EditInProject"
            );
        }
    }

    #[test]
    fn unknown_tool_defaults_to_unknown_kind() {
        assert_eq!(
            classify_action_kind("ix_definitely_not_a_real_tool"),
            ActionKind::Unknown
        );
        assert_eq!(classify_action_kind("rm_rf_slash"), ActionKind::Unknown);
    }

    #[test]
    fn empty_tool_name_is_unknown() {
        assert_eq!(classify_action_kind(""), ActionKind::Unknown);
    }

    #[test]
    fn action_kind_names_are_stable() {
        assert_eq!(ActionKind::Read.name(), "read");
        assert_eq!(ActionKind::EditInProject.name(), "edit_in_project");
        assert_eq!(ActionKind::ShellCommand.name(), "shell_command");
        assert_eq!(ActionKind::WebFetch.name(), "web_fetch");
        assert_eq!(ActionKind::EditOutOfProject.name(), "edit_out_of_project");
        assert_eq!(ActionKind::Unknown.name(), "unknown");
    }

    #[test]
    fn action_kind_serde_snake_case() {
        assert_eq!(
            serde_json::to_string(&ActionKind::EditInProject).unwrap(),
            r#""edit_in_project""#
        );
        let back: ActionKind = serde_json::from_str(r#""shell_command""#).unwrap();
        assert_eq!(back, ActionKind::ShellCommand);
    }
}
