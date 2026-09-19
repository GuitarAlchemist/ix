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
        "ix_pca",
        "ix_dbscan",
        "ix_eigen",
        "ix_silhouette",
        "ix_feature_importances",
        "ix_svd",
        "ix_gmm",
        "ix_wavelet_denoise",
        "ix_fir_filter",
        // ix#193 — pure state estimation over a caller-supplied series, no side effects.
        "ix_kalman",
        "ix_spectrogram",
        "ix_autocorrelation",
        "ix_analyze_reference",
        "ix_spectral_distance",
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
        // Pure correlation-mesh compute over caller-supplied series.
        "ix_mesh_correlate",
        // Reachability analysis of a caller-supplied Petri net (inline or PNML string).
        "ix_petri_analyze",
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
        // Assumption graph: scan @ai: annotations / replay a belief log, no writes.
        "ix_assumption_query",
        "ix_assumption_belief_at",
        "ix_assumption_drift",
        "ix_assumption_claims",
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
        // Manual (hand-registered) tools, gated since ix#350. Pure compute or
        // static data:
        "ix_tsne",
        "ix_autograd_run",
        "ix_grothendieck_delta",
        "ix_grothendieck_nearby",
        "ix_grothendieck_path",
        "ix_voicings_payload",
        "ix_catalog_list",
        "ix_code_catalog",
        "ix_grammar_catalog",
        "ix_rfc_catalog",
        // Manual tools that only read files (no writes, no processes). Each
        // confines the path its caller names through `path_confine`, as Tier 1
        // requires — see the module docs there.
        "ix_cargo_deps",
        "ix_pipeline_list",
        "ix_quality_gate_history",
        "ix_code_topology",
        "ix_annotations_scan",
        "ix_optick_search",
        "ix_ast_query",
        "ix_code_smells",
        // Asks the client LLM (MCP sampling) for a spec and validates it; runs nothing.
        "ix_pipeline_compile",
    ];

    // In-project edits — tools that write to state/ or emit trace data
    // inside the workspace.
    const EDIT_IN_PROJECT_TOOLS: &[&str] = &[
        "ix_demo",                    // writes demo outputs (images, videos) to project dirs
        "ix_trace_ingest",            // writes trace data
        "ix_governance_belief",       // writes to state/beliefs/
        "ix_governance_graph_rescan", // rebuilds graph state
        "ix_session_flywheel_export", // writes a GA Trace JSON file
        // `set` / `delete` mutate the process-wide cache that `ix_pipeline_run`
        // consults before dispatching an asset-backed step, so a write can stand
        // in for a later step's result. In-process state, not a file, but not
        // side-effect-free either; the classifier is name-only, so the whole
        // tool takes the writing kind (Tier 2, still auto-continues).
        "ix_cache",
        // Manual tools, gated since ix#350. Orchestrators: each inner tool call
        // is gated on its own; the outer call writes the step cache
        // (ix_pipeline_run) or may export a trace (ix_triage_session learn=true).
        "ix_pipeline_run",
        "ix_triage_session",
        // Writes run logs under a caller-chosen `state_dir`.
        "ix_autoresearch_run",
        // Spawn a subprocess with a fixed, read-only argument list (`git log`,
        // `ix pipeline hits`, `git cat-file`/`git status`). Not Tier 1 because
        // they start processes; not Tier 3 because Tier 3 is a hard refusal
        // (no approval path) and none of them takes a caller-chosen program or
        // writes. Tier 2 keeps them classified, logged and loop-detected.
        "ix_git_log",
        "ix_git_churn",
        "ix_thinker_hits",
        "ix_maintain_gate",
    ];

    // Gated tools — always Tier 3, which refuses the call. The tables exist
    // so a tool that shells out, fetches from the web, or writes outside the
    // workspace is classified explicitly instead of falling through to
    // `Unknown`.
    const SHELL_COMMAND_TOOLS: &[&str] = &[
        // Runs a caller-chosen executable (`sentrux_exe`) and writes
        // annotations to caller-chosen paths.
        "ix_sentrux_annotate",
        // Spawns `ix pipeline compile`, which calls an LLM provider API and,
        // with `run: true`, executes the compiled pipeline.
        "ix_nl_to_pipeline",
    ];
    const WEB_FETCH_TOOLS: &[&str] = &[];
    const EDIT_OUT_OF_PROJECT_TOOLS: &[&str] = &[];

    if READ_TOOLS.contains(&tool_name) {
        ActionKind::Read
    } else if EDIT_IN_PROJECT_TOOLS.contains(&tool_name) {
        ActionKind::EditInProject
    } else if SHELL_COMMAND_TOOLS.contains(&tool_name) {
        ActionKind::ShellCommand
    } else if WEB_FETCH_TOOLS.contains(&tool_name) {
        ActionKind::WebFetch
    } else if EDIT_OUT_OF_PROJECT_TOOLS.contains(&tool_name) {
        ActionKind::EditOutOfProject
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
        // ix_pca: pure deterministic ML compute (like ix_kmeans) — must classify
        // as Read so the approval middleware doesn't block the new MCP tool /
        // pipeline stage at Tier 3 (Codex P1 on #83: exposed-but-unusable).
        for tool in &[
            "ix_stats",
            "ix_context_walk",
            "ix_code_analyze",
            "ix_fft",
            "ix_pca",
            "ix_dbscan",
            "ix_eigen",
            "ix_silhouette",
            "ix_feature_importances",
            "ix_svd",
            "ix_gmm",
            "ix_wavelet_denoise",
            "ix_fir_filter",
            // ix#193: same "exposed-but-unusable" trap this test's header records for
            // #83 — registering ix_kalman in ToolRegistry is not enough, because an
            // unclassified tool falls through to Unknown and the middleware blocks it at
            // Tier 3. Verified: the end-to-end smoke test failed with
            // "ApprovalRequired (tier: tier_three)" until this entry was added.
            "ix_kalman",
            "ix_spectrogram",
            "ix_autocorrelation",
            "ix_analyze_reference",
            "ix_spectral_distance",
            // Same trap, found by the registry-wide parity check: all six were
            // registered and unit-tested, and refused on every MCP call.
            "ix_mesh_correlate",
            "ix_petri_analyze",
            "ix_assumption_query",
            "ix_assumption_belief_at",
            "ix_assumption_drift",
            "ix_assumption_claims",
            // Manual tools, gated since ix#350.
            "ix_tsne",
            "ix_cargo_deps",
            "ix_optick_search",
            "ix_pipeline_compile",
        ] {
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
            // Writes the in-process cache ix_pipeline_run reads (review on #345).
            "ix_cache",
            // Manual tools that spawn fixed read-only subprocesses or write run
            // state (ix#350) — never Tier 1.
            "ix_git_log",
            "ix_git_churn",
            "ix_thinker_hits",
            "ix_autoresearch_run",
            "ix_pipeline_run",
        ] {
            assert_eq!(
                classify_action_kind(tool),
                ActionKind::EditInProject,
                "expected {tool} to be EditInProject"
            );
        }
    }

    #[test]
    fn caller_chosen_process_tools_classify_as_shell_command() {
        for tool in &["ix_sentrux_annotate", "ix_nl_to_pipeline"] {
            assert_eq!(
                classify_action_kind(tool),
                ActionKind::ShellCommand,
                "expected {tool} to be ShellCommand"
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
