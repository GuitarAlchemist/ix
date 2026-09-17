//! MCP-surface tests for the `ix_nl_to_pipeline` tool.
//!
//! The tool shells out to the sibling `ix` binary, which calls an LLM provider
//! API and, with `run: true`, executes the compiled pipeline. `ix-approval`
//! classifies it `ShellCommand` (Tier 3), and Tier 3 has no approval path yet
//! (ix#350), so every MCP call is refused before anything is spawned. These
//! tests pin that refusal; the in-domain / out-of-domain behaviour of the
//! compiler itself is exercised through the `ix pipeline compile` CLI.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

fn assert_refused_by_approval_gate(args: serde_json::Value) {
    let err = ToolRegistry::new()
        .call("ix_nl_to_pipeline", args)
        .expect_err("ix_nl_to_pipeline is Tier 3 and must be refused");
    assert!(
        err.starts_with("ix_approval: action blocked (ApprovalRequired)"),
        "expected an approval refusal, got: {err}"
    );
}

#[test]
fn nl_to_pipeline_in_domain_request_is_refused_by_the_approval_gate() {
    assert_refused_by_approval_gate(
        json!({ "sentence": "compute summary statistics on the numbers 1 2 3 4 5" }),
    );
}

#[test]
fn nl_to_pipeline_run_request_is_refused_by_the_approval_gate() {
    assert_refused_by_approval_gate(
        json!({ "sentence": "scrape a news website and email me a summary", "run": true }),
    );
}
