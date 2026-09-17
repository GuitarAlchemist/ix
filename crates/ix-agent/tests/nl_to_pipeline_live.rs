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

/// The end-to-end compile path the MCP tool used to cover, exercised where it
/// still runs: the `ix` CLI. `#[ignore]`d — it needs a built `ix`, an
/// `ANTHROPIC_API_KEY` and the network. Run manually:
///
/// ```text
/// cargo build -p ix-skill
/// cargo test -p ix-agent --test nl_to_pipeline_live -- --ignored --nocapture
/// ```
#[test]
#[ignore = "live: runs the `ix` CLI + calls the Anthropic API"]
fn ix_cli_pipeline_compile_handles_in_and_out_of_domain() {
    let exe = std::env::current_exe().expect("test exe");
    let bin = exe
        .parent()
        .and_then(|deps| deps.parent())
        .map(|profile| profile.join(if cfg!(windows) { "ix.exe" } else { "ix" }))
        .expect("profile dir");
    assert!(
        bin.is_file(),
        "build the CLI first: cargo build -p ix-skill ({} missing)",
        bin.display()
    );
    let compile = |sentence: &str| -> serde_json::Value {
        let out = std::process::Command::new(&bin)
            .args(["--format", "json", "pipeline", "compile", sentence])
            .output()
            .expect("spawn ix");
        let stdout = String::from_utf8_lossy(&out.stdout);
        serde_json::from_str(stdout.trim())
            .unwrap_or_else(|e| panic!("expected JSON from ix, got {stdout:?} ({e})"))
    };

    let in_domain = compile("compute summary statistics on the numbers 1 2 3 4 5");
    let status = in_domain["status"].as_str().unwrap_or("");
    assert!(
        matches!(status, "compiled" | "ok"),
        "expected compiled/ok for an in-domain request, got {status:?}: {in_domain}"
    );

    let out_of_domain = compile("scrape a news website and email me a summary");
    assert_eq!(
        out_of_domain["status"].as_str(),
        Some("out_of_domain"),
        "out-of-domain request must be refused, not confabulated: {out_of_domain}"
    );
}
