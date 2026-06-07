//! Live end-to-end test for the `ix_nl_to_pipeline` MCP tool.
//!
//! `#[ignore]`d: it shells out to the sibling `ix` binary and calls the
//! Anthropic Messages API (needs `ANTHROPIC_API_KEY` + network + a built `ix`).
//! Run manually:
//!
//! ```text
//! cargo build -p ix-skill          # ensure target/<profile>/ix exists
//! cargo test -p ix-agent --test nl_to_pipeline_live -- --ignored --nocapture
//! ```

use ix_agent::tools::ToolRegistry;
use serde_json::json;

#[test]
#[ignore = "live: shells `ix` + calls the Anthropic API"]
fn nl_to_pipeline_compiles_in_domain_request() {
    let reg = ToolRegistry::new();
    let out = reg
        .call(
            "ix_nl_to_pipeline",
            json!({ "sentence": "compute summary statistics on the numbers 1 2 3 4 5" }),
        )
        .expect("ix_nl_to_pipeline call returns Ok");
    eprintln!("ix_nl_to_pipeline → {out}");
    let status = out["status"].as_str().unwrap_or("");
    assert!(
        matches!(status, "compiled" | "ok"),
        "expected compiled/ok for an in-domain request, got {status:?}: {out}"
    );
}

#[test]
#[ignore = "live: shells `ix` + calls the Anthropic API"]
fn nl_to_pipeline_refuses_out_of_domain() {
    let reg = ToolRegistry::new();
    let out = reg
        .call(
            "ix_nl_to_pipeline",
            json!({ "sentence": "scrape a news website and email me a summary" }),
        )
        .expect("ix_nl_to_pipeline call returns Ok");
    eprintln!("ix_nl_to_pipeline (ood) → {out}");
    assert_eq!(
        out["status"].as_str(),
        Some("out_of_domain"),
        "out-of-domain request must be refused, not confabulated: {out}"
    );
}
