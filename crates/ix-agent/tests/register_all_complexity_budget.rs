//! Regression gate for the P0.1 register_all decomposition.
//!
//! Before P0.1 landed, `register_all` in `crates/ix-agent/src/tools.rs`
//! was a single monolithic method with cyclomatic complexity 108. The
//! refactor split it into sub-methods and extracted large JSON schemas
//! into standalone `schema_*` helpers, bringing the workspace max down
//! to 22.
//!
//! This test pins the max for every `register_*` method in tools.rs
//! via the live `ix_code_analyze` tool, so the next structural drift
//! (someone stuffing a new 15-field tool into an existing sub-method
//! without extracting its schema) trips CI rather than silently
//! re-inflating complexity.
//!
//! The budget is deliberately wider than the current ceiling (currently
//! 22; budget is 30) so small additions don't require touching this
//! test. A reviewer who bumps the budget should have a concrete reason,
//! not just inertia.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

/// Maximum allowed cyclomatic complexity for any `register_*` method
/// in `tools.rs`. Current ceiling after P0.1 is 22 — this budget
/// gives ~35% headroom. Only bump this if you have a real reason
/// (new section of tools) and document it in the commit.
const REGISTER_FN_BUDGET: i64 = 30;

#[test]
fn no_register_method_exceeds_complexity_budget() {
    let result = ToolRegistry::new()
        .call(
            "ix_code_analyze",
            json!({
                "path": concat!(env!("CARGO_MANIFEST_DIR"), "/src/tools.rs"),
                "operation": "complexity"
            }),
        )
        .expect("ix_code_analyze complexity call should succeed");

    let functions = result["functions"]
        .as_array()
        .expect("complexity result must have a 'functions' array");

    let mut over_budget: Vec<(String, i64)> = Vec::new();
    let mut max_seen: (String, i64) = ("<none>".into(), 0);

    for f in functions {
        let name = f["name"].as_str().unwrap_or("");
        if !name.starts_with("register_") {
            continue;
        }
        let cyc = f["cyclomatic"].as_f64().unwrap_or(0.0) as i64;
        if cyc > max_seen.1 {
            max_seen = (name.to_string(), cyc);
        }
        if cyc > REGISTER_FN_BUDGET {
            over_budget.push((name.to_string(), cyc));
        }
    }

    assert!(
        !over_budget.is_empty() || max_seen.1 > 0,
        "expected at least one register_* method in tools.rs; got none"
    );

    if !over_budget.is_empty() {
        let details = over_budget
            .iter()
            .map(|(n, c)| format!("  {n}: {c}"))
            .collect::<Vec<_>>()
            .join("\n");
        panic!(
            "register_* methods exceed the P0.1 complexity budget of {REGISTER_FN_BUDGET}:\n\
             {details}\n\n\
             To fix: split the offending method into sub-methods (see register_all_advanced \
             or register_all_core for the pattern), or extract the largest tool's JSON schema \
             into a standalone schema_ix_*() function (see schema_ix_supervised for the \
             pattern). The current max before this regression was {}: {}.",
            max_seen.0, max_seen.1
        );
    }
}
