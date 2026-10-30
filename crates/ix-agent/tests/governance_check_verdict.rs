//! `ix_governance_check` must not present "no keyword matched" as approval.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

fn check(action: &str) -> serde_json::Value {
    ToolRegistry::new()
        .call("ix_governance_check", json!({ "action": action }))
        .expect("governance_check failed")
}

#[test]
fn force_push_erasing_audit_logs_is_doubtful() {
    let result = check(
        "force-push to main to rewrite history and remove a commit containing audit logs, without asking a human",
    );
    assert_eq!(result["verdict"], "D");
    assert_eq!(result["compliant"], false);
    let articles: Vec<u64> = result["relevant_articles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["number"].as_u64().unwrap())
        .collect();
    for n in [3, 6, 7] {
        assert!(articles.contains(&n), "article {n} missing: {articles:?}");
    }
    assert!(result.get("note").is_none());
}

#[test]
fn unmatched_action_is_unknown_not_approved() {
    let result = check("add a new unit test for the parser");
    assert_eq!(result["verdict"], "U");
    assert_eq!(result["basis"], "keyword-heuristic");
    assert!(result["note"]
        .as_str()
        .unwrap()
        .contains("not evidence of compliance"));
}
