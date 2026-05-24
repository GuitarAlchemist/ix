//! Smoke tests for `ix_git_churn` against the ix workspace itself.
//!
//! Same shape as `git_log_smoke`: we cannot assert on absolute file
//! counts or line totals (they shift with every commit), so the tests
//! pin the *invariants* — sort order, projection consistency, security
//! validation, and that the tool finds at least one file when given a
//! reasonable window.

use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};
use std::path::PathBuf;

fn workspace_root() -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace root
    p.display().to_string()
}

fn with_repo_root(mut args: Value) -> Value {
    if let Some(obj) = args.as_object_mut() {
        obj.insert("repo_root".into(), json!(workspace_root()));
    }
    args
}

fn run_git_churn(args: Value) -> Result<Value, String> {
    ToolRegistry::new().call("ix_git_churn", with_repo_root(args))
}

#[test]
fn returns_per_file_churn_for_default_window() {
    let result = run_git_churn(json!({})).expect("ix_git_churn failed");

    assert_eq!(result["window_days"], 30, "default window should be 30");
    assert_eq!(result["limit"], 50, "default limit should be 50");

    let files = result["files"].as_array().expect("files");
    let total = result["total_files"].as_u64().expect("total_files");
    assert!(
        !files.is_empty(),
        "ix workspace should have at least one churned file in the last 30 days, got 0 (total_files={total})"
    );
    assert!(
        files.len() <= 50,
        "files len ({}) must not exceed limit (50)",
        files.len()
    );
    assert!(
        (files.len() as u64) <= total,
        "files len ({}) cannot exceed total_files ({})",
        files.len(),
        total
    );

    // Each entry must carry the documented fields and non-negative counts.
    for (i, f) in files.iter().enumerate() {
        let path = f["path"].as_str().unwrap_or("");
        assert!(!path.is_empty(), "files[{i}].path missing");
        assert!(
            f["churn_count"].as_u64().is_some(),
            "files[{i}].churn_count missing or not u64"
        );
        assert!(
            f["lines_added"].as_u64().is_some(),
            "files[{i}].lines_added missing or not u64"
        );
        assert!(
            f["lines_deleted"].as_u64().is_some(),
            "files[{i}].lines_deleted missing or not u64"
        );
        let last = f["last_changed"].as_str().unwrap_or("");
        assert!(
            last.len() == 10 && last.as_bytes()[4] == b'-' && last.as_bytes()[7] == b'-',
            "files[{i}].last_changed = {last:?} should be YYYY-MM-DD"
        );
    }
}

#[test]
fn ranked_descending_by_churn_count() {
    let result = run_git_churn(json!({ "since_days": 90 })).expect("ix_git_churn failed");
    let files = result["files"].as_array().expect("files");
    let counts: Vec<u64> = files
        .iter()
        .map(|f| f["churn_count"].as_u64().unwrap())
        .collect();
    for i in 1..counts.len() {
        assert!(
            counts[i - 1] >= counts[i],
            "files must be sorted by churn_count desc; got [{i}]={} after [{}]={}",
            counts[i],
            i - 1,
            counts[i - 1]
        );
    }
}

#[test]
fn flat_projections_align_with_files() {
    let result =
        run_git_churn(json!({ "since_days": 30, "limit": 20 })).expect("ix_git_churn failed");
    let files = result["files"].as_array().expect("files");
    let paths = result["paths"].as_array().expect("paths");
    let churn = result["churn_counts"].as_array().expect("churn_counts");
    let added = result["lines_added"].as_array().expect("lines_added");
    let deleted = result["lines_deleted"].as_array().expect("lines_deleted");

    assert_eq!(files.len(), paths.len(), "paths len mismatch");
    assert_eq!(files.len(), churn.len(), "churn_counts len mismatch");
    assert_eq!(files.len(), added.len(), "lines_added len mismatch");
    assert_eq!(files.len(), deleted.len(), "lines_deleted len mismatch");

    for (i, f) in files.iter().enumerate() {
        assert_eq!(
            f["path"].as_str(),
            paths[i].as_str(),
            "paths[{i}] diverges from files[{i}].path"
        );
        assert_eq!(
            f["churn_count"].as_u64().map(|n| n as f64),
            churn[i].as_f64(),
            "churn_counts[{i}] diverges from files[{i}].churn_count"
        );
        assert_eq!(
            f["lines_added"].as_u64().map(|n| n as f64),
            added[i].as_f64(),
            "lines_added[{i}] diverges"
        );
        assert_eq!(
            f["lines_deleted"].as_u64().map(|n| n as f64),
            deleted[i].as_f64(),
            "lines_deleted[{i}] diverges"
        );
    }
}

#[test]
fn rejects_zero_since_days() {
    let err = run_git_churn(json!({ "since_days": 0 })).expect_err("since_days=0 must reject");
    assert!(
        err.contains("since_days"),
        "expected since_days error, got: {err}"
    );
}

#[test]
fn rejects_out_of_range_since_days() {
    let err =
        run_git_churn(json!({ "since_days": 99999 })).expect_err("since_days bounds must reject");
    assert!(
        err.contains("since_days"),
        "expected since_days error, got: {err}"
    );
}

#[test]
fn rejects_zero_limit() {
    let err = run_git_churn(json!({ "limit": 0 })).expect_err("limit=0 must reject");
    assert!(err.contains("limit"), "expected limit error, got: {err}");
}

#[test]
fn rejects_oversized_limit() {
    let err =
        run_git_churn(json!({ "limit": 1_000_000 })).expect_err("limit too large must reject");
    assert!(err.contains("limit"), "expected limit error, got: {err}");
}

#[test]
fn limit_truncates_but_total_files_does_not() {
    let small = run_git_churn(json!({ "since_days": 90, "limit": 5 })).expect("small limit");
    let big = run_git_churn(json!({ "since_days": 90, "limit": 1000 })).expect("big limit");

    assert!(
        small["files"].as_array().unwrap().len() <= 5,
        "files must respect limit=5"
    );
    assert_eq!(
        small["total_files"], big["total_files"],
        "total_files is the un-truncated count and should be limit-invariant"
    );
}
