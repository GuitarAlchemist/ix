//! Smoke tests for `ix_git_log` against the ix workspace itself.
//!
//! These tests run the tool on this very repository, so the output
//! is whatever `git log` is producing *right now*. We can't assert
//! on absolute commit counts (they grow over time) — instead we
//! assert the *shape* and that the total is internally consistent
//! between `commits` and `sum(series)`.

use ix_agent::tools::ToolRegistry;
use serde_json::json;

fn run_git_log(args: serde_json::Value) -> Result<serde_json::Value, String> {
    ToolRegistry::new().call("ix_git_log", args)
}

#[test]
fn produces_90_day_series_for_ix_agent() {
    let result = run_git_log(json!({
        "path": "crates/ix-agent",
        "since_days": 90,
        "bucket": "day"
    }))
    .expect("ix_git_log failed");

    assert_eq!(result["path"], "crates/ix-agent");
    assert_eq!(result["bucket"], "day");
    assert_eq!(result["window_days"], 90);
    assert_eq!(result["n_buckets"], 90);

    let series = result["series"]
        .as_array()
        .expect("series")
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(series.len(), 90);

    let dates = result["dates"].as_array().expect("dates");
    assert_eq!(dates.len(), 90);

    let total: f64 = series.iter().sum();
    let reported = result["commits"].as_u64().unwrap() as f64;
    // series only contains commits that fell inside the window; the
    // reported `commits` is the raw git log count, which for a 90-day
    // window should match exactly.
    assert!(
        (total - reported).abs() < 0.5,
        "series sum ({total}) should match reported commits ({reported})"
    );

    // Every bucket must be a non-negative integer-valued f64.
    for (i, v) in series.iter().enumerate() {
        assert!(*v >= 0.0, "series[{i}] = {v} must be non-negative");
        assert_eq!(v.fract(), 0.0, "series[{i}] = {v} must be an integer");
    }
}

#[test]
fn weekly_bucket_packs_same_total() {
    let daily = run_git_log(json!({
        "path": "crates/ix-agent",
        "since_days": 90,
        "bucket": "day"
    }))
    .expect("daily");
    let weekly = run_git_log(json!({
        "path": "crates/ix-agent",
        "since_days": 90,
        "bucket": "week"
    }))
    .expect("weekly");

    let daily_total: f64 = daily["series"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .sum();
    let weekly_total: f64 = weekly["series"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .sum();
    assert!(
        (daily_total - weekly_total).abs() < 0.5,
        "daily total ({daily_total}) must equal weekly total ({weekly_total})"
    );
    assert_eq!(weekly["n_buckets"], 13); // ceil(90/7) = 13
}

#[test]
fn rejects_path_traversal() {
    let err = run_git_log(json!({
        "path": "../../../etc/passwd",
        "since_days": 90
    }))
    .expect_err("path traversal must reject");
    assert!(
        err.contains("path") && err.contains(".."),
        "expected path-safety error mentioning '..', got: {err}"
    );
}

#[test]
fn rejects_absolute_path() {
    let err = run_git_log(json!({
        "path": "/etc/passwd",
        "since_days": 90
    }))
    .expect_err("absolute path must reject");
    assert!(
        err.contains("path"),
        "expected path-safety error, got: {err}"
    );
}

#[test]
fn rejects_shell_metacharacters() {
    let err = run_git_log(json!({
        "path": "crates/ix-agent; rm -rf /",
        "since_days": 90
    }))
    .expect_err("shell metacharacters must reject");
    assert!(
        err.contains("path"),
        "expected path-safety error, got: {err}"
    );
}

#[test]
fn rejects_out_of_range_since_days() {
    let err = run_git_log(json!({
        "path": "crates/ix-agent",
        "since_days": 99999
    }))
    .expect_err("since_days bounds must reject");
    assert!(
        err.contains("since_days"),
        "expected since_days error, got: {err}"
    );
}

#[test]
fn rejects_unknown_bucket() {
    let err = run_git_log(json!({
        "path": "crates/ix-agent",
        "bucket": "century"
    }))
    .expect_err("unknown bucket must reject");
    assert!(
        err.contains("bucket"),
        "expected bucket error, got: {err}"
    );
}
