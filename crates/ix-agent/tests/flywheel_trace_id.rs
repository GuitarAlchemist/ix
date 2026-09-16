//! `ix_session_flywheel_export` names its output `<trace_dir>/<trace_id>.json`.
//! A `trace_id` that is a path (absolute, or with `..` or separators) must be
//! refused through the MCP entry point, before anything is created or written.

use ix_agent::registry_bridge::shared_loop_detector;
use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::json;

#[test]
fn flywheel_export_refuses_a_trace_id_that_is_a_path() {
    let (ctx, _rx) = ServerContext::new();
    let registry = ToolRegistry::new();
    let call = |args: serde_json::Value| {
        shared_loop_detector().clear_key("ix_session_flywheel_export");
        registry.call_with_ctx("ix_session_flywheel_export", args, &ctx)
    };

    let work = tempfile::tempdir().unwrap();
    let log_path = work.path().join("session.jsonl");
    drop(ix_session::SessionLog::open(&log_path).unwrap());
    let log = log_path.to_str().unwrap();
    let trace_dir = work.path().join("traces");
    let trace_dir_arg = trace_dir.to_str().unwrap();

    // Existing files the trace id tries to replace, beside and outside trace_dir.
    let outside = tempfile::tempdir().unwrap();
    let absolute_target = outside.path().join("settings.json");
    std::fs::write(&absolute_target, "KEEP").unwrap();
    let sibling_target = work.path().join("settings.json");
    std::fs::write(&sibling_target, "KEEP").unwrap();

    let absolute_id = outside.path().join("settings");
    for id in [absolute_id.to_str().unwrap(), "../settings", "sub/../../settings"] {
        let err = call(json!({ "session_log": log, "trace_dir": trace_dir_arg, "trace_id": id }))
            .expect_err("a path-shaped trace_id must be refused");
        assert!(err.contains("is not a plain file name"), "{id}: {err}");
    }
    assert_eq!(std::fs::read_to_string(&absolute_target).unwrap(), "KEEP");
    assert_eq!(std::fs::read_to_string(&sibling_target).unwrap(), "KEEP");
    assert!(!trace_dir.exists(), "a refused export must not create trace_dir");

    // A plain trace id still exports into trace_dir.
    let out = call(json!({ "session_log": log, "trace_dir": trace_dir_arg, "trace_id": "run-1" }))
        .expect("a plain trace_id must export");
    let written = std::path::PathBuf::from(out["written"].as_str().unwrap());
    assert_eq!(written, trace_dir.join("run-1.json"));
    assert!(written.exists(), "{out}");
}
