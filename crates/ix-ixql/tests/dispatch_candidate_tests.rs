//! Candidate-only replay through the existing parser and executor.
use std::sync::Arc;
use ix_ixql::{Executor, MemoryHost};
use serde_json::json;

#[test]
fn dispatch_candidate_never_admits_a_live_send() {
    let host = Arc::new(MemoryHost::frozen());
    host.seed("state/dispatch/request.json", json!({}));
    let executor = Executor::new(host);
    let out = executor.run_source(
        r#"candidate <- ix.io.read("state/dispatch/request.json") → mission.dispatch_candidate"#,
    ).unwrap();
    let candidate = out.binding("candidate").unwrap();
    assert_eq!(candidate["live_dispatch_available"], false);
    assert_eq!(candidate["requires_runner_admission"], true);
    assert!(out.writes.is_empty() && out.compound.is_empty());
}
