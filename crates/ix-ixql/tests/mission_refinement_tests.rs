//! First tracer: a missing refinement handler must not masquerade as a scope decision.
use std::sync::Arc;
use ix_ixql::{Executor, MemoryHost};

#[test]
fn root_git_add_is_refused_with_a_scope_reason_before_any_effect() {
    let host = Arc::new(MemoryHost::frozen());
    let executor = Executor::new(host);
    let error = executor.run_source(
        r#"proposal <- { contract_version: "0.1.0", function_version: "0.1.0",
             implementation_sha: "0123456789012345678901234567890123456789",
             mission_id: "m-1", attempt_id: "a-1", revision: "r-1",
             refusal_id: "ref-1", owner: "Blue", paths: ["."] }
           → mission.refine_scope"#,
    ).expect_err("root staging must be refused");
    assert!(error.to_string().contains("ScopeTooBroad"), "{error}");
}
