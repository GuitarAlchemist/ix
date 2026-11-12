//! End-to-end smoke test for the `#[ix_tool]` macro + `linkme` + registry.
//!
//! Annotates three trivial functions, then verifies:
//!   1. They all appear in `IX_TOOLS` (sentinel filtered).
//!   2. Their `ToolDescriptor` metadata is correct.
//!   3. `invoke()` decodes args, calls the handler, and encodes the result.
//!   4. A `Result`-returning tool propagates `Err(_)` as `ToolError::Exec`.

use ix_registry::{by_domain, by_name, count, invoke, search, ToolError};
use ix_tool_macros::ix_tool;
use ix_types::{IxVector, Value};

/// Add two scalars.
#[ix_tool(domain = "canary")]
pub fn add(a: f64, b: f64) -> f64 {
    a + b
}

/// Sum the elements of a vector.
#[ix_tool(domain = "canary", governance = "deterministic")]
pub fn vec_sum(v: IxVector) -> f64 {
    v.0.sum()
}

/// Divide, returning `Err` on zero divisor (tests Result unwrapping).
#[ix_tool(domain = "canary")]
pub fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("division by zero".into())
    } else {
        Ok(a / b)
    }
}

#[test]
fn all_three_tools_are_registered() {
    // At least the 3 canary tools are present (the workspace may link
    // others in the future; assert a lower bound instead of equality).
    assert!(count() >= 3, "expected ≥ 3 tools, got {}", count());

    assert!(by_name("registry.canary.add").is_some());
    assert!(by_name("registry.canary.vec_sum").is_some());
    assert!(by_name("registry.canary.divide").is_some());
}

#[test]
fn sentinel_never_visible() {
    assert!(by_name("__sentinel__").is_none());
    assert!(ix_registry::all().all(|s| s.name != "__sentinel__"));
}

#[test]
fn descriptor_metadata_is_correct() {
    let desc = by_name("registry.canary.vec_sum").unwrap();
    assert_eq!(desc.domain, "canary");
    assert_eq!(desc.crate_name, "ix-registry");
    assert_eq!(desc.inputs.len(), 1);
    assert_eq!(desc.outputs.len(), 1);
    assert_eq!(desc.governance_tags, &["deterministic"]);
    assert!(desc.doc.contains("Sum the elements"));
}

#[test]
fn invoke_scalar_tool() {
    let out = invoke(
        "registry.canary.add",
        &[Value::Scalar(2.0), Value::Scalar(3.5)],
    )
    .unwrap();
    match out {
        Value::Scalar(x) => assert_eq!(x, 5.5),
        other => panic!("expected Scalar, got {other:?}"),
    }
}

#[test]
fn invoke_vector_tool() {
    let v = IxVector::new(vec![1.0, 2.0, 3.0, 4.0]);
    let out = invoke(
        "registry.canary.vec_sum",
        &[ix_types::IntoValue::into_value(v)],
    )
    .unwrap();
    match out {
        Value::Scalar(x) => assert_eq!(x, 10.0),
        other => panic!("expected Scalar, got {other:?}"),
    }
}

#[test]
fn invoke_wrong_arity_returns_error() {
    let err = invoke("registry.canary.add", &[Value::Scalar(1.0)]).unwrap_err();
    match err {
        ToolError::Arity { expected, actual } => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        }
        other => panic!("expected Arity error, got {other:?}"),
    }
}

#[test]
fn invoke_wrong_type_returns_type_error() {
    let err = invoke(
        "registry.canary.add",
        &[Value::Text("nope".into()), Value::Scalar(1.0)],
    )
    .unwrap_err();
    assert!(matches!(err, ToolError::Type { arg_index: 0, .. }));
}

#[test]
fn result_err_propagates_as_exec() {
    let err = invoke(
        "registry.canary.divide",
        &[Value::Scalar(1.0), Value::Scalar(0.0)],
    )
    .unwrap_err();
    match err {
        ToolError::Exec(msg) => assert!(msg.contains("division by zero")),
        other => panic!("expected Exec error, got {other:?}"),
    }
}

#[test]
fn by_domain_groups_canary_tools() {
    let canary = by_domain("canary");
    assert_eq!(canary.len(), 3);
}

#[test]
fn search_matches_doc_and_name() {
    let hits = search("vec_sum");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].name, "registry.canary.vec_sum");

    let hits_doc = search("divide");
    assert!(hits_doc.iter().any(|s| s.name == "registry.canary.divide"));
}
