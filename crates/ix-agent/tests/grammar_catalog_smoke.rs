//! Smoke tests for `ix_grammar_catalog`.
//!
//! Pins down the MCP handler wiring — that the tool is registered,
//! that it delegates to ix_grammar::catalog::GrammarCatalog, and
//! that filter combinations return the expected entries.

use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};

fn call(args: Value) -> Result<Value, String> {
    ToolRegistry::new().call("ix_grammar_catalog", args)
}

#[test]
fn no_filters_returns_every_entry() {
    let result = call(json!({})).expect("no-filter call");
    let total = result["counts"]["total"].as_u64().expect("total");
    let matched = result["matched"].as_u64().expect("matched");
    assert_eq!(matched, total);
    assert!(total >= 25, "expected 25+ grammar entries, got {total}");
}

#[test]
fn python_query_returns_python_grammar() {
    let result = call(json!({ "language": "python" })).expect("python call");
    let entries = result["entries"].as_array().expect("entries");
    let names: Vec<&str> = entries.iter().filter_map(|e| e["name"].as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("Python")),
        "python query should return a Python grammar; got {names:?}"
    );
}

#[test]
fn abnf_format_filter_catches_rfc_protocols() {
    let result = call(json!({ "format": "abnf" })).expect("abnf call");
    let entries = result["entries"].as_array().expect("entries");
    let names: Vec<&str> = entries.iter().filter_map(|e| e["name"].as_str()).collect();
    // Must include at least one HTTP/URI/JSON/TLS entry.
    let has_rfc_proto = names.iter().any(|n| {
        n.contains("HTTP") || n.contains("URI") || n.contains("JSON") || n.contains("TLS")
    });
    assert!(
        has_rfc_proto,
        "expected at least one RFC protocol in ABNF results, got {names:?}"
    );
}

#[test]
fn unknown_format_returns_clear_error() {
    let err = call(json!({ "format": "nonsense" })).expect_err("should reject");
    assert!(
        err.contains("unknown format") && err.contains("abnf"),
        "expected enum hint in error, got: {err}"
    );
}

#[test]
fn combined_language_and_format_filter() {
    let result = call(json!({
        "language": "http",
        "format": "abnf"
    }))
    .expect("combined call");
    let entries = result["entries"].as_array().expect("entries");
    // Every remaining entry must have format=abnf AND language=http
    // (or the meta-language "many").
    for e in entries {
        assert_eq!(e["format"], "abnf");
        let lang = e["language"].as_str().unwrap();
        assert!(lang == "http" || lang == "many");
    }
}

#[test]
fn topic_filter_finds_rfc_tag() {
    let result = call(json!({ "topic": "rfc" })).expect("topic call");
    let matched = result["matched"].as_u64().unwrap();
    assert!(
        matched >= 10,
        "expected 10+ rfc-tagged entries, got {matched}"
    );
}
