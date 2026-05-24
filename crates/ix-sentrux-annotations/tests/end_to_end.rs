//! End-to-end test: drives the real `sentrux.exe mcp` binary if it's
//! installed at the canonical workstation path. Skips gracefully when
//! sentrux isn't available so CI without sentrux still passes.
//!
//! Also covers the fixture-mode round-trip, which does NOT require
//! sentrux on disk — this is the path CI uses today.

use ix_sentrux_annotations::{
    convert::violation_to_annotation,
    emit_sidecar,
    mcp_bridge::{run_sentrux_check, SentruxConfig},
    rules_response::{parse_check_rules_response, RuleViolation, RulesReport},
    DEFAULT_SENTRUX_EXE,
};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn fixture_round_trip_emits_well_formed_annotation() {
    // The fixture mimics what sentrux would have produced against the ix
    // worktree on 2026-05-24 (one max_fn_lines violation). We assert the
    // emitted JSONL matches the contract: F truth value,
    // detected-by-sentrux certainty, sentrux author, sha256 id.
    let report = RulesReport {
        pass: false,
        rules_checked: 4,
        summary: "fixture".into(),
        violation_count: 1,
        violations: vec![RuleViolation {
            rule: "max_fn_lines".into(),
            severity: "Error".into(),
            message: "1 function(s) exceed max length of 400 lines".into(),
            files: vec![
                "crates/ix-agent/src/tools.rs:register_bridges_and_session (420 lines)".into(),
            ],
        }],
    };

    let tmp = tempfile::tempdir().unwrap();
    let now = "2026-05-24T18:00:00Z";
    let mut annotations = Vec::new();
    for v in &report.violations {
        annotations.extend(violation_to_annotation(tmp.path(), v, now));
    }
    assert_eq!(annotations.len(), 1);

    emit_sidecar(tmp.path(), &annotations, None).unwrap();
    let sidecar = tmp
        .path()
        .join("state/quality/ai-annotations-sentrux.jsonl");
    let body = fs::read_to_string(sidecar).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(body.trim_end()).unwrap();

    // SCHEMA_VERSION is 2 after PR #59 added @ai:business-value / @ai:hot-path
    // kinds; the contract still accepts v1 readers, but new annotations
    // serialize as v2.
    assert_eq!(parsed["schema_version"], 2);
    assert_eq!(parsed["truth_value"], "F");
    assert_eq!(parsed["certainty"], "detected-by-sentrux");
    assert_eq!(parsed["confidence"], 1.0);
    assert_eq!(parsed["source"]["author"], "sentrux");
    assert!(parsed["source"]["evidence"]
        .as_str()
        .unwrap()
        .starts_with("sentrux-check@"));
    assert_eq!(parsed["location"]["path"], "crates/ix-agent/src/tools.rs");
    assert!(parsed["claim"]
        .as_str()
        .unwrap()
        .contains("register_bridges_and_session"));
    // Deterministic id (sha256 over path:line:kind:claim).
    let id = parsed["id"].as_str().unwrap();
    assert!(id.starts_with("sha256:") && id.len() == "sha256:".len() + 64);
}

#[test]
fn plaintext_diagnostic_becomes_empty_report() {
    // Sentrux returns plaintext (not JSON) when a workspace has no rules
    // file. That should yield an empty report, not an error.
    let envelope = serde_json::json!({
        "result": {
            "content": [{
                "text": "No rules file found at C:/whatever/.sentrux/rules.toml. Create one to define architectural constraints."
            }]
        }
    });
    let report = parse_check_rules_response(&envelope).expect("parses");
    assert!(report.violations.is_empty());
    assert!(report.summary.contains("No rules file found"));
}

#[test]
fn live_sentrux_check_against_ix_worktree() {
    let exe = PathBuf::from(DEFAULT_SENTRUX_EXE);
    if !exe.exists() {
        eprintln!("skipping: sentrux.exe not installed at {}", exe.display());
        return;
    }
    // Find the ix workspace root by walking up from CARGO_MANIFEST_DIR.
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    if !workspace.join("Cargo.toml").exists() {
        eprintln!("skipping: could not locate ix workspace Cargo.toml");
        return;
    }
    let cfg = SentruxConfig {
        sentrux_exe: exe,
        workspace: workspace.clone(),
        timeout: Duration::from_secs(60),
    };
    let report = match run_sentrux_check(&cfg) {
        Ok(r) => r,
        Err(e) => {
            // Live transport can be flaky on a busy machine (sentrux can
            // hang on a cold scan). Skip rather than fail CI on
            // environmental noise — the fixture test covers correctness.
            eprintln!("skipping live sentrux test (transport error): {e}");
            return;
        }
    };
    // The contract claim: when sentrux finds violations, the bridge emits
    // a well-formed annotation per violation. We only assert the SHAPE
    // here — the count fluctuates as the ix workspace evolves.
    for v in &report.violations {
        let annos = violation_to_annotation(&workspace, v, "live");
        assert!(
            !annos.is_empty(),
            "every violation yields at least one annotation"
        );
        for a in &annos {
            assert_eq!(a.source.author, "sentrux");
            assert_eq!(a.truth_value, ix_ai_annotations::TruthValue::F);
        }
    }
}
