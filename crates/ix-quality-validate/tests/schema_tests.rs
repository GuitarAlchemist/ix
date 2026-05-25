//! Unit tests for the canonical dashboard-envelope schema.

use ix_quality_validate::{build_validator, validate_value};
use serde_json::json;

#[test]
fn valid_envelope_passes() {
    let validator = build_validator().expect("schema compiles");
    let envelope = json!({
        "domain": "ga-harness",
        "emitted_at": "2026-05-17T19:58:09Z",
        "metric_name": "harness_ready",
        "metric_value": 1.0,
        "oracle_status": "ok",
        "summary": "Supervised-loop kit artifacts present and parseable.",
        "problems": []
    });
    let res = validate_value(&validator, &envelope);
    assert!(
        res.is_ok(),
        "expected valid envelope to pass, got: {:?}",
        res
    );
}

#[test]
fn missing_oracle_status_fails_with_clear_message() {
    let validator = build_validator().expect("schema compiles");
    let envelope = json!({
        "domain": "ga-harness",
        "emitted_at": "2026-05-17T19:58:09Z",
        "metric_name": "harness_ready",
        "metric_value": 1.0,
        "summary": "Supervised-loop kit artifacts present and parseable."
    });
    let res = validate_value(&validator, &envelope);
    let errors = res.expect_err("expected missing oracle_status to fail");
    assert!(
        errors.iter().any(|e| e.contains("oracle_status")),
        "expected an error mentioning oracle_status, got: {errors:?}"
    );
}

#[test]
fn degraded_true_without_reason_fails_cross_field_rule() {
    let validator = build_validator().expect("schema compiles");
    let envelope = json!({
        "domain": "voicing-analysis",
        "emitted_at": "2026-05-24T08:00:00Z",
        "metric_name": "pass_pct",
        "metric_value": null,
        "oracle_status": "warn",
        "summary": "Producer ran in degraded mode.",
        "degraded": true
    });
    let res = validate_value(&validator, &envelope);
    let errors = res.expect_err("expected degraded:true without degraded_reason to fail");
    assert!(
        errors.iter().any(|e| e.contains("degraded_reason")),
        "expected an error mentioning degraded_reason, got: {errors:?}"
    );
}

#[test]
fn degraded_true_with_reason_passes() {
    let validator = build_validator().expect("schema compiles");
    let envelope = json!({
        "domain": "voicing-analysis",
        "emitted_at": "2026-05-24T08:00:00Z",
        "metric_name": "pass_pct",
        "metric_value": null,
        "oracle_status": "warn",
        "summary": "Producer ran in degraded mode; carrying last-known-good.",
        "degraded": true,
        "degraded_reason": "dotnet sdk missing",
        "last_known_good_pass_pct": 0.97,
        "last_known_good_source": "baseline",
        "last_known_good_date": "2026-05-01T00:00:00Z"
    });
    let res = validate_value(&validator, &envelope);
    assert!(
        res.is_ok(),
        "expected degraded envelope with reason to pass, got: {:?}",
        res
    );
}

#[test]
fn additional_domain_specific_fields_are_allowed() {
    let validator = build_validator().expect("schema compiles");
    let envelope = json!({
        "domain": "invariants",
        "emitted_at": "2026-05-24T10:00:00Z",
        "metric_name": "invariant_pass_ratio",
        "metric_value": 0.92,
        "oracle_status": "warn",
        "summary": "23 of 25 invariants passing (2 orphaned)",
        "problems": [
            { "code": "invariant-18", "message": "did not fire on any exemplar" }
        ],
        "exemplars": [],
        "fired": {}
    });
    let res = validate_value(&validator, &envelope);
    assert!(
        res.is_ok(),
        "expected envelope with extra domain-specific fields to pass, got: {:?}",
        res
    );
}
