//! Versioned dry-run through the real parser/evaluator/registry, with a simulated host.
//! Fixed SHA values below are fixture identities, not live deployed code receipts.
use std::sync::Arc;
use ix_ixql::{CallArgs, Executor, Host, MemoryHost, Produced, RegistrationError};
use ix_ixql::mission::{self, EnginePin, OwnershipClaim, ProbeRecord, ScopeObservation, ScopeRefinement};
use serde_json::{json, Value};

const SHA: &str = "0123456789012345678901234567890123456789";
const OTHER_SHA: &str = "abcdefabcdefabcdefabcdefabcdefabcdefabcd";

fn observation() -> ScopeObservation {
    let at = MemoryHost::frozen().now().timestamp_millis();
    ScopeObservation {
        mission_id: "m-1".into(), attempt_id: "a-1".into(), revision: "r-1".into(),
        refusal_id: "ref-1".into(), refusal_reason: "scope_too_broad".into(),
        original_paths: vec![".".into()], observed_at_ms: at, expires_at_ms: at + 30_000,
        complete: true, physical_paths_verified: true,
        claims: vec![OwnershipClaim {path: "Blue".into(), owners: vec!["Blue".into()]}],
        probe: Some(ProbeRecord {ts: "2026-01-02T03:04:05Z".into(),
            metric: "fixture_scope_snapshot".into(), value: 1.0, unit: Some("count".into()),
            probe: "fixture".into(), build: "0123456789ab".into()}),
    }
}

fn request(paths: Value) -> Value {
    json!({"contract_version":"0.1.0", "dsl_version":"1",
        "engine":{"version":"0.1.0", "source_sha":SHA},
        "function_version":"0.1.0", "implementation_sha":SHA,
        "mission_id":"m-1", "attempt_id":"a-1", "revision":"r-1",
        "refusal_id":"ref-1", "owner":"Blue", "paths":paths})
}

fn composed(host: Arc<MemoryHost>, proof: Option<ScopeObservation>) -> Executor {
    let mut executor = Executor::new(host.clone());
    let adapter = ScopeRefinement::new(host, SHA.into(),
        EnginePin {version: "0.1.0".into(), source_sha: SHA.into()}, proof).unwrap();
    executor.capabilities().register(mission::FUNCTION_NAME, adapter).unwrap();
    executor
}

fn run(value: Value, proof: Option<ScopeObservation>) -> Result<Value, String> {
    let host = Arc::new(MemoryHost::frozen());
    host.seed("state/mission/proposal.json", value);
    composed(host, proof)
        .run_source(r#"proposal <- ix.io.read("state/mission/proposal.json") → mission.refine_scope"#)
        .map(|out| out.binding("proposal").unwrap().clone()).map_err(|e| e.to_string())
}

#[test]
fn root_git_add_is_refused_with_a_scope_reason_before_any_effect() {
    // The original RED oracle, unchanged in its behavior; only host composition is now supplied.
    let error = run(request(json!(["."])), Some(observation())).expect_err("root staging must be refused");
    assert!(error.contains("ScopeTooBroad"), "{error}");
}

#[test]
fn blue_preview_is_bounded_correlated_and_never_grants_permission() {
    let result = run(request(json!(["Blue"])), Some(observation())).unwrap();
    assert_eq!(result["argv"], json!(["git", "add", "--", "Blue/"]));
    assert_eq!(result["status"], "proposal");
    assert_eq!(result["requires_runner_admission"], true);
    assert_eq!(result["mission_id"], "m-1");
    assert_eq!(result["attempt_id"], "a-1");
    assert_eq!(result["implementation_sha"], SHA);
    assert!(jsonschema::draft202012::new(&mission::output_schema()).unwrap().is_valid(&result));
}

#[test]
fn dry_run_returns_a_binding_without_writes_or_compound_effects() {
    let host = Arc::new(MemoryHost::frozen());
    host.seed("state/mission/proposal.json", request(json!(["Blue"])));
    let outcome = composed(host.clone(), Some(observation())).run_source(
        r#"proposal <- ix.io.read("state/mission/proposal.json") → mission.refine_scope"#).unwrap();
    assert!(outcome.writes.is_empty() && outcome.compound.is_empty());
    assert_eq!(host.files().len(), 1);
}

#[test]
fn unknown_refusal_and_ambiguous_ownership_cannot_offer_a_workaround() {
    let mut proof = observation();
    proof.refusal_reason = "permission_unknown".into();
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("RefusalUnknown"));
    let mut proof = observation();
    proof.claims[0].owners.push("Green".into());
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("OwnershipAmbiguous"));
    let mut proof = observation();
    proof.claims.push(OwnershipClaim {path:"blue/private".into(), owners:vec!["Green".into()]});
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("OwnershipAmbiguous"));
}

#[test]
fn duplicates_are_refused_in_scope_ownership_and_ixql_records() {
    assert!(run(request(json!(["Blue", "Blue"])), Some(observation())).unwrap_err().contains("DuplicateScope"));
    let mut proof = observation();
    let mut alias = proof.claims[0].clone();
    alias.path = "blue".into();
    proof.claims.push(alias);
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("DuplicateOwnership"));
    let executor = composed(Arc::new(MemoryHost::frozen()), Some(observation()));
    assert!(executor.run_source("x <- { paths: [], paths: [] } → mission.refine_scope")
        .unwrap_err().to_string().contains("paths"));
}

#[test]
fn missing_expired_future_and_wrong_attempt_observations_remain_unknown() {
    assert!(run(request(json!(["Blue"])), None).unwrap_err().contains("ObservationMissing"));
    let mut proof = observation();
    proof.observed_at_ms -= 30_000; proof.expires_at_ms -= 30_000;
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("ObservationExpired"));
    let mut proof = observation();
    proof.observed_at_ms += 1;
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("ObservationTimeUnknown"));
    let mut proof = observation();
    proof.attempt_id = "different".into();
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("ObservationBindingMismatch"));
}

#[test]
fn heartbeat_cannot_replace_missing_or_stale_probe_evidence() {
    let mut proof = observation();
    proof.probe = None;
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("ProbeMissing"));
    let mut proof = observation();
    proof.probe.as_mut().unwrap().ts = "2026-01-02T03:03:34Z".into();
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("ProbeExpiredOrFuture"));
    let mut proof = observation();
    proof.probe.as_mut().unwrap().ts = "2026-01-02T03:03:35Z".into();
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("ProbeExpiredOrFuture"));
    let mut value = request(json!(["Blue"]));
    value["heartbeat"] = json!({"alive":true});
    assert!(run(value, Some(observation())).unwrap_err().contains("ProposalContractInvalid"));
}

#[test]
fn candidate_cannot_escape_ownership_or_git_literal_path_semantics() {
    for path in ["Green", "Blueberry"] {
        assert!(run(request(json!([path])), Some(observation())).unwrap_err().contains("ScopeNotOwned"));
    }
    for path in ["../Blue", "Blue/../Green", "/Blue", "C:/Blue", "Blue/*",
        ":(top)Blue", "Blue/.git", "Blue//x", "Blue/./x", "Blue./x", "-Blue"] {
        assert!(run(request(json!([path])), Some(observation())).unwrap_err().contains("ScopePathInvalid"));
    }
    let mut proof = observation();
    proof.physical_paths_verified = false;
    assert!(run(request(json!(["Blue"])), Some(proof)).unwrap_err().contains("OwnershipUnknown"));
}

#[test]
fn schema_and_engine_and_function_pins_are_exact_for_replay() {
    for field in ["contract_version", "dsl_version", "function_version"] {
        let mut value = request(json!(["Blue"])); value[field] = json!("latest");
        assert!(run(value, Some(observation())).is_err());
    }
    let mut value = request(json!(["Blue"]));
    value["implementation_sha"] = json!(OTHER_SHA);
    assert!(run(value, Some(observation())).unwrap_err().contains("ContractPinMismatch"));
    let mut value = request(json!(["Blue"]));
    value["engine"]["source_sha"] = json!(OTHER_SHA);
    assert!(run(value, Some(observation())).unwrap_err().contains("ContractPinMismatch"));
}

#[test]
fn a_replayed_preview_is_pure_but_is_not_an_effect_idempotency_receipt() {
    let a = run(request(json!(["Blue"])), Some(observation())).unwrap();
    let b = run(request(json!(["Blue"])), Some(observation())).unwrap();
    assert_eq!(a, b);
    assert_eq!(a["requires_runner_admission"], true);
}

#[test]
fn duplicate_registration_cannot_replace_the_existing_handler() {
    let mut executor = composed(Arc::new(MemoryHost::frozen()), Some(observation()));
    assert!(matches!(executor.capabilities().register(mission::FUNCTION_NAME,
        |_:CallArgs| Ok(Produced::plain(Value::Null))), Err(RegistrationError::Duplicate(_))));
}

#[test]
fn canonical_schema_exports_a_typed_roundtrip_without_permission_fields() {
    let value = request(json!(["Blue"]));
    let typed: mission::ScopeProposal = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(typed).unwrap(), value);
    let mut forged = value;
    forged["grant"] = json!(true);
    assert!(!jsonschema::draft202012::new(&mission::input_schema()).unwrap().is_valid(&forged));
}
