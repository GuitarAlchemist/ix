//! Parser/adapter fixtures only: no WMUX, Python or transport effect is invoked.
//! Fixture code SHAs are declarations, not deployment authentication.
use std::sync::Arc;
use ix_ixql::{Executor, Host, MemoryHost, CallArgs, Produced, RegistrationError};
use ix_ixql::mission::EnginePin;
use ix_ixql::dispatch::{self, DispatchCandidate, DispatchRequest, DispatchObservation, Editor};
use serde_json::{json, Value};

const SHA: &str = "0123456789012345678901234567890123456789";
fn request() -> DispatchRequest {
    let at = MemoryHost::frozen().now().timestamp_millis();
    serde_json::from_value(json!({
        "contract_version":"0.1.0", "dsl_version":"1",
        "engine":{"version":"0.1.0", "source_sha":SHA}, "implementation_sha":SHA,
        "transport":dispatch::transport_pin(), "mission_id":"fixture-m1", "nonce":"0123456789abcdef",
        "binding":{"session":"fixture-session", "workspace":"ws-fixture", "surface":"surf-fixture"},
        "brief_path":"state/mission/brief.json", "brief_sha256":"a".repeat(64),
        "registry_path":"state/mission/registry.json", "ledger_path":"state/mission/ledger.jsonl",
        "created_at_ms":at, "wait_until_ms":at+30_000
    })).unwrap()
}
fn observation() -> DispatchObservation {
    DispatchObservation {request:request(), observed_at_ms:MemoryHost::frozen().now().timestamp_millis(),
        state:"prepared".into(), readiness:"ready".into(), editor:Editor::Empty, prompt_confirmed:false}
}
fn composed(host: Arc<MemoryHost>, proof: Option<DispatchObservation>) -> Executor {
    let mut executor = Executor::new(host.clone());
    executor.capabilities().register(dispatch::FUNCTION_NAME,
        DispatchCandidate::new(host, request().engine, SHA.into(), proof).unwrap()).unwrap();
    executor
}
fn run(value: Value, proof: Option<DispatchObservation>) -> Result<Value, String> {
    let host = Arc::new(MemoryHost::frozen());
    host.seed("state/dispatch/request.json", value);
    composed(host, proof).run_source(
        r#"candidate <- ix.io.read("state/dispatch/request.json") → mission.dispatch_candidate"#)
        .map(|out| out.binding("candidate").unwrap().clone()).map_err(|e|e.to_string())
}
fn value() -> Value { serde_json::to_value(request()).unwrap() }

#[test]
fn dispatch_candidate_never_admits_a_live_send() {
    let host = Arc::new(MemoryHost::frozen());
    host.seed("state/dispatch/request.json", value());
    let executor = composed(host.clone(), Some(observation()));
    let out = executor.run_source(
        r#"candidate <- ix.io.read("state/dispatch/request.json") → mission.dispatch_candidate"#,
    ).unwrap();
    let candidate = out.binding("candidate").unwrap();
    assert_eq!(candidate["live_dispatch_available"], false);
    assert_eq!(candidate["requires_runner_admission"], true);
    assert!(out.writes.is_empty() && out.compound.is_empty());
    assert_eq!(candidate["suggested_operation"], "dispatch");
    assert_eq!(candidate["status"], "candidate");
    assert!(candidate.get("argv").is_none());
    assert_eq!(host.files().len(), 1);
}

#[test]
fn three_ids_and_nonce_and_artifacts_are_exact_even_for_receipt_replay() {
    let mut proof = observation(); proof.state = "submission_unknown".into();
    for field in ["session", "workspace", "surface"] {
        let mut changed = value(); changed["binding"][field] = json!("different");
        assert!(run(changed, Some(proof.clone())).unwrap_err().contains("ObservationBindingMismatch"));
    }
    for field in ["mission_id", "nonce", "brief_path", "brief_sha256", "registry_path", "ledger_path"] {
        let mut changed = value();
        changed[field] = if field == "nonce" {json!("abcdef0123456789")} else if field == "brief_sha256" {json!("b".repeat(64))} else {json!("different")};
        assert!(run(changed, Some(proof.clone())).unwrap_err().contains("ObservationBindingMismatch"));
    }
}

#[test]
fn ambiguous_or_already_attempted_states_offer_receipt_only_even_with_nonempty_input() {
    for state in ["dispatch_attempted", "submission_unknown", "transport_accepted", "received",
        "provider_working", "blocked", "awaiting_consumption", "consumed", "interrupted", "abandoned"] {
        let mut proof = observation(); proof.state = state.into();
        proof.editor = Editor::Nonempty; proof.readiness = "blocked".into();
        let first = run(value(), Some(proof.clone())).unwrap();
        let replay = run(value(), Some(proof)).unwrap();
        assert_eq!(first, replay);
        assert_eq!(first["suggested_operation"], "observe_receipt");
        assert_eq!(first["effect_outcome"], "unknown");
        assert_eq!(first["first_action_confirmed"], false);
        assert_eq!(first["live_dispatch_available"], false);
    }
}

#[test]
fn nonempty_or_unknown_editor_refuses_a_prepared_candidate_without_clearing_anything() {
    for editor in [Editor::Nonempty, Editor::Unknown] {
        let mut proof = observation(); proof.editor = editor;
        assert!(run(value(), Some(proof)).unwrap_err().contains("InputNotVerifiedEmpty"));
    }
    for readiness in ["busy", "blocked", "unknown", "idle"] {
        let mut proof = observation(); proof.readiness = readiness.into();
        assert!(run(value(), Some(proof)).unwrap_err().contains("AdmissionUnknownOrNotReady"));
    }
}

#[test]
fn local_wait_deadline_stops_observation_without_cancelling_or_resending() {
    let mut proof = observation();
    proof.request.created_at_ms -= 30_000; proof.request.wait_until_ms -= 30_000;
    proof.state = "submission_unknown".into();
    let result = run(serde_json::to_value(&proof.request).unwrap(), Some(proof)).unwrap();
    assert_eq!(result["suggested_operation"], "stop_waiting");
    assert_eq!(result["waiting_deadline_expired"], true);
    assert_eq!(result["native_execution_cancelled"], false);
    assert_eq!(result["effect_outcome"], "unknown");
    assert_eq!(result["requires_runner_admission"], true);
}

#[test]
fn freshness_boundary_future_and_missing_observation_are_refused() {
    assert!(run(value(), None).unwrap_err().contains("ObservationMissing"));
    for shift in [-30_000, 1] {
        let mut proof = observation(); proof.observed_at_ms += shift;
        assert!(run(value(), Some(proof)).unwrap_err().contains("ObservationStaleOrFuture"));
    }
    let mut proof = observation(); proof.observed_at_ms -= 29_999;
    assert!(run(value(), Some(proof)).is_ok());
}

#[test]
fn deadline_cannot_be_extended_by_a_task_or_overflowed() {
    for offset in [0, -1, 3_600_001] {
        let mut proof = observation(); proof.request.wait_until_ms = proof.request.created_at_ms + offset;
        assert!(run(serde_json::to_value(&proof.request).unwrap(), Some(proof))
            .unwrap_err().contains("WaitingDeadlineInvalid"));
    }
    let mut proof = observation(); proof.request.created_at_ms = i64::MIN; proof.request.wait_until_ms = i64::MAX;
    assert!(run(serde_json::to_value(&proof.request).unwrap(), Some(proof)).unwrap_err().contains("WaitingDeadlineInvalid"));
    let mut changed = value(); changed["wait_until_ms"] = json!(request().wait_until_ms + 1);
    assert!(run(changed, Some(observation())).unwrap_err().contains("ObservationBindingMismatch"));
}

#[test]
fn local_archive_and_engine_pins_have_no_latest_or_upstream_alias() {
    let mut changed = value(); changed["transport"]["snapshot_sha"] = json!("a".repeat(40));
    assert!(run(changed, Some(observation())).unwrap_err().contains("ContractPinMismatch"));
    for field in ["cycle_dispatch_sha256", "wmux_control_sha256", "wmux_mission_sha256"] {
        let mut changed = value(); changed["transport"][field] = json!("b".repeat(64));
        assert!(run(changed, Some(observation())).unwrap_err().contains("ContractPinMismatch"));
    }
    for field in ["implementation_sha", "source_sha"] {
        let mut changed = value();
        if field == "source_sha" {changed["engine"][field] = json!("b".repeat(40));}
        else {changed[field] = json!("b".repeat(40));}
        assert!(run(changed, Some(observation())).unwrap_err().contains("ContractPinMismatch"));
    }
}

#[test]
fn unknown_contract_keys_budget_and_paths_cannot_smuggle_cli_or_permissions() {
    for key in ["argv", "permission", "cancel", "exactly_once"] {
        let mut changed = value(); changed[key] = json!(true);
        assert!(run(changed, Some(observation())).unwrap_err().contains("RequestInvalid"));
    }
    let mut changed = value(); changed["mission_id"] = json!("x".repeat(5000));
    assert!(run(changed, Some(observation())).unwrap_err().contains("RequestBudgetExceeded"));
    for path in ["../other", "C:/outside", ".git/index", "state//file", "state/-arg", "state/file."] {
        let mut proof = observation(); proof.request.registry_path = path.into();
        assert!(run(serde_json::to_value(&proof.request).unwrap(), Some(proof))
            .unwrap_err().contains("ArtifactPathInvalid"));
    }
    let mut proof = observation(); proof.request.registry_path = "STATE/MISSION/LEDGER.JSONL".into();
    assert!(run(serde_json::to_value(&proof.request).unwrap(), Some(proof)).unwrap_err().contains("ArtifactAlias"));
}

#[test]
fn transport_ack_or_working_or_consumed_does_not_prove_successful_tool_action() {
    for state in ["transport_accepted", "provider_working", "consumed"] {
        let mut proof = observation(); proof.state = state.into(); proof.prompt_confirmed = state != "transport_accepted";
        let result = run(value(), Some(proof)).unwrap();
        assert_eq!(result["first_action_confirmed"], false);
        assert_eq!(result["effect_outcome"], "unknown");
        assert_eq!(result["evidence_source"], "host_declaration");
        assert!(result.get("verdict").is_none());
    }
    let mut proof = observation(); proof.state = "unknown_future_state".into();
    assert!(run(value(), Some(proof)).unwrap_err().contains("TransportStateUnknown"));
}

#[test]
fn parser_duplicates_and_registry_replacement_fail_closed() {
    let mut executor = composed(Arc::new(MemoryHost::frozen()), Some(observation()));
    assert!(matches!(executor.capabilities().register(dispatch::FUNCTION_NAME,
        |_:CallArgs| Ok(Produced::plain(Value::Null))), Err(RegistrationError::Duplicate(_))));
    assert!(executor.run_source("x <- { mission_id: \"a\", mission_id: \"b\" } → mission.dispatch_candidate").is_err());
    assert!(executor.run_source("x <- {} → mission.dispatch_candidate(force: true)")
        .unwrap_err().to_string().contains("PipedContractRequired"));
}

#[test]
fn malformed_host_code_identity_is_not_registered() {
    assert!(DispatchCandidate::new(Arc::new(MemoryHost::frozen()),
        EnginePin {version:"latest".into(), source_sha:SHA.into()}, SHA.into(), Some(observation())).is_err());
}

#[test]
fn contradictory_prepared_receipt_and_empty_artifact_path_refuse() {
    let mut proof = observation(); proof.prompt_confirmed = true;
    assert!(run(value(), Some(proof)).unwrap_err().contains("ObservationInconsistent"));
    let mut proof = observation(); proof.request.brief_path.clear();
    assert!(run(serde_json::to_value(&proof.request).unwrap(), Some(proof)).unwrap_err().contains("ArtifactPathInvalid"));
}
