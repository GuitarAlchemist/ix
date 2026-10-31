//! Advisory projection of the archived Python transport contract.
//! This adapter only reads Host::now. It never calls Python, WMUX or a ledger.
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::mission::EnginePin;
use crate::{CallArgs, Capability, Host, Produced};

pub const FUNCTION_NAME: &str = "mission.dispatch_candidate";
pub const VERSION: &str = "0.1.0";
pub const SNAPSHOT_SHA: &str = "31d75c00409ff6227a199c4921be893e369c8b4e";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportPin {
    pub snapshot_sha: String,
    pub cycle_dispatch_sha256: String,
    pub wmux_control_sha256: String,
    pub wmux_mission_sha256: String,
}

/// Local archival integrity pins, not an upstream WMUX release or authentication.
pub fn transport_pin() -> TransportPin {
    TransportPin {
        snapshot_sha: SNAPSHOT_SHA.into(),
        cycle_dispatch_sha256: "fe5f0d23833d4dbd54b985f12ecbb3f64a489617a484c9b56b81f708e9e011e0".into(),
        wmux_control_sha256: "4aa7cc12881982540aed0c8e9971f5a7a912d5fb382b52621307569d75b703b9".into(),
        wmux_mission_sha256: "23f0db454cd4372fc62292fb31d756bac177f4bc7f9d8d9dfc10a7080e924f4d".into(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub session: String,
    pub workspace: String,
    pub surface: String,
}

/// Adapter request only. The Python functions do not consume this JSON shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchRequest {
    pub contract_version: String,
    pub dsl_version: String,
    pub engine: EnginePin,
    pub implementation_sha: String,
    pub transport: TransportPin,
    pub mission_id: String,
    pub nonce: String,
    pub binding: Binding,
    pub brief_path: String,
    pub brief_sha256: String,
    pub registry_path: String,
    pub ledger_path: String,
    pub created_at_ms: i64,
    /// Proposed local observation deadline; no native TTL or cancellation.
    pub wait_until_ms: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Editor { Empty, Nonempty, Unknown }

/// Host-injected declaration; there is no live snapshot producer in this module.
/// Keep the same mission/nonce/binding/artifacts for receipt-only replay.
#[derive(Debug, Clone)]
pub struct DispatchObservation {
    pub request: DispatchRequest,
    pub observed_at_ms: i64,
    /// Exact state names projected by cycle_dispatch.status.
    pub state: String,
    pub readiness: String,
    pub editor: Editor,
    pub prompt_confirmed: bool,
}

pub struct DispatchCandidate {
    host: Arc<dyn Host>,
    engine: EnginePin,
    implementation_sha: String,
    observation: Option<DispatchObservation>,
}

impl DispatchCandidate {
    pub fn new(host: Arc<dyn Host>, engine: EnginePin, implementation_sha: String,
        observation: Option<DispatchObservation>) -> Result<Self, String> {
        if engine.version != env!("CARGO_PKG_VERSION") || !hex(&engine.source_sha, 40) ||
            !hex(&implementation_sha, 40) {
            return Err("ContractPinMismatch".into());
        }
        Ok(Self {host, engine, implementation_sha, observation})
    }

    fn preview(&self, request: DispatchRequest) -> Result<Value, String> {
        if request.contract_version != VERSION || request.dsl_version != "1" ||
            request.engine != self.engine || request.implementation_sha != self.implementation_sha ||
            request.transport != transport_pin() {
            return Err("ContractPinMismatch".into());
        }
        for id in [&request.mission_id, &request.binding.session,
            &request.binding.workspace, &request.binding.surface] {
            if id.is_empty() || id.len() > 128 || !id.as_bytes()[0].is_ascii_alphanumeric() ||
                !id.bytes().all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b)) {
                return Err("IdentityInvalid".into());
            }
        }
        if !hex(&request.nonce, 16) || !hex(&request.brief_sha256, 64) {
            return Err("IdentityInvalid".into());
        }
        let paths = [&request.brief_path, &request.registry_path, &request.ledger_path];
        for path in paths {
            if path.is_empty() || path.len() > 512 || crate::path::normalize(path).map_err(|_| "ArtifactPathInvalid")? != *path ||
                !path.bytes().all(|b| b.is_ascii_alphanumeric() || b"._/-".contains(&b)) ||
                path.split('/').any(|p| p.eq_ignore_ascii_case(".git") || p.starts_with('-') || p.ends_with('.')) {
                return Err("ArtifactPathInvalid".into());
            }
        }
        if (0..paths.len()).any(|i| (i + 1..paths.len()).any(|j| paths[i].eq_ignore_ascii_case(paths[j]))) {
            return Err("ArtifactAlias".into());
        }
        let proof = self.observation.as_ref().ok_or("ObservationMissing")?;
        if proof.request != request {
            return Err("ObservationBindingMismatch".into());
        }
        let now = self.host.now().timestamp_millis();
        if request.created_at_ms > now ||
            !matches!(request.wait_until_ms.checked_sub(request.created_at_ms), Some(1..=3_600_000)) {
            return Err("WaitingDeadlineInvalid".into());
        }
        if !matches!(now.checked_sub(proof.observed_at_ms), Some(0..=29_999)) {
            return Err("ObservationStaleOrFuture".into());
        }
        let replay = match proof.state.as_str() {
            "prepared" => false,
            "dispatch_attempted" | "submission_unknown" | "transport_accepted" |
            "received" | "provider_working" | "blocked" | "awaiting_consumption" |
            "consumed" | "interrupted" | "abandoned" => true,
            _ => return Err("TransportStateUnknown".into()),
        };
        if !replay && proof.prompt_confirmed { return Err("ObservationInconsistent".into()); }
        let expired = now >= request.wait_until_ms;
        if !replay && !expired {
            if proof.readiness != "ready" { return Err("AdmissionUnknownOrNotReady".into()); }
            if proof.editor != Editor::Empty { return Err("InputNotVerifiedEmpty".into()); }
        }
        let operation = if expired { "stop_waiting" } else if replay { "observe_receipt" } else { "dispatch" };
        Ok(json!({
            "contract_version": VERSION, "function": FUNCTION_NAME,
            "request": request, "status": "candidate",
            "suggested_operation": operation, "transport_state": proof.state,
            "prompt_confirmed": proof.prompt_confirmed,
            "first_action_confirmed": false,
            "requires_runner_admission": true, "live_dispatch_available": false,
            "evidence_source": "host_declaration",
            "effect_outcome": "unknown", "native_execution_cancelled": false,
            "waiting_deadline_expired": expired, "observed_at_ms": proof.observed_at_ms
        }))
    }
}

fn hex(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

struct JsonBudget { remaining: usize }
impl std::io::Write for JsonBudget {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "request budget"));
        }
        self.remaining -= bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

impl Capability for DispatchCandidate {
    fn call(&self, args: CallArgs) -> Result<Produced, String> {
        if !args.positional.is_empty() || !args.named.is_empty() {
            return Err("PipedContractRequired".into());
        }
        let value = args.piped.ok_or("PipedContractRequired")?;
        // Adapter JSON budget only; this cannot bound the Python reader/CLI buffer.
        serde_json::to_writer(JsonBudget {remaining:4096}, &value)
            .map_err(|_| "RequestBudgetExceeded")?;
        let request = serde_json::from_value(value).map_err(|_| "RequestInvalid")?;
        self.preview(request).map(Produced::plain)
    }
}
