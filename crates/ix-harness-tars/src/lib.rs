//! Harness adapter — tars `ComprehensiveDiagnostics` → `SessionEvent` stream.
//!
//! This is the reference implementation of the harness-adapter
//! pattern specified in
//! `ix/docs/brainstorms/2026-04-11-harness-adapter-pattern.md`.
//! The projection rules are owned by
//! `demerzel/logic/harness-tars.md` — this crate mechanically
//! applies them. If the two drift, the Demerzel doc wins.
//!
//! # Shape
//!
//! One pure function: [`tars_to_observations`]. Takes a raw byte
//! slice (tars's JSON output), parses it into the relevant subset
//! of fields, and emits a `Vec<SessionEvent::ObservationAdded>`.
//!
//! The adapter is deterministic: same input bytes + same round
//! produce bit-identical output. `diagnosis_id` is the SHA-256 of
//! the canonical input bytes, so two adapter runs on the same
//! diagnosis yield identical correlation IDs.
//!
//! # Non-goals
//!
//! - Does not parse the full tars diagnostic shape — only the
//!   fields the projection rules consume. Unknown fields are
//!   silently ignored.
//! - Does not output wall-clock times, IPs, env vars, or any other
//!   non-deterministic or privacy-adjacent data.
//! - Does not sign observations. First-party use only until the
//!   signature layer ships.

use ix_agent_core::SessionEvent;
use ix_types::Hexavalent;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Fixed source name for tars observations. Must match the
/// Demerzel governance doc.
pub const SOURCE: &str = "tars";

/// Errors produced by [`tars_to_observations`].
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    /// The input bytes were not valid UTF-8 or valid JSON.
    #[error("parse: {0}")]
    Parse(String),
}

/// Subset of tars's `ComprehensiveDiagnostics` that the projection
/// rules actually use. Missing or extra fields are silently
/// ignored — the adapter extracts what it can and emits
/// observations only for rules whose inputs are present.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct TarsDiagnostics {
    #[serde(default)]
    gpu_info: Vec<GpuInfo>,
    #[serde(default)]
    git_health: Option<GitHealth>,
    #[serde(default)]
    network_diagnostics: Option<NetworkDiagnostics>,
    #[serde(default)]
    system_resources: Option<SystemResources>,
    #[serde(default)]
    service_health: Option<ServiceHealth>,
    #[serde(default)]
    overall_health_score: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GpuInfo {
    #[serde(default)]
    name: String,
    #[serde(default)]
    memory_total: u64,
    #[serde(default)]
    memory_used: u64,
    #[serde(default)]
    temperature: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct GitHealth {
    #[serde(default)]
    is_repository: bool,
    #[serde(default)]
    is_clean: bool,
    #[serde(default)]
    unstaged_changes: u32,
    #[serde(default)]
    ahead_by: u32,
    #[serde(default)]
    behind_by: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct NetworkDiagnostics {
    #[serde(default)]
    is_connected: bool,
    #[serde(default)]
    dns_resolution_time: u64,
    #[serde(default)]
    ping_latency: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct SystemResources {
    #[serde(default)]
    cpu_usage_percent: f64,
    #[serde(default)]
    memory_total_bytes: u64,
    #[serde(default)]
    memory_used_bytes: u64,
    #[serde(default)]
    disk_total_bytes: u64,
    #[serde(default)]
    disk_free_bytes: u64,
    #[serde(default)]
    process_count: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ServiceHealth {
    #[serde(default)]
    database_connectivity: bool,
    #[serde(default)]
    web_service_availability: bool,
    #[serde(default)]
    file_system_permissions: bool,
}

/// Project a tars diagnostics JSON blob into a stream of
/// [`SessionEvent::ObservationAdded`] records.
///
/// The `round` parameter is stamped into every emitted observation
/// and must match the caller's remediation-round counter. Ordinals
/// start at 0 and increment for every emitted observation (rules
/// that don't fire don't consume an ordinal).
///
/// The `diagnosis_id` of each emitted observation is the SHA-256
/// of the canonical input bytes, so two runs on the same input
/// produce correlated observations that dedup cleanly in the G-Set
/// merge.
pub fn tars_to_observations(
    input: &[u8],
    round: u32,
) -> Result<Vec<SessionEvent>, AdapterError> {
    // Parse tars JSON — permissive by default (unknown fields OK).
    let diag: TarsDiagnostics = serde_json::from_slice(input)
        .map_err(|e| AdapterError::Parse(e.to_string()))?;

    let diagnosis_id = sha256_hex(input);
    let mut out: Vec<SessionEvent> = Vec::new();
    let mut ordinal: u64 = 0;

    // ── Top-level overall health score ─────────────────────────
    if let Some(score) = diag.overall_health_score {
        let (variant, weight, evidence) = classify_overall_score(score);
        out.push(emit(
            &mut ordinal,
            &diagnosis_id,
            round,
            "tars_diagnosis::reliable",
            variant,
            weight,
            evidence,
        ));
    }

    // ── Disk health ────────────────────────────────────────────
    if let Some(sys) = diag.system_resources.as_ref() {
        if sys.disk_free_bytes > 0 && sys.disk_free_bytes < 1_000_000_000 {
            // < 1 GB
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "cleanup_disk::valuable",
                Hexavalent::True,
                0.9,
                format!("disk_free_bytes={}", sys.disk_free_bytes),
            ));
        } else if sys.disk_free_bytes > 0 && sys.disk_free_bytes < 5_000_000_000 {
            // 1–5 GB
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "cleanup_disk::valuable",
                Hexavalent::Probable,
                0.7,
                format!("disk_free_bytes={}", sys.disk_free_bytes),
            ));
        }
        if sys.disk_total_bytes > 0 {
            let used_frac = sys.disk_used_bytes_frac();
            if used_frac > 0.98 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    "system_stability::safe",
                    Hexavalent::False,
                    0.9,
                    format!("disk used {:.1}%", used_frac * 100.0),
                ));
            } else if used_frac > 0.95 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    "system_stability::safe",
                    Hexavalent::Doubtful,
                    0.7,
                    format!("disk used {:.1}%", used_frac * 100.0),
                ));
            }
        }
    }

    // ── GPU rules (per device) ─────────────────────────────────
    for (i, gpu) in diag.gpu_info.iter().enumerate() {
        let gpu_key = sanitize_name(&gpu.name);
        if let Some(temp) = gpu.temperature {
            if temp > 95.0 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    &format!("{gpu_key}::safe"),
                    Hexavalent::False,
                    1.0,
                    format!("gpu[{i}].temperature={temp}"),
                ));
            } else if temp > 85.0 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    "gpu_cooling::valuable",
                    Hexavalent::True,
                    0.9,
                    format!("gpu[{i}].temperature={temp}"),
                ));
            }
        }
        if gpu.memory_total > 0 {
            let used_frac = gpu.memory_used as f64 / gpu.memory_total as f64;
            if used_frac > 0.98 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    &format!("{gpu_key}::safe"),
                    Hexavalent::False,
                    0.9,
                    format!("gpu[{i}] memory {:.1}%", used_frac * 100.0),
                ));
            }
        }
    }

    // ── Git health ─────────────────────────────────────────────
    if let Some(git) = diag.git_health.as_ref() {
        if !git.is_repository {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "git_init::valuable",
                Hexavalent::True,
                0.9,
                "not a git repository".to_string(),
            ));
        } else if !git.is_clean && git.unstaged_changes > 100 {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "git_review::valuable",
                Hexavalent::Probable,
                0.7,
                format!("{} unstaged changes", git.unstaged_changes),
            ));
        } else if git.behind_by > 50 {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "git_pull::valuable",
                Hexavalent::Probable,
                0.8,
                format!("behind by {}", git.behind_by),
            ));
        } else if git.is_clean && git.ahead_by == 0 && git.behind_by == 0 {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "git_state::reliable",
                Hexavalent::True,
                0.9,
                "clean + synced".to_string(),
            ));
        }
    }

    // ── Network reachability ───────────────────────────────────
    if let Some(net) = diag.network_diagnostics.as_ref() {
        if !net.is_connected {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "network::reliable",
                Hexavalent::False,
                1.0,
                "no network connection".to_string(),
            ));
        } else if let Some(ping) = net.ping_latency {
            if ping > 500 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    "network::reliable",
                    Hexavalent::Doubtful,
                    0.6,
                    format!("ping_latency={ping}ms"),
                ));
            } else if ping < 50 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    "network::reliable",
                    Hexavalent::True,
                    0.8,
                    format!("ping_latency={ping}ms"),
                ));
            }
        }
        if net.dns_resolution_time > 1000 {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "dns::reliable",
                Hexavalent::Doubtful,
                0.7,
                format!("dns_resolution_time={}ms", net.dns_resolution_time),
            ));
        }
    }

    // ── System resource pressure ───────────────────────────────
    if let Some(sys) = diag.system_resources.as_ref() {
        if sys.cpu_usage_percent > 95.0 {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "cpu_throttle::valuable",
                Hexavalent::Probable,
                0.6,
                format!("cpu_usage={:.1}%", sys.cpu_usage_percent),
            ));
        }
        if sys.memory_total_bytes > 0 {
            let mem_frac = sys.memory_used_bytes as f64 / sys.memory_total_bytes as f64;
            if mem_frac > 0.98 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    "system_stability::safe",
                    Hexavalent::False,
                    0.8,
                    format!("memory used {:.1}%", mem_frac * 100.0),
                ));
            } else if mem_frac > 0.95 {
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    "memory_pressure::valuable",
                    Hexavalent::Doubtful,
                    0.7,
                    format!("memory used {:.1}%", mem_frac * 100.0),
                ));
            }
        }
        if sys.process_count > 2000 {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "process_audit::valuable",
                Hexavalent::Probable,
                0.5,
                format!("process_count={}", sys.process_count),
            ));
        }
    }

    // ── Service health ─────────────────────────────────────────
    if let Some(svc) = diag.service_health.as_ref() {
        if !svc.database_connectivity {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "database::reliable",
                Hexavalent::False,
                1.0,
                "database unreachable".to_string(),
            ));
        }
        if !svc.web_service_availability {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "web_service::reliable",
                Hexavalent::False,
                1.0,
                "web service unavailable".to_string(),
            ));
        }
        if !svc.file_system_permissions {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "fs_permissions::safe",
                Hexavalent::False,
                1.0,
                "filesystem permission failure".to_string(),
            ));
        }
        if svc.database_connectivity && svc.web_service_availability && svc.file_system_permissions {
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                "service_baseline::reliable",
                Hexavalent::True,
                0.9,
                "database+web+fs all healthy".to_string(),
            ));
        }
    }

    Ok(out)
}

impl SystemResources {
    fn disk_used_bytes_frac(&self) -> f64 {
        if self.disk_total_bytes == 0 {
            0.0
        } else {
            let used = self.disk_total_bytes.saturating_sub(self.disk_free_bytes);
            used as f64 / self.disk_total_bytes as f64
        }
    }
}

fn classify_overall_score(score: f64) -> (Hexavalent, f64, String) {
    let evidence = format!("overall_health_score={score}");
    if score >= 90.0 {
        (Hexavalent::True, 0.9, evidence)
    } else if score >= 70.0 {
        (Hexavalent::Probable, 0.7, evidence)
    } else if score >= 50.0 {
        (Hexavalent::Unknown, 0.5, evidence)
    } else if score >= 30.0 {
        (Hexavalent::Doubtful, 0.7, evidence)
    } else {
        (Hexavalent::False, 0.9, evidence)
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn emit(
    ordinal: &mut u64,
    diagnosis_id: &str,
    round: u32,
    claim_key: &str,
    variant: Hexavalent,
    weight: f64,
    evidence: String,
) -> SessionEvent {
    let ord = *ordinal;
    *ordinal += 1;
    SessionEvent::ObservationAdded {
        ordinal: ord,
        source: SOURCE.to_string(),
        diagnosis_id: diagnosis_id.to_string(),
        round,
        claim_key: claim_key.to_string(),
        variant,
        weight,
        evidence: Some(evidence),
    }
}

fn sha256_hex(input: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input);
    let hash = hasher.finalize();
    let mut out = String::with_capacity(64);
    for byte in hash.iter() {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn extract(event: &SessionEvent) -> (&str, &str, Hexavalent, f64) {
        if let SessionEvent::ObservationAdded {
            source,
            claim_key,
            variant,
            weight,
            ..
        } = event
        {
            (source, claim_key, *variant, *weight)
        } else {
            panic!("expected ObservationAdded, got {event:?}")
        }
    }

    // ── Happy path: canonical healthy system ──────────────────

    #[test]
    fn healthy_system_emits_positive_baselines() {
        let input = serde_json::json!({
            "overallHealthScore": 95.0,
            "gpuInfo": [],
            "gitHealth": {
                "isRepository": true,
                "isClean": true,
                "unstagedChanges": 0,
                "aheadBy": 0,
                "behindBy": 0
            },
            "networkDiagnostics": {
                "isConnected": true,
                "dnsResolutionTime": 20,
                "pingLatency": 15
            },
            "systemResources": {
                "cpuUsagePercent": 30.0,
                "memoryTotalBytes": 34359738368_u64,
                "memoryUsedBytes": 8000000000_u64,
                "diskTotalBytes": 1000000000000_u64,
                "diskFreeBytes": 500000000000_u64,
                "processCount": 400
            },
            "serviceHealth": {
                "databaseConnectivity": true,
                "webServiceAvailability": true,
                "fileSystemPermissions": true
            }
        })
        .to_string();
        let obs = tars_to_observations(input.as_bytes(), 1).expect("parses");

        // Expect: overall score baseline, git clean, network healthy,
        // service baseline — 4 observations with T variant.
        assert!(obs.len() >= 4, "expected >= 4 observations, got {:#?}", obs);
        for e in &obs {
            let (source, _claim, _var, _w) = extract(e);
            assert_eq!(source, "tars");
        }
        // Assert key claims fired.
        let claims: Vec<String> = obs.iter().map(|e| extract(e).1.to_string()).collect();
        assert!(claims.contains(&"tars_diagnosis::reliable".to_string()));
        assert!(claims.contains(&"git_state::reliable".to_string()));
        assert!(claims.contains(&"network::reliable".to_string()));
        assert!(claims.contains(&"service_baseline::reliable".to_string()));
    }

    // ── Critical disk scenario ────────────────────────────────

    #[test]
    fn critical_disk_emits_cleanup_and_stability_refutation() {
        let input = serde_json::json!({
            "overallHealthScore": 40.0,
            "systemResources": {
                "diskTotalBytes": 1000000000000_u64,
                "diskFreeBytes": 500000000_u64, // 500 MB — critical
                "memoryTotalBytes": 34359738368_u64,
                "memoryUsedBytes": 10000000000_u64,
                "cpuUsagePercent": 40.0,
                "processCount": 300
            }
        })
        .to_string();
        let obs = tars_to_observations(input.as_bytes(), 3).expect("parses");

        let claims: Vec<String> = obs.iter().map(|e| extract(e).1.to_string()).collect();
        assert!(claims.contains(&"tars_diagnosis::reliable".to_string()));
        assert!(claims.contains(&"cleanup_disk::valuable".to_string()));
        assert!(claims.contains(&"system_stability::safe".to_string()));

        // Find the cleanup_disk observation — should be T weight 0.9.
        let cleanup = obs
            .iter()
            .find(|e| extract(e).1 == "cleanup_disk::valuable")
            .unwrap();
        let (_, _, variant, weight) = extract(cleanup);
        assert_eq!(variant, Hexavalent::True);
        assert!((weight - 0.9).abs() < 1e-9);

        // Stability should be F (disk > 98% used — 500MB free / 1TB).
        let stability = obs
            .iter()
            .find(|e| extract(e).1 == "system_stability::safe")
            .unwrap();
        let (_, _, variant, _) = extract(stability);
        assert_eq!(variant, Hexavalent::False);
    }

    // ── GPU overheating scenario ──────────────────────────────

    #[test]
    fn gpu_overheating_emits_safe_false_on_gpu_name() {
        let input = serde_json::json!({
            "gpuInfo": [
                {
                    "name": "NVIDIA RTX 5080",
                    "memoryTotal": 17179869184_u64,
                    "memoryUsed": 8000000000_u64,
                    "temperature": 96.0
                }
            ]
        })
        .to_string();
        let obs = tars_to_observations(input.as_bytes(), 0).expect("parses");

        let gpu_obs = obs
            .iter()
            .find(|e| extract(e).1 == "nvidia_rtx_5080::safe")
            .expect("expected nvidia_rtx_5080::safe observation");
        let (_, _, variant, weight) = extract(gpu_obs);
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 1.0).abs() < 1e-9);
    }

    // ── Determinism ────────────────────────────────────────────

    #[test]
    fn same_input_produces_same_diagnosis_id() {
        let input = serde_json::json!({
            "overallHealthScore": 75.0,
            "gpuInfo": [],
            "systemResources": {
                "diskTotalBytes": 0,
                "diskFreeBytes": 0,
                "memoryTotalBytes": 0,
                "memoryUsedBytes": 0,
                "cpuUsagePercent": 0.0,
                "processCount": 0
            }
        })
        .to_string();
        let obs1 = tars_to_observations(input.as_bytes(), 0).unwrap();
        let obs2 = tars_to_observations(input.as_bytes(), 0).unwrap();
        assert_eq!(obs1.len(), obs2.len());
        for (a, b) in obs1.iter().zip(obs2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn different_inputs_produce_different_diagnosis_ids() {
        let input1 = serde_json::json!({"overallHealthScore": 90.0}).to_string();
        let input2 = serde_json::json!({"overallHealthScore": 20.0}).to_string();
        let obs1 = tars_to_observations(input1.as_bytes(), 0).unwrap();
        let obs2 = tars_to_observations(input2.as_bytes(), 0).unwrap();

        if let (
            SessionEvent::ObservationAdded { diagnosis_id: d1, .. },
            SessionEvent::ObservationAdded { diagnosis_id: d2, .. },
        ) = (&obs1[0], &obs2[0])
        {
            assert_ne!(d1, d2);
        } else {
            panic!("expected ObservationAdded");
        }
    }

    // ── Round parameter stamping ──────────────────────────────

    #[test]
    fn round_parameter_stamped_into_every_observation() {
        let input = serde_json::json!({"overallHealthScore": 85.0}).to_string();
        let obs = tars_to_observations(input.as_bytes(), 42).unwrap();
        for event in &obs {
            if let SessionEvent::ObservationAdded { round, .. } = event {
                assert_eq!(*round, 42);
            } else {
                panic!("expected ObservationAdded");
            }
        }
    }

    // ── Sanitization ──────────────────────────────────────────

    #[test]
    fn gpu_name_sanitization_handles_spaces_and_punctuation() {
        assert_eq!(sanitize_name("NVIDIA RTX 5080"), "nvidia_rtx_5080");
        assert_eq!(sanitize_name("AMD Radeon RX 7900 XTX"), "amd_radeon_rx_7900_xtx");
        assert_eq!(sanitize_name("  trailing whitespace  "), "trailing_whitespace");
        assert_eq!(sanitize_name("!!!special!!!"), "special");
    }

    // ── Parse error ───────────────────────────────────────────

    #[test]
    fn malformed_json_returns_parse_error() {
        let err = tars_to_observations(b"{not valid json", 0)
            .expect_err("should fail");
        assert!(matches!(err, AdapterError::Parse(_)));
    }

    #[test]
    fn empty_input_is_treated_as_empty_object() {
        // Empty JSON object means "no fields present" — zero
        // observations, no error. Empty byte slice is invalid.
        let obs = tars_to_observations(b"{}", 0).unwrap();
        assert_eq!(obs.len(), 0);
    }

    // ── Round-trip smoke test against ix-agent-core ───────────

    #[test]
    fn emitted_events_serialize_as_observation_added() {
        let input = serde_json::json!({"overallHealthScore": 85.0}).to_string();
        let obs = tars_to_observations(input.as_bytes(), 1).unwrap();
        for event in &obs {
            let json = serde_json::to_string(event).unwrap();
            assert!(
                json.contains(r#""kind":"observation_added""#),
                "expected observation_added tag, got {json}"
            );
            // Round-trip through the SessionEvent type.
            let back: SessionEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *event);
        }
    }
}
