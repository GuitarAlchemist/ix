//! Harness adapter — GitHub Actions workflow run summary →
//! `SessionEvent::ObservationAdded` stream.
//!
//! Third harness adapter after tars/cargo/clippy. Projects CI
//! outcomes from a standard input shape that callers assemble
//! from two GitHub API calls (workflow run + jobs).
//!
//! Spec: `demerzel/logic/harness-github-actions.md`.

use ix_agent_core::SessionEvent;
use ix_types::Hexavalent;
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const SOURCE: &str = "github-actions";

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("parse: {0}")]
    Parse(String),
}

/// Combined input shape: run summary + jobs array in one JSON.
/// Callers assemble this from GitHub API responses.
#[derive(Debug, Clone, Deserialize)]
pub struct Input {
    pub run: Run,
    #[serde(default)]
    pub jobs: Vec<Job>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Run {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub conclusion: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Job {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub conclusion: Option<String>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Step {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub conclusion: Option<String>,
}

/// Project a GitHub Actions run summary into observations.
pub fn github_actions_to_observations(
    input_bytes: &[u8],
    round: u32,
) -> Result<Vec<SessionEvent>, AdapterError> {
    let input: Input =
        serde_json::from_slice(input_bytes).map_err(|e| AdapterError::Parse(e.to_string()))?;
    let diagnosis_id = sha256_hex(input_bytes);

    let mut out: Vec<SessionEvent> = Vec::new();
    let mut ordinal: u64 = 0;

    // Rule 1: run-level reliability baseline
    let workflow_key = sanitize(&input.run.name);
    let run_claim = if workflow_key.is_empty() {
        "ci_run::reliable".to_string()
    } else {
        format!("ci_run:{workflow_key}::reliable")
    };
    let (variant, weight, evidence) = classify_run_conclusion(input.run.conclusion.as_deref());
    out.push(emit(
        &mut ordinal,
        &diagnosis_id,
        round,
        &run_claim,
        variant,
        weight,
        evidence,
    ));

    // Rule 2: per-job value observations
    for job in &input.jobs {
        if job.status != "completed" {
            // Not a terminal event; skip queued / in_progress jobs.
            continue;
        }
        let job_key = sanitize(&job.name);
        if job_key.is_empty() {
            continue;
        }
        if let Some((claim_tail, variant, weight)) =
            classify_job_conclusion(job.conclusion.as_deref())
        {
            let claim = format!("ci_job:{job_key}::{claim_tail}");
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                &claim,
                variant,
                weight,
                format!(
                    "job {job_name} → {conclusion}",
                    job_name = job.name,
                    conclusion = job
                        .conclusion
                        .clone()
                        .unwrap_or_else(|| "unknown".to_string()),
                ),
            ));
        }

        // Rule 3: slow job timely flag
        if let (Some(start), Some(end)) = (&job.started_at, &job.completed_at) {
            if let Some(duration) = duration_seconds(start, end) {
                if duration > 600.0 {
                    out.push(emit(
                        &mut ordinal,
                        &diagnosis_id,
                        round,
                        &format!("ci_job:{job_key}::timely"),
                        Hexavalent::Doubtful,
                        0.6,
                        format!("duration={}s", duration as u64),
                    ));
                }
            }
        }

        // Rule 4: step-level failures
        for step in &job.steps {
            if step.conclusion.as_deref() == Some("failure") {
                let step_key = sanitize(&step.name);
                if step_key.is_empty() {
                    continue;
                }
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    &format!("ci_step:{job_key}/{step_key}::valuable"),
                    Hexavalent::False,
                    0.9,
                    format!(
                        "step {step_name} in job {job_name} failed",
                        step_name = step.name,
                        job_name = job.name
                    ),
                ));
            }
        }
    }

    Ok(out)
}

fn classify_run_conclusion(conclusion: Option<&str>) -> (Hexavalent, f64, String) {
    let ev = conclusion.unwrap_or("missing").to_string();
    match conclusion {
        Some("success") => (Hexavalent::True, 0.9, ev),
        Some("failure") => (Hexavalent::False, 1.0, ev),
        Some("cancelled") => (Hexavalent::Unknown, 0.4, ev),
        Some("skipped") => (Hexavalent::Unknown, 0.3, ev),
        Some("timed_out") => (Hexavalent::False, 0.9, ev),
        Some("startup_failure") => (Hexavalent::False, 1.0, ev),
        _ => (Hexavalent::Unknown, 0.3, ev),
    }
}

fn classify_job_conclusion(conclusion: Option<&str>) -> Option<(&'static str, Hexavalent, f64)> {
    match conclusion {
        Some("success") => Some(("valuable", Hexavalent::True, 0.85)),
        Some("failure") => Some(("valuable", Hexavalent::False, 1.0)),
        Some("cancelled") => Some(("valuable", Hexavalent::Unknown, 0.4)),
        Some("timed_out") => Some(("timely", Hexavalent::False, 0.9)),
        Some("skipped") | None => None, // no signal
        _ => None,
    }
}

/// Sanitize a freeform name for claim_key embedding. Lowercase
/// ASCII alphanumerics + underscores. Strips leading/trailing
/// underscores.
fn sanitize(name: &str) -> String {
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

/// Parse an RFC3339-ish duration between two timestamps. Very
/// loose — we only want the delta in seconds, and we only care
/// to millisecond precision. Falls back to `None` on any parse
/// error.
fn duration_seconds(start: &str, end: &str) -> Option<f64> {
    let s = parse_epoch(start)?;
    let e = parse_epoch(end)?;
    Some((e - s).max(0.0))
}

/// Minimal RFC3339 parser — extracts the Y/M/D H:M:S components
/// and computes seconds-since-epoch via a fixed reference point.
/// Deliberately simplistic: no timezone handling (Actions uses Z),
/// no leap-second awareness, no millisecond fractional support.
fn parse_epoch(ts: &str) -> Option<f64> {
    // Expected format: "YYYY-MM-DDTHH:MM:SSZ"
    if ts.len() < 20 {
        return None;
    }
    let year: i64 = ts.get(0..4)?.parse().ok()?;
    let month: i64 = ts.get(5..7)?.parse().ok()?;
    let day: i64 = ts.get(8..10)?.parse().ok()?;
    let hour: i64 = ts.get(11..13)?.parse().ok()?;
    let minute: i64 = ts.get(14..16)?.parse().ok()?;
    let second: i64 = ts.get(17..19)?.parse().ok()?;

    // Naive days-from-epoch computation. We only compute deltas
    // between two timestamps in the same call, so accuracy drifts
    // are fine as long as both calls use the same rule.
    let days = days_from_civil(year, month, day);
    let total = days * 86400 + hour * 3600 + minute * 60 + second;
    Some(total as f64)
}

/// Howard Hinnant's days-from-civil algorithm. Returns days
/// since 1970-01-01. Works for any year >= 0.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * if m > 2 { m - 3 } else { m + 9 } + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
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

    fn extract(event: &SessionEvent) -> (&str, Hexavalent, f64) {
        if let SessionEvent::ObservationAdded {
            claim_key,
            variant,
            weight,
            ..
        } = event
        {
            (claim_key, *variant, *weight)
        } else {
            panic!("expected ObservationAdded")
        }
    }

    #[test]
    fn successful_run_emits_t_baseline() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "success"},
            "jobs": []
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        assert_eq!(obs.len(), 1);
        let (claim, variant, weight) = extract(&obs[0]);
        assert_eq!(claim, "ci_run:ci::reliable");
        assert_eq!(variant, Hexavalent::True);
        assert!((weight - 0.9).abs() < 1e-9);
    }

    #[test]
    fn failed_run_emits_f_baseline_at_full_weight() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "failure"},
            "jobs": []
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        let (_, variant, weight) = extract(&obs[0]);
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cancelled_run_emits_unknown() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "cancelled"},
            "jobs": []
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        let (_, variant, _) = extract(&obs[0]);
        assert_eq!(variant, Hexavalent::Unknown);
    }

    #[test]
    fn per_job_observations_for_completed_jobs() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "failure"},
            "jobs": [
                {"name": "build", "status": "completed", "conclusion": "success"},
                {"name": "test", "status": "completed", "conclusion": "failure"}
            ]
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        // Expect: run + 2 jobs = 3 observations
        assert_eq!(obs.len(), 3);
        let build_obs = obs
            .iter()
            .find(|e| extract(e).0 == "ci_job:build::valuable")
            .unwrap();
        let (_, variant, _) = extract(build_obs);
        assert_eq!(variant, Hexavalent::True);
        let test_obs = obs
            .iter()
            .find(|e| extract(e).0 == "ci_job:test::valuable")
            .unwrap();
        let (_, variant, weight) = extract(test_obs);
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn in_progress_jobs_are_skipped() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "success"},
            "jobs": [
                {"name": "build", "status": "in_progress", "conclusion": null}
            ]
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        // Only the run observation — in_progress job is skipped.
        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn timed_out_job_emits_timely_false() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "failure"},
            "jobs": [
                {"name": "long", "status": "completed", "conclusion": "timed_out"}
            ]
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        let timely = obs
            .iter()
            .find(|e| extract(e).0 == "ci_job:long::timely")
            .unwrap();
        let (_, variant, weight) = extract(timely);
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 0.9).abs() < 1e-9);
    }

    #[test]
    fn slow_job_emits_additional_timely_doubtful() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "success"},
            "jobs": [
                {
                    "name": "integration",
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-04-11T12:00:00Z",
                    "completed_at": "2026-04-11T12:15:00Z"
                }
            ]
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        // run + job valuable + job timely = 3
        assert_eq!(obs.len(), 3);
        let timely = obs
            .iter()
            .find(|e| extract(e).0 == "ci_job:integration::timely")
            .unwrap();
        let (_, variant, weight) = extract(timely);
        assert_eq!(variant, Hexavalent::Doubtful);
        assert!((weight - 0.6).abs() < 1e-9);
    }

    #[test]
    fn fast_job_does_not_emit_timely() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "success"},
            "jobs": [
                {
                    "name": "build",
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-04-11T12:00:00Z",
                    "completed_at": "2026-04-11T12:01:00Z"
                }
            ]
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        // Only run + job valuable = 2 (no timely for 1-minute job)
        assert_eq!(obs.len(), 2);
    }

    #[test]
    fn failed_step_emits_step_observation() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "failure"},
            "jobs": [
                {
                    "name": "test",
                    "status": "completed",
                    "conclusion": "failure",
                    "steps": [
                        {"name": "Run tests", "conclusion": "success"},
                        {"name": "Upload coverage", "conclusion": "failure"}
                    ]
                }
            ]
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        let step = obs
            .iter()
            .find(|e| extract(e).0 == "ci_step:test/upload_coverage::valuable")
            .unwrap();
        let (_, variant, weight) = extract(step);
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 0.9).abs() < 1e-9);
    }

    #[test]
    fn sanitize_lowercases_and_replaces_punctuation() {
        assert_eq!(sanitize("Build & Test"), "build___test");
        assert_eq!(sanitize("  CI  "), "ci");
        assert_eq!(sanitize("deploy-prod"), "deploy_prod");
    }

    #[test]
    fn malformed_json_is_a_parse_error() {
        let err = github_actions_to_observations(b"{not json", 0).unwrap_err();
        assert!(matches!(err, AdapterError::Parse(_)));
    }

    #[test]
    fn round_trip_through_session_event() {
        let input = serde_json::json!({
            "run": {"name": "CI", "conclusion": "success"},
            "jobs": []
        })
        .to_string();
        let obs = github_actions_to_observations(input.as_bytes(), 1).unwrap();
        for event in &obs {
            let json = serde_json::to_string(event).unwrap();
            let back: SessionEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *event);
        }
    }
}
