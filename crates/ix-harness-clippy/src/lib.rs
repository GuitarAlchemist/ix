//! Harness adapter — `cargo clippy --message-format=json` output →
//! `SessionEvent::ObservationAdded` stream.
//!
//! Third evidential-shape adapter (after tars/cargo). Focuses on
//! the static-analysis signal axis — "is the code written
//! correctly" rather than "does the code behave correctly."
//!
//! Spec: `demerzel/logic/harness-clippy.md`.

use ix_agent_core::SessionEvent;
use ix_types::Hexavalent;
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub const SOURCE: &str = "clippy";

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("input is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

#[derive(Debug, Clone, Deserialize)]
struct ClippyMessage {
    #[serde(default)]
    reason: String,
    #[serde(default)]
    message: Option<Diagnostic>,
}

#[derive(Debug, Clone, Deserialize)]
struct Diagnostic {
    #[serde(default)]
    level: String,
    #[serde(default)]
    code: Option<LintCode>,
    #[serde(default)]
    message: String,
}

#[derive(Debug, Clone, Deserialize)]
struct LintCode {
    #[serde(default)]
    code: String,
}

/// Project a clippy JSON stream into
/// [`SessionEvent::ObservationAdded`] records.
pub fn clippy_to_observations(input: &[u8], round: u32) -> Result<Vec<SessionEvent>, AdapterError> {
    let text = std::str::from_utf8(input)?;
    let diagnosis_id = sha256_hex(input);

    // Collect only clippy diagnostics.
    let diagnostics: Vec<Diagnostic> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<ClippyMessage>(line).ok())
        .filter(|m| m.reason == "compiler-message")
        .filter_map(|m| m.message)
        .filter(|d| {
            d.code
                .as_ref()
                .map(|c| c.code.starts_with("clippy::"))
                .unwrap_or(false)
        })
        .collect();

    let error_count = diagnostics.iter().filter(|d| d.level == "error").count();
    let warning_count = diagnostics.iter().filter(|d| d.level == "warning").count();

    let mut out: Vec<SessionEvent> = Vec::new();
    let mut ordinal: u64 = 0;

    // Aggregate reliability observation first.
    let (variant, weight, evidence) = classify_run(error_count, warning_count);
    out.push(emit(
        &mut ordinal,
        &diagnosis_id,
        round,
        "clippy_run::reliable",
        variant,
        weight,
        evidence,
    ));

    // Per-lint observations.
    for d in &diagnostics {
        let lint_name = d.code.as_ref().map(|c| c.code.clone()).unwrap_or_default();
        if let Some((aspect, variant, weight)) = classify_diagnostic(&d.level, &lint_name) {
            let claim_key = format!("clippy:{lint_name}::{aspect}");
            let evidence = format!("{}: {}", d.level, d.message);
            out.push(emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                &claim_key,
                variant,
                weight,
                evidence,
            ));
        }
    }

    Ok(out)
}

fn classify_run(errors: usize, warnings: usize) -> (Hexavalent, f64, String) {
    let evidence = format!("{errors} error(s), {warnings} warning(s)");
    if errors > 0 {
        (Hexavalent::False, 1.0, evidence)
    } else if warnings > 20 {
        (Hexavalent::Doubtful, 0.7, evidence)
    } else if warnings > 0 {
        (Hexavalent::Probable, 0.7, evidence)
    } else {
        (Hexavalent::True, 0.9, evidence)
    }
}

/// Classify one diagnostic into `(aspect, variant, weight)`.
/// Returns `None` for `help` and unknown levels.
fn classify_diagnostic(level: &str, lint_name: &str) -> Option<(&'static str, Hexavalent, f64)> {
    match level {
        "error" => Some(("safe", Hexavalent::False, 1.0)),
        "warning" => {
            // Inspect the lint path to pick aspect + weight.
            if lint_name.contains("::correctness::") {
                Some(("safe", Hexavalent::Doubtful, 0.8))
            } else if lint_name.contains("::suspicious::") {
                Some(("safe", Hexavalent::Doubtful, 0.7))
            } else if lint_name.contains("::perf::") {
                Some(("timely", Hexavalent::Doubtful, 0.6))
            } else {
                // style / complexity / pedantic / nursery / default
                Some(("valuable", Hexavalent::Doubtful, 0.5))
            }
        }
        "note" => Some(("valuable", Hexavalent::Unknown, 0.3)),
        _ => None, // help, unknown levels
    }
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
            panic!("expected ObservationAdded")
        }
    }

    #[test]
    fn empty_input_emits_clean_baseline() {
        let obs = clippy_to_observations(b"", 0).unwrap();
        assert_eq!(obs.len(), 1);
        let (_, claim, variant, weight) = extract(&obs[0]);
        assert_eq!(claim, "clippy_run::reliable");
        assert_eq!(variant, Hexavalent::True);
        assert!((weight - 0.9).abs() < 1e-9);
    }

    #[test]
    fn error_level_emits_safe_false() {
        let input = r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"clippy::panicking_unwrap"},"message":"unwrap on const None"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        // Summary + per-lint observation
        assert_eq!(obs.len(), 2);

        // Summary is F (errors > 0)
        let (_, _, sv, _) = extract(&obs[0]);
        assert_eq!(sv, Hexavalent::False);

        // Per-lint is safe F
        let (_, claim, variant, weight) = extract(&obs[1]);
        assert_eq!(claim, "clippy:clippy::panicking_unwrap::safe");
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 1.0).abs() < 1e-9);
    }

    #[test]
    fn style_warning_emits_valuable_doubtful() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::style::needless_return"},"message":"needless return"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let lint_obs = obs
            .iter()
            .find(|e| extract(e).1.contains("needless_return"))
            .unwrap();
        let (_, claim, variant, _) = extract(lint_obs);
        assert!(claim.ends_with("::valuable"));
        assert_eq!(variant, Hexavalent::Doubtful);
    }

    #[test]
    fn correctness_warning_emits_safe_doubtful() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::correctness::float_cmp"},"message":"strict float comparison"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let lint_obs = obs
            .iter()
            .find(|e| extract(e).1.contains("float_cmp"))
            .unwrap();
        let (_, claim, variant, weight) = extract(lint_obs);
        assert!(claim.ends_with("::safe"));
        assert_eq!(variant, Hexavalent::Doubtful);
        assert!((weight - 0.8).abs() < 1e-9);
    }

    #[test]
    fn perf_warning_emits_timely_aspect() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::perf::needless_collect"},"message":"unnecessary collect"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let lint_obs = obs
            .iter()
            .find(|e| extract(e).1.contains("needless_collect"))
            .unwrap();
        let (_, claim, _, _) = extract(lint_obs);
        assert!(claim.ends_with("::timely"), "got {claim}");
    }

    #[test]
    fn non_clippy_diagnostics_are_ignored() {
        // A plain rustc warning (no clippy:: prefix).
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"unused_variables"},"message":"unused variable"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        // Only the summary observation (clean baseline, no
        // clippy diagnostics seen).
        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn help_level_is_ignored() {
        let input = r#"{"reason":"compiler-message","message":{"level":"help","code":{"code":"clippy::style::foo"},"message":"suggestion"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        // Only summary; help is skipped.
        assert_eq!(obs.len(), 1);
    }

    #[test]
    fn warning_count_summary_scales_with_count() {
        // 10 warnings → P variant (1-20 warnings band)
        let lines: Vec<String> = (0..10)
            .map(|i| format!(
                r#"{{"reason":"compiler-message","message":{{"level":"warning","code":{{"code":"clippy::style::w{i}"}},"message":"w"}}}}"#
            ))
            .collect();
        let input = lines.join("\n");
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let (_, _, sv, _) = extract(&obs[0]);
        assert_eq!(sv, Hexavalent::Probable);
    }

    #[test]
    fn many_warnings_emit_doubtful_summary() {
        let lines: Vec<String> = (0..25)
            .map(|i| format!(
                r#"{{"reason":"compiler-message","message":{{"level":"warning","code":{{"code":"clippy::style::w{i}"}},"message":"w"}}}}"#
            ))
            .collect();
        let input = lines.join("\n");
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        let (_, _, sv, _) = extract(&obs[0]);
        assert_eq!(sv, Hexavalent::Doubtful);
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let input = concat!(
            "garbage line\n",
            r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"clippy::panic"},"message":"panic"}}"#,
            "\n",
        );
        let obs = clippy_to_observations(input.as_bytes(), 0).unwrap();
        assert_eq!(obs.len(), 2); // summary + 1 diagnostic
    }

    #[test]
    fn round_trip_through_session_event() {
        let input = r#"{"reason":"compiler-message","message":{"level":"warning","code":{"code":"clippy::style::foo"},"message":"w"}}"#;
        let obs = clippy_to_observations(input.as_bytes(), 1).unwrap();
        for event in &obs {
            let json = serde_json::to_string(event).unwrap();
            let back: SessionEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *event);
        }
    }
}
