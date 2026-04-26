//! Harness adapter — `cargo test --format=json` output →
//! `SessionEvent::ObservationAdded` stream.
//!
//! The second reference implementation of the harness-adapter
//! pattern. Where `ix-harness-tars` projects a diagnostic
//! snapshot (time series, resource metrics), this adapter
//! projects an **evidential** stream: discrete test pass/fail
//! events that together attest to code behavior.
//!
//! The projection rules are owned by
//! `demerzel/logic/harness-cargo.md`. This crate mechanically
//! applies them.
//!
//! # What this is NOT
//!
//! - Not a cargo invocation wrapper. The adapter reads cargo's
//!   JSON output; running `cargo test` is the caller's job.
//! - Not a compiler error handler. `cargo build` errors are a
//!   separate producer and need their own adapter.
//! - Not a clippy wrapper. Clippy has its own JSON shape and
//!   its own projection spec (future `ix-harness-clippy`).

use std::collections::BTreeMap;

use ix_agent_core::SessionEvent;
use ix_types::Hexavalent;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Fixed source name for cargo observations. Must match the
/// Demerzel governance doc.
pub const SOURCE: &str = "cargo";

/// Errors produced by [`cargo_to_observations`].
#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    /// Input was not valid UTF-8.
    #[error("input is not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

/// A single cargo event parsed from one JSONL line.
///
/// cargo emits a heterogeneous stream — test events, suite
/// events, and occasional stray output. This enum covers the
/// events the adapter cares about; everything else is silently
/// ignored.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CargoEvent {
    /// A per-test event. `event` is one of "started", "ok",
    /// "failed", "ignored", "bench", or unknown strings (future
    /// cargo versions may add variants).
    Test {
        #[serde(default)]
        event: String,
        #[serde(default)]
        name: String,
        #[serde(default)]
        exec_time: Option<f64>,
    },
    /// Suite-level event with aggregate counts.
    Suite {
        #[serde(default)]
        event: String,
        #[serde(default)]
        passed: u32,
        #[serde(default)]
        failed: u32,
        #[serde(default)]
        #[allow(dead_code)]
        ignored: u32,
    },
}

/// Project a cargo JSON stream into
/// [`SessionEvent::ObservationAdded`] records.
///
/// Per the spec in `demerzel/logic/harness-cargo.md`:
/// 1. Suite-level observation is emitted FIRST (bottom-line
///    signal before per-test detail)
/// 2. Per-test observations in the order cargo emitted them
/// 3. Slow tests (`exec_time > 5.0`) emit an additional `::timely`
///    observation
///
/// Malformed JSONL lines are skipped (not errors). Unknown event
/// variants and unknown fields are silently ignored.
pub fn cargo_to_observations(input: &[u8], round: u32) -> Result<Vec<SessionEvent>, AdapterError> {
    // Fail loudly on non-UTF-8 bytes (cargo always writes UTF-8
    // JSON), but tolerate JSON parse failures line-by-line.
    let text = std::str::from_utf8(input)?;
    let diagnosis_id = sha256_hex(input);

    // Walk the stream twice — once to find the suite summary, once
    // for per-test events. Rationale: the governance spec requires
    // suite observations to come first in the output, but cargo
    // emits the suite event LAST (after all tests have run). A
    // single pass would either buffer everything or emit out of
    // order.
    let events: Vec<CargoEvent> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<CargoEvent>(line).ok())
        .collect();

    let mut out: Vec<SessionEvent> = Vec::new();
    let mut ordinal: u64 = 0;

    // Suite-level observation first.
    if let Some(suite) = find_suite_event(&events) {
        let (variant, weight, evidence) = classify_suite(&suite);
        out.push(emit(
            &mut ordinal,
            &diagnosis_id,
            round,
            "cargo_suite::reliable",
            variant,
            weight,
            evidence,
        ));
    }

    // Per-test observations in stream order.
    for event in &events {
        if let CargoEvent::Test {
            event: status,
            name,
            exec_time,
        } = event
        {
            if name.is_empty() {
                continue;
            }
            if let Some((variant, weight, evidence)) = classify_test_event(status) {
                let claim_key = test_claim_key(name, "valuable");
                out.push(emit(
                    &mut ordinal,
                    &diagnosis_id,
                    round,
                    &claim_key,
                    variant,
                    weight,
                    evidence.to_string(),
                ));
            }
            // Slow-test timely flag
            if let Some(t) = exec_time {
                if *t > 5.0 {
                    let claim_key = test_claim_key(name, "timely");
                    out.push(emit(
                        &mut ordinal,
                        &diagnosis_id,
                        round,
                        &claim_key,
                        Hexavalent::Doubtful,
                        0.6,
                        format!("exec_time={t}s"),
                    ));
                }
            }
        }
    }

    Ok(out)
}

/// Classify one `{"type":"test"}` event into an optional
/// `(variant, weight, evidence)` triple. Returns `None` for
/// non-terminal events (`started`) and unknown statuses.
fn classify_test_event(status: &str) -> Option<(Hexavalent, f64, &'static str)> {
    match status {
        "ok" => Some((Hexavalent::True, 0.9, "ok")),
        "failed" => Some((Hexavalent::False, 1.0, "failed")),
        "ignored" => Some((Hexavalent::Unknown, 0.3, "ignored")),
        "bench" => Some((Hexavalent::Probable, 0.6, "bench")),
        _ => None,
    }
}

/// Classify a suite-level event into `(variant, weight, evidence)`.
/// Uses the pass ratio for partial-failure shading.
fn classify_suite(suite: &SuiteSummary) -> (Hexavalent, f64, String) {
    let total = suite.passed + suite.failed;
    let evidence = format!("passed {} of {}", suite.passed, total);
    if suite.event == "ok" || (suite.failed == 0 && suite.passed > 0) {
        return (Hexavalent::True, 0.9, evidence);
    }
    if total == 0 {
        // No tests ran — no signal.
        return (Hexavalent::Unknown, 0.3, evidence);
    }
    let passed_frac = suite.passed as f64 / total as f64;
    if suite.passed == 0 {
        (Hexavalent::False, 1.0, evidence)
    } else if passed_frac > 0.9 {
        (Hexavalent::Doubtful, 0.7, evidence)
    } else if passed_frac > 0.5 {
        (Hexavalent::Doubtful, 0.8, evidence)
    } else {
        (Hexavalent::False, 0.9, evidence)
    }
}

/// Normalized suite summary extracted from a CargoEvent::Suite.
struct SuiteSummary {
    event: String,
    passed: u32,
    failed: u32,
}

/// Find the last `{"type":"suite"}` event in the stream. cargo
/// emits one at the end of every run; if multiple fire (e.g.,
/// multiple test binaries in one invocation), we take the last.
fn find_suite_event(events: &[CargoEvent]) -> Option<SuiteSummary> {
    events.iter().rev().find_map(|e| match e {
        CargoEvent::Suite {
            event,
            passed,
            failed,
            ..
        } => Some(SuiteSummary {
            event: event.clone(),
            passed: *passed,
            failed: *failed,
        }),
        _ => None,
    })
}

/// Build a claim_key for a test observation. Uses `test:` as the
/// action prefix and the full test name as the action body, so
/// per-test observations don't collide with tool-name observations.
///
/// The aspect is appended with `::` so `rfind("::")` in the merge
/// function's `action_and_aspect` helper cleanly recovers the
/// aspect (since test names may contain `::` in module paths).
fn test_claim_key(name: &str, aspect: &str) -> String {
    format!("test:{name}::{aspect}")
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

/// Aggregate statistics about a cargo run. Useful for consumers
/// that want a quick summary without walking the observation list.
#[derive(Debug, Clone, Default)]
pub struct CargoStats {
    pub total_tests: u32,
    pub passed: u32,
    pub failed: u32,
    pub ignored: u32,
    pub slow: u32,
}

/// Compute summary stats from a parsed cargo stream. Uses a
/// `BTreeMap` keyed by test name to deduplicate re-emitted events
/// (cargo sometimes emits `started` followed by `ok` for the same
/// test; we only count terminal events).
pub fn compute_stats(input: &[u8]) -> Result<CargoStats, AdapterError> {
    let text = std::str::from_utf8(input)?;
    let mut by_name: BTreeMap<String, String> = BTreeMap::new();
    let mut slow: u32 = 0;
    for line in text.lines() {
        if let Ok(CargoEvent::Test {
            event,
            name,
            exec_time,
        }) = serde_json::from_str::<CargoEvent>(line)
        {
            if matches!(event.as_str(), "ok" | "failed" | "ignored" | "bench") {
                by_name.insert(name.clone(), event);
                if let Some(t) = exec_time {
                    if t > 5.0 {
                        slow += 1;
                    }
                }
            }
        }
    }
    let mut stats = CargoStats::default();
    for (_, status) in by_name {
        stats.total_tests += 1;
        match status.as_str() {
            "ok" | "bench" => stats.passed += 1,
            "failed" => stats.failed += 1,
            "ignored" => stats.ignored += 1,
            _ => {}
        }
    }
    stats.slow = slow;
    Ok(stats)
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

    // ── All-pass suite ────────────────────────────────────────

    #[test]
    fn all_pass_suite_emits_positive_baseline_and_test_observations() {
        let input = concat!(
            r#"{"type":"test","event":"ok","name":"foo::test_a","exec_time":0.01}"#,
            "\n",
            r#"{"type":"test","event":"ok","name":"foo::test_b","exec_time":0.02}"#,
            "\n",
            r#"{"type":"suite","event":"ok","passed":2,"failed":0,"ignored":0,"exec_time":0.5}"#,
            "\n",
        );
        let obs = cargo_to_observations(input.as_bytes(), 1).expect("parses");

        // 3 observations: 1 suite + 2 tests
        assert_eq!(obs.len(), 3);

        // Suite first
        let (_src, claim, variant, weight) = extract(&obs[0]);
        assert_eq!(claim, "cargo_suite::reliable");
        assert_eq!(variant, Hexavalent::True);
        assert!((weight - 0.9).abs() < 1e-9);

        // Per-test observations
        let (_, claim1, var1, _) = extract(&obs[1]);
        assert_eq!(claim1, "test:foo::test_a::valuable");
        assert_eq!(var1, Hexavalent::True);

        let (_, claim2, var2, _) = extract(&obs[2]);
        assert_eq!(claim2, "test:foo::test_b::valuable");
        assert_eq!(var2, Hexavalent::True);
    }

    // ── Mixed suite with partial failure ──────────────────────

    #[test]
    fn partial_failure_suite_emits_doubtful_baseline() {
        // 9 passed, 1 failed → pass ratio = 0.9 → D with weight 0.7
        let mut lines = Vec::new();
        for i in 0..9 {
            lines.push(format!(r#"{{"type":"test","event":"ok","name":"t_{i}"}}"#));
        }
        lines.push(r#"{"type":"test","event":"failed","name":"t_fail"}"#.to_string());
        lines.push(
            r#"{"type":"suite","event":"failed","passed":9,"failed":1,"ignored":0}"#.to_string(),
        );
        let input = lines.join("\n");
        let obs = cargo_to_observations(input.as_bytes(), 1).expect("parses");

        let (_, claim, variant, weight) = extract(&obs[0]);
        assert_eq!(claim, "cargo_suite::reliable");
        assert_eq!(variant, Hexavalent::Doubtful);
        // 0.9 ratio falls in the ">0.5 but !>0.9" band → weight 0.8
        // Since 9/10 == 0.9 exactly, it falls in the ">0.5" band.
        assert!((weight - 0.8).abs() < 1e-9, "expected 0.8, got {weight}");
    }

    #[test]
    fn total_failure_suite_emits_f_at_full_weight() {
        let input = concat!(
            r#"{"type":"test","event":"failed","name":"a"}"#,
            "\n",
            r#"{"type":"test","event":"failed","name":"b"}"#,
            "\n",
            r#"{"type":"suite","event":"failed","passed":0,"failed":2,"ignored":0}"#,
            "\n",
        );
        let obs = cargo_to_observations(input.as_bytes(), 1).expect("parses");
        let (_, _, variant, weight) = extract(&obs[0]);
        assert_eq!(variant, Hexavalent::False);
        assert!((weight - 1.0).abs() < 1e-9);
    }

    // ── Ignored tests ─────────────────────────────────────────

    #[test]
    fn ignored_test_emits_unknown_with_low_weight() {
        let input = concat!(
            r#"{"type":"test","event":"ignored","name":"flaky"}"#,
            "\n",
            r#"{"type":"suite","event":"ok","passed":0,"failed":0,"ignored":1}"#,
            "\n",
        );
        let obs = cargo_to_observations(input.as_bytes(), 1).expect("parses");

        let test_obs = obs
            .iter()
            .find(|e| extract(e).1 == "test:flaky::valuable")
            .unwrap();
        let (_, _, variant, weight) = extract(test_obs);
        assert_eq!(variant, Hexavalent::Unknown);
        assert!((weight - 0.3).abs() < 1e-9);
    }

    // ── Slow test flags timely ────────────────────────────────

    #[test]
    fn slow_test_emits_additional_timely_observation() {
        let input = concat!(
            r#"{"type":"test","event":"ok","name":"slow_integration","exec_time":7.5}"#,
            "\n",
            r#"{"type":"suite","event":"ok","passed":1,"failed":0,"ignored":0}"#,
            "\n",
        );
        let obs = cargo_to_observations(input.as_bytes(), 1).expect("parses");
        // Expected: 1 suite + 1 valuable + 1 timely = 3
        assert_eq!(obs.len(), 3);

        let timely = obs
            .iter()
            .find(|e| extract(e).1 == "test:slow_integration::timely")
            .expect("expected timely observation for slow test");
        let (_, _, variant, weight) = extract(timely);
        assert_eq!(variant, Hexavalent::Doubtful);
        assert!((weight - 0.6).abs() < 1e-9);
    }

    // ── Deep module path handles rfind ────────────────────────

    #[test]
    fn deep_module_path_in_test_name_parses_correctly() {
        let input = concat!(
            r#"{"type":"test","event":"ok","name":"ix_math::eigen::jacobi::basic_test"}"#,
            "\n",
            r#"{"type":"suite","event":"ok","passed":1,"failed":0}"#,
            "\n",
        );
        let obs = cargo_to_observations(input.as_bytes(), 0).expect("parses");
        let test = obs
            .iter()
            .find(|e| extract(e).1.starts_with("test:ix_math::eigen::"))
            .expect("test with module path should be emitted");
        // The claim_key is the full path + ::valuable suffix.
        let (_, claim, _, _) = extract(test);
        assert_eq!(claim, "test:ix_math::eigen::jacobi::basic_test::valuable");
    }

    // ── Malformed input tolerance ─────────────────────────────

    #[test]
    fn malformed_lines_are_skipped_not_errors() {
        let input = concat!(
            "this is not JSON\n",
            r#"{"type":"test","event":"ok","name":"valid"}"#,
            "\n",
            "{another garbage line\n",
            r#"{"type":"suite","event":"ok","passed":1,"failed":0}"#,
            "\n",
        );
        let obs = cargo_to_observations(input.as_bytes(), 0).expect("should tolerate garbage");
        // Should still emit the suite + 1 valid test.
        assert_eq!(obs.len(), 2);
    }

    #[test]
    fn empty_input_emits_nothing() {
        let obs = cargo_to_observations(b"", 0).expect("empty is ok");
        assert_eq!(obs.len(), 0);
    }

    #[test]
    fn utf8_failure_is_a_hard_error() {
        // 0xFF is not valid UTF-8 under any encoding.
        let bad = &[0xFFu8, 0xFE, 0xFD];
        let err = cargo_to_observations(bad, 0).expect_err("should reject non-UTF-8");
        assert!(matches!(err, AdapterError::Utf8(_)));
    }

    // ── Determinism ───────────────────────────────────────────

    #[test]
    fn same_input_produces_same_diagnosis_id() {
        let input = concat!(
            r#"{"type":"test","event":"ok","name":"a"}"#,
            "\n",
            r#"{"type":"suite","event":"ok","passed":1,"failed":0}"#,
            "\n",
        );
        let obs1 = cargo_to_observations(input.as_bytes(), 0).unwrap();
        let obs2 = cargo_to_observations(input.as_bytes(), 0).unwrap();
        assert_eq!(obs1, obs2);
    }

    // ── compute_stats ─────────────────────────────────────────

    #[test]
    fn compute_stats_counts_terminal_events_only() {
        let input = concat!(
            r#"{"type":"test","event":"started","name":"a"}"#,
            "\n",
            r#"{"type":"test","event":"ok","name":"a"}"#,
            "\n",
            r#"{"type":"test","event":"failed","name":"b"}"#,
            "\n",
            r#"{"type":"test","event":"ignored","name":"c"}"#,
            "\n",
            r#"{"type":"test","event":"ok","name":"d","exec_time":6.0}"#,
            "\n",
        );
        let stats = compute_stats(input.as_bytes()).unwrap();
        assert_eq!(stats.total_tests, 4);
        assert_eq!(stats.passed, 2); // a, d
        assert_eq!(stats.failed, 1); // b
        assert_eq!(stats.ignored, 1); // c
        assert_eq!(stats.slow, 1); // d
    }

    // ── Round-trip through SessionEvent ──────────────────────

    #[test]
    fn emitted_events_serialize_round_trip() {
        let input = concat!(
            r#"{"type":"test","event":"ok","name":"a"}"#,
            "\n",
            r#"{"type":"suite","event":"ok","passed":1,"failed":0}"#,
            "\n",
        );
        let obs = cargo_to_observations(input.as_bytes(), 1).unwrap();
        for event in &obs {
            let json = serde_json::to_string(event).unwrap();
            assert!(json.contains(r#""kind":"observation_added""#));
            let back: SessionEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *event);
        }
    }
}
