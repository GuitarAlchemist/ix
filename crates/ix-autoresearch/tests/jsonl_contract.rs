//! Phase 1 IX→Hari boundary contract test.
//!
//! Validates that the JSONL emitted by `run_experiment` against the
//! `GrammarTarget` smoke target conforms to the pinned schema in
//! `crates/ix-autoresearch/SCHEMA.md` and that the derived semantic
//! event view (as Hari would consume it) behaves per spec.
//!
//! Verifies the four acceptance criteria from
//! `agent-blackbox/docs/ix-real-problems-plan.md` Workflow 3:
//!
//! - Append-only event log (file order is event order).
//! - Deterministic replay: same seed ⇒ identical config_hash sequence.
//! - Contradictory findings preserved: same config_hash with differing
//!   `accepted` flags surfaces as `disposition: "contradictory"` in the
//!   derived view.
//! - Schema validation: every line matches the documented per-event
//!   shape.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

use serde_json::Value;
use tempfile::TempDir;

use ix_autoresearch::{
    log::read_log, run_experiment, GrammarConfig, GrammarScore, GrammarTarget, LogEvent, Strategy,
    TimeBudget, SCHEMA_VERSION,
};

const ITERS: usize = 20;
const SEED: u64 = 4242;

/// Run the canonical contract producer once. Returns (log_path, run_id).
fn produce_log(dir: &Path, seed: u64) -> std::path::PathBuf {
    let mut target = GrammarTarget::default_smoke();
    let outcome = run_experiment(
        &mut target,
        Strategy::SimulatedAnnealing {
            initial_temperature: Some(0.05),
            cooling_rate: 0.95,
        },
        ITERS,
        TimeBudget::soft(Duration::from_secs(5)),
        dir,
        seed,
    )
    .expect("run_experiment");
    outcome.log_path
}

#[test]
fn every_line_validates_against_pinned_schema() {
    let dir = TempDir::new().unwrap();
    let log_path = produce_log(dir.path(), SEED);
    let raw = std::fs::read_to_string(&log_path).expect("read log");
    let lines: Vec<&str> = raw.lines().filter(|l| !l.trim().is_empty()).collect();

    // Must be at least RunStart + N iterations + RunComplete.
    assert!(
        lines.len() >= ITERS + 2,
        "expected ≥{} lines, got {}",
        ITERS + 2,
        lines.len()
    );

    let mut seen_run_start = false;
    let mut iteration_count = 0;
    let mut seen_run_complete = false;
    let mut prev_iter: Option<u64> = None;

    for (i, line) in lines.iter().enumerate() {
        let v: Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {i} not valid JSON: {e}\n  line: {line}"));
        let event = v
            .get("event")
            .and_then(Value::as_str)
            .unwrap_or_else(|| panic!("line {i} missing `event` discriminator"));
        let schema_version = v
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or_else(|| panic!("line {i} missing `schema_version`"));
        assert_eq!(
            schema_version, SCHEMA_VERSION as u64,
            "line {i} schema_version mismatch"
        );

        match event {
            "run_start" => {
                assert!(!seen_run_start, "duplicate run_start at line {i}");
                assert!(
                    !seen_run_complete,
                    "run_start after run_complete at line {i}"
                );
                assert_eq!(i, 0, "run_start must be the first line");
                for req in [
                    "run_id",
                    "timestamp",
                    "target",
                    "strategy",
                    "seed",
                    "baseline_config",
                ] {
                    assert!(
                        v.get(req).is_some(),
                        "run_start line {i} missing required field `{req}`"
                    );
                }
                seen_run_start = true;
            }
            "iteration" => {
                assert!(seen_run_start, "iteration before run_start at line {i}");
                assert!(
                    !seen_run_complete,
                    "iteration after run_complete at line {i}"
                );
                for req in [
                    "iteration",
                    "timestamp",
                    "config",
                    "config_hash",
                    "accepted",
                    "elapsed_ms",
                ] {
                    assert!(
                        v.get(req).is_some(),
                        "iteration line {i} missing required field `{req}`"
                    );
                }
                let it = v
                    .get("iteration")
                    .and_then(Value::as_u64)
                    .expect("iteration usize");
                if let Some(p) = prev_iter {
                    assert!(
                        it > p,
                        "iteration numbers must be monotone-increasing (line {i}: {p} → {it})"
                    );
                }
                prev_iter = Some(it);
                iteration_count += 1;
            }
            "run_complete" => {
                assert!(!seen_run_complete, "duplicate run_complete at line {i}");
                assert_eq!(i, lines.len() - 1, "run_complete must be the last line");
                for req in ["timestamp", "iterations", "accepted"] {
                    assert!(
                        v.get(req).is_some(),
                        "run_complete line {i} missing required field `{req}`"
                    );
                }
                seen_run_complete = true;
            }
            other => panic!("line {i}: unknown event discriminator `{other}`"),
        }
    }

    assert!(seen_run_start, "log must contain a run_start");
    assert!(seen_run_complete, "log must contain a run_complete");
    assert_eq!(
        iteration_count, ITERS,
        "iteration count mismatch (expected {ITERS}, got {iteration_count})"
    );
}

#[test]
fn round_trips_through_typed_log_event() {
    // Round-tripping via `read_log` proves serde and the pinned schema agree.
    let dir = TempDir::new().unwrap();
    let log_path = produce_log(dir.path(), SEED);
    let events: Vec<LogEvent<GrammarConfig, GrammarScore>> =
        read_log(&log_path).expect("read_log parses every event");
    assert!(matches!(events.first(), Some(LogEvent::RunStart { .. })));
    assert!(matches!(events.last(), Some(LogEvent::RunComplete { .. })));
    let iter_count = events
        .iter()
        .filter(|e| matches!(e, LogEvent::Iteration { .. }))
        .count();
    assert_eq!(iter_count, ITERS);
}

#[test]
fn deterministic_replay_same_seed_produces_same_config_hash_sequence() {
    // Acceptance: "Replay is deterministic for the same log."
    // We assert that two fresh runs with the same seed produce the same
    // (iteration, config_hash, accepted) tuple sequence — the entropy
    // sources are seeded RNG only, so the candidate stream MUST be
    // bit-identical. Timestamps and run_id legitimately differ between
    // runs and are excluded from the comparison.
    let dir_a = TempDir::new().unwrap();
    let dir_b = TempDir::new().unwrap();
    let path_a = produce_log(dir_a.path(), SEED);
    let path_b = produce_log(dir_b.path(), SEED);

    let extract = |p: &Path| -> Vec<(u64, String, bool)> {
        let raw = std::fs::read_to_string(p).unwrap();
        raw.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .filter(|v| v.get("event").and_then(Value::as_str) == Some("iteration"))
            .map(|v| {
                let it = v.get("iteration").and_then(Value::as_u64).unwrap();
                let ch = v
                    .get("config_hash")
                    .and_then(Value::as_str)
                    .unwrap()
                    .to_string();
                let acc = v.get("accepted").and_then(Value::as_bool).unwrap();
                (it, ch, acc)
            })
            .collect()
    };

    let a = extract(&path_a);
    let b = extract(&path_b);
    assert_eq!(
        a, b,
        "same-seed runs must produce identical config_hash sequence"
    );
}

/// Derived semantic event view (per SCHEMA.md "Layer 2"). Produced by
/// the consumer from raw iteration lines; tested here to lock the
/// projection so consumers can verify against it.
#[derive(Debug, Clone, PartialEq)]
struct DerivedEvent {
    event_id: String,
    target: String,
    claim: String,
    confidence: f64,
    contradicted_by: Vec<String>,
    disposition: &'static str,
}

fn derive_events(log_path: &Path) -> Vec<DerivedEvent> {
    let raw = std::fs::read_to_string(log_path).unwrap();
    let mut run_id = String::new();
    let mut target = String::new();
    let mut out: Vec<DerivedEvent> = Vec::new();
    // claim -> Vec<(event_id, accepted)>, in order of appearance.
    let mut by_claim: BTreeMap<String, Vec<(String, bool)>> = BTreeMap::new();

    for line in raw.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line).unwrap();
        match v.get("event").and_then(Value::as_str) {
            Some("run_start") => {
                run_id = v
                    .get("run_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                target = v
                    .get("target")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
            }
            Some("iteration") => {
                let iter = v.get("iteration").and_then(Value::as_u64).unwrap();
                let event_id = format!("{run_id}/iteration-{iter}");
                let config_hash = v
                    .get("config_hash")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let short = &config_hash[..config_hash.len().min(12)];
                let claim = format!("{target}/config-{short}-is-an-improvement");
                let accepted = v.get("accepted").and_then(Value::as_bool).unwrap_or(false);
                let error_present = v
                    .get("error")
                    .and_then(Value::as_str)
                    .is_some_and(|s| !s.is_empty());

                let confidence = if accepted {
                    0.66
                } else if error_present {
                    0.10
                } else {
                    0.33
                };

                // Find priors with the same claim but differing `accepted`.
                let prior_disagrees: Vec<String> = by_claim
                    .get(&claim)
                    .map(|priors| {
                        priors
                            .iter()
                            .filter(|(_, prior_acc)| *prior_acc != accepted)
                            .map(|(id, _)| id.clone())
                            .collect()
                    })
                    .unwrap_or_default();

                let disposition = if !prior_disagrees.is_empty() {
                    "contradictory"
                } else if accepted {
                    "confirmed"
                } else if error_present {
                    "refuted"
                } else {
                    "pending"
                };

                by_claim
                    .entry(claim.clone())
                    .or_default()
                    .push((event_id.clone(), accepted));

                out.push(DerivedEvent {
                    event_id,
                    target: target.clone(),
                    claim,
                    confidence,
                    contradicted_by: prior_disagrees,
                    disposition,
                });
            }
            _ => {}
        }
    }

    out
}

#[test]
fn derived_event_view_event_ids_are_unique_and_ordered() {
    let dir = TempDir::new().unwrap();
    let log_path = produce_log(dir.path(), SEED);
    let derived = derive_events(&log_path);
    assert_eq!(derived.len(), ITERS);

    let mut seen: BTreeSet<String> = BTreeSet::new();
    for ev in &derived {
        assert!(
            seen.insert(ev.event_id.clone()),
            "duplicate event_id: {}",
            ev.event_id
        );
        // All event_ids start with the same run_id prefix.
        assert!(ev.event_id.contains("/iteration-"));
    }
}

#[test]
fn contradictory_findings_preserved_in_derived_view() {
    // Acceptance: "Contradictory findings are preserved as Contradictory,
    // not averaged away." We construct a tiny synthetic log on the fly
    // — two iteration lines with identical config_hash, differing
    // `accepted` — and assert the projection flags the second as
    // `disposition: contradictory` and lists the first in
    // `contradicted_by`.
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("synthetic.jsonl");

    let lines = [
        // run_start
        serde_json::json!({
            "event": "run_start",
            "schema_version": 1,
            "run_id": "test-run-0001",
            "timestamp": "2026-05-17T10:00:00Z",
            "target": "target_grammar",
            "strategy": {"kind": "Greedy"},
            "seed": 0,
            "git_sha": null,
            "git_sha_reason": "synthetic",
            "baseline_config": {"rule_weights": [0.5, 0.5], "temperature": 1.0},
            "eval_inputs_hash": null
        }),
        // accepted = true for config_hash X
        serde_json::json!({
            "event": "iteration",
            "schema_version": 1,
            "iteration": 0,
            "timestamp": "2026-05-17T10:00:01Z",
            "config": {"rule_weights": [0.6, 0.4], "temperature": 1.0},
            "config_hash": "autoresearch:samehash_aaaaaaaaaaaaaa",
            "score": {"parse_success_rate": 0.5, "ess_stability": 1.0},
            "reward": 0.6,
            "accepted": true,
            "previous_hash": null,
            "error": null,
            "elapsed_ms": 5,
            "strategy_state": null,
            "cache_hit": false
        }),
        // accepted = false for SAME config_hash X — contradiction.
        serde_json::json!({
            "event": "iteration",
            "schema_version": 1,
            "iteration": 1,
            "timestamp": "2026-05-17T10:00:02Z",
            "config": {"rule_weights": [0.6, 0.4], "temperature": 1.0},
            "config_hash": "autoresearch:samehash_aaaaaaaaaaaaaa",
            "score": {"parse_success_rate": 0.5, "ess_stability": 1.0},
            "reward": 0.55,
            "accepted": false,
            "previous_hash": null,
            "error": null,
            "elapsed_ms": 5,
            "strategy_state": null,
            "cache_hit": false
        }),
        serde_json::json!({
            "event": "run_complete",
            "schema_version": 1,
            "timestamp": "2026-05-17T10:00:03Z",
            "iterations": 2,
            "accepted": 1
        }),
    ];
    let body = lines
        .iter()
        .map(|v| serde_json::to_string(v).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&path, format!("{body}\n")).unwrap();

    let derived = derive_events(&path);
    assert_eq!(derived.len(), 2);
    assert_eq!(derived[0].disposition, "confirmed");
    assert!(derived[0].contradicted_by.is_empty());
    assert_eq!(derived[1].disposition, "contradictory");
    assert_eq!(
        derived[1].contradicted_by,
        vec![derived[0].event_id.clone()]
    );
    // Same canonical claim across both.
    assert_eq!(derived[0].claim, derived[1].claim);
}

#[test]
fn json_schema_file_is_present_and_well_formed() {
    let schema_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("jsonl-event.schema.json");
    let raw = std::fs::read_to_string(&schema_path)
        .unwrap_or_else(|e| panic!("missing {}: {e}", schema_path.display()));
    let parsed: Value = serde_json::from_str(&raw).expect("schema is valid JSON");
    assert_eq!(
        parsed.get("$schema").and_then(Value::as_str).unwrap_or(""),
        "https://json-schema.org/draft/2020-12/schema"
    );
    let one_of = parsed.get("oneOf").and_then(Value::as_array).unwrap();
    assert_eq!(
        one_of.len(),
        3,
        "schema must define run_start, iteration, run_complete"
    );
}
