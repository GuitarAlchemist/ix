//! Tests for the Methodology Guard usefulness + anti-process metrics (ix#219).
//!
//! Two rules govern this file.
//!
//! **A test that cannot fail is not a test.** Every mechanical check here is
//! exercised by seeding the violation it exists to catch and asserting it
//! fires, and each is paired with the same fixture minus the seed, asserting
//! it goes quiet. A check that fires on everything would pass the first half
//! alone.
//!
//! **A metric that cannot discriminate is not a metric.** The load-bearing
//! test is [`the_three_real_cases_land_in_three_different_classes`], and its
//! control is [`mechanism_revert_a_verdict_only_classifier_collapses_case_one_and_two`],
//! which rebuilds the drafted verdict-only metric and shows it gives the same
//! answer for two runs that are not the same.

use std::collections::BTreeSet;

use chrono::Utc;
use ix_quality_trend::gate_ledger::{
    EvidenceKind, GateDecision, GateEvidence, GateLedgerEntry, GateMetric, LedgerLine, OperatorAck,
};
use ix_quality_trend::guard_metrics::{
    assess, delta, measure, provenance, reason_fidelity, run_from_entry, runs_from_ledger,
    self_probe, AntiProcessClass, AntiProcessThresholds, GuardMetrics, GuardRun, Outcome,
    Provenance, ReasonFidelity, RunClass, Verdict,
};

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_string()).collect()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

// ---------------------------------------------------------------------------
// The three real cases, as fixtures
// ---------------------------------------------------------------------------

/// Case 1. An agent tried to edit `.github/workflows/**`, which is in
/// `agent-blackbox.policy.json`'s `blocked_paths`. The gate fired, cited the
/// file it was actually blocking on, and the change was held for a human
/// label. Working as designed.
fn case_1_useful_block() -> GuardRun {
    GuardRun {
        id: "case-1-blocked-path".to_string(),
        gate: "agent-blackbox/risk-report".to_string(),
        decision: GateDecision::Fail,
        verdict: Verdict::Correct,
        outcome: Outcome::Blocked,
        cited_evidence: strs(&[".github/workflows/agent-blackbox.yml"]),
        gated_changeset: strs(&[".github/workflows/agent-blackbox.yml", "crates/foo/src/lib.rs"]),
        reviewed_changeset: strs(&[
            ".github/workflows/agent-blackbox.yml",
            "crates/foo/src/lib.rs",
        ]),
        ack: None,
    }
}

/// Case 2, ix#316. The gate blocked, and blocking was the right call — but the
/// paths it cited as its evidence have zero commits in this repository. The
/// verdict is right; the reason is invented.
///
/// Note both changesets are the *same* here, isolating the reason axis. The
/// real incident had both defects at once; case 3 isolates the other.
fn case_2_correct_verdict_false_reason() -> GuardRun {
    GuardRun {
        id: "case-2-ix316-reason".to_string(),
        gate: "agent-blackbox/risk-report".to_string(),
        decision: GateDecision::Fail,
        verdict: Verdict::Correct,
        outcome: Outcome::Blocked,
        cited_evidence: strs(&["src/policy/blocked.rs", "src/agent/runner.rs"]),
        gated_changeset: strs(&["crates/ix-duck/sql/pareto_frontier.sql", "Cargo.toml"]),
        reviewed_changeset: strs(&["crates/ix-duck/sql/pareto_frontier.sql", "Cargo.toml"]),
        ack: None,
    }
}

/// Case 3, the other half of ix#316. `enforce --report dist/risk-report.json`
/// gated on an artifact listing 12 files while the `risk-report.md` humans
/// read listed 25. The gate and the reviewer were looking at different
/// changes.
fn case_3_provenance_break() -> GuardRun {
    GuardRun {
        id: "case-3-ix316-provenance".to_string(),
        gate: "agent-blackbox/risk-report".to_string(),
        decision: GateDecision::Fail,
        verdict: Verdict::Correct,
        outcome: Outcome::Blocked,
        cited_evidence: strs(&["Cargo.toml"]),
        gated_changeset: strs(&["Cargo.toml", "crates/a/src/lib.rs"]),
        reviewed_changeset: strs(&[
            "Cargo.toml",
            "crates/a/src/lib.rs",
            "crates/b/src/lib.rs",
            "docs/plans/2026-09-07-x.md",
        ]),
        ack: None,
    }
}

#[test]
fn the_three_real_cases_land_in_three_different_classes() {
    let one = assess(&case_1_useful_block());
    let two = assess(&case_2_correct_verdict_false_reason());
    let three = assess(&case_3_provenance_break());

    assert_eq!(one.class, RunClass::UsefulBlock);
    assert_eq!(two.class, RunClass::CorrectVerdictFalseReason);
    assert_eq!(three.class, RunClass::ProvenanceBreak);

    let distinct: BTreeSet<RunClass> = [one.class, two.class, three.class].into_iter().collect();
    assert_eq!(
        distinct.len(),
        3,
        "a usefulness metric that cannot distinguish these three cases is not \
         measuring anything"
    );
}

#[test]
fn mechanism_revert_a_verdict_only_classifier_collapses_case_one_and_two() {
    // The drafted metric, rebuilt: false_positive_rate over human verdicts
    // alone. This is the control. It must give the SAME answer for cases 1
    // and 2 — that is the defect — while the three-axis assessment must not.
    fn verdict_only(run: &GuardRun) -> &'static str {
        match run.verdict {
            Verdict::Correct => "true-positive",
            Verdict::Incorrect => "false-positive",
            Verdict::Unadjudicated => "ungraded",
        }
    }

    let one = case_1_useful_block();
    let two = case_2_correct_verdict_false_reason();

    assert_eq!(
        verdict_only(&one),
        verdict_only(&two),
        "the drafted verdict-only metric is supposed to collapse these two; if \
         this assertion ever fails, the control has stopped being a control"
    );
    assert_eq!(
        verdict_only(&two),
        "true-positive",
        "and it scores ix#316 as a SUCCESS, which is the whole problem"
    );

    assert_ne!(
        assess(&one).class,
        assess(&two).class,
        "the three-axis assessment must separate what the verdict axis cannot"
    );
}

#[test]
fn provenance_break_outranks_a_correct_verdict_and_a_sound_reason() {
    // Case 3's verdict is Correct and its cited path is genuinely in the gated
    // changeset. Only the provenance axis objects — and it must win, because a
    // verdict about a changeset the gate never saw is not a verdict.
    let run = case_3_provenance_break();
    assert_eq!(reason_fidelity(&run), ReasonFidelity::Sound);
    assert_eq!(run.verdict, Verdict::Correct);
    assert_eq!(assess(&run).class, RunClass::ProvenanceBreak);
}

#[test]
fn a_fabricated_reason_without_human_adjudication_is_not_called_a_correct_verdict() {
    let mut run = case_2_correct_verdict_false_reason();
    run.verdict = Verdict::Unadjudicated;
    assert_eq!(assess(&run).class, RunClass::FabricatedReason);
}

// ---------------------------------------------------------------------------
// Seeded violations, each paired with its own removal
// ---------------------------------------------------------------------------

#[test]
fn seeding_a_path_that_is_not_in_the_gated_changeset_is_caught() {
    let mut run = case_1_useful_block();
    run.cited_evidence.push("src/never-committed.rs".to_string());

    match reason_fidelity(&run) {
        ReasonFidelity::Fabricated { paths } => {
            assert_eq!(paths, vec!["src/never-committed.rs".to_string()]);
        }
        other => panic!("seeded fabrication not caught: {other:?}"),
    }
    assert_eq!(assess(&run).class, RunClass::CorrectVerdictFalseReason);
}

#[test]
fn removing_the_seeded_path_makes_the_fidelity_check_go_quiet() {
    // The other half of the pair: a checker that fires on everything would
    // pass the seeded test above on its own.
    let run = case_1_useful_block();
    assert_eq!(reason_fidelity(&run), ReasonFidelity::Sound);
    assert_eq!(assess(&run).class, RunClass::UsefulBlock);
}

#[test]
fn seeding_one_extra_file_on_the_reviewed_side_is_caught() {
    let mut run = case_1_useful_block();
    run.reviewed_changeset.push("docs/seeded.md".to_string());

    match provenance(&run) {
        Provenance::Diverged {
            only_gated,
            only_reviewed,
        } => {
            assert!(only_gated.is_empty());
            assert_eq!(only_reviewed, vec!["docs/seeded.md".to_string()]);
        }
        other => panic!("seeded provenance divergence not caught: {other:?}"),
    }
    assert_eq!(assess(&run).class, RunClass::ProvenanceBreak);
}

#[test]
fn seeding_one_extra_file_on_the_gated_side_is_caught_too() {
    // Divergence is symmetric: the gate seeing a file the reviewer did not is
    // as broken as the reverse, and ix#316 was this direction — the enforced
    // JSON listed files the reviewed report did not.
    let mut run = case_1_useful_block();
    run.gated_changeset.push("crates/ghost/src/lib.rs".to_string());

    match provenance(&run) {
        Provenance::Diverged { only_gated, .. } => {
            assert_eq!(only_gated, vec!["crates/ghost/src/lib.rs".to_string()]);
        }
        other => panic!("seeded provenance divergence not caught: {other:?}"),
    }
}

#[test]
fn removing_the_seeded_file_makes_the_provenance_check_go_quiet() {
    assert_eq!(provenance(&case_1_useful_block()), Provenance::Agreed);
}

#[test]
fn an_unrecorded_changeset_reads_as_unknown_and_never_as_clean() {
    // The failure mode this guards against: an absent artifact scoring the
    // same as a verified one.
    let mut run = case_1_useful_block();
    run.gated_changeset.clear();
    assert_eq!(reason_fidelity(&run), ReasonFidelity::Unknown);
    assert_eq!(provenance(&run), Provenance::Unknown);

    let m = measure(&[run]);
    assert_eq!(
        m.reason_fidelity_rate, None,
        "an unverifiable window must report unknown, not a perfect score"
    );
    assert_eq!(m.provenance_agreement_rate, None);
}

#[test]
fn the_shipped_self_probe_watches_its_own_checkers_fail() {
    let p = self_probe();
    assert!(p.fidelity_check_bites);
    assert!(p.provenance_check_bites);
    assert!(p.clean_run_stays_clean);
    assert!(p.all_pass());
}

// ---------------------------------------------------------------------------
// Anti-process
// ---------------------------------------------------------------------------

fn window(fired: usize, silent: usize, sanctioned: usize, clean: usize) -> Vec<GuardRun> {
    let blocked = fired - silent - sanctioned;
    let mut runs = Vec::new();
    let mut push = |n: usize, decision: GateDecision, outcome: Outcome, tag: &str| {
        for i in 0..n {
            runs.push(GuardRun {
                id: format!("{tag}-{i}"),
                gate: "agent-blackbox/risk-report".to_string(),
                decision,
                verdict: Verdict::Unadjudicated,
                outcome,
                cited_evidence: Vec::new(),
                gated_changeset: Vec::new(),
                reviewed_changeset: Vec::new(),
                ack: None,
            });
        }
    };
    push(blocked, GateDecision::Fail, Outcome::Blocked, "blocked");
    push(silent, GateDecision::Fail, Outcome::MergedWithoutAck, "silent");
    push(
        sanctioned,
        GateDecision::Fail,
        Outcome::OverriddenWithAck,
        "acked",
    );
    push(clean, GateDecision::Pass, Outcome::Clean, "pass");
    runs
}

#[test]
fn a_gate_nobody_has_watched_fail_is_not_a_gate() {
    // Perfect numbers on every other axis. It still does not pass, because
    // nobody has seeded a violation and watched it caught.
    let m = measure(&window(4, 0, 0, 96));
    let ap = m.anti_process(&AntiProcessThresholds::default());
    assert_eq!(ap.class, AntiProcessClass::Unwatched);
    assert!(ap.reason.contains("not a gate"), "{}", ap.reason);
}

#[test]
fn recording_the_seed_probe_is_what_lifts_the_unwatched_verdict() {
    // Mechanism control for the test above: the ONLY difference is the probe.
    let base = window(4, 0, 0, 96);
    let t = AntiProcessThresholds::default();
    assert_eq!(
        measure(&base).anti_process(&t).class,
        AntiProcessClass::Unwatched
    );
    assert_eq!(
        measure(&base).with_seed_probe(true).anti_process(&t).class,
        AntiProcessClass::Discriminating
    );
    assert_eq!(
        measure(&base).with_seed_probe(false).anti_process(&t).class,
        AntiProcessClass::Unwatched,
        "a probe that ran and did not catch the seed is worse than no probe, \
         never better"
    );
}

#[test]
fn a_gate_that_always_fires_has_stopped_discriminating() {
    let m = measure(&window(100, 0, 0, 0)).with_seed_probe(true);
    let ap = m.anti_process(&AntiProcessThresholds::default());
    assert_eq!(ap.class, AntiProcessClass::NotDiscriminating);
    assert!(ap.reason.contains("always fires"), "{}", ap.reason);
}

#[test]
fn a_gate_that_never_fires_has_stopped_discriminating_too() {
    let m = measure(&window(0, 0, 0, 100)).with_seed_probe(true);
    let ap = m.anti_process(&AntiProcessThresholds::default());
    assert_eq!(ap.class, AntiProcessClass::NotDiscriminating);
    assert!(ap.reason.contains("silent"), "{}", ap.reason);
}

#[test]
fn too_few_runs_reports_insufficient_rather_than_a_rate() {
    let m = measure(&window(1, 0, 0, 2)).with_seed_probe(true);
    assert_eq!(
        m.anti_process(&AntiProcessThresholds::default()).class,
        AntiProcessClass::Insufficient
    );
}

#[test]
fn empty_window_rates_are_unknown_not_zero() {
    let m = measure(&[]);
    assert_eq!(m.runs, 0);
    assert_eq!(m.fire_rate, None);
    assert_eq!(m.silent_bypass_rate, None);
    assert_eq!(m.adjudication_coverage, None);
}

// ---------------------------------------------------------------------------
// The measured anchor: ix's own Agent Blackbox history
// ---------------------------------------------------------------------------
//
// Sampled 2026-09-08 over the 120 most recent GuitarAlchemist/ix pull
// requests; 119 carried an Agent Blackbox check. 40 runs were red. 29 of those
// merged anyway. Only 4 red runs ever carried the `agent-blackbox-reviewed`
// label, so 25 merged with no acknowledgement of any kind and 11 were held.
//
// Split at 2026-07-20 into two windows of 59 and 60:
//   older  n=59  red=21  merged-despite-red=20  labelled=0
//   newer  n=60  red=19  merged-despite-red=9   labelled=4

fn ix_history_older() -> Vec<GuardRun> {
    window(21, 20, 0, 38)
}

fn ix_history_newer() -> Vec<GuardRun> {
    window(19, 5, 4, 41)
}

fn ix_history_total() -> Vec<GuardRun> {
    let mut v = ix_history_older();
    v.extend(ix_history_newer());
    v
}

#[test]
fn the_measured_ix_history_reproduces_the_published_rates() {
    let m = measure(&ix_history_total());
    assert_eq!(m.runs, 119);
    assert_eq!(m.fired, 40);
    assert!(close(m.fire_rate.unwrap(), 40.0 / 119.0));
    assert!(close(m.silent_bypass_rate.unwrap(), 25.0 / 40.0));
    assert!(close(m.sanctioned_override_rate.unwrap(), 4.0 / 40.0));
}

#[test]
fn on_measured_ix_history_the_risk_report_gate_classifies_as_ceremonial() {
    let m = measure(&ix_history_total()).with_seed_probe(true);
    let ap = m.anti_process(&AntiProcessThresholds::default());
    assert_eq!(ap.class, AntiProcessClass::Ceremonial);
    assert!(ap.reason.contains("62.5%"), "{}", ap.reason);
}

#[test]
fn folding_acknowledged_overrides_into_silent_bypass_hides_the_ceremonial_verdict() {
    // Mechanism control for splitting bypass in two. If all 29 merges-despite-
    // red were counted as sanctioned overrides, the gate reads as governed.
    let governed_reading = window(40, 0, 29, 79);
    let m = measure(&governed_reading).with_seed_probe(true);
    assert_eq!(
        m.anti_process(&AntiProcessThresholds::default()).class,
        AntiProcessClass::Discriminating,
        "collapsing the two bypass kinds is what makes an ignored gate look fine"
    );
}

#[test]
fn the_boundary_between_the_two_windows_shows_movement_the_total_hides() {
    let d = delta(&ix_history_older(), &ix_history_newer());
    let t = AntiProcessThresholds::default();

    // The cumulative number says Ceremonial.
    assert_eq!(
        measure(&ix_history_total())
            .with_seed_probe(true)
            .anti_process(&t)
            .class,
        AntiProcessClass::Ceremonial
    );
    // The older window alone is far worse than the total suggests.
    assert!(close(d.before.silent_bypass_rate.unwrap(), 20.0 / 21.0));
    // The newer window alone is already under the threshold.
    assert!(close(d.after.silent_bypass_rate.unwrap(), 5.0 / 19.0));
    assert_eq!(
        d.after
            .clone()
            .with_seed_probe(true)
            .anti_process(&t)
            .class,
        AntiProcessClass::Discriminating,
        "the boundary says the gate is recovering; the total says it is ceremony"
    );

    assert!(d.silent_bypass_delta.unwrap() < 0.0);
    assert!(d.sanctioned_override_delta.unwrap() > 0.0);
}

#[test]
fn a_delta_is_unknown_when_either_side_is_unknown() {
    let d = delta(&[], &ix_history_newer());
    assert_eq!(d.fire_rate_delta, None);
    assert_eq!(d.silent_bypass_delta, None);
}

// ---------------------------------------------------------------------------
// Adjudication coverage
// ---------------------------------------------------------------------------

#[test]
fn an_ungraded_window_reads_as_ungraded_rather_than_scoring_well() {
    let m = measure(&ix_history_total());
    assert_eq!(
        m.adjudication_coverage,
        Some(0.0),
        "40 fires, none of them judged by a human"
    );
    assert_eq!(
        m.useful_block_rate, None,
        "with no adjudicated fires there is no useful-block rate to report"
    );
}

#[test]
fn adjudicating_some_fires_moves_coverage_without_inventing_a_verdict() {
    let mut runs = ix_history_total();
    for r in runs.iter_mut().filter(|r| r.fired()).take(10) {
        r.verdict = Verdict::Correct;
        r.cited_evidence = strs(&["a.rs"]);
        r.gated_changeset = strs(&["a.rs"]);
        r.reviewed_changeset = strs(&["a.rs"]);
    }
    let m = measure(&runs);
    assert!(close(m.adjudication_coverage.unwrap(), 10.0 / 40.0));
    assert!(close(m.useful_block_rate.unwrap(), 1.0));
    assert_eq!(m.class_counts.get(&RunClass::UsefulBlock), Some(&10));
}

// ---------------------------------------------------------------------------
// Ledger bridge — reuse of the shipped substrate, not a second store
// ---------------------------------------------------------------------------

fn entry(decision: GateDecision) -> GateLedgerEntry {
    GateLedgerEntry::new(
        "agent-blackbox",
        "risk-report",
        decision,
        GateMetric {
            name: "risk_score".to_string(),
            value: 1.0,
            threshold: Some(0.6),
            trend: None,
        },
    )
}

#[test]
fn guard_fields_ride_in_the_existing_extra_slot() {
    let mut e = entry(GateDecision::Fail);
    e.extra = Some(serde_json::json!({
        "guard": {
            "cited_evidence": ["src/ghost.rs"],
            "gated_changeset": ["Cargo.toml"],
            "reviewed_changeset": ["Cargo.toml"],
            "verdict": "correct",
            "outcome": "blocked"
        }
    }));

    let run = run_from_entry(&e);
    assert_eq!(run.gate, "agent-blackbox/risk-report");
    assert_eq!(run.verdict, Verdict::Correct);
    assert_eq!(run.outcome, Outcome::Blocked);
    assert_eq!(assess(&run).class, RunClass::CorrectVerdictFalseReason);
}

#[test]
fn an_entry_with_no_guard_block_still_lifts_and_never_guesses_a_silent_bypass() {
    let e = entry(GateDecision::Fail);
    let run = run_from_entry(&e);
    assert_eq!(
        run.outcome,
        Outcome::Blocked,
        "the ledger cannot see whether the change landed, so it must not claim \
         a bypass"
    );
    assert_eq!(run.verdict, Verdict::Unadjudicated);
    assert_eq!(reason_fidelity(&run), ReasonFidelity::Unknown);
}

#[test]
fn an_operator_ack_lifts_to_a_sanctioned_override() {
    let mut e = entry(GateDecision::Fail);
    e.operator_ack = Some(OperatorAck {
        by: "spareilleux".to_string(),
        at: Utc::now(),
        note: Some("agent-blackbox-reviewed".to_string()),
    });
    assert_eq!(run_from_entry(&e).outcome, Outcome::OverriddenWithAck);
}

#[test]
fn a_lone_evidence_ref_becomes_the_cited_path() {
    let mut e = entry(GateDecision::Fail);
    e.evidence = Some(GateEvidence::new(EvidenceKind::File, "src/ghost.rs"));
    e.extra = Some(serde_json::json!({ "guard": { "gated_changeset": ["Cargo.toml"] } }));
    let run = run_from_entry(&e);
    assert_eq!(run.cited_evidence, strs(&["src/ghost.rs"]));
    assert!(matches!(
        reason_fidelity(&run),
        ReasonFidelity::Fabricated { .. }
    ));
}

#[test]
fn legacy_v0_rows_do_not_inflate_the_denominator() {
    let lines = vec![
        LedgerLine::V1(Box::new(entry(GateDecision::Fail))),
        LedgerLine::LegacyV0(serde_json::json!({ "pr": 42, "gate": "chatbot-qa" })),
        LedgerLine::V1(Box::new(entry(GateDecision::Pass))),
    ];
    let runs = runs_from_ledger(&lines);
    assert_eq!(runs.len(), 2);
    assert_eq!(measure(&runs).runs, 2);
}

#[test]
fn metrics_round_trip_through_json() {
    let m = measure(&ix_history_total()).with_seed_probe(true);
    let text = serde_json::to_string(&m).expect("serialise");
    let back: GuardMetrics = serde_json::from_str(&text).expect("deserialise");
    assert_eq!(back.runs, m.runs);
    assert_eq!(back.fired, m.fired);
    assert_eq!(back.seeded_violation_caught, Some(true));
    assert_eq!(back.class_counts, m.class_counts);
}
