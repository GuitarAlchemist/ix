//! Work-history distillation on a real snapshot: 1,255 PRs across
//! ix/ga/tars/Demerzel/hari/gaia, fetched 2026-09-14 by
//! `scripts/fetch-pr-lifecycle.sh`. Values below are pinned to that
//! fixture; they are snapshot tripwires, not general properties, and
//! re-fetching means re-deriving them.

use memristive_markov::work_history::{
    parse_jsonl, report, WorkHistoryReport, MIN_WORKER_OUTGOING, SMOOTHING_SWEEP,
};
use std::collections::BTreeMap;

const FIXTURE: &str = include_str!("fixtures/pr-lifecycle-2026-09-14.jsonl");

/// `from_state`/`to_state` enum of Demerzel `schemas/seldon/markov-transition.schema.json`,
/// including the origin-aware stuck states proposed in Demerzel#1088.
const SCHEMA_STATES: [&str; 11] = [
    "issue.grooming",
    "issue.ready",
    "issue.delegated",
    "pr.draft",
    "pr.ready_for_review",
    "pr.merge_candidate",
    "pr.merged",
    "pr.rejected",
    "issue.stuck",
    "pr.stuck_draft",
    "pr.stuck_ready",
];

fn default_report() -> WorkHistoryReport {
    let prs = parse_jsonl(FIXTURE).unwrap();
    report(&prs, 14, 0.8, 3, 5).unwrap()
}

#[test]
fn transitions_fit_the_seldon_schema() {
    let r = default_report();
    assert_eq!(r.prs, 1255);
    let mut out_mass: BTreeMap<(&str, Option<&str>), (f64, u64)> = BTreeMap::new();
    for t in r.transitions.iter().chain(&r.transitions_by_worker) {
        assert!(SCHEMA_STATES.contains(&t.from_state.as_str()), "{t:?}");
        assert!(SCHEMA_STATES.contains(&t.to_state.as_str()), "{t:?}");
        assert!((0.0..=1.0).contains(&t.probability));
        assert!(t.sample_size >= 1);
        // Stuck durations are the threshold, so they are not published.
        assert_eq!(
            t.median_duration_hours.is_some(),
            !t.to_state.contains("stuck"),
            "{t:?}"
        );
        let e = out_mass
            .entry((t.from_state.as_str(), t.worker.as_deref()))
            .or_default();
        e.0 += t.probability;
        e.1 += t.sample_size;
    }
    for ((from, worker), (mass, n)) in out_mass {
        assert!(
            (mass - 1.0).abs() < 1e-9,
            "{from} {worker:?} sums to {mass}"
        );
        if worker.is_some() {
            assert!(n >= MIN_WORKER_OUTGOING, "{from} {worker:?} has {n}");
        }
    }
}

#[test]
fn no_merge_candidate_to_ready_regression() {
    // Merge-candidate labels applied to drafts used to create
    // draft -> merge_candidate -> ready_for_review paths (10 PRs).
    let r = default_report();
    assert!(!r
        .transitions
        .iter()
        .any(|t| t.from_state == "pr.merge_candidate" && t.to_state == "pr.ready_for_review"));
    assert_eq!(
        (r.ready_flips.ready_flips, r.ready_flips.merged_within_60s),
        (138, 75)
    );
}

#[test]
fn absorption_snapshot_values() {
    let r = default_report();
    let get = |s: &str| r.absorption.iter().find(|a| a.state == s).unwrap();
    let ready = get("pr.ready_for_review");
    assert_eq!(ready.sample_size, 1204);
    assert!(ready.p_merged > 0.91, "{ready:?}");
    // Stuck states carry censoring: many are still open at as_of, so compare
    // the conditional merge rate, not p_merged. Both are what the first-order
    // chain implies, not observed outcomes: all 17 stuck drafts that were
    // later flipped ready merged. Observed rates (stuck_draft 17/21 = 0.810,
    // stuck_ready 24/35 = 0.686) fall outside this range, which pins the
    // chain values on this fixture.
    for s in ["pr.stuck_draft", "pr.stuck_ready"] {
        let a = get(s);
        assert!(a.p_unresolved > 0.2, "{a:?}");
        let resolved = a.p_merged_given_resolved.unwrap();
        assert!((0.7..0.76).contains(&resolved), "{a:?}");
        assert!(resolved < ready.p_merged_given_resolved.unwrap());
    }
}

#[test]
fn held_out_snapshot_values() {
    let r = default_report();
    let h = &r.held_out;
    assert_eq!((h.train_sequences, h.test_sequences), (1004, 251));
    assert_eq!(h.test_transitions, 355);
    let score = |m: &str| h.scores.iter().find(|s| s.model == m).unwrap();
    let (marginal, first, vlmm) = (
        score("order0_marginal"),
        score("first_order"),
        score("vlmm"),
    );
    // With origin-aware stuck states VLMM adds nothing: same accuracy, no
    // disagreement. The log-loss ordering below is a snapshot value of the
    // 80/20 split only: across stale {7,14,30} x split {0.7,0.8,0.9} x
    // (order, min obs) {(3,5),(2,5),(3,2),(3,10)}, VLMM never beats
    // first-order on accuracy, but it has the lower log-loss at every
    // epsilon at 0.9 (all 12 settings). A flip here after a refetch or a
    // split change is not a code regression.
    assert!(first.accuracy > marginal.accuracy);
    assert_eq!(first.accuracy, vlmm.accuracy);
    assert_eq!(
        (
            h.disagreement.vlmm_only_correct,
            h.disagreement.first_order_only_correct
        ),
        (0, 0)
    );
    assert_eq!(h.disagreement.cluster_sign_test_p, 1.0);
    assert_eq!(first.log_loss.len(), SMOOTHING_SWEEP.len());
    for (f, v) in first.log_loss.iter().zip(&vlmm.log_loss) {
        assert!(f.log_loss < v.log_loss, "eps {}: {f:?} vs {v:?}", f.epsilon);
    }
}

#[test]
fn report_is_deterministic() {
    let a = serde_json::to_string(&default_report()).unwrap();
    let b = serde_json::to_string(&default_report()).unwrap();
    assert_eq!(a, b);
}
