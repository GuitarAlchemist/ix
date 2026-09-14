//! Work-history distillation on a real snapshot: 1,255 PRs across
//! ix/ga/tars/Demerzel/hari/gaia, fetched 2026-09-14 by
//! `scripts/fetch-pr-lifecycle.sh`. Values below are pinned to that
//! fixture; re-fetching means re-deriving them.

use memristive_markov::work_history::{parse_jsonl, report, WorkHistoryReport};
use std::collections::BTreeMap;

const FIXTURE: &str = include_str!("fixtures/pr-lifecycle-2026-09-14.jsonl");

/// `from_state`/`to_state` enum of Demerzel `schemas/seldon/markov-transition.schema.json`.
const SCHEMA_STATES: [&str; 9] = [
    "issue.grooming",
    "issue.ready",
    "issue.delegated",
    "pr.draft",
    "pr.ready_for_review",
    "pr.merge_candidate",
    "pr.merged",
    "pr.rejected",
    "issue.stuck",
];

fn default_report() -> WorkHistoryReport {
    let prs = parse_jsonl(FIXTURE).unwrap();
    report(&prs, 14, 0.8, 3, 5).unwrap()
}

fn score(r: &WorkHistoryReport, model: &str) -> (f64, f64) {
    let s = r.held_out.scores.iter().find(|s| s.model == model).unwrap();
    (s.log_loss, s.accuracy)
}

#[test]
fn transitions_fit_the_seldon_schema() {
    let r = default_report();
    assert_eq!(r.prs, 1255);
    let mut out_mass: BTreeMap<(&str, Option<&str>), f64> = BTreeMap::new();
    for t in r.transitions.iter().chain(&r.transitions_by_worker) {
        assert!(SCHEMA_STATES.contains(&t.from_state.as_str()), "{t:?}");
        assert!(SCHEMA_STATES.contains(&t.to_state.as_str()), "{t:?}");
        assert!((0.0..=1.0).contains(&t.probability));
        assert!(t.sample_size >= 1);
        assert!(t.median_duration_hours.is_some_and(|h| h >= 0.0));
        *out_mass
            .entry((t.from_state.as_str(), t.worker.as_deref()))
            .or_default() += t.probability;
    }
    for (key, mass) in out_mass {
        assert!((mass - 1.0).abs() < 1e-9, "{key:?} sums to {mass}");
    }
}

#[test]
fn ready_prs_mostly_merge_and_stuck_ones_split() {
    let r = default_report();
    let get = |s: &str| r.absorption.iter().find(|a| a.state == s).unwrap();
    // 1,010 of 1,214 transitions out of ready_for_review go straight to merge.
    let ready = get("pr.ready_for_review");
    assert_eq!(ready.sample_size, 1214);
    assert!(ready.p_merged > 0.9, "{ready:?}");
    // Stuck PRs are far less likely to land.
    let stuck = get("issue.stuck");
    assert!(
        stuck.p_merged < 0.55 && stuck.p_rejected > 0.15,
        "{stuck:?}"
    );
    let draft = get("pr.draft");
    assert!(draft.p_merged < ready.p_merged && draft.p_merged > stuck.p_merged);
}

#[test]
fn vlmm_beats_first_order_on_held_out_transitions() {
    let r = default_report();
    let h = &r.held_out;
    assert_eq!((h.train_sequences, h.test_sequences), (1004, 251));
    assert_eq!(h.test_transitions, 361);
    let (m_loss, _) = score(&r, "order0_marginal");
    let (f_loss, f_acc) = score(&r, "first_order");
    let (v_loss, v_acc) = score(&r, "vlmm");
    assert!(f_loss < m_loss);
    assert!(v_loss < f_loss, "vlmm {v_loss} vs first-order {f_loss}");
    assert!(v_acc > f_acc);
    // Every disagreement goes VLMM's way (exact McNemar p ~ 2.4e-4). All 13
    // are the context [pr.draft, issue.stuck] -> pr.ready_for_review: a
    // stuck *draft* revives to review, where first-order predicts merge.
    assert_eq!((h.vlmm_only_correct, h.first_order_only_correct), (13, 0));
}

#[test]
fn report_is_deterministic() {
    let a = serde_json::to_string(&default_report()).unwrap();
    let b = serde_json::to_string(&default_report()).unwrap();
    assert_eq!(a, b);
}
