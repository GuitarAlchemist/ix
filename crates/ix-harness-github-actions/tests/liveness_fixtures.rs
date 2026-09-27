//! Liveness verdicts over recorded `gh run list` history.
//!
//! `fixtures/ix-ga-nightly-quality.runs.json` was captured 2026-09-14 with
//! `gh run list -R GuitarAlchemist/ix --workflow ga-nightly-quality.yml
//! --limit 30 --json createdAt,event,status,conclusion,headBranch`
//! (all events, so the pull-request runs that pass in between stay in).

use ix_harness_github_actions::liveness::{assess, RunRecord, WorkflowRuns};
use ix_types::Hexavalent;

const NOW: &str = "2026-09-14T12:00:00Z";

fn ga_nightly_quality(runs: Vec<RunRecord>) -> WorkflowRuns {
    WorkflowRuns {
        repo: "GuitarAlchemist/ix".to_string(),
        workflow: "GA Nightly Quality".to_string(),
        path: Some(".github/workflows/ga-nightly-quality.yml".to_string()),
        state: Some("active".to_string()),
        crons: Some(vec!["15 5 * * *".to_string()]),
        schedule_since: None,
        runs,
        runs_truncated: false,
        fetch_error: None,
    }
}

fn recorded() -> Vec<RunRecord> {
    serde_json::from_str(include_str!("fixtures/ix-ga-nightly-quality.runs.json")).unwrap()
}

#[test]
fn test_ga_nightly_quality_four_night_failure_streak_is_false() {
    // The miss: four scheduled nights failed in a row, a passing PR run in
    // between, nobody reacted.
    let window: Vec<RunRecord> = recorded()
        .into_iter()
        .filter(|r| r.created_at.as_str() >= "2026-09-11")
        .collect();
    assert_eq!(window.iter().filter(|r| r.event == "schedule").count(), 4);
    assert!(window
        .iter()
        .any(|r| r.event == "pull_request" && r.conclusion.as_deref() == Some("success")));

    let r = assess(&ga_nightly_quality(window), NOW, 0)
        .unwrap()
        .unwrap();
    assert_eq!(r.failure_streak, 4);
    assert_eq!(r.verdict, Hexavalent::False);
    assert!(r.is_unhealthy());
    assert_eq!(
        r.claim_key,
        "gha_loop:guitaralchemist:ix:ga_nightly_quality::reliable"
    );
}

#[test]
fn test_ga_nightly_quality_full_window_never_succeeds_on_schedule() {
    let r = assess(&ga_nightly_quality(recorded()), NOW, 0)
        .unwrap()
        .unwrap();
    assert_eq!(r.scheduled_runs, 22, "8 pull_request runs are excluded");
    assert_eq!(r.failure_streak, 22);
    assert!(r.no_success_in_window);
    assert_eq!(r.last_success_age_hours, None);
    assert!(
        (r.cadence_hours - 24.0).abs() < 2.0,
        "nightly cadence inferred"
    );
    assert_eq!(r.verdict, Hexavalent::False);
    // Two sources, one claim; the schedule still fires (T@0.3) while the
    // outcome is F@1.0, so the merge synthesizes C but does not escalate.
    assert_eq!(r.observations.len(), 2);
    assert!(r.distribution["C"] > 0.0);
    assert!(r.distribution["F"] > r.distribution["C"]);
}
