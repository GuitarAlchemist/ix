//! Phase 2 acceptance — Target C grammar adapter end-to-end.
//!
//! Plan §Phase 2 acceptance: "100 iterations of SA, asserts
//! `final_reward > baseline_reward + 0.05` (some improvement) on a
//! deterministically-seeded run."
//!
//! Plus: "Target C runs sub-second per iter; 100 iters complete in < 30 s;
//! reward improves measurably."

use std::time::{Duration, Instant};

use tempfile::TempDir;

use ix_autoresearch::{
    run_experiment, Experiment, GrammarTarget, LogEvent, Strategy, TimeBudget,
};

#[test]
fn sa_finds_improvement_over_uniform_baseline_on_skewed_held_out() {
    // Skewed held-out (Q²-sum = 0.24); uniform baseline scores 1/6 ≈ 0.167.
    // SA should pull rule_weights toward larger weight on the high-frequency
    // rules and recover at least +0.05 reward.
    let dir = TempDir::new().unwrap();
    let mut target = GrammarTarget::default_smoke();

    // Compute baseline reward (deterministic, no eval cost — just calls evaluate once).
    let baseline_score = target
        .evaluate(
            &target.baseline(),
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    let baseline_reward = target.score_to_reward(&baseline_score);
    // Sanity: 1/6 + 0.1 * ess_stability_at_uniform.
    assert!(baseline_reward >= 1.0 / 6.0 - 1e-9);

    // Run 100 iterations of SA at moderate T₀; cooling ratio 0.95.
    let outcome = run_experiment(
        &mut target,
        Strategy::SimulatedAnnealing {
            initial_temperature: Some(0.05),
            cooling_rate: 0.95,
        },
        100,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        2026,
    )
    .unwrap();

    let final_reward = outcome.best_reward.expect("best_reward must be Some");
    let delta = final_reward - baseline_reward;
    assert!(
        delta > 0.05,
        "final reward should beat baseline by >0.05; baseline={baseline_reward:.4}, final={final_reward:.4}, delta={delta:.4}"
    );
    assert_eq!(outcome.iterations, 100);
}

#[test]
fn one_hundred_iters_complete_under_thirty_seconds() {
    // Plan acceptance: "Target C runs sub-second per iter; 100 iters
    // complete in < 30 s." Generous bound to absorb cold-build noise on CI.
    let dir = TempDir::new().unwrap();
    let mut target = GrammarTarget::default_smoke();
    let start = Instant::now();
    let outcome = run_experiment(
        &mut target,
        Strategy::Greedy,
        100,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        7,
    )
    .unwrap();
    let elapsed = start.elapsed();
    assert_eq!(outcome.iterations, 100);
    assert!(
        elapsed < Duration::from_secs(30),
        "100 grammar iters should finish in <30s; took {elapsed:?}"
    );
}

#[test]
fn run_log_carries_grammar_target_type_name() {
    // RunStart.target field includes the type name; verifies the kernel
    // is wiring our adapter, not a mock.
    let dir = TempDir::new().unwrap();
    let mut target = GrammarTarget::default_smoke();
    let outcome = run_experiment(
        &mut target,
        Strategy::Greedy,
        3,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        1,
    )
    .unwrap();
    let events: Vec<LogEvent<ix_autoresearch::GrammarConfig, ix_autoresearch::GrammarScore>> =
        ix_autoresearch::log::read_log(&outcome.log_path).unwrap();
    if let LogEvent::RunStart { target, .. } = &events[0] {
        assert!(
            target.contains("GrammarTarget"),
            "RunStart.target should name GrammarTarget, got {target}"
        );
    } else {
        panic!("first event should be RunStart");
    }
}

#[test]
fn eval_inputs_hash_is_present_in_run_start() {
    // The grammar adapter returns a content-hash of the held-out
    // distribution; verify it lands on RunStart.
    let dir = TempDir::new().unwrap();
    let mut target = GrammarTarget::default_smoke();
    let outcome = run_experiment(
        &mut target,
        Strategy::Greedy,
        2,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        3,
    )
    .unwrap();
    let events: Vec<LogEvent<ix_autoresearch::GrammarConfig, ix_autoresearch::GrammarScore>> =
        ix_autoresearch::log::read_log(&outcome.log_path).unwrap();
    if let LogEvent::RunStart {
        eval_inputs_hash, ..
    } = &events[0]
    {
        assert!(
            eval_inputs_hash.is_some(),
            "GrammarTarget should provide eval_inputs_hash"
        );
    } else {
        panic!("first event should be RunStart");
    }
}

#[test]
fn cost_ledger_reports_no_eval_failures_on_clean_run() {
    let dir = TempDir::new().unwrap();
    let mut target = GrammarTarget::default_smoke();
    let outcome = run_experiment(
        &mut target,
        Strategy::Greedy,
        20,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        99,
    )
    .unwrap();
    assert_eq!(outcome.cost.eval_failure_count, 0);
    // Some iterations should be rejected (Greedy + Gaussian noise on a
    // continuous landscape rejects more often than not at small steps).
    assert!(outcome.cost.rejected_count > 0);
}
