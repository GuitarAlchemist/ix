//! Phase 5 acceptance — Target A OPTIC-K adapter (mocked rebuild).
//!
//! Plan §Phase 5 acceptance: 30 iters with mock; validates schema
//! parsing and Dirichlet-on-simplex perturbation correctness. The
//! adapter is "code-complete on the IX side"; live runs are gated on
//! the GA-side `--weights-config` flag (Phase 6).

use std::time::{Duration, Instant};

use tempfile::TempDir;

use ix_autoresearch::{
    run_experiment, Experiment, LogEvent, OpticKConfig, OpticKScore, OpticKTarget, Strategy,
    TimeBudget,
};

#[test]
fn dirichlet_perturbation_preserves_simplex_across_thirty_iterations() {
    let dir = TempDir::new().unwrap();
    let mut target = OpticKTarget::default_smoke();

    // Run 30 SA iterations and verify *every accepted iter* has a
    // simplex-valid config (sum to 1 within 1e-6, no negatives).
    let outcome = run_experiment(
        &mut target,
        Strategy::SimulatedAnnealing {
            initial_temperature: Some(1e-3),
            cooling_rate: 0.95,
        },
        30,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        2026,
    )
    .unwrap();

    let events: Vec<LogEvent<OpticKConfig, OpticKScore>> =
        ix_autoresearch::log::read_log(&outcome.log_path).unwrap();

    let mut iter_count = 0usize;
    for ev in &events {
        if let LogEvent::Iteration { config, .. } = ev {
            iter_count += 1;
            // Sum invariant.
            let sum = config.sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "iteration config must sum to ≈ 1.0 (got {sum})"
            );
            // Non-negativity.
            for &w in &config.as_array() {
                assert!(w >= 0.0, "weight must be ≥ 0; got {w}");
            }
        }
    }
    assert!(iter_count == 30);
}

#[test]
fn evaluate_at_optimum_returns_zero_leak_through_kernel_path() {
    // End-to-end smoke through the kernel: feed the true optimum, run
    // 1 iteration with random_search (always accepts), and verify the
    // kernel landed a near-zero leak score.
    let dir = TempDir::new().unwrap();
    // Manually invoke evaluate to confirm shape — kernel does the same.
    let mut target = OpticKTarget::default_smoke();
    let opt = target.true_optimum().clone();
    let score = target
        .evaluate(&opt, Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert!(score.structure_leak_pct < 1e-12);

    // And via run_experiment with greedy (uniform baseline, will perturb away
    // from optimum but the test is just to exercise the pipeline):
    let outcome = run_experiment(
        &mut target,
        Strategy::Greedy,
        5,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        7,
    )
    .unwrap();
    assert_eq!(outcome.iterations, 5);
    let events: Vec<LogEvent<OpticKConfig, OpticKScore>> =
        ix_autoresearch::log::read_log(&outcome.log_path).unwrap();
    // RunStart should carry the OpticKTarget type name.
    if let LogEvent::RunStart { target, .. } = &events[0] {
        assert!(target.contains("OpticKTarget"));
    } else {
        panic!("first event should be RunStart");
    }
}

#[test]
fn lex_order_reward_makes_low_leak_dominate() {
    // Build a synthetic high-retrieval-but-leaky score against a
    // zero-everything-but-low-leak score; the leaky one must lose
    // despite higher retrieval.
    let target = OpticKTarget::default_smoke();
    let leaky = OpticKScore {
        structure_leak_pct: 0.1,
        retrieval_match_pct: 1.0,
        inv_25_pass_rate: 1.0,
        inv_32_pass_rate: 1.0,
        inv_36_pass_rate: 1.0,
    };
    let cleaner = OpticKScore {
        structure_leak_pct: 0.0,
        retrieval_match_pct: 0.5,
        inv_25_pass_rate: 0.5,
        inv_32_pass_rate: 0.5,
        inv_36_pass_rate: 0.5,
    };
    assert!(target.score_to_reward(&cleaner) > target.score_to_reward(&leaky));
}

#[test]
fn cost_ledger_reports_clean_run() {
    let dir = TempDir::new().unwrap();
    let mut target = OpticKTarget::default_smoke();
    let outcome = run_experiment(
        &mut target,
        Strategy::Greedy,
        15,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        99,
    )
    .unwrap();
    assert_eq!(outcome.cost.eval_failure_count, 0);
    assert_eq!(
        outcome.cost.rejected_count + outcome.accepted as u32,
        outcome.iterations as u32
    );
}

#[test]
fn eval_inputs_hash_lands_in_run_start() {
    // The OpticKTarget should provide an eval_inputs_hash so replays
    // can detect "the true_optimum changed between runs".
    let dir = TempDir::new().unwrap();
    let mut target = OpticKTarget::default_smoke();
    let outcome = run_experiment(
        &mut target,
        Strategy::Greedy,
        2,
        TimeBudget::soft(Duration::from_secs(5)),
        dir.path(),
        3,
    )
    .unwrap();
    let events: Vec<LogEvent<OpticKConfig, OpticKScore>> =
        ix_autoresearch::log::read_log(&outcome.log_path).unwrap();
    if let LogEvent::RunStart {
        eval_inputs_hash, ..
    } = &events[0]
    {
        assert!(
            eval_inputs_hash.is_some(),
            "OpticKTarget should set eval_inputs_hash"
        );
    } else {
        panic!("first event should be RunStart");
    }
}
