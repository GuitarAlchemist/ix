//! Executable fixtures for the agent-loop metrics proposed in
//! `docs/research/tars-v1-state-space-ix-metrics.md` (Refs #193).
//!
//! Each test is the "local test fixture" for one metric. The point is not to ship the
//! metrics as public API — it is to prove, with a live binding, that a metric composed
//! only from already-shipped IX primitives actually discriminates the four loop regimes
//! (converging / oscillating / stalling / diverging) on synthetic traces.
//!
//! Every metric below is a *test-local* composition. Nothing here grows a public surface.

use ix_chaos::lyapunov::{classify_dynamics, DynamicsType};
use ix_signal::correlation::autocorrelation;
use ix_signal::timeseries::{ddm_detect, difference, rolling_std, DdmConfig, DriftState};

// ---------------------------------------------------------------------------
// M1 — Loop progress drift, over `ix_signal::timeseries::ddm_detect`
// ---------------------------------------------------------------------------

/// Index of the first step at which DDM reports `Drift`, if any.
///
/// Input is one boolean per loop step: `true` = the step failed to make progress.
/// DDM was designed for exactly this shape (degradation of a learner's error rate),
/// so the metric is a direct reuse with no new mathematics.
fn first_drift_step(step_failed: &[bool], config: DdmConfig) -> Option<usize> {
    ddm_detect(step_failed, config)
        .iter()
        .position(|s| s.state == DriftState::Drift)
}

/// Deterministic pseudo-random booleans at a target failure rate.
///
/// A tiny LCG keeps the fixture reproducible without pulling in `rand`, so the
/// asserted step indices are stable across platforms.
fn failure_sequence(n: usize, rate: f64, seed: u64) -> Vec<bool> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u = ((state >> 33) as f64) / ((1u64 << 31) as f64);
            u < rate
        })
        .collect()
}

#[test]
fn m1_drift_fires_when_progress_degrades() {
    // 150 healthy steps (10% failure) then 150 degraded steps (70% failure).
    let mut trace = failure_sequence(150, 0.10, 42);
    trace.extend(failure_sequence(150, 0.70, 43));

    let drift = first_drift_step(&trace, DdmConfig::default());
    assert!(
        drift.is_some(),
        "DDM must report Drift on a loop whose failure rate degrades 0.10 -> 0.70"
    );
    // Drift must be attributed to the degraded half, not the healthy prefix.
    assert!(
        drift.unwrap() >= 150,
        "drift reported at step {:?}, before the regime change at 150",
        drift
    );
}

#[test]
fn m1_drift_silent_on_a_stable_loop() {
    // Same failure rate throughout: a loop that is consistently imperfect but not degrading.
    let trace = failure_sequence(300, 0.10, 42);
    assert_eq!(
        first_drift_step(&trace, DdmConfig::default()),
        None,
        "DDM must stay silent when the failure rate is stationary (no false positive)"
    );
}

// ---------------------------------------------------------------------------
// M2 — Loop oscillation index, over `ix_signal::correlation::autocorrelation`
// ---------------------------------------------------------------------------

/// Dominant repeat period of a loop's per-step action signature, and its strength.
///
/// Returns `(lag, normalized_autocorrelation)` for the strongest positive lag in
/// `1..=max_lag`. A high value at lag `k` means the loop repeats itself every `k`
/// steps — the "edits file A, test fails, reverts to B, repeats" failure.
///
/// Two preprocessing steps are mandatory, and both were found by a failing test
/// rather than by inspection (see the research note's caveat for M2):
///
/// 1. **First-difference** the signature. Raw autocorrelation at short lags measures
///    *smoothness*, not periodicity: a monotone ramp scores 0.95 at lag 1 and would be
///    misreported as a tight 1-cycle. Differencing removes the trend.
/// 2. **Mean-center** afterwards. `autocorrelation` normalizes by the zero-lag value
///    but does not center, so any DC component otherwise dominates every lag.
fn oscillation_index(signature: &[f64], max_lag: usize) -> (usize, f64) {
    let diffed = difference(signature, 1);
    let n = diffed.len();
    if n < 2 {
        return (0, 0.0);
    }
    let mean = diffed.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = diffed.iter().map(|v| v - mean).collect();

    let ac = autocorrelation(&centered);
    // `autocorrelation` returns 2n-1 values; zero lag sits at index n-1.
    let mut best = (0usize, f64::NEG_INFINITY);
    for lag in 1..=max_lag.min(n - 1) {
        let v = ac[n - 1 + lag];
        if v > best.1 {
            best = (lag, v);
        }
    }
    best
}

#[test]
fn m2_detects_a_three_step_cycle() {
    // The loop cycles through three distinct actions forever.
    let signature: Vec<f64> = (0..60).map(|i| [1.0, 2.0, 3.0][i % 3]).collect();

    let (lag, strength) = oscillation_index(&signature, 12);
    assert_eq!(lag, 3, "a period-3 loop must peak at lag 3, got lag {lag}");
    assert!(
        strength > 0.5,
        "period-3 peak strength {strength} is too weak to act on"
    );
}

#[test]
fn m2_quiet_on_a_non_repeating_loop() {
    // Monotone progress: each step visits a new state, no cycle.
    let signature: Vec<f64> = (0..60).map(|i| i as f64).collect();

    let (_, strength) = oscillation_index(&signature, 12);
    assert!(
        strength < 0.9,
        "a monotone (non-repeating) loop must not present a near-perfect cycle peak, got {strength}"
    );
}

// ---------------------------------------------------------------------------
// M3 — Loop contraction rate, classified by `ix_chaos::lyapunov::classify_dynamics`
// ---------------------------------------------------------------------------

/// Mean log-ratio contraction rate of a residual sequence.
///
/// `lambda = (1/(n-1)) * sum ln(|r_{k+1}| / |r_k|)`
///
/// Negative = the loop contracts toward its target; zero = marginal; positive = it
/// diverges. This is **not** a Lyapunov exponent of a dynamical system — see the
/// caveat in the research note. It is a finite-time contraction rate on a 1-D
/// residual, which is why it is fed to `classify_dynamics` rather than computed by
/// `mle_1d`.
fn contraction_rate(residual: &[f64]) -> f64 {
    const FLOOR: f64 = 1e-12;
    let ratios: Vec<f64> = residual
        .windows(2)
        .map(|w| (w[1].abs().max(FLOOR) / w[0].abs().max(FLOOR)).ln())
        .collect();
    ratios.iter().sum::<f64>() / ratios.len() as f64
}

#[test]
fn m3_converging_loop_classifies_as_fixed_point() {
    // Geometric decay: the textbook converging loop.
    let residual: Vec<f64> = (0..40).map(|k| 0.5_f64.powi(k)).collect();

    let lambda = contraction_rate(&residual);
    assert!(
        (lambda - 0.5_f64.ln()).abs() < 1e-9,
        "contraction rate {lambda} should equal ln(0.5) = {}",
        0.5_f64.ln()
    );
    assert_eq!(classify_dynamics(lambda, 0.05), DynamicsType::FixedPoint);
}

#[test]
fn m3_diverging_loop_classifies_as_chaotic_or_divergent() {
    // Geometric growth: the loop is making things worse each step.
    let residual: Vec<f64> = (0..40).map(|k| 1.5_f64.powi(k)).collect();

    let lambda = contraction_rate(&residual);
    assert!(lambda > 0.0, "growing residual must give lambda > 0, got {lambda}");
    assert!(
        matches!(
            classify_dynamics(lambda, 0.05),
            DynamicsType::Chaotic | DynamicsType::Divergent
        ),
        "growing residual must not be classified as converging"
    );
}

#[test]
fn m3_flat_loop_is_marginal_not_converging() {
    // A stalled loop: the residual never moves. lambda == 0 exactly.
    let residual = vec![0.5_f64; 40];

    let lambda = contraction_rate(&residual);
    assert!(lambda.abs() < 1e-12, "flat residual must give lambda 0, got {lambda}");
    assert_eq!(
        classify_dynamics(lambda, 0.05),
        DynamicsType::Periodic,
        "a stalled loop is marginal, and M3 alone cannot tell it from a limit cycle"
    );
}

// ---------------------------------------------------------------------------
// M4 — Loop stall index, over `ix_signal::timeseries::rolling_std`
// ---------------------------------------------------------------------------

/// True when the loop has gone quiet *without* reaching its target.
///
/// This exists because M3 returns lambda ~ 0 for a stalled loop *and* for a healthy
/// limit cycle *and* for a converged one. M4 is the disambiguator: flat **and** far
/// from target = stalled; flat **and** at target = done.
fn is_stalled(residual: &[f64], window: usize, flat_eps: f64, target: f64) -> bool {
    let sd = rolling_std(residual, window);
    let Some(&last_sd) = sd.last() else {
        return false;
    };
    let Some(&last_r) = residual.last() else {
        return false;
    };
    last_sd.is_finite() && last_sd < flat_eps && last_r.abs() > target
}

#[test]
fn m4_flat_and_far_from_target_is_a_stall() {
    let residual = vec![0.5_f64; 40];
    assert!(
        is_stalled(&residual, 10, 1e-6, 0.01),
        "a residual pinned at 0.5 with target 0.01 is a stall"
    );
}

#[test]
fn m4_flat_and_at_target_is_convergence_not_a_stall() {
    let residual = vec![0.001_f64; 40];
    assert!(
        !is_stalled(&residual, 10, 1e-6, 0.01),
        "a residual at 0.001 with target 0.01 has converged, and must not be flagged as a stall"
    );
}

// ---------------------------------------------------------------------------
// Regime separation — the four metrics together must not collapse
// ---------------------------------------------------------------------------

#[test]
fn metrics_separate_the_four_loop_regimes() {
    let converging: Vec<f64> = (0..40).map(|k| 0.5_f64.powi(k)).collect();
    let diverging: Vec<f64> = (0..40).map(|k| 1.5_f64.powi(k)).collect();
    let stalling = vec![0.5_f64; 40];
    let oscillating: Vec<f64> = (0..40).map(|i| [1.0, 2.0, 3.0][i % 3]).collect();

    // M3 separates converging from diverging.
    assert!(contraction_rate(&converging) < -0.1);
    assert!(contraction_rate(&diverging) > 0.1);

    // M3 alone cannot separate stalling from oscillating: both are marginal.
    assert!(contraction_rate(&stalling).abs() < 1e-12);

    // M4 catches the stall; M2 catches the oscillation. This is why >1 metric is needed.
    assert!(is_stalled(&stalling, 10, 1e-6, 0.01));
    assert_eq!(oscillation_index(&oscillating, 12).0, 3);
}
