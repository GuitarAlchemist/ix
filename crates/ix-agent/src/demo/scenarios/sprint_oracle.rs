//! "Sprint Oracle" — Predict sprint velocity and model ticket state transitions.
//!
//! Chains: ix_stats → ix_linear_regression → ix_markov → ix_bandit
//! The "aha": the Markov chain reveals that tickets in "Review" have a 25% chance
//! of bouncing back to "In Progress" — a hidden bottleneck the PM didn't see.

use serde_json::{json, Value};

use crate::demo::{DemoScenario, DemoStep, Difficulty, ScenarioMeta, StepInput};

pub struct SprintOracle;

pub static META: ScenarioMeta = ScenarioMeta {
    id: "sprint-oracle",
    title: "Sprint Oracle",
    tagline: "Your team says 'it depends' — the data says otherwise",
    description: "A PM managing a 6-person team wants to predict next sprint's velocity and \
                  find process bottlenecks. Stats baseline the history, regression finds the \
                  trend, a Markov chain models ticket state transitions (revealing a hidden \
                  review bottleneck), and a bandit optimizes story point estimation strategy.",
    difficulty: Difficulty::Beginner,
    tags: &["product-management", "agile", "prediction", "process"],
    tools_used: &["ix_stats", "ix_linear_regression", "ix_markov", "ix_bandit"],
};

impl DemoScenario for SprintOracle {
    fn meta(&self) -> &ScenarioMeta {
        &META
    }

    fn steps(&self, seed: u64, verbosity: u8) -> Vec<DemoStep> {
        // 12 sprints of velocity data (story points completed)
        let velocities = generate_velocity_history(12, seed);
        let velocity_for_regression = velocities.clone();

        vec![
            // Step 1: ix_stats — baseline velocity
            DemoStep {
                label: "Baseline sprint velocity".into(),
                tool: "ix_stats".into(),
                input: StepInput::Static(json!({ "data": velocities })),
                narrative: if verbosity >= 1 {
                    "We start with 12 sprints of velocity data (story points completed \
                     per 2-week sprint). Basic stats tell us what 'normal' looks like \
                     for this team."
                        .into()
                } else {
                    "Stats on 12 sprints of velocity.".into()
                },
                interpret: Some(|output: &Value| {
                    let mean = output.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let std = output
                        .get("std_dev")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let min = output.get("min").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let max = output.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    format!(
                        "Average velocity: {mean:.0} points/sprint (±{std:.0}). \
                         Range: {min:.0}–{max:.0}. The team is variable but trending. \
                         Next question: which direction?"
                    )
                }),
            },
            // Step 2: ix_linear_regression — velocity trend
            DemoStep {
                label: "Fit velocity trend line".into(),
                tool: "ix_linear_regression".into(),
                input: StepInput::Static({
                    // X = sprint number (1-12), y = velocity.
                    // `ix_linear_regression` takes `x` + `y`; prediction
                    // for future sprints is derived from the fitted
                    // slope/bias at interpret time, not from a separate
                    // `x_test` payload.
                    let x: Vec<Vec<f64>> = (1..=12).map(|i| vec![i as f64]).collect();
                    json!({
                        "x": x,
                        "y": velocity_for_regression,
                    })
                }),
                narrative: if verbosity >= 1 {
                    "We fit a linear regression on sprint number vs velocity to find \
                     the trend. Is the team ramping up, plateauing, or slowing down? \
                     And what's the predicted velocity for the next 3 sprints?"
                        .into()
                } else {
                    "Regression: velocity trend + prediction.".into()
                },
                interpret: Some(|output: &Value| {
                    let slope = output
                        .get("weights")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.first())
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let bias = output.get("bias").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let next_sprint = slope * 13.0 + bias;
                    let direction = if slope > 0.5 {
                        "trending up — the team is ramping"
                    } else if slope < -0.5 {
                        "trending down — possible burnout or scope creep"
                    } else {
                        "stable — consistent delivery"
                    };
                    format!(
                        "Slope: {slope:+.1} points/sprint ({direction}). \
                         Predicted next sprint: {next_sprint:.0} points. \
                         Plan capacity accordingly."
                    )
                }),
            },
            // Step 3: ix_markov — ticket state transitions (THE AHA MOMENT)
            DemoStep {
                label: "Model ticket state transitions".into(),
                tool: "ix_markov".into(),
                input: StepInput::Static({
                    // Transition matrix: Open → InProgress → Review → Done
                    // Hidden insight: Review has 25% bounce-back to InProgress
                    json!({
                        "transition_matrix": [
                            [0.2, 0.7, 0.0, 0.1],  // Open: 70% → InProgress, 10% → Done (auto-close)
                            [0.0, 0.3, 0.6, 0.1],   // InProgress: 60% → Review, 10% → Done
                            [0.0, 0.25, 0.35, 0.4],  // Review: 25% BOUNCE BACK, 40% → Done
                            [0.0, 0.0, 0.0, 1.0]    // Done: absorbing state
                        ],
                        "initial_state": [1.0, 0.0, 0.0, 0.0],
                        "steps": 10
                    })
                }),
                narrative: if verbosity >= 1 {
                    "Now the process insight. We model ticket lifecycle as a Markov chain \
                     with 4 states: Open → InProgress → Review → Done. The transition \
                     probabilities come from 6 months of JIRA data. Where do tickets get stuck?"
                        .into()
                } else {
                    "Markov chain: ticket state transitions.".into()
                },
                interpret: Some(|_output: &Value| {
                    "The Review state has a 25% probability of bouncing back to InProgress. \
                     That means 1 in 4 tickets goes through the review cycle twice. This is \
                     the hidden bottleneck: the team thinks review is fast, but the bounce-back \
                     rate adds ~2 days per affected ticket. Fix the review process, and velocity \
                     jumps 15-20%."
                        .into()
                }),
            },
            // Step 4: ix_bandit — optimize estimation strategy
            DemoStep {
                label: "Optimize story point estimation".into(),
                tool: "ix_bandit".into(),
                input: StepInput::Static(json!({
                    "algorithm": "thompson",
                    "true_means": [0.6, 0.75, 0.55],
                    "rounds": 100
                })),
                narrative: if verbosity >= 1 {
                    "Finally, we use a Thompson sampling bandit to A/B test three estimation \
                     strategies: (1) planning poker, (2) t-shirt sizing with historical \
                     calibration, (3) gut feel. Each 'arm' represents the accuracy rate of \
                     that method (how often the estimate is within 20% of actual)."
                        .into()
                } else {
                    "Thompson bandit: best estimation strategy.".into()
                },
                interpret: Some(|output: &Value| {
                    let best = output.get("best_arm").and_then(|v| v.as_u64()).unwrap_or(0);
                    let strategies = ["planning poker", "t-shirt sizing + calibration", "gut feel"];
                    let name = strategies.get(best as usize).unwrap_or(&"unknown");
                    let regret = output
                        .get("cumulative_regret")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    format!(
                        "Best strategy: '{name}' (arm {best}), converged after ~30 rounds. \
                         Cumulative regret: {regret:.1}. T-shirt sizing with historical \
                         calibration wins because it combines human judgment with data — \
                         the same principle behind this entire demo."
                    )
                }),
            },
        ]
    }
}

/// Generate 12 sprints of velocity with slight upward trend + noise.
fn generate_velocity_history(sprints: usize, seed: u64) -> Vec<f64> {
    let mut rng_state = seed;
    let mut data = Vec::with_capacity(sprints);
    for i in 0..sprints {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 10.0;
        // Base 30 points + slight upward trend + noise
        let velocity = 30.0 + 1.5 * i as f64 + noise;
        data.push(velocity.round());
    }
    data
}
