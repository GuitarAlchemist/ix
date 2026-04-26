//! "Chaos Detective" — A time series that looks random is deterministic chaos.
//!
//! Chains: ix_stats → ix_fft → ix_chaos_lyapunov → ix_topo
//! The "aha": positive Lyapunov exponent proves hidden order in apparent noise.

use serde_json::{json, Value};

use crate::demo::{DemoScenario, DemoStep, Difficulty, ScenarioMeta, StepInput};

pub struct ChaosDetective;

pub static META: ScenarioMeta = ScenarioMeta {
    id: "chaos-detective",
    title: "Chaos Detective",
    tagline: "That noise is not noise",
    description:
        "An IoT sensor signal looks like pure noise. Classical stats and FFT see nothing. \
                  But Lyapunov exponents reveal deterministic chaos, and persistent homology \
                  confirms the attractor has topological structure.",
    difficulty: Difficulty::Intermediate,
    tags: &["signal", "chaos", "topology", "math"],
    tools_used: &["ix_stats", "ix_fft", "ix_chaos_lyapunov", "ix_topo"],
};

impl DemoScenario for ChaosDetective {
    fn meta(&self) -> &ScenarioMeta {
        &META
    }

    fn steps(&self, seed: u64, verbosity: u8) -> Vec<DemoStep> {
        // Generate logistic map data: x_{n+1} = r * x_n * (1 - x_n), r=3.99 (chaotic regime)
        let data = generate_logistic_map(512, 3.99, seed);

        let narrative_1 = if verbosity >= 2 {
            "We compute descriptive statistics on 512 samples from what appears to be a noisy \
             sensor. If this were truly random, we'd expect a uniform-ish distribution. Let's \
             check the mean, standard deviation, and range."
                .into()
        } else {
            "Compute stats on 512 samples from a suspicious sensor signal.".into()
        };

        let narrative_2 = if verbosity >= 2 {
            "Next we run FFT to look for periodicity. If there's a hidden signal, FFT will \
             reveal dominant frequencies. If it's noise, energy will be spread across all bins."
                .into()
        } else {
            "FFT analysis — looking for hidden periodicity.".into()
        };

        let data_for_fft = data.clone();

        vec![
            // Step 1: ix_stats — looks unremarkable
            DemoStep {
                label: "Compute descriptive statistics".into(),
                tool: "ix_stats".into(),
                input: StepInput::Static(json!({ "data": data })),
                narrative: narrative_1,
                interpret: Some(|output: &Value| {
                    let mean = output.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let std = output
                        .get("std_dev")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    format!(
                        "Mean={mean:.3}, StdDev={std:.3}. Looks like uniform noise — \
                         nothing obviously structured here."
                    )
                }),
            },
            // Step 2: ix_fft — no dominant frequency
            DemoStep {
                label: "Frequency analysis via FFT".into(),
                tool: "ix_fft".into(),
                input: StepInput::Static(json!({ "signal": data_for_fft })),
                narrative: narrative_2,
                interpret: Some(|output: &Value| {
                    if let Some(magnitudes) = output.get("magnitudes").and_then(|v| v.as_array()) {
                        let max_mag = magnitudes
                            .iter()
                            .filter_map(|v| v.as_f64())
                            .fold(0.0_f64, f64::max);
                        format!(
                            "Peak magnitude={max_mag:.2}. Energy is spread across all frequencies — \
                             classical frequency analysis says: noise. But is it?"
                        )
                    } else {
                        "FFT complete — no dominant frequency peak detected.".into()
                    }
                }),
            },
            // Step 3: ix_chaos_lyapunov — THE AHA MOMENT
            DemoStep {
                label: "Lyapunov exponent analysis".into(),
                tool: "ix_chaos_lyapunov".into(),
                input: StepInput::Static(json!({
                    "map": "logistic",
                    "parameter": 3.99,
                    "iterations": 1000
                })),
                narrative: "Now the twist. We compute the Lyapunov exponent — a measure of \
                            sensitive dependence on initial conditions. Positive = chaos. \
                            Zero = periodic. Negative = stable."
                    .into(),
                interpret: Some(|output: &Value| {
                    let exponent = output
                        .get("lyapunov_exponent")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    if exponent > 0.0 {
                        format!(
                            "Lyapunov exponent = {exponent:.4} (POSITIVE). This signal is \
                             deterministic chaos, not random noise! Sensitive dependence on \
                             initial conditions means it's unpredictable but governed by \
                             a simple rule: x → 3.99·x·(1−x)."
                        )
                    } else {
                        format!("Lyapunov exponent = {exponent:.4}.")
                    }
                }),
            },
            // Step 4: ix_topo — topological confirmation
            DemoStep {
                label: "Topological fingerprint via persistent homology".into(),
                tool: "ix_topo".into(),
                input: StepInput::Static({
                    // Delay-embed: pairs [[x_i, x_{i+1}], ...] reveal the attractor shape
                    let pairs: Vec<Vec<f64>> = data_for_fft
                        .windows(2)
                        .take(50)
                        .map(|w| vec![w[0], w[1]])
                        .collect();
                    json!({
                        "operation": "persistence",
                        "points": pairs,
                        "max_dim": 1,
                        "max_radius": 1.0
                    })
                }),
                narrative: "Final confirmation: we delay-embed the series into 2D and run \
                            persistent homology. If the attractor has a loop, we'll see a \
                            long-lived 1-cycle in the barcode."
                    .into(),
                interpret: Some(|output: &Value| {
                    if let Some(bars) = output.get("bars").and_then(|v| v.as_array()) {
                        let h1_bars: Vec<_> = bars
                            .iter()
                            .filter(|b| b.get("dimension").and_then(|d| d.as_u64()) == Some(1))
                            .collect();
                        if h1_bars.is_empty() {
                            "No persistent 1-cycles found.".into()
                        } else {
                            format!(
                                "Found {} persistent 1-cycle(s) — the attractor has loop \
                                 structure. This is the topological fingerprint of chaos: \
                                 not random, not periodic, but a strange attractor with \
                                 detectable geometry.",
                                h1_bars.len()
                            )
                        }
                    } else {
                        "Persistent homology complete.".into()
                    }
                }),
            },
        ]
    }
}

/// Generate logistic map: x_{n+1} = r * x_n * (1 - x_n)
fn generate_logistic_map(n: usize, r: f64, seed: u64) -> Vec<f64> {
    let mut x = (seed % 1000) as f64 / 1000.0;
    // Clamp to (0, 1) to avoid degenerate fixed points
    if x <= 0.0 || x >= 1.0 {
        x = 0.42;
    }
    let mut data = Vec::with_capacity(n);
    // Burn in 100 iterations to reach the attractor
    for _ in 0..100 {
        x = r * x * (1.0 - x);
    }
    for _ in 0..n {
        x = r * x * (1.0 - x);
        data.push(x);
    }
    data
}
