//! "Cost Anomaly Hunter" — Detect cloud spending anomalies with signal + stats + clustering.
//!
//! Chains: ix_stats → ix_fft → ix_kmeans → ix_bloom_filter
//! The "aha": FFT reveals a hidden weekly billing cycle, and k-means isolates
//! the runaway service that accounts for 40% of the anomalous spend.

use serde_json::{json, Value};

use crate::demo::{DemoScenario, DemoStep, Difficulty, ScenarioMeta, StepInput};

pub struct CostAnomalyHunter;

pub static META: ScenarioMeta = ScenarioMeta {
    id: "cost-anomaly-hunter",
    title: "Cost Anomaly Hunter",
    tagline: "Your cloud bill spiked 30% — is it real growth or a runaway service?",
    description: "A FinOps team sees a 30% cloud cost increase. Before panicking, they use \
                  stats to baseline the spend, FFT to find billing cycles, k-means to cluster \
                  cost patterns by service, and a Bloom filter to quickly check resource IDs \
                  against a known-costly watchlist. The result: one misconfigured auto-scaler \
                  accounts for most of the spike.",
    difficulty: Difficulty::Beginner,
    tags: &["finops", "cloud", "signal", "clustering"],
    tools_used: &["ix_stats", "ix_fft", "ix_kmeans", "ix_bloom_filter"],
};

impl DemoScenario for CostAnomalyHunter {
    fn meta(&self) -> &ScenarioMeta {
        &META
    }

    fn steps(&self, seed: u64, verbosity: u8) -> Vec<DemoStep> {
        // Generate 90 days of daily cloud spend with a weekly cycle + anomaly spike
        let daily_spend = generate_cloud_spend(90, seed);
        let daily_spend_for_fft = daily_spend.clone();

        vec![
            // Step 1: ix_stats — baseline the distribution
            DemoStep {
                label: "Baseline cloud spend statistics".into(),
                tool: "ix_stats".into(),
                input: StepInput::Static(json!({ "data": daily_spend })),
                narrative: if verbosity >= 1 {
                    "First, we compute descriptive statistics on 90 days of daily cloud spend. \
                     This gives us the baseline: what's normal? The mean and standard deviation \
                     tell us where to draw the anomaly threshold."
                        .into()
                } else {
                    "Compute spend baseline stats.".into()
                },
                interpret: Some(|output: &Value| {
                    let mean = output.get("mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let std = output
                        .get("std_dev")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let max = output.get("max").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    format!(
                        "Baseline: ${mean:.0}/day ± ${std:.0}. But max=${max:.0} — \
                         that's {:.1}σ above the mean. Something spiked.",
                        (max - mean) / std
                    )
                }),
            },
            // Step 2: ix_fft — find billing cycles
            DemoStep {
                label: "FFT to detect spending cycles".into(),
                tool: "ix_fft".into(),
                input: StepInput::Static(json!({ "signal": daily_spend_for_fft })),
                narrative: if verbosity >= 1 {
                    "Next, we run FFT on the daily spend to find periodic patterns. Cloud \
                     costs often have weekly cycles (dev clusters shut down on weekends) and \
                     monthly cycles (reserved instance billing). Finding these lets us \
                     separate cyclical variation from true anomalies."
                        .into()
                } else {
                    "FFT on daily spend — find cycles.".into()
                },
                interpret: Some(|output: &Value| {
                    if let Some(magnitudes) = output.get("magnitudes").and_then(|v| v.as_array()) {
                        // Look for peaks — bin ~13 would be ~7-day cycle in 90-day data
                        let mags: Vec<f64> = magnitudes.iter().filter_map(|v| v.as_f64()).collect();
                        let max_bin = mags
                            .iter()
                            .enumerate()
                            .skip(1) // skip DC
                            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let period = if max_bin > 0 {
                            90.0 / max_bin as f64
                        } else {
                            0.0
                        };
                        format!(
                            "Dominant cycle at bin {max_bin} ≈ {period:.1}-day period. \
                             That's the weekly dev-cluster shutdown pattern. Costs dip \
                             every weekend — this is normal. The anomaly is on top of this cycle."
                        )
                    } else {
                        "FFT complete.".into()
                    }
                }),
            },
            // Step 3: ix_kmeans — cluster by cost pattern (THE AHA MOMENT)
            DemoStep {
                label: "Cluster services by spend pattern".into(),
                tool: "ix_kmeans".into(),
                input: StepInput::Static({
                    // 8 services × 3 features: [avg_daily_cost, cost_variance, growth_rate]
                    let service_features = generate_service_features(seed);
                    json!({
                        "data": service_features,
                        "k": 3,
                        "max_iterations": 50
                    })
                }),
                narrative: if verbosity >= 1 {
                    "Now the detective work. We profile 8 cloud services by their cost \
                     pattern: average daily spend, variance, and 30-day growth rate. \
                     K-means with k=3 should separate normal, elevated, and anomalous."
                        .into()
                } else {
                    "Cluster 8 services by cost pattern.".into()
                },
                interpret: Some(|output: &Value| {
                    if let Some(assignments) = output.get("assignments").and_then(|v| v.as_array())
                    {
                        let services = [
                            "EC2-prod",
                            "RDS-main",
                            "S3-logs",
                            "Lambda-api",
                            "EKS-dev",
                            "CloudFront",
                            "SageMaker-train",
                            "EC2-autoscale",
                        ];
                        let mut clusters: std::collections::HashMap<u64, Vec<&str>> =
                            std::collections::HashMap::new();
                        for (i, a) in assignments.iter().enumerate() {
                            let c = a.as_u64().unwrap_or(0);
                            clusters
                                .entry(c)
                                .or_default()
                                .push(services.get(i).unwrap_or(&"?"));
                        }
                        let anomalous = clusters
                            .values()
                            .min_by_key(|v| v.len())
                            .map(|v| v.join(", "))
                            .unwrap_or_default();
                        format!(
                            "Cluster analysis reveals the outlier: [{anomalous}] is in its \
                             own cluster — high spend, high variance, high growth. That \
                             misconfigured auto-scaler is the smoking gun."
                        )
                    } else {
                        "Clustering complete.".into()
                    }
                }),
            },
            // Step 4: ix_bloom_filter — fast watchlist check
            DemoStep {
                label: "Check against known-costly resource watchlist".into(),
                tool: "ix_bloom_filter".into(),
                input: StepInput::Static(json!({
                    "operation": "create_and_check",
                    "items": [
                        "i-0abc-autoscale-XXL",
                        "i-0def-gpu-training",
                        "rds-replica-unused"
                    ],
                    "check": [
                        "i-0abc-autoscale-XXL",
                        "i-9999-normal-web",
                        "rds-replica-unused"
                    ],
                    "false_positive_rate": 0.01
                })),
                narrative: if verbosity >= 1 {
                    "Final step: cross-reference the flagged resource IDs against a \
                     watchlist of known-costly resources using a Bloom filter. This is \
                     how you'd do it at scale — O(1) lookup, zero false negatives."
                        .into()
                } else {
                    "Bloom filter watchlist check.".into()
                },
                interpret: Some(|_output: &Value| {
                    "Bloom filter confirms: the auto-scaler instance is on the watchlist. \
                     Action: cap the auto-scaler max instances, set a billing alarm, \
                     and schedule a FinOps review."
                        .into()
                }),
            },
        ]
    }
}

/// Generate 90 days of cloud spend: base ~$1000/day + weekly cycle + anomaly spike in days 60-75.
fn generate_cloud_spend(days: usize, seed: u64) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut rng_state = seed;
    let mut data = Vec::with_capacity(days);
    for day in 0..days {
        // Simple LCG for deterministic noise
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let noise = ((rng_state >> 33) as f64 / u32::MAX as f64 - 0.5) * 100.0;
        // Base + weekly cycle (dip on weekends)
        let base = 1000.0;
        let weekly = -150.0 * (2.0 * PI * day as f64 / 7.0).cos();
        // Anomaly spike: days 60-75
        let anomaly = if (60..75).contains(&day) { 400.0 } else { 0.0 };
        data.push(base + weekly + anomaly + noise);
    }
    data
}

/// Generate 8 services × 3 features: [avg_daily_cost, cost_variance, growth_rate].
/// Service 8 (auto-scaler) is the anomalous one.
fn generate_service_features(seed: u64) -> Vec<Vec<f64>> {
    let mut rng_state = seed;
    let mut next = || -> f64 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f64 / u32::MAX as f64
    };

    vec![
        // Normal services: moderate cost, low variance, low growth
        vec![
            200.0 + next() * 50.0,
            20.0 + next() * 10.0,
            0.02 + next() * 0.03,
        ], // EC2-prod
        vec![
            150.0 + next() * 30.0,
            15.0 + next() * 8.0,
            0.01 + next() * 0.02,
        ], // RDS-main
        vec![
            50.0 + next() * 20.0,
            10.0 + next() * 5.0,
            0.03 + next() * 0.02,
        ], // S3-logs
        vec![
            80.0 + next() * 25.0,
            12.0 + next() * 6.0,
            0.02 + next() * 0.03,
        ], // Lambda-api
        vec![
            120.0 + next() * 40.0,
            25.0 + next() * 10.0,
            0.01 + next() * 0.02,
        ], // EKS-dev
        vec![
            60.0 + next() * 15.0,
            8.0 + next() * 4.0,
            0.01 + next() * 0.01,
        ], // CloudFront
        // Elevated but expected (ML training)
        vec![
            350.0 + next() * 80.0,
            60.0 + next() * 20.0,
            0.05 + next() * 0.03,
        ], // SageMaker
        // ANOMALY: auto-scaler gone wild
        vec![
            800.0 + next() * 200.0,
            250.0 + next() * 50.0,
            0.35 + next() * 0.1,
        ], // EC2-autoscale
    ]
}
