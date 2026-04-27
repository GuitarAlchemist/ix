//! Target C — grammar-rule-weight tuning.
//!
//! Smoke-test target for the kernel: a small weighted CFG with N rules,
//! a deterministic held-out corpus drawn from a "true" rule-frequency
//! distribution, and an autoresearch loop that perturbs rule weights to
//! recover the underlying distribution.
//!
//! ## Eval metric
//!
//! Given current weights and a temperature, we form a softmax distribution
//! `P` over rule IDs and compare it against the empirical held-out
//! distribution `Q`. The primary score is the **expected match rate**:
//!
//! ```text
//! parse_success_rate = Σᵢ Q(i) · P(i)
//! ```
//!
//! At uniform weights, P is uniform (1/N each) so the score is `1/N`.
//! At the optimum (`weights = ln(Q)` under T=1), `P = Q` and the score
//! is `Σᵢ Q(i)²` — strictly higher when Q is non-uniform.
//!
//! Secondary score: replicator dynamics treat `P` as initial proportions
//! and `Q` as fitness; we run a short simulation and report
//! `ess_stability = 1.0 − L1(final_proportions, P)`. Stable populations
//! (where `P ≈ Q`) barely move during simulation, giving high
//! ess_stability; unstable populations move significantly, giving low
//! values.
//!
//! ## Why this is a useful smoke test
//!
//! - In-process, sub-second per iter — no shell-out, no flakiness.
//! - Real `ix_grammar` API surface (`WeightedRule`, `softmax`,
//!   `replicator::simulate`, `detect_ess`) — exercises the actual
//!   primitives, not a mock.
//! - Improvement is provable analytically (parse_success_rate goes from
//!   `1/N` to `Σ Q²`) so the integration test asserts a deterministic
//!   reward delta on a seeded run.

use std::collections::HashMap;
use std::time::Instant;

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use ix_grammar::replicator::{detect_ess, simulate, GrammarSpecies};
use ix_grammar::weighted::{softmax, WeightedRule};

use crate::error::{AutoresearchError, EvalCategory};
use crate::Experiment;

/// Tunable knobs for the grammar target.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GrammarConfig {
    /// One weight per rule (positions match `GrammarTarget::rule_ids`).
    /// Must be non-negative; perturbation reflects at 0.
    pub rule_weights: Vec<f64>,
    /// Softmax temperature, must be > 0; perturbation log-uniform in
    /// [0.1, 10.0].
    pub temperature: f64,
}

/// Multi-objective score. `score_to_reward` collapses to a scalar via
/// `parse_success_rate + 0.1 · ess_stability`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct GrammarScore {
    pub parse_success_rate: f64,
    pub ess_stability: f64,
}

/// Target adapter: holds the rule names, the held-out distribution, and
/// the perturbation hyperparameters. The kernel owns the loop; this
/// object owns the eval pipeline.
pub struct GrammarTarget {
    /// Rule identifiers, in the canonical order used by `Config::rule_weights`.
    rule_ids: Vec<String>,
    /// Empirical "held-out" distribution over the rule IDs (sums to 1).
    held_out_freq: Vec<f64>,
    /// Gaussian σ for weight perturbation. Default 0.1.
    weight_sigma: f64,
    /// Replicator simulation steps. Default 50.
    replicator_steps: usize,
    /// Replicator simulation step size. Default 0.1.
    replicator_dt: f64,
    /// Threshold below which a species is pruned. Default 1e-6.
    replicator_prune: f64,
}

impl GrammarTarget {
    /// Construct from explicit rule IDs + held-out frequencies.
    /// The two slices must be the same length and frequencies must
    /// be non-negative (they're normalized to sum to 1 on construction).
    pub fn new(rule_ids: Vec<String>, held_out_freq: Vec<f64>) -> Result<Self, String> {
        if rule_ids.is_empty() {
            return Err("rule_ids must be non-empty".to_string());
        }
        if rule_ids.len() != held_out_freq.len() {
            return Err(format!(
                "rule_ids ({}) and held_out_freq ({}) length mismatch",
                rule_ids.len(),
                held_out_freq.len()
            ));
        }
        let total: f64 = held_out_freq.iter().sum();
        if total <= 0.0 || !total.is_finite() {
            return Err(format!("held_out_freq sum invalid: {total}"));
        }
        if held_out_freq.iter().any(|f| *f < 0.0 || !f.is_finite()) {
            return Err("held_out_freq contains negative or non-finite value".to_string());
        }
        let normalized: Vec<f64> = held_out_freq.iter().map(|f| f / total).collect();
        Ok(Self {
            rule_ids,
            held_out_freq: normalized,
            weight_sigma: 0.1,
            replicator_steps: 50,
            replicator_dt: 0.1,
            replicator_prune: 1e-6,
        })
    }

    /// Default smoke-test instance: 6 rules with skewed held-out
    /// frequencies. Used by the integration test.
    pub fn default_smoke() -> Self {
        Self::new(
            (0..6).map(|i| format!("r{i}")).collect(),
            vec![0.05, 0.05, 0.10, 0.20, 0.25, 0.35],
        )
        .expect("default_smoke is well-formed")
    }

    pub fn n_rules(&self) -> usize {
        self.rule_ids.len()
    }

    pub fn held_out(&self) -> &[f64] {
        &self.held_out_freq
    }

    /// Theoretical maximum `parse_success_rate` = Σᵢ Q(i)².
    /// Useful for tests that assert "we got within X of the optimum".
    pub fn theoretical_optimum(&self) -> f64 {
        self.held_out_freq.iter().map(|q| q * q).sum()
    }

    /// Convenience: assemble a `WeightedRule` vector with the given weights.
    fn rules_with_weights(&self, weights: &[f64]) -> Vec<WeightedRule> {
        self.rule_ids
            .iter()
            .zip(weights.iter())
            .map(|(id, &w)| {
                let mut r = WeightedRule::new(id.clone(), 0, "ix-autoresearch:smoke");
                r.weight = w.max(0.0);
                r
            })
            .collect()
    }
}

impl Experiment for GrammarTarget {
    type Config = GrammarConfig;
    type Score = GrammarScore;

    fn baseline(&self) -> GrammarConfig {
        let n = self.n_rules();
        GrammarConfig {
            // Uniform weights ⇒ softmax gives uniform distribution at T=1.
            rule_weights: vec![1.0 / n as f64; n],
            temperature: 1.0,
        }
    }

    fn perturb(&mut self, current: &GrammarConfig, rng: &mut ChaCha8Rng) -> GrammarConfig {
        // Gaussian σ on each weight, reflective at 0.
        let new_weights: Vec<f64> = current
            .rule_weights
            .iter()
            .map(|&w| {
                // Box-Muller approximation via two uniform draws.
                let u1: f64 = rng.random_range(1e-12..1.0);
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let perturbed = w + z * self.weight_sigma;
                // Reflective bound at 0: |x|.
                perturbed.abs()
            })
            .collect();

        // Log-uniform on temperature within [0.1, 10.0].
        let log_lo = 0.1f64.ln();
        let log_hi = 10.0f64.ln();
        let log_t = log_lo + rng.random::<f64>() * (log_hi - log_lo);
        let new_temperature = log_t.exp();

        GrammarConfig {
            rule_weights: new_weights,
            temperature: new_temperature.max(1e-3),
        }
    }

    fn evaluate(
        &mut self,
        config: &GrammarConfig,
        _soft_deadline: Instant,
    ) -> Result<GrammarScore, AutoresearchError> {
        if config.rule_weights.len() != self.rule_ids.len() {
            return Err(AutoresearchError::EvalFailed(EvalCategory::InvalidConfig {
                reason: format!(
                    "rule_weights length {} != n_rules {}",
                    config.rule_weights.len(),
                    self.rule_ids.len()
                ),
            }));
        }
        if !config.temperature.is_finite() || config.temperature <= 0.0 {
            return Err(AutoresearchError::EvalFailed(EvalCategory::InvalidConfig {
                reason: format!("temperature must be > 0, got {}", config.temperature),
            }));
        }

        // 1. Build WeightedRule vector and softmax it at the requested T.
        let rules = self.rules_with_weights(&config.rule_weights);
        let probs = softmax(&rules, config.temperature);
        let prob_map: HashMap<String, f64> = probs.into_iter().collect();

        // 2. parse_success_rate = Σᵢ Q(i) · P(i).
        let parse_success_rate: f64 = self
            .rule_ids
            .iter()
            .zip(self.held_out_freq.iter())
            .map(|(id, &q)| q * prob_map.get(id).copied().unwrap_or(0.0))
            .sum();

        // 3. Replicator dynamics: initial proportions = current P, fitness = Q.
        //    After simulation, ess_stability = 1 - L1(final, P).
        let initial_species: Vec<GrammarSpecies> = self
            .rule_ids
            .iter()
            .zip(self.held_out_freq.iter())
            .map(|(id, &q)| {
                GrammarSpecies::new(id.clone(), prob_map.get(id).copied().unwrap_or(0.0), q)
            })
            .collect();
        let sim = simulate(
            &initial_species,
            self.replicator_steps,
            self.replicator_dt,
            self.replicator_prune,
        );
        // Use detect_ess as a sanity-check side effect (we don't consume its output here, but
        // it ratifies the replicator API surface promised by the plan).
        let _ = detect_ess(&sim.final_species, 0.5);
        let initial_by_id: HashMap<&str, f64> = initial_species
            .iter()
            .map(|s| (s.id.as_str(), s.proportion))
            .collect();
        let l1: f64 = sim
            .final_species
            .iter()
            .map(|s| {
                let init = initial_by_id.get(s.id.as_str()).copied().unwrap_or(0.0);
                (s.proportion - init).abs()
            })
            .sum();
        let ess_stability = (1.0 - l1).clamp(0.0, 1.0);

        Ok(GrammarScore {
            parse_success_rate,
            ess_stability,
        })
    }

    fn score_to_reward(&self, score: &GrammarScore) -> f64 {
        // Weighted sum: parse_success_rate dominates (0..~0.5),
        // ess_stability is a tie-break (0..1) scaled to 0.1.
        score.parse_success_rate + 0.1 * score.ess_stability
    }

    fn cache_salt(&self) -> Option<String> {
        // Eval is deterministic given config.
        Some("ix-autoresearch:target_grammar:v1".to_string())
    }

    fn eval_inputs_hash(&self) -> Option<String> {
        // Content-hash the held-out distribution + rule IDs so a
        // reload-with-different-corpus invalidates downstream caches and
        // makes the change visible in the run log.
        let payload = serde_json::json!({
            "rule_ids": self.rule_ids,
            "held_out_freq": self.held_out_freq,
        });
        serde_json::to_vec(&payload)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn default_smoke_constructs_with_six_rules() {
        let t = GrammarTarget::default_smoke();
        assert_eq!(t.n_rules(), 6);
        let total: f64 = t.held_out().iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn baseline_is_uniform_at_temperature_one() {
        let t = GrammarTarget::default_smoke();
        let b = t.baseline();
        assert_eq!(b.rule_weights.len(), 6);
        for w in &b.rule_weights {
            assert!((w - 1.0 / 6.0).abs() < 1e-12);
        }
        assert_eq!(b.temperature, 1.0);
    }

    #[test]
    fn theoretical_optimum_matches_sum_of_squares() {
        let t = GrammarTarget::default_smoke();
        // Q² = 0.05² + 0.05² + 0.10² + 0.20² + 0.25² + 0.35²
        //    = 0.0025 + 0.0025 + 0.01 + 0.04 + 0.0625 + 0.1225 = 0.2400
        let opt = t.theoretical_optimum();
        assert!((opt - 0.24).abs() < 1e-12, "got {opt}");
    }

    #[test]
    fn evaluate_at_baseline_returns_one_over_n() {
        // Uniform weights ⇒ uniform softmax ⇒ Σ Q(i)·(1/N) = 1/N.
        let mut t = GrammarTarget::default_smoke();
        let baseline = t.baseline();
        let score = t
            .evaluate(&baseline, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        assert!(
            (score.parse_success_rate - 1.0 / 6.0).abs() < 1e-12,
            "expected 1/6, got {}",
            score.parse_success_rate
        );
    }

    #[test]
    fn evaluate_rejects_mismatched_weight_length() {
        let mut t = GrammarTarget::default_smoke();
        let bad = GrammarConfig {
            rule_weights: vec![0.5, 0.5], // length 2, not 6
            temperature: 1.0,
        };
        let err = t
            .evaluate(&bad, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(
            err,
            AutoresearchError::EvalFailed(EvalCategory::InvalidConfig { .. })
        ));
    }

    #[test]
    fn evaluate_rejects_non_positive_temperature() {
        let mut t = GrammarTarget::default_smoke();
        let bad = GrammarConfig {
            rule_weights: vec![1.0; 6],
            temperature: 0.0,
        };
        let err = t
            .evaluate(&bad, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(
            err,
            AutoresearchError::EvalFailed(EvalCategory::InvalidConfig { .. })
        ));
    }

    #[test]
    fn perturb_keeps_weights_non_negative() {
        let mut t = GrammarTarget::default_smoke();
        let baseline = t.baseline();
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..50 {
            let candidate = t.perturb(&baseline, &mut rng);
            assert!(candidate.rule_weights.iter().all(|w| *w >= 0.0));
            assert!(candidate.temperature > 0.0);
            assert!(candidate.temperature.is_finite());
        }
    }

    #[test]
    fn cache_salt_is_target_specific() {
        let t = GrammarTarget::default_smoke();
        let salt = t.cache_salt().unwrap();
        assert!(salt.contains("target_grammar"));
    }

    #[test]
    fn eval_inputs_hash_changes_when_held_out_changes() {
        let t1 = GrammarTarget::new(
            (0..3).map(|i| format!("r{i}")).collect(),
            vec![0.5, 0.3, 0.2],
        )
        .unwrap();
        let t2 = GrammarTarget::new(
            (0..3).map(|i| format!("r{i}")).collect(),
            vec![0.4, 0.3, 0.3],
        )
        .unwrap();
        assert_ne!(t1.eval_inputs_hash(), t2.eval_inputs_hash());
    }
}
