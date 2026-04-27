//! Acceptance strategies for the autoresearch loop.
//!
//! v1 ships three strategies:
//!
//! - [`Strategy::Greedy`] — accept iff candidate strictly beats current.
//!   Karpathy's autoresearch uses this exclusively.
//! - [`Strategy::SimulatedAnnealing`] — Metropolis acceptance with
//!   geometric cooling. `initial_temperature: None` triggers calibration
//!   via [`calibrate_initial_temperature`] (Ben-Ameur 2004 heuristic).
//! - [`Strategy::RandomSearch`] — every candidate is "accepted" for
//!   logging purposes; best-so-far is tracked separately. Used as a
//!   reward-hacking baseline (Cerebras lessons).
//!
//! The [`Strategy::Custom`] variant is reserved (skipped from serde) to
//! prevent a one-way door at the MCP `input_schema` layer when v1.5+
//! adds UCB-over-directions or Bayesian opt; v1 does not wire it.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Decision returned by a policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Accept,
    Reject,
}

/// Trait for acceptance policies. v1 implementations are owned by
/// [`Strategy`]; the trait exists primarily so [`Strategy::Custom`]
/// can plug in without breaking the kernel API in v1.5+.
pub trait AcceptancePolicy: Send {
    fn decide(
        &mut self,
        prev_reward: f64,
        candidate_reward: f64,
        iteration: usize,
        rng: &mut ChaCha8Rng,
    ) -> Decision;

    /// Strategy state to log on each iteration (e.g. SA temperature).
    /// Returned as a `serde_json::Value` so the kernel can stash it in
    /// `Iteration.strategy_state` without leaking strategy-specific
    /// types into the log schema.
    fn log_state(&self) -> Option<serde_json::Value> {
        None
    }
}

/// Public configuration enum for acceptance behavior. Serialized into
/// the `RunStart.strategy` field for replay reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Strategy {
    /// Accept iff `candidate_reward > prev_reward`.
    Greedy,

    /// Metropolis acceptance with geometric cooling
    /// (`T_{n+1} = cooling_rate · T_n`). `initial_temperature: None`
    /// requests calibration via [`calibrate_initial_temperature`].
    SimulatedAnnealing {
        #[serde(default)]
        initial_temperature: Option<f64>,
        #[serde(default = "default_cooling_rate")]
        cooling_rate: f64,
    },

    /// Always accept. The kernel still tracks best-so-far separately.
    RandomSearch,

    /// Reserved for v1.5+ plug-ins. Never serialized; deserialization
    /// of `{"kind": "custom", ...}` will produce a parse error.
    #[serde(skip)]
    Custom,
}

fn default_cooling_rate() -> f64 {
    0.95
}

impl Strategy {
    /// Materialize a runnable policy.
    ///
    /// `experiment_baseline_reward` is used for SA T₀ calibration when
    /// `initial_temperature` is `None`. Pass `0.0` if calibration isn't
    /// applicable (other strategies ignore it).
    pub fn into_policy(self) -> Box<dyn AcceptancePolicy> {
        match self {
            Self::Greedy => Box::new(GreedyPolicy),
            Self::SimulatedAnnealing {
                initial_temperature,
                cooling_rate,
            } => {
                let t0 = initial_temperature.unwrap_or(1.0);
                Box::new(SimulatedAnnealingPolicy {
                    temperature: t0,
                    cooling_rate,
                })
            }
            Self::RandomSearch => Box::new(RandomSearchPolicy),
            Self::Custom => panic!(
                "Strategy::Custom is reserved for v1.5+; cannot construct a policy in v1"
            ),
        }
    }

    /// Helper: build SA strategy with a calibrated T₀, given mean uphill ΔE.
    /// `target_accept` is typically 0.8 (Ben-Ameur 2004).
    pub fn sa_with_calibrated_t0(mean_uphill_delta_e: f64, target_accept: f64) -> Self {
        let t0 = ben_ameur_t0(mean_uphill_delta_e, target_accept);
        Self::SimulatedAnnealing {
            initial_temperature: Some(t0),
            cooling_rate: default_cooling_rate(),
        }
    }
}

// ───────────────────────── Greedy ─────────────────────────

#[derive(Debug, Default)]
pub struct GreedyPolicy;

impl AcceptancePolicy for GreedyPolicy {
    fn decide(
        &mut self,
        prev_reward: f64,
        candidate_reward: f64,
        _iteration: usize,
        _rng: &mut ChaCha8Rng,
    ) -> Decision {
        if candidate_reward > prev_reward {
            Decision::Accept
        } else {
            Decision::Reject
        }
    }
}

// ───────────────────────── Random search ─────────────────────────

#[derive(Debug, Default)]
pub struct RandomSearchPolicy;

impl AcceptancePolicy for RandomSearchPolicy {
    fn decide(
        &mut self,
        _prev_reward: f64,
        _candidate_reward: f64,
        _iteration: usize,
        _rng: &mut ChaCha8Rng,
    ) -> Decision {
        Decision::Accept
    }
}

// ───────────────────────── Simulated annealing ─────────────────────────

#[derive(Debug, Clone)]
pub struct SimulatedAnnealingPolicy {
    pub temperature: f64,
    pub cooling_rate: f64,
}

impl AcceptancePolicy for SimulatedAnnealingPolicy {
    fn decide(
        &mut self,
        prev_reward: f64,
        candidate_reward: f64,
        _iteration: usize,
        rng: &mut ChaCha8Rng,
    ) -> Decision {
        // Higher reward is better. ΔE = candidate − prev; positive ΔE is
        // a good move (accept always); negative is a bad move (accept
        // with probability exp(ΔE / T)).
        let delta_e = candidate_reward - prev_reward;

        let decision = if delta_e > 0.0 {
            Decision::Accept
        } else if self.temperature <= 0.0 {
            // T = 0 ⇒ greedy.
            Decision::Reject
        } else {
            let p = (delta_e / self.temperature).exp();
            if rng.random::<f64>() < p {
                Decision::Accept
            } else {
                Decision::Reject
            }
        };

        // Cool after every decision.
        self.temperature *= self.cooling_rate;
        decision
    }

    fn log_state(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "temperature": self.temperature }))
    }
}

// ───────────────────────── T₀ calibration ─────────────────────────

/// Ben-Ameur 2004: pick T₀ so that bad moves are accepted with
/// probability ≈ `target_accept` initially. With mean uphill ΔE > 0
/// (where uphill means *worse* in our higher-is-better convention,
/// i.e. ΔE_uphill = -|delta|): `T₀ = -ΔE / ln(target_accept)`.
///
/// Robust to degenerate inputs:
/// - If `mean_uphill_delta_e == 0.0` (flat surface), returns a small
///   positive default (1e-6) so SA at iter 0 is ~indistinguishable
///   from Greedy. This avoids `T₀ = NaN` from `0/-Inf`.
/// - If `target_accept` is outside `(0, 1)`, returns `1.0` as a
///   safe default.
pub fn ben_ameur_t0(mean_uphill_delta_e: f64, target_accept: f64) -> f64 {
    if !(0.0 < target_accept && target_accept < 1.0) {
        return 1.0;
    }
    if mean_uphill_delta_e <= 0.0 {
        return 1e-6;
    }
    let t = -mean_uphill_delta_e / target_accept.ln();
    if t.is_finite() && t > 0.0 {
        t
    } else {
        1.0
    }
}

/// Sample `n_samples` rewards via the supplied `eval_one_random` closure
/// (typically: perturb the baseline, evaluate, return reward) and compute
/// the mean *uphill* ΔE — i.e. the mean magnitude of decreases below the
/// baseline reward (we're maximizing). The result is suitable for
/// [`ben_ameur_t0`].
///
/// Used by `Strategy::SimulatedAnnealing { initial_temperature: None }`
/// to calibrate T₀ before the main loop starts.
pub fn calibrate_initial_temperature<F>(
    baseline_reward: f64,
    n_samples: usize,
    target_accept: f64,
    mut eval_one_random: F,
) -> f64
where
    F: FnMut() -> f64,
{
    let mut downhill_sum = 0.0;
    let mut downhill_count = 0usize;
    for _ in 0..n_samples {
        let r = eval_one_random();
        let delta = r - baseline_reward;
        if delta < 0.0 {
            downhill_sum += -delta;
            downhill_count += 1;
        }
    }
    let mean_uphill = if downhill_count == 0 {
        0.0
    } else {
        downhill_sum / downhill_count as f64
    };
    ben_ameur_t0(mean_uphill, target_accept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn greedy_accepts_strict_improvement() {
        let mut p = GreedyPolicy;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        assert_eq!(p.decide(0.5, 0.6, 0, &mut rng), Decision::Accept);
        assert_eq!(p.decide(0.5, 0.5, 0, &mut rng), Decision::Reject);
        assert_eq!(p.decide(0.5, 0.4, 0, &mut rng), Decision::Reject);
    }

    #[test]
    fn random_search_always_accepts() {
        let mut p = RandomSearchPolicy;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        for prev in [0.0, 0.5, 1.0] {
            for cand in [0.0, 0.5, 1.0] {
                assert_eq!(p.decide(prev, cand, 0, &mut rng), Decision::Accept);
            }
        }
    }

    #[test]
    fn sa_accepts_uphill_at_high_temperature() {
        let mut p = SimulatedAnnealingPolicy {
            temperature: 1000.0,
            cooling_rate: 0.99,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let trials = 200;
        let downhill_accepts = (0..trials)
            .filter(|_| {
                p.temperature = 1000.0; // hold T fixed for this stat
                p.decide(0.5, 0.4, 0, &mut rng) == Decision::Accept
            })
            .count();
        let rate = downhill_accepts as f64 / trials as f64;
        assert!(rate > 0.5, "high-T SA should accept most downhill moves; rate={rate}");
    }

    #[test]
    fn sa_at_zero_temperature_is_greedy() {
        let mut p = SimulatedAnnealingPolicy {
            temperature: 0.0,
            cooling_rate: 0.95,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        // Improvement: accept.
        assert_eq!(p.decide(0.5, 0.6, 0, &mut rng), Decision::Accept);
        // Worse: reject.
        p.temperature = 0.0;
        assert_eq!(p.decide(0.5, 0.4, 0, &mut rng), Decision::Reject);
    }

    #[test]
    fn sa_log_state_carries_temperature() {
        let p = SimulatedAnnealingPolicy {
            temperature: 0.42,
            cooling_rate: 0.95,
        };
        let state = p.log_state().unwrap();
        assert_eq!(state["temperature"].as_f64(), Some(0.42));
    }

    #[test]
    fn ben_ameur_t0_handles_flat_surface() {
        // Mean uphill ΔE = 0 (flat) ⇒ tiny positive T₀, not NaN.
        let t = ben_ameur_t0(0.0, 0.8);
        assert!(t.is_finite() && t > 0.0);
        assert!(t < 1e-3);
    }

    #[test]
    fn ben_ameur_t0_typical_case() {
        // ΔE = 0.1, target_accept = 0.8: T₀ = -0.1 / ln(0.8) ≈ 0.448
        let t = ben_ameur_t0(0.1, 0.8);
        assert!((t - 0.4481).abs() < 1e-3, "t={t}");
    }

    #[test]
    fn calibrate_initial_temperature_with_const_eval_returns_tiny_t0() {
        // Always-baseline eval ⇒ no downhill samples ⇒ T₀ = 1e-6.
        let t = calibrate_initial_temperature(0.5, 10, 0.8, || 0.5);
        assert!(t > 0.0 && t < 1e-3);
    }

    #[test]
    fn strategy_serializes_with_kind_tag() {
        let s = Strategy::Greedy;
        let json = serde_json::to_string(&s).unwrap();
        assert!(json.contains("\"kind\":\"greedy\""));

        let s2 = Strategy::SimulatedAnnealing {
            initial_temperature: Some(0.5),
            cooling_rate: 0.9,
        };
        let json2 = serde_json::to_string(&s2).unwrap();
        assert!(json2.contains("\"kind\":\"simulated_annealing\""));
        assert!(json2.contains("\"initial_temperature\":0.5"));
        assert!(json2.contains("\"cooling_rate\":0.9"));
    }
}
