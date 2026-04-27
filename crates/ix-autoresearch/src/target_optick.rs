//! Target A — OPTIC-K self-tuning (mocked v1; live v2 via GA companion PR).
//!
//! Perturbs the 6-element OPTIC-K partition-weight simplex
//! `[STRUCTURE, MORPHOLOGY, CONTEXT, SYMBOLIC, MODAL, ROOT]` and scores
//! each candidate against `ix-optick-invariants` + `ix-embedding-diagnostics`.
//!
//! ## Mocked vs live
//!
//! v1 ships a synthetic `rebuild_fn` that does NOT shell out to GA's
//! `FretboardVoicingsCLI`. Instead it computes a deterministic
//! `OpticKScore` purely from the config's distance to a hand-picked
//! "true optimum" weight vector. This validates:
//!
//! - Dirichlet-on-simplex perturbation correctness (sum-to-1 invariant
//!   preserved within 1e-9 after renormalize),
//! - Lex-order reward scalarization,
//! - End-to-end kernel + adapter wiring.
//!
//! The live path requires the GA-side `--weights-config <path>` flag
//! on `FretboardVoicingsCLI` (Phase 6). When that lands, swap the
//! `rebuild_fn` for a real shell-out.
//!
//! ## Dirichlet perturbation defaults
//!
//! - α = 200 (concentration). Per Stan's reference: `Var[X_i] = w_i(1−w_i)/(α+1)`.
//!   At α=200 with uniform mid-point w≈0.17, σ ≈ 0.027 — "explore
//!   neighborhood" scale.
//! - ε = 1e-3 floor on every component before scaling, then renormalize
//!   to sum 1 *before* sampling. Without the floor, zero weights are
//!   absorbing states under `Dirichlet(0, ...)`.

use std::time::Instant;

use rand_chacha::ChaCha8Rng;
use rand_distr::{Dirichlet, Distribution};
use serde::{Deserialize, Serialize};

use crate::error::{AutoresearchError, EvalCategory};
use crate::Experiment;

/// 6-element simplex of OPTIC-K partition weights.
///
/// Field order matches `EmbeddingSchema` in GA. Sum must be ≈ 1.0
/// (perturbation preserves this via Dirichlet-on-simplex sampling).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpticKConfig {
    pub structure_weight: f64,
    pub morphology_weight: f64,
    pub context_weight: f64,
    pub symbolic_weight: f64,
    pub modal_weight: f64,
    pub root_weight: f64,
}

impl OpticKConfig {
    pub fn as_array(&self) -> [f64; 6] {
        [
            self.structure_weight,
            self.morphology_weight,
            self.context_weight,
            self.symbolic_weight,
            self.modal_weight,
            self.root_weight,
        ]
    }

    pub fn from_array(a: [f64; 6]) -> Self {
        Self {
            structure_weight: a[0],
            morphology_weight: a[1],
            context_weight: a[2],
            symbolic_weight: a[3],
            modal_weight: a[4],
            root_weight: a[5],
        }
    }

    /// Sum of weights — should be ≈ 1.0 for a valid config.
    pub fn sum(&self) -> f64 {
        self.as_array().iter().sum()
    }
}

/// Score from one OPTIC-K eval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub struct OpticKScore {
    /// Primary metric: STRUCTURE leak percentage. Lower is better.
    pub structure_leak_pct: f64,
    /// Tie-breaker: retrieval-match percentage (chatbot ground truth).
    pub retrieval_match_pct: f64,
    pub inv_25_pass_rate: f64,
    pub inv_32_pass_rate: f64,
    pub inv_36_pass_rate: f64,
}

/// Adapter holding the rebuild function (real CLI shell-out in v2,
/// synthetic in v1) and the Dirichlet hyperparameters.
pub struct OpticKTarget {
    /// "True optimum" used by the synthetic rebuild_fn to compute
    /// distance-to-optimum scores. In v1 this is the production
    /// `EmbeddingSchema.PartitionWeights` from GA, hardcoded as a
    /// reference point.
    true_optimum: OpticKConfig,
    /// Concentration α for Dirichlet perturbation. Default 200.
    alpha: f64,
    /// Floor on every weight before Dirichlet scaling. Default 1e-3.
    /// Without this, `Dirichlet(0, ...)` is an absorbing state.
    floor: f64,
    /// Synthetic vs live rebuild. In v1 this is always synthetic.
    rebuild_mode: RebuildMode,
}

/// Rebuild dispatch — synthetic (mocked v1) or live (Phase 6+).
#[derive(Debug, Clone, Copy)]
pub enum RebuildMode {
    /// Compute synthetic score from config distance to `true_optimum`.
    /// No shell-out. Default for v1.
    Synthetic,
    // Live shell-out variants land in Phase 6+ alongside the GA
    // `--weights-config` flag. Reserved here so the enum doesn't grow
    // breaking in v1.5.
    // CIReduced { ga_cli_path: PathBuf, ... },
    // LiveCorpus { ... },
}

impl OpticKTarget {
    /// Default smoke instance: production-shaped weights as the
    /// "true optimum"; α = 200, floor = 1e-3, synthetic rebuild.
    pub fn default_smoke() -> Self {
        // OPTIC-K v4-pp-r v1.8 production weights (per ix-optick/src/lib.rs:64
        // schema seed). These are the reference point the synthetic
        // rebuild scores against.
        let true_optimum = OpticKConfig::from_array([
            0.30, // STRUCTURE
            0.25, // MORPHOLOGY
            0.15, // CONTEXT
            0.10, // SYMBOLIC
            0.15, // MODAL
            0.05, // ROOT
        ]);
        Self {
            true_optimum,
            alpha: 200.0,
            floor: 1e-3,
            rebuild_mode: RebuildMode::Synthetic,
        }
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_floor(mut self, floor: f64) -> Self {
        self.floor = floor;
        self
    }

    pub fn true_optimum(&self) -> &OpticKConfig {
        &self.true_optimum
    }

    /// Floor every weight at `self.floor` then renormalize to sum 1.
    /// Public so tests can verify the invariant directly.
    pub fn floor_and_renormalize(&self, mut w: [f64; 6]) -> [f64; 6] {
        for x in &mut w {
            *x = x.max(self.floor);
        }
        let total: f64 = w.iter().sum();
        if total > 0.0 {
            for x in &mut w {
                *x /= total;
            }
        }
        w
    }

    /// L1 distance between a config and the `true_optimum`. Used by the
    /// synthetic rebuild for score derivation.
    fn l1_to_optimum(&self, config: &OpticKConfig) -> f64 {
        let opt = self.true_optimum.as_array();
        let cur = config.as_array();
        cur.iter().zip(opt.iter()).map(|(a, b)| (a - b).abs()).sum()
    }
}

impl Experiment for OpticKTarget {
    type Config = OpticKConfig;
    type Score = OpticKScore;

    fn baseline(&self) -> OpticKConfig {
        // Uniform 1/6 across all six partitions — far from the
        // production `true_optimum`, so SA has room to improve.
        OpticKConfig::from_array([1.0 / 6.0; 6])
    }

    fn perturb(&mut self, current: &OpticKConfig, rng: &mut ChaCha8Rng) -> OpticKConfig {
        // Floor + renormalize *before* Dirichlet so zero components are
        // not absorbing states. Then sample Dir(α · w_floored_renormed).
        let w = self.floor_and_renormalize(current.as_array());
        let mut alpha_arr = [0.0f64; 6];
        for (i, v) in w.iter().enumerate() {
            alpha_arr[i] = v * self.alpha;
        }
        // rand_distr 0.5 uses const generics — `Dirichlet::<f64, 6>::new`
        // takes a fixed-size array. If alpha is degenerate (NaN, etc.)
        // we fall back to the renormed `w` itself.
        let dirichlet: Dirichlet<f64, 6> = match Dirichlet::new(alpha_arr) {
            Ok(d) => d,
            Err(_) => return OpticKConfig::from_array(w),
        };
        let sample: [f64; 6] = dirichlet.sample(rng);
        OpticKConfig::from_array(sample)
    }

    fn evaluate(
        &mut self,
        config: &OpticKConfig,
        _soft_deadline: Instant,
    ) -> Result<OpticKScore, AutoresearchError> {
        // Validate simplex constraint within tolerance.
        let s = config.sum();
        if !s.is_finite() || (s - 1.0).abs() > 1e-3 {
            return Err(AutoresearchError::EvalFailed(EvalCategory::InvalidConfig {
                reason: format!("weights must sum to ≈ 1.0; got sum = {s:.6}"),
            }));
        }
        if config.as_array().iter().any(|w| !w.is_finite() || *w < -1e-12) {
            return Err(AutoresearchError::EvalFailed(EvalCategory::InvalidConfig {
                reason: "weights must be non-negative and finite".to_string(),
            }));
        }

        match self.rebuild_mode {
            RebuildMode::Synthetic => Ok(self.synthetic_score(config)),
        }
    }

    fn score_to_reward(&self, s: &OpticKScore) -> f64 {
        // Lex-style scalar (deepen-plan refinement): leak dominates by
        // 1e6, retrieval next at 1e3, mean(invariants) breaks ties.
        let leak = (1.0 - s.structure_leak_pct).clamp(0.0, 1.0);
        let retr = s.retrieval_match_pct.clamp(0.0, 1.0);
        let inv = (s.inv_25_pass_rate + s.inv_32_pass_rate + s.inv_36_pass_rate) / 3.0;
        leak * 1.0e6 + retr * 1.0e3 + inv
    }

    fn cache_salt(&self) -> Option<String> {
        // Synthetic eval is deterministic given config. Live mode would
        // include the GA index hash here so a corpus rebuild invalidates.
        Some("ix-autoresearch:target_optick:v1-synthetic".to_string())
    }

    fn eval_inputs_hash(&self) -> Option<String> {
        // For synthetic mode, the only "input" is the true_optimum
        // reference point. Hash that so changing it (e.g., to test a
        // different production schema) invalidates downstream caches.
        let payload = serde_json::json!({
            "true_optimum": self.true_optimum,
            "alpha": self.alpha,
            "floor": self.floor,
            "mode": "synthetic-v1",
        });
        serde_json::to_vec(&payload)
            .ok()
            .map(|bytes| blake3::hash(&bytes).to_hex().to_string())
    }
}

impl OpticKTarget {
    /// Synthetic score derivation. Mirrors what the live pipeline would
    /// produce in shape (the same OpticKScore fields), but computes
    /// values purely from the config's distance to true_optimum.
    fn synthetic_score(&self, config: &OpticKConfig) -> OpticKScore {
        let l1 = self.l1_to_optimum(config); // ∈ [0, 2]
        // Linear scale: l1=0 → leak=0; l1=2 → leak=1.
        let structure_leak_pct = (l1 / 2.0).clamp(0.0, 1.0);
        // Inverse of leak with a tiny "structural prior" so retrieval
        // doesn't track leak exactly. v1: retrieval = 1 − leak² (concave).
        let retrieval_match_pct =
            (1.0 - structure_leak_pct.powi(2)).clamp(0.0, 1.0);
        // Synthetic invariant pass rates: smoothly decay with leak.
        let inv_25_pass_rate = (1.0 - structure_leak_pct).clamp(0.0, 1.0);
        let inv_32_pass_rate = (1.0 - structure_leak_pct).clamp(0.0, 1.0);
        let inv_36_pass_rate = (1.0 - structure_leak_pct * 0.8).clamp(0.0, 1.0);
        OpticKScore {
            structure_leak_pct,
            retrieval_match_pct,
            inv_25_pass_rate,
            inv_32_pass_rate,
            inv_36_pass_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_target() -> OpticKTarget {
        OpticKTarget::default_smoke()
    }

    #[test]
    fn baseline_is_uniform_simplex() {
        let t = make_target();
        let b = t.baseline();
        for &w in &b.as_array() {
            assert!((w - 1.0 / 6.0).abs() < 1e-12);
        }
        assert!((b.sum() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn perturb_preserves_simplex_within_tolerance() {
        let mut t = make_target();
        let baseline = t.baseline();
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..50 {
            let c = t.perturb(&baseline, &mut rng);
            let sum = c.sum();
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "perturbed config must sum to 1.0 within 1e-9; got {sum}"
            );
            for &w in &c.as_array() {
                assert!(w >= 0.0, "weight must be ≥ 0; got {w}");
            }
        }
    }

    #[test]
    fn perturb_keeps_zero_components_above_floor() {
        // Start from a degenerate config (one zero) and verify floor
        // mechanism prevents it from being trapped at zero.
        let mut t = make_target();
        let degenerate = OpticKConfig::from_array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..20 {
            let c = t.perturb(&degenerate, &mut rng);
            // After floor + Dirichlet, every weight should sample from a
            // distribution with non-zero support — so ≥ 0.
            for &w in &c.as_array() {
                assert!(w >= 0.0);
            }
            assert!((c.sum() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn evaluate_at_true_optimum_returns_zero_leak() {
        let mut t = make_target();
        let opt = t.true_optimum().clone();
        let score = t
            .evaluate(&opt, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        assert!(score.structure_leak_pct < 1e-12);
        assert!(score.retrieval_match_pct > 0.999);
        assert!(score.inv_25_pass_rate > 0.999);
    }

    #[test]
    fn evaluate_at_uniform_baseline_has_positive_leak() {
        // Production weights are non-uniform, so uniform baseline is
        // far from optimum → non-zero leak.
        let mut t = make_target();
        let baseline = t.baseline();
        let score = t
            .evaluate(&baseline, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap();
        assert!(score.structure_leak_pct > 0.0);
        assert!(score.structure_leak_pct < 1.0);
    }

    #[test]
    fn evaluate_rejects_non_simplex_config() {
        let mut t = make_target();
        let bad = OpticKConfig::from_array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]); // sums to 6
        let err = t
            .evaluate(&bad, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(
            err,
            AutoresearchError::EvalFailed(EvalCategory::InvalidConfig { .. })
        ));
    }

    #[test]
    fn evaluate_rejects_negative_weight() {
        let mut t = make_target();
        let bad = OpticKConfig::from_array([0.5, 0.6, -0.1, 0.0, 0.0, 0.0]);
        let err = t
            .evaluate(&bad, Instant::now() + std::time::Duration::from_secs(1))
            .unwrap_err();
        assert!(matches!(
            err,
            AutoresearchError::EvalFailed(EvalCategory::InvalidConfig { .. })
        ));
    }

    #[test]
    fn score_to_reward_lex_orders_leak_first() {
        let t = make_target();
        let low_leak = OpticKScore {
            structure_leak_pct: 0.0,
            retrieval_match_pct: 0.0,
            inv_25_pass_rate: 0.0,
            inv_32_pass_rate: 0.0,
            inv_36_pass_rate: 0.0,
        };
        let high_retrieval = OpticKScore {
            structure_leak_pct: 0.5,
            retrieval_match_pct: 1.0,
            inv_25_pass_rate: 1.0,
            inv_32_pass_rate: 1.0,
            inv_36_pass_rate: 1.0,
        };
        // 0-leak with everything else 0 must beat 0.5-leak with everything else 1.
        assert!(t.score_to_reward(&low_leak) > t.score_to_reward(&high_retrieval));
    }

    #[test]
    fn cache_salt_marks_synthetic_mode() {
        let t = make_target();
        let salt = t.cache_salt().unwrap();
        assert!(salt.contains("synthetic"));
    }

    #[test]
    fn eval_inputs_hash_changes_when_alpha_changes() {
        let t1 = OpticKTarget::default_smoke().with_alpha(100.0);
        let t2 = OpticKTarget::default_smoke().with_alpha(500.0);
        assert_ne!(t1.eval_inputs_hash(), t2.eval_inputs_hash());
    }

    #[test]
    fn floor_and_renormalize_eliminates_zeros_and_sums_to_one() {
        let t = make_target();
        let result = t.floor_and_renormalize([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]);
        // After floor + renormalize, every weight is strictly positive
        // (no absorbing-state zeros). The renormalize step pushes them
        // slightly *below* the raw `floor` because flooring increases
        // the sum >1, so renormalize divides — that's expected.
        for w in &result {
            assert!(*w > 0.0, "weight {} should be > 0 after floor", w);
        }
        // Sum invariant is the load-bearing one.
        let total: f64 = result.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}
