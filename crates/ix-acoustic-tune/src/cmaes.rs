//! CMA-ES — Covariance Matrix Adaptation Evolution Strategy.
//!
//! A derivative-free optimizer for non-convex, low-to-moderate-dimensional
//! black-box objectives; the de-facto standard for synthesizer parameter
//! matching (~5 continuous params against a target recording). Implements the
//! basic (μ_w, λ) variant with positive recombination weights (Hansen, "The CMA
//! Evolution Strategy: A Tutorial"). Box constraints are handled by clamping
//! (repairing) sampled candidates to the feasible region.
//!
//! Promotion candidate: this is a general optimizer and belongs in `ix-optimize`
//! once its API stabilizes. It reuses `ix_math::eigen::symmetric_eigen` for the
//! covariance eigendecomposition rather than adding a linear-algebra dependency.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::AskTell;

/// CMA-ES optimizer state, driven via the [`AskTell`] interface.
pub struct CmaEs {
    dim: usize,
    mean: Array1<f64>,
    sigma: f64,

    // Strategy parameters (constant after construction).
    lambda: usize,
    mu: usize,
    weights: Array1<f64>,
    mu_eff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    chi_n: f64,

    // Dynamic state.
    pc: Array1<f64>,
    ps: Array1<f64>,
    cov: Array2<f64>,
    eig_b: Array2<f64>, // eigenvectors of cov (columns)
    eig_d: Array1<f64>, // sqrt of eigenvalues (clamped ≥ tiny)

    lower: Option<Array1<f64>>,
    upper: Option<Array1<f64>>,
    rng: StdRng,
    generation: usize,
    best: Option<(Array1<f64>, f64)>,
}

impl CmaEs {
    /// Create a CMA-ES instance with an initial `mean`, initial step size
    /// `sigma` (roughly 1/4 of the search range per dimension is a good start),
    /// and a `seed` for reproducibility.
    pub fn new(mean: Array1<f64>, sigma: f64, seed: u64) -> Self {
        let n = mean.len();
        assert!(n >= 1, "CMA-ES needs at least one dimension");
        assert!(sigma > 0.0 && sigma.is_finite(), "sigma must be > 0");

        let nf = n as f64;
        let lambda = 4 + (3.0 * nf.ln()).floor() as usize;
        let mu = lambda / 2;

        // Recombination weights w_i ∝ ln(mu + 0.5) - ln(i), normalized to sum 1.
        let mut w: Vec<f64> = (1..=mu)
            .map(|i| (mu as f64 + 0.5).ln() - (i as f64).ln())
            .collect();
        let w_sum: f64 = w.iter().sum();
        for wi in &mut w {
            *wi /= w_sum;
        }
        let weights = Array1::from_vec(w);
        let mu_eff = 1.0 / weights.iter().map(|x| x * x).sum::<f64>();

        let cc = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
        let cs = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mu_eff);
        let cmu = ((1.0 - c1)
            .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).powi(2) + mu_eff)))
        .max(0.0);
        let damps = 1.0 + 2.0 * (((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        Self {
            dim: n,
            mean,
            sigma,
            lambda,
            mu,
            weights,
            mu_eff,
            cc,
            cs,
            c1,
            cmu,
            damps,
            chi_n,
            pc: Array1::zeros(n),
            ps: Array1::zeros(n),
            cov: Array2::eye(n),
            eig_b: Array2::eye(n),
            eig_d: Array1::ones(n),
            lower: None,
            upper: None,
            rng: StdRng::seed_from_u64(seed),
            generation: 0,
            best: None,
        }
    }

    /// Box-constrain the search to `[lower, upper]` per dimension. Sampled
    /// candidates are clamped (repaired) into the box before evaluation.
    pub fn with_bounds(mut self, lower: Array1<f64>, upper: Array1<f64>) -> Self {
        assert_eq!(lower.len(), self.dim, "lower bound dim mismatch");
        assert_eq!(upper.len(), self.dim, "upper bound dim mismatch");
        self.lower = Some(lower);
        self.upper = Some(upper);
        self
    }

    /// Population size (λ) — the number of candidates [`ask`](AskTell::ask) returns.
    pub fn population_size(&self) -> usize {
        self.lambda
    }

    /// Current step size σ.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Current distribution mean (the running best-estimate center).
    pub fn mean(&self) -> &Array1<f64> {
        &self.mean
    }

    fn clamp(&self, mut x: Array1<f64>) -> Array1<f64> {
        if let (Some(lo), Some(hi)) = (&self.lower, &self.upper) {
            for i in 0..self.dim {
                x[i] = x[i].clamp(lo[i], hi[i]);
            }
        }
        x
    }

    /// Run the optimizer in-process against a closure objective (convenience for
    /// testing and for objectives that *can* be evaluated synchronously). For
    /// the external-render case, drive [`ask`](AskTell::ask) / [`tell`](AskTell::tell)
    /// directly. Returns the best `(params, loss)` found.
    // @ai:invariant CMA-ES drives the distribution to the optimum of a smooth non-convex objective — verified by recovering the Rosenbrock valley minimum (1,1), which requires correct covariance adaptation to the curved valley [T:test conf:0.85 src:cmaes::tests::recovers_rosenbrock_minimum]
    pub fn minimize<F>(mut self, f: F, max_generations: usize) -> (Array1<f64>, f64)
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        for _ in 0..max_generations {
            let candidates = self.ask();
            let scored: Vec<(Array1<f64>, f64)> = candidates
                .into_iter()
                .map(|p| {
                    let loss = f(&p);
                    (p, loss)
                })
                .collect();
            self.tell(&scored);
        }
        self.recommend()
            .expect("minimize ran at least one generation")
    }

    /// `C^{-1/2} = B · diag(1/d) · Bᵀ`, from the current eigendecomposition.
    fn inv_sqrt_cov(&self) -> Array2<f64> {
        let inv_d = self.eig_d.mapv(|di| 1.0 / di);
        let bd = &self.eig_b * &inv_d; // scales column j by inv_d[j] (broadcast over rows)
        bd.dot(&self.eig_b.t())
    }

    /// Recompute B and D from the (symmetric) covariance matrix.
    fn update_eigensystem(&mut self) {
        // Enforce exact symmetry to keep the solver honest.
        let sym = 0.5 * (&self.cov + &self.cov.t());
        match ix_math::eigen::symmetric_eigen(&sym) {
            Ok((vals, vecs)) => {
                self.eig_b = vecs;
                // Clamp eigenvalues to a tiny positive floor before sqrt (C is PSD
                // in theory; guard numerical drift).
                self.eig_d = vals.mapv(|v| v.max(1e-20).sqrt());
            }
            Err(_) => {
                // Degenerate covariance: reset to isotropic (C = I) rather than
                // panic. `eig_d` holds sqrt of the COVARIANCE eigenvalues, so it is
                // ones (NOT sqrt(sigma) — sigma is applied separately in `ask`, and
                // folding it in here would double-count the step size).
                self.cov = Array2::eye(self.dim);
                self.eig_b = Array2::eye(self.dim);
                self.eig_d = Array1::ones(self.dim);
            }
        }
    }
}

fn outer(y: &Array1<f64>) -> Array2<f64> {
    let n = y.len();
    Array2::from_shape_fn((n, n), |(i, j)| y[i] * y[j])
}

impl AskTell for CmaEs {
    fn ask(&mut self) -> Vec<Array1<f64>> {
        (0..self.lambda)
            .map(|_| {
                // z ~ N(0, I); y = B·D·z; x = mean + sigma·y; repaired into the box.
                let z = Array1::from_shape_fn(self.dim, |_| -> f64 {
                    StandardNormal.sample(&mut self.rng)
                });
                let dz = &self.eig_d * &z;
                let y = self.eig_b.dot(&dz);
                let x = &self.mean + &(self.sigma * &y);
                self.clamp(x)
            })
            .collect()
    }

    // @ai:invariant after tell, the distribution mean equals the weighted recombination Σ w_i·x_(i) of the μ best candidates — independent of the step-size σ update (uses sigma_old, not the post-update sigma) [T:test conf:0.9 src:cmaes::tests::mean_update_is_the_weighted_centroid_independent_of_sigma]
    fn tell(&mut self, evaluated: &[(Array1<f64>, f64)]) {
        if evaluated.is_empty() {
            return;
        }
        // Track global best.
        for (p, loss) in evaluated {
            if self.best.as_ref().map(|(_, b)| *loss < *b).unwrap_or(true) {
                self.best = Some((p.clone(), *loss));
            }
        }

        // Sort ascending by loss; select the μ best.
        let mut idx: Vec<usize> = (0..evaluated.len()).collect();
        idx.sort_by(|&a, &b| {
            evaluated[a]
                .1
                .partial_cmp(&evaluated[b].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mu = self.mu.min(idx.len());

        // Renormalize the selection weights to sum to 1 over the μ candidates
        // actually returned (a host driving ask/tell may report fewer than λ);
        // otherwise a short batch would systematically under-scale the mean and
        // covariance updates. With a full λ batch this is a no-op.
        let w_sum: f64 = (0..mu).map(|k| self.weights[k]).sum::<f64>().max(1e-300);
        let w_used: Vec<f64> = (0..mu).map(|k| self.weights[k] / w_sum).collect();

        // Recompute the steps y_i = (x_i - mean)/sigma from the returned params
        // (robust to any host-side repair), and the weighted mean step y_w.
        let mut y_w: Array1<f64> = Array1::zeros(self.dim);
        let mut ys: Vec<Array1<f64>> = Vec::with_capacity(mu);
        for k in 0..mu {
            let x = &evaluated[idx[k]].0;
            let y = (x - &self.mean) / self.sigma;
            y_w = &y_w + &(w_used[k] * &y);
            ys.push(y);
        }

        // inv_sqrt_C of the distribution that GENERATED these samples (pre-update).
        let inv_sqrt_c = self.inv_sqrt_cov();

        // Step-size evolution path.
        let cs_factor = (self.cs * (2.0 - self.cs) * self.mu_eff).sqrt();
        self.ps = &((1.0 - self.cs) * &self.ps) + &(cs_factor * &inv_sqrt_c.dot(&y_w));
        let ps_norm = self.ps.dot(&self.ps).sqrt();

        // Heaviside stall guard for the rank-one update.
        let hsig = if ps_norm
            / (1.0 - (1.0 - self.cs).powi(2 * (self.generation as i32 + 1))).sqrt()
            < (1.4 + 2.0 / (self.dim as f64 + 1.0)) * self.chi_n
        {
            1.0
        } else {
            0.0
        };

        // Covariance evolution path.
        let cc_factor = (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt();
        self.pc = &((1.0 - self.cc) * &self.pc) + &(hsig * cc_factor * &y_w);

        // Rank-μ term: Σ w_i y_i y_iᵀ.
        let mut rank_mu: Array2<f64> = Array2::zeros((self.dim, self.dim));
        for (k, y) in ys.iter().enumerate() {
            rank_mu = &rank_mu + &(w_used[k] * &outer(y));
        }
        // Rank-one term (with the stall correction).
        let rank_one = &outer(&self.pc) + &((1.0 - hsig) * self.cc * (2.0 - self.cc) * &self.cov);

        self.cov = &((1.0 - self.c1 - self.cmu) * &self.cov)
            + &(self.c1 * &rank_one)
            + &(self.cmu * &rank_mu);

        // Step-size update. Capture the OLD sigma FIRST: the mean step below must
        // use the sigma that generated the samples, so mean_new lands exactly on
        // the weighted centroid Σ w_i x_i (= mean_old + sigma_old·y_w), independent
        // of this step-size change. Using the post-update sigma would scale every
        // mean step by exp((cs/damps)(‖ps‖/χ−1)) ≠ 1 — a wrong distribution update.
        let sigma_old = self.sigma;
        self.sigma *= ((self.cs / self.damps) * (ps_norm / self.chi_n - 1.0)).exp();
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            self.sigma = 1e-10;
        }

        // Mean update (with sigma_old), then refresh the eigensystem for next ask.
        self.mean = &self.mean + &(sigma_old * &y_w);
        self.generation += 1;
        self.update_eigensystem();
    }

    fn recommend(&self) -> Option<(Array1<f64>, f64)> {
        self.best.clone()
    }

    fn generation(&self) -> usize {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CMA-ES must drive a 5-D shifted sphere to its known minimum at x = 3.
    #[test]
    fn recovers_shifted_sphere_minimum() {
        let opt = CmaEs::new(Array1::zeros(5), 2.0, 42);
        let (best, loss) = opt.minimize(|x| x.iter().map(|&v| (v - 3.0).powi(2)).sum(), 200);
        assert!(loss < 1e-6, "should reach the minimum, loss={loss}");
        for (i, &v) in best.iter().enumerate() {
            assert!((v - 3.0).abs() < 1e-2, "x[{i}]={v} should be ≈3");
        }
    }

    // The classic 2-D Rosenbrock valley — minimum at (1, 1).
    #[test]
    fn recovers_rosenbrock_minimum() {
        let opt = CmaEs::new(Array1::from_vec(vec![-1.0, 1.0]), 0.5, 7);
        let (best, loss) = opt.minimize(
            |x| {
                let a = 1.0 - x[0];
                let b = x[1] - x[0] * x[0];
                a * a + 100.0 * b * b
            },
            400,
        );
        assert!(loss < 1e-4, "rosenbrock loss={loss}");
        assert!((best[0] - 1.0).abs() < 0.05 && (best[1] - 1.0).abs() < 0.05);
    }

    // Review P0 regression: after one tell the mean must equal the weighted
    // recombination of the μ best points, Σ w_i x_(i) — INDEPENDENT of the
    // step-size update. The old bug scaled this by sigma_new/sigma_old ≠ 1.
    #[test]
    fn mean_update_is_the_weighted_centroid_independent_of_sigma() {
        // 1-D ⇒ λ = 4, μ = 2, weights ∝ [ln2.5−ln1, ln2.5−ln2].
        let mut opt = CmaEs::new(Array1::from_vec(vec![0.0]), 1.0, 99);
        let cands = opt.ask();
        assert_eq!(cands.len(), 4);
        // Fitness = the candidate value, so the two SMALLEST are selected.
        let scored: Vec<(Array1<f64>, f64)> = cands.iter().map(|c| (c.clone(), c[0])).collect();

        let mut vals: Vec<f64> = cands.iter().map(|c| c[0]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let w0 = (2.5_f64).ln(); // ln(2.5) − ln(1)
        let w1 = (2.5_f64).ln() - (2.0_f64).ln();
        let wsum = w0 + w1;
        let expected = (w0 / wsum) * vals[0] + (w1 / wsum) * vals[1];

        opt.tell(&scored);
        assert!(
            (opt.mean()[0] - expected).abs() < 1e-9,
            "mean {} should equal the weighted centroid {expected} (σ-independent)",
            opt.mean()[0]
        );
    }

    // Box constraints: every asked candidate stays inside [lower, upper].
    #[test]
    fn respects_bounds() {
        let lower = Array1::from_vec(vec![0.0, 0.0, 0.0]);
        let upper = Array1::from_vec(vec![1.0, 0.5, 0.9]);
        let mut opt = CmaEs::new(Array1::from_vec(vec![0.5, 0.25, 0.45]), 0.4, 3)
            .with_bounds(lower.clone(), upper.clone());
        for _ in 0..10 {
            for cand in opt.ask() {
                for i in 0..3 {
                    assert!(
                        cand[i] >= lower[i] - 1e-12 && cand[i] <= upper[i] + 1e-12,
                        "candidate out of bounds at dim {i}: {}",
                        cand[i]
                    );
                }
            }
            // feed a dummy loss to advance the state
            let scored: Vec<_> = opt.ask().into_iter().map(|p| (p, 0.0)).collect();
            opt.tell(&scored);
        }
    }
}
