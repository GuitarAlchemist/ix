//! # ix-manifold
//!
//! Pure-Rust manifold learning. v1 ships **t-SNE** (van der Maaten &
//! Hinton, 2008): the workhorse for 2D / 3D embedding visualization.
//! UMAP is reserved for v2 if a concrete consumer asks; t-SNE alone
//! covers the OPTIC-K voicing-cluster use case and any general
//! "what's actually in this embedding" question.
//!
//! ## Quick start
//!
//! ```no_run
//! use ndarray::Array2;
//! use ix_manifold::Tsne;
//!
//! // 1000 points × 124 dims, e.g. OPTIC-K embeddings.
//! let x: Array2<f64> = Array2::zeros((1000, 124));
//! let y: Array2<f64> = Tsne::new()
//!     .with_perplexity(30.0)
//!     .with_n_iter(500)
//!     .with_seed(42)
//!     .fit_transform(x.view());
//! assert_eq!(y.shape(), &[1000, 2]);
//! ```
//!
//! ## Algorithm
//!
//! The standard O(n²) implementation:
//!
//! 1. Compute pairwise squared distances `D_ij = ||x_i − x_j||²`.
//! 2. For each row, bisect σ_i so that Shannon entropy of
//!    `P_{j|i} ∝ exp(−D_ij / 2σ_i²)` matches `log(perplexity)`.
//! 3. Symmetrize: `P_ij = (P_{j|i} + P_{i|j}) / 2n`.
//! 4. Initialize low-dim points `y_i ∼ N(0, 1e-4 · I)`.
//! 5. Iterate: Student-t `Q_ij ∝ (1 + ||y_i − y_j||²)^-1`,
//!    `∂C/∂y_i = 4 Σ_j (P_ij − Q_ij) · (1 + ||y_i − y_j||²)^-1 · (y_i − y_j)`,
//!    update with momentum + learning rate. Early exaggeration
//!    multiplies P by 12 for the first quarter of iters to spread
//!    clusters apart before fine-grained placement.
//!
//! Complexity is `O(n² · target_dim)` per iter, so n ≤ ~5000 is the
//! practical sweet spot. For 313K voicings, sample first or wait for
//! Barnes-Hut (v3).

#![deny(unsafe_code)]

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// t-SNE configuration. Sensible defaults match scikit-learn.
#[derive(Debug, Clone)]
pub struct Tsne {
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    target_dim: usize,
    early_exaggeration: f64,
    early_exaggeration_iters: usize,
    seed: u64,
}

impl Default for Tsne {
    fn default() -> Self {
        Self::new()
    }
}

impl Tsne {
    /// Defaults: perplexity 30, learning rate 200, 1000 iters,
    /// 2D output, early exaggeration 12× for first 250 iters, seed 0.
    pub fn new() -> Self {
        Self {
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            target_dim: 2,
            early_exaggeration: 12.0,
            early_exaggeration_iters: 250,
            seed: 0,
        }
    }

    pub fn with_perplexity(mut self, p: f64) -> Self {
        self.perplexity = p;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_n_iter(mut self, n: usize) -> Self {
        self.n_iter = n;
        self
    }
    pub fn with_target_dim(mut self, d: usize) -> Self {
        assert!(d >= 1, "target_dim must be ≥ 1");
        self.target_dim = d;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_early_exaggeration(mut self, factor: f64, iters: usize) -> Self {
        self.early_exaggeration = factor;
        self.early_exaggeration_iters = iters;
        self
    }

    /// Fit on `x` (rows = samples, cols = features) and return the
    /// low-dim embedding (rows = samples, cols = `target_dim`).
    pub fn fit_transform(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let n = x.nrows();
        assert!(
            n >= 2,
            "t-SNE needs ≥ 2 samples; got {n}"
        );
        assert!(
            self.perplexity < ((n - 1) as f64) / 3.0,
            "perplexity {} too large for {} samples (must be < (n-1)/3 = {})",
            self.perplexity,
            n,
            (n - 1) as f64 / 3.0
        );

        let dist_sq = pairwise_sq_dist(x);
        let mut p = compute_p(&dist_sq, self.perplexity);
        // Symmetrize and normalize
        p = (&p + &p.t()) / (2.0 * n as f64);
        // Floor for numerical stability
        p.mapv_inplace(|v| v.max(1e-12));

        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let normal = Normal::new(0.0, 1e-4_f64.sqrt()).unwrap();
        let mut y: Array2<f64> = Array2::from_shape_fn((n, self.target_dim), |_| {
            normal.sample(&mut rng)
        });
        let mut y_prev = y.clone();
        let mut p_eff = p.clone();
        p_eff *= self.early_exaggeration;

        for iter in 0..self.n_iter {
            if iter == self.early_exaggeration_iters {
                // End of early-exaggeration phase
                p_eff = p.clone();
            }
            let momentum = if iter < 250 { 0.5 } else { 0.8 };
            let grad = compute_gradient(&y, &p_eff);
            let next = &y - &(self.learning_rate * &grad) + &(momentum * (&y - &y_prev));
            y_prev = y;
            y = next;
            // Recenter to keep numerical drift away from origin small.
            let mean = y.mean_axis(Axis(0)).unwrap();
            y = &y - &mean;
        }
        y
    }
}

/// Pairwise squared Euclidean distances. `out[i, j] = ||x_i − x_j||²`.
fn pairwise_sq_dist(x: ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        let xi = x.row(i);
        for j in (i + 1)..n {
            let xj = x.row(j);
            let v = sq_dist(xi, xj);
            d[[i, j]] = v;
            d[[j, i]] = v;
        }
    }
    d
}

#[inline]
fn sq_dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum()
}

/// Compute the conditional probability matrix P with per-row σ_i
/// chosen to match the target perplexity by binary search on β = 1/(2σ²).
fn compute_p(dist_sq: &Array2<f64>, perplexity: f64) -> Array2<f64> {
    let n = dist_sq.nrows();
    let target_entropy = perplexity.ln();
    let mut p = Array2::<f64>::zeros((n, n));
    let max_iter = 50;
    let tol = 1e-5;

    for i in 0..n {
        let row = dist_sq.row(i);
        // Bisect β: low/high are heuristic bounds; reset on overshoot.
        let mut beta = 1.0_f64;
        let mut beta_low = f64::NEG_INFINITY;
        let mut beta_high = f64::INFINITY;
        let mut p_row = row_probs(row, beta, i);
        let mut entropy = entropy_of(&p_row);

        for _ in 0..max_iter {
            let diff = entropy - target_entropy;
            if diff.abs() < tol {
                break;
            }
            if diff > 0.0 {
                // Entropy too high → distribution too spread → increase β
                beta_low = beta;
                beta = if beta_high.is_infinite() {
                    beta * 2.0
                } else {
                    (beta + beta_high) / 2.0
                };
            } else {
                beta_high = beta;
                beta = if beta_low.is_infinite() {
                    beta / 2.0
                } else {
                    (beta + beta_low) / 2.0
                };
            }
            p_row = row_probs(row, beta, i);
            entropy = entropy_of(&p_row);
        }
        p.row_mut(i).assign(&p_row);
    }
    p
}

fn row_probs(row: ArrayView1<f64>, beta: f64, skip: usize) -> Array1<f64> {
    let n = row.len();
    let mut e = Array1::<f64>::zeros(n);
    for j in 0..n {
        e[j] = if j == skip {
            0.0
        } else {
            (-row[j] * beta).exp()
        };
    }
    let s: f64 = e.iter().sum();
    if s > 0.0 {
        e.mapv_inplace(|v| v / s);
    }
    e
}

fn entropy_of(p: &Array1<f64>) -> f64 {
    // Shannon entropy in nats: -Σ p_i log p_i (skip zeros).
    let mut h = 0.0;
    for &v in p.iter() {
        if v > 0.0 {
            h -= v * v.ln();
        }
    }
    h
}

/// Gradient of the KL divergence cost wrt low-dim points.
fn compute_gradient(y: &Array2<f64>, p: &Array2<f64>) -> Array2<f64> {
    let n = y.nrows();
    let d = y.ncols();
    // Q numerator: (1 + ||y_i − y_j||²)^-1 with diagonal = 0.
    let dist_sq = pairwise_sq_dist(y.view());
    let mut num = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            num[[i, j]] = if i == j { 0.0 } else { 1.0 / (1.0 + dist_sq[[i, j]]) };
        }
    }
    let z: f64 = num.sum();
    let z_safe = z.max(1e-12);
    // Q with floor so KL stays finite when comparing tiny values.
    let q = &num / z_safe;
    // PQ-diff weighted by num gives the per-pair attractive/repulsive force.
    let pq_diff = p - &q;
    let coeff = &pq_diff * &num;

    let mut grad = Array2::<f64>::zeros((n, d));
    for i in 0..n {
        let mut g = Array1::<f64>::zeros(d);
        let yi = y.row(i);
        for j in 0..n {
            if i == j {
                continue;
            }
            let yj = y.row(j);
            let c = coeff[[i, j]];
            for k in 0..d {
                g[k] += c * (yi[k] - yj[k]);
            }
        }
        let g = 4.0 * g;
        grad.slice_mut(s![i, ..]).assign(&g);
    }
    grad
}

// ─── Barnes-Hut t-SNE ──────────────────────────────────────────────
//
// Wrapper around the `bhtsne` crate (pinned to 0.5.3, edition 2021).
// O(n log n) per iteration via space-partitioning trees on both
// the kNN-perplexity step and the gradient step. Practical ceiling
// is ~100K-300K points where the exact `Tsne` above caps at ~5K.

/// Barnes-Hut t-SNE configuration. Defaults match scikit-learn /
/// openTSNE: perplexity=30, theta=0.5, 1000 epochs, 2D output.
#[derive(Debug, Clone)]
pub struct BarnesHutTsne {
    perplexity: f32,
    theta: f32,
    epochs: usize,
    target_dim: u8,
    seed: u64,
}

impl Default for BarnesHutTsne {
    fn default() -> Self {
        Self::new()
    }
}

impl BarnesHutTsne {
    /// Defaults: perplexity=30, theta=0.5 (Barnes-Hut accuracy knob;
    /// smaller = more accurate, slower), 1000 epochs, 2D, seed 0.
    pub fn new() -> Self {
        Self {
            perplexity: 30.0,
            theta: 0.5,
            epochs: 1000,
            target_dim: 2,
            seed: 0,
        }
    }

    pub fn with_perplexity(mut self, p: f32) -> Self {
        self.perplexity = p;
        self
    }
    /// Barnes-Hut θ: tradeoff between accuracy (small θ) and speed
    /// (large θ). 0.5 is the standard default. Range typically
    /// [0.1, 0.8].
    pub fn with_theta(mut self, t: f32) -> Self {
        self.theta = t;
        self
    }
    pub fn with_epochs(mut self, n: usize) -> Self {
        self.epochs = n;
        self
    }
    pub fn with_target_dim(mut self, d: u8) -> Self {
        assert!(d >= 1 && d <= 3, "target_dim must be 1..=3 (got {d})");
        self.target_dim = d;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Fit on `x` (rows = samples, cols = features) and return the
    /// low-dim embedding (rows = samples, cols = `target_dim`).
    ///
    /// Note: bhtsne uses f32 internally, so we down-cast on entry
    /// and up-cast on exit. The input matrix must outlive the call
    /// because bhtsne takes `&[&[f32]]` slices.
    pub fn fit_transform(&self, x: ArrayView2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let d = x.ncols();
        assert!(n >= 2, "Barnes-Hut t-SNE needs ≥ 2 samples; got {n}");
        assert!(
            self.perplexity < ((n - 1) as f32) / 3.0,
            "perplexity {} too large for {} samples (must be < (n-1)/3 = {})",
            self.perplexity,
            n,
            (n - 1) as f32 / 3.0
        );

        // Flatten to row-major f32 buffer + slice-of-slices view.
        let mut flat: Vec<f32> = Vec::with_capacity(n * d);
        for i in 0..n {
            for j in 0..d {
                flat.push(x[[i, j]] as f32);
            }
        }
        let samples: Vec<&[f32]> = flat.chunks(d).collect();

        // bhtsne uses thread_rng internally for the embedding init;
        // it is not seeded by us. For determinism guarantees we'd
        // need to fork bhtsne or use the exact `Tsne`. Document this
        // in the type docs. For now, seed is reserved for forward
        // compatibility (e.g. a future `with_init_seed` if bhtsne
        // exposes one).
        let _ = self.seed;

        let mut tsne_builder = bhtsne::tSNE::new(&samples);
        tsne_builder
            .embedding_dim(self.target_dim)
            .perplexity(self.perplexity)
            .epochs(self.epochs);
        tsne_builder.barnes_hut(self.theta, |a: &&[f32], b: &&[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        });

        // Extract the embedding. bhtsne stores it as a flat Vec<f32>
        // of length n × target_dim, row-major.
        let emb: Vec<f32> = tsne_builder.embedding();
        let target_dim = self.target_dim as usize;
        Array2::from_shape_fn((n, target_dim), |(i, j)| emb[i * target_dim + j] as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn output_shape_matches_target_dim() {
        let x = Array2::<f64>::eye(40);
        let y = Tsne::new()
            .with_n_iter(50)
            .with_perplexity(5.0)
            .with_seed(7)
            .fit_transform(x.view());
        assert_eq!(y.shape(), &[40, 2]);
        for v in y.iter() {
            assert!(v.is_finite(), "non-finite output at {v}");
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let x = Array2::<f64>::from_shape_fn((30, 5), |(i, j)| ((i + j) as f64).sin());
        let cfg = Tsne::new().with_n_iter(100).with_perplexity(5.0).with_seed(42);
        let a = cfg.clone().fit_transform(x.view());
        let b = cfg.fit_transform(x.view());
        for (av, bv) in a.iter().zip(b.iter()) {
            assert!((av - bv).abs() < 1e-9, "non-deterministic: {av} vs {bv}");
        }
    }

    #[test]
    fn well_separated_clusters_remain_separated() {
        // Two tight Gaussians in 10D centered far apart.
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let n_per = 30;
        let dim = 10;
        let normal = Normal::new(0.0, 0.05).unwrap();
        let mut x = Array2::<f64>::zeros((2 * n_per, dim));
        for i in 0..n_per {
            for k in 0..dim {
                x[[i, k]] = normal.sample(&mut rng);
                x[[n_per + i, k]] = 10.0 + normal.sample(&mut rng);
            }
        }
        let y = Tsne::new()
            .with_n_iter(500)
            .with_perplexity(8.0)
            .with_seed(1)
            .fit_transform(x.view());

        // Centroids of the two clusters in 2D.
        let c0 = y.slice(s![0..n_per, ..]).mean_axis(Axis(0)).unwrap();
        let c1 = y.slice(s![n_per.., ..]).mean_axis(Axis(0)).unwrap();
        let inter = sq_dist(c0.view(), c1.view()).sqrt();

        // Mean intra-cluster radius for each cluster.
        let mut intra0 = 0.0;
        for i in 0..n_per {
            intra0 += sq_dist(y.row(i), c0.view()).sqrt();
        }
        intra0 /= n_per as f64;
        let mut intra1 = 0.0;
        for i in 0..n_per {
            intra1 += sq_dist(y.row(n_per + i), c1.view()).sqrt();
        }
        intra1 /= n_per as f64;

        // Inter-cluster distance must be at least 2× the larger intra
        // radius — i.e. the two clusters don't visually overlap. A
        // tighter ratio would chase tunings of the optimizer rather
        // than verify correctness.
        let intra_max = intra0.max(intra1);
        assert!(
            inter > 2.0 * intra_max,
            "clusters not separated: inter={inter:.3}, intra_max={intra_max:.3}"
        );
    }

    #[test]
    fn target_dim_3_works() {
        let x = Array2::<f64>::from_shape_fn((20, 8), |(i, j)| (i * j) as f64);
        let y = Tsne::new()
            .with_n_iter(50)
            .with_perplexity(5.0)
            .with_target_dim(3)
            .with_seed(11)
            .fit_transform(x.view());
        assert_eq!(y.shape(), &[20, 3]);
    }

    #[test]
    #[should_panic(expected = "perplexity")]
    fn perplexity_too_large_panics() {
        // n=10, perplexity=30 → (n-1)/3 = 3, so 30 is invalid.
        let x = Array2::<f64>::eye(10);
        let _ = Tsne::new()
            .with_perplexity(30.0)
            .fit_transform(x.view());
    }

    #[test]
    fn pairwise_sq_dist_symmetric() {
        let x = array![[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]];
        let d = pairwise_sq_dist(x.view());
        // d[0,1] = 9 + 16 = 25, d[0,2] = 36 + 64 = 100
        assert!((d[[0, 1]] - 25.0).abs() < 1e-9);
        assert!((d[[1, 0]] - 25.0).abs() < 1e-9);
        assert!((d[[0, 2]] - 100.0).abs() < 1e-9);
        assert!(d[[0, 0]].abs() < 1e-9);
    }

    #[test]
    fn barnes_hut_output_shape_matches_target_dim() {
        // bhtsne needs n much greater than perplexity. Use 200 points,
        // perplexity 5, 2D output.
        let x = Array2::<f64>::from_shape_fn((200, 6), |(i, j)| ((i * j) as f64).cos());
        let y = BarnesHutTsne::new()
            .with_perplexity(5.0)
            .with_epochs(50)
            .with_target_dim(2)
            .fit_transform(x.view());
        assert_eq!(y.shape(), &[200, 2]);
        for v in y.iter() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn barnes_hut_separates_two_clusters() {
        // Two well-separated Gaussians in 8D — Barnes-Hut should split
        // them in 2D.
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        let n_per = 100;
        let dim = 8;
        let normal = Normal::new(0.0, 0.1).unwrap();
        let mut x = Array2::<f64>::zeros((2 * n_per, dim));
        for i in 0..n_per {
            for k in 0..dim {
                x[[i, k]] = normal.sample(&mut rng);
                x[[n_per + i, k]] = 8.0 + normal.sample(&mut rng);
            }
        }
        let y = BarnesHutTsne::new()
            .with_perplexity(20.0)
            .with_epochs(500)
            .with_theta(0.5)
            .fit_transform(x.view());

        let c0 = y.slice(s![0..n_per, ..]).mean_axis(Axis(0)).unwrap();
        let c1 = y.slice(s![n_per.., ..]).mean_axis(Axis(0)).unwrap();
        let inter = sq_dist(c0.view(), c1.view()).sqrt();
        let mut intra = 0.0;
        for i in 0..n_per {
            intra += sq_dist(y.row(i), c0.view()).sqrt();
            intra += sq_dist(y.row(n_per + i), c1.view()).sqrt();
        }
        intra /= (2 * n_per) as f64;
        assert!(
            inter > 2.0 * intra,
            "Barnes-Hut clusters not separated: inter={inter:.3}, intra_avg={intra:.3}"
        );
    }

    #[test]
    #[should_panic(expected = "perplexity")]
    fn barnes_hut_perplexity_too_large_panics() {
        let x = Array2::<f64>::eye(10);
        let _ = BarnesHutTsne::new()
            .with_perplexity(30.0)
            .fit_transform(x.view());
    }

    #[test]
    fn entropy_of_uniform_is_log_n() {
        let p = Array1::from_elem(8, 1.0 / 8.0);
        let h = entropy_of(&p);
        assert!((h - 8.0_f64.ln()).abs() < 1e-9);
    }
}
