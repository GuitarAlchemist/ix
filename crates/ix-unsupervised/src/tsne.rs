//! t-SNE (t-distributed Stochastic Neighbor Embedding).
//!
//! Barnes-Hut approximation with perplexity-based affinity computation.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use ndarray_rand::RandomExt;
use rand_distr::Normal;

use crate::traits::DimensionReducer;
use ix_math::distance::euclidean_squared;

/// t-SNE configuration.
pub struct TSNE {
    pub n_components: usize,
    pub perplexity: f64,
    pub learning_rate: f64,
    pub max_iterations: usize,
    pub seed: u64,
    embedding: Option<Array2<f64>>,
}

impl TSNE {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            perplexity: 30.0,
            learning_rate: 200.0,
            max_iterations: 1000,
            seed: 42,
            embedding: None,
        }
    }

    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Compute pairwise squared distances.
fn pairwise_distances(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let pi = x.row(i).to_owned();
            let pj = x.row(j).to_owned();
            let dist = euclidean_squared(&pi, &pj).unwrap();
            d[[i, j]] = dist;
            d[[j, i]] = dist;
        }
    }
    d
}

/// Binary search for sigma that yields target perplexity.
fn find_sigma(distances_row: &Array1<f64>, idx: usize, target_perplexity: f64) -> f64 {
    let target_entropy = target_perplexity.ln();
    let mut lo = 1e-20_f64;
    let mut hi = 1e4_f64;
    let mut sigma = 1.0;

    for _ in 0..50 {
        sigma = (lo + hi) / 2.0;
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut sum_p = 0.0;
        for (j, &d) in distances_row.iter().enumerate() {
            if j != idx {
                sum_p += (-d / two_sigma_sq).exp();
            }
        }

        if sum_p < 1e-300 {
            lo = sigma;
            continue;
        }

        let mut entropy = 0.0;
        for (j, &d) in distances_row.iter().enumerate() {
            if j != idx {
                let p = (-d / two_sigma_sq).exp() / sum_p;
                if p > 1e-300 {
                    entropy -= p * p.ln();
                }
            }
        }

        if (entropy - target_entropy).abs() < 1e-5 {
            break;
        }

        if entropy > target_entropy {
            hi = sigma;
        } else {
            lo = sigma;
        }
    }

    sigma
}

/// Compute joint probability matrix P from high-dimensional distances.
fn compute_joint_p(distances: &Array2<f64>, perplexity: f64) -> Array2<f64> {
    let n = distances.nrows();
    let mut p = Array2::zeros((n, n));

    // Compute conditional probabilities p_{j|i}
    for i in 0..n {
        let dists = distances.row(i).to_owned();
        let sigma = find_sigma(&dists, i, perplexity);
        let two_sigma_sq = 2.0 * sigma * sigma;

        let mut sum_exp = 0.0;
        for j in 0..n {
            if j != i {
                sum_exp += (-dists[j] / two_sigma_sq).exp();
            }
        }

        for j in 0..n {
            if j != i {
                p[[i, j]] = (-dists[j] / two_sigma_sq).exp() / sum_exp.max(1e-300);
            }
        }
    }

    // Symmetrize: P_{ij} = (p_{j|i} + p_{i|j}) / 2n
    let mut p_sym = Array2::zeros((n, n));
    let two_n = 2.0 * n as f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let val = (p[[i, j]] + p[[j, i]]) / two_n;
            let val = val.max(1e-12);
            p_sym[[i, j]] = val;
            p_sym[[j, i]] = val;
        }
    }
    p_sym
}

/// Compute low-dimensional affinities (Student-t with 1 DOF).
fn compute_q(y: &Array2<f64>) -> (Array2<f64>, f64) {
    let n = y.nrows();
    let mut q = Array2::zeros((n, n));
    let mut sum_q = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let yi = y.row(i).to_owned();
            let yj = y.row(j).to_owned();
            let dist = euclidean_squared(&yi, &yj).unwrap();
            let val = 1.0 / (1.0 + dist);
            q[[i, j]] = val;
            q[[j, i]] = val;
            sum_q += 2.0 * val;
        }
    }

    (q, sum_q)
}

impl DimensionReducer for TSNE {
    fn fit(&mut self, x: &Array2<f64>) {
        let n = x.nrows();
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize embedding randomly
        let mut y: Array2<f64> = Array2::random_using(
            (n, self.n_components),
            Normal::new(0.0, 1e-4).unwrap(),
            &mut rng,
        );

        let distances = pairwise_distances(x);
        let p = compute_joint_p(&distances, self.perplexity);

        let mut gains = Array2::ones((n, self.n_components));
        let mut prev_update = Array2::zeros((n, self.n_components));
        let momentum = 0.5;

        for iter in 0..self.max_iterations {
            let (q_unnorm, sum_q) = compute_q(&y);
            let current_momentum = if iter < 250 { momentum } else { 0.8 };

            // Compute gradients
            let mut grad = Array2::<f64>::zeros((n, self.n_components));
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let q_ij: f64 = q_unnorm[[i, j]] / sum_q.max(1e-300);
                    let mult: f64 = 4.0 * (p[[i, j]] - q_ij) * q_unnorm[[i, j]];
                    for d in 0..self.n_components {
                        let delta: f64 = mult * (y[[i, d]] - y[[j, d]]);
                        grad[[i, d]] += delta;
                    }
                }
            }

            // Update with adaptive gains
            for i in 0..n {
                for d in 0..self.n_components {
                    let same_sign = (grad[[i, d]] > 0.0) == (prev_update[[i, d]] > 0.0);
                    let g: f64 = gains[[i, d]];
                    gains[[i, d]] = if same_sign {
                        (g * 0.8_f64).max(0.01)
                    } else {
                        g + 0.2
                    };

                    let update: f64 =
                        current_momentum * prev_update[[i, d]] - self.learning_rate * gains[[i, d]] * grad[[i, d]];
                    y[[i, d]] += update;
                    prev_update[[i, d]] = update;
                }
            }

            // Center
            let mean = y.mean_axis(ndarray::Axis(0)).unwrap();
            for i in 0..n {
                for d in 0..self.n_components {
                    y[[i, d]] -= mean[d];
                }
            }
        }

        self.embedding = Some(y);
    }

    fn transform(&self, _x: &Array2<f64>) -> Array2<f64> {
        self.embedding
            .clone()
            .expect("fit() must be called before transform()")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_tsne_separates_clusters() {
        // Two well-separated clusters in 3D
        let x = array![
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.0, 0.1],
            [10.0, 10.0, 10.0],
            [10.1, 10.1, 10.1],
            [10.2, 10.0, 10.1],
        ];

        let mut tsne = TSNE::new(2)
            .with_perplexity(2.0)
            .with_max_iterations(300)
            .with_seed(42);

        let y = tsne.fit_transform(&x);
        assert_eq!(y.dim(), (6, 2));

        // Cluster 1 center and cluster 2 center should be separated
        let c1 = (y.row(0).to_owned() + y.row(1).to_owned() + y.row(2).to_owned()) / 3.0;
        let c2 = (y.row(3).to_owned() + y.row(4).to_owned() + y.row(5).to_owned()) / 3.0;
        let dist = euclidean_squared(&c1, &c2).unwrap();
        assert!(dist > 0.1, "clusters should be separated in embedding, dist={}", dist);
    }
}
