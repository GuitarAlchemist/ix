//! Gaussian Mixture Model with Expectation-Maximization.

use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::traits::Clusterer;

/// Gaussian Mixture Model.
pub struct GMM {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub seed: u64,
    /// Mixture weights (k,)
    pub weights: Option<Array1<f64>>,
    /// Component means (k, d)
    pub means: Option<Array2<f64>>,
    /// Component covariance diagonals (k, d) — diagonal covariance for simplicity
    pub covariances: Option<Array2<f64>>,
}

impl GMM {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 100,
            tolerance: 1e-6,
            seed: 42,
            weights: None,
            means: None,
            covariances: None,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Compute log-likelihood of data under current model.
    pub fn log_likelihood(&self, x: &Array2<f64>) -> f64 {
        let means = self.means.as_ref().expect("not fitted");
        let covs = self.covariances.as_ref().expect("not fitted");
        let weights = self.weights.as_ref().expect("not fitted");

        let mut ll = 0.0;
        for i in 0..x.nrows() {
            let mut sum = 0.0;
            for k in 0..self.k {
                sum += weights[k] * gaussian_pdf(&x.row(i).to_owned(), &means.row(k).to_owned(), &covs.row(k).to_owned());
            }
            ll += sum.max(1e-300).ln();
        }
        ll
    }
}

/// Diagonal Gaussian PDF.
fn gaussian_pdf(x: &Array1<f64>, mean: &Array1<f64>, cov_diag: &Array1<f64>) -> f64 {
    let d = x.len() as f64;
    let diff = x - mean;
    let exponent: f64 = diff.iter().zip(cov_diag.iter())
        .map(|(&di, &ci)| di * di / ci.max(1e-10))
        .sum();
    let det: f64 = cov_diag.iter().map(|&c| c.max(1e-10)).product();
    let norm = (2.0 * std::f64::consts::PI).powf(d / 2.0) * det.sqrt();
    (-0.5 * exponent).exp() / norm.max(1e-300)
}

impl Clusterer for GMM {
    fn fit(&mut self, x: &Array2<f64>) {
        let n = x.nrows();
        let d = x.ncols();
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Initialize with K-Means++ style: pick k random points as means
        use rand::Rng;
        let indices: Vec<usize> = {
            let mut idxs = Vec::with_capacity(self.k);
            for _ in 0..self.k {
                let mut idx = rng.random_range(0..n);
                while idxs.contains(&idx) {
                    idx = rng.random_range(0..n);
                }
                idxs.push(idx);
            }
            idxs
        };

        let mut means = Array2::zeros((self.k, d));
        for (ki, &idx) in indices.iter().enumerate() {
            means.row_mut(ki).assign(&x.row(idx));
        }

        // Initialize covariances to data variance
        let data_var = x.var_axis(Axis(0), 0.0);
        let mut covariances = Array2::zeros((self.k, d));
        for ki in 0..self.k {
            covariances.row_mut(ki).assign(&data_var);
        }

        // Equal weights
        let mut weights = Array1::from_elem(self.k, 1.0 / self.k as f64);

        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..self.max_iterations {
            // E-step: compute responsibilities
            let mut resp = Array2::zeros((n, self.k));
            for i in 0..n {
                let xi = x.row(i).to_owned();
                let mut row_sum = 0.0;
                for k in 0..self.k {
                    let p = weights[k] * gaussian_pdf(&xi, &means.row(k).to_owned(), &covariances.row(k).to_owned());
                    resp[[i, k]] = p;
                    row_sum += p;
                }
                if row_sum > 1e-300 {
                    for k in 0..self.k {
                        resp[[i, k]] /= row_sum;
                    }
                }
            }

            // M-step
            let nk: Array1<f64> = resp.sum_axis(Axis(0));

            for k in 0..self.k {
                let nk_val = nk[k].max(1e-10);

                // Update means
                let mut new_mean = Array1::zeros(d);
                for i in 0..n {
                    for j in 0..d {
                        new_mean[j] += resp[[i, k]] * x[[i, j]];
                    }
                }
                new_mean /= nk_val;
                means.row_mut(k).assign(&new_mean);

                // Update covariances (diagonal)
                let mut new_cov = Array1::zeros(d);
                for i in 0..n {
                    for j in 0..d {
                        let diff = x[[i, j]] - new_mean[j];
                        new_cov[j] += resp[[i, k]] * diff * diff;
                    }
                }
                new_cov /= nk_val;
                // Add regularization to prevent singular covariances
                new_cov.mapv_inplace(|v| v.max(1e-6));
                covariances.row_mut(k).assign(&new_cov);

                // Update weights
                weights[k] = nk_val / n as f64;
            }

            // Check convergence
            self.means = Some(means.clone());
            self.covariances = Some(covariances.clone());
            self.weights = Some(weights.clone());

            let ll = self.log_likelihood(x);
            if (ll - prev_ll).abs() < self.tolerance {
                break;
            }
            prev_ll = ll;
        }

        self.means = Some(means);
        self.covariances = Some(covariances);
        self.weights = Some(weights);
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let means = self.means.as_ref().expect("not fitted");
        let covs = self.covariances.as_ref().expect("not fitted");
        let weights = self.weights.as_ref().expect("not fitted");

        Array1::from_iter((0..x.nrows()).map(|i| {
            let xi = x.row(i).to_owned();
            let mut best_k = 0;
            let mut best_p = f64::NEG_INFINITY;
            for k in 0..self.k {
                let p = weights[k] * gaussian_pdf(&xi, &means.row(k).to_owned(), &covs.row(k).to_owned());
                if p > best_p {
                    best_p = p;
                    best_k = k;
                }
            }
            best_k
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_gmm_two_clusters() {
        let x = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [-0.1, 0.2],
            [10.0, 10.0], [10.1, 10.1], [10.2, 10.0], [9.9, 10.2]
        ];

        let mut gmm = GMM::new(2).with_seed(42);
        let labels = gmm.fit_predict(&x);

        // First 4 should be same cluster, last 4 same cluster
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[2], labels[3]);
        assert_eq!(labels[4], labels[5]);
        assert_eq!(labels[5], labels[6]);
        assert_eq!(labels[6], labels[7]);
        assert_ne!(labels[0], labels[4]);
    }

    #[test]
    fn test_gmm_log_likelihood_increases() {
        let x = array![
            [0.0, 0.0], [0.5, 0.5], [1.0, 0.0],
            [10.0, 10.0], [10.5, 10.5], [11.0, 10.0]
        ];

        let mut gmm = GMM::new(2).with_seed(42).with_max_iterations(1);
        gmm.fit(&x);
        let ll1 = gmm.log_likelihood(&x);

        let mut gmm2 = GMM::new(2).with_seed(42).with_max_iterations(50);
        gmm2.fit(&x);
        let ll2 = gmm2.log_likelihood(&x);

        assert!(ll2 >= ll1 - 1e-6, "ll should increase: {} vs {}", ll1, ll2);
    }
}
