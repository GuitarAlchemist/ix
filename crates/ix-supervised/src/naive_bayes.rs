//! Gaussian Naive Bayes classifier.

use ndarray::{Array1, Array2};

use crate::traits::Classifier;

/// Gaussian Naive Bayes: assumes features are normally distributed per class.
pub struct GaussianNaiveBayes {
    /// Per-class means: means[class][feature]
    means: Vec<Array1<f64>>,
    /// Per-class variances
    variances: Vec<Array1<f64>>,
    /// Prior probabilities
    priors: Vec<f64>,
    n_classes: usize,
}

impl GaussianNaiveBayes {
    pub fn new() -> Self {
        Self {
            means: Vec::new(),
            variances: Vec::new(),
            priors: Vec::new(),
            n_classes: 0,
        }
    }
}

impl Default for GaussianNaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for GaussianNaiveBayes {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        self.n_classes = *y.iter().max().unwrap() + 1;
        let n = x.nrows() as f64;

        self.means.clear();
        self.variances.clear();
        self.priors.clear();

        for c in 0..self.n_classes {
            let mask: Vec<usize> = (0..y.len()).filter(|&i| y[i] == c).collect();
            let count = mask.len() as f64;
            self.priors.push(count / n);

            let class_data: Array2<f64> =
                Array2::from_shape_fn((mask.len(), x.ncols()), |(i, j)| x[[mask[i], j]]);

            let mean = class_data.mean_axis(ndarray::Axis(0)).unwrap();
            let var = class_data
                .mapv(|v| v * v)
                .mean_axis(ndarray::Axis(0))
                .unwrap()
                - &mean.mapv(|v| v * v);
            // Add small epsilon to avoid division by zero
            let var = var.mapv(|v| v.max(1e-9));

            self.means.push(mean);
            self.variances.push(var);
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let proba = self.predict_proba(x);
        Array1::from_iter((0..x.nrows()).map(|i| {
            proba
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }))
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut log_proba = Array2::zeros((n, self.n_classes));

        for c in 0..self.n_classes {
            let log_prior = self.priors[c].ln();
            for i in 0..n {
                let mut log_likelihood = log_prior;
                for j in 0..x.ncols() {
                    let mean = self.means[c][j];
                    let var = self.variances[c][j];
                    // Log of Gaussian PDF
                    let diff = x[[i, j]] - mean;
                    log_likelihood +=
                        -0.5 * (2.0 * std::f64::consts::PI * var).ln() - 0.5 * diff * diff / var;
                }
                log_proba[[i, c]] = log_likelihood;
            }
        }

        // Convert log probabilities to probabilities (softmax-style)
        for i in 0..n {
            let max_log = log_proba
                .row(i)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut row_sum = 0.0;
            for c in 0..self.n_classes {
                log_proba[[i, c]] = (log_proba[[i, c]] - max_log).exp();
                row_sum += log_proba[[i, c]];
            }
            for c in 0..self.n_classes {
                log_proba[[i, c]] /= row_sum;
            }
        }

        log_proba
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::accuracy;
    use ndarray::array;

    #[test]
    fn test_gaussian_nb() {
        let x = array![
            [1.0, 1.0],
            [1.5, 2.0],
            [2.0, 1.0],
            [6.0, 5.0],
            [7.0, 7.0],
            [6.5, 6.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let mut model = GaussianNaiveBayes::new();
        model.fit(&x, &y);

        let pred = model.predict(&x);
        assert!(accuracy(&y, &pred) > 0.8);
    }
}
