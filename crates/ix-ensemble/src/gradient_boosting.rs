//! Gradient Boosted Trees for classification.
//!
//! Implements gradient boosting with shallow regression trees (stumps) as
//! weak learners. Uses log-loss (cross-entropy) for binary and multiclass
//! problems with one-vs-rest pseudo-residual fitting.
//!
//! # Algorithm
//!
//! 1. Initialize predictions with class prior log-odds
//! 2. For each boosting round and each class:
//!    a. Compute pseudo-residuals (negative gradient of log-loss)
//!    b. Fit a regression tree to the residuals
//!    c. Update predictions: F += learning_rate * tree(x)
//! 3. Final prediction uses softmax over accumulated class scores
//!
//! # Example: Binary Classification
//!
//! ```
//! use ndarray::{array, Array2};
//! use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
//! use ix_ensemble::traits::EnsembleClassifier;
//!
//! let x = Array2::from_shape_vec((8, 2), vec![
//!     0.0, 0.0,  0.5, 0.5,  1.0, 0.0,  0.3, 0.2,
//!     5.0, 5.0,  5.5, 5.5,  6.0, 5.0,  5.3, 5.2,
//! ]).unwrap();
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! let mut gbc = GradientBoostedClassifier::new(50, 0.1, 3);
//! gbc.fit(&x, &y);
//!
//! let pred = gbc.predict(&x);
//! // Well-separated data → should classify correctly
//! assert_eq!(pred, y);
//! ```
//!
//! # Example: Multiclass Classification
//!
//! ```
//! use ndarray::{array, Array2};
//! use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
//! use ix_ensemble::traits::EnsembleClassifier;
//! use ix_supervised::metrics::accuracy;
//!
//! let x = Array2::from_shape_vec((9, 2), vec![
//!     0.0, 0.0,  0.5, 0.0,  0.0, 0.5,
//!     5.0, 0.0,  5.5, 0.0,  5.0, 0.5,
//!     0.0, 5.0,  0.5, 5.0,  0.0, 5.5,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
//!
//! let mut gbc = GradientBoostedClassifier::new(50, 0.1, 3);
//! gbc.fit(&x, &y);
//!
//! let pred = gbc.predict(&x);
//! let acc = accuracy(&y, &pred);
//! assert!(acc >= 0.8, "Should handle 3 classes, got acc={}", acc);
//! ```
//!
//! # Example: Tuning Hyperparameters
//!
//! ```
//! use ndarray::{array, Array2};
//! use ix_ensemble::gradient_boosting::GradientBoostedClassifier;
//! use ix_ensemble::traits::EnsembleClassifier;
//!
//! let x = Array2::from_shape_vec((6, 1), vec![
//!     0.0,  1.0,  2.0,  8.0,  9.0,  10.0,
//! ]).unwrap();
//! let y = array![0, 0, 0, 1, 1, 1];
//!
//! // More trees + lower learning rate = smoother fit
//! let mut gbc = GradientBoostedClassifier::new(100, 0.05, 2)
//!     .with_min_samples_leaf(1);
//! gbc.fit(&x, &y);
//!
//! let proba = gbc.predict_proba(&x);
//! // Class 0 samples should have high probability for class 0
//! assert!(proba[[0, 0]] > 0.7);
//! // Class 1 samples should have high probability for class 1
//! assert!(proba[[5, 1]] > 0.7);
//! ```

use ndarray::{Array1, Array2};

use crate::traits::EnsembleClassifier;

// ─────────────────────── Regression Stump ───────────────────────────────

/// A shallow regression tree used as a weak learner in gradient boosting.
///
/// Splits on a single feature/threshold. Each leaf predicts the mean of
/// its training targets.
#[derive(Clone, Debug)]
struct RegressionStump {
    feature: usize,
    threshold: f64,
    left_value: f64,
    right_value: f64,
}

impl RegressionStump {
    /// Fit a depth-1 regression tree to minimize squared error.
    fn fit(
        x: &Array2<f64>,
        residuals: &Array1<f64>,
        min_samples_leaf: usize,
    ) -> Self {
        let (n, p) = x.dim();
        let overall_mean = residuals.mean().unwrap_or(0.0);

        let mut best_mse = f64::INFINITY;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_left_val = overall_mean;
        let mut best_right_val = overall_mean;

        let mut indices: Vec<usize> = (0..n).collect();
        let total_sum: f64 = residuals.sum();

        for feat in 0..p {
            indices.sort_by(|&a, &b| x[[a, feat]].partial_cmp(&x[[b, feat]]).unwrap());

            // Running sums for incremental MSE computation
            let mut left_sum = 0.0;
            let mut left_count = 0usize;

            for split_pos in 0..n - 1 {
                let idx = indices[split_pos];
                left_sum += residuals[idx];
                left_count += 1;
                let right_count = n - left_count;

                if left_count < min_samples_leaf || right_count < min_samples_leaf {
                    continue;
                }

                // Skip if same value
                let val = x[[indices[split_pos], feat]];
                let next_val = x[[indices[split_pos + 1], feat]];
                if (val - next_val).abs() < 1e-12 {
                    continue;
                }

                let left_mean = left_sum / left_count as f64;
                let right_sum = total_sum - left_sum;
                let right_mean = right_sum / right_count as f64;

                // MSE = sum of squared deviations from mean in each partition
                // Simplified: total_var - left_count * left_mean^2 - right_count * right_mean^2
                // We just need relative comparison, so use negative "variance explained"
                let mse = -(left_count as f64 * left_mean * left_mean
                    + right_count as f64 * right_mean * right_mean);

                if mse < best_mse {
                    best_mse = mse;
                    best_feature = feat;
                    best_threshold = (val + next_val) / 2.0;
                    best_left_val = left_mean;
                    best_right_val = right_mean;
                }
            }
        }

        RegressionStump {
            feature: best_feature,
            threshold: best_threshold,
            left_value: best_left_val,
            right_value: best_right_val,
        }
    }

    fn predict_one(&self, sample: &ndarray::ArrayView1<f64>) -> f64 {
        if sample[self.feature] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        Array1::from_iter((0..x.nrows()).map(|i| self.predict_one(&x.row(i))))
    }
}

// ────────────────── Gradient Boosted Classifier ─────────────────────────

/// Gradient Boosted Trees classifier.
///
/// Uses gradient boosting with regression stumps to fit pseudo-residuals
/// from the log-loss objective. Supports binary and multiclass problems.
pub struct GradientBoostedClassifier {
    /// Number of boosting rounds.
    pub n_estimators: usize,
    /// Learning rate (shrinkage factor).
    pub learning_rate: f64,
    /// Maximum depth per weak learner (currently depth-1 stumps).
    pub max_depth: usize,
    /// Minimum samples required in each leaf.
    pub min_samples_leaf: usize,

    // Learned state
    n_classes: usize,
    /// Per-class initial log-odds (class priors).
    init_scores: Vec<f64>,
    /// trees[round][class] = regression stump
    trees: Vec<Vec<RegressionStump>>,
}

impl GradientBoostedClassifier {
    /// Create a new gradient boosting classifier.
    ///
    /// # Arguments
    /// - `n_estimators` — number of boosting rounds
    /// - `learning_rate` — step size shrinkage (0.01–0.3 typical)
    /// - `max_depth` — maximum tree depth (1 = stump)
    pub fn new(n_estimators: usize, learning_rate: f64, max_depth: usize) -> Self {
        Self {
            n_estimators,
            learning_rate,
            max_depth,
            min_samples_leaf: 1,
            n_classes: 0,
            init_scores: Vec::new(),
            trees: Vec::new(),
        }
    }

    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Raw scores (log-odds) for each sample and class.
    fn raw_scores(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let mut scores = Array2::from_shape_fn((n, self.n_classes), |(_, c)| self.init_scores[c]);

        for round_trees in &self.trees {
            for (c, tree) in round_trees.iter().enumerate() {
                let preds = tree.predict(x);
                for i in 0..n {
                    scores[[i, c]] += self.learning_rate * preds[i];
                }
            }
        }

        scores
    }
}

/// Softmax: convert raw scores to probabilities.
fn softmax_row(scores: &ndarray::ArrayView1<f64>) -> Vec<f64> {
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

impl EnsembleClassifier for GradientBoostedClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        let n = x.nrows();
        self.n_classes = *y.iter().max().unwrap() + 1;

        // Initialize with class log-priors
        let mut class_counts = vec![0usize; self.n_classes];
        for &label in y.iter() {
            class_counts[label] += 1;
        }
        self.init_scores = class_counts.iter()
            .map(|&c| ((c as f64 + 1.0) / (n as f64 + self.n_classes as f64)).ln())
            .collect();

        // Working scores: F[i, c]
        let mut scores = Array2::from_shape_fn((n, self.n_classes), |(_, c)| self.init_scores[c]);

        self.trees.clear();

        for _round in 0..self.n_estimators {
            // Compute probabilities via inline softmax (avoids per-row Vec alloc)
            let mut proba = Array2::zeros((n, self.n_classes));
            for i in 0..n {
                let max_s = scores.row(i).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0;
                for c in 0..self.n_classes {
                    proba[[i, c]] = (scores[[i, c]] - max_s).exp();
                    sum += proba[[i, c]];
                }
                for c in 0..self.n_classes {
                    proba[[i, c]] /= sum;
                }
            }

            // Fit one stump per class on pseudo-residuals
            let mut round_trees = Vec::with_capacity(self.n_classes);
            for c in 0..self.n_classes {
                // Pseudo-residual: y_one_hot - p
                let residuals = Array1::from_iter((0..n).map(|i| {
                    let target = if y[i] == c { 1.0 } else { 0.0 };
                    target - proba[[i, c]]
                }));

                let stump = RegressionStump::fit(x, &residuals, self.min_samples_leaf);

                // Update working scores
                let preds = stump.predict(x);
                for i in 0..n {
                    scores[[i, c]] += self.learning_rate * preds[i];
                }

                round_trees.push(stump);
            }
            self.trees.push(round_trees);
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let proba = self.predict_proba(x);
        Array1::from_iter((0..x.nrows()).map(|i| {
            proba
                .row(i)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        }))
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let scores = self.raw_scores(x);
        let n = x.nrows();
        let mut proba = Array2::zeros((n, self.n_classes));
        for i in 0..n {
            let p = softmax_row(&scores.row(i));
            for c in 0..self.n_classes {
                proba[[i, c]] = p[c];
            }
        }
        proba
    }

    fn n_estimators(&self) -> usize {
        self.trees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ix_supervised::metrics::accuracy;

    #[test]
    fn test_gbc_binary_separable() {
        let x = Array2::from_shape_vec((8, 2), vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2,
            5.0, 5.0, 5.5, 5.5, 6.0, 5.0, 5.3, 5.2,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut gbc = GradientBoostedClassifier::new(50, 0.1, 3);
        gbc.fit(&x, &y);

        let pred = gbc.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 1.0, "Should perfectly classify separable data, got acc={}", acc);
    }

    #[test]
    fn test_gbc_multiclass() {
        let x = Array2::from_shape_vec((9, 2), vec![
            0.0, 0.0, 0.5, 0.0, 0.0, 0.5,
            5.0, 0.0, 5.5, 0.0, 5.0, 0.5,
            0.0, 5.0, 0.5, 5.0, 0.0, 5.5,
        ]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let mut gbc = GradientBoostedClassifier::new(50, 0.1, 3);
        gbc.fit(&x, &y);

        let pred = gbc.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 0.8, "Should handle 3 classes, got acc={}", acc);
    }

    #[test]
    fn test_gbc_predict_proba_sums_to_one() {
        let x = Array2::from_shape_vec((4, 2), vec![
            0.0, 0.0, 1.0, 1.0,
            5.0, 5.0, 6.0, 6.0,
        ]).unwrap();
        let y = array![0, 0, 1, 1];

        let mut gbc = GradientBoostedClassifier::new(20, 0.1, 2);
        gbc.fit(&x, &y);

        let proba = gbc.predict_proba(&x);
        for i in 0..proba.nrows() {
            let sum: f64 = proba.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-6, "Row {} probabilities sum to {}, expected 1.0", i, sum);
        }
    }

    #[test]
    fn test_gbc_n_estimators() {
        let x = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let mut gbc = GradientBoostedClassifier::new(30, 0.1, 2);
        gbc.fit(&x, &y);

        assert_eq!(gbc.n_estimators(), 30);
    }

    #[test]
    fn test_gbc_learning_rate_effect() {
        let x = Array2::from_shape_vec((6, 1), vec![
            0.0, 1.0, 2.0, 8.0, 9.0, 10.0,
        ]).unwrap();
        let y = array![0, 0, 0, 1, 1, 1];

        // High LR with few trees
        let mut gbc_fast = GradientBoostedClassifier::new(10, 0.5, 2);
        gbc_fast.fit(&x, &y);
        let pred_fast = gbc_fast.predict(&x);
        let acc_fast = accuracy(&y, &pred_fast);

        // Low LR with many trees
        let mut gbc_slow = GradientBoostedClassifier::new(100, 0.05, 2);
        gbc_slow.fit(&x, &y);
        let pred_slow = gbc_slow.predict(&x);
        let acc_slow = accuracy(&y, &pred_slow);

        // Both should work on this easy data
        assert!(acc_fast >= 0.8, "High LR should still work, got {}", acc_fast);
        assert!(acc_slow >= 0.8, "Low LR should work with enough rounds, got {}", acc_slow);
    }

    #[test]
    fn test_gbc_with_min_samples_leaf() {
        let x = Array2::from_shape_vec((8, 2), vec![
            0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.3, 0.2,
            5.0, 5.0, 5.5, 5.5, 6.0, 5.0, 5.3, 5.2,
        ]).unwrap();
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut gbc = GradientBoostedClassifier::new(50, 0.1, 3)
            .with_min_samples_leaf(2);
        gbc.fit(&x, &y);

        let pred = gbc.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 0.9, "min_samples_leaf=2 should still classify well, got {}", acc);
    }

    #[test]
    fn test_regression_stump_basic() {
        // Simple 1D data: left half should predict negative, right half positive
        let x = Array2::from_shape_vec((6, 1), vec![
            0.0, 1.0, 2.0, 8.0, 9.0, 10.0,
        ]).unwrap();
        let residuals = array![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];

        let stump = RegressionStump::fit(&x, &residuals, 1);
        let preds = stump.predict(&x);

        // Left partition should predict ~-1, right should predict ~+1
        for i in 0..3 {
            assert!(preds[i] < 0.0, "Left prediction should be negative, got {}", preds[i]);
        }
        for i in 3..6 {
            assert!(preds[i] > 0.0, "Right prediction should be positive, got {}", preds[i]);
        }
    }
}
