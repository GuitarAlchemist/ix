//! Support Vector Machine (linear, binary classification).
//!
//! Implements linear SVM with hinge loss and subgradient descent.
//! Minimizes: (1/2)||w||^2 + C * sum(max(0, 1 - yi*(w.xi + b)))

use ndarray::{Array1, Array2};
use crate::traits::Classifier;

/// Linear SVM for binary classification (classes 0 and 1).
pub struct LinearSVM {
    pub c: f64, // Regularization parameter
    pub learning_rate: f64,
    pub max_iterations: usize,
    weights: Option<Array1<f64>>,
    bias: f64,
}

impl LinearSVM {
    pub fn new(c: f64) -> Self {
        Self {
            c,
            learning_rate: 0.001,
            max_iterations: 1000,
            weights: None,
            bias: 0.0,
        }
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Raw decision function: w.x + b for each sample.
    fn decision_function(&self, x: &Array2<f64>) -> Array1<f64> {
        let w = self.weights.as_ref().expect("Model not fitted");
        x.dot(w) + self.bias
    }
}

impl Classifier for LinearSVM {
    #[allow(clippy::needless_range_loop)]
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        let (n, p) = x.dim();
        let mut w = Array1::zeros(p);
        let mut b = 0.0;

        // Map labels {0, 1} -> {-1, +1}
        let y_signed: Vec<f64> = y.iter().map(|&v| if v == 0 { -1.0 } else { 1.0 }).collect();

        for t in 0..self.max_iterations {
            // Decaying learning rate
            let lr = self.learning_rate / (1.0 + 0.001 * t as f64);

            let mut dw: Array1<f64> = Array1::zeros(p);
            let mut db = 0.0;

            for i in 0..n {
                let xi = x.row(i);
                let yi = y_signed[i];
                let margin = yi * (xi.dot(&w) + b);

                if margin < 1.0 {
                    // Hinge loss active: subgradient is -yi*xi
                    dw = &dw - &(yi * &xi);
                    db -= yi;
                }
            }

            // Average over samples, add regularization gradient for w
            let dw_reg = &w + &(&dw * (self.c / n as f64));
            let db_avg = db * self.c / n as f64;

            w = &w - &(lr * &dw_reg);
            b -= lr * db_avg;
        }

        self.weights = Some(w);
        self.bias = b;
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let scores = self.decision_function(x);
        scores.mapv(|s| if s >= 0.0 { 1 } else { 0 })
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        // Approximate probabilities via sigmoid (Platt scaling).
        let scores = self.decision_function(x);
        let n = x.nrows();
        let mut proba = Array2::zeros((n, 2));
        for i in 0..n {
            let p1 = 1.0 / (1.0 + (-scores[i]).exp());
            proba[[i, 1]] = p1;
            proba[[i, 0]] = 1.0 - p1;
        }
        proba
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::accuracy;
    use ndarray::array;

    #[test]
    fn test_linear_svm_separable() {
        let x = array![
            [0.0, 0.0], [0.5, 0.5], [0.3, 0.2],
            [3.0, 3.0], [3.5, 3.5], [3.2, 3.3]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let mut svm = LinearSVM::new(1.0)
            .with_learning_rate(0.01)
            .with_max_iterations(500);
        svm.fit(&x, &y);

        let pred = svm.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 1.0, "SVM should classify separable data, got acc={}", acc);
    }

    #[test]
    fn test_linear_svm_predict_proba() {
        let x = array![
            [0.0, 0.0], [0.5, 0.5],
            [5.0, 5.0], [5.5, 5.5]
        ];
        let y = array![0, 0, 1, 1];

        let mut svm = LinearSVM::new(1.0)
            .with_learning_rate(0.01)
            .with_max_iterations(500);
        svm.fit(&x, &y);

        let proba = svm.predict_proba(&x);
        for i in 0..proba.nrows() {
            let sum: f64 = proba.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1, got {}", sum);
        }
        // Class-0 sample should have higher prob for class 0
        assert!(proba[[0, 0]] > proba[[0, 1]]);
        // Class-1 sample should have higher prob for class 1
        assert!(proba[[3, 1]] > proba[[3, 0]]);
    }

    #[test]
    fn test_linear_svm_high_c() {
        let x = array![
            [-2.0, -2.0], [-1.0, -1.0],
            [1.0, 1.0], [2.0, 2.0]
        ];
        let y = array![0, 0, 1, 1];

        let mut svm = LinearSVM::new(100.0)
            .with_learning_rate(0.001)
            .with_max_iterations(1000);
        svm.fit(&x, &y);

        let pred = svm.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(acc >= 1.0, "High-C SVM should be perfect on separable data, got acc={}", acc);
    }
}
