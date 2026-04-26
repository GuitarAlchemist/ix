//! Logistic Regression (binary classification via gradient descent).

use ndarray::{Array1, Array2};

use crate::traits::Classifier;
use ix_math::activation::sigmoid;

/// Binary Logistic Regression.
pub struct LogisticRegression {
    pub weights: Option<Array1<f64>>,
    pub bias: f64,
    pub learning_rate: f64,
    pub max_iterations: usize,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: 0.0,
            learning_rate: 0.01,
            max_iterations: 1000,
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
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for LogisticRegression {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>) {
        let (n, p) = x.dim();
        let mut w = Array1::zeros(p);
        let mut b = 0.0;

        let y_f: Array1<f64> = y.mapv(|v| v as f64);

        for _ in 0..self.max_iterations {
            // Forward: z = Xw + b, a = sigmoid(z)
            let z = x.dot(&w) + b;
            let a = z.mapv(sigmoid);

            // Gradients
            let error = &a - &y_f;
            let dw = x.t().dot(&error) / n as f64;
            let db = error.sum() / n as f64;

            // Update
            w = w - self.learning_rate * &dw;
            b -= self.learning_rate * db;
        }

        self.weights = Some(w);
        self.bias = b;
    }

    fn predict(&self, x: &Array2<f64>) -> Array1<usize> {
        let proba = self.predict_proba(x);
        Array1::from_iter(
            proba
                .column(1)
                .iter()
                .map(|&p| if p >= 0.5 { 1 } else { 0 }),
        )
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let w = self.weights.as_ref().expect("Model not fitted");
        let z = x.dot(w) + self.bias;
        let p1 = z.mapv(sigmoid);
        let p0 = p1.mapv(|p| 1.0 - p);

        let n = x.nrows();
        let mut proba = Array2::zeros((n, 2));
        proba.column_mut(0).assign(&p0);
        proba.column_mut(1).assign(&p1);
        proba
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::accuracy;
    use ndarray::array;

    #[test]
    fn test_logistic_regression_linearly_separable() {
        // Simple linearly separable data
        let x = array![
            [0.0, 0.0],
            [0.5, 0.5],
            [0.3, 0.2],
            [3.0, 3.0],
            [3.5, 3.5],
            [3.2, 3.3]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iterations(1000);
        model.fit(&x, &y);

        let pred = model.predict(&x);
        let acc = accuracy(&y, &pred);
        assert!(
            acc > 0.8,
            "Should classify linearly separable data, got acc={}",
            acc
        );
    }
}
