//! Ensemble traits.

use ndarray::{Array1, Array2};

/// An ensemble classifier that aggregates multiple base classifiers.
pub trait EnsembleClassifier {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<usize>);
    fn predict(&self, x: &Array2<f64>) -> Array1<usize>;
    fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64>;
    fn n_estimators(&self) -> usize;
}
