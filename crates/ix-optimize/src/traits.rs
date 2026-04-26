//! Core traits for optimization.

use ndarray::Array1;

/// An objective function that can be evaluated and (optionally) differentiated.
pub trait ObjectiveFunction {
    /// Evaluate f(x).
    fn evaluate(&self, x: &Array1<f64>) -> f64;

    /// Gradient of f at x. Default uses numerical gradient.
    fn gradient(&self, x: &Array1<f64>) -> Array1<f64> {
        ix_math::calculus::numerical_gradient(&|p: &Array1<f64>| self.evaluate(p), x, 1e-7)
    }

    /// Dimensionality of the search space.
    fn dim(&self) -> usize;
}

/// An iterative parameter optimizer.
pub trait Optimizer {
    /// Perform one optimization step. Returns updated parameters.
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64>;

    /// Algorithm name.
    fn name(&self) -> &str;
}

/// Result of an optimization run.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub best_params: Array1<f64>,
    pub best_value: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Convenience function: wrap a closure as an ObjectiveFunction.
pub struct ClosureObjective<F: Fn(&Array1<f64>) -> f64> {
    pub f: F,
    pub dimensions: usize,
}

impl<F: Fn(&Array1<f64>) -> f64> ObjectiveFunction for ClosureObjective<F> {
    fn evaluate(&self, x: &Array1<f64>) -> f64 {
        (self.f)(x)
    }

    fn dim(&self) -> usize {
        self.dimensions
    }
}
