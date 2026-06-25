//! Convergence criterion for iterative numerical algorithms.
//!
//! Iterative fits (k-means, GMM, NMF, gradient descent, …) all share the same
//! two knobs — a `max_iterations` cap and a `tolerance` on the per-iteration
//! change — and the same decision: *stop when the change has fallen below
//! tolerance*. Historically each algorithm re-declared those fields under
//! slightly different names (`max_iter`/`max_iterations`, `tol`/`tolerance`)
//! and inlined the `delta < tol` test (k-means even hard-coded `1e-10`).
//!
//! [`Convergence`] is the one place that owns both the config and the
//! [`converged`](Convergence::converged) decision. The *delta* itself stays with
//! each algorithm (centroid shift, log-likelihood change, reconstruction error
//! are genuinely different) — only the criterion is shared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Convergence {
    /// Hard cap on the number of iterations.
    pub max_iterations: usize,
    /// Stop once the per-iteration change drops below this.
    pub tolerance: f64,
}

impl Convergence {
    /// A criterion with an explicit cap and tolerance.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Builder: set the iteration cap.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Builder: set the convergence tolerance.
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Has the loop converged? `true` when the iteration's change `delta` (a
    /// non-negative magnitude) has fallen below [`tolerance`](Self::tolerance).
    /// Uses strict `<` to match the historical inlined checks exactly.
    pub fn converged(&self, delta: f64) -> bool {
        delta < self.tolerance
    }
}

impl Default for Convergence {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Convergence;

    #[test]
    fn converged_uses_strict_less_than() {
        let c = Convergence::new(100, 1e-6);
        assert!(c.converged(1e-9));
        assert!(!c.converged(1e-6)); // strict: equal is not converged
        assert!(!c.converged(1e-3));
    }

    #[test]
    fn builders_and_default() {
        let c = Convergence::default()
            .with_max_iterations(50)
            .with_tolerance(1e-4);
        assert_eq!(c.max_iterations, 50);
        assert!((c.tolerance - 1e-4).abs() < f64::EPSILON);
    }
}
