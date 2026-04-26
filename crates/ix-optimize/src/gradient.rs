//! Gradient-based optimizers: SGD, Momentum, Adam.

use ndarray::Array1;

use crate::convergence::ConvergenceCriteria;
use crate::traits::{ObjectiveFunction, OptimizeResult, Optimizer};

/// Stochastic Gradient Descent.
pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64> {
        params - &(self.learning_rate * gradient)
    }

    fn name(&self) -> &str {
        "SGD"
    }
}

/// SGD with Momentum.
pub struct Momentum {
    pub learning_rate: f64,
    pub momentum: f64,
    velocity: Option<Array1<f64>>,
}

impl Momentum {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            learning_rate,
            momentum,
            velocity: None,
        }
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64> {
        let v = match &self.velocity {
            Some(v) => self.momentum * v - self.learning_rate * gradient,
            None => -self.learning_rate * gradient,
        };
        let new_params = params + &v;
        self.velocity = Some(v);
        new_params
    }

    fn name(&self) -> &str {
        "Momentum"
    }
}

/// Adam optimizer.
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: Option<Array1<f64>>,
    v: Option<Array1<f64>>,
    t: usize,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: None,
            v: None,
            t: 0,
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &Array1<f64>, gradient: &Array1<f64>) -> Array1<f64> {
        self.t += 1;
        let t = self.t as f64;

        let m = match &self.m {
            Some(m) => self.beta1 * m + (1.0 - self.beta1) * gradient,
            None => (1.0 - self.beta1) * gradient.clone(),
        };
        let v = match &self.v {
            Some(v) => self.beta2 * v + (1.0 - self.beta2) * &gradient.mapv(|g| g * g),
            None => (1.0 - self.beta2) * gradient.mapv(|g| g * g),
        };

        let m_hat = &m / (1.0 - self.beta1.powf(t));
        let v_hat = &v / (1.0 - self.beta2.powf(t));

        let new_params =
            params - &(self.learning_rate * &m_hat / &(v_hat.mapv(f64::sqrt) + self.epsilon));

        self.m = Some(m);
        self.v = Some(v);
        new_params
    }

    fn name(&self) -> &str {
        "Adam"
    }
}

/// Run gradient-based optimization loop.
pub fn minimize<O, F>(
    objective: &F,
    optimizer: &mut O,
    initial_params: Array1<f64>,
    criteria: &ConvergenceCriteria,
) -> OptimizeResult
where
    O: Optimizer,
    F: ObjectiveFunction,
{
    let mut params = initial_params;
    let mut best_value = objective.evaluate(&params);
    let mut best_params = params.clone();

    for i in 0..criteria.max_iterations {
        let grad = objective.gradient(&params);
        let new_params = optimizer.step(&params, &grad);
        let value = objective.evaluate(&new_params);

        if value < best_value {
            best_value = value;
            best_params = new_params.clone();
        }

        // Check gradient norm convergence
        let grad_norm: f64 = grad.dot(&grad).sqrt();
        if grad_norm < criteria.tolerance {
            return OptimizeResult {
                best_params,
                best_value,
                iterations: i + 1,
                converged: true,
            };
        }

        params = new_params;
    }

    OptimizeResult {
        best_params,
        best_value,
        iterations: criteria.max_iterations,
        converged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ClosureObjective;
    use ndarray::array;

    #[test]
    fn test_sgd_quadratic() {
        // Minimize f(x) = x^2, minimum at x=0
        let obj = ClosureObjective {
            f: |x: &Array1<f64>| x[0].powi(2),
            dimensions: 1,
        };
        let mut sgd = SGD::new(0.1);
        let criteria = ConvergenceCriteria {
            max_iterations: 100,
            tolerance: 1e-6,
        };
        let result = minimize(&obj, &mut sgd, array![5.0], &criteria);
        assert!(result.best_value < 1e-6);
    }

    #[test]
    fn test_adam_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, min at (1,1)
        let obj = ClosureObjective {
            f: |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2),
            dimensions: 2,
        };
        let mut adam = Adam::new(0.01);
        let criteria = ConvergenceCriteria {
            max_iterations: 10000,
            tolerance: 1e-8,
        };
        let result = minimize(&obj, &mut adam, array![0.0, 0.0], &criteria);
        assert!(result.best_value < 0.1); // Adam may not converge perfectly on Rosenbrock
    }
}
