//! Simulated Annealing optimization.

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use crate::traits::{ObjectiveFunction, OptimizeResult};

/// Temperature schedule for simulated annealing.
#[derive(Debug, Clone)]
pub enum CoolingSchedule {
    /// T(k) = T0 * alpha^k
    Exponential { alpha: f64 },
    /// T(k) = T0 / (1 + alpha * k)
    Linear { alpha: f64 },
    /// T(k) = T0 / ln(1 + k)
    Logarithmic,
}

/// Simulated Annealing configuration.
pub struct SimulatedAnnealing {
    pub initial_temp: f64,
    pub min_temp: f64,
    pub max_iterations: usize,
    pub cooling: CoolingSchedule,
    pub step_size: f64,
    pub seed: u64,
}

impl Default for SimulatedAnnealing {
    fn default() -> Self {
        Self {
            initial_temp: 100.0,
            min_temp: 1e-8,
            max_iterations: 10000,
            cooling: CoolingSchedule::Exponential { alpha: 0.995 },
            step_size: 1.0,
            seed: 42,
        }
    }
}

impl SimulatedAnnealing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_temp(mut self, initial: f64, min: f64) -> Self {
        self.initial_temp = initial;
        self.min_temp = min;
        self
    }

    pub fn with_cooling(mut self, schedule: CoolingSchedule) -> Self {
        self.cooling = schedule;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_step_size(mut self, s: f64) -> Self {
        self.step_size = s;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    fn temperature(&self, iteration: usize) -> f64 {
        let k = iteration as f64;
        match &self.cooling {
            CoolingSchedule::Exponential { alpha } => self.initial_temp * alpha.powf(k),
            CoolingSchedule::Linear { alpha } => self.initial_temp / (1.0 + alpha * k),
            CoolingSchedule::Logarithmic => {
                if iteration == 0 {
                    self.initial_temp
                } else {
                    self.initial_temp / (1.0 + k).ln()
                }
            }
        }
    }

    /// Run simulated annealing on the given objective function.
    pub fn minimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
        initial: Array1<f64>,
    ) -> OptimizeResult {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let normal = Normal::new(0.0, self.step_size).unwrap();

        let mut current = initial;
        let mut current_value = objective.evaluate(&current);
        let mut best = current.clone();
        let mut best_value = current_value;

        for i in 0..self.max_iterations {
            let temp = self.temperature(i);
            if temp < self.min_temp {
                return OptimizeResult {
                    best_params: best,
                    best_value,
                    iterations: i,
                    converged: true,
                };
            }

            // Generate neighbor
            let perturbation: Array1<f64> =
                Array1::from_iter((0..current.len()).map(|_| normal.sample(&mut rng)));
            let neighbor = &current + &perturbation;
            let neighbor_value = objective.evaluate(&neighbor);

            // Acceptance criterion
            let delta = neighbor_value - current_value;
            let accept = if delta < 0.0 {
                true
            } else {
                let prob = (-delta / temp).exp();
                rng.random::<f64>() < prob
            };

            if accept {
                current = neighbor;
                current_value = neighbor_value;

                if current_value < best_value {
                    best = current.clone();
                    best_value = current_value;
                }
            }
        }

        OptimizeResult {
            best_params: best,
            best_value,
            iterations: self.max_iterations,
            converged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ClosureObjective;
    use ndarray::array;

    #[test]
    fn test_annealing_quadratic() {
        let obj = ClosureObjective {
            f: |x: &Array1<f64>| x[0].powi(2) + x[1].powi(2),
            dimensions: 2,
        };
        let sa = SimulatedAnnealing::new()
            .with_temp(100.0, 1e-10)
            .with_max_iterations(50000)
            .with_step_size(0.5)
            .with_seed(42);

        let result = sa.minimize(&obj, array![10.0, 10.0]);
        assert!(
            result.best_value < 1.0,
            "SA should find near-zero for quadratic, got {}",
            result.best_value
        );
    }

    #[test]
    fn test_cooling_schedules() {
        let sa = SimulatedAnnealing::new().with_temp(100.0, 0.0);

        // Exponential: should decrease
        let t0 = sa.temperature(0);
        let t100 = sa.temperature(100);
        assert!(t100 < t0);

        // Linear
        let sa_lin = SimulatedAnnealing::new()
            .with_temp(100.0, 0.0)
            .with_cooling(CoolingSchedule::Linear { alpha: 0.1 });
        assert!(sa_lin.temperature(100) < sa_lin.temperature(0));
    }
}
