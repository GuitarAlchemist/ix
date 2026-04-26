//! Particle Swarm Optimization (PSO).

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::traits::{ObjectiveFunction, OptimizeResult};

/// PSO configuration.
pub struct ParticleSwarm {
    pub num_particles: usize,
    pub max_iterations: usize,
    pub inertia: f64,
    pub cognitive: f64, // c1: pull toward personal best
    pub social: f64,    // c2: pull toward global best
    pub bounds: (f64, f64),
    pub seed: u64,
}

impl Default for ParticleSwarm {
    fn default() -> Self {
        Self {
            num_particles: 30,
            max_iterations: 1000,
            inertia: 0.7,
            cognitive: 1.5,
            social: 1.5,
            bounds: (-10.0, 10.0),
            seed: 42,
        }
    }
}

impl ParticleSwarm {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_particles(mut self, n: usize) -> Self {
        self.num_particles = n;
        self
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_bounds(mut self, lo: f64, hi: f64) -> Self {
        self.bounds = (lo, hi);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Run PSO minimization.
    pub fn minimize<F: ObjectiveFunction>(&self, objective: &F) -> OptimizeResult {
        let dim = objective.dim();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let (lo, hi) = self.bounds;

        // Initialize particles
        let mut positions: Vec<Array1<f64>> = (0..self.num_particles)
            .map(|_| Array1::from_iter((0..dim).map(|_| rng.random_range(lo..hi))))
            .collect();

        let mut velocities: Vec<Array1<f64>> = (0..self.num_particles)
            .map(|_| {
                let range = (hi - lo) * 0.1;
                Array1::from_iter((0..dim).map(|_| rng.random_range(-range..range)))
            })
            .collect();

        let mut personal_best: Vec<Array1<f64>> = positions.clone();
        let mut personal_best_values: Vec<f64> =
            positions.iter().map(|p| objective.evaluate(p)).collect();

        let mut global_best_idx = personal_best_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let mut global_best = personal_best[global_best_idx].clone();
        let mut global_best_value = personal_best_values[global_best_idx];

        for iter in 0..self.max_iterations {
            for i in 0..self.num_particles {
                // Update velocity
                for d in 0..dim {
                    let r1: f64 = rng.random();
                    let r2: f64 = rng.random();
                    velocities[i][d] = self.inertia * velocities[i][d]
                        + self.cognitive * r1 * (personal_best[i][d] - positions[i][d])
                        + self.social * r2 * (global_best[d] - positions[i][d]);
                }

                // Update position
                positions[i] = &positions[i] + &velocities[i];

                // Clamp to bounds
                positions[i].mapv_inplace(|v| v.clamp(lo, hi));

                // Evaluate
                let value = objective.evaluate(&positions[i]);

                if value < personal_best_values[i] {
                    personal_best_values[i] = value;
                    personal_best[i] = positions[i].clone();

                    if value < global_best_value {
                        global_best_value = value;
                        global_best = positions[i].clone();
                        global_best_idx = i;
                    }
                }
            }

            // Early termination
            if global_best_value < 1e-12 {
                return OptimizeResult {
                    best_params: global_best,
                    best_value: global_best_value,
                    iterations: iter + 1,
                    converged: true,
                };
            }
        }

        let _ = global_best_idx;

        OptimizeResult {
            best_params: global_best,
            best_value: global_best_value,
            iterations: self.max_iterations,
            converged: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::ClosureObjective;

    #[test]
    fn test_pso_sphere() {
        // f(x) = sum(x_i^2), min at origin
        let obj = ClosureObjective {
            f: |x: &Array1<f64>| x.mapv(|v| v * v).sum(),
            dimensions: 3,
        };
        let pso = ParticleSwarm::new()
            .with_particles(40)
            .with_max_iterations(500)
            .with_bounds(-10.0, 10.0)
            .with_seed(42);

        let result = pso.minimize(&obj);
        assert!(
            result.best_value < 0.01,
            "PSO should find near-zero, got {}",
            result.best_value
        );
    }
}
