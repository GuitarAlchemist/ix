//! Differential Evolution (DE/rand/1/bin).

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::traits::EvolutionResult;

/// Differential Evolution configuration.
pub struct DifferentialEvolution {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_factor: f64, // F: typically 0.5-1.0
    pub crossover_prob: f64,  // CR: typically 0.7-0.9
    pub bounds: (f64, f64),
    pub seed: u64,
}

impl Default for DifferentialEvolution {
    fn default() -> Self {
        Self {
            population_size: 50,
            generations: 1000,
            mutation_factor: 0.8,
            crossover_prob: 0.9,
            bounds: (-10.0, 10.0),
            seed: 42,
        }
    }
}

impl DifferentialEvolution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_population_size(mut self, n: usize) -> Self {
        self.population_size = n;
        self
    }

    pub fn with_generations(mut self, n: usize) -> Self {
        self.generations = n;
        self
    }

    pub fn with_mutation_factor(mut self, f: f64) -> Self {
        self.mutation_factor = f;
        self
    }

    pub fn with_crossover_prob(mut self, cr: f64) -> Self {
        self.crossover_prob = cr;
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

    /// Run DE minimization.
    pub fn minimize<F>(&self, fitness_fn: &F, dim: usize) -> EvolutionResult
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let (lo, hi) = self.bounds;
        let np = self.population_size;

        // Initialize population
        let mut pop: Vec<Array1<f64>> = (0..np)
            .map(|_| Array1::from_iter((0..dim).map(|_| rng.random_range(lo..hi))))
            .collect();
        let mut fitness: Vec<f64> = pop.iter().map(fitness_fn).collect();

        let mut fitness_history = Vec::with_capacity(self.generations);

        for _gen in 0..self.generations {
            let best_f = fitness.iter().cloned().fold(f64::INFINITY, f64::min);
            fitness_history.push(best_f);

            for i in 0..np {
                // Select 3 distinct random indices != i
                let (r1, r2, r3) = pick_three(&mut rng, np, i);

                // Mutation: v = x_r1 + F * (x_r2 - x_r3)
                let mutant = &pop[r1] + &(self.mutation_factor * (&pop[r2] - &pop[r3]));

                // Crossover
                let j_rand = rng.random_range(0..dim);
                let trial = Array1::from_iter((0..dim).map(|j| {
                    if j == j_rand || rng.random::<f64>() < self.crossover_prob {
                        mutant[j].clamp(lo, hi)
                    } else {
                        pop[i][j]
                    }
                }));

                // Selection
                let trial_fitness = fitness_fn(&trial);
                if trial_fitness <= fitness[i] {
                    pop[i] = trial;
                    fitness[i] = trial_fitness;
                }
            }
        }

        // Find best
        let best_idx = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        EvolutionResult {
            best_genes: pop[best_idx].clone(),
            best_fitness: fitness[best_idx],
            generations: self.generations,
            fitness_history,
        }
    }
}

fn pick_three(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize, usize) {
    let mut r1 = rng.random_range(0..n);
    while r1 == exclude {
        r1 = rng.random_range(0..n);
    }
    let mut r2 = rng.random_range(0..n);
    while r2 == exclude || r2 == r1 {
        r2 = rng.random_range(0..n);
    }
    let mut r3 = rng.random_range(0..n);
    while r3 == exclude || r3 == r1 || r3 == r2 {
        r3 = rng.random_range(0..n);
    }
    (r1, r2, r3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_de_sphere() {
        let de = DifferentialEvolution::new()
            .with_population_size(30)
            .with_generations(500)
            .with_bounds(-5.0, 5.0)
            .with_seed(42);

        let result = de.minimize(&|x| x.mapv(|v| v * v).sum(), 3);
        assert!(
            result.best_fitness < 0.01,
            "DE should optimize sphere function well, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_de_rosenbrock() {
        let rosenbrock = |x: &Array1<f64>| {
            (0..x.len() - 1)
                .map(|i| 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2))
                .sum::<f64>()
        };

        let de = DifferentialEvolution::new()
            .with_population_size(60)
            .with_generations(2000)
            .with_bounds(-5.0, 5.0)
            .with_seed(42);

        let result = de.minimize(&rosenbrock, 2);
        assert!(
            result.best_fitness < 1.0,
            "DE should do reasonably on Rosenbrock, got {}",
            result.best_fitness
        );
    }
}
