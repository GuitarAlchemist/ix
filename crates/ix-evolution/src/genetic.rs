//! Genetic Algorithm (GA) for continuous optimization.

use ndarray::Array1;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::selection;
use crate::traits::{EvolutionResult, Individual, RealIndividual};

/// Genetic Algorithm configuration.
pub struct GeneticAlgorithm {
    pub population_size: usize,
    pub generations: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub tournament_size: usize,
    pub elitism: usize,
    pub bounds: (f64, f64),
    pub seed: u64,
}

impl Default for GeneticAlgorithm {
    fn default() -> Self {
        Self {
            population_size: 100,
            generations: 500,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            tournament_size: 3,
            elitism: 2,
            bounds: (-10.0, 10.0),
            seed: 42,
        }
    }
}

impl GeneticAlgorithm {
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

    pub fn with_mutation_rate(mut self, r: f64) -> Self {
        self.mutation_rate = r;
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

    /// Run the GA. `fitness_fn` evaluates a candidate solution (lower is better).
    pub fn minimize<F>(&self, fitness_fn: &F, dim: usize) -> EvolutionResult
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let (lo, hi) = self.bounds;

        // Initialize population
        let mut population: Vec<RealIndividual> = (0..self.population_size)
            .map(|_| {
                use rand::Rng;
                let genes = Array1::from_iter((0..dim).map(|_| rng.random_range(lo..hi)));
                let f = fitness_fn(&genes);
                RealIndividual::new(genes).with_fitness(f)
            })
            .collect();

        let mut fitness_history = Vec::with_capacity(self.generations);

        for _gen in 0..self.generations {
            // Sort by fitness (ascending = best first)
            population.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

            let best_fitness = population[0].fitness();
            fitness_history.push(best_fitness);

            // Build next generation
            let mut next_gen: Vec<RealIndividual> = Vec::with_capacity(self.population_size);

            // Elitism: keep top individuals
            for ind in population
                .iter()
                .take(self.elitism.min(self.population_size))
            {
                next_gen.push(ind.clone());
            }

            // Fill rest with offspring
            while next_gen.len() < self.population_size {
                let parent1 = selection::tournament(&population, self.tournament_size, &mut rng);
                let parent2 = selection::tournament(&population, self.tournament_size, &mut rng);

                use rand::Rng;
                let mut child = if rng.random::<f64>() < self.crossover_rate {
                    parent1.crossover(&parent2, &mut rng)
                } else {
                    parent1.clone()
                };

                child.mutate(self.mutation_rate, &mut rng);

                // Clamp to bounds
                child.genes.mapv_inplace(|v| v.clamp(lo, hi));

                // Evaluate fitness
                child.fitness_value = fitness_fn(&child.genes);

                next_gen.push(child);
            }

            population = next_gen;
        }

        // Final sort
        population.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

        EvolutionResult {
            best_genes: population[0].genes.clone(),
            best_fitness: population[0].fitness(),
            generations: self.generations,
            fitness_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ga_sphere() {
        // Minimize f(x) = sum(x_i^2)
        let ga = GeneticAlgorithm::new()
            .with_population_size(50)
            .with_generations(200)
            .with_mutation_rate(0.2)
            .with_bounds(-5.0, 5.0)
            .with_seed(42);

        let result = ga.minimize(&|x| x.mapv(|v| v * v).sum(), 3);
        assert!(
            result.best_fitness < 1.0,
            "GA should optimize sphere function, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_ga_rastrigin() {
        // Rastrigin: f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
        let rastrigin = |x: &Array1<f64>| {
            let n = x.len() as f64;
            10.0 * n
                + x.iter()
                    .map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
                    .sum::<f64>()
        };

        let ga = GeneticAlgorithm::new()
            .with_population_size(100)
            .with_generations(500)
            .with_mutation_rate(0.15)
            .with_bounds(-5.12, 5.12)
            .with_seed(123);

        let result = ga.minimize(&rastrigin, 2);
        assert!(
            result.best_fitness < 5.0,
            "GA should do reasonably on Rastrigin, got {}",
            result.best_fitness
        );
    }

    #[test]
    fn test_fitness_history_decreasing() {
        let ga = GeneticAlgorithm::new()
            .with_population_size(30)
            .with_generations(50)
            .with_seed(42);

        let result = ga.minimize(&|x| x.mapv(|v| v * v).sum(), 2);

        // First fitness should be >= last fitness (improving over time)
        assert!(result.fitness_history.first().unwrap() >= result.fitness_history.last().unwrap());
    }
}
