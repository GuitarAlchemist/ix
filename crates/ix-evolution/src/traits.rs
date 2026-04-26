//! Core traits for evolutionary algorithms.

use ndarray::Array1;

/// An individual in the population.
pub trait Individual: Clone + Send {
    /// Fitness value (lower is better for minimization).
    fn fitness(&self) -> f64;

    /// Create offspring via crossover with another individual.
    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self;

    /// Mutate in place with given mutation rate.
    fn mutate(&mut self, rate: f64, rng: &mut impl rand::Rng);
}

/// A real-valued individual for continuous optimization.
#[derive(Debug, Clone)]
pub struct RealIndividual {
    pub genes: Array1<f64>,
    pub fitness_value: f64,
}

impl RealIndividual {
    pub fn new(genes: Array1<f64>) -> Self {
        Self {
            genes,
            fitness_value: f64::INFINITY,
        }
    }

    pub fn with_fitness(mut self, fitness: f64) -> Self {
        self.fitness_value = fitness;
        self
    }
}

impl Individual for RealIndividual {
    fn fitness(&self) -> f64 {
        self.fitness_value
    }

    fn crossover(&self, other: &Self, rng: &mut impl rand::Rng) -> Self {
        // BLX-alpha crossover
        let alpha = 0.5;
        let genes = Array1::from_iter(self.genes.iter().zip(other.genes.iter()).map(|(&a, &b)| {
            let lo = a.min(b) - alpha * (a - b).abs();
            let hi = a.max(b) + alpha * (a - b).abs();
            rng.random_range(lo..=hi)
        }));
        RealIndividual::new(genes)
    }

    fn mutate(&mut self, rate: f64, rng: &mut impl rand::Rng) {
        use rand_distr::{Distribution, Normal};
        let normal = Normal::new(0.0, rate).unwrap();
        for g in self.genes.iter_mut() {
            if rng.random::<f64>() < 0.3 {
                *g += normal.sample(rng);
            }
        }
    }
}

/// Result of an evolutionary optimization run.
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    pub best_genes: Array1<f64>,
    pub best_fitness: f64,
    pub generations: usize,
    pub fitness_history: Vec<f64>,
}
