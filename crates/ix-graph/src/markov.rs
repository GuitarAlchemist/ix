//! Markov chains: transition matrices, stationary distributions, simulation.

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

/// A discrete-time Markov chain defined by a transition matrix.
/// P[i][j] = probability of transitioning from state i to state j.
#[derive(Debug, Clone)]
pub struct MarkovChain {
    /// Transition matrix (row-stochastic: rows sum to 1).
    pub transition: Array2<f64>,
    /// State labels (optional).
    pub state_names: Vec<String>,
}

impl MarkovChain {
    /// Create from a transition matrix. Validates row-stochastic property.
    pub fn new(transition: Array2<f64>) -> Result<Self, String> {
        let (n, m) = transition.dim();
        if n != m {
            return Err(format!("Transition matrix must be square, got {}x{}", n, m));
        }
        for i in 0..n {
            let row_sum: f64 = transition.row(i).sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(format!("Row {} sums to {}, expected 1.0", i, row_sum));
            }
        }
        Ok(Self {
            transition,
            state_names: (0..n).map(|i| format!("S{}", i)).collect(),
        })
    }

    pub fn with_names(mut self, names: Vec<String>) -> Self {
        self.state_names = names;
        self
    }

    pub fn n_states(&self) -> usize {
        self.transition.nrows()
    }

    /// Probability of being in each state after `steps` transitions from `initial`.
    pub fn state_distribution(&self, initial: &Array1<f64>, steps: usize) -> Array1<f64> {
        let mut dist = initial.clone();
        for _ in 0..steps {
            dist = dist.dot(&self.transition);
        }
        dist
    }

    /// Estimate stationary distribution via power iteration.
    pub fn stationary_distribution(&self, max_iter: usize, tol: f64) -> Array1<f64> {
        let n = self.n_states();
        let mut dist = Array1::from_elem(n, 1.0 / n as f64);

        for _ in 0..max_iter {
            let new_dist = dist.dot(&self.transition);
            let diff = (&new_dist - &dist).mapv(f64::abs).sum();
            dist = new_dist;
            if diff < tol {
                break;
            }
        }
        dist
    }

    /// Simulate a random walk for `steps` from `start_state`.
    pub fn simulate(&self, start_state: usize, steps: usize, seed: u64) -> Vec<usize> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut state = start_state;
        let mut path = vec![state];

        for _ in 0..steps {
            let row = self.transition.row(state);
            let r: f64 = rng.random();
            let mut cumulative = 0.0;
            for (j, &p) in row.iter().enumerate() {
                cumulative += p;
                if r < cumulative {
                    state = j;
                    break;
                }
            }
            path.push(state);
        }
        path
    }

    /// Mean first passage time from state i to state j (estimated via simulation).
    pub fn mean_first_passage(
        &self,
        from: usize,
        to: usize,
        n_simulations: usize,
        max_steps: usize,
        seed: u64,
    ) -> f64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut total_steps = 0u64;
        let mut reached = 0u64;

        for _ in 0..n_simulations {
            let mut state = from;
            for step in 1..=max_steps {
                let row = self.transition.row(state);
                let r: f64 = rng.random();
                let mut cumulative = 0.0;
                for (j, &p) in row.iter().enumerate() {
                    cumulative += p;
                    if r < cumulative {
                        state = j;
                        break;
                    }
                }
                if state == to {
                    total_steps += step as u64;
                    reached += 1;
                    break;
                }
            }
        }

        if reached == 0 {
            f64::INFINITY
        } else {
            total_steps as f64 / reached as f64
        }
    }

    /// Check if the chain is ergodic (irreducible + aperiodic).
    /// Simple check: after many steps, all states have positive probability.
    pub fn is_ergodic(&self, steps: usize) -> bool {
        let _n = self.n_states();
        // Compute P^steps
        let mut power = self.transition.clone();
        for _ in 1..steps {
            power = power.dot(&self.transition);
        }
        // Check all entries are positive
        power.iter().all(|&v| v > 1e-10)
    }
}

/// Absorbing Markov chain analysis.
pub struct AbsorbingChain {
    pub chain: MarkovChain,
    pub absorbing_states: Vec<usize>,
}

impl AbsorbingChain {
    pub fn new(chain: MarkovChain) -> Self {
        let absorbing: Vec<usize> = (0..chain.n_states())
            .filter(|&i| chain.transition[[i, i]] == 1.0)
            .collect();
        Self {
            chain,
            absorbing_states: absorbing,
        }
    }

    pub fn is_absorbing_state(&self, state: usize) -> bool {
        self.absorbing_states.contains(&state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_markov_chain_creation() {
        let p = array![[0.7, 0.3], [0.4, 0.6]];
        let mc = MarkovChain::new(p).unwrap();
        assert_eq!(mc.n_states(), 2);
    }

    #[test]
    fn test_invalid_transition_matrix() {
        let p = array![[0.5, 0.3], [0.4, 0.6]]; // Row 0 doesn't sum to 1
        assert!(MarkovChain::new(p).is_err());
    }

    #[test]
    fn test_stationary_distribution() {
        // Simple 2-state chain
        let p = array![[0.7, 0.3], [0.4, 0.6]];
        let mc = MarkovChain::new(p).unwrap();
        let stationary = mc.stationary_distribution(1000, 1e-10);

        // Analytical: pi = [4/7, 3/7]
        assert!((stationary[0] - 4.0 / 7.0).abs() < 1e-6);
        assert!((stationary[1] - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_state_distribution() {
        let p = array![[0.0, 1.0], [1.0, 0.0]]; // Alternating
        let mc = MarkovChain::new(p).unwrap();
        let init = array![1.0, 0.0]; // Start in state 0

        let after_1 = mc.state_distribution(&init, 1);
        assert!((after_1[0] - 0.0).abs() < 1e-10);
        assert!((after_1[1] - 1.0).abs() < 1e-10);

        let after_2 = mc.state_distribution(&init, 2);
        assert!((after_2[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simulate() {
        let p = array![[0.0, 1.0], [1.0, 0.0]]; // Deterministic alternation
        let mc = MarkovChain::new(p).unwrap();
        let path = mc.simulate(0, 4, 42);
        assert_eq!(path, vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn test_ergodic() {
        let p = array![[0.7, 0.3], [0.4, 0.6]];
        let mc = MarkovChain::new(p).unwrap();
        assert!(mc.is_ergodic(100));
    }
}
