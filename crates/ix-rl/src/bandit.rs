//! Multi-armed bandit algorithms.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Epsilon-greedy bandit.
pub struct EpsilonGreedy {
    pub epsilon: f64,
    pub q_values: Vec<f64>,
    pub counts: Vec<usize>,
    rng: StdRng,
}

impl EpsilonGreedy {
    pub fn new(n_arms: usize, epsilon: f64, seed: u64) -> Self {
        Self {
            epsilon,
            q_values: vec![0.0; n_arms],
            counts: vec![0; n_arms],
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn select_arm(&mut self) -> usize {
        if self.rng.random::<f64>() < self.epsilon {
            self.rng.random_range(0..self.q_values.len())
        } else {
            self.q_values
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    pub fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1;
        let n = self.counts[arm] as f64;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;
    }
}

/// Upper Confidence Bound (UCB1).
pub struct UCB1 {
    pub q_values: Vec<f64>,
    pub counts: Vec<usize>,
    pub total_count: usize,
}

impl UCB1 {
    pub fn new(n_arms: usize) -> Self {
        Self {
            q_values: vec![0.0; n_arms],
            counts: vec![0; n_arms],
            total_count: 0,
        }
    }

    pub fn select_arm(&self) -> usize {
        // Play each arm at least once
        for (i, &c) in self.counts.iter().enumerate() {
            if c == 0 {
                return i;
            }
        }

        let total = self.total_count as f64;
        self.q_values
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let bonus = (2.0 * total.ln() / self.counts[i] as f64).sqrt();
                (i, q + bonus)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    pub fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1;
        self.total_count += 1;
        let n = self.counts[arm] as f64;
        self.q_values[arm] += (reward - self.q_values[arm]) / n;
    }
}

/// Thompson Sampling (Gaussian).
pub struct ThompsonSampling {
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
    pub counts: Vec<usize>,
    rng: StdRng,
}

impl ThompsonSampling {
    pub fn new(n_arms: usize, seed: u64) -> Self {
        Self {
            means: vec![0.0; n_arms],
            variances: vec![1.0; n_arms],
            counts: vec![0; n_arms],
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn select_arm(&mut self) -> usize {
        let samples: Vec<f64> = (0..self.means.len())
            .map(|i| {
                let dist = Normal::new(self.means[i], self.variances[i].sqrt().max(0.01)).unwrap();
                dist.sample(&mut self.rng)
            })
            .collect();

        samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    pub fn update(&mut self, arm: usize, reward: f64) {
        self.counts[arm] += 1;
        let n = self.counts[arm] as f64;
        self.means[arm] += (reward - self.means[arm]) / n;
        // Simplified variance update
        if n > 1.0 {
            self.variances[arm] = 1.0 / n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsilon_greedy_converges() {
        let mut bandit = EpsilonGreedy::new(3, 0.1, 42);
        let true_means = [1.0, 2.0, 3.0]; // Arm 2 is best
        let mut rng = StdRng::seed_from_u64(123);

        for _ in 0..1000 {
            let arm = bandit.select_arm();
            let reward = true_means[arm] + Normal::new(0.0, 0.5).unwrap().sample(&mut rng);
            bandit.update(arm, reward);
        }

        // Best arm should have highest Q-value
        let best = bandit
            .q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(best, 2);
    }

    #[test]
    fn test_ucb1_explores_all() {
        let mut ucb = UCB1::new(5);
        // First 5 selections should be 0,1,2,3,4
        for i in 0..5 {
            assert_eq!(ucb.select_arm(), i);
            ucb.update(i, 1.0);
        }
    }
}
