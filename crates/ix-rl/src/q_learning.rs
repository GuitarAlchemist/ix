//! Tabular Q-Learning and SARSA.

use ndarray::Array2;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::traits::{Environment, Agent};

/// Tabular Q-Learning agent.
pub struct QLearning {
    /// Q-table: (num_states, num_actions)
    pub q_table: Array2<f64>,
    pub learning_rate: f64,
    pub discount: f64,
    pub epsilon: f64,
    rng: StdRng,
}

impl QLearning {
    pub fn new(num_states: usize, num_actions: usize, seed: u64) -> Self {
        Self {
            q_table: Array2::zeros((num_states, num_actions)),
            learning_rate: 0.1,
            discount: 0.99,
            epsilon: 0.1,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_discount(mut self, gamma: f64) -> Self {
        self.discount = gamma;
        self
    }

    pub fn with_epsilon(mut self, eps: f64) -> Self {
        self.epsilon = eps;
        self
    }

    /// Epsilon-greedy action selection given a flat state index.
    pub fn select_action_index(&mut self, state_idx: usize) -> usize {
        let num_actions = self.q_table.ncols();
        if self.rng.random::<f64>() < self.epsilon {
            self.rng.random_range(0..num_actions)
        } else {
            let row = self.q_table.row(state_idx);
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    /// Q-learning update: Q(s,a) += lr * (r + gamma * max_a' Q(s',a') - Q(s,a))
    pub fn update_index(
        &mut self,
        state_idx: usize,
        action: usize,
        reward: f64,
        next_state_idx: usize,
        done: bool,
    ) {
        let max_next_q = if done {
            0.0
        } else {
            self.q_table
                .row(next_state_idx)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let td_target = reward + self.discount * max_next_q;
        let td_error = td_target - self.q_table[[state_idx, action]];
        self.q_table[[state_idx, action]] += self.learning_rate * td_error;
    }

    /// Train on a GridWorld-like environment for `episodes` episodes.
    /// Returns total reward per episode.
    pub fn train_gridworld(
        &mut self,
        env: &mut crate::env::GridWorld,
        episodes: usize,
        max_steps: usize,
    ) -> Vec<f64> {
        let mut rewards = Vec::with_capacity(episodes);

        for _ in 0..episodes {
            let state = env.reset();
            let mut state_idx = env.state_index(&state);
            let mut total_reward = 0.0;

            for _ in 0..max_steps {
                let action = self.select_action_index(state_idx);
                let (next_state, reward, done) = env.step(&action);
                let next_idx = env.state_index(&next_state);

                self.update_index(state_idx, action, reward, next_idx, done);
                total_reward += reward;
                state_idx = next_idx;

                if done {
                    break;
                }
            }

            rewards.push(total_reward);
        }

        rewards
    }
}

/// Implement the generic Agent trait for GridWorld.
impl Agent<crate::env::GridWorld> for QLearning {
    fn select_action(&self, state: &(usize, usize)) -> usize {
        let state_idx = state.0 * self.q_table.ncols().max(1) + state.1;
        // Greedy (no exploration in trait method)
        let row = self.q_table.row(state_idx.min(self.q_table.nrows() - 1));
        row.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }

    fn update(
        &mut self,
        _state: &(usize, usize),
        _action: &usize,
        _reward: f64,
        _next_state: &(usize, usize),
        _done: bool,
    ) {
        // Use update_index directly for GridWorld training
    }
}

/// Tabular SARSA agent (on-policy).
pub struct Sarsa {
    pub q_table: Array2<f64>,
    pub learning_rate: f64,
    pub discount: f64,
    pub epsilon: f64,
    rng: StdRng,
}

impl Sarsa {
    pub fn new(num_states: usize, num_actions: usize, seed: u64) -> Self {
        Self {
            q_table: Array2::zeros((num_states, num_actions)),
            learning_rate: 0.1,
            discount: 0.99,
            epsilon: 0.1,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn select_action_index(&mut self, state_idx: usize) -> usize {
        let num_actions = self.q_table.ncols();
        if self.rng.random::<f64>() < self.epsilon {
            self.rng.random_range(0..num_actions)
        } else {
            let row = self.q_table.row(state_idx);
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0
        }
    }

    /// SARSA update: Q(s,a) += lr * (r + gamma * Q(s',a') - Q(s,a))
    pub fn update_index(
        &mut self,
        state_idx: usize,
        action: usize,
        reward: f64,
        next_state_idx: usize,
        next_action: usize,
        done: bool,
    ) {
        let next_q = if done {
            0.0
        } else {
            self.q_table[[next_state_idx, next_action]]
        };
        let td_target = reward + self.discount * next_q;
        let td_error = td_target - self.q_table[[state_idx, action]];
        self.q_table[[state_idx, action]] += self.learning_rate * td_error;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::GridWorld;

    #[test]
    fn test_qlearning_learns_gridworld() {
        let mut env = GridWorld::new(3, 3, (0, 0), (2, 2));
        let mut agent = QLearning::new(9, 4, 42)
            .with_learning_rate(0.1)
            .with_discount(0.95)
            .with_epsilon(0.2);

        let rewards = agent.train_gridworld(&mut env, 500, 100);

        // Average reward should improve
        let first_100: f64 = rewards[..100].iter().sum::<f64>() / 100.0;
        let last_100: f64 = rewards[400..].iter().sum::<f64>() / 100.0;
        assert!(
            last_100 > first_100,
            "agent should improve: first_100={}, last_100={}",
            first_100,
            last_100
        );
    }

    #[test]
    fn test_qlearning_q_table_shape() {
        let agent = QLearning::new(25, 4, 42);
        assert_eq!(agent.q_table.dim(), (25, 4));
    }

    #[test]
    fn test_sarsa_update() {
        let mut agent = Sarsa::new(9, 4, 42);
        agent.update_index(0, 1, 1.0, 1, 2, false);
        assert!(agent.q_table[[0, 1]] > 0.0);
    }
}
