//! Simple test environments for RL agents.

use crate::traits::Environment;

/// GridWorld — a simple 2D grid environment.
///
/// The agent starts at `start` and tries to reach `goal`.
/// Actions: 0=Up, 1=Right, 2=Down, 3=Left
/// Reward: -1 per step, +10 on reaching goal.
pub struct GridWorld {
    pub rows: usize,
    pub cols: usize,
    pub start: (usize, usize),
    pub goal: (usize, usize),
    pub current: (usize, usize),
}

impl GridWorld {
    pub fn new(rows: usize, cols: usize, start: (usize, usize), goal: (usize, usize)) -> Self {
        Self {
            rows,
            cols,
            start,
            goal,
            current: start,
        }
    }

    /// Default 5x5 grid with start at (0,0) and goal at (4,4).
    pub fn default_5x5() -> Self {
        Self::new(5, 5, (0, 0), (4, 4))
    }

    /// Convert (row, col) to a flat state index.
    pub fn state_index(&self, state: &(usize, usize)) -> usize {
        state.0 * self.cols + state.1
    }

    /// Total number of states.
    pub fn num_states(&self) -> usize {
        self.rows * self.cols
    }

    /// Number of actions (always 4: Up, Right, Down, Left).
    pub fn num_actions(&self) -> usize {
        4
    }
}

impl Environment for GridWorld {
    type State = (usize, usize);
    type Action = usize;

    fn reset(&mut self) -> Self::State {
        self.current = self.start;
        self.current
    }

    fn step(&mut self, action: &Self::Action) -> (Self::State, f64, bool) {
        let (r, c) = self.current;
        self.current = match action {
            0 => (r.saturating_sub(1), c),           // Up
            1 => (r, (c + 1).min(self.cols - 1)),    // Right
            2 => ((r + 1).min(self.rows - 1), c),    // Down
            3 => (r, c.saturating_sub(1)),            // Left
            _ => (r, c),
        };

        let done = self.current == self.goal;
        let reward = if done { 10.0 } else { -1.0 };
        (self.current, reward, done)
    }

    fn actions(&self) -> Vec<Self::Action> {
        vec![0, 1, 2, 3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gridworld_reset() {
        let mut env = GridWorld::default_5x5();
        let state = env.reset();
        assert_eq!(state, (0, 0));
    }

    #[test]
    fn test_gridworld_step_right() {
        let mut env = GridWorld::default_5x5();
        env.reset();
        let (state, reward, done) = env.step(&1); // Right
        assert_eq!(state, (0, 1));
        assert_eq!(reward, -1.0);
        assert!(!done);
    }

    #[test]
    fn test_gridworld_boundary() {
        let mut env = GridWorld::default_5x5();
        env.reset();
        let (state, _, _) = env.step(&0); // Up from (0,0) stays at (0,0)
        assert_eq!(state, (0, 0));
    }

    #[test]
    fn test_gridworld_reach_goal() {
        let mut env = GridWorld::new(2, 2, (0, 0), (1, 1));
        env.reset();
        env.step(&1); // (0,1)
        let (state, reward, done) = env.step(&2); // (1,1) = goal
        assert_eq!(state, (1, 1));
        assert_eq!(reward, 10.0);
        assert!(done);
    }

    #[test]
    fn test_gridworld_state_index() {
        let env = GridWorld::default_5x5();
        assert_eq!(env.state_index(&(0, 0)), 0);
        assert_eq!(env.state_index(&(2, 3)), 13);
        assert_eq!(env.state_index(&(4, 4)), 24);
        assert_eq!(env.num_states(), 25);
    }
}
