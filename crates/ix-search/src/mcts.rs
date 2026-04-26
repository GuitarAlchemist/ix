//! Monte Carlo Tree Search (MCTS).
//!
//! Uses random rollouts to evaluate game positions.
//! UCB1 selection, random default policy.

use rand::prelude::*;

/// A state for MCTS.
pub trait MctsState: Clone {
    type Action: Clone;

    fn legal_actions(&self) -> Vec<Self::Action>;
    fn apply(&self, action: &Self::Action) -> Self;
    fn is_terminal(&self) -> bool;

    /// Evaluate terminal state: 1.0 = win, 0.0 = loss, 0.5 = draw.
    fn reward(&self) -> f64;
}

/// MCTS tree node.
struct MctsNode<S: MctsState> {
    state: S,
    action: Option<S::Action>,
    visits: u64,
    total_reward: f64,
    children: Vec<usize>, // Indices into node pool
    parent: Option<usize>,
    untried_actions: Vec<S::Action>,
}

/// Run MCTS and return the best action.
pub fn mcts_search<S: MctsState>(
    root: &S,
    iterations: usize,
    exploration: f64,
    seed: u64,
) -> Option<S::Action> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut nodes: Vec<MctsNode<S>> = Vec::new();

    // Create root node
    let actions = root.legal_actions();
    if actions.is_empty() {
        return None;
    }

    nodes.push(MctsNode {
        state: root.clone(),
        action: None,
        visits: 0,
        total_reward: 0.0,
        children: Vec::new(),
        parent: None,
        untried_actions: actions,
    });

    for _ in 0..iterations {
        // Selection: walk tree using UCB1
        let mut node_idx = 0;
        while nodes[node_idx].untried_actions.is_empty()
            && !nodes[node_idx].children.is_empty()
            && !nodes[node_idx].state.is_terminal()
        {
            node_idx = select_child(&nodes, node_idx, exploration);
        }

        // Expansion: add a child for an untried action
        if !nodes[node_idx].untried_actions.is_empty() && !nodes[node_idx].state.is_terminal() {
            let action_idx = rng.random_range(0..nodes[node_idx].untried_actions.len());
            let action = nodes[node_idx].untried_actions.swap_remove(action_idx);
            let new_state = nodes[node_idx].state.apply(&action);
            let new_actions = new_state.legal_actions();

            let new_idx = nodes.len();
            nodes[node_idx].children.push(new_idx);

            nodes.push(MctsNode {
                state: new_state,
                action: Some(action),
                visits: 0,
                total_reward: 0.0,
                children: Vec::new(),
                parent: Some(node_idx),
                untried_actions: new_actions,
            });

            node_idx = new_idx;
        }

        // Simulation: random rollout
        let reward = rollout(&nodes[node_idx].state, &mut rng);

        // Backpropagation
        let mut idx = Some(node_idx);
        while let Some(i) = idx {
            nodes[i].visits += 1;
            nodes[i].total_reward += reward;
            idx = nodes[i].parent;
        }
    }

    // Select most visited child of root
    nodes[0]
        .children
        .iter()
        .max_by_key(|&&c| nodes[c].visits)
        .and_then(|&c| nodes[c].action.clone())
}

/// UCB1 child selection.
fn select_child<S: MctsState>(nodes: &[MctsNode<S>], parent: usize, c: f64) -> usize {
    let parent_visits = nodes[parent].visits as f64;

    *nodes[parent]
        .children
        .iter()
        .max_by(|&&a, &&b| {
            let ucb_a = ucb1(nodes[a].total_reward, nodes[a].visits, parent_visits, c);
            let ucb_b = ucb1(nodes[b].total_reward, nodes[b].visits, parent_visits, c);
            ucb_a.partial_cmp(&ucb_b).unwrap()
        })
        .unwrap()
}

fn ucb1(total_reward: f64, visits: u64, parent_visits: f64, c: f64) -> f64 {
    if visits == 0 {
        return f64::INFINITY;
    }
    let v = visits as f64;
    total_reward / v + c * (parent_visits.ln() / v).sqrt()
}

/// Grammar-weighted UCB1: multiplies the exploration bonus by a rule weight.
///
/// This variant is used by grammar-guided MCTS where `weight` encodes the
/// Bayesian confidence in a production rule (range 0–1, higher is preferred).
///
/// ```
/// use ix_search::mcts::weighted_ucb1;
/// let score = weighted_ucb1(5.0, 10, 100.0, 1.41, 0.8);
/// assert!(score > 0.0);
/// ```
pub fn weighted_ucb1(
    total_reward: f64,
    visits: u64,
    parent_visits: f64,
    c: f64,
    weight: f64,
) -> f64 {
    if visits == 0 {
        return f64::INFINITY;
    }
    let v = visits as f64;
    total_reward / v + c * weight * (parent_visits.ln() / v).sqrt()
}

/// Random rollout from a state.
fn rollout<S: MctsState>(state: &S, rng: &mut impl Rng) -> f64 {
    let mut current = state.clone();
    let mut depth = 0;

    while !current.is_terminal() && depth < 500 {
        let actions = current.legal_actions();
        if actions.is_empty() {
            break;
        }
        let idx = rng.random_range(0..actions.len());
        current = current.apply(&actions[idx]);
        depth += 1;
    }

    current.reward()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple "get to 21" game — Nim-like.
    #[derive(Clone, Debug)]
    struct NimState {
        count: i32,
        my_turn: bool,
    }

    impl MctsState for NimState {
        type Action = i32;

        fn legal_actions(&self) -> Vec<i32> {
            if self.is_terminal() {
                return vec![];
            }
            let max = 3.min(21 - self.count);
            (1..=max).collect()
        }

        fn apply(&self, action: &i32) -> Self {
            NimState {
                count: self.count + action,
                my_turn: !self.my_turn,
            }
        }

        fn is_terminal(&self) -> bool {
            self.count >= 21
        }

        fn reward(&self) -> f64 {
            // Player who reaches 21 wins
            if self.count >= 21 {
                if self.my_turn {
                    0.0
                } else {
                    1.0
                } // Previous player won
            } else {
                0.5
            }
        }
    }

    #[test]
    fn test_mcts_finds_move() {
        let state = NimState {
            count: 0,
            my_turn: true,
        };
        let action = mcts_search(&state, 1000, 1.41, 42);
        assert!(action.is_some());
        let a = action.unwrap();
        assert!((1..=3).contains(&a));
    }
}
