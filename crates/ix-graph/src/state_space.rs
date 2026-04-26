//! State space exploration and search algorithms.
//!
//! Used for agent decision-making: explore a space of possible actions/states
//! to find optimal paths (e.g., which skill to invoke, in what order).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::Hash;

/// A state in a state space.
pub trait State: Clone + Eq + Hash {
    type Action: Clone;

    /// Available actions from this state.
    fn actions(&self) -> Vec<Self::Action>;

    /// Apply action, return (new_state, cost).
    fn apply(&self, action: &Self::Action) -> (Self, f64);

    /// Is this a goal state?
    fn is_goal(&self) -> bool;

    /// Heuristic estimate of cost to goal (for A*). Default = 0.
    fn heuristic(&self) -> f64 {
        0.0
    }
}

/// Result of a state space search.
#[derive(Debug, Clone)]
pub struct SearchResult<S> {
    pub path: Vec<S>,
    pub total_cost: f64,
    pub nodes_expanded: usize,
}

/// A* search over a state space.
pub fn astar<S: State>(initial: S) -> Option<SearchResult<S>> {
    let mut open = BinaryHeap::new();
    let mut g_scores: HashMap<S, f64> = HashMap::new();
    let mut came_from: HashMap<S, S> = HashMap::new();
    let mut expanded = 0usize;

    g_scores.insert(initial.clone(), 0.0);
    open.push(AStarNode {
        state: initial.clone(),
        f_score: initial.heuristic(),
        g_score: 0.0,
    });

    while let Some(AStarNode { state, g_score, .. }) = open.pop() {
        if state.is_goal() {
            // Reconstruct path
            let mut path = vec![state.clone()];
            let mut current = state;
            while let Some(prev) = came_from.get(&current) {
                path.push(prev.clone());
                current = prev.clone();
            }
            path.reverse();
            return Some(SearchResult {
                total_cost: g_score,
                path,
                nodes_expanded: expanded,
            });
        }

        if g_score > *g_scores.get(&state).unwrap_or(&f64::INFINITY) {
            continue;
        }

        expanded += 1;

        for action in state.actions() {
            let (next_state, cost) = state.apply(&action);
            let tentative_g = g_score + cost;

            if tentative_g < *g_scores.get(&next_state).unwrap_or(&f64::INFINITY) {
                g_scores.insert(next_state.clone(), tentative_g);
                came_from.insert(next_state.clone(), state.clone());
                open.push(AStarNode {
                    f_score: tentative_g + next_state.heuristic(),
                    g_score: tentative_g,
                    state: next_state,
                });
            }
        }
    }

    None
}

/// Beam search: keeps only top-k states at each level.
/// Useful for agent routing where you want fast approximate solutions.
pub fn beam_search<S: State>(
    initial: S,
    beam_width: usize,
    max_depth: usize,
) -> Option<SearchResult<S>> {
    let mut beam: Vec<(S, f64, Vec<S>)> = vec![(initial.clone(), 0.0, vec![initial])];
    let mut expanded = 0usize;

    for _ in 0..max_depth {
        let mut candidates: Vec<(S, f64, Vec<S>)> = Vec::new();

        for (state, cost, path) in &beam {
            if state.is_goal() {
                return Some(SearchResult {
                    path: path.clone(),
                    total_cost: *cost,
                    nodes_expanded: expanded,
                });
            }

            expanded += 1;
            for action in state.actions() {
                let (next, step_cost) = state.apply(&action);
                let new_cost = cost + step_cost;
                let mut new_path = path.clone();
                new_path.push(next.clone());
                candidates.push((next, new_cost, new_path));
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Keep top beam_width by f-score (g + h)
        candidates.sort_by(|(s1, c1, _), (s2, c2, _)| {
            let f1 = c1 + s1.heuristic();
            let f2 = c2 + s2.heuristic();
            f1.partial_cmp(&f2).unwrap_or(Ordering::Equal)
        });
        candidates.truncate(beam_width);
        beam = candidates;
    }

    // Return best found even if not goal
    beam.into_iter()
        .filter(|(s, _, _)| s.is_goal())
        .min_by(|(_, c1, _), (_, c2, _)| c1.partial_cmp(c2).unwrap_or(Ordering::Equal))
        .map(|(_, cost, path)| SearchResult {
            path,
            total_cost: cost,
            nodes_expanded: expanded,
        })
}

struct AStarNode<S> {
    state: S,
    f_score: f64,
    g_score: f64,
}

impl<S: Eq> Eq for AStarNode<S> {}

impl<S: Eq> PartialEq for AStarNode<S> {
    fn eq(&self, other: &Self) -> bool {
        self.state == other.state
    }
}

impl<S: Eq> Ord for AStarNode<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl<S: Eq> PartialOrd for AStarNode<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple grid state for testing.
    #[derive(Clone, Debug, Eq, PartialEq, Hash)]
    struct GridPos {
        x: i32,
        y: i32,
        goal_x: i32,
        goal_y: i32,
    }

    impl State for GridPos {
        type Action = (i32, i32);

        fn actions(&self) -> Vec<Self::Action> {
            vec![(0, 1), (0, -1), (1, 0), (-1, 0)]
        }

        fn apply(&self, action: &Self::Action) -> (Self, f64) {
            let new = GridPos {
                x: self.x + action.0,
                y: self.y + action.1,
                goal_x: self.goal_x,
                goal_y: self.goal_y,
            };
            (new, 1.0)
        }

        fn is_goal(&self) -> bool {
            self.x == self.goal_x && self.y == self.goal_y
        }

        fn heuristic(&self) -> f64 {
            ((self.x - self.goal_x).abs() + (self.y - self.goal_y).abs()) as f64
        }
    }

    #[test]
    fn test_astar_grid() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 3,
            goal_y: 3,
        };
        let result = astar(start).unwrap();
        assert!((result.total_cost - 6.0).abs() < 1e-10); // Manhattan distance
    }

    #[test]
    fn test_beam_search_grid() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 2,
            goal_y: 2,
        };
        let result = beam_search(start, 10, 20).unwrap();
        assert!((result.total_cost - 4.0).abs() < 1e-10);
    }
}
