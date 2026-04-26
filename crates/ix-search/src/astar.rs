//! A* search and variants.
//!
//! Optimal pathfinding using f(n) = g(n) + h(n).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::Hash;

/// A state in a search problem.
pub trait SearchState: Clone + Eq + Hash {
    /// Unique identifier for the state.
    type Action: Clone;

    /// Generate successor states with (action, successor, step_cost).
    fn successors(&self) -> Vec<(Self::Action, Self, f64)>;

    /// Check if this is a goal state.
    fn is_goal(&self) -> bool;
}

/// Node in the search frontier.
#[allow(dead_code)]
struct SearchNode<S: SearchState> {
    state: S,
    g_cost: f64, // Path cost from start
    f_cost: f64, // g + h (estimated total)
    parent: Option<S>,
    action: Option<S::Action>,
}

impl<S: SearchState> PartialEq for SearchNode<S> {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
    }
}

impl<S: SearchState> Eq for SearchNode<S> {}

impl<S: SearchState> Ord for SearchNode<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl<S: SearchState> PartialOrd for SearchNode<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A* search result.
#[derive(Debug, Clone)]
pub struct SearchResult<S: SearchState> {
    /// The path of states from start to goal.
    pub path: Vec<S>,
    /// Actions taken.
    pub actions: Vec<S::Action>,
    /// Total path cost.
    pub cost: f64,
    /// Number of nodes expanded.
    pub nodes_expanded: usize,
    /// Number of nodes generated.
    pub nodes_generated: usize,
}

/// Standard A* search.
///
/// `heuristic`: h(state) -> estimated cost to goal. Must be admissible (never overestimates).
pub fn astar<S, H>(start: S, heuristic: H) -> Option<SearchResult<S>>
where
    S: SearchState,
    H: Fn(&S) -> f64,
{
    weighted_astar(start, heuristic, 1.0)
}

/// Weighted A* search (WA*).
///
/// Uses f(n) = g(n) + w * h(n). With w > 1, trades optimality for speed.
/// Solution cost is at most w times the optimal cost.
pub fn weighted_astar<S, H>(start: S, heuristic: H, weight: f64) -> Option<SearchResult<S>>
where
    S: SearchState,
    H: Fn(&S) -> f64,
{
    let mut open = BinaryHeap::new();
    let mut g_scores: HashMap<S, f64> = HashMap::new();
    let mut came_from: HashMap<S, (S, S::Action)> = HashMap::new();
    let mut closed: HashSet<S> = HashSet::new();
    let mut nodes_expanded = 0usize;
    let mut nodes_generated = 0usize;

    let h = heuristic(&start);
    open.push(SearchNode {
        state: start.clone(),
        g_cost: 0.0,
        f_cost: weight * h,
        parent: None,
        action: None,
    });
    g_scores.insert(start.clone(), 0.0);

    while let Some(current) = open.pop() {
        if current.state.is_goal() {
            // Reconstruct path
            let mut path = vec![current.state.clone()];
            let mut actions = Vec::new();
            let mut state = current.state.clone();

            while let Some((parent, action)) = came_from.get(&state) {
                actions.push(action.clone());
                path.push(parent.clone());
                state = parent.clone();
            }

            path.reverse();
            actions.reverse();

            return Some(SearchResult {
                path,
                actions,
                cost: current.g_cost,
                nodes_expanded,
                nodes_generated,
            });
        }

        if closed.contains(&current.state) {
            continue;
        }
        closed.insert(current.state.clone());
        nodes_expanded += 1;

        for (action, successor, step_cost) in current.state.successors() {
            nodes_generated += 1;

            if closed.contains(&successor) {
                continue;
            }

            let new_g = current.g_cost + step_cost;
            let prev_g = g_scores.get(&successor).copied().unwrap_or(f64::INFINITY);

            if new_g < prev_g {
                g_scores.insert(successor.clone(), new_g);
                came_from.insert(successor.clone(), (current.state.clone(), action.clone()));

                let h = heuristic(&successor);
                open.push(SearchNode {
                    state: successor,
                    g_cost: new_g,
                    f_cost: new_g + weight * h,
                    parent: Some(current.state.clone()),
                    action: Some(action),
                });
            }
        }
    }

    None // No path found
}

/// Greedy Best-First Search — uses only h(n), ignoring path cost.
pub fn greedy_best_first<S, H>(start: S, heuristic: H) -> Option<SearchResult<S>>
where
    S: SearchState,
    H: Fn(&S) -> f64,
{
    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<S, (S, S::Action, f64)> = HashMap::new();
    let mut visited: HashSet<S> = HashSet::new();
    let mut nodes_expanded = 0usize;
    let mut nodes_generated = 0usize;

    open.push(SearchNode {
        state: start.clone(),
        g_cost: 0.0,
        f_cost: heuristic(&start),
        parent: None,
        action: None,
    });

    while let Some(current) = open.pop() {
        if current.state.is_goal() {
            let mut path = vec![current.state.clone()];
            let mut actions = Vec::new();
            let mut cost = 0.0;
            let mut state = current.state.clone();

            while let Some((parent, action, step_cost)) = came_from.get(&state) {
                actions.push(action.clone());
                cost += step_cost;
                path.push(parent.clone());
                state = parent.clone();
            }

            path.reverse();
            actions.reverse();

            return Some(SearchResult {
                path,
                actions,
                cost,
                nodes_expanded,
                nodes_generated,
            });
        }

        if !visited.insert(current.state.clone()) {
            continue;
        }
        nodes_expanded += 1;

        for (action, successor, step_cost) in current.state.successors() {
            nodes_generated += 1;
            if !visited.contains(&successor) {
                came_from.insert(
                    successor.clone(),
                    (current.state.clone(), action.clone(), step_cost),
                );
                let h = heuristic(&successor);
                open.push(SearchNode {
                    state: successor,
                    g_cost: 0.0,
                    f_cost: h,
                    parent: None,
                    action: Some(action),
                });
            }
        }
    }

    None
}

/// Uniform Cost Search (Dijkstra's) — A* with h = 0.
pub fn uniform_cost_search<S>(start: S) -> Option<SearchResult<S>>
where
    S: SearchState,
{
    astar(start, |_| 0.0)
}

/// Bidirectional A* search.
///
/// Searches from both start and goal simultaneously.
/// Requires a `reverse_successors` function (predecessors from goal side).
pub fn bidirectional_astar<S, H, R>(
    start: S,
    goal: S,
    forward_heuristic: H,
    reverse_heuristic: R,
) -> Option<SearchResult<S>>
where
    S: SearchState,
    H: Fn(&S) -> f64,
    R: Fn(&S) -> f64,
{
    // Forward search
    let mut open_f = BinaryHeap::new();
    let mut g_f: HashMap<S, f64> = HashMap::new();
    let mut closed_f: HashSet<S> = HashSet::new();
    let mut parent_f: HashMap<S, (S, S::Action)> = HashMap::new();

    // Backward search
    let mut open_b = BinaryHeap::new();
    let mut g_b: HashMap<S, f64> = HashMap::new();
    let mut closed_b: HashSet<S> = HashSet::new();
    let mut parent_b: HashMap<S, (S, S::Action)> = HashMap::new();

    let mut nodes_expanded = 0usize;
    let mut nodes_generated = 0usize;

    open_f.push(SearchNode {
        state: start.clone(),
        g_cost: 0.0,
        f_cost: forward_heuristic(&start),
        parent: None,
        action: None,
    });
    g_f.insert(start.clone(), 0.0);

    open_b.push(SearchNode {
        state: goal.clone(),
        g_cost: 0.0,
        f_cost: reverse_heuristic(&goal),
        parent: None,
        action: None,
    });
    g_b.insert(goal.clone(), 0.0);

    let mut best_cost = f64::INFINITY;
    let mut meeting_point: Option<S> = None;

    for _ in 0..1_000_000 {
        if open_f.is_empty() && open_b.is_empty() {
            break;
        }

        // Expand forward
        if let Some(node) = open_f.pop() {
            if !closed_f.contains(&node.state) {
                closed_f.insert(node.state.clone());
                nodes_expanded += 1;

                if let Some(&gb) = g_b.get(&node.state) {
                    let total = node.g_cost + gb;
                    if total < best_cost {
                        best_cost = total;
                        meeting_point = Some(node.state.clone());
                    }
                }

                for (action, succ, cost) in node.state.successors() {
                    nodes_generated += 1;
                    let new_g = node.g_cost + cost;
                    if new_g < *g_f.get(&succ).unwrap_or(&f64::INFINITY) {
                        g_f.insert(succ.clone(), new_g);
                        parent_f.insert(succ.clone(), (node.state.clone(), action.clone()));
                        open_f.push(SearchNode {
                            state: succ,
                            g_cost: new_g,
                            f_cost: new_g + forward_heuristic(&node.state),
                            parent: None,
                            action: Some(action),
                        });
                    }
                }
            }
        }

        // Expand backward
        if let Some(node) = open_b.pop() {
            if !closed_b.contains(&node.state) {
                closed_b.insert(node.state.clone());
                nodes_expanded += 1;

                if let Some(&gf) = g_f.get(&node.state) {
                    let total = gf + node.g_cost;
                    if total < best_cost {
                        best_cost = total;
                        meeting_point = Some(node.state.clone());
                    }
                }

                for (action, succ, cost) in node.state.successors() {
                    nodes_generated += 1;
                    let new_g = node.g_cost + cost;
                    if new_g < *g_b.get(&succ).unwrap_or(&f64::INFINITY) {
                        g_b.insert(succ.clone(), new_g);
                        parent_b.insert(succ.clone(), (node.state.clone(), action.clone()));
                        open_b.push(SearchNode {
                            state: succ,
                            g_cost: new_g,
                            f_cost: new_g + reverse_heuristic(&node.state),
                            parent: None,
                            action: Some(action),
                        });
                    }
                }
            }
        }

        // Termination check
        let min_f = open_f.peek().map(|n| n.f_cost).unwrap_or(f64::INFINITY);
        let min_b = open_b.peek().map(|n| n.f_cost).unwrap_or(f64::INFINITY);
        if min_f.min(min_b) >= best_cost {
            break;
        }
    }

    meeting_point.map(|mp| {
        // Reconstruct forward path
        let mut fwd_path = vec![mp.clone()];
        let mut fwd_actions = Vec::new();
        let mut state = mp.clone();
        while let Some((parent, action)) = parent_f.get(&state) {
            fwd_actions.push(action.clone());
            fwd_path.push(parent.clone());
            state = parent.clone();
        }
        fwd_path.reverse();
        fwd_actions.reverse();

        // Reconstruct backward path
        state = mp;
        while let Some((parent, action)) = parent_b.get(&state) {
            fwd_path.push(parent.clone());
            fwd_actions.push(action.clone());
            state = parent.clone();
        }

        SearchResult {
            path: fwd_path,
            actions: fwd_actions,
            cost: best_cost,
            nodes_expanded,
            nodes_generated,
        }
    })
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
        width: i32,
        height: i32,
    }

    impl SearchState for GridPos {
        type Action = (i32, i32); // dx, dy

        fn successors(&self) -> Vec<(Self::Action, Self, f64)> {
            let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
            dirs.iter()
                .filter_map(|&(dx, dy)| {
                    let nx = self.x + dx;
                    let ny = self.y + dy;
                    if nx >= 0 && nx < self.width && ny >= 0 && ny < self.height {
                        Some((
                            (dx, dy),
                            GridPos {
                                x: nx,
                                y: ny,
                                ..*self
                            },
                            1.0,
                        ))
                    } else {
                        None
                    }
                })
                .collect()
        }

        fn is_goal(&self) -> bool {
            self.x == self.goal_x && self.y == self.goal_y
        }
    }

    fn manhattan(s: &GridPos) -> f64 {
        ((s.x - s.goal_x).abs() + (s.y - s.goal_y).abs()) as f64
    }

    #[test]
    fn test_astar_finds_shortest() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 4,
            goal_y: 4,
            width: 10,
            height: 10,
        };
        let result = astar(start, manhattan).unwrap();
        assert_eq!(result.cost, 8.0, "Manhattan distance on grid should be 8");
        assert_eq!(result.path.len(), 9);
    }

    #[test]
    fn test_weighted_astar_faster() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 9,
            goal_y: 9,
            width: 20,
            height: 20,
        };

        let opt = astar(start.clone(), manhattan).unwrap();
        let weighted = weighted_astar(start, manhattan, 2.0).unwrap();

        // Weighted should expand fewer nodes (or equal)
        assert!(weighted.nodes_expanded <= opt.nodes_expanded + 5);
        // But solution may be suboptimal (at most 2x)
        assert!(weighted.cost <= opt.cost * 2.0);
    }

    #[test]
    fn test_greedy_finds_path() {
        let start = GridPos {
            x: 0,
            y: 0,
            goal_x: 3,
            goal_y: 3,
            width: 10,
            height: 10,
        };
        let result = greedy_best_first(start, manhattan).unwrap();
        assert!(result.path.last().unwrap().is_goal());
    }
}
