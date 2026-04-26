//! Uninformed (blind) search algorithms: BFS, DFS, iterative deepening.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::astar::{SearchResult, SearchState};

/// Breadth-First Search.
pub fn bfs<S>(start: S) -> Option<SearchResult<S>>
where
    S: SearchState,
{
    let mut queue = VecDeque::new();
    let mut visited: HashSet<S> = HashSet::new();
    let mut came_from: HashMap<S, (S, S::Action, f64)> = HashMap::new();
    let mut nodes_expanded = 0usize;
    let mut nodes_generated = 0usize;

    queue.push_back(start.clone());
    visited.insert(start.clone());

    while let Some(current) = queue.pop_front() {
        nodes_expanded += 1;

        if current.is_goal() {
            return Some(reconstruct(
                current,
                &came_from,
                nodes_expanded,
                nodes_generated,
            ));
        }

        for (action, successor, cost) in current.successors() {
            nodes_generated += 1;
            if visited.insert(successor.clone()) {
                came_from.insert(successor.clone(), (current.clone(), action, cost));
                queue.push_back(successor);
            }
        }
    }

    None
}

/// Depth-First Search.
pub fn dfs<S>(start: S) -> Option<SearchResult<S>>
where
    S: SearchState,
{
    let mut stack = vec![start.clone()];
    let mut visited: HashSet<S> = HashSet::new();
    let mut came_from: HashMap<S, (S, S::Action, f64)> = HashMap::new();
    let mut nodes_expanded = 0usize;
    let mut nodes_generated = 0usize;

    while let Some(current) = stack.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }
        nodes_expanded += 1;

        if current.is_goal() {
            return Some(reconstruct(
                current,
                &came_from,
                nodes_expanded,
                nodes_generated,
            ));
        }

        for (action, successor, cost) in current.successors() {
            nodes_generated += 1;
            if !visited.contains(&successor) {
                came_from.insert(successor.clone(), (current.clone(), action, cost));
                stack.push(successor);
            }
        }
    }

    None
}

/// Depth-Limited Search.
pub fn depth_limited_search<S>(start: S, limit: usize) -> Option<SearchResult<S>>
where
    S: SearchState,
{
    let mut nodes_expanded = 0usize;
    let mut nodes_generated = 0usize;

    fn dls_recursive<S: SearchState>(
        state: &S,
        depth: usize,
        limit: usize,
        visited: &mut HashSet<S>,
        came_from: &mut HashMap<S, (S, S::Action, f64)>,
        expanded: &mut usize,
        generated: &mut usize,
    ) -> bool {
        *expanded += 1;

        if state.is_goal() {
            return true;
        }

        if depth >= limit {
            return false;
        }

        visited.insert(state.clone());

        for (action, successor, cost) in state.successors() {
            *generated += 1;
            if !visited.contains(&successor) {
                came_from.insert(successor.clone(), (state.clone(), action, cost));
                if dls_recursive(
                    &successor,
                    depth + 1,
                    limit,
                    visited,
                    came_from,
                    expanded,
                    generated,
                ) {
                    return true;
                }
            }
        }

        visited.remove(state);
        false
    }

    let mut visited = HashSet::new();
    let mut came_from = HashMap::new();

    if dls_recursive(
        &start,
        0,
        limit,
        &mut visited,
        &mut came_from,
        &mut nodes_expanded,
        &mut nodes_generated,
    ) {
        // Find the goal state
        let goal = came_from
            .keys()
            .chain(std::iter::once(&start))
            .find(|s| s.is_goal())
            .cloned();

        if let Some(goal) = goal {
            return Some(reconstruct(
                goal,
                &came_from,
                nodes_expanded,
                nodes_generated,
            ));
        }
    }

    None
}

/// Iterative Deepening Depth-First Search (IDDFS).
///
/// Combines the optimality of BFS with the memory efficiency of DFS.
pub fn iddfs<S>(start: S, max_depth: usize) -> Option<SearchResult<S>>
where
    S: SearchState,
{
    for depth in 0..=max_depth {
        if let Some(result) = depth_limited_search(start.clone(), depth) {
            return Some(result);
        }
    }
    None
}

/// Reconstruct path from came_from map.
fn reconstruct<S: SearchState>(
    goal: S,
    came_from: &HashMap<S, (S, S::Action, f64)>,
    nodes_expanded: usize,
    nodes_generated: usize,
) -> SearchResult<S> {
    let mut path = vec![goal.clone()];
    let mut actions = Vec::new();
    let mut cost = 0.0;
    let mut state = goal;

    while let Some((parent, action, step_cost)) = came_from.get(&state) {
        actions.push(action.clone());
        cost += step_cost;
        path.push(parent.clone());
        state = parent.clone();
    }

    path.reverse();
    actions.reverse();

    SearchResult {
        path,
        actions,
        cost,
        nodes_expanded,
        nodes_generated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq, Hash)]
    struct SimpleState {
        val: i32,
        goal: i32,
    }

    impl SearchState for SimpleState {
        type Action = i32;

        fn successors(&self) -> Vec<(Self::Action, Self, f64)> {
            vec![
                (
                    1,
                    SimpleState {
                        val: self.val + 1,
                        goal: self.goal,
                    },
                    1.0,
                ),
                (
                    -1,
                    SimpleState {
                        val: self.val - 1,
                        goal: self.goal,
                    },
                    1.0,
                ),
            ]
        }

        fn is_goal(&self) -> bool {
            self.val == self.goal
        }
    }

    #[test]
    fn test_bfs_shortest() {
        let start = SimpleState { val: 0, goal: 3 };
        let result = bfs(start).unwrap();
        assert_eq!(result.cost, 3.0);
    }

    #[test]
    fn test_iddfs_finds_optimal() {
        let start = SimpleState { val: 0, goal: 3 };
        let result = iddfs(start, 10).unwrap();
        assert_eq!(result.cost, 3.0);
    }
}
