//! Core DAG data structure with cycle detection and topological sort.

use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a node in the pipeline.
pub type NodeId = String;

/// A directed acyclic graph.
///
/// Generic over node data `N`. Enforces acyclicity on every edge insertion.
#[derive(Debug, Clone)]
pub struct Dag<N> {
    /// Node data, keyed by ID.
    nodes: HashMap<NodeId, N>,
    /// Adjacency list: node → set of successors.
    edges: HashMap<NodeId, HashSet<NodeId>>,
    /// Reverse adjacency: node → set of predecessors.
    reverse: HashMap<NodeId, HashSet<NodeId>>,
    /// Insertion order (for deterministic iteration).
    order: Vec<NodeId>,
}

/// Errors that can occur when building a DAG.
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    #[error("node '{0}' already exists")]
    DuplicateNode(NodeId),

    #[error("node '{0}' not found")]
    NodeNotFound(NodeId),

    #[error("adding edge {0} → {1} would create a cycle")]
    CycleDetected(NodeId, NodeId),

    #[error("self-loop on node '{0}'")]
    SelfLoop(NodeId),
}

impl<N> Dag<N> {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            reverse: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Add a node. Returns error if ID already exists.
    pub fn add_node(&mut self, id: impl Into<NodeId>, data: N) -> Result<(), DagError> {
        let id = id.into();
        if self.nodes.contains_key(&id) {
            return Err(DagError::DuplicateNode(id));
        }
        self.nodes.insert(id.clone(), data);
        self.edges.entry(id.clone()).or_default();
        self.reverse.entry(id.clone()).or_default();
        self.order.push(id);
        Ok(())
    }

    /// Add a directed edge from → to. Returns error if it would create a cycle.
    pub fn add_edge(&mut self, from: impl Into<NodeId>, to: impl Into<NodeId>) -> Result<(), DagError> {
        let from = from.into();
        let to = to.into();

        if from == to {
            return Err(DagError::SelfLoop(from));
        }
        if !self.nodes.contains_key(&from) {
            return Err(DagError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(DagError::NodeNotFound(to));
        }

        // Check if adding this edge would create a cycle:
        // There's a cycle iff there's already a path from `to` → `from`.
        if self.has_path(&to, &from) {
            return Err(DagError::CycleDetected(from, to));
        }

        self.edges.entry(from.clone()).or_default().insert(to.clone());
        self.reverse.entry(to).or_default().insert(from);
        Ok(())
    }

    /// Check if a path exists from `start` to `end` (BFS).
    pub fn has_path(&self, start: &str, end: &str) -> bool {
        if start == end {
            return true;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());
        visited.insert(start.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = self.edges.get(&current) {
                for next in neighbors {
                    if next == end {
                        return true;
                    }
                    if visited.insert(next.clone()) {
                        queue.push_back(next.clone());
                    }
                }
            }
        }

        false
    }

    /// Get node data by ID.
    pub fn get(&self, id: &str) -> Option<&N> {
        self.nodes.get(id)
    }

    /// Get mutable node data.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut N> {
        self.nodes.get_mut(id)
    }

    /// Get the predecessors (dependencies) of a node.
    pub fn predecessors(&self, id: &str) -> Vec<&NodeId> {
        self.reverse.get(id)
            .map(|s| s.iter().collect())
            .unwrap_or_default()
    }

    /// Get the successors (dependents) of a node.
    pub fn successors(&self, id: &str) -> Vec<&NodeId> {
        self.edges.get(id)
            .map(|s| s.iter().collect())
            .unwrap_or_default()
    }

    /// Get all root nodes (no predecessors).
    pub fn roots(&self) -> Vec<&NodeId> {
        self.order.iter()
            .filter(|id| {
                self.reverse.get(id.as_str())
                    .is_none_or(|preds| preds.is_empty())
            })
            .collect()
    }

    /// Get all leaf nodes (no successors).
    pub fn leaves(&self) -> Vec<&NodeId> {
        self.order.iter()
            .filter(|id| {
                self.edges.get(id.as_str())
                    .is_none_or(|succs| succs.is_empty())
            })
            .collect()
    }

    /// In-degree of a node (number of predecessors).
    pub fn in_degree(&self, id: &str) -> usize {
        self.reverse.get(id).map_or(0, |s| s.len())
    }

    /// Out-degree of a node (number of successors).
    pub fn out_degree(&self, id: &str) -> usize {
        self.edges.get(id).map_or(0, |s| s.len())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|s| s.len()).sum()
    }

    /// All node IDs in insertion order.
    pub fn node_ids(&self) -> &[NodeId] {
        &self.order
    }

    /// Topological sort using Kahn's algorithm.
    ///
    /// Returns nodes in an order where every node comes after its dependencies.
    /// Panics if the graph has a cycle (should be impossible if built via add_edge).
    pub fn topological_sort(&self) -> Vec<&NodeId> {
        let mut in_degrees: HashMap<&str, usize> = HashMap::new();
        for id in &self.order {
            in_degrees.insert(id, self.in_degree(id));
        }

        let mut queue: VecDeque<&str> = in_degrees.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            sorted.push(node);
            if let Some(succs) = self.edges.get(node) {
                for succ in succs {
                    if let Some(deg) = in_degrees.get_mut(succ.as_str()) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        // Map back to NodeId references from self.order, preserving topological order
        let pos_map: HashMap<&str, usize> = sorted.iter().enumerate().map(|(i, &s)| (s, i)).collect();
        let mut result: Vec<&NodeId> = self.order.iter()
            .filter(|id| pos_map.contains_key(id.as_str()))
            .collect();
        result.sort_by_key(|id| pos_map.get(id.as_str()).copied().unwrap_or(usize::MAX));
        result
    }

    /// Group nodes into execution levels (for parallel execution).
    ///
    /// Level 0: all roots (no dependencies).
    /// Level 1: nodes whose dependencies are all in level 0.
    /// Level N: nodes whose dependencies are all in levels < N.
    ///
    /// Nodes within the same level can execute in parallel.
    pub fn parallel_levels(&self) -> Vec<Vec<&NodeId>> {
        let mut levels: Vec<Vec<&NodeId>> = Vec::new();
        let mut assigned: HashMap<&str, usize> = HashMap::new();

        let topo = self.topological_sort();

        for id in &topo {
            let preds = self.predecessors(id);
            let level = if preds.is_empty() {
                0
            } else {
                preds.iter()
                    .map(|p| assigned.get(p.as_str()).copied().unwrap_or(0) + 1)
                    .max()
                    .unwrap_or(0)
            };

            assigned.insert(id, level);

            while levels.len() <= level {
                levels.push(Vec::new());
            }
            levels[level].push(id);
        }

        levels
    }

    /// Find the critical path (longest path through the DAG).
    ///
    /// Requires a cost function for each node.
    pub fn critical_path<F>(&self, cost_fn: F) -> (Vec<&NodeId>, f64)
    where
        F: Fn(&NodeId, &N) -> f64,
    {
        let topo = self.topological_sort();
        let mut dist: HashMap<&str, f64> = HashMap::new();
        let mut prev: HashMap<&str, &str> = HashMap::new();

        for id in &topo {
            let node_cost = cost_fn(id, self.nodes.get(id.as_str()).unwrap());
            let preds = self.predecessors(id);

            let max_pred = preds.iter()
                .map(|p| dist.get(p.as_str()).copied().unwrap_or(0.0))
                .fold(0.0f64, f64::max);

            let best_pred = preds.iter()
                .max_by(|a, b| {
                    let da = dist.get(a.as_str()).unwrap_or(&0.0);
                    let db = dist.get(b.as_str()).unwrap_or(&0.0);
                    da.partial_cmp(db).unwrap()
                });

            dist.insert(id, max_pred + node_cost);
            if let Some(pred) = best_pred {
                prev.insert(id, pred);
            }
        }

        // Find the end node with maximum distance
        let (&end_node, &total_cost) = dist.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap_or((&"", &0.0));

        // Trace back the critical path
        let mut path = vec![];
        let mut current = end_node;
        loop {
            if let Some(id) = self.order.iter().find(|id| id.as_str() == current) {
                path.push(id);
            }
            if let Some(&p) = prev.get(current) {
                current = p;
            } else {
                break;
            }
        }
        path.reverse();

        (path, total_cost)
    }
}

impl<N> Default for Dag<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_dag() {
        let mut dag = Dag::new();
        dag.add_node("a", "Load data").unwrap();
        dag.add_node("b", "Process").unwrap();
        dag.add_node("c", "Output").unwrap();

        dag.add_edge("a", "b").unwrap();
        dag.add_edge("b", "c").unwrap();

        assert_eq!(dag.node_count(), 3);
        assert_eq!(dag.edge_count(), 2);
    }

    #[test]
    fn test_cycle_detection() {
        let mut dag = Dag::new();
        dag.add_node("a", ()).unwrap();
        dag.add_node("b", ()).unwrap();
        dag.add_node("c", ()).unwrap();

        dag.add_edge("a", "b").unwrap();
        dag.add_edge("b", "c").unwrap();

        // This would create a cycle: c → a
        let result = dag.add_edge("c", "a");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), DagError::CycleDetected(_, _)));
    }

    #[test]
    fn test_self_loop() {
        let mut dag = Dag::new();
        dag.add_node("a", ()).unwrap();
        assert!(matches!(dag.add_edge("a", "a"), Err(DagError::SelfLoop(_))));
    }

    #[test]
    fn test_topological_sort() {
        let mut dag = Dag::new();
        dag.add_node("d", ()).unwrap();
        dag.add_node("c", ()).unwrap();
        dag.add_node("b", ()).unwrap();
        dag.add_node("a", ()).unwrap();

        dag.add_edge("a", "b").unwrap();
        dag.add_edge("a", "c").unwrap();
        dag.add_edge("b", "d").unwrap();
        dag.add_edge("c", "d").unwrap();

        let sorted = dag.topological_sort();
        let ids: Vec<&str> = sorted.iter().map(|s| s.as_str()).collect();

        // "a" must come before "b" and "c", both must come before "d"
        let pos_a = ids.iter().position(|&x| x == "a").unwrap();
        let pos_b = ids.iter().position(|&x| x == "b").unwrap();
        let pos_c = ids.iter().position(|&x| x == "c").unwrap();
        let pos_d = ids.iter().position(|&x| x == "d").unwrap();

        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_parallel_levels() {
        //   a ──→ c ──→ e
        //   b ──→ d ──↗
        let mut dag = Dag::new();
        dag.add_node("a", ()).unwrap();
        dag.add_node("b", ()).unwrap();
        dag.add_node("c", ()).unwrap();
        dag.add_node("d", ()).unwrap();
        dag.add_node("e", ()).unwrap();

        dag.add_edge("a", "c").unwrap();
        dag.add_edge("b", "d").unwrap();
        dag.add_edge("c", "e").unwrap();
        dag.add_edge("d", "e").unwrap();

        let levels = dag.parallel_levels();
        assert_eq!(levels.len(), 3);

        // Level 0: a, b (roots — can run in parallel)
        let l0: HashSet<&str> = levels[0].iter().map(|s| s.as_str()).collect();
        assert!(l0.contains("a"));
        assert!(l0.contains("b"));

        // Level 1: c, d (depend on level 0 — can run in parallel)
        let l1: HashSet<&str> = levels[1].iter().map(|s| s.as_str()).collect();
        assert!(l1.contains("c"));
        assert!(l1.contains("d"));

        // Level 2: e (depends on level 1)
        let l2: HashSet<&str> = levels[2].iter().map(|s| s.as_str()).collect();
        assert!(l2.contains("e"));
    }

    #[test]
    fn test_roots_and_leaves() {
        let mut dag = Dag::new();
        dag.add_node("a", ()).unwrap();
        dag.add_node("b", ()).unwrap();
        dag.add_node("c", ()).unwrap();

        dag.add_edge("a", "b").unwrap();
        dag.add_edge("b", "c").unwrap();

        let roots: Vec<&str> = dag.roots().iter().map(|s| s.as_str()).collect();
        assert_eq!(roots, vec!["a"]);

        let leaves: Vec<&str> = dag.leaves().iter().map(|s| s.as_str()).collect();
        assert_eq!(leaves, vec!["c"]);
    }

    #[test]
    fn test_critical_path() {
        let mut dag = Dag::new();
        dag.add_node("a", 2.0f64).unwrap();
        dag.add_node("b", 5.0f64).unwrap();
        dag.add_node("c", 1.0f64).unwrap();
        dag.add_node("d", 3.0f64).unwrap();

        dag.add_edge("a", "b").unwrap();
        dag.add_edge("a", "c").unwrap();
        dag.add_edge("b", "d").unwrap();
        dag.add_edge("c", "d").unwrap();

        let (path, cost) = dag.critical_path(|_, &c| c);
        let ids: Vec<&str> = path.iter().map(|s| s.as_str()).collect();

        // Critical path: a(2) → b(5) → d(3) = 10
        assert_eq!(ids, vec!["a", "b", "d"]);
        assert!((cost - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_diamond_dag() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let mut dag = Dag::new();
        dag.add_node("a", ()).unwrap();
        dag.add_node("b", ()).unwrap();
        dag.add_node("c", ()).unwrap();
        dag.add_node("d", ()).unwrap();

        dag.add_edge("a", "b").unwrap();
        dag.add_edge("a", "c").unwrap();
        dag.add_edge("b", "d").unwrap();
        dag.add_edge("c", "d").unwrap();

        let levels = dag.parallel_levels();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].len(), 1); // a
        assert_eq!(levels[1].len(), 2); // b, c in parallel
        assert_eq!(levels[2].len(), 1); // d
    }
}
