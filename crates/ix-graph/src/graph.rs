//! Core graph data structures and algorithms.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

/// A weighted directed graph using adjacency list.
#[derive(Debug, Clone)]
pub struct Graph {
    /// adjacency[node] = vec![(neighbor, weight)]
    pub adjacency: HashMap<usize, Vec<(usize, f64)>>,
    pub node_count: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            node_count: 0,
        }
    }

    pub fn with_nodes(n: usize) -> Self {
        let mut g = Self::new();
        g.node_count = n;
        for i in 0..n {
            g.adjacency.entry(i).or_default();
        }
        g
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.adjacency.entry(from).or_default().push((to, weight));
        self.adjacency.entry(to).or_default();
        self.node_count = self.node_count.max(from + 1).max(to + 1);
    }

    pub fn add_undirected_edge(&mut self, a: usize, b: usize, weight: f64) {
        self.add_edge(a, b, weight);
        self.add_edge(b, a, weight);
    }

    pub fn neighbors(&self, node: usize) -> &[(usize, f64)] {
        self.adjacency
            .get(&node)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Breadth-first search. Returns distances from source (-1 = unreachable).
    pub fn bfs(&self, source: usize) -> HashMap<usize, i64> {
        let mut dist: HashMap<usize, i64> = HashMap::new();
        let mut queue = VecDeque::new();

        for &node in self.adjacency.keys() {
            dist.insert(node, -1);
        }
        dist.insert(source, 0);
        queue.push_back(source);

        while let Some(node) = queue.pop_front() {
            let d = dist[&node];
            for &(neighbor, _) in self.neighbors(node) {
                if dist.get(&neighbor) == Some(&-1) {
                    dist.insert(neighbor, d + 1);
                    queue.push_back(neighbor);
                }
            }
        }
        dist
    }

    /// Depth-first search. Returns visit order.
    pub fn dfs(&self, source: usize) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.dfs_recursive(source, &mut visited, &mut order);
        order
    }

    fn dfs_recursive(&self, node: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        if visited.contains(&node) {
            return;
        }
        visited.insert(node);
        order.push(node);
        for &(neighbor, _) in self.neighbors(node) {
            self.dfs_recursive(neighbor, visited, order);
        }
    }

    /// Dijkstra's shortest path. Returns (distances, predecessors).
    pub fn dijkstra(&self, source: usize) -> (HashMap<usize, f64>, HashMap<usize, Option<usize>>) {
        let mut dist: HashMap<usize, f64> = HashMap::new();
        let mut prev: HashMap<usize, Option<usize>> = HashMap::new();
        let mut heap = BinaryHeap::new();

        for &node in self.adjacency.keys() {
            dist.insert(node, f64::INFINITY);
            prev.insert(node, None);
        }
        dist.insert(source, 0.0);
        heap.push(DijkstraState {
            cost: 0.0,
            node: source,
        });

        while let Some(DijkstraState { cost, node }) = heap.pop() {
            if cost > dist[&node] {
                continue;
            }
            for &(neighbor, weight) in self.neighbors(node) {
                let new_cost = cost + weight;
                if new_cost < *dist.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    dist.insert(neighbor, new_cost);
                    prev.insert(neighbor, Some(node));
                    heap.push(DijkstraState {
                        cost: new_cost,
                        node: neighbor,
                    });
                }
            }
        }

        (dist, prev)
    }

    /// Reconstruct shortest path from Dijkstra predecessors.
    pub fn shortest_path(&self, source: usize, target: usize) -> Option<Vec<usize>> {
        let (_, prev) = self.dijkstra(source);
        let mut path = Vec::new();
        let mut current = target;

        loop {
            path.push(current);
            if current == source {
                path.reverse();
                return Some(path);
            }
            match prev.get(&current) {
                Some(Some(p)) => current = *p,
                _ => return None,
            }
        }
    }

    /// Topological sort (Kahn's algorithm). Returns None if cycle exists.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for &node in self.adjacency.keys() {
            in_degree.entry(node).or_insert(0);
        }
        for edges in self.adjacency.values() {
            for &(neighbor, _) in edges {
                *in_degree.entry(neighbor).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&node, _)| node)
            .collect();
        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &(neighbor, _) in self.neighbors(node) {
                if let Some(d) = in_degree.get_mut(&neighbor) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if order.len() == self.adjacency.len() {
            Some(order)
        } else {
            None // Cycle detected
        }
    }

    /// PageRank algorithm.
    pub fn pagerank(&self, damping: f64, iterations: usize) -> HashMap<usize, f64> {
        let n = self.adjacency.len() as f64;
        let mut rank: HashMap<usize, f64> =
            self.adjacency.keys().map(|&node| (node, 1.0 / n)).collect();

        let out_degree: HashMap<usize, usize> = self
            .adjacency
            .iter()
            .map(|(&node, edges)| (node, edges.len()))
            .collect();

        for _ in 0..iterations {
            let mut new_rank: HashMap<usize, f64> = self
                .adjacency
                .keys()
                .map(|&node| (node, (1.0 - damping) / n))
                .collect();

            for (&node, edges) in &self.adjacency {
                let out = out_degree[&node] as f64;
                if out > 0.0 {
                    let share = damping * rank[&node] / out;
                    for &(neighbor, _) in edges {
                        *new_rank.get_mut(&neighbor).unwrap() += share;
                    }
                }
            }

            rank = new_rank;
        }

        rank
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(PartialEq)]
struct DijkstraState {
    cost: f64,
    node: usize,
}

impl Eq for DijkstraState {}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bfs() {
        let mut g = Graph::new();
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(1, 3, 1.0);
        g.add_edge(2, 3, 1.0);

        let dist = g.bfs(0);
        assert_eq!(dist[&0], 0);
        assert_eq!(dist[&1], 1);
        assert_eq!(dist[&3], 2);
    }

    #[test]
    fn test_dijkstra() {
        let mut g = Graph::new();
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 4.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(1, 3, 5.0);
        g.add_edge(2, 3, 1.0);

        let (dist, _) = g.dijkstra(0);
        assert!((dist[&0] - 0.0).abs() < 1e-10);
        assert!((dist[&1] - 1.0).abs() < 1e-10);
        assert!((dist[&2] - 3.0).abs() < 1e-10); // 0->1->2
        assert!((dist[&3] - 4.0).abs() < 1e-10); // 0->1->2->3
    }

    #[test]
    fn test_shortest_path() {
        let mut g = Graph::new();
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 10.0);

        let path = g.shortest_path(0, 2).unwrap();
        assert_eq!(path, vec![0, 1, 2]);
    }

    #[test]
    fn test_topological_sort() {
        let mut g = Graph::new();
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(1, 3, 1.0);
        g.add_edge(2, 3, 1.0);

        let order = g.topological_sort().unwrap();
        assert_eq!(order[0], 0); // 0 must come first
        assert_eq!(*order.last().unwrap(), 3); // 3 must come last
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = Graph::new();
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);

        assert!(g.topological_sort().is_none());
    }

    #[test]
    fn test_pagerank() {
        let mut g = Graph::new();
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);

        let rank = g.pagerank(0.85, 100);
        // In a cycle of 3, all ranks should be equal
        let r0 = rank[&0];
        let r1 = rank[&1];
        let r2 = rank[&2];
        assert!((r0 - r1).abs() < 0.01);
        assert!((r1 - r2).abs() < 0.01);
    }
}
