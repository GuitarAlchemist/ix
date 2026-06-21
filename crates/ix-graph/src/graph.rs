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

    /// Weakly-connected components: returns each node's component id, indexed by
    /// node (`result[node]`). Edge direction is ignored, so two nodes share an id
    /// iff one is reachable from the other along edges in either direction. Ids are
    /// assigned `0, 1, …` in increasing node order (the smallest node in a component
    /// fixes its id), so the count of distinct ids is the number of components.
    pub fn connected_components(&self) -> Vec<usize> {
        let n = self.node_count;
        // Undirected adjacency view built from the directed adjacency list.
        let mut undirected: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (&u, edges) in &self.adjacency {
            if u >= n {
                continue;
            }
            for &(v, _) in edges {
                if v >= n {
                    continue;
                }
                undirected[u].push(v);
                undirected[v].push(u);
            }
        }
        let mut comp = vec![usize::MAX; n];
        let mut next_id = 0;
        for start in 0..n {
            if comp[start] != usize::MAX {
                continue;
            }
            comp[start] = next_id;
            let mut queue = VecDeque::from([start]);
            while let Some(node) = queue.pop_front() {
                for &nb in &undirected[node] {
                    if comp[nb] == usize::MAX {
                        comp[nb] = next_id;
                        queue.push_back(nb);
                    }
                }
            }
            next_id += 1;
        }
        comp
    }

    /// Undirected adjacency view (deduplicated neighbour sets), built from the
    /// directed adjacency list. Shared by the centrality measures, which treat the
    /// graph as undirected.
    fn undirected_adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.node_count;
        let mut sets: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for (&u, edges) in &self.adjacency {
            if u >= n {
                continue;
            }
            for &(v, _) in edges {
                if v >= n || v == u {
                    continue;
                }
                sets[u].insert(v);
                sets[v].insert(u);
            }
        }
        sets.into_iter().map(|s| s.into_iter().collect()).collect()
    }

    /// Degree centrality (undirected): each node's neighbour count normalized by the
    /// maximum possible (`n − 1`). Indexed by node.
    pub fn degree_centrality(&self) -> Vec<f64> {
        let n = self.node_count;
        let adj = self.undirected_adjacency();
        let denom = (n as f64 - 1.0).max(1.0);
        adj.iter().map(|nb| nb.len() as f64 / denom).collect()
    }

    /// Closeness centrality (undirected, unweighted, Wasserman–Faust form for
    /// possibly-disconnected graphs): for node `v` with `r` other reachable nodes at
    /// total hop-distance `d`, `C(v) = r² / ((n − 1) · d)`. Indexed by node.
    pub fn closeness_centrality(&self) -> Vec<f64> {
        let n = self.node_count;
        let adj = self.undirected_adjacency();
        let denom = (n as f64 - 1.0).max(1.0);
        let mut out = vec![0.0; n];
        for s in 0..n {
            // BFS hop distances from s over the undirected view.
            let mut dist = vec![-1i64; n];
            dist[s] = 0;
            let mut q = VecDeque::from([s]);
            while let Some(v) = q.pop_front() {
                for &w in &adj[v] {
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        q.push_back(w);
                    }
                }
            }
            let (mut r, mut sumd) = (0.0, 0.0);
            for (i, &d) in dist.iter().enumerate() {
                if i != s && d > 0 {
                    r += 1.0;
                    sumd += d as f64;
                }
            }
            out[s] = if sumd > 0.0 { (r * r) / (denom * sumd) } else { 0.0 };
        }
        out
    }

    /// Eigenvector centrality (undirected) by power iteration on the **shifted**
    /// adjacency `A + I`, L2-normalized. Indexed by node.
    ///
    /// The `+ I` shift (each node sees itself) is load-bearing: a raw `A·x` iteration
    /// oscillates on *bipartite* graphs (a star, any path) because `A` has eigenvalues
    /// `±λ`, so even iteration counts return a near-uniform vector that mislabels hubs
    /// and leaves as equal. `A + I` has strictly positive eigenvalues `λ + 1`, so power
    /// iteration converges to the Perron vector — which ranks nodes identically to `A`
    /// on connected graphs while remaining stable on bipartite ones.
    pub fn eigenvector_centrality(&self, iterations: usize) -> Vec<f64> {
        let n = self.node_count;
        if n == 0 {
            return Vec::new();
        }
        let adj = self.undirected_adjacency();
        let mut x = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..iterations {
            // next = (A + I)·x — start each node at its own value, then add neighbours.
            let mut next = x.clone();
            for (v, nb) in adj.iter().enumerate() {
                for &w in nb {
                    next[v] += x[w];
                }
            }
            let norm: f64 = next.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm <= f64::EPSILON {
                break; // degenerate → leave the previous estimate
            }
            for v in &mut next {
                *v /= norm;
            }
            x = next;
        }
        x
    }

    /// Betweenness centrality (undirected, unweighted) via Brandes' algorithm: the
    /// fraction of shortest paths through each node, halved to undo the
    /// each-pair-counted-twice double count. Indexed by node.
    pub fn betweenness_centrality(&self) -> Vec<f64> {
        let n = self.node_count;
        let adj = self.undirected_adjacency();
        let mut bc = vec![0.0; n];
        for s in 0..n {
            let mut stack: Vec<usize> = Vec::new();
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0; n];
            sigma[s] = 1.0;
            let mut dist = vec![-1i64; n];
            dist[s] = 0;
            let mut q = VecDeque::from([s]);
            while let Some(v) = q.pop_front() {
                stack.push(v);
                for &w in &adj[v] {
                    if dist[w] < 0 {
                        dist[w] = dist[v] + 1;
                        q.push_back(w);
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        preds[w].push(v);
                    }
                }
            }
            let mut delta = vec![0.0; n];
            while let Some(w) = stack.pop() {
                for &v in &preds[w] {
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
                if w != s {
                    bc[w] += delta[w];
                }
            }
        }
        // Undirected: each shortest path is counted from both endpoints.
        for v in &mut bc {
            *v /= 2.0;
        }
        bc
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
    fn test_connected_components() {
        // Two components: {0,1,2} (a directed chain — direction ignored) and {3,4}.
        // Node 5 is isolated → its own component. Three components total.
        let mut g = Graph::with_nodes(6);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(3, 4, 1.0);

        let comp = g.connected_components();
        assert_eq!(comp.len(), 6, "one id per node");
        // Same component within {0,1,2} and within {3,4}.
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
        assert_eq!(comp[3], comp[4]);
        // Different components across the three groups.
        assert_ne!(comp[0], comp[3]);
        assert_ne!(comp[0], comp[5]);
        assert_ne!(comp[3], comp[5]);

        let distinct: std::collections::HashSet<_> = comp.iter().collect();
        assert_eq!(distinct.len(), 3, "exactly three weakly-connected components");
    }

    #[test]
    fn test_centrality_star_center_dominates() {
        // Star: center 0 joined to leaves 1,2,3 (undirected). The hub must score
        // strictly highest on every centrality measure; leaves tie.
        let mut g = Graph::with_nodes(4);
        g.add_undirected_edge(0, 1, 1.0);
        g.add_undirected_edge(0, 2, 1.0);
        g.add_undirected_edge(0, 3, 1.0);

        let deg = g.degree_centrality();
        assert!((deg[0] - 1.0).abs() < 1e-12, "hub degree centrality = 3/3 = 1");
        assert!(deg[0] > deg[1] && (deg[1] - deg[2]).abs() < 1e-12);

        let clo = g.closeness_centrality();
        assert!(clo[0] > clo[1], "hub is closest to everything");

        let eig = g.eigenvector_centrality(100);
        // A+I shift converges to the Perron vector: hub/leaf ratio is √3 ≈ 1.73 on a
        // 3-leaf star. Assert a real gap (the pre-fix bipartite oscillation returned
        // hub ≈ leaf, passing a bare `>` only by float luck).
        assert!(
            eig[0] > 1.5 * eig[1],
            "hub eigenvector centrality dominates leaves: {} vs {}",
            eig[0],
            eig[1]
        );
        assert!((eig[1] - eig[2]).abs() < 1e-9 && (eig[2] - eig[3]).abs() < 1e-9, "leaves tie");

        let bc = g.betweenness_centrality();
        // Every shortest path between two leaves passes through the hub → bc[0] > 0,
        // leaves lie on no one else's path → 0.
        assert!(bc[0] > 0.0, "hub has positive betweenness, got {}", bc[0]);
        assert!(bc[1].abs() < 1e-12, "a leaf has zero betweenness, got {}", bc[1]);
    }

    #[test]
    fn test_betweenness_path_middle_dominates() {
        // Path 0-1-2: only the middle node 1 lies on the 0↔2 shortest path.
        let mut g = Graph::with_nodes(3);
        g.add_undirected_edge(0, 1, 1.0);
        g.add_undirected_edge(1, 2, 1.0);
        let bc = g.betweenness_centrality();
        assert!((bc[1] - 1.0).abs() < 1e-12, "middle node betweenness = 1, got {}", bc[1]);
        assert!(bc[0].abs() < 1e-12 && bc[2].abs() < 1e-12, "endpoints have zero betweenness");
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
