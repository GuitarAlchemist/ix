//! Binary Space Partition tree for arbitrary dimensions.
//!
//! Provides a `BspNode<D>` that partitions D-dimensional space using axis-aligned
//! hyperplanes (cycling through dimensions). Supports insertion, region queries,
//! and nearest-neighbor search.
//!
//! Uses const generics so the dimension is fixed at compile time, enabling
//! efficient stack-allocated point arrays from 2D up to 16D (sedenion space).
//!
//! # Examples
//!
//! ```
//! use machin_math::bsp::BspTree;
//!
//! let mut tree = BspTree::<2>::new();
//! tree.insert([1.0, 2.0]);
//! tree.insert([3.0, 4.0]);
//! tree.insert([5.0, 1.0]);
//!
//! // Nearest neighbor
//! let (nearest, dist_sq) = tree.nearest_neighbor(&[2.0, 3.0]).unwrap();
//! assert_eq!(nearest, [1.0, 2.0]); // closest point
//!
//! // Region query (axis-aligned bounding box)
//! let results = tree.query_region(&[0.0, 0.0], &[4.0, 5.0]);
//! assert_eq!(results.len(), 2); // [1,2] and [3,4] are inside
//! ```

use crate::error::MathError;

/// A point in D-dimensional space.
pub type Point<const D: usize> = [f64; D];

/// A BSP tree for D-dimensional points.
///
/// Internally wraps an optional root node. An empty tree has no root.
pub struct BspTree<const D: usize> {
    root: Option<Box<BspNode<D>>>,
    size: usize,
}

/// A single node in the BSP tree.
struct BspNode<const D: usize> {
    point: Point<D>,
    /// Which dimension (axis) this node splits on.
    split_dim: usize,
    left: Option<Box<BspNode<D>>>,
    right: Option<Box<BspNode<D>>>,
}

impl<const D: usize> BspTree<D> {
    /// Create an empty BSP tree.
    pub fn new() -> Self {
        assert!(D > 0, "BSP tree dimension must be at least 1");
        Self {
            root: None,
            size: 0,
        }
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Insert a point into the tree.
    pub fn insert(&mut self, point: Point<D>) {
        self.root = Some(Self::insert_node(self.root.take(), point, 0));
        self.size += 1;
    }

    /// Build a balanced BSP tree from a set of points.
    ///
    /// This produces a more balanced tree than sequential insertion,
    /// giving O(log n) expected query time.
    pub fn from_points(mut points: Vec<Point<D>>) -> Self {
        let size = points.len();
        let root = Self::build_balanced(&mut points, 0);
        Self { root, size }
    }

    /// Find the nearest neighbor to `query`.
    ///
    /// Returns `(point, squared_distance)` or `None` if tree is empty.
    pub fn nearest_neighbor(&self, query: &Point<D>) -> Option<(Point<D>, f64)> {
        let mut best_point = None;
        let mut best_dist_sq = f64::INFINITY;
        if let Some(ref root) = self.root {
            Self::nn_search(root, query, &mut best_point, &mut best_dist_sq);
        }
        best_point.map(|p| (p, best_dist_sq))
    }

    /// Find all points within an axis-aligned bounding box.
    ///
    /// `min_corner[i] <= point[i] <= max_corner[i]` for all dimensions.
    pub fn query_region(&self, min_corner: &Point<D>, max_corner: &Point<D>) -> Vec<Point<D>> {
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::region_search(root, min_corner, max_corner, &mut results);
        }
        results
    }

    /// Find all points within `radius` of `center`.
    ///
    /// Returns points sorted by distance (ascending).
    pub fn query_radius(&self, center: &Point<D>, radius: f64) -> Vec<(Point<D>, f64)> {
        let radius_sq = radius * radius;
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            Self::radius_search(root, center, radius_sq, &mut results);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Find the k nearest neighbors to `query`.
    ///
    /// Returns up to `k` points with their squared distances, sorted ascending.
    pub fn k_nearest(&self, query: &Point<D>, k: usize) -> Result<Vec<(Point<D>, f64)>, MathError> {
        if k == 0 {
            return Err(MathError::InvalidParameter("k must be >= 1".into()));
        }
        let mut heap: Vec<(Point<D>, f64)> = Vec::with_capacity(k + 1);
        if let Some(ref root) = self.root {
            Self::knn_search(root, query, k, &mut heap);
        }
        heap.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(heap)
    }

    // ─── Internal helpers ────────────────────────────────────────────────

    fn insert_node(
        node: Option<Box<BspNode<D>>>,
        point: Point<D>,
        depth: usize,
    ) -> Box<BspNode<D>> {
        match node {
            None => Box::new(BspNode {
                point,
                split_dim: depth % D,
                left: None,
                right: None,
            }),
            Some(mut n) => {
                let dim = n.split_dim;
                if point[dim] < n.point[dim] {
                    n.left = Some(Self::insert_node(n.left.take(), point, depth + 1));
                } else {
                    n.right = Some(Self::insert_node(n.right.take(), point, depth + 1));
                }
                n
            }
        }
    }

    fn build_balanced(points: &mut [Point<D>], depth: usize) -> Option<Box<BspNode<D>>> {
        if points.is_empty() {
            return None;
        }
        let dim = depth % D;
        points.sort_by(|a, b| a[dim].partial_cmp(&b[dim]).unwrap_or(std::cmp::Ordering::Equal));
        let mid = points.len() / 2;

        let point = points[mid];
        let (left_slice, right_slice) = points.split_at_mut(mid);
        // right_slice[0] is the median point, skip it
        let right_slice = if right_slice.len() > 1 {
            &mut right_slice[1..]
        } else {
            &mut []
        };

        Some(Box::new(BspNode {
            point,
            split_dim: dim,
            left: Self::build_balanced(left_slice, depth + 1),
            right: Self::build_balanced(right_slice, depth + 1),
        }))
    }

    fn nn_search(
        node: &BspNode<D>,
        query: &Point<D>,
        best_point: &mut Option<Point<D>>,
        best_dist_sq: &mut f64,
    ) {
        let dist_sq = squared_distance(&node.point, query);
        if dist_sq < *best_dist_sq {
            *best_dist_sq = dist_sq;
            *best_point = Some(node.point);
        }

        let dim = node.split_dim;
        let diff = query[dim] - node.point[dim];
        let diff_sq = diff * diff;

        // Search the side of the split that query falls on first
        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::nn_search(child, query, best_point, best_dist_sq);
        }
        // Only search the other side if the splitting plane is closer than current best
        if diff_sq < *best_dist_sq {
            if let Some(ref child) = second {
                Self::nn_search(child, query, best_point, best_dist_sq);
            }
        }
    }

    fn region_search(
        node: &BspNode<D>,
        min_corner: &Point<D>,
        max_corner: &Point<D>,
        results: &mut Vec<Point<D>>,
    ) {
        // Check if this point is inside the region
        let inside = (0..D).all(|i| node.point[i] >= min_corner[i] && node.point[i] <= max_corner[i]);
        if inside {
            results.push(node.point);
        }

        let dim = node.split_dim;
        // Search left if min_corner is below the split
        if min_corner[dim] <= node.point[dim] {
            if let Some(ref left) = node.left {
                Self::region_search(left, min_corner, max_corner, results);
            }
        }
        // Search right if max_corner is above the split
        if max_corner[dim] >= node.point[dim] {
            if let Some(ref right) = node.right {
                Self::region_search(right, min_corner, max_corner, results);
            }
        }
    }

    fn radius_search(
        node: &BspNode<D>,
        center: &Point<D>,
        radius_sq: f64,
        results: &mut Vec<(Point<D>, f64)>,
    ) {
        let dist_sq = squared_distance(&node.point, center);
        if dist_sq <= radius_sq {
            results.push((node.point, dist_sq));
        }

        let dim = node.split_dim;
        let diff = center[dim] - node.point[dim];
        let diff_sq = diff * diff;

        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::radius_search(child, center, radius_sq, results);
        }
        if diff_sq <= radius_sq {
            if let Some(ref child) = second {
                Self::radius_search(child, center, radius_sq, results);
            }
        }
    }

    fn knn_search(
        node: &BspNode<D>,
        query: &Point<D>,
        k: usize,
        heap: &mut Vec<(Point<D>, f64)>,
    ) {
        let dist_sq = squared_distance(&node.point, query);

        // Insert if heap isn't full or this point is closer than the worst
        let worst_dist = if heap.len() >= k {
            heap.iter()
                .map(|(_, d)| *d)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            f64::INFINITY
        };

        if heap.len() < k || dist_sq < worst_dist {
            heap.push((node.point, dist_sq));
            if heap.len() > k {
                // Remove the farthest point
                let max_idx = heap
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap();
                heap.swap_remove(max_idx);
            }
        }

        let dim = node.split_dim;
        let diff = query[dim] - node.point[dim];
        let diff_sq = diff * diff;

        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            Self::knn_search(child, query, k, heap);
        }

        let current_worst = if heap.len() >= k {
            heap.iter()
                .map(|(_, d)| *d)
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            f64::INFINITY
        };

        if diff_sq < current_worst || heap.len() < k {
            if let Some(ref child) = second {
                Self::knn_search(child, query, k, heap);
            }
        }
    }
}

impl<const D: usize> Default for BspTree<D> {
    fn default() -> Self {
        Self::new()
    }
}

/// Squared Euclidean distance between two D-dimensional points.
fn squared_distance<const D: usize>(a: &Point<D>, b: &Point<D>) -> f64 {
    (0..D).map(|i| (a[i] - b[i]) * (a[i] - b[i])).sum()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_tree() {
        let tree = BspTree::<3>::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.nearest_neighbor(&[0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_insert_and_len() {
        let mut tree = BspTree::<2>::new();
        tree.insert([1.0, 2.0]);
        tree.insert([3.0, 4.0]);
        assert_eq!(tree.len(), 2);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_nearest_neighbor_2d() {
        let mut tree = BspTree::<2>::new();
        tree.insert([1.0, 1.0]);
        tree.insert([5.0, 5.0]);
        tree.insert([3.0, 3.0]);

        let (nearest, dist_sq) = tree.nearest_neighbor(&[2.0, 2.0]).unwrap();
        assert_eq!(nearest, [1.0, 1.0]);
        assert!((dist_sq - 2.0).abs() < 1e-10);

        let (nearest, dist_sq) = tree.nearest_neighbor(&[4.0, 4.0]).unwrap();
        // [3,3] and [5,5] are equidistant; accept either
        assert!(nearest == [3.0, 3.0] || nearest == [5.0, 5.0]);
        assert!((dist_sq - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbor_3d() {
        let mut tree = BspTree::<3>::new();
        tree.insert([0.0, 0.0, 0.0]);
        tree.insert([10.0, 10.0, 10.0]);
        tree.insert([1.0, 1.0, 1.0]);

        let (nearest, _) = tree.nearest_neighbor(&[0.5, 0.5, 0.5]).unwrap();
        assert_eq!(nearest, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_nearest_neighbor_exact_match() {
        let mut tree = BspTree::<2>::new();
        tree.insert([3.0, 4.0]);
        tree.insert([1.0, 2.0]);

        let (nearest, dist_sq) = tree.nearest_neighbor(&[3.0, 4.0]).unwrap();
        assert_eq!(nearest, [3.0, 4.0]);
        assert!(dist_sq < 1e-15);
    }

    #[test]
    fn test_region_query_2d() {
        let mut tree = BspTree::<2>::new();
        tree.insert([1.0, 2.0]);
        tree.insert([3.0, 4.0]);
        tree.insert([5.0, 1.0]);
        tree.insert([7.0, 8.0]);

        let results = tree.query_region(&[0.0, 0.0], &[4.0, 5.0]);
        assert_eq!(results.len(), 2);
        assert!(results.contains(&[1.0, 2.0]));
        assert!(results.contains(&[3.0, 4.0]));
    }

    #[test]
    fn test_region_query_empty() {
        let mut tree = BspTree::<2>::new();
        tree.insert([1.0, 2.0]);

        let results = tree.query_region(&[10.0, 10.0], &[20.0, 20.0]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_radius_query() {
        let mut tree = BspTree::<2>::new();
        tree.insert([0.0, 0.0]);
        tree.insert([1.0, 0.0]);
        tree.insert([0.0, 1.0]);
        tree.insert([10.0, 10.0]);

        let results = tree.query_radius(&[0.0, 0.0], 1.5);
        assert_eq!(results.len(), 3); // [0,0], [1,0], [0,1]
        // Should be sorted by distance
        assert_eq!(results[0].0, [0.0, 0.0]);
    }

    #[test]
    fn test_k_nearest() {
        let mut tree = BspTree::<2>::new();
        tree.insert([0.0, 0.0]);
        tree.insert([1.0, 0.0]);
        tree.insert([2.0, 0.0]);
        tree.insert([10.0, 0.0]);

        let knn = tree.k_nearest(&[0.5, 0.0], 2).unwrap();
        assert_eq!(knn.len(), 2);
        assert_eq!(knn[0].0, [0.0, 0.0]);
        assert_eq!(knn[1].0, [1.0, 0.0]);
    }

    #[test]
    fn test_k_nearest_zero_fails() {
        let tree = BspTree::<2>::new();
        assert!(tree.k_nearest(&[0.0, 0.0], 0).is_err());
    }

    #[test]
    fn test_from_points_balanced() {
        let points = vec![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 7.0],
        ];
        let tree = BspTree::from_points(points);
        assert_eq!(tree.len(), 7);

        let (nearest, _) = tree.nearest_neighbor(&[3.5, 3.5]).unwrap();
        assert!(nearest == [3.0, 3.0] || nearest == [4.0, 4.0]);
    }

    #[test]
    fn test_high_dimension_16d() {
        let mut tree = BspTree::<16>::new();
        let mut p1 = [0.0; 16];
        let mut p2 = [0.0; 16];
        p1[0] = 1.0;
        p2[0] = 2.0;
        tree.insert(p1);
        tree.insert(p2);

        let query = [0.5; 16];
        let (nearest, _) = tree.nearest_neighbor(&query).unwrap();
        // p1 is closer: dist = sqrt(0.25 + 15*0.25) = sqrt(4) = 2
        // p2: dist = sqrt(2.25 + 15*0.25) = sqrt(6) ≈ 2.45
        assert_eq!(nearest[0], 1.0);
    }

    #[test]
    fn test_single_point() {
        let mut tree = BspTree::<2>::new();
        tree.insert([42.0, 99.0]);

        let (nearest, dist_sq) = tree.nearest_neighbor(&[0.0, 0.0]).unwrap();
        assert_eq!(nearest, [42.0, 99.0]);
        assert!((dist_sq - (42.0_f64.powi(2) + 99.0_f64.powi(2))).abs() < 1e-10);
    }

    #[test]
    fn test_many_points_nn() {
        // Insert 100 points on a grid, verify NN correctness
        let mut tree = BspTree::<2>::new();
        for i in 0..10 {
            for j in 0..10 {
                tree.insert([i as f64, j as f64]);
            }
        }
        assert_eq!(tree.len(), 100);

        let (nearest, dist_sq) = tree.nearest_neighbor(&[4.1, 4.1]).unwrap();
        assert_eq!(nearest, [4.0, 4.0]);
        assert!(dist_sq < 0.1);
    }

    #[test]
    fn test_default() {
        let tree: BspTree<3> = BspTree::default();
        assert!(tree.is_empty());
    }
}
