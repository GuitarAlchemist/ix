/// Binary Space Partitioning tree for arbitrary-dimensional point sets.
///
/// Supports k-nearest neighbor and radius queries.
///
/// A BSP tree node.
#[derive(Debug, Clone)]
pub enum BspNode {
    /// Leaf containing point indices.
    Leaf { indices: Vec<usize> },
    /// Internal split node.
    Split {
        axis: usize,
        value: f64,
        left: Box<BspNode>,
        right: Box<BspNode>,
    },
}

impl BspNode {
    /// Build a BSP tree from a set of points.
    ///
    /// Points are represented as `Vec<Vec<f64>>` for arbitrary dimensionality.
    /// `max_leaf_size` controls when to stop splitting.
    pub fn build(points: &[Vec<f64>], max_leaf_size: usize) -> BspNode {
        let indices: Vec<usize> = (0..points.len()).collect();
        build_recursive(points, &indices, max_leaf_size)
    }

    /// Find the k nearest neighbors to `point`.
    ///
    /// Returns a vector of `(point_index, distance)` sorted by distance.
    pub fn query_nearest(&self, points: &[Vec<f64>], query: &[f64], k: usize) -> Vec<(usize, f64)> {
        let mut best: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
        knn_search(self, points, query, k, &mut best);
        best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        best.truncate(k);
        best
    }

    /// Find all points within `radius` of `point`.
    ///
    /// Returns a vector of point indices.
    pub fn query_radius(&self, points: &[Vec<f64>], query: &[f64], radius: f64) -> Vec<usize> {
        let mut result = Vec::new();
        radius_search(self, points, query, radius, &mut result);
        result
    }
}

fn build_recursive(points: &[Vec<f64>], indices: &[usize], max_leaf_size: usize) -> BspNode {
    if indices.len() <= max_leaf_size || indices.is_empty() {
        return BspNode::Leaf {
            indices: indices.to_vec(),
        };
    }

    let dim = points[indices[0]].len();

    // Choose the axis with the greatest spread
    let mut best_axis = 0;
    let mut best_spread = f64::NEG_INFINITY;

    {
        #![allow(clippy::needless_range_loop)]
        for axis in 0..dim {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &idx in indices {
                let v = points[idx][axis];
                if v < min_val {
                    min_val = v;
                }
                if v > max_val {
                    max_val = v;
                }
            }
            let spread = max_val - min_val;
            if spread > best_spread {
                best_spread = spread;
                best_axis = axis;
            }
        }
    }

    // If no spread at all, make a leaf
    if best_spread < 1e-15 {
        return BspNode::Leaf {
            indices: indices.to_vec(),
        };
    }

    // Find median value along best axis
    let mut values: Vec<f64> = indices.iter().map(|&i| points[i][best_axis]).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = values[values.len() / 2];

    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for &idx in indices {
        if points[idx][best_axis] <= median {
            left_indices.push(idx);
        } else {
            right_indices.push(idx);
        }
    }

    // Avoid infinite recursion: if all points went to one side, force a split
    if left_indices.is_empty() || right_indices.is_empty() {
        // Split in half by index order
        let mid = indices.len() / 2;
        let mut sorted_indices = indices.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            points[a][best_axis]
                .partial_cmp(&points[b][best_axis])
                .unwrap()
        });
        left_indices = sorted_indices[..mid].to_vec();
        right_indices = sorted_indices[mid..].to_vec();

        // If still degenerate, leaf it
        if left_indices.is_empty() || right_indices.is_empty() {
            return BspNode::Leaf {
                indices: indices.to_vec(),
            };
        }
    }

    let left = build_recursive(points, &left_indices, max_leaf_size);
    let right = build_recursive(points, &right_indices, max_leaf_size);

    BspNode::Split {
        axis: best_axis,
        value: median,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn worst_dist(best: &[(usize, f64)], k: usize) -> f64 {
    if best.len() < k {
        f64::INFINITY
    } else {
        best.iter()
            .map(|(_, d)| *d)
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

fn knn_search(
    node: &BspNode,
    points: &[Vec<f64>],
    query: &[f64],
    k: usize,
    best: &mut Vec<(usize, f64)>,
) {
    match node {
        BspNode::Leaf { indices } => {
            for &idx in indices {
                let d = euclidean_dist(query, &points[idx]);
                if best.len() < k || d < worst_dist(best, k) {
                    best.push((idx, d));
                    best.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    if best.len() > k {
                        best.truncate(k);
                    }
                }
            }
        }
        BspNode::Split {
            axis,
            value,
            left,
            right,
        } => {
            let goes_left = query[*axis] <= *value;
            let (first, second) = if goes_left {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };

            knn_search(first, points, query, k, best);

            // Check if we need to search the other side
            let plane_dist = (query[*axis] - *value).abs();
            if plane_dist < worst_dist(best, k) {
                knn_search(second, points, query, k, best);
            }
        }
    }
}

fn radius_search(
    node: &BspNode,
    points: &[Vec<f64>],
    query: &[f64],
    radius: f64,
    result: &mut Vec<usize>,
) {
    match node {
        BspNode::Leaf { indices } => {
            for &idx in indices {
                let d = euclidean_dist(query, &points[idx]);
                if d <= radius {
                    result.push(idx);
                }
            }
        }
        BspNode::Split {
            axis,
            value,
            left,
            right,
        } => {
            let plane_dist = query[*axis] - *value;

            if plane_dist <= 0.0 {
                // Query is on the left side
                radius_search(left, points, query, radius, result);
                if plane_dist.abs() <= radius {
                    radius_search(right, points, query, radius, result);
                }
            } else {
                // Query is on the right side
                radius_search(right, points, query, radius, result);
                if plane_dist.abs() <= radius {
                    radius_search(left, points, query, radius, result);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_2d_points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![0.1, 0.1],
        ]
    }

    #[test]
    fn test_build_tree() {
        let points = make_2d_points();
        let tree = BspNode::build(&points, 2);
        // Just verify it doesn't panic and produces a valid tree
        match &tree {
            BspNode::Leaf { .. } => {}  // small enough for leaf
            BspNode::Split { .. } => {} // split is fine
        }
    }

    #[test]
    fn test_nearest_neighbor_correctness() {
        let points = make_2d_points();
        let tree = BspNode::build(&points, 2);

        let query = [0.05, 0.05];
        let result = tree.query_nearest(&points, &query, 1);
        assert_eq!(result.len(), 1);

        // Brute force nearest
        let mut brute: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, euclidean_dist(&query, p)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        assert_eq!(
            result[0].0, brute[0].0,
            "BSP nearest: idx={}, brute force: idx={}",
            result[0].0, brute[0].0
        );
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let points = make_2d_points();
        let tree = BspNode::build(&points, 2);

        let query = [0.5, 0.5];
        let k = 3;
        let result = tree.query_nearest(&points, &query, k);
        assert_eq!(result.len(), k);

        // Brute force k-nearest
        let mut brute: Vec<(usize, f64)> = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, euclidean_dist(&query, p)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut bsp_indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        let mut brute_indices: Vec<usize> = brute[..k].iter().map(|(i, _)| *i).collect();
        bsp_indices.sort();
        brute_indices.sort();
        assert_eq!(
            bsp_indices, brute_indices,
            "BSP k-NN indices don't match brute force"
        );
    }

    #[test]
    fn test_radius_query() {
        let points = make_2d_points();
        let tree = BspNode::build(&points, 2);

        let query = [0.0, 0.0];
        let radius = 1.1;

        let mut result = tree.query_radius(&points, &query, radius);
        result.sort();

        // Brute force
        let mut brute: Vec<usize> = points
            .iter()
            .enumerate()
            .filter(|(_, p)| euclidean_dist(&query, p) <= radius)
            .map(|(i, _)| i)
            .collect();
        brute.sort();

        assert_eq!(
            result, brute,
            "BSP radius query: {:?}, brute force: {:?}",
            result, brute
        );
    }

    #[test]
    fn test_empty_points() {
        let points: Vec<Vec<f64>> = vec![];
        let tree = BspNode::build(&points, 2);
        match tree {
            BspNode::Leaf { indices } => assert!(indices.is_empty()),
            _ => panic!("empty points should produce a leaf"),
        }
    }

    #[test]
    fn test_single_point() {
        let points = vec![vec![1.0, 2.0, 3.0]];
        let tree = BspNode::build(&points, 2);
        let result = tree.query_nearest(&points, &[1.0, 2.0, 3.0], 1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0);
        assert!((result[0].1).abs() < 1e-10);
    }

    #[test]
    fn test_high_dimensional() {
        // Test with 16D points (sedenion space)
        let mut points = Vec::new();
        for i in 0..50 {
            let p: Vec<f64> = (0..16).map(|d| (i * 7 + d * 3) as f64 % 10.0).collect();
            points.push(p);
        }
        let tree = BspNode::build(&points, 4);

        let query: Vec<f64> = vec![5.0; 16];
        let result = tree.query_nearest(&points, &query, 3);
        assert_eq!(result.len(), 3);

        // Verify sorted by distance
        for w in result.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }

        // Verify all returned indices are valid
        for (idx, dist) in &result {
            assert!(*idx < points.len());
            let actual = euclidean_dist(&query, &points[*idx]);
            assert!((actual - dist).abs() < 1e-10, "distance mismatch");
        }

        // For low-D (3D) we verify exact brute-force match
        let points_3d: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                vec![
                    (i * 3) as f64 % 7.0,
                    (i * 5) as f64 % 11.0,
                    (i * 7) as f64 % 13.0,
                ]
            })
            .collect();
        let tree_3d = BspNode::build(&points_3d, 4);
        let query_3d = vec![3.0, 5.0, 6.0];
        let result_3d = tree_3d.query_nearest(&points_3d, &query_3d, 3);

        let mut brute: Vec<(usize, f64)> = points_3d
            .iter()
            .enumerate()
            .map(|(i, p)| (i, euclidean_dist(&query_3d, p)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut bsp_idx: Vec<usize> = result_3d.iter().map(|(i, _)| *i).collect();
        let mut brute_idx: Vec<usize> = brute[..3].iter().map(|(i, _)| *i).collect();
        bsp_idx.sort();
        brute_idx.sort();
        assert_eq!(bsp_idx, brute_idx);
    }
}
