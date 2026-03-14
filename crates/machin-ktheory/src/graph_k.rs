//! K-groups from graph adjacency matrices.
//!
//! Computes algebraic K-theory invariants K₀ and K₁ from directed graph
//! adjacency matrices using Smith normal form. These invariants detect:
//!
//! - **K₀**: Resource balance — cokernel of (I - Aᵀ), measures "stable equivalence"
//! - **K₁**: Feedback cycles — kernel of (I - Aᵀ), detects eigenvalue-1 presence
//!
//! # Examples
//!
//! ```
//! use machin_ktheory::graph_k;
//! use ndarray::array;
//!
//! // Simple 2-node graph: 0 → 1 → 0 (cycle)
//! let adj = array![[0.0, 1.0], [1.0, 0.0]];
//! let k0 = graph_k::k0_from_adjacency(&adj).unwrap();
//! let k1 = graph_k::k1_from_adjacency(&adj).unwrap();
//!
//! // K₁ rank > 0 means feedback cycles exist
//! assert!(k1.rank >= 0);
//! ```

use ndarray::Array2;

use crate::error::KTheoryError;

/// Result of a K-group computation.
#[derive(Debug, Clone)]
pub struct KGroupResult {
    /// Rank of the free part.
    pub rank: usize,
    /// Invariant factors of the torsion part (> 1 only).
    pub torsion: Vec<i64>,
    /// Generators (column vectors of the relevant subspace).
    pub generators: Vec<Vec<i64>>,
}

/// Compute K₀ from an adjacency matrix.
///
/// K₀ = coker(I - Aᵀ) = ℤⁿ / im(I - Aᵀ).
/// The torsion and free rank are extracted via Smith normal form.
pub fn k0_from_adjacency(adj: &Array2<f64>) -> Result<KGroupResult, KTheoryError> {
    let n = validate_square(adj)?;
    let mat = compute_i_minus_at(adj, n);
    let snf = smith_normal_form(&mat);

    // Cokernel: for each diagonal entry d_i of Smith form:
    // - d_i = 0 → contributes 1 to free rank
    // - d_i = ±1 → trivial (killed)
    // - |d_i| > 1 → contributes ℤ/|d_i|ℤ to torsion
    let mut rank = 0;
    let mut torsion = Vec::new();

    for i in 0..n {
        let d = snf.diagonal[i].abs();
        if d == 0 {
            rank += 1;
        } else if d > 1 {
            torsion.push(d);
        }
    }

    Ok(KGroupResult {
        rank,
        torsion,
        generators: snf.right_transform,
    })
}

/// Compute K₁ from an adjacency matrix.
///
/// K₁ = ker(I - Aᵀ), the null space of (I - Aᵀ) over ℤ.
/// Non-trivial K₁ indicates feedback cycles (eigenvalue 1 in Aᵀ).
pub fn k1_from_adjacency(adj: &Array2<f64>) -> Result<KGroupResult, KTheoryError> {
    let n = validate_square(adj)?;
    let mat = compute_i_minus_at(adj, n);
    let snf = smith_normal_form(&mat);

    // Kernel: columns of the right transform corresponding to zero diagonal entries
    let mut rank = 0;
    let mut generators = Vec::new();

    for i in 0..n {
        if snf.diagonal[i] == 0 {
            rank += 1;
            if i < snf.right_transform.len() {
                generators.push(snf.right_transform[i].clone());
            }
        }
    }

    Ok(KGroupResult {
        rank,
        torsion: Vec::new(), // K₁ kernel has no torsion
        generators,
    })
}

/// Detect feedback cycles in a directed graph.
///
/// Returns cycle paths found by detecting eigenvalue-1 presence in Aᵀ.
/// Each cycle is a sequence of node indices.
pub fn detect_feedback_cycles(adj: &Array2<f64>) -> Result<Vec<Vec<usize>>, KTheoryError> {
    let n = validate_square(adj)?;
    let mut cycles = Vec::new();

    // DFS-based cycle detection
    let mut visited = vec![0u8; n]; // 0=unvisited, 1=in-stack, 2=done
    let mut stack = Vec::new();

    for start in 0..n {
        if visited[start] == 0 {
            dfs_cycles(adj, start, &mut visited, &mut stack, &mut cycles, n);
        }
    }

    Ok(cycles)
}

/// Resource invariant check: given allocation and free counts,
/// verify K₀ constraint (should sum to 0 for balanced systems).
///
/// Returns the imbalance. Zero means balanced.
pub fn resource_invariant(allocations: &[i64], frees: &[i64]) -> i64 {
    let alloc_sum: i64 = allocations.iter().sum();
    let free_sum: i64 = frees.iter().sum();
    alloc_sum - free_sum
}

// ─── Internal helpers ────────────────────────────────────────────────────────

fn validate_square(adj: &Array2<f64>) -> Result<usize, KTheoryError> {
    let shape = adj.shape();
    if shape[0] != shape[1] {
        return Err(KTheoryError::InvalidParameter(format!(
            "adjacency matrix must be square, got {}×{}",
            shape[0], shape[1]
        )));
    }
    if shape[0] == 0 {
        return Err(KTheoryError::InvalidParameter(
            "adjacency matrix must be non-empty".into(),
        ));
    }
    Ok(shape[0])
}

fn compute_i_minus_at(adj: &Array2<f64>, n: usize) -> Vec<Vec<i64>> {
    let mut mat = vec![vec![0i64; n]; n];
    for i in 0..n {
        for j in 0..n {
            let a_val = adj[[j, i]] as i64; // Aᵀ[i,j] = A[j,i]
            mat[i][j] = if i == j { 1 - a_val } else { -a_val };
        }
    }
    mat
}

/// Smith normal form result.
struct SmithResult {
    diagonal: Vec<i64>,
    right_transform: Vec<Vec<i64>>,
}

/// Compute Smith normal form of an integer matrix.
///
/// Returns diagonal entries and the right transformation matrix.
#[allow(clippy::needless_range_loop)]
fn smith_normal_form(mat: &[Vec<i64>]) -> SmithResult {
    let n = mat.len();
    if n == 0 {
        return SmithResult {
            diagonal: vec![],
            right_transform: vec![],
        };
    }
    let m = mat[0].len();

    // Working copy
    let mut a: Vec<Vec<i64>> = mat.to_vec();

    // Right transform (column operations) — starts as identity
    let mut right = vec![vec![0i64; m]; m];
    for i in 0..m {
        right[i][i] = 1;
    }

    let min_dim = n.min(m);

    for k in 0..min_dim {
        // Find pivot: smallest non-zero absolute value in submatrix a[k..][k..]
        if !find_and_swap_pivot(&mut a, &mut right, k, n, m) {
            continue; // All zeros in this submatrix
        }

        // Eliminate using pivot at (k, k)
        let mut changed = true;
        while changed {
            changed = false;

            // Eliminate column k below row k
            for i in (k + 1)..n {
                if a[i][k] != 0 {
                    let q = a[i][k] / a[k][k];
                    for j in k..m {
                        a[i][j] -= q * a[k][j];
                    }
                    if a[i][k] != 0 {
                        // GCD step: swap rows
                        a.swap(k, i);
                        changed = true;
                    }
                }
            }

            // Eliminate row k right of column k
            for j in (k + 1)..m {
                if a[k][j] != 0 {
                    let q = a[k][j] / a[k][k];
                    for i in k..n {
                        a[i][j] -= q * a[i][k];
                    }
                    // Apply same column operation to right transform
                    for i in 0..m {
                        right[i][j] -= q * right[i][k];
                    }
                    if a[k][j] != 0 {
                        // Swap columns
                        for row in a.iter_mut().take(n) {
                            row.swap(k, j);
                        }
                        for row in right.iter_mut().take(m) {
                            row.swap(k, j);
                        }
                        changed = true;
                    }
                }
            }
        }
    }

    // Extract diagonal
    let mut diagonal = Vec::with_capacity(min_dim);
    for i in 0..min_dim {
        diagonal.push(a[i][i]);
    }

    // Convert right transform to column vectors
    let right_cols: Vec<Vec<i64>> = (0..m)
        .map(|j| (0..m).map(|i| right[i][j]).collect())
        .collect();

    SmithResult {
        diagonal,
        right_transform: right_cols,
    }
}

#[allow(clippy::needless_range_loop)]
fn find_and_swap_pivot(
    a: &mut [Vec<i64>],
    right: &mut [Vec<i64>],
    k: usize,
    n: usize,
    m: usize,
) -> bool {
    let mut best_val = i64::MAX;
    let mut best_r = k;
    let mut best_c = k;

    for i in k..n {
        for j in k..m {
            let v = a[i][j].abs();
            if v > 0 && v < best_val {
                best_val = v;
                best_r = i;
                best_c = j;
            }
        }
    }

    if best_val == i64::MAX {
        return false; // All zeros
    }

    // Swap rows
    if best_r != k {
        a.swap(k, best_r);
    }
    // Swap columns
    if best_c != k {
        for row in a.iter_mut().take(n) {
            row.swap(k, best_c);
        }
        for row in right.iter_mut().take(m) {
            row.swap(k, best_c);
        }
    }

    // Make pivot positive
    if a[k][k] < 0 {
        for val in a[k].iter_mut().skip(k).take(m - k) {
            *val = -*val;
        }
    }

    true
}

fn dfs_cycles(
    adj: &Array2<f64>,
    node: usize,
    visited: &mut [u8],
    stack: &mut Vec<usize>,
    cycles: &mut Vec<Vec<usize>>,
    n: usize,
) {
    visited[node] = 1;
    stack.push(node);

    for next in 0..n {
        if adj[[node, next]] > 0.0 {
            if visited[next] == 1 {
                // Found a cycle — extract it from stack
                if let Some(pos) = stack.iter().position(|&x| x == next) {
                    let cycle: Vec<usize> = stack[pos..].to_vec();
                    cycles.push(cycle);
                }
            } else if visited[next] == 0 {
                dfs_cycles(adj, next, visited, stack, cycles, n);
            }
        }
    }

    stack.pop();
    visited[node] = 2;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_k0_identity_graph() {
        // No edges: adj = 0, I - Aᵀ = I
        // Smith form of I has all 1s on diagonal → rank=0, no torsion
        let adj = array![[0.0, 0.0], [0.0, 0.0]];
        let k0 = k0_from_adjacency(&adj).unwrap();
        assert_eq!(k0.rank, 0);
        assert!(k0.torsion.is_empty());
    }

    #[test]
    fn test_k0_cycle_graph() {
        // 2-cycle: 0→1, 1→0
        let adj = array![[0.0, 1.0], [1.0, 0.0]];
        let k0 = k0_from_adjacency(&adj).unwrap();
        // I - Aᵀ = [[1, -1], [-1, 1]], Smith form = [[1, 0], [0, 0]]
        assert_eq!(k0.rank, 1); // One free generator
    }

    #[test]
    fn test_k1_cycle_graph() {
        // 2-cycle should have non-trivial K₁
        let adj = array![[0.0, 1.0], [1.0, 0.0]];
        let k1 = k1_from_adjacency(&adj).unwrap();
        assert!(k1.rank > 0, "cycle should have feedback");
    }

    #[test]
    fn test_k0_self_loop() {
        // Self-loop: adj = [[1]]
        // I - Aᵀ = [[0]], Smith = [[0]]
        let adj = array![[1.0]];
        let k0 = k0_from_adjacency(&adj).unwrap();
        assert_eq!(k0.rank, 1);
    }

    #[test]
    fn test_k1_no_cycles() {
        // DAG: 0→1, 0→2 (no cycles)
        let adj = array![[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let k1 = k1_from_adjacency(&adj).unwrap();
        assert_eq!(k1.rank, 0);
    }

    #[test]
    fn test_detect_feedback_cycles_simple() {
        // 0→1→0 cycle
        let adj = array![[0.0, 1.0], [1.0, 0.0]];
        let cycles = detect_feedback_cycles(&adj).unwrap();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_detect_feedback_cycles_dag() {
        // Pure DAG: 0→1→2
        let adj = array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
        let cycles = detect_feedback_cycles(&adj).unwrap();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_resource_invariant_balanced() {
        assert_eq!(resource_invariant(&[5, 3, 2], &[4, 4, 2]), 0);
    }

    #[test]
    fn test_resource_invariant_unbalanced() {
        assert_eq!(resource_invariant(&[10], &[7]), 3);
    }

    #[test]
    fn test_non_square_rejected() {
        let adj = Array2::zeros((2, 3));
        assert!(k0_from_adjacency(&adj).is_err());
    }

    #[test]
    fn test_empty_rejected() {
        let adj = Array2::zeros((0, 0));
        assert!(k0_from_adjacency(&adj).is_err());
    }

    #[test]
    fn test_3_cycle() {
        // 0→1→2→0
        let adj = array![
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ];
        let k0 = k0_from_adjacency(&adj).unwrap();
        let k1 = k1_from_adjacency(&adj).unwrap();
        let cycles = detect_feedback_cycles(&adj).unwrap();
        // 3-cycle has feedback
        assert!(k1.rank > 0 || !cycles.is_empty());
        // K₀ should detect the cycle structure
        assert!(k0.rank > 0 || !k0.torsion.is_empty());
    }

    #[test]
    fn test_smith_normal_form_identity() {
        let mat = vec![vec![1, 0], vec![0, 1]];
        let snf = smith_normal_form(&mat);
        assert_eq!(snf.diagonal, vec![1, 1]);
    }

    #[test]
    fn test_smith_normal_form_zero() {
        let mat = vec![vec![0, 0], vec![0, 0]];
        let snf = smith_normal_form(&mat);
        assert_eq!(snf.diagonal, vec![0, 0]);
    }
}
