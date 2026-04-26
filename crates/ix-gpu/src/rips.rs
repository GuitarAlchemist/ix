//! GPU-accelerated Vietoris-Rips complex construction.
//!
//! Uses the pairwise distance matrix (computed on GPU) to build a Rips complex.
//! The GPU computes the N×N distance matrix, then the CPU extracts simplices
//! at a given radius threshold.
//!
//! # Examples
//!
//! ```no_run
//! use ix_gpu::rips::{rips_edges_cpu, rips_edges_gpu};
//! use ix_gpu::context::GpuContext;
//!
//! // 3 points in 2D: close triangle
//! let points = vec![0.0f32, 0.0, 1.0, 0.0, 0.5, 0.866];
//! let edges = rips_edges_cpu(&points, 2, 1.5);
//! assert!(!edges.is_empty()); // edges within radius 1.5
//! ```

use crate::context::GpuContext;
use crate::distance::{pairwise_distance_cpu, pairwise_distance_gpu};

/// Build Rips 1-skeleton (edges) on the GPU.
///
/// Returns pairs `(i, j)` where `i < j` and `dist(point_i, point_j) <= radius`.
/// Uses GPU-accelerated distance matrix computation.
pub fn rips_edges_gpu(
    ctx: &GpuContext,
    points: &[f32],
    dim: usize,
    radius: f32,
) -> Vec<(usize, usize)> {
    let dist_matrix = pairwise_distance_gpu(ctx, points, dim);
    let n = points.len() / dim;
    extract_edges(&dist_matrix, n, radius)
}

/// Build Rips 1-skeleton (edges) on the CPU.
pub fn rips_edges_cpu(points: &[f32], dim: usize, radius: f32) -> Vec<(usize, usize)> {
    let dist_matrix = pairwise_distance_cpu(points, dim);
    let n = points.len() / dim;
    extract_edges(&dist_matrix, n, radius)
}

/// Build full Rips complex up to `max_dim` simplices on GPU.
///
/// Returns simplices as vectors of vertex indices, along with their birth radii.
/// Uses the GPU for distance matrix, then CPU for simplex enumeration.
pub fn rips_complex_gpu(
    ctx: &GpuContext,
    points: &[f32],
    dim: usize,
    max_simplex_dim: usize,
    max_radius: f32,
) -> Vec<(Vec<usize>, f32)> {
    let dist_matrix = pairwise_distance_gpu(ctx, points, dim);
    let n = points.len() / dim;
    build_rips_complex(&dist_matrix, n, max_simplex_dim, max_radius)
}

/// Build full Rips complex on CPU.
pub fn rips_complex_cpu(
    points: &[f32],
    dim: usize,
    max_simplex_dim: usize,
    max_radius: f32,
) -> Vec<(Vec<usize>, f32)> {
    let dist_matrix = pairwise_distance_cpu(points, dim);
    let n = points.len() / dim;
    build_rips_complex(&dist_matrix, n, max_simplex_dim, max_radius)
}

fn extract_edges(dist_matrix: &[f32], n: usize, radius: f32) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if dist_matrix[i * n + j] <= radius {
                edges.push((i, j));
            }
        }
    }
    edges
}

fn build_rips_complex(
    dist_matrix: &[f32],
    n: usize,
    max_simplex_dim: usize,
    max_radius: f32,
) -> Vec<(Vec<usize>, f32)> {
    let mut simplices: Vec<(Vec<usize>, f32)> = Vec::new();

    // 0-simplices (vertices) — born at 0
    for i in 0..n {
        simplices.push((vec![i], 0.0));
    }

    if max_simplex_dim == 0 {
        return simplices;
    }

    // 1-simplices (edges)
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist_matrix[i * n + j];
            if d <= max_radius {
                edges.push((i, j, d));
                simplices.push((vec![i, j], d));
            }
        }
    }

    if max_simplex_dim == 1 {
        return simplices;
    }

    // Higher simplices via clique enumeration
    // Build adjacency for clique finding
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(i, j, _) in &edges {
        adj[i].push(j);
        adj[j].push(i);
    }

    // Enumerate cliques using Bron-Kerbosch style expansion
    // For each edge, try to extend to triangles, tetrahedra, etc.
    if max_simplex_dim >= 2 {
        // Triangles
        for &(i, j, _) in &edges {
            for &k in &adj[i] {
                if k > j && adj[j].contains(&k) {
                    let d_ij = dist_matrix[i * n + j];
                    let d_ik = dist_matrix[i * n + k];
                    let d_jk = dist_matrix[j * n + k];
                    let birth = d_ij.max(d_ik).max(d_jk);
                    if birth <= max_radius {
                        simplices.push((vec![i, j, k], birth));

                        // Tetrahedra
                        if max_simplex_dim >= 3 {
                            for &l in &adj[i] {
                                if l > k && adj[j].contains(&l) && adj[k].contains(&l) {
                                    let d_il = dist_matrix[i * n + l];
                                    let d_jl = dist_matrix[j * n + l];
                                    let d_kl = dist_matrix[k * n + l];
                                    let birth4 = birth.max(d_il).max(d_jl).max(d_kl);
                                    if birth4 <= max_radius {
                                        simplices.push((vec![i, j, k, l], birth4));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    simplices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_rips_edges_triangle() {
        // Equilateral triangle with side ~1
        let points = vec![0.0f32, 0.0, 1.0, 0.0, 0.5, 0.866];
        let edges = rips_edges_cpu(&points, 2, 1.5);
        assert_eq!(edges.len(), 3, "equilateral triangle should have 3 edges");
    }

    #[test]
    fn test_cpu_rips_edges_no_edges() {
        // Points far apart
        let points = vec![0.0f32, 0.0, 100.0, 100.0];
        let edges = rips_edges_cpu(&points, 2, 1.0);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_cpu_rips_complex_vertices_only() {
        let points = vec![0.0f32, 0.0, 1.0, 1.0];
        let complex = rips_complex_cpu(&points, 2, 0, 10.0);
        assert_eq!(complex.len(), 2); // just vertices
    }

    #[test]
    fn test_cpu_rips_complex_with_triangle() {
        let points = vec![0.0f32, 0.0, 1.0, 0.0, 0.5, 0.866];
        let complex = rips_complex_cpu(&points, 2, 2, 2.0);
        // Should have 3 vertices + 3 edges + 1 triangle = 7
        let vertices = complex.iter().filter(|(s, _)| s.len() == 1).count();
        let edges = complex.iter().filter(|(s, _)| s.len() == 2).count();
        let triangles = complex.iter().filter(|(s, _)| s.len() == 3).count();
        assert_eq!(vertices, 3);
        assert_eq!(edges, 3);
        assert_eq!(triangles, 1);
    }

    #[test]
    fn test_cpu_rips_complex_birth_times() {
        let points = vec![0.0f32, 0.0, 1.0, 0.0];
        let complex = rips_complex_cpu(&points, 2, 1, 2.0);
        // Edge (0,1) should be born at distance 1.0
        let edge = complex.iter().find(|(s, _)| s.len() == 2).unwrap();
        assert!((edge.1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_rips_radius_filter() {
        let points = vec![0.0f32, 0.0, 0.5, 0.0, 10.0, 0.0];
        let edges = rips_edges_cpu(&points, 2, 1.0);
        // Only (0,1) should be within radius 1.0
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (0, 1));
    }

    #[test]
    fn test_cpu_rips_tetrahedron() {
        // 4 points close together
        let points = vec![
            0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.866, 0.0, 0.5, 0.289, 0.816,
        ];
        let complex = rips_complex_cpu(&points, 3, 3, 2.0);
        let tets = complex.iter().filter(|(s, _)| s.len() == 4).count();
        assert_eq!(tets, 1, "regular tetrahedron should produce 1 3-simplex");
    }

    // GPU tests require hardware
    // #[test]
    // fn test_gpu_rips_matches_cpu() {
    //     let ctx = GpuContext::new().expect("Need GPU");
    //     let points = vec![0.0f32, 0.0, 1.0, 0.0, 0.5, 0.866];
    //     let gpu = rips_edges_gpu(&ctx, &points, 2, 1.5);
    //     let cpu = rips_edges_cpu(&points, 2, 1.5);
    //     assert_eq!(gpu, cpu);
    // }
}
