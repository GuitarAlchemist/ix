//! Batch vector operations — similarity matrix, top-k search.
//!
//! Designed for the "Pro-Tip": compute entire similarity matrices in one GPU pass
//! rather than one-pair-at-a-time.

use crate::context::GpuContext;
use crate::similarity::cosine_similarity_cpu;

/// Compute a full cosine similarity matrix for a set of vectors.
///
/// Given N vectors of dimension D, returns an N×N similarity matrix.
/// Uses GPU if context is provided, otherwise CPU fallback.
pub fn similarity_matrix(ctx: Option<&GpuContext>, vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let _n = vectors.len();

    match ctx {
        Some(gpu) => similarity_matrix_gpu(gpu, vectors),
        None => similarity_matrix_cpu(vectors),
    }
}

/// CPU fallback for similarity matrix.
pub fn similarity_matrix_cpu(vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let mut matrix = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n {
            let sim = cosine_similarity_cpu(&vectors[i], &vectors[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }

    matrix
}

/// GPU-accelerated similarity matrix.
///
/// Flattens all vectors into a single buffer, then dispatches a compute shader
/// that computes all pairwise similarities in parallel.
pub fn similarity_matrix_gpu(ctx: &GpuContext, vectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = vectors.len();
    if n == 0 {
        return vec![];
    }
    let dim = vectors[0].len();

    // Flatten vectors into a single buffer
    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

    // Pre-compute norms
    let norms: Vec<f32> = vectors
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();

    // For now, use the GPU matmul approach: similarity = V × V^T, then normalize
    // V is N×D, V^T is D×N, result is N×N
    let flat_ref = &flat;
    let v_t: Vec<f32> = (0..dim)
        .flat_map(|d| (0..n).map(move |i| flat_ref[i * dim + d]))
        .collect();

    let dot_matrix = crate::matmul::matmul_gpu(ctx, &flat, &v_t, n, dim, n);

    // Normalize by norms to get cosine similarity
    let mut matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            let denom = norms[i] * norms[j];
            matrix[i][j] = if denom > 1e-10 {
                dot_matrix[i * n + j] / denom
            } else {
                0.0
            };
        }
    }

    matrix
}

/// Find top-k most similar vectors to a query.
///
/// Returns Vec<(index, similarity)> sorted by similarity descending.
pub fn top_k_similar(
    _ctx: Option<&GpuContext>,
    query: &[f32],
    corpus: &[Vec<f32>],
    k: usize,
) -> Vec<(usize, f32)> {
    let similarities: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, vec)| {
            let sim = cosine_similarity_cpu(query, vec);
            (i, sim)
        })
        .collect();

    let mut sorted = similarities;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    sorted.truncate(k);
    sorted
}

/// Batch top-k: for multiple queries against a corpus.
///
/// This is the high-performance path — computes the entire query×corpus
/// similarity matrix on the GPU in one pass.
pub fn batch_top_k(
    ctx: Option<&GpuContext>,
    queries: &[Vec<f32>],
    corpus: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<(usize, f32)>> {
    let nq = queries.len();
    let nc = corpus.len();
    if nq == 0 || nc == 0 {
        return vec![vec![]; nq];
    }
    let dim = queries[0].len();

    // Flatten
    let q_flat: Vec<f32> = queries.iter().flat_map(|v| v.iter().copied()).collect();
    let c_flat: Vec<f32> = corpus.iter().flat_map(|v| v.iter().copied()).collect();

    // Compute norms
    let q_norms: Vec<f32> = queries
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();
    let c_norms: Vec<f32> = corpus
        .iter()
        .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
        .collect();

    // Transpose corpus: D×NC
    let c_flat_ref = &c_flat;
    let c_t: Vec<f32> = (0..dim)
        .flat_map(|d| (0..nc).map(move |i| c_flat_ref[i * dim + d]))
        .collect();

    // Dot product matrix: NQ × NC
    let dot_matrix = match ctx {
        Some(gpu) => crate::matmul::matmul_gpu(gpu, &q_flat, &c_t, nq, dim, nc),
        None => crate::matmul::matmul_cpu(&q_flat, &c_t, nq, dim, nc),
    };

    // For each query, compute cosine similarities and find top-k
    (0..nq)
        .map(|qi| {
            let mut sims: Vec<(usize, f32)> = (0..nc)
                .map(|ci| {
                    let denom = q_norms[qi] * c_norms[ci];
                    let sim = if denom > 1e-10 {
                        dot_matrix[qi * nc + ci] / denom
                    } else {
                        0.0
                    };
                    (ci, sim)
                })
                .collect();

            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sims.truncate(k);
            sims
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similarity_matrix_cpu() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];

        let matrix = similarity_matrix_cpu(&vectors);
        assert!((matrix[0][0] - 1.0).abs() < 1e-5);
        assert!(matrix[0][1].abs() < 1e-5); // Orthogonal
        assert!(matrix[0][2] > 0.5); // Partially aligned
    }

    #[test]
    fn test_top_k() {
        let query = vec![1.0, 0.0, 0.0];
        let corpus = vec![
            vec![1.0, 0.0, 0.0],  // Identical
            vec![0.9, 0.1, 0.0],  // Very similar
            vec![0.0, 1.0, 0.0],  // Orthogonal
            vec![-1.0, 0.0, 0.0], // Opposite
        ];

        let results = top_k_similar(None, &query, &corpus, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Most similar
        assert_eq!(results[1].0, 1); // Second most
    }

    #[test]
    fn test_batch_top_k_cpu() {
        let queries = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];

        let results = batch_top_k(None, &queries, &corpus, 1);
        assert_eq!(results[0][0].0, 0); // Query [1,0] most similar to corpus [1,0]
        assert_eq!(results[1][0].0, 1); // Query [0,1] most similar to corpus [0,1]
    }

    #[test]
    fn test_similarity_matrix_empty() {
        let matrix = similarity_matrix_cpu(&[]);
        assert!(matrix.is_empty());
    }

    #[test]
    fn test_similarity_matrix_single_vector() {
        let vectors = vec![vec![1.0, 2.0, 3.0]];
        let matrix = similarity_matrix_cpu(&vectors);
        assert_eq!(matrix.len(), 1);
        assert!((matrix[0][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_similarity_matrix_symmetry() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let matrix = similarity_matrix_cpu(&vectors);
        for (i, row) in matrix.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!(
                    (*value - matrix[j][i]).abs() < 1e-10,
                    "similarity matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_similarity_matrix_diagonal_is_one() {
        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![-1.0, 0.0, 1.0],
        ];
        let matrix = similarity_matrix_cpu(&vectors);
        for (i, row) in matrix.iter().enumerate() {
            assert!(
                (row[i] - 1.0).abs() < 1e-5,
                "diagonal should be 1.0 (self-similarity)"
            );
        }
    }

    #[test]
    fn test_top_k_all() {
        let query = vec![1.0, 0.0];
        let corpus = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let results = top_k_similar(None, &query, &corpus, 10);
        assert_eq!(results.len(), 2); // truncated to corpus size
    }

    #[test]
    fn test_top_k_ordering() {
        let query = vec![1.0, 0.0, 0.0];
        let corpus = vec![
            vec![0.0, 1.0, 0.0],  // orthogonal (sim ≈ 0)
            vec![1.0, 0.1, 0.0],  // very similar
            vec![-1.0, 0.0, 0.0], // opposite (sim ≈ -1)
            vec![0.9, 0.3, 0.0],  // similar
        ];
        let results = top_k_similar(None, &query, &corpus, 4);
        // Should be sorted descending by similarity
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 >= results[i + 1].1,
                "results should be sorted descending by similarity"
            );
        }
    }

    #[test]
    fn test_batch_top_k_empty_queries() {
        let corpus = vec![vec![1.0, 0.0]];
        let results = batch_top_k(None, &[], &corpus, 1);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_top_k_empty_corpus() {
        let queries = vec![vec![1.0, 0.0]];
        let results = batch_top_k(None, &queries, &[], 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    #[test]
    fn test_batch_top_k_multiple_k() {
        let queries = vec![vec![1.0, 0.0]];
        let corpus = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.0, 1.0]];
        let results = batch_top_k(None, &queries, &corpus, 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[0][0].0, 0); // most similar
    }

    #[test]
    fn test_similarity_matrix_with_context_none() {
        let vectors = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let matrix = similarity_matrix(None, &vectors);
        assert!((matrix[0][1]).abs() < 1e-5); // orthogonal
        assert!((matrix[0][0] - 1.0).abs() < 1e-5); // self
    }
}
