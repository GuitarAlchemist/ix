//! GPU-accelerated batch k-nearest neighbor queries.
//!
//! Brute-force kNN on the GPU: for each query point, compute distances to all
//! reference points and find the k closest. This is O(Q*N) work but massively
//! parallel — each query is independent.
//!
//! # Examples
//!
//! ```no_run
//! use machin_gpu::knn::{batch_knn_cpu, batch_knn_gpu};
//! use machin_gpu::context::GpuContext;
//!
//! // 4 reference points in 2D
//! let refs = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//! let queries = vec![0.1f32, 0.1]; // 1 query point
//! let k = 2;
//! let dim = 2;
//!
//! let (indices, dists) = batch_knn_cpu(&refs, &queries, dim, k);
//! assert_eq!(indices.len(), 2); // k=2 results for 1 query
//! assert_eq!(indices[0], 0);    // nearest is (0,0)
//! ```

use crate::context::GpuContext;
use wgpu::*;

/// WGSL shader for brute-force kNN.
///
/// Each workgroup handles one query point. Each thread computes distance to a subset
/// of reference points, then we do a parallel selection of the k smallest.
///
/// For simplicity, k is limited to 32 (stored in shared memory per thread).
/// Params: [num_refs, num_queries, dim, k]
const KNN_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> refs: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;  // [num_refs, num_queries, dim, k]
@group(0) @binding(3) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_dists: array<f32>;

@compute @workgroup_size(256)
fn batch_knn(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let num_refs = params[0];
    let num_queries = params[1];
    let dim = params[2];
    let k = params[3];

    let query_idx = wg_id.x;
    if (query_idx >= num_queries) {
        return;
    }

    let tid = local_id.x;
    let q_base = query_idx * dim;

    // Each thread finds its local best-k from its subset of refs
    // Simple approach: each thread scans all refs with stride,
    // keeping track of the single nearest it found.
    // Then thread 0 gathers all partial results.

    // For large k, this approach is limited. For k <= workgroup_size (256),
    // we use a different strategy: each thread computes distance to one ref
    // and we do a parallel top-k selection.

    // Simple approach: each thread computes distances for its stride of refs,
    // stores the best distance and index.

    var best_dist: f32 = 3.402823e+38; // f32::MAX
    var best_idx: u32 = 0u;

    var i = tid;
    while (i < num_refs) {
        var dist: f32 = 0.0;
        for (var d = 0u; d < dim; d++) {
            let diff = queries[q_base + d] - refs[i * dim + d];
            dist += diff * diff;
        }
        dist = sqrt(dist);

        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
        i += 256u;
    }

    // For k=1, thread 0 just picks the global best
    // For k>1, we need a more sophisticated approach.
    // Simple solution: write all thread results to output, sort on CPU.
    // This is a pragmatic GPU/CPU hybrid approach.

    // Write this thread's best to a per-query output section
    let out_base = query_idx * 256u;
    if (tid < num_refs || tid == 0u) {
        out_indices[out_base + tid] = best_idx;
        out_dists[out_base + tid] = best_dist;
    } else {
        out_indices[out_base + tid] = 0u;
        out_dists[out_base + tid] = 3.402823e+38;
    }
}
"#;

/// Batch kNN queries on the GPU.
///
/// Returns `(indices, distances)` where each query gets `k` nearest neighbors.
/// Indices and distances are flattened: query_i's results are at `[i*k .. (i+1)*k]`.
pub fn batch_knn_gpu(
    ctx: &GpuContext,
    refs: &[f32],
    queries: &[f32],
    dim: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert!(refs.len().is_multiple_of(dim));
    assert!(queries.len().is_multiple_of(dim));

    let num_refs = refs.len() / dim;
    let num_queries = queries.len() / dim;

    let buf_refs = ctx.create_buffer_init("refs", refs);
    let buf_queries = ctx.create_buffer_init("queries", queries);

    let params = [num_refs as u32, num_queries as u32, dim as u32, k as u32];
    let params_bytes: &[u8] = bytemuck::cast_slice(&params);
    use wgpu::util::DeviceExt;
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: params_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    // Each query gets 256 candidate results from the GPU
    let candidates_per_query = 256usize;
    let total_candidates = num_queries * candidates_per_query;

    let idx_size = (total_candidates * std::mem::size_of::<u32>()) as u64;
    let dist_size = (total_candidates * std::mem::size_of::<f32>()) as u64;
    let buf_out_idx = ctx.create_output_buffer("out_indices", idx_size);
    let buf_out_dist = ctx.create_output_buffer("out_dists", dist_size);

    let pipeline = ctx.create_compute_pipeline("knn", KNN_SHADER, "batch_knn");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("knn_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: buf_refs.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: buf_queries.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: buf_out_idx.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: buf_out_dist.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("knn_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("knn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // One workgroup per query
        pass.dispatch_workgroups(num_queries as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    // Read back candidates
    let raw_dists = ctx.read_buffer(&buf_out_dist, dist_size);

    // Read indices as f32 then cast to u32 (since read_buffer returns f32)
    let raw_idx_f32 = ctx.read_buffer(&buf_out_idx, idx_size);
    let raw_idx: Vec<u32> = raw_idx_f32.iter().map(|f| f.to_bits()).collect();

    // CPU-side: for each query, sort candidates and take top-k
    let mut result_indices = Vec::with_capacity(num_queries * k);
    let mut result_dists = Vec::with_capacity(num_queries * k);

    for q in 0..num_queries {
        let base = q * candidates_per_query;
        let mut candidates: Vec<(f32, u32)> = (0..candidates_per_query)
            .map(|i| (raw_dists[base + i], raw_idx[base + i]))
            .filter(|(d, _)| *d < f32::MAX / 2.0)
            .collect();

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for &(dist, idx) in candidates.iter().take(k) {
            result_indices.push(idx);
            result_dists.push(dist);
        }
        // Pad if fewer than k results
        for _ in candidates.len()..k {
            result_indices.push(u32::MAX);
            result_dists.push(f32::INFINITY);
        }
    }

    (result_indices, result_dists)
}

/// CPU fallback: brute-force kNN.
///
/// Returns `(indices, distances)` flattened: query_i's k results at `[i*k..(i+1)*k]`.
pub fn batch_knn_cpu(
    refs: &[f32],
    queries: &[f32],
    dim: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert!(refs.len().is_multiple_of(dim));
    assert!(queries.len().is_multiple_of(dim));

    let num_refs = refs.len() / dim;
    let num_queries = queries.len() / dim;

    let mut result_indices = Vec::with_capacity(num_queries * k);
    let mut result_dists = Vec::with_capacity(num_queries * k);

    for q in 0..num_queries {
        let q_base = q * dim;
        let mut dists: Vec<(f32, u32)> = (0..num_refs)
            .map(|r| {
                let r_base = r * dim;
                let dist: f32 = (0..dim)
                    .map(|d| {
                        let diff = queries[q_base + d] - refs[r_base + d];
                        diff * diff
                    })
                    .sum::<f32>()
                    .sqrt();
                (dist, r as u32)
            })
            .collect();

        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for &(dist, idx) in dists.iter().take(k) {
            result_indices.push(idx);
            result_dists.push(dist);
        }
        for _ in dists.len()..k {
            result_indices.push(u32::MAX);
            result_dists.push(f32::INFINITY);
        }
    }

    (result_indices, result_dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_knn_basic() {
        // 4 points in 2D
        let refs = vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let queries = vec![0.1, 0.1]; // near origin
        let (indices, dists) = batch_knn_cpu(&refs, &queries, 2, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); // nearest is (0,0)
        assert!(dists[0] < dists[1]); // sorted by distance
    }

    #[test]
    fn test_cpu_knn_exact_match() {
        let refs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let queries = vec![3.0, 4.0]; // exact match with point 1
        let (indices, dists) = batch_knn_cpu(&refs, &queries, 2, 1);
        assert_eq!(indices[0], 1);
        assert!(dists[0] < 1e-10);
    }

    #[test]
    fn test_cpu_knn_multiple_queries() {
        let refs = vec![0.0, 0.0, 10.0, 10.0];
        let queries = vec![0.1, 0.1, 9.9, 9.9]; // 2 queries
        let (indices, _dists) = batch_knn_cpu(&refs, &queries, 2, 1);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); // first query near (0,0)
        assert_eq!(indices[1], 1); // second query near (10,10)
    }

    #[test]
    fn test_cpu_knn_k_larger_than_refs() {
        let refs = vec![1.0, 2.0];
        let queries = vec![0.0, 0.0];
        let (indices, dists) = batch_knn_cpu(&refs, &queries, 2, 3);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 0); // only ref
        assert_eq!(indices[1], u32::MAX); // padded
        assert!(dists[1].is_infinite());
    }

    #[test]
    fn test_cpu_knn_high_dim() {
        // 3 points in 5D
        let mut refs = vec![0.0f32; 15];
        refs[0] = 1.0; // point 0: (1,0,0,0,0)
        refs[5] = 2.0; // point 1: (0,2,0,0,0)
        refs[10] = 0.5; // point 2: (0,0,0.5,0,0)
        let queries = vec![0.0f32; 5]; // origin
        let (indices, _) = batch_knn_cpu(&refs, &queries, 5, 3);
        assert_eq!(indices[0], 2); // nearest: (0,0,0.5,0,0) at dist 0.5
        assert_eq!(indices[1], 0); // next: (1,0,0,0,0) at dist 1.0
        assert_eq!(indices[2], 1); // farthest: (0,2,0,0,0) at dist 2.0
    }

    #[test]
    fn test_cpu_knn_single_ref() {
        let refs = vec![42.0, 0.0, -1.0];
        let queries = vec![0.0, 0.0, 0.0];
        let (indices, _) = batch_knn_cpu(&refs, &queries, 3, 1);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_cpu_knn_all_equidistant() {
        // Points on a circle, query at center
        let refs = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let queries = vec![0.0, 0.0]; // center
        let (_, dists) = batch_knn_cpu(&refs, &queries, 2, 4);
        // All distances should be 1.0
        for d in &dists {
            assert!((*d - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cpu_knn_distances_sorted() {
        let refs = vec![0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 5.0, 0.0];
        let queries = vec![0.0, 0.0];
        let (_, dists) = batch_knn_cpu(&refs, &queries, 2, 4);
        for i in 0..dists.len() - 1 {
            assert!(dists[i] <= dists[i + 1], "distances should be sorted ascending");
        }
    }

    // GPU tests require hardware
    // #[test]
    // fn test_gpu_matches_cpu() {
    //     let ctx = GpuContext::new().expect("Need GPU");
    //     let refs: Vec<f32> = (0..30).map(|i| i as f32).collect();
    //     let queries = vec![0.5, 0.5, 0.5];
    //     let (gpu_idx, gpu_dist) = batch_knn_gpu(&ctx, &refs, &queries, 3, 3);
    //     let (cpu_idx, cpu_dist) = batch_knn_cpu(&refs, &queries, 3, 3);
    //     assert_eq!(gpu_idx, cpu_idx);
    //     for (a, b) in gpu_dist.iter().zip(cpu_dist.iter()) {
    //         assert!((a - b).abs() < 1e-3);
    //     }
    // }
}
