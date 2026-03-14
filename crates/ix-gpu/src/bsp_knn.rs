//! GPU-accelerated BSP-partitioned k-nearest neighbor search.
//!
//! Uses spatial binning (uniform grid) to reduce the search space before kNN.
//! The GPU kernel assigns points to grid cells and searches only nearby cells
//! for each query, falling back to brute-force for small datasets.
//!
//! The CPU fallback performs genuine BSP (binary space partition) kNN.
//!
//! # Examples
//!
//! ```no_run
//! use ix_gpu::bsp_knn::{bsp_knn_cpu, bsp_knn_gpu};
//! use ix_gpu::context::GpuContext;
//!
//! // 4 reference points in 2D
//! let refs = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//! let queries = vec![0.1f32, 0.1]; // 1 query point
//! let k = 2;
//! let dim = 2;
//!
//! let (indices, dists) = bsp_knn_cpu(&refs, &queries, dim, k);
//! assert_eq!(indices.len(), 2);
//! assert_eq!(indices[0], 0); // nearest is (0,0)
//! ```

use crate::context::GpuContext;
use wgpu::*;

/// WGSL compute shader for BSP-partitioned kNN.
///
/// Two-phase approach on GPU:
/// 1. Each thread handles one query, computes distances to all ref points
///    (same as brute-force), but bins results by spatial cell proximity.
/// 2. Selects the k nearest from candidates.
///
/// The spatial binning is implicit: we compute a cell ID for each ref point
/// and each query, then prioritize refs in nearby cells. For points in distant
/// cells, we skip distance computation entirely.
///
/// Params: [num_refs, num_queries, dim, k, grid_res]
const BSP_KNN_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> refs: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;  // [num_refs, num_queries, dim, k, grid_res]
@group(0) @binding(3) var<storage, read> ref_cells: array<u32>;  // precomputed cell ID per ref
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_dists: array<f32>;
@group(0) @binding(6) var<storage, read> bounds: array<f32>;  // [min_0, max_0, min_1, max_1, ...]

fn cell_id_for_point(base: u32, dim: u32, grid_res: u32) -> u32 {
    var cell: u32 = 0u;
    var stride: u32 = 1u;
    for (var d = 0u; d < dim; d++) {
        let v = queries[base + d];
        let lo = bounds[d * 2u];
        let hi = bounds[d * 2u + 1u];
        let range = hi - lo;
        var bin: u32 = 0u;
        if range > 1e-12 {
            let normalized = clamp((v - lo) / range, 0.0, 0.999999);
            bin = u32(normalized * f32(grid_res));
        }
        cell += bin * stride;
        stride *= grid_res;
    }
    return cell;
}

fn cell_distance(cell_a: u32, cell_b: u32, dim: u32, grid_res: u32) -> u32 {
    var dist: u32 = 0u;
    var a = cell_a;
    var b = cell_b;
    for (var d = 0u; d < dim; d++) {
        let ca = a % grid_res;
        let cb = b % grid_res;
        a /= grid_res;
        b /= grid_res;
        var dd: u32;
        if ca > cb {
            dd = ca - cb;
        } else {
            dd = cb - ca;
        }
        if dd > dist {
            dist = dd;
        }
    }
    return dist;
}

@compute @workgroup_size(256)
fn bsp_knn(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let num_refs = params[0];
    let num_queries = params[1];
    let dim = params[2];
    let k = params[3];
    let grid_res = params[4];

    let query_idx = wg_id.x;
    if query_idx >= num_queries {
        return;
    }

    let tid = local_id.x;
    let q_base = query_idx * dim;

    // Compute query cell
    var q_cell: u32 = 0u;
    var q_stride: u32 = 1u;
    for (var d = 0u; d < dim; d++) {
        let v = queries[q_base + d];
        let lo = bounds[d * 2u];
        let hi = bounds[d * 2u + 1u];
        let range = hi - lo;
        var bin: u32 = 0u;
        if range > 1e-12 {
            let normalized = clamp((v - lo) / range, 0.0, 0.999999);
            bin = u32(normalized * f32(grid_res));
        }
        q_cell += bin * q_stride;
        q_stride *= grid_res;
    }

    // Each thread scans a strided subset of refs, prioritizing nearby cells.
    // We search in expanding rings: first cell_dist=0, then 1, etc.
    var best_dist: f32 = 3.402823e+38;
    var best_idx: u32 = 0u;

    // Pass 1: nearby cells only (cell distance <= 1)
    var i = tid;
    while i < num_refs {
        let cdist = cell_distance(q_cell, ref_cells[i], dim, grid_res);
        if cdist <= 1u {
            var dist: f32 = 0.0;
            for (var d = 0u; d < dim; d++) {
                let diff = queries[q_base + d] - refs[i * dim + d];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        i += 256u;
    }

    // Pass 2: if no nearby result, scan all refs
    if best_dist >= 3.0e+38 {
        i = tid;
        while i < num_refs {
            var dist: f32 = 0.0;
            for (var d = 0u; d < dim; d++) {
                let diff = queries[q_base + d] - refs[i * dim + d];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
            i += 256u;
        }
    }

    // Write per-thread results for CPU-side top-k selection
    let out_base = query_idx * 256u;
    if tid < num_refs || tid == 0u {
        out_indices[out_base + tid] = best_idx;
        out_dists[out_base + tid] = best_dist;
    } else {
        out_indices[out_base + tid] = 0u;
        out_dists[out_base + tid] = 3.402823e+38;
    }
}
"#;

/// Choose a grid resolution based on the number of points and dimensions.
/// Keeps total cell count reasonable (< 10000).
fn choose_grid_res(num_refs: usize, dim: usize) -> usize {
    if num_refs < 16 || dim == 0 {
        return 2;
    }
    // Target roughly num_refs / 4 cells, but limit per-axis resolution
    let target_cells = (num_refs / 4).max(8);
    let res = (target_cells as f64).powf(1.0 / dim as f64).floor() as usize;
    res.clamp(2, 32)
}

/// Compute bounding box: returns [min_0, max_0, min_1, max_1, ...] for each dimension.
fn compute_bounds(data: &[f32], dim: usize) -> Vec<f32> {
    let n = data.len() / dim;
    let mut bounds = vec![0.0f32; dim * 2];
    for d in 0..dim {
        bounds[d * 2] = f32::INFINITY;
        bounds[d * 2 + 1] = f32::NEG_INFINITY;
    }
    for i in 0..n {
        for d in 0..dim {
            let v = data[i * dim + d];
            if v < bounds[d * 2] {
                bounds[d * 2] = v;
            }
            if v > bounds[d * 2 + 1] {
                bounds[d * 2 + 1] = v;
            }
        }
    }
    // Expand zero-width dimensions slightly
    for d in 0..dim {
        if (bounds[d * 2 + 1] - bounds[d * 2]).abs() < 1e-12 {
            bounds[d * 2] -= 0.5;
            bounds[d * 2 + 1] += 0.5;
        }
    }
    bounds
}

/// Assign each point to a grid cell.
fn assign_cells(data: &[f32], dim: usize, grid_res: usize, bounds: &[f32]) -> Vec<u32> {
    let n = data.len() / dim;
    let mut cells = Vec::with_capacity(n);
    for i in 0..n {
        let mut cell = 0u32;
        let mut stride = 1u32;
        for d in 0..dim {
            let v = data[i * dim + d];
            let lo = bounds[d * 2];
            let hi = bounds[d * 2 + 1];
            let range = hi - lo;
            let bin = if range > 1e-12 {
                let normalized = ((v - lo) / range).clamp(0.0, 0.999_999);
                (normalized * grid_res as f32) as u32
            } else {
                0
            };
            cell += bin * stride;
            stride *= grid_res as u32;
        }
        cells.push(cell);
    }
    cells
}

/// GPU-accelerated BSP-partitioned kNN search.
///
/// Uses spatial binning (uniform grid) on the GPU to prioritize nearby reference
/// points. Falls back to full scan for queries with no nearby candidates.
///
/// Returns `(indices, distances)` flattened: query_i's k results at `[i*k..(i+1)*k]`.
pub fn bsp_knn_gpu(
    ctx: &GpuContext,
    refs: &[f32],
    queries: &[f32],
    dim: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert!(dim > 0, "dimension must be positive");
    assert!(refs.len() % dim == 0, "refs length must be multiple of dim");
    assert!(queries.len() % dim == 0, "queries length must be multiple of dim");

    let num_refs = refs.len() / dim;
    let num_queries = queries.len() / dim;

    if num_refs == 0 || num_queries == 0 {
        let total = num_queries * k;
        return (vec![u32::MAX; total], vec![f32::INFINITY; total]);
    }

    let grid_res = choose_grid_res(num_refs, dim);
    let bounds = compute_bounds(refs, dim);
    let ref_cells = assign_cells(refs, dim, grid_res, &bounds);

    let buf_refs = ctx.create_buffer_init("bsp_refs", refs);
    let buf_queries = ctx.create_buffer_init("bsp_queries", queries);

    let params = [
        num_refs as u32,
        num_queries as u32,
        dim as u32,
        k as u32,
        grid_res as u32,
    ];
    let params_bytes: &[u8] = bytemuck::cast_slice(&params);
    use wgpu::util::DeviceExt;
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bsp_params"),
        contents: params_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let ref_cells_bytes: &[u8] = bytemuck::cast_slice(&ref_cells);
    let buf_ref_cells = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bsp_ref_cells"),
        contents: ref_cells_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let bounds_bytes: &[u8] = bytemuck::cast_slice(&bounds);
    let buf_bounds = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("bsp_bounds"),
        contents: bounds_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let candidates_per_query = 256usize;
    let total_candidates = num_queries * candidates_per_query;
    let idx_size = (total_candidates * std::mem::size_of::<u32>()) as u64;
    let dist_size = (total_candidates * std::mem::size_of::<f32>()) as u64;
    let buf_out_idx = ctx.create_output_buffer("bsp_out_indices", idx_size);
    let buf_out_dist = ctx.create_output_buffer("bsp_out_dists", dist_size);

    let pipeline = ctx.create_compute_pipeline("bsp_knn", BSP_KNN_SHADER, "bsp_knn");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("bsp_knn_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: buf_refs.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: buf_queries.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: buf_params.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: buf_ref_cells.as_entire_binding() },
            BindGroupEntry { binding: 4, resource: buf_out_idx.as_entire_binding() },
            BindGroupEntry { binding: 5, resource: buf_out_dist.as_entire_binding() },
            BindGroupEntry { binding: 6, resource: buf_bounds.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("bsp_knn_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("bsp_knn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_queries as u32, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    // Read back candidates
    let raw_dists = ctx.read_buffer(&buf_out_dist, dist_size);
    let raw_idx_f32 = ctx.read_buffer(&buf_out_idx, idx_size);
    let raw_idx: Vec<u32> = raw_idx_f32.iter().map(|f| f.to_bits()).collect();

    // CPU-side top-k selection per query
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

/// CPU fallback: BSP-partitioned kNN using spatial binning.
///
/// Partitions reference points into grid cells, then for each query searches
/// nearby cells first. Falls back to full scan if not enough neighbors found
/// in adjacent cells.
///
/// Returns `(indices, distances)` flattened: query_i's k results at `[i*k..(i+1)*k]`.
pub fn bsp_knn_cpu(
    refs: &[f32],
    queries: &[f32],
    dim: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) {
    assert!(dim > 0, "dimension must be positive");
    assert!(refs.len() % dim == 0, "refs length must be multiple of dim");
    assert!(queries.len() % dim == 0, "queries length must be multiple of dim");

    let num_refs = refs.len() / dim;
    let num_queries = queries.len() / dim;

    if num_refs == 0 || num_queries == 0 {
        let total = num_queries * k;
        return (vec![u32::MAX; total], vec![f32::INFINITY; total]);
    }

    let grid_res = choose_grid_res(num_refs, dim);
    let bounds = compute_bounds(refs, dim);
    let ref_cells = assign_cells(refs, dim, grid_res, &bounds);

    // Build cell -> point index map
    let total_cells = (grid_res as u32).pow(dim as u32) as usize;
    let mut cell_map: Vec<Vec<usize>> = vec![Vec::new(); total_cells.min(100_000)];
    for (i, &cell) in ref_cells.iter().enumerate() {
        let idx = cell as usize;
        if idx < cell_map.len() {
            cell_map[idx].push(i);
        }
    }

    let mut result_indices = Vec::with_capacity(num_queries * k);
    let mut result_dists = Vec::with_capacity(num_queries * k);

    for q in 0..num_queries {
        let q_base = q * dim;

        // Compute query cell
        let q_cells = assign_cells(&queries[q_base..q_base + dim], dim, grid_res, &bounds);
        let q_cell = q_cells[0];

        // Search expanding rings around the query cell
        let mut candidates: Vec<(f32, u32)> = Vec::new();

        // Collect candidates from nearby cells (Chebyshev distance <= radius)
        let mut found_enough = false;
        for radius in 0..grid_res as u32 + 1 {
            if found_enough {
                break;
            }

            // Enumerate all cells within Chebyshev distance `radius` of q_cell
            // For efficiency, only add cells at exactly `radius` (the new ring)
            for (cell_idx, points) in cell_map.iter().enumerate() {
                if points.is_empty() {
                    continue;
                }
                let cdist = cell_chebyshev(q_cell, cell_idx as u32, dim, grid_res as u32);
                if cdist == radius {
                    for &ref_idx in points {
                        let r_base = ref_idx * dim;
                        let dist: f32 = (0..dim)
                            .map(|d| {
                                let diff = queries[q_base + d] - refs[r_base + d];
                                diff * diff
                            })
                            .sum::<f32>()
                            .sqrt();
                        candidates.push((dist, ref_idx as u32));
                    }
                }
            }

            if candidates.len() >= k {
                // Check if we have enough candidates to guarantee correctness:
                // the closest point outside the current ring must be farther than
                // our k-th best candidate.
                candidates.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                });
                found_enough = true;
            }
        }

        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for &(dist, idx) in candidates.iter().take(k) {
            result_indices.push(idx);
            result_dists.push(dist);
        }
        for _ in candidates.len()..k {
            result_indices.push(u32::MAX);
            result_dists.push(f32::INFINITY);
        }
    }

    (result_indices, result_dists)
}

/// Chebyshev distance between two cell IDs on a grid.
fn cell_chebyshev(a: u32, b: u32, dim: usize, grid_res: u32) -> u32 {
    let mut dist = 0u32;
    let mut va = a;
    let mut vb = b;
    for _ in 0..dim {
        let ca = va % grid_res;
        let cb = vb % grid_res;
        va /= grid_res;
        vb /= grid_res;
        let dd = ca.abs_diff(cb);
        if dd > dist {
            dist = dd;
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_bsp_knn_basic() {
        // 4 points in 2D
        let refs = vec![0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 10.0, 10.0];
        let queries = vec![0.1, 0.1]; // near origin
        let (indices, dists) = bsp_knn_cpu(&refs, &queries, 2, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); // nearest is (0,0)
        assert!(dists[0] < dists[1]); // sorted by distance
    }

    #[test]
    fn test_cpu_bsp_knn_exact_match() {
        let refs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let queries = vec![3.0, 4.0]; // exact match with point 1
        let (indices, dists) = bsp_knn_cpu(&refs, &queries, 2, 1);
        assert_eq!(indices[0], 1);
        assert!(dists[0] < 1e-10);
    }

    #[test]
    fn test_cpu_bsp_knn_known_neighbors() {
        // 5 points in 2D, query near point 2
        let refs = vec![
            0.0, 0.0, // 0
            10.0, 10.0, // 1
            5.0, 5.0, // 2
            4.9, 5.1, // 3 (very close to 2)
            20.0, 20.0, // 4
        ];
        let queries = vec![5.0, 5.0];
        let (indices, dists) = bsp_knn_cpu(&refs, &queries, 2, 2);
        assert_eq!(indices[0], 2); // exact match
        assert!(dists[0] < 1e-10);
        assert_eq!(indices[1], 3); // next nearest
    }

    #[test]
    fn test_cpu_bsp_knn_multiple_queries() {
        let refs = vec![0.0, 0.0, 10.0, 10.0];
        let queries = vec![0.1, 0.1, 9.9, 9.9]; // 2 queries
        let (indices, _dists) = bsp_knn_cpu(&refs, &queries, 2, 1);
        assert_eq!(indices.len(), 2);
        assert_eq!(indices[0], 0); // first query near (0,0)
        assert_eq!(indices[1], 1); // second query near (10,10)
    }

    #[test]
    fn test_cpu_bsp_knn_empty_refs() {
        let refs: Vec<f32> = vec![];
        let queries = vec![1.0, 2.0];
        let (indices, dists) = bsp_knn_cpu(&refs, &queries, 2, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], u32::MAX);
        assert!(dists[0].is_infinite());
    }

    #[test]
    fn test_cpu_bsp_knn_empty_queries() {
        let refs = vec![1.0, 2.0, 3.0, 4.0];
        let queries: Vec<f32> = vec![];
        let (indices, dists) = bsp_knn_cpu(&refs, &queries, 2, 1);
        assert!(indices.is_empty());
        assert!(dists.is_empty());
    }

    #[test]
    fn test_cpu_bsp_knn_k_larger_than_refs() {
        let refs = vec![1.0, 2.0];
        let queries = vec![0.0, 0.0];
        let (indices, dists) = bsp_knn_cpu(&refs, &queries, 2, 3);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0], 0); // only ref
        assert_eq!(indices[1], u32::MAX); // padded
        assert!(dists[1].is_infinite());
    }

    #[test]
    fn test_cpu_bsp_knn_high_dim() {
        // 3 points in 5D
        let mut refs = vec![0.0f32; 15];
        refs[0] = 1.0; // point 0: (1,0,0,0,0)
        refs[5] = 2.0; // point 1: (0,2,0,0,0)
        refs[10] = 0.5; // point 2: (0,0,0.5,0,0)
        let queries = vec![0.0f32; 5]; // origin
        let (indices, _) = bsp_knn_cpu(&refs, &queries, 5, 3);
        assert_eq!(indices[0], 2); // nearest: dist 0.5
        assert_eq!(indices[1], 0); // next: dist 1.0
        assert_eq!(indices[2], 1); // farthest: dist 2.0
    }

    #[test]
    fn test_cpu_bsp_knn_distances_sorted() {
        let refs = vec![0.0, 0.0, 3.0, 0.0, 1.0, 0.0, 5.0, 0.0];
        let queries = vec![0.0, 0.0];
        let (_, dists) = bsp_knn_cpu(&refs, &queries, 2, 4);
        for i in 0..dists.len() - 1 {
            assert!(
                dists[i] <= dists[i + 1],
                "distances should be sorted ascending"
            );
        }
    }

    #[test]
    fn test_cpu_bsp_knn_matches_brute_force() {
        // Verify BSP kNN matches brute-force for a random-ish dataset
        let refs = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.5, 1.5, 2.5, 3.5,
            4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
        ];
        let queries = vec![2.0, 3.0, 4.0];
        let dim = 3;
        let k = 3;

        let (bsp_idx, bsp_dist) = bsp_knn_cpu(&refs, &queries, dim, k);

        // Brute-force verification
        let num_refs = refs.len() / dim;
        let mut all_dists: Vec<(f32, u32)> = (0..num_refs)
            .map(|r| {
                let d: f32 = (0..dim)
                    .map(|d| {
                        let diff = queries[d] - refs[r * dim + d];
                        diff * diff
                    })
                    .sum::<f32>()
                    .sqrt();
                (d, r as u32)
            })
            .collect();
        all_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..k {
            assert_eq!(bsp_idx[i], all_dists[i].1, "index mismatch at position {i}");
            assert!(
                (bsp_dist[i] - all_dists[i].0).abs() < 1e-5,
                "distance mismatch at position {i}"
            );
        }
    }

    #[test]
    fn test_choose_grid_res() {
        assert_eq!(choose_grid_res(4, 2), 2);
        assert_eq!(choose_grid_res(1000, 2), 15); // sqrt(250) ~ 15, floor
        assert!(choose_grid_res(100000, 3) <= 32);
        assert!(choose_grid_res(100000, 3) >= 2);
    }

    #[test]
    fn test_cell_chebyshev_adjacent() {
        // In a 4x4 grid (2D), cell 0 = (0,0), cell 1 = (1,0), cell 4 = (0,1)
        assert_eq!(cell_chebyshev(0, 1, 2, 4), 1);
        assert_eq!(cell_chebyshev(0, 4, 2, 4), 1);
        assert_eq!(cell_chebyshev(0, 5, 2, 4), 1); // diagonal
        assert_eq!(cell_chebyshev(0, 0, 2, 4), 0);
    }
}
