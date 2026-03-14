//! GPU-accelerated pairwise distance matrix.
//!
//! Computes the N×N Euclidean distance matrix from N points using a 2D WGSL compute shader.
//! Each thread computes one element of the distance matrix.
//!
//! # Examples
//!
//! ```no_run
//! use ix_gpu::context::GpuContext;
//! use ix_gpu::distance::{pairwise_distance_gpu, pairwise_distance_cpu};
//!
//! // 3 points in 2D: (0,0), (3,4), (1,0)
//! let points = vec![0.0f32, 0.0,  3.0, 4.0,  1.0, 0.0];
//! let dim = 2;
//!
//! let dist = pairwise_distance_cpu(&points, dim);
//! assert_eq!(dist.len(), 9); // 3×3 matrix
//! assert!((dist[1] - 5.0).abs() < 1e-5); // d(0,1) = 5
//! ```

use crate::context::GpuContext;
use wgpu::*;

/// WGSL shader for pairwise Euclidean distance matrix.
///
/// 2D dispatch: each thread (i, j) computes dist(point_i, point_j).
/// Points are stored flat: [x0,y0,z0, x1,y1,z1, ...] with stride = dim.
/// Params buffer: [num_points, dim].
const DISTANCE_MATRIX_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<u32>;  // [num_points, dim]
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;

@compute @workgroup_size(16, 16)
fn pairwise_distance(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let n = params[0];
    let dim = params[1];
    let i = global_id.x;
    let j = global_id.y;

    if (i >= n || j >= n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var d = 0u; d < dim; d++) {
        let diff = points[i * dim + d] - points[j * dim + d];
        sum += diff * diff;
    }

    distances[i * n + j] = sqrt(sum);
}
"#;

/// Compute N×N pairwise Euclidean distance matrix on the GPU.
///
/// - `points`: flat array [x0,y0,..., x1,y1,...] with `dim` values per point
/// - `dim`: dimensionality of each point
///
/// Returns flat row-major N×N distance matrix.
pub fn pairwise_distance_gpu(ctx: &GpuContext, points: &[f32], dim: usize) -> Vec<f32> {
    assert!(points.len() % dim == 0, "Points length must be multiple of dim");
    let n = points.len() / dim;

    let buf_points = ctx.create_buffer_init("points", points);

    // Pack params as u32
    let params = [n as u32, dim as u32];
    let params_bytes: &[u8] = bytemuck::cast_slice(&params);
    use wgpu::util::DeviceExt;
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: params_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    });

    let output_size = (n * n * std::mem::size_of::<f32>()) as u64;
    let buf_output = ctx.create_output_buffer("distances", output_size);

    let pipeline = ctx.create_compute_pipeline("dist_matrix", DISTANCE_MATRIX_SHADER, "pairwise_distance");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("dist_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: buf_points.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: buf_params.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: buf_output.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("dist_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dist_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (n as u32).div_ceil(16);
        let wg_y = (n as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.read_buffer(&buf_output, output_size)
}

/// CPU fallback: compute N×N pairwise Euclidean distance matrix.
pub fn pairwise_distance_cpu(points: &[f32], dim: usize) -> Vec<f32> {
    assert!(points.len() % dim == 0, "Points length must be multiple of dim");
    let n = points.len() / dim;
    let mut dist = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for d in 0..dim {
                let diff = points[i * dim + d] - points[j * dim + d];
                sum += diff * diff;
            }
            dist[i * n + j] = sum.sqrt();
        }
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_self_distance_zero() {
        let points = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dist = pairwise_distance_cpu(&points, 3);
        // Diagonal should be zero
        assert!(dist[0].abs() < 1e-10); // d(0,0)
        assert!(dist[3].abs() < 1e-10); // d(1,1)
    }

    #[test]
    fn test_cpu_known_distance() {
        // (0,0) and (3,4) → distance = 5
        let points = vec![0.0, 0.0, 3.0, 4.0];
        let dist = pairwise_distance_cpu(&points, 2);
        assert!((dist[1] - 5.0).abs() < 1e-5); // d(0,1)
        assert!((dist[2] - 5.0).abs() < 1e-5); // d(1,0) symmetric
    }

    #[test]
    fn test_cpu_symmetry() {
        let points = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dist = pairwise_distance_cpu(&points, 3);
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                assert!((dist[i * n + j] - dist[j * n + i]).abs() < 1e-10,
                    "Distance matrix should be symmetric");
            }
        }
    }

    #[test]
    fn test_cpu_triangle_inequality() {
        let points = vec![0.0, 0.0, 3.0, 0.0, 0.0, 4.0];
        let dist = pairwise_distance_cpu(&points, 2);
        let d01 = dist[1]; // d(0,1)
        let d02 = dist[2]; // d(0,2)
        let d12 = dist[3 + 2]; // d(1,2)
        assert!(d01 + d12 >= d02 - 1e-5, "Triangle inequality violated");
    }

    #[test]
    fn test_cpu_single_point() {
        let points = vec![42.0, 0.0, -1.0];
        let dist = pairwise_distance_cpu(&points, 3);
        assert_eq!(dist.len(), 1);
        assert!(dist[0].abs() < 1e-10);
    }

    #[test]
    fn test_cpu_high_dimensional() {
        // Two points in 10D
        let mut p1 = vec![0.0f32; 10];
        let mut p2 = vec![0.0f32; 10];
        p1[0] = 1.0;
        p2[0] = 4.0;
        let mut points = p1;
        points.extend(p2);
        let dist = pairwise_distance_cpu(&points, 10);
        assert!((dist[1] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_diagonal_zero() {
        let points = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let dist = pairwise_distance_cpu(&points, 3);
        let n = 3;
        for i in 0..n {
            assert!(dist[i * n + i].abs() < 1e-10, "diagonal must be zero");
        }
    }

    #[test]
    fn test_cpu_non_negative() {
        let points = vec![-5.0, 3.0, 2.0, -1.0, 0.0, 7.0];
        let dist = pairwise_distance_cpu(&points, 2);
        for d in &dist {
            assert!(*d >= 0.0, "distances must be non-negative");
        }
    }

    // GPU tests require hardware
    // #[test]
    // fn test_gpu_matches_cpu() {
    //     let ctx = GpuContext::new().expect("Need GPU");
    //     let points: Vec<f32> = (0..30).map(|i| i as f32).collect();
    //     let gpu = pairwise_distance_gpu(&ctx, &points, 3);
    //     let cpu = pairwise_distance_cpu(&points, 3);
    //     for (a, b) in gpu.iter().zip(cpu.iter()) {
    //         assert!((a - b).abs() < 1e-3);
    //     }
    // }
}
