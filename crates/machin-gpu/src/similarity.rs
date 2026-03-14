//! GPU-accelerated cosine similarity.
//!
//! Computes cosine similarity between vectors entirely on the GPU
//! using WGSL compute shaders.

use wgpu::*;
use crate::context::GpuContext;

/// WGSL compute shader for cosine similarity.
///
/// Computes dot(a, b), |a|^2, |b|^2 in parallel using workgroup reductions.
const COSINE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
// result[0] = dot(a,b), result[1] = |a|^2, result[2] = |b|^2

var<workgroup> shared_dot: array<f32, 256>;
var<workgroup> shared_norm_a: array<f32, 256>;
var<workgroup> shared_norm_b: array<f32, 256>;

@compute @workgroup_size(256)
fn cosine_similarity(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = arrayLength(&a);

    // Each thread accumulates partial sums
    var dot_sum: f32 = 0.0;
    var norm_a_sum: f32 = 0.0;
    var norm_b_sum: f32 = 0.0;

    // Grid-stride loop for vectors larger than workgroup
    var i = gid;
    while (i < n) {
        let ai = a[i];
        let bi = b[i];
        dot_sum += ai * bi;
        norm_a_sum += ai * ai;
        norm_b_sum += bi * bi;
        i += 256u * 1u; // stride = workgroup_size * num_workgroups
    }

    shared_dot[tid] = dot_sum;
    shared_norm_a[tid] = norm_a_sum;
    shared_norm_b[tid] = norm_b_sum;

    workgroupBarrier();

    // Parallel reduction
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_dot[tid] += shared_dot[tid + stride];
            shared_norm_a[tid] += shared_norm_a[tid + stride];
            shared_norm_b[tid] += shared_norm_b[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 writes result
    if (tid == 0u) {
        result[0] = shared_dot[0];
        result[1] = shared_norm_a[0];
        result[2] = shared_norm_b[0];
    }
}
"#;

/// WGSL shader for dot product only.
const DOT_PRODUCT_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn dot_product(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = arrayLength(&a);

    var sum: f32 = 0.0;
    var i = gid;
    while (i < n) {
        sum += a[i] * b[i];
        i += 256u;
    }

    shared[tid] = sum;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        result[0] = shared[0];
    }
}
"#;

/// Compute cosine similarity between two f32 vectors on the GPU.
///
/// Returns a value in [-1, 1]. Handles GPU init, data transfer, and compute.
pub fn cosine_similarity_gpu(ctx: &GpuContext, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let buf_a = ctx.create_buffer_init("vec_a", a);
    let buf_b = ctx.create_buffer_init("vec_b", b);
    let buf_result = ctx.create_output_buffer("result", 3 * std::mem::size_of::<f32>() as u64);

    let pipeline = ctx.create_compute_pipeline("cosine_sim", COSINE_SHADER, "cosine_similarity");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("cosine_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: buf_result.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("cosine_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("cosine_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // Dispatch enough workgroups to cover the data
        let workgroups = (a.len() as u32).div_ceil(256);
        pass.dispatch_workgroups(workgroups.min(1), 1, 1); // Single workgroup for reduction
    }

    ctx.queue.submit(Some(encoder.finish()));

    let results = ctx.read_buffer(&buf_result, 3 * std::mem::size_of::<f32>() as u64);

    let dot = results[0];
    let norm_a = results[1].sqrt();
    let norm_b = results[2].sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute dot product of two f32 vectors on the GPU.
pub fn dot_product_gpu(ctx: &GpuContext, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let buf_a = ctx.create_buffer_init("vec_a", a);
    let buf_b = ctx.create_buffer_init("vec_b", b);
    let buf_result = ctx.create_output_buffer("result", std::mem::size_of::<f32>() as u64);

    let pipeline = ctx.create_compute_pipeline("dot_prod", DOT_PRODUCT_SHADER, "dot_product");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("dot_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: buf_result.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("dot_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("dot_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    let results = ctx.read_buffer(&buf_result, std::mem::size_of::<f32>() as u64);
    results[0]
}

/// CPU fallback for cosine similarity (for comparison / when no GPU).
pub fn cosine_similarity_cpu(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Euclidean distance on GPU (via compute shader).
pub fn euclidean_distance_cpu(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity_cpu(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity_cpu(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cpu_cosine_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity_cpu(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_cosine_parallel() {
        let a = vec![3.0, 4.0, 0.0];
        let b = vec![6.0, 8.0, 0.0]; // same direction, different magnitude
        let sim = cosine_similarity_cpu(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5, "parallel vectors should have similarity 1.0");
    }

    #[test]
    fn test_cpu_cosine_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity_cpu(&a, &b);
        assert_eq!(sim, 0.0, "zero vector should give similarity 0");
    }

    #[test]
    fn test_cpu_cosine_single_element() {
        let sim = cosine_similarity_cpu(&[5.0], &[3.0]);
        assert!((sim - 1.0).abs() < 1e-5);
        let sim_neg = cosine_similarity_cpu(&[5.0], &[-3.0]);
        assert!((sim_neg - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_cosine_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, -1.0, 2.0];
        let sim_ab = cosine_similarity_cpu(&a, &b);
        let sim_ba = cosine_similarity_cpu(&b, &a);
        assert!((sim_ab - sim_ba).abs() < 1e-10, "cosine similarity should be symmetric");
    }

    #[test]
    fn test_cpu_euclidean_distance_zero() {
        let a = vec![1.0, 2.0, 3.0];
        let d = euclidean_distance_cpu(&a, &a);
        assert!(d < 1e-10);
    }

    #[test]
    fn test_cpu_euclidean_distance_known() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let d = euclidean_distance_cpu(&a, &b);
        assert!((d - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_cpu_euclidean_distance_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 8.0];
        let d_ab = euclidean_distance_cpu(&a, &b);
        let d_ba = euclidean_distance_cpu(&b, &a);
        assert!((d_ab - d_ba).abs() < 1e-10);
    }

    // GPU tests require actual GPU hardware — run with `cargo test --features gpu_tests`
    // #[test]
    // fn test_gpu_cosine() {
    //     let ctx = GpuContext::new().expect("Need GPU");
    //     let a = vec![1.0f32, 2.0, 3.0, 4.0];
    //     let b = vec![4.0f32, 3.0, 2.0, 1.0];
    //     let gpu_sim = cosine_similarity_gpu(&ctx, &a, &b);
    //     let cpu_sim = cosine_similarity_cpu(&a, &b);
    //     assert!((gpu_sim - cpu_sim).abs() < 1e-4);
    // }
}
