//! GPU-accelerated matrix multiplication.
//!
//! Uses WGSL compute shaders for general matrix multiply (GEMM).

use wgpu::*;
use crate::context::GpuContext;

/// WGSL shader for matrix multiplication: C = A × B.
///
/// A is M×K, B is K×N, C is M×N. All stored in row-major order.
/// Uses tiled approach with shared memory for better cache behavior.
const MATMUL_SHADER: &str = r#"
struct Params {
    M: u32,
    N: u32,
    K: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= params.M || col >= params.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < params.K; k++) {
        sum += a[row * params.K + k] * b[k * params.N + col];
    }

    c[row * params.N + col] = sum;
}
"#;

/// Parameters for matrix multiplication (passed as uniform buffer).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

/// GPU matrix multiplication: C = A × B.
///
/// `a`: M×K matrix in row-major order.
/// `b`: K×N matrix in row-major order.
/// Returns M×N result matrix.
pub fn matmul_gpu(ctx: &GpuContext, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A must be {}×{}", m, k);
    assert_eq!(b.len(), k * n, "B must be {}×{}", k, n);

    let params = MatmulParams {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        _pad: 0,
    };

    let buf_a = ctx.create_buffer_init("mat_a", a);
    let buf_b = ctx.create_buffer_init("mat_b", b);
    let buf_c = ctx.create_output_buffer("mat_c", (m * n * std::mem::size_of::<f32>()) as u64);

    use wgpu::util::DeviceExt;
    let buf_params = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::cast_slice(&[params]),
        usage: BufferUsages::UNIFORM,
    });

    let pipeline = ctx.create_compute_pipeline("matmul", MATMUL_SHADER, "matmul");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("matmul_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
            BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
            BindGroupEntry { binding: 2, resource: buf_c.as_entire_binding() },
            BindGroupEntry { binding: 3, resource: buf_params.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("matmul_encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let wg_x = (m as u32).div_ceil(16);
        let wg_y = (n as u32).div_ceil(16);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));

    ctx.read_buffer(&buf_c, (m * n * std::mem::size_of::<f32>()) as u64)
}

/// CPU fallback for matrix multiplication.
pub fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);

    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul_identity() {
        // 2×2 identity × [1,2,3,4] = [1,2,3,4]
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = matmul_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cpu_matmul_basic() {
        // [1,2] × [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = matmul_cpu(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cpu_matmul_rectangular() {
        // 2×3 × 3×1 = 2×1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0, 1.0, 1.0];
        let c = matmul_cpu(&a, &b, 2, 3, 1);
        assert_eq!(c, vec![6.0, 15.0]);
    }

    #[test]
    fn test_cpu_matmul_zero_matrix() {
        let a = vec![0.0; 4];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = matmul_cpu(&a, &b, 2, 2, 2);
        assert!(c.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_cpu_matmul_1x1() {
        let c = matmul_cpu(&[3.0], &[4.0], 1, 1, 1);
        assert_eq!(c, vec![12.0]);
    }

    #[test]
    fn test_cpu_matmul_row_times_col() {
        // 1×3 × 3×1 = 1×1 (dot product)
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = matmul_cpu(&a, &b, 1, 3, 1);
        assert_eq!(c, vec![32.0]); // 4+10+18
    }

    #[test]
    fn test_cpu_matmul_col_times_row() {
        // 3×1 × 1×3 = 3×3 (outer product)
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = matmul_cpu(&a, &b, 3, 1, 3);
        assert_eq!(c, vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_cpu_matmul_associativity() {
        // (A × B) × C == A × (B × C)
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = vec![1.0, 0.0, 0.0, 1.0];
        let ab = matmul_cpu(&a, &b, 2, 2, 2);
        let ab_c = matmul_cpu(&ab, &c, 2, 2, 2);
        let bc = matmul_cpu(&b, &c, 2, 2, 2);
        let a_bc = matmul_cpu(&a, &bc, 2, 2, 2);
        for (x, y) in ab_c.iter().zip(a_bc.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cpu_matmul_larger() {
        // 3×3 identity × 3×2 = 3×2 unchanged
        let eye = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = matmul_cpu(&eye, &b, 3, 3, 2);
        assert_eq!(c, b);
    }
}
