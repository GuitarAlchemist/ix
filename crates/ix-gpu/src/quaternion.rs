//! GPU-accelerated batch quaternion transforms.
//!
//! Rotates N 3D points by a quaternion entirely on the GPU using WGSL compute shaders.
//! Uses the formula: p' = q * p * q⁻¹ (sandwich product).
//!
//! # Examples
//!
//! ```no_run
//! use ix_gpu::context::GpuContext;
//! use ix_gpu::quaternion::{batch_quaternion_rotate_gpu, batch_quaternion_rotate_cpu};
//!
//! let points = vec![1.0f32, 0.0, 0.0,  0.0, 1.0, 0.0]; // 2 points
//! // 90° rotation around Z axis: q = (cos(45°), 0, 0, sin(45°))
//! let s = std::f32::consts::FRAC_1_SQRT_2;
//! let quat = [s, 0.0, 0.0, s]; // [w, x, y, z]
//!
//! // CPU version (always works)
//! let result = batch_quaternion_rotate_cpu(&points, &quat);
//! assert_eq!(result.len(), 6); // 2 points × 3 components
//! ```

use crate::context::GpuContext;
use wgpu::*;

/// WGSL shader for batch quaternion rotation.
///
/// Each thread rotates one point by the quaternion using the sandwich product.
/// Quaternion layout: [w, x, y, z] (scalar-first convention).
/// Points layout: flat [x0,y0,z0, x1,y1,z1, ...].
const QUAT_ROTATE_SHADER: &str = r#"
struct Quaternion {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> quat: array<f32>;  // [w, x, y, z]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn batch_rotate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let num_points = arrayLength(&points) / 3u;
    let idx = global_id.x;
    if (idx >= num_points) {
        return;
    }

    let base = idx * 3u;
    let px = points[base];
    let py = points[base + 1u];
    let pz = points[base + 2u];

    let qw = quat[0];
    let qx = quat[1];
    let qy = quat[2];
    let qz = quat[3];

    // t = 2 * cross(q.xyz, p)
    let tx = 2.0 * (qy * pz - qz * py);
    let ty = 2.0 * (qz * px - qx * pz);
    let tz = 2.0 * (qx * py - qy * px);

    // p' = p + qw * t + cross(q.xyz, t)
    output[base]      = px + qw * tx + (qy * tz - qz * ty);
    output[base + 1u] = py + qw * ty + (qz * tx - qx * tz);
    output[base + 2u] = pz + qw * tz + (qx * ty - qy * tx);
}
"#;

/// Rotate N 3D points by a quaternion on the GPU.
///
/// - `points`: flat array of 3D points [x0,y0,z0, x1,y1,z1, ...]
/// - `quat`: quaternion [w, x, y, z] (must be unit quaternion)
///
/// Returns flat array of rotated points, same layout.
pub fn batch_quaternion_rotate_gpu(ctx: &GpuContext, points: &[f32], quat: &[f32; 4]) -> Vec<f32> {
    assert!(points.len() % 3 == 0, "Points must be triples (x,y,z)");
    let n = points.len();

    let buf_points = ctx.create_buffer_init("points", points);
    let buf_quat = ctx.create_buffer_init("quat", quat);
    let buf_output = ctx.create_output_buffer("output", std::mem::size_of_val(points) as u64);

    let pipeline = ctx.create_compute_pipeline("quat_rotate", QUAT_ROTATE_SHADER, "batch_rotate");

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("quat_bind"),
        layout: &bind_group_layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: buf_points.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: buf_quat.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: buf_output.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("quat_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("quat_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let num_points = (n / 3) as u32;
        let workgroups = num_points.div_ceil(256);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.read_buffer(&buf_output, std::mem::size_of_val(points) as u64)
}

/// CPU fallback: rotate N 3D points by a unit quaternion.
///
/// Uses the optimized formula: p' = p + 2w(q×p) + 2(q×(q×p))
/// where q = (qx, qy, qz) is the vector part.
pub fn batch_quaternion_rotate_cpu(points: &[f32], quat: &[f32; 4]) -> Vec<f32> {
    assert!(points.len() % 3 == 0, "Points must be triples (x,y,z)");

    let [qw, qx, qy, qz] = *quat;
    let mut out = vec![0.0f32; points.len()];

    for i in (0..points.len()).step_by(3) {
        let px = points[i];
        let py = points[i + 1];
        let pz = points[i + 2];

        // t = 2 * cross(q.xyz, p)
        let tx = 2.0 * (qy * pz - qz * py);
        let ty = 2.0 * (qz * px - qx * pz);
        let tz = 2.0 * (qx * py - qy * px);

        // p' = p + qw * t + cross(q.xyz, t)
        out[i] = px + qw * tx + (qy * tz - qz * ty);
        out[i + 1] = py + qw * ty + (qz * tx - qx * tz);
        out[i + 2] = pz + qw * tz + (qx * ty - qy * tx);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_identity_rotation() {
        let points = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let identity = [1.0, 0.0, 0.0, 0.0]; // no rotation
        let result = batch_quaternion_rotate_cpu(&points, &identity);
        for (a, b) in result.iter().zip(points.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Identity rotation should not change points"
            );
        }
    }

    #[test]
    fn test_cpu_90_deg_z_rotation() {
        // 90° around Z: (cos(45°), 0, 0, sin(45°))
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let quat = [s, 0.0, 0.0, s];
        let points = vec![1.0, 0.0, 0.0]; // x-axis
        let result = batch_quaternion_rotate_cpu(&points, &quat);
        // Should map to y-axis: (0, 1, 0)
        assert!(result[0].abs() < 1e-5, "x should be ~0, got {}", result[0]);
        assert!(
            (result[1] - 1.0).abs() < 1e-5,
            "y should be ~1, got {}",
            result[1]
        );
        assert!(result[2].abs() < 1e-5, "z should be ~0, got {}", result[2]);
    }

    #[test]
    fn test_cpu_180_deg_rotation() {
        // 180° around Y: (0, 0, 1, 0)
        let quat = [0.0, 0.0, 1.0, 0.0];
        let points = vec![1.0, 0.0, 0.0];
        let result = batch_quaternion_rotate_cpu(&points, &quat);
        // Should map to (-1, 0, 0)
        assert!((result[0] - (-1.0)).abs() < 1e-5);
        assert!(result[1].abs() < 1e-5);
        assert!(result[2].abs() < 1e-5);
    }

    #[test]
    fn test_cpu_batch_multiple_points() {
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let quat = [s, 0.0, 0.0, s]; // 90° around Z
        let points = vec![
            1.0, 0.0, 0.0, // → (0, 1, 0)
            0.0, 1.0, 0.0, // → (-1, 0, 0)
        ];
        let result = batch_quaternion_rotate_cpu(&points, &quat);
        // Point 0: x→y
        assert!(result[0].abs() < 1e-5);
        assert!((result[1] - 1.0).abs() < 1e-5);
        // Point 1: y→-x
        assert!((result[3] - (-1.0)).abs() < 1e-5);
        assert!(result[4].abs() < 1e-5);
    }

    #[test]
    fn test_cpu_rotation_preserves_length() {
        let s = 0.5f32;
        let c = (1.0 - 3.0 * s * s).sqrt();
        let quat = [c, s, s, s]; // arbitrary rotation
                                 // Normalize quaternion
        let norm =
            (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt();
        let quat = [
            quat[0] / norm,
            quat[1] / norm,
            quat[2] / norm,
            quat[3] / norm,
        ];

        let points = vec![3.0, 4.0, 0.0];
        let result = batch_quaternion_rotate_cpu(&points, &quat);
        let len_before =
            (points[0] * points[0] + points[1] * points[1] + points[2] * points[2]).sqrt();
        let len_after =
            (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
        assert!(
            (len_before - len_after).abs() < 1e-4,
            "Rotation should preserve length"
        );
    }

    #[test]
    fn test_cpu_empty_points() {
        let quat = [1.0, 0.0, 0.0, 0.0];
        let result = batch_quaternion_rotate_cpu(&[], &quat);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cpu_double_rotation() {
        // Two 90° rotations around Z = 180° rotation
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let quat = [s, 0.0, 0.0, s]; // 90° around Z
        let points = vec![1.0, 0.0, 0.0];
        let r1 = batch_quaternion_rotate_cpu(&points, &quat);
        let r2 = batch_quaternion_rotate_cpu(&r1, &quat);
        // After 180°: (1,0,0) → (-1,0,0)
        assert!((r2[0] - (-1.0)).abs() < 1e-4);
        assert!(r2[1].abs() < 1e-4);
        assert!(r2[2].abs() < 1e-4);
    }

    #[test]
    fn test_cpu_rotation_axis_invariant() {
        // Rotating around Z should not change Z component
        let s = std::f32::consts::FRAC_1_SQRT_2;
        let quat = [s, 0.0, 0.0, s]; // 90° around Z
        let points = vec![1.0, 2.0, 7.0];
        let result = batch_quaternion_rotate_cpu(&points, &quat);
        assert!(
            (result[2] - 7.0).abs() < 1e-4,
            "Z component should be unchanged"
        );
    }

    // GPU tests require hardware — commented out
    // #[test]
    // fn test_gpu_matches_cpu() {
    //     let ctx = GpuContext::new().expect("Need GPU");
    //     let s = std::f32::consts::FRAC_1_SQRT_2;
    //     let quat = [s, 0.0, 0.0, s];
    //     let points: Vec<f32> = (0..300).map(|i| i as f32 * 0.1).collect();
    //     let gpu = batch_quaternion_rotate_gpu(&ctx, &points, &quat);
    //     let cpu = batch_quaternion_rotate_cpu(&points, &quat);
    //     for (a, b) in gpu.iter().zip(cpu.iter()) {
    //         assert!((a - b).abs() < 1e-3);
    //     }
    // }
}
