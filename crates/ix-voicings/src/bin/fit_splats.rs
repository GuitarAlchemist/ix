//! `fit-splats` — fit a Gaussian-Splat representation to the voicing
//! position cloud and write a binary little-endian PLY in Mark Kellogg's
//! `gaussian-splat-3d` schema.
//!
//! Pipeline:
//!
//! 1. Read `state/viz/voicing-positions.bin` (`N × 3 × f32`, packed LE) and
//!    its sidecar `voicing-positions.meta.json`. Promote to `f64` at the
//!    boundary; all math runs in `f64`.
//! 2. Downsample with `--stride` (default 200) to feed k-means a tractable
//!    centroid-fitting set (~3.4K points at stride 200).
//! 3. Run `ix_unsupervised::kmeans::KMeans` with `--k` (default 512) and
//!    `max_iterations = 50`.
//! 4. Assign **all** N points to nearest centroid for honest covariance.
//! 5. Per cluster: compute mean + 3×3 sample covariance (N-1 denominator,
//!    biased fallback when count == 1).
//! 6. Symmetric eigendecomposition via `ix_math::eigen::symmetric_eigen`.
//! 7. Clamp eigenvalues to `max(1e-12)` before log; `scale_k = 0.5 *
//!    ln(λ_k)` (Mark Kellogg stores log-stddev, and std = sqrt(λ)).
//! 8. Reflection correction: if `det(V) < 0`, negate first eigenvector
//!    column to make V a proper rotation (right-handed) before quaternion
//!    extraction.
//! 9. Quaternion from rotation matrix (`[w, x, y, z]`), normalized.
//! 10. SH degree-0 colour: uniform white (RGB = 1, 1, 1) baked via the
//!     standard `f_dc = (rgb - 0.5) / SH_C0` with `SH_C0 = 0.28209479`.
//! 11. Opacity logit: `p_i = count_i / max_count`, clamped to `[0.01,
//!     0.99]`, then `logit(p) = ln(p / (1-p))`. Empty clusters skipped.
//! 12. Write PLY: header + 14 × f32 records per vertex (`x y z f_dc_0
//!     f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2
//!     rot_3`), binary little-endian.
//!
//! ```sh
//! cargo run --release -p ix-voicings --bin fit-splats
//! cargo run --release -p ix-voicings --bin fit-splats -- --k 256 --stride 500
//! ```

use std::path::PathBuf;

use clap::Parser;
use ix_math::eigen::symmetric_eigen;
use ix_unsupervised::kmeans::KMeans;
use ix_unsupervised::traits::Clusterer;
use ndarray::Array2;
use serde::Deserialize;

/// Mark Kellogg / 3DGS spherical-harmonic degree-0 constant: `1 / (2 * sqrt(PI))`.
const SH_C0: f32 = 0.28209479;
/// Header magic for the position buffer record size.
const POS_RECORD_F32S: usize = 3;
/// Per-vertex floats in the output PLY (xyz + 3 SH DC + opacity + 3 scale + 4 rot).
const VERTEX_F32S: usize = 14;
/// Numerical floor for eigenvalues before taking `ln`.
const EIGEN_FLOOR: f64 = 1e-12;

#[derive(Parser, Debug)]
#[command(about = "Fit a 3DGS-compatible Gaussian Splat PLY to the voicing position cloud")]
struct Cli {
    /// Path to the positions binary (`N × 3 × f32`, packed little-endian).
    #[arg(long, default_value = "state/viz/voicing-positions.bin")]
    positions: PathBuf,

    /// Path to the positions metadata sidecar.
    #[arg(long, default_value = "state/viz/voicing-positions.meta.json")]
    meta: PathBuf,

    /// Output PLY path.
    #[arg(long, default_value = "state/viz/voicing-cloud.ply")]
    output: PathBuf,

    /// Number of clusters (k). 512 produces ~30 KB; 1024 ~60 KB.
    #[arg(long, default_value_t = 512)]
    k: usize,

    /// Downsample stride for k-means centroid fitting. The full N points
    /// are always used for the covariance pass — only the centroid search
    /// is downsampled. Default 200 ≈ 3.4K points for the 688K cloud.
    #[arg(long, default_value_t = 200)]
    stride: usize,

    /// Max k-means iterations.
    #[arg(long, default_value_t = 50)]
    max_iterations: usize,

    /// Deterministic seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Deserialize)]
struct PositionMeta {
    total: usize,
}

fn main() {
    let cli = Cli::parse();
    let started = std::time::Instant::now();

    // --- Step 1: read positions + meta -------------------------------------
    let raw = std::fs::read(&cli.positions).unwrap_or_else(|e| {
        eprintln!("error: cannot read {:?}: {e}", cli.positions);
        std::process::exit(2);
    });
    let meta_bytes = std::fs::read(&cli.meta).unwrap_or_else(|e| {
        eprintln!("error: cannot read {:?}: {e}", cli.meta);
        std::process::exit(2);
    });
    let meta: PositionMeta = serde_json::from_slice(&meta_bytes).unwrap_or_else(|e| {
        eprintln!("error: cannot parse meta JSON: {e}");
        std::process::exit(2);
    });

    let n_total = meta.total;
    let expected_bytes = n_total * POS_RECORD_F32S * std::mem::size_of::<f32>();
    if raw.len() != expected_bytes {
        eprintln!(
            "error: positions buffer is {} bytes, expected {} ({} voicings × 3 × 4)",
            raw.len(),
            expected_bytes,
            n_total
        );
        std::process::exit(2);
    }
    eprintln!(
        "loaded positions: n_total={n_total}, k={}, stride={}",
        cli.k, cli.stride
    );

    // Promote f32 → f64 once at the read boundary.
    let mut positions = Array2::<f64>::zeros((n_total, 3));
    for i in 0..n_total {
        for j in 0..3 {
            let off = (i * 3 + j) * 4;
            let val = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            positions[[i, j]] = val as f64;
        }
    }

    // --- Step 2: downsample for k-means ------------------------------------
    let sample_idx: Vec<usize> = (0..n_total).step_by(cli.stride.max(1)).collect();
    let n_sample = sample_idx.len();
    if n_sample < cli.k {
        eprintln!(
            "error: stride {} yields {} sample points but k={}; need n_sample >= k",
            cli.stride, n_sample, cli.k
        );
        std::process::exit(2);
    }
    let mut sample = Array2::<f64>::zeros((n_sample, 3));
    for (row, &i) in sample_idx.iter().enumerate() {
        for j in 0..3 {
            sample[[row, j]] = positions[[i, j]];
        }
    }
    eprintln!("downsampled: n_sample={n_sample}");

    // --- Step 3: k-means on the subset -------------------------------------
    let mut km = KMeans::new(cli.k).with_seed(cli.seed);
    km.max_iterations = cli.max_iterations;
    eprintln!(
        "fitting k-means (k={}, max_iter={})...",
        cli.k, cli.max_iterations
    );
    km.fit(&sample);
    eprintln!(
        "  k-means fitted in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    // --- Step 4: full-resolution membership assignment ---------------------
    let labels = km.predict(&positions);
    let centroids = km.centroids.as_ref().expect("fitted").clone();

    // --- Step 5: per-cluster stats (mean already in centroid, recompute on
    // the full assignment because k-means' centroid is from the *sampled*
    // subset). Recomputing the mean from full membership keeps the
    // covariance honest and centred on the true cluster centroid.
    let k = cli.k;
    let mut counts = vec![0usize; k];
    let mut sums = vec![[0.0f64; 3]; k];
    for i in 0..n_total {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..3 {
            sums[c][j] += positions[[i, j]];
        }
    }
    let mut means = vec![[0.0f64; 3]; k];
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..3 {
                means[c][j] = sums[c][j] / counts[c] as f64;
            }
        } else {
            // Fall back to the k-means centroid (which lives in the
            // sampled subset's coordinate system) so the splat still has
            // a defined position. Cluster will be skipped at emit time
            // because count == 0 means no covariance to extract.
            for j in 0..3 {
                means[c][j] = centroids[[c, j]];
            }
        }
    }

    // 3×3 sample covariance per cluster, single pass over assignments.
    let mut covs: Vec<[[f64; 3]; 3]> = vec![[[0.0; 3]; 3]; k];
    for i in 0..n_total {
        let c = labels[i];
        let dx = positions[[i, 0]] - means[c][0];
        let dy = positions[[i, 1]] - means[c][1];
        let dz = positions[[i, 2]] - means[c][2];
        let d = [dx, dy, dz];
        for a in 0..3 {
            for b in 0..3 {
                covs[c][a][b] += d[a] * d[b];
            }
        }
    }
    // Normalize: N-1 unbiased (or N when count == 1 to avoid div-by-zero).
    for (c, cov) in covs.iter_mut().enumerate().take(k) {
        let denom = match counts[c] {
            0 => continue,
            1 => 1.0, // single-point cluster: biased estimator (covariance ≈ 0)
            n => (n - 1) as f64,
        };
        for row in cov.iter_mut() {
            for entry in row.iter_mut() {
                *entry /= denom;
            }
        }
    }

    // --- Steps 6-11: eigen → scale + rotation; opacity ---------------------
    let max_count = counts.iter().copied().max().unwrap_or(1).max(1) as f64;

    let mut emitted: Vec<VertexF32> = Vec::with_capacity(k);
    let mut eigen_clamped = 0usize;
    let mut reflection_fixed = 0usize;
    let mut empty_clusters = 0usize;

    for c in 0..k {
        if counts[c] == 0 {
            empty_clusters += 1;
            continue;
        }

        // Build the symmetric covariance Array2<f64>.
        let mut sigma = Array2::<f64>::zeros((3, 3));
        for a in 0..3 {
            for b in 0..3 {
                sigma[[a, b]] = covs[c][a][b];
            }
        }

        // Symmetric eigendecomposition: V's columns are eigenvectors of Σ,
        // eigenvalues sorted descending.
        let (mut lambda, mut v) = match symmetric_eigen(&sigma) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("warning: eigen failed for cluster {c}: {e:?}; skipping");
                continue;
            }
        };

        // Clamp eigenvalues before log.
        for k_i in 0..3 {
            if lambda[k_i] < EIGEN_FLOOR {
                lambda[k_i] = EIGEN_FLOOR;
                eigen_clamped += 1;
            }
        }

        // Reflection check: ensure right-handed orthonormal basis.
        let det = det3(&v);
        if det < 0.0 {
            for row in 0..3 {
                v[[row, 0]] = -v[[row, 0]];
            }
            reflection_fixed += 1;
        }

        // Log-scale per eigen-axis.
        let scale = [
            0.5 * lambda[0].ln(),
            0.5 * lambda[1].ln(),
            0.5 * lambda[2].ln(),
        ];

        // Quaternion from rotation matrix (Shoemake / standard branch-free
        // form). Output (w, x, y, z); we normalize defensively.
        let m = [
            [v[[0, 0]], v[[0, 1]], v[[0, 2]]],
            [v[[1, 0]], v[[1, 1]], v[[1, 2]]],
            [v[[2, 0]], v[[2, 1]], v[[2, 2]]],
        ];
        let q = rotation_matrix_to_quaternion(&m);

        // Opacity: density relative to most-populous cluster.
        let p_raw = counts[c] as f64 / max_count;
        let p = p_raw.clamp(0.01, 0.99);
        let opacity_logit = (p / (1.0 - p)).ln();

        // Uniform white SH DC. f_dc = (rgb - 0.5) / SH_C0 with rgb = 1.
        let f_dc = (1.0 - 0.5) / SH_C0;

        emitted.push(VertexF32 {
            x: means[c][0] as f32,
            y: means[c][1] as f32,
            z: means[c][2] as f32,
            f_dc_0: f_dc,
            f_dc_1: f_dc,
            f_dc_2: f_dc,
            opacity: opacity_logit as f32,
            scale_0: scale[0] as f32,
            scale_1: scale[1] as f32,
            scale_2: scale[2] as f32,
            rot_0: q[0] as f32,
            rot_1: q[1] as f32,
            rot_2: q[2] as f32,
            rot_3: q[3] as f32,
        });
    }

    eprintln!(
        "clusters: {} emitted, {} empty; eigen clamps fired {}, reflections fixed {}",
        emitted.len(),
        empty_clusters,
        eigen_clamped,
        reflection_fixed
    );

    // --- Step 12: PLY write -----------------------------------------------
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
            eprintln!("error: cannot create output dir: {e}");
            std::process::exit(2);
        });
    }
    let n = emitted.len();
    let mut bytes: Vec<u8> = Vec::with_capacity(512 + n * VERTEX_F32S * 4);

    // Header (matches the original 2023 INRIA 3DGS layout that Kellogg's
    // gaussian-splat-3d viewer reads). Property order must match the
    // record layout below.
    let header = format!(
        "ply\n\
         format binary_little_endian 1.0\n\
         element vertex {n}\n\
         property float x\n\
         property float y\n\
         property float z\n\
         property float f_dc_0\n\
         property float f_dc_1\n\
         property float f_dc_2\n\
         property float opacity\n\
         property float scale_0\n\
         property float scale_1\n\
         property float scale_2\n\
         property float rot_0\n\
         property float rot_1\n\
         property float rot_2\n\
         property float rot_3\n\
         end_header\n"
    );
    bytes.extend_from_slice(header.as_bytes());

    for v in &emitted {
        for f in [
            v.x, v.y, v.z, v.f_dc_0, v.f_dc_1, v.f_dc_2, v.opacity, v.scale_0, v.scale_1,
            v.scale_2, v.rot_0, v.rot_1, v.rot_2, v.rot_3,
        ] {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
    }

    std::fs::write(&cli.output, &bytes).unwrap_or_else(|e| {
        eprintln!("error: cannot write {:?}: {e}", cli.output);
        std::process::exit(2);
    });

    eprintln!(
        "wrote {} vertices ({} bytes) to {} in {:.1}s",
        n,
        bytes.len(),
        cli.output.display(),
        started.elapsed().as_secs_f64()
    );
}

struct VertexF32 {
    x: f32,
    y: f32,
    z: f32,
    f_dc_0: f32,
    f_dc_1: f32,
    f_dc_2: f32,
    opacity: f32,
    scale_0: f32,
    scale_1: f32,
    scale_2: f32,
    rot_0: f32,
    rot_1: f32,
    rot_2: f32,
    rot_3: f32,
}

/// 3×3 determinant of an ndarray-backed rotation matrix.
fn det3(v: &Array2<f64>) -> f64 {
    let a = v[[0, 0]];
    let b = v[[0, 1]];
    let c = v[[0, 2]];
    let d = v[[1, 0]];
    let e = v[[1, 1]];
    let f = v[[1, 2]];
    let g = v[[2, 0]];
    let h = v[[2, 1]];
    let i = v[[2, 2]];
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Convert a 3×3 rotation matrix to a normalized quaternion `[w, x, y, z]`
/// using Shoemake's branch-by-trace formulation (numerically stable across
/// all sign quadrants). Assumes the input is a proper rotation
/// (`det(m) == +1` within numerical tolerance).
fn rotation_matrix_to_quaternion(m: &[[f64; 3]; 3]) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0; // s = 4w
        let w = 0.25 * s;
        let x = (m[2][1] - m[1][2]) / s;
        let y = (m[0][2] - m[2][0]) / s;
        let z = (m[1][0] - m[0][1]) / s;
        (w, x, y, z)
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0; // s = 4x
        let w = (m[2][1] - m[1][2]) / s;
        let x = 0.25 * s;
        let y = (m[0][1] + m[1][0]) / s;
        let z = (m[0][2] + m[2][0]) / s;
        (w, x, y, z)
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0; // s = 4y
        let w = (m[0][2] - m[2][0]) / s;
        let x = (m[0][1] + m[1][0]) / s;
        let y = 0.25 * s;
        let z = (m[1][2] + m[2][1]) / s;
        (w, x, y, z)
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0; // s = 4z
        let w = (m[1][0] - m[0][1]) / s;
        let x = (m[0][2] + m[2][0]) / s;
        let y = (m[1][2] + m[2][1]) / s;
        let z = 0.25 * s;
        (w, x, y, z)
    };

    let norm = (w * w + x * x + y * y + z * z).sqrt();
    if norm > 0.0 {
        [w / norm, x / norm, y / norm, z / norm]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn identity_quaternion() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let q = rotation_matrix_to_quaternion(&m);
        assert!((q[0] - 1.0).abs() < 1e-9);
        assert!(q[1].abs() < 1e-9);
        assert!(q[2].abs() < 1e-9);
        assert!(q[3].abs() < 1e-9);
    }

    #[test]
    fn quaternion_normalized() {
        // 90° rotation about Z.
        let m = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let q = rotation_matrix_to_quaternion(&m);
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
        // Expect (cos(45°), 0, 0, sin(45°)) up to sign.
        assert!((q[0].abs() - (0.5f64.sqrt())).abs() < 1e-9);
        assert!((q[3].abs() - (0.5f64.sqrt())).abs() < 1e-9);
    }

    #[test]
    fn det_identity_is_one() {
        let m = Array2::<f64>::eye(3);
        assert!((det3(&m) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn det_reflection_is_negative() {
        let m = array![[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(det3(&m) < 0.0);
    }
}
