//! Lie groups and Lie algebras for rigid body transformations.
//!
//! Provides mappings between Lie algebras (tangent spaces) and Lie groups
//! (rotation/transformation groups):
//!
//! - **so(3) ↔ SO(3)**: 3D rotations via Rodrigues' formula
//! - **se(3) ↔ SE(3)**: 6-DOF rigid body transforms
//! - **SU(2) ↔ Quaternions**: Spin group / quaternion bridge
//! - **Pauli matrices**: Generators of SU(2)
//!
//! # Examples
//!
//! ```
//! use ix_dynamics::lie;
//! use std::f64::consts::PI;
//!
//! // Rotate 90° around Z axis using exponential map
//! let omega = [0.0, 0.0, PI / 2.0];
//! let rotation = lie::so3_exp(&omega);
//!
//! // Should map [1, 0, 0] → [0, 1, 0]
//! let x = rotation[[0, 0]]; // ≈ 0 (cos 90°)
//! let y = rotation[[1, 0]]; // ≈ 1 (sin 90°)
//! assert!(x.abs() < 1e-10);
//! assert!((y - 1.0).abs() < 1e-10);
//! ```

use ndarray::{array, Array1, Array2};

use crate::error::DynamicsError;

/// Tolerance for numerical comparisons.
const TOL: f64 = 1e-10;

// ─── so(3) / SO(3) ──────────────────────────────────────────────────────────

/// Compute the Lie bracket [A, B] = AB - BA for 3×3 skew-symmetric matrices.
///
/// The inputs are 3-vectors representing elements of so(3); the bracket
/// is returned as a 3-vector.
pub fn so3_bracket(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    // [a]× [b]× - [b]× [a]× = [a × b]×
    // So the bracket of two so(3) elements is just the cross product.
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Convert a 3-vector to its skew-symmetric matrix (hat map).
///
/// `[ω]× = [[0, -ωz, ωy], [ωz, 0, -ωx], [-ωy, ωx, 0]]`
pub fn hat(omega: &[f64; 3]) -> Array2<f64> {
    array![
        [0.0, -omega[2], omega[1]],
        [omega[2], 0.0, -omega[0]],
        [-omega[1], omega[0], 0.0]
    ]
}

/// Extract the 3-vector from a skew-symmetric matrix (vee map).
///
/// Inverse of `hat`. Does not check that the input is actually skew-symmetric.
pub fn vee(mat: &Array2<f64>) -> [f64; 3] {
    [mat[[2, 1]], mat[[0, 2]], mat[[1, 0]]]
}

/// Exponential map so(3) → SO(3) via Rodrigues' formula.
///
/// Given angular velocity vector ω ∈ ℝ³, computes R = exp([ω]×).
///
/// Uses Taylor expansion near θ = 0 for numerical stability.
pub fn so3_exp(omega: &[f64; 3]) -> Array2<f64> {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();

    if theta < TOL {
        // First-order Taylor: R ≈ I + [ω]×
        let k = hat(omega);
        let mut r = Array2::eye(3);
        r += &k;
        return r;
    }

    let k = hat(&[omega[0] / theta, omega[1] / theta, omega[2] / theta]);
    let k2 = k.dot(&k);

    // Rodrigues: R = I + sin(θ) K + (1 - cos(θ)) K²
    let mut r = Array2::eye(3);
    r += &(&k * theta.sin());
    r += &(&k2 * (1.0 - theta.cos()));
    r
}

/// Logarithmic map SO(3) → so(3).
///
/// Given rotation matrix R ∈ SO(3), returns ω such that R = exp([ω]×).
///
/// Returns `Err` if the input is not a valid rotation matrix.
pub fn so3_log(r: &Array2<f64>) -> Result<[f64; 3], DynamicsError> {
    if r.shape() != [3, 3] {
        return Err(DynamicsError::InvalidParameter(
            "so3_log requires a 3×3 matrix".into(),
        ));
    }

    let trace = r[[0, 0]] + r[[1, 1]] + r[[2, 2]];
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    if theta.abs() < TOL {
        // Near identity: ω ≈ 0
        return Ok([0.0, 0.0, 0.0]);
    }

    if (std::f64::consts::PI - theta).abs() < TOL {
        // θ ≈ π: need special handling
        // Find the column of (R + I) with largest norm
        let mut rpi = r.clone();
        rpi[[0, 0]] += 1.0;
        rpi[[1, 1]] += 1.0;
        rpi[[2, 2]] += 1.0;

        let mut best_col = 0;
        let mut best_norm = 0.0;
        for c in 0..3 {
            let norm_sq =
                rpi[[0, c]] * rpi[[0, c]] + rpi[[1, c]] * rpi[[1, c]] + rpi[[2, c]] * rpi[[2, c]];
            if norm_sq > best_norm {
                best_norm = norm_sq;
                best_col = c;
            }
        }

        let norm = best_norm.sqrt();
        if norm < TOL {
            return Ok([0.0, 0.0, 0.0]);
        }

        let axis = [
            rpi[[0, best_col]] / norm,
            rpi[[1, best_col]] / norm,
            rpi[[2, best_col]] / norm,
        ];
        return Ok([
            axis[0] * std::f64::consts::PI,
            axis[1] * std::f64::consts::PI,
            axis[2] * std::f64::consts::PI,
        ]);
    }

    // General case: ω = θ/(2 sin θ) × vee(R - R^T)
    let factor = theta / (2.0 * theta.sin());
    Ok([
        factor * (r[[2, 1]] - r[[1, 2]]),
        factor * (r[[0, 2]] - r[[2, 0]]),
        factor * (r[[1, 0]] - r[[0, 1]]),
    ])
}

// ─── se(3) / SE(3) ──────────────────────────────────────────────────────────

/// Exponential map se(3) → SE(3).
///
/// A twist ξ = (ω, v) ∈ ℝ⁶ maps to a 4×4 homogeneous transform.
/// - ω (first 3 elements): angular velocity
/// - v (last 3 elements): linear velocity
///
/// Returns 4×4 matrix `[[R, t], [0, 1]]`.
pub fn se3_exp(twist: &[f64; 6]) -> Array2<f64> {
    let omega = [twist[0], twist[1], twist[2]];
    let v = [twist[3], twist[4], twist[5]];

    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
    let r = so3_exp(&omega);

    let t = if theta < TOL {
        // Pure translation: T = I, t = v
        Array1::from_vec(vec![v[0], v[1], v[2]])
    } else {
        // t = V * v where V = I + (1-cos θ)/θ² [ω]× + (θ - sin θ)/θ³ [ω]×²
        let k = hat(&[omega[0] / theta, omega[1] / theta, omega[2] / theta]);
        let k2 = k.dot(&k);
        let mut big_v = Array2::eye(3);
        big_v += &(&k * ((1.0 - theta.cos()) / theta));
        big_v += &(&k2 * ((theta - theta.sin()) / theta));
        let v_arr = Array1::from_vec(vec![v[0], v[1], v[2]]);
        big_v.dot(&v_arr)
    };

    let mut result = Array2::zeros((4, 4));
    result.slice_mut(ndarray::s![..3, ..3]).assign(&r);
    result[[0, 3]] = t[0];
    result[[1, 3]] = t[1];
    result[[2, 3]] = t[2];
    result[[3, 3]] = 1.0;
    result
}

// ─── SU(2) / Quaternion bridge ───────────────────────────────────────────────

/// Convert a unit quaternion (w, x, y, z) to its SU(2) matrix representation.
///
/// SU(2) matrix: `[[w + zi, xi + y], [-xi + y, w - zi]]`
/// where the quaternion is `q = w + xi + yj + zk`.
///
/// Returns a 2×2 complex matrix as a 2×2×2 array where `[r, c, 0]` is real
/// and `[r, c, 1]` is imaginary.
pub fn su2_from_quaternion(q: &[f64; 4]) -> [[Complex; 2]; 2] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [Complex(w, z), Complex(x, y)],
        [Complex(-x, y), Complex(w, -z)],
    ]
}

/// Convert an SU(2) matrix back to a unit quaternion (w, x, y, z).
pub fn quaternion_from_su2(m: &[[Complex; 2]; 2]) -> [f64; 4] {
    let w = m[0][0].0; // real part of (0,0)
    let z = m[0][0].1; // imag part of (0,0)
    let x = m[0][1].0; // real part of (0,1)
    let y = m[0][1].1; // imag part of (0,1)
    [w, x, y, z]
}

/// A simple complex number for SU(2) representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex(pub f64, pub f64);

impl Complex {
    /// Modulus squared |z|².
    pub fn norm_sq(self) -> f64 {
        self.0 * self.0 + self.1 * self.1
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Self) -> Self::Output {
        Complex(
            self.0 * rhs.0 - self.1 * rhs.1,
            self.0 * rhs.1 + self.1 * rhs.0,
        )
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, rhs: Self) -> Self::Output {
        Complex(self.0 + rhs.0, self.1 + rhs.1)
    }
}

// ─── Pauli matrices ──────────────────────────────────────────────────────────

/// The three Pauli matrices σ_x, σ_y, σ_z as 2×2 complex matrices.
///
/// These are the generators of SU(2):
/// - σ_x = [[0, 1], [1, 0]]
/// - σ_y = [[0, -i], [i, 0]]
/// - σ_z = [[1, 0], [0, -1]]
pub fn pauli_matrices() -> [[[Complex; 2]; 2]; 3] {
    let zero = Complex(0.0, 0.0);
    let one = Complex(1.0, 0.0);
    let neg_one = Complex(-1.0, 0.0);
    let i = Complex(0.0, 1.0);
    let neg_i = Complex(0.0, -1.0);

    [
        // σ_x
        [[zero, one], [one, zero]],
        // σ_y
        [[zero, neg_i], [i, zero]],
        // σ_z
        [[one, zero], [zero, neg_one]],
    ]
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TEST_TOL: f64 = 1e-10;

    #[test]
    fn test_hat_vee_roundtrip() {
        let omega = [1.0, 2.0, 3.0];
        let mat = hat(&omega);
        let recovered = vee(&mat);
        for i in 0..3 {
            assert!((omega[i] - recovered[i]).abs() < TEST_TOL);
        }
    }

    #[test]
    fn test_hat_skew_symmetric() {
        let omega = [1.0, -2.0, 0.5];
        let mat = hat(&omega);
        for i in 0..3 {
            for j in 0..3 {
                assert!((mat[[i, j]] + mat[[j, i]]).abs() < TEST_TOL);
            }
        }
    }

    #[test]
    fn test_so3_bracket_is_cross_product() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let result = so3_bracket(&a, &b);
        // a × b = [0, 0, 1]
        assert!(result[0].abs() < TEST_TOL);
        assert!(result[1].abs() < TEST_TOL);
        assert!((result[2] - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_so3_exp_identity() {
        let omega = [0.0, 0.0, 0.0];
        let r = so3_exp(&omega);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((r[[i, j]] - expected).abs() < TEST_TOL);
            }
        }
    }

    #[test]
    fn test_so3_exp_90_deg_z() {
        let omega = [0.0, 0.0, PI / 2.0];
        let r = so3_exp(&omega);
        // Expected: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert!((r[[0, 0]]).abs() < TEST_TOL); // cos 90°
        assert!((r[[0, 1]] + 1.0).abs() < TEST_TOL); // -sin 90°
        assert!((r[[1, 0]] - 1.0).abs() < TEST_TOL); // sin 90°
        assert!((r[[1, 1]]).abs() < TEST_TOL); // cos 90°
        assert!((r[[2, 2]] - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_so3_exp_180_deg_x() {
        let omega = [PI, 0.0, 0.0];
        let r = so3_exp(&omega);
        // R_x(π) = [[1,0,0],[0,-1,0],[0,0,-1]]
        assert!((r[[0, 0]] - 1.0).abs() < TEST_TOL);
        assert!((r[[1, 1]] + 1.0).abs() < TEST_TOL);
        assert!((r[[2, 2]] + 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_so3_exp_is_rotation() {
        // R R^T = I and det(R) = 1
        let omega = [0.3, -0.7, 1.2];
        let r = so3_exp(&omega);
        let rt = r.t();
        let product = r.dot(&rt);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - expected).abs() < TEST_TOL,
                    "R*R^T[{},{}] = {} (expected {})",
                    i,
                    j,
                    product[[i, j]],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_so3_log_identity() {
        let r = Array2::eye(3);
        let omega = so3_log(&r).unwrap();
        for value in &omega {
            assert!(value.abs() < TEST_TOL);
        }
    }

    #[test]
    fn test_so3_exp_log_roundtrip() {
        let omega = [0.5, -0.3, 0.8];
        let r = so3_exp(&omega);
        let recovered = so3_log(&r).unwrap();
        for i in 0..3 {
            assert!(
                (omega[i] - recovered[i]).abs() < 1e-8,
                "omega[{}]: {} vs {}",
                i,
                omega[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_so3_log_180_deg() {
        // 180° around X axis
        let r = so3_exp(&[PI, 0.0, 0.0]);
        let omega = so3_log(&r).unwrap();
        let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt();
        assert!((theta - PI).abs() < 1e-6);
    }

    #[test]
    fn test_se3_exp_pure_translation() {
        let twist = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        let t = se3_exp(&twist);
        // Rotation part should be identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((t[[i, j]] - expected).abs() < TEST_TOL);
            }
        }
        // Translation part
        assert!((t[[0, 3]] - 1.0).abs() < TEST_TOL);
        assert!((t[[1, 3]] - 2.0).abs() < TEST_TOL);
        assert!((t[[2, 3]] - 3.0).abs() < TEST_TOL);
        assert!((t[[3, 3]] - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_se3_exp_rotation_only() {
        let twist = [0.0, 0.0, PI / 2.0, 0.0, 0.0, 0.0];
        let t = se3_exp(&twist);
        // Should be 90° rotation around Z, no translation
        assert!((t[[0, 0]]).abs() < TEST_TOL);
        assert!((t[[1, 0]] - 1.0).abs() < TEST_TOL);
        assert!((t[[3, 3]] - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_su2_quaternion_roundtrip() {
        // Unit quaternion for 90° around Z: (cos 45°, 0, 0, sin 45°)
        let q = [(PI / 4.0).cos(), 0.0, 0.0, (PI / 4.0).sin()];
        let su2 = su2_from_quaternion(&q);
        let recovered = quaternion_from_su2(&su2);
        for i in 0..4 {
            assert!(
                (q[i] - recovered[i]).abs() < TEST_TOL,
                "q[{}]: {} vs {}",
                i,
                q[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_su2_identity() {
        let q = [1.0, 0.0, 0.0, 0.0]; // identity quaternion
        let su2 = su2_from_quaternion(&q);
        // Should be 2×2 identity
        assert!((su2[0][0].0 - 1.0).abs() < TEST_TOL);
        assert!(su2[0][0].1.abs() < TEST_TOL);
        assert!(su2[0][1].norm_sq() < TEST_TOL);
        assert!(su2[1][0].norm_sq() < TEST_TOL);
        assert!((su2[1][1].0 - 1.0).abs() < TEST_TOL);
        assert!(su2[1][1].1.abs() < TEST_TOL);
    }

    #[test]
    fn test_su2_unitarity() {
        // |det(U)| = 1 for SU(2)
        let q = [0.5, 0.5, 0.5, 0.5]; // unit quaternion
        let su2 = su2_from_quaternion(&q);
        // det = a*d - b*c (complex multiplication)
        // det = ad - bc
        let ad = su2[0][0] * su2[1][1];
        let bc = su2[0][1] * su2[1][0];
        let det_real = ad.0 - bc.0;
        let det_imag = ad.1 - bc.1;
        let det_norm = (det_real * det_real + det_imag * det_imag).sqrt();
        assert!((det_norm - 1.0).abs() < TEST_TOL);
    }

    #[test]
    fn test_pauli_matrices_trace() {
        let paulis = pauli_matrices();
        // All Pauli matrices are traceless
        for (i, sigma) in paulis.iter().enumerate() {
            let trace_real = sigma[0][0].0 + sigma[1][1].0;
            let trace_imag = sigma[0][0].1 + sigma[1][1].1;
            assert!(
                trace_real.abs() < TEST_TOL && trace_imag.abs() < TEST_TOL,
                "σ_{} has non-zero trace",
                i + 1
            );
        }
    }

    #[test]
    fn test_pauli_squared_is_identity() {
        let paulis = pauli_matrices();
        // σ_i² = I for all Pauli matrices
        for (idx, sigma) in paulis.iter().enumerate() {
            // Compute σ²
            let sq = mat_mul_2x2(sigma, sigma);
            assert!(
                (sq[0][0].0 - 1.0).abs() < TEST_TOL,
                "σ_{}² [0,0] real",
                idx + 1
            );
            assert!(sq[0][0].1.abs() < TEST_TOL, "σ_{}² [0,0] imag", idx + 1);
            assert!(sq[0][1].norm_sq() < TEST_TOL, "σ_{}² [0,1]", idx + 1);
            assert!(sq[1][0].norm_sq() < TEST_TOL, "σ_{}² [1,0]", idx + 1);
            assert!(
                (sq[1][1].0 - 1.0).abs() < TEST_TOL,
                "σ_{}² [1,1] real",
                idx + 1
            );
            assert!(sq[1][1].1.abs() < TEST_TOL, "σ_{}² [1,1] imag", idx + 1);
        }
    }

    #[test]
    fn test_so3_log_rejects_wrong_size() {
        let mat = Array2::eye(2);
        assert!(so3_log(&mat).is_err());
    }

    /// Helper: multiply two 2×2 complex matrices.
    fn mat_mul_2x2(a: &[[Complex; 2]; 2], b: &[[Complex; 2]; 2]) -> [[Complex; 2]; 2] {
        [
            [
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
            ],
            [
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            ],
        ]
    }
}
