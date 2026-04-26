//! SO(3) rotation matrix utilities.

use crate::quaternion::Quaternion;

/// Convert a unit quaternion to a 3×3 rotation matrix.
pub fn from_quaternion(q: &Quaternion) -> [[f64; 3]; 3] {
    q.to_rotation_matrix()
}

/// Convert a 3×3 rotation matrix to a unit quaternion.
///
/// Uses Shepperd's method for numerical stability.
pub fn to_quaternion(m: &[[f64; 3]; 3]) -> Quaternion {
    let trace = m[0][0] + m[1][1] + m[2][2];

    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0; // s = 4w
        Quaternion::new(
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0; // s = 4x
        Quaternion::new(
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0; // s = 4y
        Quaternion::new(
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0; // s = 4z
        Quaternion::new(
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    }
}

/// Check if a 3×3 matrix is a valid rotation matrix (orthogonal with determinant ≈ 1).
pub fn is_rotation_matrix(m: &[[f64; 3]; 3], tol: f64) -> bool {
    // Check orthogonality: M^T * M ≈ I
    for i in 0..3 {
        for j in 0..3 {
            let mut dot = 0.0;
            for item in m.iter().take(3) {
                dot += item[i] * item[j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tol {
                return false;
            }
        }
    }

    // Check determinant ≈ 1
    let det = determinant(m);
    (det - 1.0).abs() <= tol
}

/// Orthogonalize a 3×3 matrix using Gram-Schmidt on its columns.
pub fn orthogonalize(m: &mut [[f64; 3]; 3]) {
    // Extract columns
    let mut c0 = [m[0][0], m[1][0], m[2][0]];
    let mut c1 = [m[0][1], m[1][1], m[2][1]];
    let mut c2 = [m[0][2], m[1][2], m[2][2]];

    // Normalize c0
    let n0 = vec_norm(&c0);
    if n0 > 1e-12 {
        c0[0] /= n0;
        c0[1] /= n0;
        c0[2] /= n0;
    }

    // c1 = c1 - proj(c1, c0)
    let d10 = vec_dot(&c1, &c0);
    c1[0] -= d10 * c0[0];
    c1[1] -= d10 * c0[1];
    c1[2] -= d10 * c0[2];
    let n1 = vec_norm(&c1);
    if n1 > 1e-12 {
        c1[0] /= n1;
        c1[1] /= n1;
        c1[2] /= n1;
    }

    // c2 = c0 × c1 (ensures right-handed and orthogonal)
    c2[0] = c0[1] * c1[2] - c0[2] * c1[1];
    c2[1] = c0[2] * c1[0] - c0[0] * c1[2];
    c2[2] = c0[0] * c1[1] - c0[1] * c1[0];

    // Write back
    m[0][0] = c0[0];
    m[0][1] = c1[0];
    m[0][2] = c2[0];
    m[1][0] = c0[1];
    m[1][1] = c1[1];
    m[1][2] = c2[1];
    m[2][0] = c0[2];
    m[2][1] = c1[2];
    m[2][2] = c2[2];
}

fn determinant(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn vec_dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec_norm(a: &[f64; 3]) -> f64 {
    vec_dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_identity_matrix() {
        let q = Quaternion::identity();
        let m = from_quaternion(&q);
        for (i, row) in m.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx_eq(*value, expected));
            }
        }
    }

    #[test]
    fn test_is_rotation_matrix() {
        let q = Quaternion::from_axis_angle([1.0, 2.0, 3.0], 0.7);
        let m = from_quaternion(&q);
        assert!(is_rotation_matrix(&m, 1e-10));
    }

    #[test]
    fn test_not_rotation_matrix() {
        // Scaling matrix is not a rotation
        let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        assert!(!is_rotation_matrix(&m, 1e-6));
    }

    #[test]
    fn test_determinant_one() {
        let q = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 1.5);
        let m = from_quaternion(&q);
        let det = determinant(&m);
        assert!(approx_eq(det, 1.0));
    }

    #[test]
    fn test_round_trip_quaternion_matrix() {
        let q_orig = Quaternion::from_axis_angle([1.0, 0.0, 0.0], PI / 3.0);
        let m = from_quaternion(&q_orig);
        let q_back = to_quaternion(&m);
        // Quaternions q and -q are the same rotation
        let dot = q_orig.dot(&q_back).abs();
        assert!(approx_eq(dot, 1.0));
    }

    #[test]
    fn test_orthogonalize() {
        // Start with a slightly skewed matrix
        let mut m = [[1.0, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        orthogonalize(&mut m);
        assert!(is_rotation_matrix(&m, 1e-10));
    }

    #[test]
    fn test_orthogonalize_preserves_rotation() {
        let q = Quaternion::from_axis_angle([1.0, 1.0, 1.0], 0.5);
        let mut m = from_quaternion(&q);
        // Perturb slightly
        m[0][1] += 0.001;
        m[1][0] -= 0.001;
        orthogonalize(&mut m);
        assert!(is_rotation_matrix(&m, 1e-10));
    }
}
