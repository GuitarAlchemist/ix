//! Quaternion representation and operations for 3D rotations.

use std::ops::Mul;

/// A quaternion q = w + xi + yj + zk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    /// Create a new quaternion with the given components.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// The identity quaternion (no rotation).
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a quaternion from an axis and angle (radians).
    /// The axis does not need to be pre-normalized.
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len < 1e-12 {
            return Self::identity();
        }
        let half = angle * 0.5;
        let s = half.sin() / len;
        Self {
            w: half.cos(),
            x: axis[0] * s,
            y: axis[1] * s,
            z: axis[2] * s,
        }
    }

    /// Hamilton product: self * other.
    pub fn mul(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    /// Quaternion conjugate: q* = w - xi - yj - zk.
    pub fn conjugate(&self) -> Quaternion {
        Quaternion {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternion norm (magnitude).
    pub fn norm(&self) -> f64 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize to unit quaternion.
    pub fn normalize(&self) -> Quaternion {
        let n = self.norm();
        if n < 1e-12 {
            return Self::identity();
        }
        Quaternion {
            w: self.w / n,
            x: self.x / n,
            y: self.y / n,
            z: self.z / n,
        }
    }

    /// Quaternion inverse: q^{-1} = q* / |q|^2.
    pub fn inverse(&self) -> Quaternion {
        let n2 = self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z;
        if n2 < 1e-24 {
            return Self::identity();
        }
        let c = self.conjugate();
        Quaternion {
            w: c.w / n2,
            x: c.x / n2,
            y: c.y / n2,
            z: c.z / n2,
        }
    }

    /// Dot product of two quaternions.
    pub fn dot(&self, other: &Quaternion) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Rotate a 3D point using the sandwich product: q p q*.
    pub fn rotate_point(&self, point: [f64; 3]) -> [f64; 3] {
        let p = Quaternion::new(0.0, point[0], point[1], point[2]);
        let conj = self.conjugate();
        let rotated = Quaternion::mul(&Quaternion::mul(self, &p), &conj);
        [rotated.x, rotated.y, rotated.z]
    }

    /// Convert to a 3×3 rotation matrix.
    pub fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let q = self.normalize();
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }
}

impl Mul for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Quaternion {
        Quaternion::mul(&self, &rhs)
    }
}

impl Mul<&Quaternion> for &Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: &Quaternion) -> Quaternion {
        Quaternion::mul(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    fn approx_point(a: [f64; 3], b: [f64; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_identity_rotation() {
        let q = Quaternion::identity();
        let p = [1.0, 2.0, 3.0];
        let r = q.rotate_point(p);
        assert!(approx_point(r, p));
    }

    #[test]
    fn test_90_degree_rotation_z() {
        // 90° around Z: (1,0,0) -> (0,1,0)
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let r = q.rotate_point([1.0, 0.0, 0.0]);
        assert!(approx_point(r, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_norm_preservation() {
        let q = Quaternion::from_axis_angle([1.0, 1.0, 0.0], 1.23);
        let p = [3.0, 4.0, 0.0];
        let r = q.rotate_point(p);
        let orig_len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let rot_len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        assert!(approx_eq(orig_len, rot_len));
    }

    #[test]
    fn test_composition() {
        // Two 90° rotations around Z = 180° around Z
        let q90 = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let q180 = q90 * q90;
        let r = q180.rotate_point([1.0, 0.0, 0.0]);
        assert!(approx_point(r, [-1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_inverse() {
        let q = Quaternion::from_axis_angle([1.0, 2.0, 3.0], 0.7);
        let qi = q.inverse();
        let product = q * qi;
        let id = Quaternion::identity();
        assert!(approx_eq(product.w, id.w));
        assert!(approx_eq(product.x, id.x));
        assert!(approx_eq(product.y, id.y));
        assert!(approx_eq(product.z, id.z));
    }

    #[test]
    fn test_unit_quaternion_norm() {
        let q = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 1.0);
        assert!(approx_eq(q.norm(), 1.0));
    }
}
