//! Quaternion algebra for 3D rotations.
//!
//! Uses **Hamilton convention**: ijk = −1, scalar-first storage `(w, x, y, z)`.
//! Rotation application: `q * v * q.conjugate()` rotates vector v.
//! Composition: `q1 * q2` applies q2 first, then q1 (right-to-left, matching matrix convention).
//!
//! # Examples
//!
//! ```
//! use machin_math::quaternion::{Quaternion, slerp};
//! use std::f64::consts::FRAC_PI_2;
//!
//! // Create a 90° rotation around the Z axis
//! let q = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
//!
//! // Rotate the X-axis unit vector → should become Y-axis
//! let rotated = q.rotate_vector(&[1.0, 0.0, 0.0]);
//! assert!((rotated[0]).abs() < 1e-10);
//! assert!((rotated[1] - 1.0).abs() < 1e-10);
//!
//! // Compose rotations: two 90° rotations = 180°
//! let q180 = q * q;
//! let r2 = q180.rotate_vector(&[1.0, 0.0, 0.0]);
//! assert!((r2[0] + 1.0).abs() < 1e-10); // (1,0,0) → (-1,0,0)
//!
//! // Interpolate halfway between identity and 90° rotation
//! let mid = slerp(&Quaternion::identity(), &q, 0.5);
//! assert!(mid.is_unit(1e-10));
//! ```

use ndarray::{array, Array1, Array2};

use crate::error::MathError;

// ─── Constants ──────────────────────────────────────────────────────────────

/// Threshold for near-zero norm checks (normalize, inverse, exp, ln).
const NORM_EPSILON: f64 = 1e-12;

/// SLERP falls back to NLERP when `|dot|` exceeds this (avoids sin(~0)/sin(~0)).
const SLERP_DOT_THRESHOLD: f64 = 1.0 - 1e-6;

// ─── Quaternion ─────────────────────────────────────────────────────────────

/// Unit quaternion for 3D rotation (Hamilton convention, scalar-first).
///
/// # Layout
/// - `w` — scalar (real) part
/// - `x, y, z` — vector (imaginary) part
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    /// Create a new quaternion from components.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Identity quaternion (no rotation): `(1, 0, 0, 0)`.
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create a unit quaternion from an axis–angle representation.
    ///
    /// The axis is internally normalized. Returns `Err` if the axis has near-zero length.
    ///
    /// # Examples
    ///
    /// ```
    /// use machin_math::quaternion::Quaternion;
    /// use std::f64::consts::PI;
    ///
    /// // 180° around X axis — unnormalized axis is fine
    /// let q = Quaternion::from_axis_angle(&[5.0, 0.0, 0.0], PI).unwrap();
    /// assert!(q.is_unit(1e-12));
    ///
    /// // Zero axis is rejected
    /// assert!(Quaternion::from_axis_angle(&[0.0, 0.0, 0.0], 1.0).is_err());
    /// ```
    pub fn from_axis_angle(axis: &[f64; 3], angle: f64) -> Result<Self, MathError> {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if len < NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "rotation axis has near-zero length".into(),
            ));
        }
        let inv = 1.0 / len;
        let half = angle * 0.5;
        let s = half.sin();
        Ok(Self {
            w: half.cos(),
            x: axis[0] * inv * s,
            y: axis[1] * inv * s,
            z: axis[2] * inv * s,
        })
    }

    /// Convert to a 3×3 rotation matrix.
    ///
    /// Re-normalizes the quaternion before building the matrix to guard against drift.
    ///
    /// # Examples
    ///
    /// ```
    /// use machin_math::quaternion::Quaternion;
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let q = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
    /// let m = q.to_rotation_matrix();
    /// assert_eq!(m.shape(), &[3, 3]);
    /// assert!((m[[0, 1]] + 1.0).abs() < 1e-10); // -sin(90°)
    /// ```
    pub fn to_rotation_matrix(&self) -> Array2<f64> {
        let q = self.normalize().unwrap_or(*self);
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);

        let xx = 2.0 * x * x;
        let yy = 2.0 * y * y;
        let zz = 2.0 * z * z;
        let xy = 2.0 * x * y;
        let xz = 2.0 * x * z;
        let yz = 2.0 * y * z;
        let wx = 2.0 * w * x;
        let wy = 2.0 * w * y;
        let wz = 2.0 * w * z;

        array![
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy]
        ]
    }

    /// Rotate a 3D vector using `q * v * q⁻¹` (optimized, no matrix).
    ///
    /// # Examples
    ///
    /// ```
    /// use machin_math::quaternion::Quaternion;
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let q = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
    /// let v = q.rotate_vector(&[1.0, 0.0, 0.0]);
    /// assert!((v[1] - 1.0).abs() < 1e-10); // X → Y
    /// ```
    pub fn rotate_vector(&self, v: &[f64; 3]) -> [f64; 3] {
        // Rodrigues-like expansion of q*v*q⁻¹ for unit quaternion:
        //   v' = v + 2w(u × v) + 2(u × (u × v))
        // where u = (x, y, z)
        let ux = self.x;
        let uy = self.y;
        let uz = self.z;

        // t = 2(u × v)
        let tx = 2.0 * (uy * v[2] - uz * v[1]);
        let ty = 2.0 * (uz * v[0] - ux * v[2]);
        let tz = 2.0 * (ux * v[1] - uy * v[0]);

        [
            v[0] + self.w * tx + (uy * tz - uz * ty),
            v[1] + self.w * ty + (uz * tx - ux * tz),
            v[2] + self.w * tz + (ux * ty - uy * tx),
        ]
    }

    /// Quaternion conjugate: `(w, -x, -y, -z)`.
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternion inverse: `conjugate / norm²`. Fails if norm ≈ 0.
    pub fn inverse(&self) -> Result<Self, MathError> {
        let ns = self.norm_squared();
        if ns < NORM_EPSILON * NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "cannot invert zero-norm quaternion".into(),
            ));
        }
        let inv = 1.0 / ns;
        Ok(Self {
            w: self.w * inv,
            x: -self.x * inv,
            y: -self.y * inv,
            z: -self.z * inv,
        })
    }

    /// Euclidean norm: `sqrt(w² + x² + y² + z²)`.
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Squared norm (avoids sqrt): `w² + x² + y² + z²`.
    pub fn norm_squared(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Normalize to unit quaternion. Returns `Err` if norm < `NORM_EPSILON`.
    pub fn normalize(&self) -> Result<Self, MathError> {
        let n = self.norm();
        if n < NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "zero-norm quaternion cannot be normalized".into(),
            ));
        }
        let inv = 1.0 / n;
        Ok(Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        })
    }

    /// Dot product of two quaternions.
    pub fn dot(&self, other: &Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Returns `true` if the quaternion is unit within `tolerance`.
    pub fn is_unit(&self, tolerance: f64) -> bool {
        (self.norm_squared() - 1.0).abs() < tolerance
    }

    /// Scalar multiplication: `s * (w, x, y, z)`.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            w: self.w * s,
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Quaternion exponential map.
    ///
    /// For `q = (0, v)`: `exp(q) = (cos(||v||), sin(||v||)/||v|| * v)`.
    /// Uses Taylor expansion `sinc(x) ≈ 1 − x²/6` near `||v|| → 0`.
    pub fn exp(&self) -> Self {
        let vn = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        let ew = self.w.exp();

        let sinc = if vn < 1e-8 {
            // Taylor expansion: sin(x)/x ≈ 1 - x²/6 + x⁴/120
            1.0 - vn * vn / 6.0
        } else {
            vn.sin() / vn
        };

        Self {
            w: ew * vn.cos(),
            x: ew * sinc * self.x,
            y: ew * sinc * self.y,
            z: ew * sinc * self.z,
        }
    }

    /// Quaternion logarithmic map.
    ///
    /// For unit quaternion: `ln(q) = (0, θ * v̂)` where `θ = acos(clamp(w))`.
    /// Returns `Err` if the quaternion has near-zero norm.
    pub fn ln(&self) -> Result<Self, MathError> {
        let n = self.norm();
        if n < NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "cannot take logarithm of zero-norm quaternion".into(),
            ));
        }

        let vn = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        let lnn = n.ln();

        let coeff = if vn < 1e-8 {
            // Near identity: θ/sin(θ) ≈ 1 when v ≈ 0
            // Use acos(w/n)/vn but approximate when vn is small
            1.0 / n
        } else {
            let w_clamped = (self.w / n).clamp(-1.0, 1.0);
            w_clamped.acos() / vn
        };

        Ok(Self {
            w: lnn,
            x: coeff * self.x,
            y: coeff * self.y,
            z: coeff * self.z,
        })
    }

    /// Normalized linear interpolation (NLERP).
    ///
    /// Faster than SLERP but not constant-velocity. Always produces a valid unit quaternion.
    /// Chooses the shorter path by negating if `dot < 0`.
    pub fn nlerp(&self, other: &Self, t: f64) -> Self {
        let d = self.dot(other);
        let other = if d < 0.0 { -*other } else { *other };
        let result = Self {
            w: self.w + t * (other.w - self.w),
            x: self.x + t * (other.x - self.x),
            y: self.y + t * (other.y - self.y),
            z: self.z + t * (other.z - self.z),
        };
        // nlerp always normalizes — safe unless both inputs are zero
        result.normalize().unwrap_or_default()
    }
}

impl Default for Quaternion {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── Operator traits ────────────────────────────────────────────────────────

impl std::ops::Mul for Quaternion {
    type Output = Self;

    /// Hamilton product.
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
    }
}

impl std::ops::Mul<f64> for Quaternion {
    type Output = Self;

    fn mul(self, s: f64) -> Self {
        self.scale(s)
    }
}

impl std::ops::Add for Quaternion {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl std::ops::Sub for Quaternion {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl std::ops::Neg for Quaternion {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            w: -self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::fmt::Display for Quaternion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}i + {}j + {}k", self.w, self.x, self.y, self.z)
    }
}

impl From<Quaternion> for Array1<f64> {
    fn from(q: Quaternion) -> Array1<f64> {
        array![q.w, q.x, q.y, q.z]
    }
}

impl TryFrom<Array1<f64>> for Quaternion {
    type Error = MathError;

    fn try_from(arr: Array1<f64>) -> Result<Self, MathError> {
        if arr.len() != 4 {
            return Err(MathError::DimensionMismatch {
                expected: 4,
                got: arr.len(),
            });
        }
        Ok(Self {
            w: arr[0],
            x: arr[1],
            y: arr[2],
            z: arr[3],
        })
    }
}

// ─── Free functions ─────────────────────────────────────────────────────────

/// Spherical linear interpolation. Returns `None` for near-antipodal quaternions
/// where the rotation path is ambiguous.
///
/// # Examples
///
/// ```
/// use machin_math::quaternion::{Quaternion, try_slerp, slerp};
/// use std::f64::consts::FRAC_PI_2;
///
/// let q1 = Quaternion::identity();
/// let q2 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
///
/// // Midpoint of 0° and 90° = 45°
/// let mid = try_slerp(&q1, &q2, 0.5).unwrap();
/// assert!(mid.is_unit(1e-10));
///
/// // slerp always returns a value (NLERP fallback)
/// let also_mid = slerp(&q1, &q2, 0.5);
/// assert!((mid.dot(&also_mid) - 1.0).abs() < 1e-10);
/// ```
pub fn try_slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Option<Quaternion> {
    let mut d = q1.dot(q2);
    let q2 = if d < 0.0 {
        d = -d;
        -*q2
    } else {
        *q2
    };

    // Near-parallel: NLERP is a fine approximation
    if d > SLERP_DOT_THRESHOLD {
        return Some(q1.nlerp(&q2, t));
    }

    // Near-antipodal: rotation path is ambiguous
    if d < -1.0 + NORM_EPSILON {
        return None;
    }

    let theta = d.clamp(-1.0, 1.0).acos();
    let sin_theta = theta.sin();

    // Should not happen given threshold check, but guard anyway
    if sin_theta.abs() < NORM_EPSILON {
        return Some(q1.nlerp(&q2, t));
    }

    let a = ((1.0 - t) * theta).sin() / sin_theta;
    let b = (t * theta).sin() / sin_theta;

    let result = Quaternion {
        w: a * q1.w + b * q2.w,
        x: a * q1.x + b * q2.x,
        y: a * q1.y + b * q2.y,
        z: a * q1.z + b * q2.z,
    };

    // Re-normalize to guard against drift
    Some(result.normalize().unwrap_or_default())
}

/// SLERP with NLERP fallback for degenerate cases.
///
/// Unlike `try_slerp`, this always returns a valid quaternion.
pub fn slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Quaternion {
    try_slerp(q1, q2, t).unwrap_or_else(|| q1.nlerp(q2, t))
}

// ─── Helper ─────────────────────────────────────────────────────────────────

/// Cross product of two 3D vectors (ndarray does not provide this).
pub(crate) fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn quat_approx_eq(a: &Quaternion, b: &Quaternion) -> bool {
        approx_eq(a.w, b.w) && approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    #[test]
    fn test_identity() {
        let q = Quaternion::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
        assert!(q.is_unit(1e-15));
    }

    #[test]
    fn test_default_is_identity() {
        assert_eq!(Quaternion::default(), Quaternion::identity());
    }

    #[test]
    fn test_norm_and_normalize() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let expected_norm = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert!(approx_eq(q.norm(), expected_norm));

        let qn = q.normalize().unwrap();
        assert!(approx_eq(qn.norm(), 1.0));
    }

    #[test]
    fn test_normalize_zero_fails() {
        let q = Quaternion::new(0.0, 0.0, 0.0, 0.0);
        assert!(q.normalize().is_err());
    }

    #[test]
    fn test_conjugate() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let c = q.conjugate();
        assert_eq!(c.w, 1.0);
        assert_eq!(c.x, -2.0);
        assert_eq!(c.y, -3.0);
        assert_eq!(c.z, -4.0);
    }

    #[test]
    fn test_inverse() {
        let q = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_4).unwrap();
        let qi = q.inverse().unwrap();
        let prod = q * qi;
        assert!(quat_approx_eq(&prod, &Quaternion::identity()));
    }

    #[test]
    fn test_inverse_zero_fails() {
        let q = Quaternion::new(0.0, 0.0, 0.0, 0.0);
        assert!(q.inverse().is_err());
    }

    #[test]
    fn test_hamilton_product_ijk() {
        // i*j = k, j*k = i, k*i = j
        let i = Quaternion::new(0.0, 1.0, 0.0, 0.0);
        let j = Quaternion::new(0.0, 0.0, 1.0, 0.0);
        let k = Quaternion::new(0.0, 0.0, 0.0, 1.0);

        let ij = i * j;
        assert!(quat_approx_eq(&ij, &k));

        let jk = j * k;
        assert!(quat_approx_eq(&jk, &i));

        let ki = k * i;
        assert!(quat_approx_eq(&ki, &j));

        // i*j*k = -1
        let ijk = i * j * k;
        let neg1 = Quaternion::new(-1.0, 0.0, 0.0, 0.0);
        assert!(quat_approx_eq(&ijk, &neg1));
    }

    #[test]
    fn test_from_axis_angle_90deg_z() {
        let q = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        assert!(q.is_unit(1e-12));
        assert!(approx_eq(q.w, (FRAC_PI_4).cos()));
        assert!(approx_eq(q.z, (FRAC_PI_4).sin()));
    }

    #[test]
    fn test_from_axis_angle_zero_axis_fails() {
        assert!(Quaternion::from_axis_angle(&[0.0, 0.0, 0.0], 1.0).is_err());
    }

    #[test]
    fn test_from_axis_angle_normalizes_axis() {
        let q1 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let q2 = Quaternion::from_axis_angle(&[0.0, 0.0, 5.0], FRAC_PI_2).unwrap();
        assert!(quat_approx_eq(&q1, &q2));
    }

    #[test]
    fn test_rotate_vector_90deg_z() {
        // 90° around Z: (1,0,0) → (0,1,0)
        let q = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let v = [1.0, 0.0, 0.0];
        let r = q.rotate_vector(&v);
        assert!(approx_eq(r[0], 0.0));
        assert!(approx_eq(r[1], 1.0));
        assert!(approx_eq(r[2], 0.0));
    }

    #[test]
    fn test_rotate_vector_180deg_x() {
        // 180° around X: (0,1,0) → (0,-1,0)
        let q = Quaternion::from_axis_angle(&[1.0, 0.0, 0.0], PI).unwrap();
        let r = q.rotate_vector(&[0.0, 1.0, 0.0]);
        assert!(approx_eq(r[0], 0.0));
        assert!(approx_eq(r[1], -1.0));
        assert!(approx_eq(r[2], 0.0));
    }

    #[test]
    fn test_to_rotation_matrix() {
        // 90° around Z should produce [[0,-1,0],[1,0,0],[0,0,1]]
        let q = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let m = q.to_rotation_matrix();
        assert!(approx_eq(m[[0, 0]], 0.0));
        assert!(approx_eq(m[[0, 1]], -1.0));
        assert!(approx_eq(m[[1, 0]], 1.0));
        assert!(approx_eq(m[[1, 1]], 0.0));
        assert!(approx_eq(m[[2, 2]], 1.0));
    }

    #[test]
    fn test_rotation_matrix_matches_rotate_vector() {
        let q = Quaternion::from_axis_angle(&[1.0, 1.0, 1.0], 1.23).unwrap();
        let v = [3.0, -1.5, 2.7];
        let r1 = q.rotate_vector(&v);
        let m = q.to_rotation_matrix();
        let va = ndarray::array![v[0], v[1], v[2]];
        let r2 = m.dot(&va);
        assert!(approx_eq(r1[0], r2[0]));
        assert!(approx_eq(r1[1], r2[1]));
        assert!(approx_eq(r1[2], r2[2]));
    }

    #[test]
    fn test_exp_ln_roundtrip() {
        let q = Quaternion::from_axis_angle(&[0.0, 1.0, 0.0], 1.0).unwrap();
        let logged = q.ln().unwrap();
        let recovered = logged.exp();
        let recovered = recovered.normalize().unwrap();
        // exp(ln(q)) should recover q (up to sign — double cover)
        let d = q.dot(&recovered).abs();
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_exp_near_identity() {
        // exp of a very small pure quaternion should be near identity
        let q = Quaternion::new(0.0, 1e-10, 0.0, 0.0);
        let e = q.exp();
        assert!(approx_eq(e.w, 1.0));
        assert!(e.is_unit(1e-8));
    }

    #[test]
    fn test_ln_identity() {
        let q = Quaternion::identity();
        let l = q.ln().unwrap();
        assert!(approx_eq(l.w, 0.0));
        assert!(approx_eq(l.x, 0.0));
        assert!(approx_eq(l.y, 0.0));
        assert!(approx_eq(l.z, 0.0));
    }

    #[test]
    fn test_slerp_endpoints() {
        let q1 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], 0.0).unwrap();
        let q2 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();

        let s0 = slerp(&q1, &q2, 0.0);
        let s1 = slerp(&q1, &q2, 1.0);
        assert!(quat_approx_eq(&s0, &q1));
        assert!(quat_approx_eq(&s1, &q2));
    }

    #[test]
    fn test_slerp_midpoint() {
        let q1 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], 0.0).unwrap();
        let q2 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let mid = slerp(&q1, &q2, 0.5);
        let expected = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_4).unwrap();
        assert!(quat_approx_eq(&mid, &expected));
    }

    #[test]
    fn test_slerp_shortest_path() {
        // q and -q represent the same rotation; SLERP should take the shorter path
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], 0.1).unwrap();
        let q2_neg = -q2;
        let s1 = slerp(&q1, &q2, 0.5);
        let s2 = slerp(&q1, &q2_neg, 0.5);
        // Both should produce approximately the same rotation
        let d = s1.dot(&s2).abs();
        assert!(approx_eq(d, 1.0));
    }

    #[test]
    fn test_try_slerp_near_parallel() {
        // Nearly identical quaternions — should still work (NLERP fallback)
        let q = Quaternion::identity();
        let q2 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], 1e-10).unwrap();
        let result = try_slerp(&q, &q2, 0.5);
        assert!(result.is_some());
        assert!(result.unwrap().is_unit(1e-8));
    }

    #[test]
    fn test_nlerp() {
        let q1 = Quaternion::identity();
        let q2 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let mid = q1.nlerp(&q2, 0.5);
        assert!(mid.is_unit(1e-10));
    }

    #[test]
    fn test_operator_add_sub_neg() {
        let a = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let b = Quaternion::new(5.0, 6.0, 7.0, 8.0);
        let sum = a + b;
        assert_eq!(sum.w, 6.0);
        let diff = a - b;
        assert_eq!(diff.w, -4.0);
        let neg = -a;
        assert_eq!(neg.w, -1.0);
        assert_eq!(neg.x, -2.0);
    }

    #[test]
    fn test_scalar_mul() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let s = q * 2.0;
        assert_eq!(s.w, 2.0);
        assert_eq!(s.z, 8.0);
    }

    #[test]
    fn test_display() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let s = format!("{}", q);
        assert!(s.contains("1"));
        assert!(s.contains("i"));
        assert!(s.contains("j"));
        assert!(s.contains("k"));
    }

    #[test]
    fn test_from_into_array1() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let arr: Array1<f64> = q.into();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0], 1.0);

        let q2 = Quaternion::try_from(arr).unwrap();
        assert_eq!(q, q2);
    }

    #[test]
    fn test_try_from_wrong_len() {
        let arr = ndarray::array![1.0, 2.0, 3.0];
        assert!(Quaternion::try_from(arr).is_err());
    }

    #[test]
    fn test_dot_product() {
        let a = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let b = Quaternion::new(0.0, 1.0, 0.0, 0.0);
        assert!(approx_eq(a.dot(&b), 0.0));
        assert!(approx_eq(a.dot(&a), 1.0));
    }

    #[test]
    fn test_composition_order() {
        // q1 * q2 should apply q2 first, then q1 (like matrices)
        let q_z90 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let q_x90 = Quaternion::from_axis_angle(&[1.0, 0.0, 0.0], FRAC_PI_2).unwrap();

        let v = [1.0, 0.0, 0.0];

        // Apply q_x90 first, then q_z90
        let composed = q_z90 * q_x90;
        let r1 = composed.rotate_vector(&v);

        // Manual: q_x90 on (1,0,0) = (1,0,0), then q_z90 on (1,0,0) = (0,1,0)
        let step1 = q_x90.rotate_vector(&v);
        let r2 = q_z90.rotate_vector(&step1);

        assert!(approx_eq(r1[0], r2[0]));
        assert!(approx_eq(r1[1], r2[1]));
        assert!(approx_eq(r1[2], r2[2]));
    }

    #[test]
    fn test_cross3() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross3(&a, &b);
        assert!(approx_eq(c[0], 0.0));
        assert!(approx_eq(c[1], 0.0));
        assert!(approx_eq(c[2], 1.0));
    }

    #[test]
    fn test_successive_rotations_drift() {
        // After many multiplications, quaternion should still be normalizable
        let q = Quaternion::from_axis_angle(&[1.0, 1.0, 0.0], 0.01).unwrap();
        let mut acc = Quaternion::identity();
        for _ in 0..1000 {
            acc = acc * q;
        }
        // Norm may have drifted, but normalize should recover it
        let n = acc.normalize().unwrap();
        assert!(n.is_unit(1e-10));
    }
}
