//! Dual quaternion algebra for rigid-body transforms.
//!
//! A dual quaternion `dq = q_r + ε * q_d` encodes rotation + translation
//! in 8 elements. Unit constraint: `||q_r|| = 1` AND `q_r · q_d = 0`.
//!
//! # Examples
//!
//! ```
//! use ix_math::quaternion::Quaternion;
//! use ix_math::dual_quaternion::{DualQuaternion, sclerp};
//! use std::f64::consts::FRAC_PI_2;
//!
//! // Rotate 90° around Z, then translate by (0, 0, 5)
//! let rot = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
//! let dq = DualQuaternion::from_rotation_translation(&rot, &[0.0, 0.0, 5.0]).unwrap();
//!
//! // Transform a point: (1,0,0) → rotate → (0,1,0) → translate → (0,1,5)
//! let p = dq.transform_point(&[1.0, 0.0, 0.0]).unwrap();
//! assert!((p[0]).abs() < 1e-10);
//! assert!((p[1] - 1.0).abs() < 1e-10);
//! assert!((p[2] - 5.0).abs() < 1e-10);
//!
//! // Round-trip: extract rotation and translation back
//! let (r, t) = dq.to_rotation_translation();
//! assert!(r.is_unit(1e-10));
//! assert!((t[2] - 5.0).abs() < 1e-10);
//! ```

use crate::error::MathError;
use crate::quaternion::{self, Quaternion};

// ─── Constants ──────────────────────────────────────────────────────────────

const NORM_EPSILON: f64 = 1e-12;

// ─── DualQuaternion ─────────────────────────────────────────────────────────

/// Dual quaternion for 6-DOF rigid-body transforms (rotation + translation).
///
/// Stores a real (rotation) and dual (translation encoding) quaternion part.
/// For a unit dual quaternion: `||real|| = 1` and `real · dual = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualQuaternion {
    pub real: Quaternion,
    pub dual: Quaternion,
}

impl DualQuaternion {
    /// Identity transform (no rotation, no translation).
    pub fn identity() -> Self {
        Self {
            real: Quaternion::identity(),
            dual: Quaternion::new(0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Create from a unit rotation quaternion and a translation vector.
    ///
    /// The rotation quaternion is re-normalized internally.
    /// `dual = 0.5 * t_quat * rotation` where `t_quat = (0, tx, ty, tz)`.
    pub fn from_rotation_translation(
        rotation: &Quaternion,
        translation: &[f64; 3],
    ) -> Result<Self, MathError> {
        let real = rotation.normalize()?;
        let t = Quaternion::new(0.0, translation[0], translation[1], translation[2]);
        let dual = (t * real).scale(0.5);
        Ok(Self { real, dual })
    }

    /// Extract rotation quaternion and translation vector.
    pub fn to_rotation_translation(&self) -> (Quaternion, [f64; 3]) {
        let real = self.real;
        // t_quat = 2 * dual * conjugate(real)
        let t = (self.dual * real.conjugate()).scale(2.0);
        (real, [t.x, t.y, t.z])
    }

    /// Dual quaternion conjugate: `(conj(real), conj(dual))`.
    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real.conjugate(),
            dual: self.dual.conjugate(),
        }
    }

    /// Returns `(real_norm, dual_scalar)` where `dual_scalar = real · dual / real_norm`.
    pub fn norm(&self) -> (f64, f64) {
        let rn = self.real.norm();
        if rn < NORM_EPSILON {
            return (0.0, 0.0);
        }
        let ds = self.real.dot(&self.dual) / rn;
        (rn, ds)
    }

    /// Normalize to a unit dual quaternion.
    ///
    /// Ensures `||real|| = 1` and projects out the parallel component of `dual`
    /// to enforce `real · dual = 0`.
    pub fn normalize(&self) -> Result<Self, MathError> {
        let real = self.real.normalize()?;
        // Project out parallel component: dual = dual - (real · dual) * real
        let d = real.dot(&self.dual);
        let dual = Quaternion::new(
            self.dual.w - d * real.w,
            self.dual.x - d * real.x,
            self.dual.y - d * real.y,
            self.dual.z - d * real.z,
        );
        // Scale dual by 1/real_norm (real was already normalized, but dual wasn't scaled)
        let rn = self.real.norm();
        let inv = 1.0 / rn;
        let dual = dual.scale(inv);
        Ok(Self { real, dual })
    }

    /// Inverse of the dual quaternion.
    ///
    /// For unit dual quaternions this equals the conjugate, but we use the full
    /// formula to handle floating-point drift.
    pub fn inverse(&self) -> Result<Self, MathError> {
        let ns = self.real.norm_squared();
        if ns < NORM_EPSILON * NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "cannot invert zero-norm dual quaternion".into(),
            ));
        }
        let real_inv = self.real.inverse()?;
        // dual_inv = -real_inv * dual * real_inv
        let dual_inv = (real_inv * self.dual * real_inv).scale(-1.0);
        Ok(Self {
            real: real_inv,
            dual: dual_inv,
        })
    }

    /// Transform a 3D point by this dual quaternion.
    ///
    /// Applies `dq * p * conj(dq)` where `p = (1, point) + ε(0, 0, 0, 0)`.
    pub fn transform_point(&self, point: &[f64; 3]) -> Result<[f64; 3], MathError> {
        let (rot, trans) = self.to_rotation_translation();
        let rotated = rot.rotate_vector(point);
        Ok([
            rotated[0] + trans[0],
            rotated[1] + trans[1],
            rotated[2] + trans[2],
        ])
    }
}

impl Default for DualQuaternion {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── Operator traits ────────────────────────────────────────────────────────

impl std::ops::Mul for DualQuaternion {
    type Output = Self;

    /// Dual quaternion multiplication:
    /// `(r1, d1) * (r2, d2) = (r1*r2, r1*d2 + d1*r2)`
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl std::ops::Neg for DualQuaternion {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            real: -self.real,
            dual: -self.dual,
        }
    }
}

// ─── Free functions ─────────────────────────────────────────────────────────

/// Screw linear interpolation (ScLERP) between two unit dual quaternions.
///
/// Interpolates both rotation and translation along a screw axis.
/// Falls back to component-wise SLERP + LERP for pure-translation or
/// near-identity cases.
///
/// # Examples
///
/// ```
/// use ix_math::quaternion::Quaternion;
/// use ix_math::dual_quaternion::{DualQuaternion, sclerp};
///
/// let dq1 = DualQuaternion::from_rotation_translation(
///     &Quaternion::identity(), &[0.0, 0.0, 0.0],
/// ).unwrap();
/// let dq2 = DualQuaternion::from_rotation_translation(
///     &Quaternion::identity(), &[10.0, 0.0, 0.0],
/// ).unwrap();
///
/// // Halfway interpolation of pure translation
/// let mid = sclerp(&dq1, &dq2, 0.5).unwrap();
/// let (_, t) = mid.to_rotation_translation();
/// assert!((t[0] - 5.0).abs() < 1e-10);
/// ```
pub fn sclerp(
    dq1: &DualQuaternion,
    dq2: &DualQuaternion,
    t: f64,
) -> Result<DualQuaternion, MathError> {
    // Compute relative transform: diff = dq1⁻¹ * dq2
    let dq1_inv = dq1.inverse()?;
    let mut diff = dq1_inv * *dq2;

    // Shortest path: negate if real.w < 0
    if diff.real.w < 0.0 {
        diff = -diff;
    }

    // Extract screw parameters from diff
    let half_angle = diff.real.w.clamp(-1.0, 1.0).acos();
    let sin_half = half_angle.sin();

    let powered = if sin_half.abs() < NORM_EPSILON {
        // Near-identity or pure translation: linear interpolate the dual part
        let real = quaternion::slerp(&Quaternion::identity(), &diff.real, t);
        let dual = diff.dual.scale(t);
        DualQuaternion { real, dual }
    } else {
        // Full screw decomposition
        let inv_sin = 1.0 / sin_half;

        // Screw axis from real part
        let axis = [
            diff.real.x * inv_sin,
            diff.real.y * inv_sin,
            diff.real.z * inv_sin,
        ];

        // Pitch from dual part: pitch = -2 * dual.w / sin(half_angle)
        let pitch = -2.0 * diff.dual.w * inv_sin;
        let cos_half = half_angle.cos();

        // Moment from dual part: m = (dual.xyz - (d/2)*cos(θ/2)*l) / sin(θ/2)
        let half_pitch_cos = pitch * 0.5 * cos_half;
        let moment = [
            (diff.dual.x - half_pitch_cos * axis[0]) * inv_sin,
            (diff.dual.y - half_pitch_cos * axis[1]) * inv_sin,
            (diff.dual.z - half_pitch_cos * axis[2]) * inv_sin,
        ];

        // Scale by t
        let t_angle = half_angle * t;
        let t_pitch = pitch * t;

        let sin_t = t_angle.sin();
        let cos_t = t_angle.cos();

        let real = Quaternion::new(cos_t, sin_t * axis[0], sin_t * axis[1], sin_t * axis[2]);

        let dual = Quaternion::new(
            -0.5 * t_pitch * sin_t,
            sin_t * moment[0] + 0.5 * t_pitch * cos_t * axis[0],
            sin_t * moment[1] + 0.5 * t_pitch * cos_t * axis[1],
            sin_t * moment[2] + 0.5 * t_pitch * cos_t * axis[2],
        );

        DualQuaternion { real, dual }
    };

    // result = dq1 * powered, re-normalize
    let result = *dq1 * powered;
    result.normalize()
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn vec3_approx_eq(a: &[f64; 3], b: &[f64; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_identity() {
        let dq = DualQuaternion::identity();
        assert_eq!(dq.real, Quaternion::identity());
        assert_eq!(dq.dual, Quaternion::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_default_is_identity() {
        assert_eq!(DualQuaternion::default(), DualQuaternion::identity());
    }

    #[test]
    fn test_pure_translation() {
        let rot = Quaternion::identity();
        let trans = [1.0, 2.0, 3.0];
        let dq = DualQuaternion::from_rotation_translation(&rot, &trans).unwrap();

        let (r, t) = dq.to_rotation_translation();
        assert!(r.is_unit(1e-12));
        assert!(vec3_approx_eq(&t, &trans));
    }

    #[test]
    fn test_pure_rotation() {
        let rot = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let trans = [0.0, 0.0, 0.0];
        let dq = DualQuaternion::from_rotation_translation(&rot, &trans).unwrap();

        let (r, t) = dq.to_rotation_translation();
        assert!(approx_eq(r.dot(&rot).abs(), 1.0));
        assert!(vec3_approx_eq(&t, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_rotation_translation_roundtrip() {
        let rot = Quaternion::from_axis_angle(&[1.0, 0.0, 0.0], FRAC_PI_4).unwrap();
        let trans = [5.0, -3.0, 2.0];
        let dq = DualQuaternion::from_rotation_translation(&rot, &trans).unwrap();

        let (r, t) = dq.to_rotation_translation();
        assert!(approx_eq(r.dot(&rot).abs(), 1.0));
        assert!(vec3_approx_eq(&t, &trans));
    }

    #[test]
    fn test_transform_point_translation_only() {
        let dq =
            DualQuaternion::from_rotation_translation(&Quaternion::identity(), &[1.0, 2.0, 3.0])
                .unwrap();
        let p = dq.transform_point(&[0.0, 0.0, 0.0]).unwrap();
        assert!(vec3_approx_eq(&p, &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_transform_point_rotation_only() {
        let rot = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let dq = DualQuaternion::from_rotation_translation(&rot, &[0.0, 0.0, 0.0]).unwrap();
        let p = dq.transform_point(&[1.0, 0.0, 0.0]).unwrap();
        assert!(vec3_approx_eq(&p, &[0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_transform_point_combined() {
        // Rotate 90° around Z, then translate (0, 0, 5)
        let rot = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let dq = DualQuaternion::from_rotation_translation(&rot, &[0.0, 0.0, 5.0]).unwrap();
        let p = dq.transform_point(&[1.0, 0.0, 0.0]).unwrap();
        // (1,0,0) rotated 90° Z → (0,1,0), then + (0,0,5) → (0,1,5)
        assert!(vec3_approx_eq(&p, &[0.0, 1.0, 5.0]));
    }

    #[test]
    fn test_conjugate() {
        let rot = Quaternion::from_axis_angle(&[0.0, 1.0, 0.0], 1.0).unwrap();
        let dq = DualQuaternion::from_rotation_translation(&rot, &[1.0, 2.0, 3.0]).unwrap();
        let c = dq.conjugate();
        assert_eq!(c.real, dq.real.conjugate());
        assert_eq!(c.dual, dq.dual.conjugate());
    }

    #[test]
    fn test_inverse() {
        let rot = Quaternion::from_axis_angle(&[1.0, 1.0, 0.0], 0.7).unwrap();
        let dq = DualQuaternion::from_rotation_translation(&rot, &[2.0, -1.0, 3.0]).unwrap();
        let inv = dq.inverse().unwrap();
        let prod = dq * inv;
        let (r, t) = prod.to_rotation_translation();
        assert!(approx_eq(r.dot(&Quaternion::identity()).abs(), 1.0));
        assert!(vec3_approx_eq(&t, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_normalize() {
        let rot = Quaternion::new(0.9, 0.1, 0.2, 0.3); // not unit
        let dual = Quaternion::new(0.1, 0.5, 0.3, 0.2);
        let dq = DualQuaternion { real: rot, dual };
        let n = dq.normalize().unwrap();
        assert!(n.real.is_unit(1e-10));
        // Orthogonality: real · dual ≈ 0
        assert!(n.real.dot(&n.dual).abs() < 1e-10);
    }

    #[test]
    fn test_mul_composition() {
        // dq1 then dq2: transform a point through both
        let dq1 =
            DualQuaternion::from_rotation_translation(&Quaternion::identity(), &[1.0, 0.0, 0.0])
                .unwrap();
        let dq2 =
            DualQuaternion::from_rotation_translation(&Quaternion::identity(), &[0.0, 2.0, 0.0])
                .unwrap();
        let composed = dq2 * dq1;
        let p = composed.transform_point(&[0.0, 0.0, 0.0]).unwrap();
        assert!(vec3_approx_eq(&p, &[1.0, 2.0, 0.0]));
    }

    #[test]
    fn test_sclerp_endpoints() {
        let dq1 =
            DualQuaternion::from_rotation_translation(&Quaternion::identity(), &[0.0, 0.0, 0.0])
                .unwrap();
        let rot2 = Quaternion::from_axis_angle(&[0.0, 0.0, 1.0], FRAC_PI_2).unwrap();
        let dq2 = DualQuaternion::from_rotation_translation(&rot2, &[1.0, 2.0, 3.0]).unwrap();

        let s0 = sclerp(&dq1, &dq2, 0.0).unwrap();
        let s1 = sclerp(&dq1, &dq2, 1.0).unwrap();

        let (_, t0) = s0.to_rotation_translation();
        let (_, t1) = s1.to_rotation_translation();
        assert!(vec3_approx_eq(&t0, &[0.0, 0.0, 0.0]));
        assert!(vec3_approx_eq(&t1, &[1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_sclerp_pure_translation() {
        let dq1 =
            DualQuaternion::from_rotation_translation(&Quaternion::identity(), &[0.0, 0.0, 0.0])
                .unwrap();
        let dq2 =
            DualQuaternion::from_rotation_translation(&Quaternion::identity(), &[10.0, 0.0, 0.0])
                .unwrap();

        let mid = sclerp(&dq1, &dq2, 0.5).unwrap();
        let (r, t) = mid.to_rotation_translation();
        assert!(r.is_unit(1e-10));
        assert!(approx_eq(t[0], 5.0));
        assert!(approx_eq(t[1], 0.0));
        assert!(approx_eq(t[2], 0.0));
    }

    #[test]
    fn test_neg() {
        let dq =
            DualQuaternion::from_rotation_translation(&Quaternion::identity(), &[1.0, 2.0, 3.0])
                .unwrap();
        let neg = -dq;
        assert_eq!(neg.real, -dq.real);
        assert_eq!(neg.dual, -dq.dual);
    }

    #[test]
    fn test_unit_constraint_after_from_rotation_translation() {
        let rot = Quaternion::from_axis_angle(&[1.0, 1.0, 1.0], 1.5).unwrap();
        let dq = DualQuaternion::from_rotation_translation(&rot, &[4.0, -2.0, 7.0]).unwrap();
        assert!(dq.real.is_unit(1e-12));
        // real · dual should be 0 for unit dual quaternion
        assert!(dq.real.dot(&dq.dual).abs() < 1e-10);
    }
}
