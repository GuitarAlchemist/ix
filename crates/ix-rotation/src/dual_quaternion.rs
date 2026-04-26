//! Dual quaternion representation for rigid body transformations (rotation + translation).

use crate::quaternion::Quaternion;

/// A dual quaternion: q = q_real + ε q_dual.
/// Encodes both rotation and translation in a single algebraic object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualQuaternion {
    /// The real (rotation) part.
    pub real: Quaternion,
    /// The dual (translation-encoding) part.
    pub dual: Quaternion,
}

impl DualQuaternion {
    /// Create a dual quaternion from a unit rotation quaternion and a translation vector.
    pub fn from_rotation_translation(q: Quaternion, t: [f64; 3]) -> Self {
        let q = q.normalize();
        // dual = 0.5 * t_quat * real
        let t_quat = Quaternion::new(0.0, t[0], t[1], t[2]);
        let half_t = Quaternion::new(
            t_quat.w * 0.5,
            t_quat.x * 0.5,
            t_quat.y * 0.5,
            t_quat.z * 0.5,
        );
        let dual = half_t.mul(&q);
        Self { real: q, dual }
    }

    /// Transform a 3D point by this dual quaternion (rotation then translation).
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        // First rotate
        let rotated = self.real.rotate_point(p);
        // Extract translation: t = 2 * dual * real*
        let conj = self.real.conjugate();
        let t_quat = Quaternion::new(
            self.dual.w * 2.0,
            self.dual.x * 2.0,
            self.dual.y * 2.0,
            self.dual.z * 2.0,
        )
        .mul(&conj);
        [
            rotated[0] + t_quat.x,
            rotated[1] + t_quat.y,
            rotated[2] + t_quat.z,
        ]
    }

    /// Dual quaternion multiplication.
    pub fn mul(&self, other: &DualQuaternion) -> DualQuaternion {
        DualQuaternion {
            real: self.real.mul(&other.real),
            dual: Quaternion::new(
                self.real.mul(&other.dual).w + self.dual.mul(&other.real).w,
                self.real.mul(&other.dual).x + self.dual.mul(&other.real).x,
                self.real.mul(&other.dual).y + self.dual.mul(&other.real).y,
                self.real.mul(&other.dual).z + self.dual.mul(&other.real).z,
            ),
        }
    }

    /// Dual quaternion conjugate (conjugate both parts).
    pub fn conjugate(&self) -> DualQuaternion {
        DualQuaternion {
            real: self.real.conjugate(),
            dual: self.dual.conjugate(),
        }
    }

    /// Normalize the dual quaternion so the real part is unit length.
    pub fn normalize(&self) -> DualQuaternion {
        let n = self.real.norm();
        if n < 1e-12 {
            return *self;
        }
        DualQuaternion {
            real: Quaternion::new(
                self.real.w / n,
                self.real.x / n,
                self.real.y / n,
                self.real.z / n,
            ),
            dual: Quaternion::new(
                self.dual.w / n,
                self.dual.x / n,
                self.dual.y / n,
                self.dual.z / n,
            ),
        }
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
    fn test_pure_translation() {
        let dq = DualQuaternion::from_rotation_translation(Quaternion::identity(), [1.0, 2.0, 3.0]);
        let r = dq.transform_point([0.0, 0.0, 0.0]);
        assert!(approx_point(r, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_pure_rotation() {
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let dq = DualQuaternion::from_rotation_translation(q, [0.0, 0.0, 0.0]);
        let r = dq.transform_point([1.0, 0.0, 0.0]);
        assert!(approx_point(r, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_composition() {
        // Translate by (1,0,0) then rotate 90° around Z
        let dq1 =
            DualQuaternion::from_rotation_translation(Quaternion::identity(), [1.0, 0.0, 0.0]);
        let q_rot = Quaternion::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let dq2 = DualQuaternion::from_rotation_translation(q_rot, [0.0, 0.0, 0.0]);

        // dq2 * dq1 applies dq1 first, then dq2
        let dq_combined = dq2.mul(&dq1);
        let r = dq_combined.transform_point([0.0, 0.0, 0.0]);
        // (0,0,0) -> translate -> (1,0,0) -> rotate 90° Z -> (0,1,0)
        assert!(approx_point(r, [0.0, 1.0, 0.0]));
    }
}
