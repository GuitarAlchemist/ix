//! Inverse kinematics for articulated chains.
//!
//! Supports both forward kinematics (joint angles → end-effector pose) and
//! inverse kinematics (target pose → joint angles) using two methods:
//!
//! - **CCD** (Cyclic Coordinate Descent): fast, iterative, good for real-time
//! - **Jacobian** (Damped Least-Squares): precise, uses geometric Jacobian
//!
//! Joint transforms use SO(3)/SE(3) from the `lie` module.
//!
//! # Examples
//!
//! ```
//! use machin_dynamics::ik::{Chain, Joint, JointType};
//! use std::f64::consts::FRAC_PI_2;
//!
//! // Simple 2-link planar arm (2 revolute joints around Z, links along X)
//! let chain = Chain::new(vec![
//!     Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
//!     Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
//! ]);
//!
//! let angles = vec![FRAC_PI_2, 0.0]; // first joint 90°, second 0°
//! let pose = chain.forward(&angles);
//! // End-effector should be at approximately (0, 2, 0)
//! assert!((pose[0]).abs() < 1e-6);
//! assert!((pose[1] - 2.0).abs() < 1e-6);
//! ```

use ndarray::{Array1, Array2};

use crate::error::DynamicsError;
use crate::lie;

/// Type of joint in the kinematic chain.
#[derive(Debug, Clone, Copy)]
pub enum JointType {
    /// Revolute joint: rotates around an axis. Parameter = angle (radians).
    Revolute,
    /// Prismatic joint: translates along an axis. Parameter = displacement.
    Prismatic,
}

/// A single joint in a kinematic chain.
#[derive(Debug, Clone)]
pub struct Joint {
    /// Type of joint.
    pub joint_type: JointType,
    /// Axis of rotation (revolute) or translation (prismatic), in local frame.
    pub axis: [f64; 3],
    /// Offset from previous joint's frame to this joint's origin (link length).
    pub offset: [f64; 3],
    /// Joint limits: (min, max). `None` means unlimited.
    pub limits: Option<(f64, f64)>,
}

impl Joint {
    /// Create a revolute joint with given rotation axis and link offset.
    pub fn revolute(axis: [f64; 3], offset: [f64; 3]) -> Self {
        Self {
            joint_type: JointType::Revolute,
            axis: normalize_axis(axis),
            offset,
            limits: None,
        }
    }

    /// Create a prismatic joint with given translation axis and link offset.
    pub fn prismatic(axis: [f64; 3], offset: [f64; 3]) -> Self {
        Self {
            joint_type: JointType::Prismatic,
            axis: normalize_axis(axis),
            offset,
            limits: None,
        }
    }

    /// Set joint limits.
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.limits = Some((min, max));
        self
    }

    /// Compute the 4×4 homogeneous transform for this joint at parameter `q`.
    ///
    /// Convention: rotate/translate first (joint motion), then extend link (offset).
    /// T = JointMotion * Trans(offset)
    ///
    /// This matches standard robotics convention where the link extends
    /// after the joint rotates, so the last joint's rotation matters.
    fn transform(&self, q: f64) -> Array2<f64> {
        // Link offset translation
        let mut link_t = Array2::eye(4);
        link_t[[0, 3]] = self.offset[0];
        link_t[[1, 3]] = self.offset[1];
        link_t[[2, 3]] = self.offset[2];

        match self.joint_type {
            JointType::Revolute => {
                // Rotation around axis by angle q, then translate along link
                let omega = [self.axis[0] * q, self.axis[1] * q, self.axis[2] * q];
                let r = lie::so3_exp(&omega);
                let mut joint_t = Array2::eye(4);
                joint_t.slice_mut(ndarray::s![..3, ..3]).assign(&r);
                joint_t.dot(&link_t)
            }
            JointType::Prismatic => {
                // Translation along axis by distance q, then link offset
                let mut joint_t = Array2::eye(4);
                joint_t[[0, 3]] = self.axis[0] * q;
                joint_t[[1, 3]] = self.axis[1] * q;
                joint_t[[2, 3]] = self.axis[2] * q;
                joint_t.dot(&link_t)
            }
        }
    }

    /// Clamp parameter to joint limits.
    fn clamp(&self, q: f64) -> f64 {
        match self.limits {
            Some((min, max)) => q.clamp(min, max),
            None => q,
        }
    }
}

/// An articulated kinematic chain.
#[derive(Debug, Clone)]
pub struct Chain {
    /// Joints in order from base to end-effector.
    pub joints: Vec<Joint>,
}

impl Chain {
    /// Create a new kinematic chain from a list of joints.
    pub fn new(joints: Vec<Joint>) -> Self {
        Self { joints }
    }

    /// Number of degrees of freedom (number of joints).
    pub fn dof(&self) -> usize {
        self.joints.len()
    }

    /// Forward kinematics: compute end-effector position from joint parameters.
    ///
    /// Returns the 3D position `[x, y, z]` of the end-effector.
    pub fn forward(&self, params: &[f64]) -> [f64; 3] {
        let t = self.forward_transform(params);
        [t[[0, 3]], t[[1, 3]], t[[2, 3]]]
    }

    /// Forward kinematics: compute the full 4×4 end-effector transform.
    pub fn forward_transform(&self, params: &[f64]) -> Array2<f64> {
        assert_eq!(params.len(), self.joints.len(), "params must match joint count");
        let mut t = Array2::eye(4);
        for (joint, &q) in self.joints.iter().zip(params.iter()) {
            let q = joint.clamp(q);
            t = t.dot(&joint.transform(q));
        }
        t
    }

    /// Compute the 4×4 transform up to (but NOT including) joint `i`.
    ///
    /// This gives the frame at joint i's pivot point, before its rotation.
    /// Used by IK solvers to find joint positions and world-space axes.
    fn pre_joint_transform(&self, params: &[f64], joint_idx: usize) -> Array2<f64> {
        let mut t = Array2::eye(4);
        for (idx, (joint, &q)) in self.joints.iter().zip(params.iter()).enumerate() {
            if idx >= joint_idx {
                break;
            }
            let q = joint.clamp(q);
            t = t.dot(&joint.transform(q));
        }
        t
    }

    /// Inverse kinematics using Cyclic Coordinate Descent (CCD).
    ///
    /// Iteratively adjusts each joint to minimize end-effector distance to target.
    /// Fast and robust, good for real-time applications.
    ///
    /// - `target`: desired end-effector position [x, y, z]
    /// - `initial`: initial joint parameters (starting guess)
    /// - `max_iter`: maximum number of CCD iterations
    /// - `tolerance`: convergence threshold (distance to target)
    ///
    /// Returns the solved joint parameters, or error if not converged.
    pub fn solve_ccd(
        &self,
        target: &[f64; 3],
        initial: &[f64],
        max_iter: usize,
        tolerance: f64,
    ) -> Result<Vec<f64>, DynamicsError> {
        assert_eq!(initial.len(), self.joints.len());
        let mut params: Vec<f64> = initial.to_vec();
        let target_arr = Array1::from_vec(vec![target[0], target[1], target[2]]);

        for _iter in 0..max_iter {
            // Check convergence
            let ee = self.forward(&params);
            let err = ((ee[0] - target[0]).powi(2)
                + (ee[1] - target[1]).powi(2)
                + (ee[2] - target[2]).powi(2))
            .sqrt();

            if err < tolerance {
                return Ok(params);
            }

            // Iterate joints from end-effector to base
            for i in (0..self.joints.len()).rev() {
                // Pre-joint transform: frame at joint i's pivot, before its rotation
                let t_pre = self.pre_joint_transform(&params, i);
                let joint_pos = Array1::from_vec(vec![
                    t_pre[[0, 3]],
                    t_pre[[1, 3]],
                    t_pre[[2, 3]],
                ]);
                let r_pre = t_pre.slice(ndarray::s![..3, ..3]).to_owned();
                let local_axis = Array1::from_vec(vec![
                    self.joints[i].axis[0],
                    self.joints[i].axis[1],
                    self.joints[i].axis[2],
                ]);
                let world_axis = r_pre.dot(&local_axis);

                match self.joints[i].joint_type {
                    JointType::Revolute => {
                        // Get current end-effector position
                        let ee = self.forward(&params);
                        let ee_pos_arr = Array1::from_vec(vec![ee[0], ee[1], ee[2]]);

                        // Vectors from joint pivot to end-effector and to target
                        let to_ee = &ee_pos_arr - &joint_pos;
                        let to_target = &target_arr - &joint_pos;

                        let to_ee_len = to_ee.dot(&to_ee).sqrt();
                        let to_target_len = to_target.dot(&to_target).sqrt();

                        if to_ee_len < 1e-12 || to_target_len < 1e-12 {
                            continue;
                        }

                        // Project vectors onto the plane perpendicular to the joint axis
                        let ee_proj = project_onto_plane(&to_ee, &world_axis);
                        let target_proj = project_onto_plane(&to_target, &world_axis);

                        let ee_proj_len = ee_proj.dot(&ee_proj).sqrt();
                        let target_proj_len = target_proj.dot(&target_proj).sqrt();

                        if ee_proj_len < 1e-12 || target_proj_len < 1e-12 {
                            continue;
                        }

                        // Angle between projections
                        let cos_angle = ee_proj.dot(&target_proj)
                            / (ee_proj_len * target_proj_len);
                        let cos_angle = cos_angle.clamp(-1.0, 1.0);
                        let mut angle = cos_angle.acos();

                        // Determine sign using cross product
                        let cross = cross3(&ee_proj, &target_proj);
                        if cross.dot(&world_axis) < 0.0 {
                            angle = -angle;
                        }

                        params[i] = self.joints[i].clamp(params[i] + angle);
                    }
                    JointType::Prismatic => {
                        // For prismatic joints: adjust displacement to reduce error
                        let ee = self.forward(&params);
                        let ee_arr = Array1::from_vec(vec![ee[0], ee[1], ee[2]]);
                        let error = &target_arr - &ee_arr;

                        // Project error onto joint axis
                        let delta = error.dot(&world_axis);
                        params[i] = self.joints[i].clamp(params[i] + delta);
                    }
                }
            }
        }

        // Check final convergence
        let ee = self.forward(&params);
        let err = ((ee[0] - target[0]).powi(2)
            + (ee[1] - target[1]).powi(2)
            + (ee[2] - target[2]).powi(2))
        .sqrt();

        if err < tolerance {
            Ok(params)
        } else {
            Err(DynamicsError::NumericalError(format!(
                "CCD did not converge: error = {:.6} (tolerance = {:.6})",
                err, tolerance
            )))
        }
    }

    /// Inverse kinematics using Damped Least-Squares (Jacobian method).
    ///
    /// Uses the geometric Jacobian with Levenberg-Marquardt damping.
    /// More precise than CCD, better for high-DOF chains.
    ///
    /// - `target`: desired end-effector position [x, y, z]
    /// - `initial`: initial joint parameters
    /// - `max_iter`: maximum iterations
    /// - `tolerance`: convergence threshold
    /// - `damping`: damping factor λ (typical: 0.01–1.0)
    pub fn solve_jacobian(
        &self,
        target: &[f64; 3],
        initial: &[f64],
        max_iter: usize,
        tolerance: f64,
        damping: f64,
    ) -> Result<Vec<f64>, DynamicsError> {
        assert_eq!(initial.len(), self.joints.len());
        let mut params: Vec<f64> = initial.to_vec();
        let n = self.joints.len();

        for _iter in 0..max_iter {
            let ee = self.forward(&params);
            let error = Array1::from_vec(vec![
                target[0] - ee[0],
                target[1] - ee[1],
                target[2] - ee[2],
            ]);

            let err_norm = error.dot(&error).sqrt();
            if err_norm < tolerance {
                return Ok(params);
            }

            // Build 3×n Jacobian
            let jac = self.jacobian(&params);

            // Damped least-squares: Δq = J^T (J J^T + λ²I)^{-1} e
            let jjt = jac.dot(&jac.t());
            let mut damped = jjt;
            for i in 0..3 {
                damped[[i, i]] += damping * damping;
            }

            // Solve (JJ^T + λ²I) x = e
            let x = solve_3x3(&damped, &error)?;

            // Δq = J^T x
            let delta_q = jac.t().dot(&x);

            // Update params
            for i in 0..n {
                params[i] = self.joints[i].clamp(params[i] + delta_q[i]);
            }
        }

        let ee = self.forward(&params);
        let err = ((ee[0] - target[0]).powi(2)
            + (ee[1] - target[1]).powi(2)
            + (ee[2] - target[2]).powi(2))
        .sqrt();

        if err < tolerance {
            Ok(params)
        } else {
            Err(DynamicsError::NumericalError(format!(
                "Jacobian IK did not converge: error = {:.6} (tolerance = {:.6})",
                err, tolerance
            )))
        }
    }

    /// Compute the 3×n geometric Jacobian (position part only).
    ///
    /// Column i is the contribution of joint i to end-effector velocity.
    pub fn jacobian(&self, params: &[f64]) -> Array2<f64> {
        let n = self.joints.len();
        let mut jac = Array2::zeros((3, n));

        let ee = self.forward(params);
        let ee_arr = Array1::from_vec(vec![ee[0], ee[1], ee[2]]);

        for i in 0..n {
            // Pre-joint transform: frame at joint i's pivot
            let t_pre = self.pre_joint_transform(params, i);
            let joint_pos = Array1::from_vec(vec![
                t_pre[[0, 3]],
                t_pre[[1, 3]],
                t_pre[[2, 3]],
            ]);
            let r_pre = t_pre.slice(ndarray::s![..3, ..3]).to_owned();
            let local_axis = Array1::from_vec(vec![
                self.joints[i].axis[0],
                self.joints[i].axis[1],
                self.joints[i].axis[2],
            ]);
            let world_axis = r_pre.dot(&local_axis);

            match self.joints[i].joint_type {
                JointType::Revolute => {
                    // J_i = axis × (ee - joint_pos)
                    let r = &ee_arr - &joint_pos;
                    let col = cross3(&world_axis, &r);
                    jac[[0, i]] = col[0];
                    jac[[1, i]] = col[1];
                    jac[[2, i]] = col[2];
                }
                JointType::Prismatic => {
                    // J_i = axis (linear velocity along axis)
                    jac[[0, i]] = world_axis[0];
                    jac[[1, i]] = world_axis[1];
                    jac[[2, i]] = world_axis[2];
                }
            }
        }

        jac
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn normalize_axis(a: [f64; 3]) -> [f64; 3] {
    let len = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if len < 1e-15 {
        return [0.0, 0.0, 1.0]; // default to Z
    }
    [a[0] / len, a[1] / len, a[2] / len]
}

fn cross3(a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    Array1::from_vec(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

fn project_onto_plane(v: &Array1<f64>, normal: &Array1<f64>) -> Array1<f64> {
    let dot = v.dot(normal);
    v - &(normal * dot)
}

/// Solve 3×3 linear system Ax = b via Cramer's rule.
fn solve_3x3(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, DynamicsError> {
    let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
        - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
        + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);

    if det.abs() < 1e-15 {
        return Err(DynamicsError::NumericalError(
            "Singular matrix in Jacobian solve".into(),
        ));
    }

    let inv_det = 1.0 / det;

    let x0 = inv_det
        * (b[0] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
            - a[[0, 1]] * (b[1] * a[[2, 2]] - a[[1, 2]] * b[2])
            + a[[0, 2]] * (b[1] * a[[2, 1]] - a[[1, 1]] * b[2]));

    let x1 = inv_det
        * (a[[0, 0]] * (b[1] * a[[2, 2]] - a[[1, 2]] * b[2])
            - b[0] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
            + a[[0, 2]] * (a[[1, 0]] * b[2] - b[1] * a[[2, 0]]));

    let x2 = inv_det
        * (a[[0, 0]] * (a[[1, 1]] * b[2] - b[1] * a[[2, 1]])
            - a[[0, 1]] * (a[[1, 0]] * b[2] - b[1] * a[[2, 0]])
            + b[0] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]));

    Ok(Array1::from_vec(vec![x0, x1, x2]))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    const TOL: f64 = 1e-6;

    // ─── Joint construction ─────────────────────────────────────────────

    #[test]
    fn test_revolute_joint_creation() {
        let j = Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
        assert!(matches!(j.joint_type, JointType::Revolute));
        assert!((j.axis[2] - 1.0).abs() < TOL);
        assert!(j.limits.is_none());
    }

    #[test]
    fn test_prismatic_joint_creation() {
        let j = Joint::prismatic([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(matches!(j.joint_type, JointType::Prismatic));
        assert!((j.axis[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn test_joint_limits() {
        let j = Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
            .with_limits(-PI, PI);
        assert_eq!(j.limits, Some((-PI, PI)));
        assert!((j.clamp(10.0) - PI).abs() < TOL);
        assert!((j.clamp(-10.0) - (-PI)).abs() < TOL);
        assert!((j.clamp(0.5) - 0.5).abs() < TOL);
    }

    #[test]
    fn test_axis_normalization() {
        let j = Joint::revolute([0.0, 0.0, 2.0], [1.0, 0.0, 0.0]);
        let len = (j.axis[0].powi(2) + j.axis[1].powi(2) + j.axis[2].powi(2)).sqrt();
        assert!((len - 1.0).abs() < TOL, "axis should be normalized");
    }

    // ─── Forward kinematics ─────────────────────────────────────────────

    #[test]
    fn test_fk_single_revolute_zero_angle() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let pos = chain.forward(&[0.0]);
        // Link along X, no rotation → end at (1, 0, 0)
        assert!((pos[0] - 1.0).abs() < TOL);
        assert!(pos[1].abs() < TOL);
        assert!(pos[2].abs() < TOL);
    }

    #[test]
    fn test_fk_single_revolute_90_deg() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let pos = chain.forward(&[FRAC_PI_2]);
        // Convention: Rot(90°Z) * Trans(1,0,0) → rotates (1,0,0) to (0,1,0)
        assert!(pos[0].abs() < TOL, "x should be ~0, got {}", pos[0]);
        assert!((pos[1] - 1.0).abs() < TOL, "y should be ~1, got {}", pos[1]);
    }

    #[test]
    fn test_fk_two_link_planar_zero() {
        // Two revolute joints around Z, each with link length 1 along X
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let pos = chain.forward(&[0.0, 0.0]);
        // Both links straight out along X → end at (2, 0, 0)
        assert!((pos[0] - 2.0).abs() < TOL);
        assert!(pos[1].abs() < TOL);
    }

    #[test]
    fn test_fk_two_link_planar_90_deg() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let pos = chain.forward(&[FRAC_PI_2, 0.0]);
        // Convention: T_i = Rot(q_i) * Trans(offset_i)
        // T1 = RotZ(90°) * Trans(1,0,0) → position (0,1,0), frame rotated 90°
        // T2 = RotZ(0°) * Trans(1,0,0) → in rotated frame, (1,0,0) becomes (0,1,0) in world
        // Total position: (0,1,0) + (0,1,0) = (0,2,0)
        assert!(pos[0].abs() < TOL, "x should be ~0, got {}", pos[0]);
        assert!((pos[1] - 2.0).abs() < TOL, "y should be ~2, got {}", pos[1]);
    }

    #[test]
    fn test_fk_two_link_folded() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let pos = chain.forward(&[0.0, PI]);
        // Convention: T_i = Rot(q_i) * Trans(offset_i)
        // T1 = Rot(0) * Trans(1,0,0) → position (1,0,0)
        // T2 = Rot(PI) * Trans(1,0,0) → in parent frame: RotZ(PI)*(1,0,0) = (-1,0,0)
        // Total: (1,0,0) + (-1,0,0) = (0,0,0)
        assert!(pos[0].abs() < TOL, "folded arm should be at origin x, got {}", pos[0]);
        assert!(pos[1].abs() < TOL, "y should be ~0, got {}", pos[1]);
    }

    #[test]
    fn test_fk_prismatic_joint() {
        let chain = Chain::new(vec![
            Joint::prismatic([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ]);
        let pos = chain.forward(&[5.0]);
        assert!((pos[0] - 5.0).abs() < TOL);
        assert!(pos[1].abs() < TOL);
    }

    #[test]
    fn test_fk_mixed_joints() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::prismatic([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ]);
        // Convention: Rot * Trans
        // Joint 1: RotZ(90°) * Trans(1,0,0) → position (0,1,0), frame rotated 90°
        // Joint 2: prismatic along local X (= world Y after rotation) by 3.0
        let pos = chain.forward(&[FRAC_PI_2, 3.0]);
        assert!(pos[0].abs() < TOL, "x={}", pos[0]);
        assert!((pos[1] - 4.0).abs() < TOL, "y={}", pos[1]);
    }

    #[test]
    fn test_fk_3d_chain() {
        // Joint rotating around X, then link along Z
        let chain = Chain::new(vec![
            Joint::revolute([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ]);
        let pos = chain.forward(&[FRAC_PI_2]);
        // Convention: RotX(90°) * Trans(0,0,1)
        // RotX(90°) maps Z to Y: (0,0,1) → (0,1,0)... wait:
        // RotX(90°): y→-z, z→y, so (0,0,1) → (0,-1,0)... no:
        // RotX(θ): [[1,0,0],[0,cos,-sin],[0,sin,cos]]
        // (0,0,1) → (0, -sin(90°), cos(90°)) = (0, -1, 0)
        // Hmm, actually RotX(90°): (0,0,1) → (0, 0*cos90-1*sin90, 0*sin90+1*cos90) = (0,-1,0)
        // Wait: [[1,0,0],[0,c,-s],[0,s,c]] * [0,0,1] = [0, -s, c] = [0, -1, 0] for θ=90°
        assert!(pos[0].abs() < TOL);
        assert!((pos[1] - (-1.0)).abs() < TOL, "y={}", pos[1]);
        assert!(pos[2].abs() < TOL, "z={}", pos[2]);
    }

    #[test]
    fn test_fk_transform_is_proper() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ]);
        let t = chain.forward_transform(&[0.5, -0.3]);
        // Bottom row should be [0, 0, 0, 1]
        assert!(t[[3, 0]].abs() < TOL);
        assert!(t[[3, 1]].abs() < TOL);
        assert!(t[[3, 2]].abs() < TOL);
        assert!((t[[3, 3]] - 1.0).abs() < TOL);

        // Rotation part should be orthogonal
        let r = t.slice(ndarray::s![..3, ..3]).to_owned();
        let rrt = r.dot(&r.t());
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((rrt[[i, j]] - expected).abs() < 1e-8,
                    "R*R^T[{},{}] = {}", i, j, rrt[[i, j]]);
            }
        }
    }

    // ─── CCD Inverse Kinematics ─────────────────────────────────────────

    #[test]
    fn test_ccd_already_at_target() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let target = chain.forward(&[0.3, -0.5]);
        let result = chain.solve_ccd(&target, &[0.3, -0.5], 1, 1e-6).unwrap();
        let ee = chain.forward(&result);
        for i in 0..3 {
            assert!((ee[i] - target[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_ccd_2link_reachable() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        // Target at (0, 2, 0) — reachable with both joints at 90°
        // Note: CCD converges slowly near fully-extended/symmetric poses
        let target = [0.0, 2.0, 0.0];
        let result = chain.solve_ccd(&target, &[0.0, 0.0], 500, 1e-2).unwrap();
        let ee = chain.forward(&result);
        let err = ((ee[0] - target[0]).powi(2) + (ee[1] - target[1]).powi(2)).sqrt();
        assert!(err < 1e-1, "CCD should reach target, error = {}", err);
    }

    #[test]
    fn test_ccd_2link_straight() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        // Target at (2, 0, 0) — fully extended (singularity for CCD)
        let target = [2.0, 0.0, 0.0];
        let result = chain.solve_ccd(&target, &[0.1, 0.1], 500, 1e-2).unwrap();
        let ee = chain.forward(&result);
        let err = ((ee[0] - 2.0).powi(2) + ee[1].powi(2)).sqrt();
        assert!(err < 1e-1, "error = {}", err);
    }

    #[test]
    fn test_ccd_unreachable() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        // Target at (10, 0, 0) — max reach is 2.0
        let result = chain.solve_ccd(&[10.0, 0.0, 0.0], &[0.0, 0.0], 50, 1e-4);
        assert!(result.is_err(), "unreachable target should fail");
    }

    #[test]
    fn test_ccd_with_limits() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
                .with_limits(-FRAC_PI_4, FRAC_PI_4),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
                .with_limits(-FRAC_PI_4, FRAC_PI_4),
        ]);
        let target = [1.5, 0.5, 0.0];
        let result = chain.solve_ccd(&target, &[0.0, 0.0], 100, 1e-3);
        if let Ok(params) = &result {
            for (i, &p) in params.iter().enumerate() {
                assert!(p >= -FRAC_PI_4 - TOL && p <= FRAC_PI_4 + TOL,
                    "joint {} exceeds limits: {}", i, p);
            }
        }
    }

    #[test]
    fn test_ccd_prismatic_chain() {
        let chain = Chain::new(vec![
            Joint::prismatic([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            Joint::prismatic([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        ]);
        let target = [3.0, 4.0, 0.0];
        let result = chain.solve_ccd(&target, &[0.0, 0.0], 10, 1e-6).unwrap();
        let ee = chain.forward(&result);
        assert!((ee[0] - 3.0).abs() < 1e-4);
        assert!((ee[1] - 4.0).abs() < 1e-4);
    }

    // ─── Jacobian Inverse Kinematics ────────────────────────────────────

    #[test]
    fn test_jacobian_computation() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let jac = chain.jacobian(&[0.0, 0.0]);
        // At zero angles: chain is along X axis
        // Joint 0 at (1,0,0), EE at (2,0,0): axis × (2-1, 0, 0) = Z × (1,0,0) = (0,1,0)
        // But joint 0 axis is at the joint frame origin...
        // The Jacobian should be 3×2
        assert_eq!(jac.shape(), &[3, 2]);
    }

    #[test]
    fn test_jacobian_ik_2link() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let target = [1.0, 1.0, 0.0];
        let result = chain.solve_jacobian(&target, &[0.1, 0.1], 200, 1e-4, 0.1).unwrap();
        let ee = chain.forward(&result);
        let err = ((ee[0] - target[0]).powi(2) + (ee[1] - target[1]).powi(2)).sqrt();
        assert!(err < 1e-3, "Jacobian IK error = {}", err);
    }

    #[test]
    fn test_jacobian_ik_straight() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let target = [2.0, 0.0, 0.0];
        let result = chain.solve_jacobian(&target, &[0.1, 0.1], 200, 1e-3, 0.1).unwrap();
        let ee = chain.forward(&result);
        let err = ((ee[0] - 2.0).powi(2) + ee[1].powi(2)).sqrt();
        assert!(err < 1e-2, "error = {}", err);
    }

    #[test]
    fn test_jacobian_ik_unreachable() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let result = chain.solve_jacobian(&[10.0, 0.0, 0.0], &[0.0], 50, 1e-4, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_jacobian_vs_ccd_agreement() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ]);
        let target = [1.5, 1.0, 0.0];
        let init = [0.1, 0.1, 0.1];
        let ccd_result = chain.solve_ccd(&target, &init, 200, 1e-4).unwrap();
        let jac_result = chain.solve_jacobian(&target, &init, 200, 1e-4, 0.1).unwrap();

        let ccd_ee = chain.forward(&ccd_result);
        let jac_ee = chain.forward(&jac_result);

        // Both should reach the target (possibly different joint configs)
        let ccd_err = ((ccd_ee[0] - target[0]).powi(2) + (ccd_ee[1] - target[1]).powi(2)).sqrt();
        let jac_err = ((jac_ee[0] - target[0]).powi(2) + (jac_ee[1] - target[1]).powi(2)).sqrt();

        assert!(ccd_err < 1e-3, "CCD error = {}", ccd_err);
        assert!(jac_err < 1e-3, "Jacobian error = {}", jac_err);
    }

    // ─── Chain properties ───────────────────────────────────────────────

    #[test]
    fn test_chain_dof() {
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::prismatic([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ]);
        assert_eq!(chain.dof(), 3);
    }

    #[test]
    fn test_fk_consistency_with_lie() {
        // Verify FK uses the same rotation as lie::so3_exp
        let angle = 0.7;
        let axis = [0.0, 0.0, 1.0];
        let chain = Chain::new(vec![
            Joint::revolute(axis, [0.0, 0.0, 0.0]),
        ]);
        let t = chain.forward_transform(&[angle]);
        let r_fk = t.slice(ndarray::s![..3, ..3]).to_owned();
        let r_lie = lie::so3_exp(&[0.0, 0.0, angle]);
        for i in 0..3 {
            for j in 0..3 {
                assert!((r_fk[[i, j]] - r_lie[[i, j]]).abs() < 1e-10,
                    "FK rotation should match lie::so3_exp");
            }
        }
    }

    #[test]
    fn test_ccd_3d_target() {
        // 3-joint chain with joints rotating around different axes
        let chain = Chain::new(vec![
            Joint::revolute([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            Joint::revolute([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            Joint::revolute([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ]);
        // Pick a reachable 3D target
        let test_angles = [0.5, -0.3, 0.8];
        let target = chain.forward(&test_angles);
        let result = chain.solve_ccd(&target, &[0.0, 0.0, 0.0], 200, 1e-3).unwrap();
        let ee = chain.forward(&result);
        let err = ((ee[0] - target[0]).powi(2) + (ee[1] - target[1]).powi(2) + (ee[2] - target[2]).powi(2)).sqrt();
        assert!(err < 1e-2, "3D CCD error = {}", err);
    }

    // ─── Helper tests ───────────────────────────────────────────────────

    #[test]
    fn test_solve_3x3() {
        // Simple system: I * x = b
        let a = Array2::eye(3);
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let x = solve_3x3(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < TOL);
        assert!((x[1] - 2.0).abs() < TOL);
        assert!((x[2] - 3.0).abs() < TOL);
    }

    #[test]
    fn test_solve_3x3_singular() {
        let a = Array2::zeros((3, 3));
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        assert!(solve_3x3(&a, &b).is_err());
    }
}
