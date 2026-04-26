//! Plücker coordinates for lines in 3D space.
//!
//! Convention: direction-first `(l, m)` where `l` is unit direction,
//! `m = p × l` is the moment. Follows Clifford/Study screw theory convention.
//!
//! Uses `[f64; 3]` for stack allocation and `Copy` semantics.
//!
//! # Examples
//!
//! ```
//! use ix_math::plucker::PluckerLine;
//!
//! // X-axis and Y-axis through origin — they intersect
//! let l1 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
//! let l2 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).unwrap();
//! assert!(l1.intersects(&l2, 1e-10));
//!
//! // Skew lines: X-axis and a Y-direction line offset by z=3
//! let l3 = PluckerLine::from_two_points(&[0.0, 0.0, 3.0], &[0.0, 1.0, 3.0]).unwrap();
//! assert!(!l1.intersects(&l3, 1e-10));
//! assert!((l1.distance_between(&l3) - 3.0).abs() < 1e-10);
//! ```

use crate::error::MathError;
use crate::quaternion::cross3;

const NORM_EPSILON: f64 = 1e-12;

// ─── PluckerLine ────────────────────────────────────────────────────────────

/// Plücker coordinates for a line in 3D space.
///
/// A line is represented by a unit direction vector `l` and a moment vector
/// `m = p × l`, where `p` is any point on the line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PluckerLine {
    /// Unit direction vector.
    pub direction: [f64; 3],
    /// Moment vector: `point × direction`.
    pub moment: [f64; 3],
}

impl PluckerLine {
    /// Create a Plücker line from two distinct points.
    ///
    /// Direction is `(p2 - p1) / ||p2 - p1||`. Returns `Err` if points are coincident.
    pub fn from_two_points(p1: &[f64; 3], p2: &[f64; 3]) -> Result<Self, MathError> {
        let d = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if len < NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "coincident points cannot define a line".into(),
            ));
        }
        let inv = 1.0 / len;
        let direction = [d[0] * inv, d[1] * inv, d[2] * inv];
        let moment = cross3(p1, &direction);
        Ok(Self { direction, moment })
    }

    /// Create a Plücker line from a point and a direction.
    ///
    /// Direction is internally normalized. Returns `Err` if direction has near-zero length.
    pub fn from_point_direction(point: &[f64; 3], direction: &[f64; 3]) -> Result<Self, MathError> {
        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if len < NORM_EPSILON {
            return Err(MathError::InvalidParameter(
                "direction has near-zero length".into(),
            ));
        }
        let inv = 1.0 / len;
        let dir = [direction[0] * inv, direction[1] * inv, direction[2] * inv];
        let moment = cross3(point, &dir);
        Ok(Self {
            direction: dir,
            moment,
        })
    }

    /// Reciprocal product of two Plücker lines.
    ///
    /// `l1.direction · l2.moment + l2.direction · l1.moment`
    ///
    /// This equals zero if and only if the lines are coplanar (intersect or are parallel).
    pub fn reciprocal_product(&self, other: &PluckerLine) -> f64 {
        dot3(&self.direction, &other.moment) + dot3(&other.direction, &self.moment)
    }

    /// Returns `true` if the lines are coplanar (intersect or are parallel)
    /// within the given tolerance.
    pub fn intersects(&self, other: &PluckerLine, tolerance: f64) -> bool {
        self.reciprocal_product(other).abs() < tolerance
    }

    /// Perpendicular distance between two lines.
    ///
    /// `|reciprocal_product| / ||l1.direction × l2.direction||`
    /// Returns 0 for parallel lines (distance is along the moment in that case,
    /// but the minimum perpendicular distance between parallel lines requires a
    /// different computation — this returns the skew-line distance).
    pub fn distance_between(&self, other: &PluckerLine) -> f64 {
        let c = cross3(&self.direction, &other.direction);
        let cn = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        if cn < NORM_EPSILON {
            // Parallel lines: compute distance via moment difference
            // d = ||m2/||l2|| - m1/||l1|| × l1||
            // Since directions are unit, d = ||m2 - m1|| projected perpendicular to l
            let dm = [
                other.moment[0] - self.moment[0],
                other.moment[1] - self.moment[1],
                other.moment[2] - self.moment[2],
            ];
            return (dm[0] * dm[0] + dm[1] * dm[1] + dm[2] * dm[2]).sqrt();
        }
        self.reciprocal_product(other).abs() / cn
    }

    /// Closest point on this line to the origin.
    ///
    /// `p = direction × moment` (for unit-direction Plücker lines).
    pub fn closest_point_to_origin(&self) -> [f64; 3] {
        cross3(&self.direction, &self.moment)
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    fn vec3_approx_eq(a: &[f64; 3], b: &[f64; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_from_two_points_x_axis() {
        let l = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        assert!(vec3_approx_eq(&l.direction, &[1.0, 0.0, 0.0]));
        assert!(vec3_approx_eq(&l.moment, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_from_two_points_normalizes() {
        let l = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[5.0, 0.0, 0.0]).unwrap();
        assert!(vec3_approx_eq(&l.direction, &[1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_from_two_points_coincident_fails() {
        let p = [1.0, 2.0, 3.0];
        assert!(PluckerLine::from_two_points(&p, &p).is_err());
    }

    #[test]
    fn test_from_point_direction() {
        let l = PluckerLine::from_point_direction(&[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]).unwrap();
        assert!(vec3_approx_eq(&l.direction, &[0.0, 0.0, 1.0]));
        // moment = (0,1,0) × (0,0,1) = (1,0,0)
        assert!(vec3_approx_eq(&l.moment, &[1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_from_point_direction_zero_fails() {
        assert!(PluckerLine::from_point_direction(&[0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_intersecting_lines() {
        // X-axis and Y-axis through origin — they intersect
        let l1 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        let l2 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).unwrap();
        assert!(l1.intersects(&l2, 1e-10));
    }

    #[test]
    fn test_skew_lines() {
        // X-axis through origin, and a line parallel to Y at z=1
        let l1 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        let l2 = PluckerLine::from_two_points(&[0.0, 0.0, 1.0], &[0.0, 1.0, 1.0]).unwrap();
        assert!(!l1.intersects(&l2, 1e-10));
    }

    #[test]
    fn test_parallel_lines_coplanar() {
        // Two parallel lines in the XY plane
        let l1 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        let l2 = PluckerLine::from_two_points(&[0.0, 1.0, 0.0], &[1.0, 1.0, 0.0]).unwrap();
        // Parallel lines are coplanar → reciprocal product = 0
        assert!(l1.intersects(&l2, 1e-10));
    }

    #[test]
    fn test_distance_skew_lines() {
        // X-axis and Y-axis offset by z=3
        let l1 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        let l2 = PluckerLine::from_two_points(&[0.0, 0.0, 3.0], &[0.0, 1.0, 3.0]).unwrap();
        assert!(approx_eq(l1.distance_between(&l2), 3.0));
    }

    #[test]
    fn test_distance_intersecting_lines() {
        let l1 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        let l2 = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0]).unwrap();
        assert!(approx_eq(l1.distance_between(&l2), 0.0));
    }

    #[test]
    fn test_closest_point_to_origin() {
        // Line through (0,1,0) in X direction: closest to origin is (0,1,0)
        // Wait: direction × moment. moment = (0,1,0)×(1,0,0) = (0,0,-1)
        // direction × moment = (1,0,0)×(0,0,-1) = (0,1,0) ✓
        let l = PluckerLine::from_point_direction(&[0.0, 1.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        let p = l.closest_point_to_origin();
        assert!(vec3_approx_eq(&p, &[0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_closest_point_on_axis_line() {
        // X-axis through origin: closest point is origin
        let l = PluckerLine::from_two_points(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        let p = l.closest_point_to_origin();
        assert!(vec3_approx_eq(&p, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn test_reciprocal_product_symmetric() {
        let l1 = PluckerLine::from_two_points(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        let l2 = PluckerLine::from_two_points(&[0.0, 1.0, 2.0], &[3.0, 0.0, 1.0]).unwrap();
        assert!(approx_eq(
            l1.reciprocal_product(&l2),
            l2.reciprocal_product(&l1)
        ));
    }

    #[test]
    fn test_two_constructors_equivalent() {
        // from_two_points and from_point_direction should agree
        let p = [1.0, 2.0, 3.0];
        let d = [4.0, 5.0, 6.0];
        let p2 = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];

        let l1 = PluckerLine::from_two_points(&p, &p2).unwrap();
        let l2 = PluckerLine::from_point_direction(&p, &d).unwrap();

        assert!(vec3_approx_eq(&l1.direction, &l2.direction));
        assert!(vec3_approx_eq(&l1.moment, &l2.moment));
    }
}
