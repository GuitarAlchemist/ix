//! De Rham fractal curves — IFS-based interpolation.
//!
//! Roughness decays 0.5× per recursion level. Seeded RNG for reproducibility.
//! Uses iterative implementation (explicit worklist) to avoid stack overflow.
//!
//! # Examples
//!
//! ```
//! use machin_chaos::de_rham;
//! use ndarray::Array1;
//! use rand::SeedableRng;
//!
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//!
//! // Generate a fractal path between two 2D points
//! let p0 = Array1::from_vec(vec![0.0, 0.0]);
//! let p1 = Array1::from_vec(vec![1.0, 1.0]);
//! let path = de_rham::de_rham_interpolate(&p0, &p1, 5, 0.3, &mut rng);
//! assert_eq!(path.len(), 33); // 2^5 + 1
//! assert_eq!(&path[0], &p0);  // endpoints preserved
//!
//! // Generate a 1D fractal signal
//! let mut rng2 = rand::rngs::StdRng::seed_from_u64(99);
//! let signal = de_rham::de_rham_curve_1d(6, 0.2, &mut rng2);
//! assert_eq!(signal.len(), 65); // 2^6 + 1
//! ```

use ndarray::Array1;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Maximum depth (2^20 ≈ 1M points).
const MAX_DEPTH: usize = 20;

/// Generate a de Rham fractal curve by IFS interpolation between two points.
///
/// Produces `2^depth + 1` points (including endpoints) by iteratively subdividing
/// the segment and displacing midpoints with Gaussian noise scaled by roughness.
///
/// Roughness decays 0.5× per level. Depth is silently capped at 20.
/// Returns `vec![p0, p1]` for depth=0.
pub fn de_rham_interpolate(
    p0: &Array1<f64>,
    p1: &Array1<f64>,
    depth: usize,
    roughness: f64,
    rng: &mut impl Rng,
) -> Vec<Array1<f64>> {
    let depth = depth.min(MAX_DEPTH);

    if depth == 0 {
        return vec![p0.clone(), p1.clone()];
    }

    let n_points = (1usize << depth) + 1;
    let dim = p0.len();

    // Initialize with linearly interpolated points
    let mut points: Vec<Array1<f64>> = (0..n_points)
        .map(|i| {
            let t = i as f64 / (n_points - 1) as f64;
            p0 * (1.0 - t) + p1 * t
        })
        .collect();

    // Iterative midpoint displacement: from coarse to fine
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut current_roughness = roughness;

    for level in 0..depth {
        let step = 1usize << (depth - level); // distance between existing points at this level
        let half = step / 2;

        let mut i = half;
        while i < n_points - 1 {
            // Midpoint of the two bracketing points
            let left = i - half;
            let right = i + half;
            let mid = (&points[left] + &points[right]) * 0.5;

            // Displace by Gaussian noise scaled by roughness and segment length
            let segment_len = {
                let diff = &points[right] - &points[left];
                diff.dot(&diff).sqrt()
            };
            let scale = current_roughness * segment_len;

            let displacement: Array1<f64> =
                Array1::from_vec((0..dim).map(|_| normal.sample(rng) * scale).collect());

            points[i] = mid + displacement;
            i += step;
        }

        current_roughness *= 0.5;
    }

    points
}

/// Generate a 1D de Rham fractal signal from 0.0 to 1.0.
///
/// Produces `2^depth + 1` samples. Depth is silently capped at 20.
pub fn de_rham_curve_1d(depth: usize, roughness: f64, rng: &mut impl Rng) -> Array1<f64> {
    let p0 = Array1::from_vec(vec![0.0]);
    let p1 = Array1::from_vec(vec![1.0]);
    let points = de_rham_interpolate(&p0, &p1, depth, roughness, rng);
    Array1::from_vec(points.into_iter().map(|p| p[0]).collect())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_depth_zero() {
        let p0 = Array1::from_vec(vec![0.0, 0.0]);
        let p1 = Array1::from_vec(vec![1.0, 1.0]);
        let mut rng = make_rng();
        let result = de_rham_interpolate(&p0, &p1, 0, 0.5, &mut rng);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], p0);
        assert_eq!(result[1], p1);
    }

    #[test]
    fn test_point_count() {
        let p0 = Array1::from_vec(vec![0.0]);
        let p1 = Array1::from_vec(vec![1.0]);
        let mut rng = make_rng();

        // depth=1 → 3 points, depth=3 → 9 points, depth=5 → 33 points
        assert_eq!(de_rham_interpolate(&p0, &p1, 1, 0.5, &mut rng).len(), 3);
        assert_eq!(de_rham_interpolate(&p0, &p1, 3, 0.5, &mut rng).len(), 9);
        assert_eq!(de_rham_interpolate(&p0, &p1, 5, 0.5, &mut rng).len(), 33);
    }

    #[test]
    fn test_endpoints_preserved() {
        let p0 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let p1 = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let mut rng = make_rng();
        let result = de_rham_interpolate(&p0, &p1, 5, 0.3, &mut rng);
        assert_eq!(&result[0], &p0);
        assert_eq!(result.last().unwrap(), &p1);
    }

    #[test]
    fn test_zero_roughness_is_straight_line() {
        let p0 = Array1::from_vec(vec![0.0]);
        let p1 = Array1::from_vec(vec![10.0]);
        let mut rng = make_rng();
        let result = de_rham_interpolate(&p0, &p1, 4, 0.0, &mut rng);
        // With roughness=0, all points should lie on the line [0, 10]
        let n = result.len();
        for (i, p) in result.iter().enumerate() {
            let expected = 10.0 * i as f64 / (n - 1) as f64;
            assert!(
                (p[0] - expected).abs() < 1e-10,
                "point {} = {}, expected {}",
                i,
                p[0],
                expected
            );
        }
    }

    #[test]
    fn test_reproducibility() {
        let p0 = Array1::from_vec(vec![0.0, 0.0]);
        let p1 = Array1::from_vec(vec![1.0, 1.0]);

        let mut rng1 = make_rng();
        let r1 = de_rham_interpolate(&p0, &p1, 5, 0.3, &mut rng1);

        let mut rng2 = make_rng();
        let r2 = de_rham_interpolate(&p0, &p1, 5, 0.3, &mut rng2);

        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_depth_capped() {
        let p0 = Array1::from_vec(vec![0.0]);
        let p1 = Array1::from_vec(vec![1.0]);
        let mut rng = make_rng();
        // depth=25 should be silently capped to 20
        let result = de_rham_interpolate(&p0, &p1, 25, 0.1, &mut rng);
        assert_eq!(result.len(), (1 << 20) + 1);
    }

    #[test]
    fn test_de_rham_curve_1d() {
        let mut rng = make_rng();
        let curve = de_rham_curve_1d(5, 0.3, &mut rng);
        assert_eq!(curve.len(), 33);
    }

    #[test]
    fn test_de_rham_curve_1d_endpoints() {
        let mut rng = make_rng();
        let curve = de_rham_curve_1d(5, 0.0, &mut rng);
        assert!((curve[0] - 0.0).abs() < 1e-10);
        assert!((curve[curve.len() - 1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_higher_roughness_more_variation() {
        let p0 = Array1::from_vec(vec![0.0]);
        let p1 = Array1::from_vec(vec![1.0]);

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(123);
        let low = de_rham_interpolate(&p0, &p1, 8, 0.01, &mut rng1);

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(123);
        let high = de_rham_interpolate(&p0, &p1, 8, 1.0, &mut rng2);

        // Compute variance of deviation from straight line
        let n = low.len();
        let var = |points: &[Array1<f64>]| -> f64 {
            points
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let expected = i as f64 / (n - 1) as f64;
                    let d = p[0] - expected;
                    d * d
                })
                .sum::<f64>()
                / n as f64
        };

        assert!(var(&high) > var(&low));
    }
}
