//! Iterated Function Systems (IFS) — chaos game algorithm.
//!
//! Provides affine map definitions and the chaos game iteration for generating
//! fractal attractors. Includes predefined IFS for Sierpinski triangle,
//! Barnsley fern, and Koch snowflake.
//!
//! # Examples
//!
//! ```
//! use ix_fractal::ifs;
//! use rand::SeedableRng;
//!
//! let maps = ifs::sierpinski_maps();
//! let mut rng = rand::rngs::StdRng::seed_from_u64(42);
//! let points = ifs::ifs_iterate(&maps, 1000, &mut rng);
//! assert_eq!(points.len(), 1000);
//! ```

use rand::Rng;

/// A 2D affine transformation with selection probability weight.
///
/// Transforms point `[x, y]` as:
/// ```text
/// x' = a[0][0]*x + a[0][1]*y + b[0]
/// y' = a[1][0]*x + a[1][1]*y + b[1]
/// ```
#[derive(Debug, Clone)]
pub struct AffineMap {
    /// 2x2 linear transformation matrix.
    pub a: [[f64; 2]; 2],
    /// Translation vector.
    pub b: [f64; 2],
    /// Selection probability weight (will be normalized).
    pub weight: f64,
}

impl AffineMap {
    /// Apply this affine map to a 2D point.
    fn apply(&self, point: &[f64; 2]) -> [f64; 2] {
        [
            self.a[0][0] * point[0] + self.a[0][1] * point[1] + self.b[0],
            self.a[1][0] * point[0] + self.a[1][1] * point[1] + self.b[1],
        ]
    }
}

/// Run the chaos game: iterate an IFS for the given number of iterations.
///
/// Starting from origin `[0, 0]`, at each step a map is chosen with probability
/// proportional to its weight, and the current point is transformed.
/// Returns all generated points (including the initial transient).
pub fn ifs_iterate(maps: &[AffineMap], iterations: usize, rng: &mut impl Rng) -> Vec<[f64; 2]> {
    assert!(!maps.is_empty(), "IFS requires at least one map");

    // Build cumulative weight distribution
    let total_weight: f64 = maps.iter().map(|m| m.weight).sum();
    let mut cumulative = Vec::with_capacity(maps.len());
    let mut acc = 0.0;
    for m in maps {
        acc += m.weight / total_weight;
        cumulative.push(acc);
    }

    let mut point = [0.0, 0.0];
    let mut points = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        // Select a map based on weighted probability
        let r: f64 = rng.random();
        let idx = cumulative
            .iter()
            .position(|&c| r <= c)
            .unwrap_or(maps.len() - 1);

        point = maps[idx].apply(&point);
        points.push(point);
    }

    points
}

/// Standard Sierpinski triangle IFS.
///
/// Three maps, each contracting by 0.5 towards a vertex of the unit triangle:
/// (0,0), (1,0), (0.5, sqrt(3)/2).
pub fn sierpinski_maps() -> Vec<AffineMap> {
    vec![
        AffineMap {
            a: [[0.5, 0.0], [0.0, 0.5]],
            b: [0.0, 0.0],
            weight: 1.0,
        },
        AffineMap {
            a: [[0.5, 0.0], [0.0, 0.5]],
            b: [0.5, 0.0],
            weight: 1.0,
        },
        AffineMap {
            a: [[0.5, 0.0], [0.0, 0.5]],
            b: [0.25, 0.433012701892],
            weight: 1.0,
        },
    ]
}

/// Barnsley fern IFS (classic four-map system).
pub fn barnsley_fern_maps() -> Vec<AffineMap> {
    vec![
        // Stem
        AffineMap {
            a: [[0.0, 0.0], [0.0, 0.16]],
            b: [0.0, 0.0],
            weight: 0.01,
        },
        // Successively larger leaflets
        AffineMap {
            a: [[0.85, 0.04], [-0.04, 0.85]],
            b: [0.0, 1.6],
            weight: 0.85,
        },
        // Left leaflet
        AffineMap {
            a: [[0.20, -0.26], [0.23, 0.22]],
            b: [0.0, 1.6],
            weight: 0.07,
        },
        // Right leaflet
        AffineMap {
            a: [[-0.15, 0.28], [0.26, 0.24]],
            b: [0.0, 0.44],
            weight: 0.07,
        },
    ]
}

/// Koch snowflake approximation IFS.
///
/// Four maps that approximate the Koch curve by contracting each segment
/// by 1/3 with appropriate rotations.
pub fn koch_snowflake_maps() -> Vec<AffineMap> {
    let cos60 = 0.5_f64;
    let sin60 = (3.0_f64).sqrt() / 2.0;

    vec![
        // First segment: scale by 1/3
        AffineMap {
            a: [[1.0 / 3.0, 0.0], [0.0, 1.0 / 3.0]],
            b: [0.0, 0.0],
            weight: 1.0,
        },
        // Second segment: scale by 1/3 and rotate +60 degrees
        AffineMap {
            a: [[cos60 / 3.0, -sin60 / 3.0], [sin60 / 3.0, cos60 / 3.0]],
            b: [1.0 / 3.0, 0.0],
            weight: 1.0,
        },
        // Third segment: scale by 1/3 and rotate -60 degrees
        AffineMap {
            a: [[cos60 / 3.0, sin60 / 3.0], [-sin60 / 3.0, cos60 / 3.0]],
            b: [0.5, sin60 / 3.0],
            weight: 1.0,
        },
        // Fourth segment: scale by 1/3, translate to end
        AffineMap {
            a: [[1.0 / 3.0, 0.0], [0.0, 1.0 / 3.0]],
            b: [2.0 / 3.0, 0.0],
            weight: 1.0,
        },
    ]
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
    fn test_sierpinski_bounded() {
        let maps = sierpinski_maps();
        let mut rng = make_rng();
        let points = ifs_iterate(&maps, 10_000, &mut rng);

        // Skip first 100 transient points
        for &[x, y] in &points[100..] {
            assert!((-0.01..=1.01).contains(&x), "x={} out of bounds", x);
            assert!((-0.01..=0.87).contains(&y), "y={} out of bounds", y);
        }
    }

    #[test]
    fn test_barnsley_fern_bounded() {
        let maps = barnsley_fern_maps();
        let mut rng = make_rng();
        let points = ifs_iterate(&maps, 10_000, &mut rng);

        // Fern should be roughly in [-3, 3] x [0, 10]
        for &[x, y] in &points[100..] {
            assert!((-3.0..=3.0).contains(&x), "fern x={} out of bounds", x);
            assert!((-0.5..=11.0).contains(&y), "fern y={} out of bounds", y);
        }
    }

    #[test]
    fn test_correct_number_of_points() {
        let maps = sierpinski_maps();
        let mut rng = make_rng();
        let points = ifs_iterate(&maps, 500, &mut rng);
        assert_eq!(points.len(), 500);
    }

    #[test]
    fn test_reproducibility_with_seeded_rng() {
        let maps = sierpinski_maps();

        let mut rng1 = make_rng();
        let p1 = ifs_iterate(&maps, 100, &mut rng1);

        let mut rng2 = make_rng();
        let p2 = ifs_iterate(&maps, 100, &mut rng2);

        assert_eq!(p1.len(), p2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert_eq!(a[0], b[0]);
            assert_eq!(a[1], b[1]);
        }
    }

    #[test]
    fn test_koch_snowflake_maps() {
        let maps = koch_snowflake_maps();
        assert_eq!(maps.len(), 4);

        let mut rng = make_rng();
        let points = ifs_iterate(&maps, 5000, &mut rng);
        assert_eq!(points.len(), 5000);

        // Koch curve should stay roughly bounded
        for &[x, y] in &points[100..] {
            assert!((-0.5..=1.5).contains(&x), "koch x={} out of bounds", x);
            assert!((-0.5..=1.0).contains(&y), "koch y={} out of bounds", y);
        }
    }

    #[test]
    fn test_zero_iterations() {
        let maps = sierpinski_maps();
        let mut rng = make_rng();
        let points = ifs_iterate(&maps, 0, &mut rng);
        assert_eq!(points.len(), 0);
    }
}
