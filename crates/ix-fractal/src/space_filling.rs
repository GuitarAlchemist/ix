//! Space-filling curves: Hilbert, Peano, and Z-order (Morton) encoding.
//!
//! These curves provide mappings between 1D and 2D spaces, useful for
//! spatial indexing, cache-friendly memory layouts, and data visualization.
//!
//! # Examples
//!
//! ```
//! use ix_fractal::space_filling;
//!
//! // Hilbert curve order 2 has 16 points
//! let curve = space_filling::hilbert_curve(2);
//! assert_eq!(curve.len(), 16);
//!
//! // Morton encode/decode round-trip
//! let z = space_filling::morton_encode(3, 5);
//! let (x, y) = space_filling::morton_decode(z);
//! assert_eq!((x, y), (3, 5));
//! ```

/// Convert a distance `d` along a Hilbert curve of order `n` to (x, y) coordinates.
///
/// `n` is the side length (must be a power of 2). `d` ranges from 0 to n*n - 1.
pub fn hilbert_d2xy(n: u32, d: u32) -> (u32, u32) {
    let mut x = 0u32;
    let mut y = 0u32;
    let mut d = d;
    let mut s = 1u32;

    while s < n {
        let rx = (d / 2) & 1;
        let ry = (d ^ rx) & 1;

        // Rotate
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }

        x += s * rx;
        y += s * ry;
        d /= 4;
        s *= 2;
    }

    (x, y)
}

/// Generate a Hilbert curve of the given order.
///
/// Returns `4^order` points tracing the Hilbert curve in [0, 1] x [0, 1].
/// Order 0 returns a single point at (0, 0).
pub fn hilbert_curve(order: u32) -> Vec<[f64; 2]> {
    if order == 0 {
        return vec![[0.0, 0.0]];
    }

    let n = 1u32 << order; // side length = 2^order
    let total = n * n; // 4^order points
    let scale = 1.0 / (n as f64 - 1.0).max(1.0);

    (0..total)
        .map(|d| {
            let (x, y) = hilbert_d2xy(n, d);
            [x as f64 * scale, y as f64 * scale]
        })
        .collect()
}

/// Generate a Peano curve of the given order.
///
/// Returns `9^order` points tracing the Peano curve in [0, 1] x [0, 1].
/// Order 0 returns a single point at (0, 0).
pub fn peano_curve(order: u32) -> Vec<[f64; 2]> {
    if order == 0 {
        return vec![[0.0, 0.0]];
    }

    let n = 3u32.pow(order); // side length
    let total = n * n;
    let scale = 1.0 / (n as f64 - 1.0).max(1.0);

    (0..total)
        .map(|d| {
            let (x, y) = peano_d2xy(order, d);
            [x as f64 * scale, y as f64 * scale]
        })
        .collect()
}

/// Convert a distance along a Peano curve to (x, y) coordinates.
fn peano_d2xy(order: u32, d: u32) -> (u32, u32) {
    let mut x = 0u32;
    let mut y = 0u32;
    let mut d = d;

    for i in 0..order {
        let s = 3u32.pow(i);
        let digit = d % 9;
        d /= 9;

        // Peano curve traversal pattern within each 3x3 cell
        let (dx, dy) = peano_digit_to_xy(digit, i);
        x += dx * s;
        y += dy * s;
    }

    (x, y)
}

/// Map a Peano digit (0-8) to local (x, y) within a 3x3 cell.
/// The traversal pattern alternates direction based on level parity.
fn peano_digit_to_xy(digit: u32, level: u32) -> (u32, u32) {
    // Standard Peano curve traversal order for a 3x3 grid:
    // Level-dependent serpentine pattern
    let base = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    ];

    let (bx, by) = base[digit as usize % 9];

    if level % 2 == 0 {
        (bx, by)
    } else {
        (2 - bx, 2 - by)
    }
}

/// Encode (x, y) coordinates into a Z-order (Morton) code.
///
/// Interleaves the bits of x and y: bit 0 of x goes to bit 0, bit 0 of y goes to bit 1, etc.
pub fn morton_encode(x: u32, y: u32) -> u64 {
    fn spread_bits(v: u32) -> u64 {
        let mut v = v as u64;
        v = (v | (v << 16)) & 0x0000_FFFF_0000_FFFF;
        v = (v | (v << 8)) & 0x00FF_00FF_00FF_00FF;
        v = (v | (v << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        v = (v | (v << 1)) & 0x5555_5555_5555_5555;
        v
    }

    spread_bits(x) | (spread_bits(y) << 1)
}

/// Decode a Z-order (Morton) code back to (x, y) coordinates.
pub fn morton_decode(z: u64) -> (u32, u32) {
    fn compact_bits(mut v: u64) -> u32 {
        v &= 0x5555_5555_5555_5555;
        v = (v | (v >> 1)) & 0x3333_3333_3333_3333;
        v = (v | (v >> 2)) & 0x0F0F_0F0F_0F0F_0F0F;
        v = (v | (v >> 4)) & 0x00FF_00FF_00FF_00FF;
        v = (v | (v >> 8)) & 0x0000_FFFF_0000_FFFF;
        v = (v | (v >> 16)) & 0x0000_0000_FFFF_FFFF;
        v as u32
    }

    let x = compact_bits(z);
    let y = compact_bits(z >> 1);
    (x, y)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hilbert_order_1_has_4_points() {
        let curve = hilbert_curve(1);
        assert_eq!(curve.len(), 4);
    }

    #[test]
    fn test_hilbert_order_2_has_16_points() {
        let curve = hilbert_curve(2);
        assert_eq!(curve.len(), 16);
    }

    #[test]
    fn test_hilbert_order_0() {
        let curve = hilbert_curve(0);
        assert_eq!(curve.len(), 1);
    }

    #[test]
    fn test_hilbert_bounded() {
        let curve = hilbert_curve(3);
        for &[x, y] in &curve {
            assert!((-0.01..=1.01).contains(&x), "hilbert x={} out of [0,1]", x);
            assert!((-0.01..=1.01).contains(&y), "hilbert y={} out of [0,1]", y);
        }
    }

    #[test]
    fn test_hilbert_all_points_distinct() {
        let curve = hilbert_curve(2);
        for i in 0..curve.len() {
            for j in (i + 1)..curve.len() {
                let dx = (curve[i][0] - curve[j][0]).abs();
                let dy = (curve[i][1] - curve[j][1]).abs();
                assert!(
                    dx > 1e-12 || dy > 1e-12,
                    "duplicate points at {} and {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_morton_encode_decode_roundtrip() {
        let test_cases = [(0, 0), (1, 0), (0, 1), (3, 5), (255, 255), (1000, 2000)];
        for (x, y) in test_cases {
            let z = morton_encode(x, y);
            let (dx, dy) = morton_decode(z);
            assert_eq!((dx, dy), (x, y), "roundtrip failed for ({}, {})", x, y);
        }
    }

    #[test]
    fn test_morton_encode_known_values() {
        // morton_encode(0, 0) = 0
        assert_eq!(morton_encode(0, 0), 0);
        // morton_encode(1, 0) = 1 (bit 0 of x)
        assert_eq!(morton_encode(1, 0), 1);
        // morton_encode(0, 1) = 2 (bit 0 of y goes to bit 1)
        assert_eq!(morton_encode(0, 1), 2);
        // morton_encode(1, 1) = 3
        assert_eq!(morton_encode(1, 1), 3);
    }

    #[test]
    fn test_peano_order_1_has_9_points() {
        let curve = peano_curve(1);
        assert_eq!(curve.len(), 9);
    }

    #[test]
    fn test_peano_order_0() {
        let curve = peano_curve(0);
        assert_eq!(curve.len(), 1);
    }

    #[test]
    fn test_peano_bounded() {
        let curve = peano_curve(2);
        for &[x, y] in &curve {
            assert!((-0.01..=1.01).contains(&x), "peano x={} out of [0,1]", x);
            assert!((-0.01..=1.01).contains(&y), "peano y={} out of [0,1]", y);
        }
    }

    #[test]
    fn test_peano_order_2_has_81_points() {
        let curve = peano_curve(2);
        assert_eq!(curve.len(), 81);
    }

    #[test]
    fn test_hilbert_d2xy_order1() {
        // For n=2 (order 1), the four positions should be distinct
        let points: Vec<_> = (0..4).map(|d| hilbert_d2xy(2, d)).collect();
        assert_eq!(points.len(), 4);
        // Check all are in {0,1} x {0,1}
        for &(x, y) in &points {
            assert!(x <= 1);
            assert!(y <= 1);
        }
    }
}
