//! Plane geometry for flat folded states: polygon area and convexity, half-plane clipping, the
//! overlay of the folded faces into convex cells, and how far a segment runs inside a face.

use std::collections::BTreeSet;

use crate::fold::Point;

/// Distances, in units of the crease pattern.
pub const LEN_TOL: f64 = 1e-9;
/// A smaller common area is two faces touching, not overlapping.
pub const AREA_TOL: f64 = 1e-10;
/// How far inside a face a crease must run to cross it.
pub const INSIDE_TOL: f64 = 1e-7;

/// Signed area: positive for a counterclockwise polygon.
pub fn area(p: &[Point]) -> f64 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    0.5 * (0..n)
        .map(|i| {
            let (a, b) = (p[i], p[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
}

/// Whether every turn of the polygon goes the same way.
pub fn is_convex(p: &[Point]) -> bool {
    let n = p.len();
    let turns: Vec<f64> = (0..n)
        .map(|i| {
            let (a, b, c) = (p[i], p[(i + 1) % n], p[(i + 2) % n]);
            (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        })
        .collect();
    turns.iter().all(|&t| t > -1e-12) || turns.iter().all(|&t| t < 1e-12)
}

pub fn dist(a: Point, b: Point) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn cross(a: Point, b: Point, p: Point) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

/// The part of convex polygon `p` on the left of the directed line `a -> b`.
pub fn clip(p: &[Point], a: Point, b: Point) -> Vec<Point> {
    let n = p.len();
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let (s, e) = (p[i], p[(i + 1) % n]);
        let (cs, ce) = (cross(a, b, s), cross(a, b, e));
        if cs >= 0.0 {
            out.push(s);
        }
        if (cs >= 0.0) != (ce >= 0.0) {
            let t = cs / (cs - ce);
            out.push([s[0] + t * (e[0] - s[0]), s[1] + t * (e[1] - s[1])]);
        }
    }
    out
}

/// `[x0, y0, x1, y1]`.
fn bbox(points: &[Point]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in points {
        b = [
            b[0].min(p[0]),
            b[1].min(p[1]),
            b[2].max(p[0]),
            b[3].max(p[1]),
        ];
    }
    b
}

/// A convex piece of the plane covered by one fixed set of faces.
#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub polygon: Vec<Point>,
    pub faces: BTreeSet<usize>,
}

/// Cuts the plane into convex cells, each covered by a fixed set of faces, and returns the
/// covered cells. The faces must be convex and counterclockwise. The number of cells depends on
/// the order the faces are cut in; which faces overlap does not.
pub fn overlay(polys: &[Vec<Point>]) -> Vec<Cell> {
    let all: Vec<Point> = polys.iter().flatten().copied().collect();
    if all.is_empty() {
        return Vec::new();
    }
    let [x0, y0, x1, y1] = bbox(&all);
    let mut cells = vec![Cell {
        polygon: vec![
            [x0 - 1.0, y0 - 1.0],
            [x1 + 1.0, y0 - 1.0],
            [x1 + 1.0, y1 + 1.0],
            [x0 - 1.0, y1 + 1.0],
        ],
        faces: BTreeSet::new(),
    }];
    for (i, q) in polys.iter().enumerate() {
        let qb = bbox(q);
        let mut next = Vec::with_capacity(cells.len() + q.len());
        for cell in cells {
            let cb = bbox(&cell.polygon);
            if cb[2] <= qb[0] || qb[2] <= cb[0] || cb[3] <= qb[1] || qb[3] <= cb[1] {
                next.push(cell);
                continue;
            }
            let mut rem = Some(cell.polygon);
            for j in 0..q.len() {
                let (a, b) = (q[j], q[(j + 1) % q.len()]);
                let r = rem.take().expect("a remainder is left while sides remain");
                let outside = clip(&r, b, a);
                if area(&outside) > AREA_TOL {
                    next.push(Cell {
                        polygon: outside,
                        faces: cell.faces.clone(),
                    });
                }
                let inside = clip(&r, a, b);
                if area(&inside) <= AREA_TOL {
                    break;
                }
                rem = Some(inside);
            }
            if let Some(r) = rem {
                let mut faces = cell.faces;
                faces.insert(i);
                next.push(Cell { polygon: r, faces });
            }
        }
        cells = next;
    }
    cells.into_iter().filter(|c| !c.faces.is_empty()).collect()
}

/// Length of segment `pq` lying inside convex counterclockwise polygon `q` by more than `tol`.
pub fn inside_length(p: Point, r: Point, q: &[Point], tol: f64) -> f64 {
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for j in 0..q.len() {
        let (a, b) = (q[j], q[(j + 1) % q.len()]);
        let len = dist(a, b);
        let fp = cross(a, b, p) / len - tol;
        let fr = cross(a, b, r) / len - tol;
        if fp < 0.0 && fr < 0.0 {
            return 0.0;
        }
        if fp < 0.0 {
            t0 = t0.max(fp / (fp - fr));
        } else if fr < 0.0 {
            t1 = t1.min(fp / (fp - fr));
        }
    }
    (t1 - t0).max(0.0) * dist(p, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: [Point; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    #[test]
    fn area_is_signed_by_orientation() {
        let mut cw = SQUARE.to_vec();
        cw.reverse();
        assert_eq!(area(&SQUARE), 1.0);
        assert_eq!(area(&cw), -1.0);
    }

    #[test]
    fn clip_keeps_the_left_half() {
        let left = clip(&SQUARE, [0.5, 0.0], [0.5, 1.0]);
        assert!((area(&left) - 0.5).abs() < 1e-12);
        assert!(left.iter().all(|p| p[0] <= 0.5 + 1e-12));
    }

    #[test]
    fn overlay_of_two_offset_squares_has_one_shared_cell() {
        let a = SQUARE.to_vec();
        let b: Vec<Point> = SQUARE.iter().map(|p| [p[0] + 0.5, p[1]]).collect();
        let cells = overlay(&[a, b]);
        let shared: Vec<_> = cells.iter().filter(|c| c.faces.len() == 2).collect();
        assert_eq!(shared.len(), 1);
        assert!((area(&shared[0].polygon) - 0.5).abs() < 1e-12);
        let total: f64 = cells.iter().map(|c| area(&c.polygon)).sum();
        assert!((total - 1.5).abs() < 1e-12);
    }

    #[test]
    fn a_segment_along_a_side_is_not_inside() {
        assert_eq!(
            inside_length([0.0, 0.0], [1.0, 0.0], &SQUARE, INSIDE_TOL),
            0.0
        );
        let through = inside_length([-1.0, 0.5], [2.0, 0.5], &SQUARE, INSIDE_TOL);
        assert!((through - 1.0).abs() < 1e-6);
    }
}
