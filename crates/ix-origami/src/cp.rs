//! `.cp` crease-pattern files, read into a [`Fold`] (crease pattern only, no folded frame).
//!
//! One segment per line, `code x1 y1 x2 y2`, separated by blanks. Codes 1, 2 and 3 are border,
//! mountain and valley. Every other code is an auxiliary line: it is left out and counted in
//! [`CpRead::skipped_aux`]. That table is what both ORIPA's loader (`LoaderCP`) and Oriedita's
//! (`CreasePatternReader`) do; see `tests/cp.rs` for the sources.
//!
//! A `.cp` file has no vertices, faces, folded state or layer order. The reader makes the
//! vertices (endpoints within `tol` of each other are one vertex) and refuses a file whose
//! segments are not already split where they meet, because the vertex checks would read a
//! crossing as two unrelated creases. It does not split them. The result carries no folded
//! frame, so [`Fold::structure`] and the folded-state checks refuse it; the crease-pattern
//! checks ([`crate::local::local_theorems`]) run on it.
//!
//! `tol` is absolute, in the file's own units.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::fold::{Assignment, Fold, Point};

/// Most segments one file may hold: the not-split check compares every pair.
pub const MAX_LINES: usize = 5_000;

/// A crease pattern read from a `.cp` file.
#[derive(Debug, Clone, PartialEq)]
pub struct CpRead {
    pub fold: Fold,
    /// Lines whose code is not 1, 2 or 3, left out of the fold.
    pub skipped_aux: usize,
}

/// Why a file could not be read as a `.cp` crease pattern. Line numbers count from 1 in the file.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CpError {
    #[error("the tolerance must be a positive finite number")]
    Tolerance,
    #[error("line {line}: {problem}")]
    Line { line: usize, problem: String },
    #[error("no border, mountain or valley line")]
    Empty,
    #[error("more than {limit} segments")]
    TooMany { limit: usize },
    #[error("line {line}: the segment has zero length")]
    ZeroLength { line: usize },
    #[error("lines {} and {} are the same segment", lines[0], lines[1])]
    Duplicate { lines: [usize; 2] },
    #[error(
        "lines {} and {} meet other than at a shared end: split the pattern where they meet",
        lines[0], lines[1]
    )]
    NotSplit { lines: [usize; 2] },
}

fn bad(line: usize, problem: &str) -> CpError {
    CpError::Line {
        line,
        problem: problem.to_string(),
    }
}

/// Distance from `p` to the segment `a`-`b`.
fn point_segment(p: Point, a: Point, b: Point) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 == 0.0 {
        0.0
    } else {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0)
    };
    (p[0] - (a[0] + t * dx)).hypot(p[1] - (a[1] + t * dy))
}

fn orient(a: Point, b: Point, c: Point) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Distance between two segments: 0 when they cross.
fn segment_segment(a0: Point, a1: Point, b0: Point, b1: Point) -> f64 {
    let (d0, d1) = (orient(a0, a1, b0), orient(a0, a1, b1));
    let (d2, d3) = (orient(b0, b1, a0), orient(b0, b1, a1));
    if d0 * d1 < 0.0 && d2 * d3 < 0.0 {
        return 0.0;
    }
    point_segment(a0, b0, b1)
        .min(point_segment(a1, b0, b1))
        .min(point_segment(b0, a0, a1))
        .min(point_segment(b1, a0, a1))
}

/// The vertex within `tol` of `p`, or a new one. Vertices are found through a grid of cells
/// `tol` wide, so a lookup visits nine cells.
struct Vertices {
    tol: f64,
    coords: Vec<Point>,
    grid: HashMap<(i64, i64), Vec<usize>>,
}

impl Vertices {
    fn cell(&self, p: Point) -> (i64, i64) {
        (
            (p[0] / self.tol).floor() as i64,
            (p[1] / self.tol).floor() as i64,
        )
    }

    fn id(&mut self, p: Point) -> usize {
        let (cx, cy) = self.cell(p);
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                for &v in self.grid.get(&(x, y)).into_iter().flatten() {
                    let q = self.coords[v];
                    if (q[0] - p[0]).hypot(q[1] - p[1]) <= self.tol {
                        return v;
                    }
                }
            }
        }
        self.coords.push(p);
        self.grid
            .entry((cx, cy))
            .or_default()
            .push(self.coords.len() - 1);
        self.coords.len() - 1
    }
}

/// Reads a `.cp` file. See the module docs for what is read, left out and refused.
pub fn read_cp(text: &str, tol: f64) -> Result<CpRead, CpError> {
    if !(tol.is_finite() && tol > 0.0) {
        return Err(CpError::Tolerance);
    }
    // Coordinates must fit the vertex grid.
    let reach = 1e15 * tol;
    let mut verts = Vertices {
        tol,
        coords: Vec::new(),
        grid: HashMap::new(),
    };
    let mut edges: Vec<[usize; 2]> = Vec::new();
    let mut assignment = Vec::new();
    let mut at_line = Vec::new();
    let mut skipped_aux = 0;
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let tokens: Vec<&str> = raw.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        if tokens.len() != 5 {
            return Err(bad(
                line,
                &format!("expected `code x1 y1 x2 y2`, found {} fields", tokens.len()),
            ));
        }
        let code: u32 = tokens[0]
            .parse()
            .map_err(|_| bad(line, "the line code is not a non-negative integer"))?;
        let mut xy = [0.0; 4];
        for (k, t) in tokens[1..].iter().enumerate() {
            let v: f64 = t
                .parse()
                .map_err(|_| bad(line, &format!("`{t}` is not a number")))?;
            if !v.is_finite() || v.abs() > reach {
                return Err(bad(line, &format!("`{t}` is out of range")));
            }
            xy[k] = v;
        }
        let a = match code {
            1 => Assignment::B,
            2 => Assignment::M,
            3 => Assignment::V,
            _ => {
                skipped_aux += 1;
                continue;
            }
        };
        if edges.len() == MAX_LINES {
            return Err(CpError::TooMany { limit: MAX_LINES });
        }
        let (u, w) = (verts.id([xy[0], xy[1]]), verts.id([xy[2], xy[3]]));
        if u == w {
            return Err(CpError::ZeroLength { line });
        }
        edges.push([u, w]);
        assignment.push(a);
        at_line.push(line);
    }
    if edges.is_empty() {
        return Err(CpError::Empty);
    }
    let xy = &verts.coords;
    let mut seen = HashSet::new();
    for (k, &[u, w]) in edges.iter().enumerate() {
        if !seen.insert((u.min(w), u.max(w))) {
            let first = edges[..k]
                .iter()
                .position(|e| (e[0].min(e[1]), e[0].max(e[1])) == (u.min(w), u.max(w)))
                .expect("the pair was seen");
            return Err(CpError::Duplicate {
                lines: [at_line[first], at_line[k]],
            });
        }
    }
    for i in 0..edges.len() {
        for j in i + 1..edges.len() {
            let [a0, a1] = edges[i];
            let [b0, b1] = edges[j];
            let near = match [
                (a0, a1, b0, b1),
                (a0, a1, b1, b0),
                (a1, a0, b0, b1),
                (a1, a0, b1, b0),
            ]
            .into_iter()
            .find(|&(s, _, t, _)| s == t)
            {
                // One shared end: each free end must keep off the other segment.
                Some((_, ea, _, eb)) => {
                    point_segment(xy[eb], xy[a0], xy[a1]) <= tol
                        || point_segment(xy[ea], xy[b0], xy[b1]) <= tol
                }
                None => segment_segment(xy[a0], xy[a1], xy[b0], xy[b1]) <= tol,
            };
            if near {
                return Err(CpError::NotSplit {
                    lines: [at_line[i], at_line[j]],
                });
            }
        }
    }
    Ok(CpRead {
        fold: Fold {
            file_spec: Some(1.2),
            frame_classes: vec!["creasePattern".to_string()],
            vertices_coords: verts.coords,
            edges_vertices: edges,
            edges_assignment: assignment,
            faces_vertices: Vec::new(),
            frames: Vec::new(),
            other: serde_json::Map::new(),
        },
        skipped_aux,
    })
}
