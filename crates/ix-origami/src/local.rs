//! Local flat-foldability conditions at a vertex of the crease pattern.
//!
//! Kawasaki (Kawasaki-Justin): a one-vertex crease pattern folds flat iff the alternating sum of
//! the sector angles around the vertex is 0. Maekawa (Maekawa-Justin): at a flat-folded vertex the
//! numbers of mountain and valley folds differ by two. Both are local: passing them says nothing
//! about the whole sheet, whose flat-foldability is NP-complete to test (Bern and Hayes 1996).

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::TAU;

use serde::Serialize;

use crate::fold::{Assignment, Fold};

fn incident(fold: &Fold) -> Vec<Vec<(usize, usize)>> {
    let mut inc = vec![Vec::new(); fold.vertices_coords.len()];
    for (k, &[u, w]) in fold.edges_vertices.iter().enumerate() {
        inc[u].push((k, w));
        inc[w].push((k, u));
    }
    inc
}

fn border_vertices(fold: &Fold) -> BTreeSet<usize> {
    let mut border = BTreeSet::new();
    for (&[u, w], a) in fold.edges_vertices.iter().zip(&fold.edges_assignment) {
        if *a == Assignment::B {
            border.extend([u, w]);
        }
    }
    border
}

/// Vertices on at least one edge and on no border edge.
pub fn interior_vertices(fold: &Fold) -> Vec<usize> {
    let border = border_vertices(fold);
    let inc = incident(fold);
    (0..fold.vertices_coords.len())
        .filter(|&v| !border.contains(&v) && !inc[v].is_empty())
        .collect()
}

/// The alternating sum of sector angles (radians) at each interior vertex, and the vertices
/// where it is not 0 within `tol` or where the number of creases is odd.
pub fn kawasaki(fold: &Fold, tol: f64) -> (BTreeMap<usize, f64>, Vec<usize>) {
    let xy = &fold.vertices_coords;
    let inc = incident(fold);
    let (mut sums, mut failing) = (BTreeMap::new(), Vec::new());
    for v in interior_vertices(fold) {
        let mut dirs: Vec<f64> = inc[v]
            .iter()
            .map(|&(_, w)| (xy[w][1] - xy[v][1]).atan2(xy[w][0] - xy[v][0]))
            .collect();
        dirs.sort_by(f64::total_cmp);
        let n = dirs.len();
        let alt: f64 = (0..n)
            .map(|i| {
                let sector = (dirs[(i + 1) % n] - dirs[i]).rem_euclid(TAU);
                if i % 2 == 0 {
                    sector
                } else {
                    -sector
                }
            })
            .sum();
        sums.insert(v, alt);
        if n % 2 == 1 || alt.abs() > tol {
            failing.push(v);
        }
    }
    (sums, failing)
}

/// Mountains minus valleys at each interior vertex, and the vertices where it is not ±2.
pub fn maekawa(fold: &Fold) -> (BTreeMap<usize, i64>, Vec<usize>) {
    let inc = incident(fold);
    let asg = &fold.edges_assignment;
    let (mut diff, mut failing) = (BTreeMap::new(), Vec::new());
    for v in interior_vertices(fold) {
        let count =
            |want: Assignment| inc[v].iter().filter(|&&(k, _)| asg[k] == want).count() as i64;
        let d = count(Assignment::M) - count(Assignment::V);
        diff.insert(v, d);
        if d.abs() != 2 {
            failing.push(v);
        }
    }
    (diff, failing)
}

/// Kawasaki and Maekawa where their hypotheses hold.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LocalTheorems {
    pub applicable: usize,
    pub skipped_border: usize,
    pub skipped_other_creases: usize,
    /// Vertices on no edge.
    pub skipped_isolated: usize,
    pub kawasaki_failing: Vec<usize>,
    pub kawasaki_max_abs_alt_sum: f64,
    pub maekawa_failing: Vec<usize>,
    /// How many applicable vertices have each value of M − V.
    pub maekawa_m_minus_v: BTreeMap<i64, usize>,
}

/// Kawasaki and Maekawa at the vertices where they apply: interior (on edges, none of them
/// border), with every crease a mountain or a valley, in a state folded flat. Other vertices
/// are skipped and counted by reason.
pub fn local_theorems(fold: &Fold, tol: f64) -> LocalTheorems {
    let border = border_vertices(fold);
    let mut other = BTreeSet::new();
    for (&[u, w], a) in fold.edges_vertices.iter().zip(&fold.edges_assignment) {
        if *a != Assignment::B && !a.is_fold() {
            other.extend([u, w]);
        }
    }
    let interior: BTreeSet<usize> = interior_vertices(fold).into_iter().collect();
    let applicable: BTreeSet<usize> = interior.difference(&other).copied().collect();
    let on_edges = border.len() + interior.len();
    let (sums, kfail) = kawasaki(fold, tol);
    let (diff, mfail) = maekawa(fold);
    let mut m_minus_v = BTreeMap::new();
    for v in &applicable {
        *m_minus_v.entry(diff[v]).or_insert(0) += 1;
    }
    LocalTheorems {
        applicable: applicable.len(),
        skipped_border: border.len(),
        skipped_other_creases: other.difference(&border).count(),
        skipped_isolated: fold.vertices_coords.len() - on_edges,
        kawasaki_failing: kfail
            .into_iter()
            .filter(|v| applicable.contains(v))
            .collect(),
        kawasaki_max_abs_alt_sum: applicable.iter().map(|v| sums[v].abs()).fold(0.0, f64::max),
        maekawa_failing: mfail
            .into_iter()
            .filter(|v| applicable.contains(v))
            .collect(),
        maekawa_m_minus_v: m_minus_v,
    }
}
