//! Controls: changes that describe the same folded object, so no check may change its verdict.
//! A check whose result moves under one of them depends on numbering or placement, not on the
//! fold.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::fold::{Fold, Point};

/// Where [`renumber`] sent each old index: `vertices[old] = new`, and so on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Permutation {
    pub vertices: Vec<usize>,
    pub edges: Vec<usize>,
    pub faces: Vec<usize>,
}

fn place<T: Clone>(items: &[T], perm: &[usize]) -> Vec<T> {
    let mut out = items.to_vec();
    for (old, x) in items.iter().enumerate() {
        out[perm[old]] = x.clone();
    }
    out
}

/// The same object with vertices, edges and faces renumbered, `faceOrders` reshuffled, each
/// face's vertex list rotated (still counterclockwise) and each edge's ends swapped at random.
pub fn renumber(fold: &Fold, seed: u64) -> (Fold, Permutation) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm = |n: usize| {
        let mut p: Vec<usize> = (0..n).collect();
        p.shuffle(&mut rng);
        p
    };
    let vp = perm(fold.vertices_coords.len());
    let ep = perm(fold.edges_vertices.len());
    let fp = perm(fold.faces_vertices.len());
    let mut out = fold.clone();
    out.vertices_coords = place(&fold.vertices_coords, &vp);
    let edges: Vec<[usize; 2]> = fold
        .edges_vertices
        .iter()
        .map(|&[u, w]| {
            if rng.random_bool(0.5) {
                [vp[u], vp[w]]
            } else {
                [vp[w], vp[u]]
            }
        })
        .collect();
    out.edges_vertices = place(&edges, &ep);
    out.edges_assignment = place(&fold.edges_assignment, &ep);
    let faces: Vec<Vec<usize>> = fold
        .faces_vertices
        .iter()
        .map(|f| {
            let mut g: Vec<usize> = f.iter().map(|&v| vp[v]).collect();
            g.rotate_left(rng.random_range(0..f.len()));
            g
        })
        .collect();
    out.faces_vertices = place(&faces, &fp);
    for (new, old) in out.frames.iter_mut().zip(&fold.frames) {
        new.vertices_coords = place(&old.vertices_coords, &vp);
        new.edges_fold_angle = old.edges_fold_angle.as_ref().map(|a| place(a, &ep));
        new.face_orders = old
            .face_orders
            .iter()
            .map(|&[f, g, s]| [fp[f as usize] as i64, fp[g as usize] as i64, s])
            .collect();
        new.face_orders.shuffle(&mut rng);
    }
    let permutation = Permutation {
        vertices: vp,
        edges: ep,
        faces: fp,
    };
    (out, permutation)
}

/// A rigid in-plane motion of the folded frame: a rotation by `angle` (radians), then a shift.
/// Orientations are kept.
pub fn move_folded(fold: &Fold, angle: f64, shift: Point) -> Fold {
    let (c, s) = (angle.cos(), angle.sin());
    let mut out = fold.clone();
    for p in &mut out.folded_mut().vertices_coords {
        *p = [
            c * p[0] - s * p[1] + shift[0],
            s * p[0] + c * p[1] + shift[1],
        ];
    }
    out
}

/// The folded model turned over (`y -> -y`): every orientation flips and so does "up", so
/// `faceOrders`, which are relative to the faces' own normals, stay the same.
pub fn turn_over(fold: &Fold) -> Fold {
    let mut out = fold.clone();
    for p in &mut out.folded_mut().vertices_coords {
        p[1] = -p[1];
    }
    out
}
