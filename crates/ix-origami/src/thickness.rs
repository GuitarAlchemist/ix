//! What a material thickness implies for a stated flat folded state.
//!
//! The other checks treat the sheet as having no thickness. Given a thickness `t` (the full
//! thickness, in crease-pattern units), this module reports two readings of the stated layer
//! order:
//!
//! - **ply**: the most faces stacked at one point. Any stack that keeps the stated positions is
//!   at least `ply · t` thick, and a sheet whose faces may sit at a different height in each
//!   overlay cell (paper that drapes) reaches it. That reading refuses nothing the layer rules
//!   do not already refuse.
//! - **parallel rigid panels**: each panel (faces joined by flat joints) keeps one height and
//!   stays parallel to the sheet. Such a stack exists if and only if the stated "above" over
//!   overlapping panels has no cycle. When every overlapping pair is stated, its least height is
//!   the longest chain of panels, times `t`.
//!
//! A cycle is refused only for `t > 0`: at zero thickness every face lies in one plane, and the
//! FOLD spec allows a stated order with cycles. So at `t = 0` nothing here refuses anything, and
//! every height is 0. A refusal means "no stack of parallel panels", not "cannot be made": boards
//! tilted by about `t` can escape it. Nothing here can say that a state can be made either: the
//! two faces of a folded crease touch along it, which no exact thick state allows.
//!
//! The decisions behind this reading are in `docs/adr/0007-origami-thickness-as-parallel-rigid-panels.md`.

use std::collections::{btree_map::Entry, BTreeMap};

use ix_graph::graph::Graph;
use serde::Serialize;
use thiserror::Error;

use crate::layers::{above, Context, Crease};

/// The stack a thickness implies.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Stack {
    pub t: f64,
    /// The most faces in one overlay cell.
    pub ply: usize,
    /// `ply · t`: the least height of any stack at the stated positions.
    pub ply_height: f64,
    pub rigid: Rigid,
}

/// The stated order read as parallel rigid panels, one height each.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Rigid {
    /// The stated order over overlapping panels has no cycle.
    pub acyclic: bool,
    /// One directed cycle when `acyclic` is false, as stated pairs `[below, above]`. Each pair's
    /// upper face is on the same panel as the next pair's lower face.
    pub cycle: Option<Vec<[usize; 2]>>,
    /// The panels in the longest chain; `max(chain, ply)`, a lower bound, when `levels_exact` is
    /// false. `None` with a cycle.
    pub levels: Option<usize>,
    /// Every overlapping pair of panels is stated, so `levels` is the least.
    pub levels_exact: bool,
    /// `levels · t`.
    pub height: Option<f64>,
    /// Each face's level, from 0 at the bottom: one stack that respects every stated pair, among
    /// others. `None` with a cycle, or when `levels_exact` is false.
    pub face_levels: Option<Vec<usize>>,
    /// Overlapping pairs of faces on different panels with no stated order.
    pub undetermined_pairs: usize,
    /// `t > 0` and the stated order has a cycle: no stack of parallel rigid panels exists.
    pub refused: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[error("thickness.t must be a finite number, 0 or more; got {0}")]
pub struct BadThickness(pub f64);

/// The faces' panels: each face mapped to the smallest face it is joined to by flat joints.
fn panels(n: usize, joints: &[Crease]) -> Vec<usize> {
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for j in joints {
        let (a, b) = (root(&mut parent, j.faces[0]), root(&mut parent, j.faces[1]));
        parent[a.max(b)] = a.min(b);
    }
    (0..n).map(|f| root(&mut parent, f)).collect()
}

/// One directed cycle of `g`, as its nodes in order, by depth-first search.
fn find_cycle(g: &Graph, n: usize) -> Option<Vec<usize>> {
    // 0: not seen, 1: on the current path, 2: done.
    let (mut state, mut parent) = (vec![0u8; n], vec![usize::MAX; n]);
    for s in 0..n {
        if state[s] != 0 {
            continue;
        }
        state[s] = 1;
        let mut path = vec![(s, 0)];
        while let Some(top) = path.last_mut() {
            let u = top.0;
            let Some(&(v, _)) = g.neighbors(u).get(top.1) else {
                state[u] = 2;
                path.pop();
                continue;
            };
            top.1 += 1;
            match state[v] {
                0 => {
                    state[v] = 1;
                    parent[v] = u;
                    path.push((v, 0));
                }
                1 => {
                    let mut cycle = vec![u];
                    while *cycle.last().expect("starts with u") != v {
                        cycle.push(parent[*cycle.last().expect("not empty")]);
                    }
                    cycle.reverse();
                    return Some(cycle);
                }
                _ => {}
            }
        }
    }
    None
}

/// The stack `t` implies for the fold behind `ctx`. `Err` for a `t` that is negative or not
/// finite.
// @ai:invariant at t = 0 stack() refuses nothing and every height is 0 [T:test conf:0.9 src:thickness::at_zero_thickness_nothing_changes]
pub fn stack(ctx: &Context, t: f64) -> Result<Stack, BadThickness> {
    if !(t.is_finite() && t >= 0.0) {
        return Err(BadThickness(t));
    }
    let n = ctx.orient.len();
    let ply = ctx
        .geo
        .cells
        .iter()
        .map(|c| c.faces.len())
        .max()
        .unwrap_or(0);
    let panel = panels(n, &ctx.geo.joints);
    // An arc from the lower panel to the upper one for every stated overlapping pair, and the
    // first stated pair behind each arc.
    let mut g = Graph::with_nodes(n);
    let mut witness: BTreeMap<(usize, usize), [usize; 2]> = BTreeMap::new();
    let mut undetermined_pairs = 0;
    for &(f, h) in &ctx.geo.overlapping {
        let Some(f_above) = above(&ctx.rel, &ctx.orient, f, h) else {
            if panel[f] != panel[h] {
                undetermined_pairs += 1;
            }
            continue;
        };
        let [lo, hi] = if f_above { [h, f] } else { [f, h] };
        let arc = (panel[lo], panel[hi]);
        if let Entry::Vacant(slot) = witness.entry(arc) {
            slot.insert([lo, hi]);
            g.add_edge(arc.0, arc.1, 1.0);
        }
    }
    let levels_of = |order: Vec<usize>| {
        let mut level = vec![0usize; n];
        for u in order {
            for &(v, _) in g.neighbors(u) {
                level[v] = level[v].max(level[u] + 1);
            }
        }
        level
    };
    let rigid = match g.topological_sort() {
        None => {
            let cycle = find_cycle(&g, n).expect("a graph with no topological order has a cycle");
            let k = cycle.len();
            Rigid {
                acyclic: false,
                cycle: Some(
                    (0..k)
                        .map(|i| witness[&(cycle[i], cycle[(i + 1) % k])])
                        .collect(),
                ),
                levels: None,
                levels_exact: false,
                height: None,
                face_levels: None,
                undetermined_pairs,
                refused: t > 0.0,
            }
        }
        Some(order) => {
            let level = levels_of(order);
            let chain = level.iter().max().map_or(0, |&l| l + 1);
            let exact = undetermined_pairs == 0;
            let levels = if exact { chain } else { chain.max(ply) };
            Rigid {
                acyclic: true,
                cycle: None,
                levels: Some(levels),
                levels_exact: exact,
                height: Some(levels as f64 * t),
                face_levels: exact.then(|| panel.iter().map(|&p| level[p]).collect()),
                undetermined_pairs,
                refused: false,
            }
        }
    };
    Ok(Stack {
        t,
        ply,
        ply_height: ply as f64 * t,
        rigid,
    })
}
