//! The stated layer order of a flat folded state, and the rules it must obey.
//!
//! The order comes from the folded frame's `faceOrders`; nothing here computes one. Four rules
//! test it: the adjacency rule across every crease, no cycle among the faces stacked in any
//! overlay cell, taco-tortilla (a face a crease runs through cannot lie between the crease's
//! faces) and taco-taco (two creases folded onto one line, on the same side, nest or stack but
//! never interleave).
//!
//! The functions here index faces and vertices without checking: they expect a fold that passed
//! [`Fold::structure`]. [`analyse`] and [`Context::new`] check it first.

use std::collections::{BTreeMap, BTreeSet};

use ix_graph::graph::Graph;
use serde::Serialize;

use crate::fold::{Assignment, Fold, Point};
use crate::geometry::{area, dist, inside_length, is_convex, overlay, Cell, INSIDE_TOL, LEN_TOL};
use crate::local::{local_theorems, LocalTheorems};

/// +1 for a face whose vertices stay counterclockwise in the folded frame (its normal faces the
/// viewer, +z), -1 for a face turned over.
pub fn orientation(fold: &Fold) -> Vec<i8> {
    let y = &fold.folded().vertices_coords;
    fold.faces_vertices
        .iter()
        .map(|f| {
            let p: Vec<Point> = f.iter().map(|&v| y[v]).collect();
            if area(&p) > 0.0 {
                1
            } else {
                -1
            }
        })
        .collect()
}

/// The stated `s` of every ordered face pair, 0 where nothing is stated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relations {
    n: usize,
    s: Vec<i8>,
}

impl Relations {
    fn new(n: usize) -> Self {
        Self {
            n,
            s: vec![0; n * n],
        }
    }

    /// `s` for `[f, g, s]`, or `None` if the pair is not stated.
    pub fn get(&self, f: usize, g: usize) -> Option<i8> {
        let s = self.s[f * self.n + g];
        (s != 0).then_some(s)
    }

    /// The relations with the order of faces `f` and `g` reversed, both ways.
    pub fn flipped(&self, f: usize, g: usize) -> Self {
        let mut out = self.clone();
        out.s[f * self.n + g] = -self.s[f * self.n + g];
        out.s[g * self.n + f] = -self.s[g * self.n + f];
        out
    }

    /// The stated pairs, as `(lower face, higher face)`.
    pub fn stated_pairs(&self) -> BTreeSet<(usize, usize)> {
        (0..self.n)
            .flat_map(|f| (0..self.n).map(move |g| (f, g)))
            .filter(|&(f, g)| self.s[f * self.n + g] != 0)
            .map(|(f, g)| (f.min(g), f.max(g)))
            .collect()
    }
}

/// The relations stated by `faceOrders`, each converse filled in by the spec's rule; and the
/// ordered pairs where two statements disagree.
// @ai:invariant on the crane every converse the spec's rule fills in agrees with any converse the file states, so relations() reports no conflict [T:test conf:0.9 src:crane::the_crane_passes_every_check]
pub fn relations(fold: &Fold, orient: &[i8]) -> (Relations, Vec<(usize, usize)>) {
    let mut rel = Relations::new(fold.faces_vertices.len());
    let mut conflicts = Vec::new();
    for &[f, g, s] in &fold.folded().face_orders {
        if s == 0 {
            continue;
        }
        let (f, g, s) = (f as usize, g as usize, s as i8);
        let t = if orient[f] != orient[g] { s } else { -s };
        for (a, b, val) in [(f, g, s), (g, f, t)] {
            let old = rel.s[a * rel.n + b];
            if old != 0 && old != val {
                conflicts.push((a, b));
            }
            rel.s[a * rel.n + b] = val;
        }
    }
    (rel, conflicts)
}

/// Whether `f` lies above `g` (towards +z, the viewer); `None` if not stated.
pub fn above(rel: &Relations, orient: &[i8], f: usize, g: usize) -> Option<bool> {
    rel.get(f, g).map(|s| s * orient[g] > 0)
}

/// Face shapes, crease pattern against folded frame.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Isometry {
    /// Folded lengths divided by crease-pattern lengths, over every vertex pair of every face.
    pub scale: f64,
    pub max_length_change_at_scale_1: f64,
    pub max_length_change: f64,
    pub ok: bool,
}

/// The distance between any two vertices of one face, crease pattern against folded frame,
/// after dividing the folded one by one global scale. In the plane those distances fix each
/// face up to a rigid motion or a reflection.
pub fn face_isometry(fold: &Fold, tol: f64) -> Isometry {
    let (x, y) = (&fold.vertices_coords, &fold.folded().vertices_coords);
    let pairs: Vec<(usize, usize)> = fold
        .faces_vertices
        .iter()
        .flat_map(|f| (0..f.len()).flat_map(move |i| (i + 1..f.len()).map(move |j| (f[i], f[j]))))
        .collect();
    let folded: f64 = pairs.iter().map(|&(a, b)| dist(y[a], y[b])).sum();
    let flat: f64 = pairs.iter().map(|&(a, b)| dist(x[a], x[b])).sum();
    let k = folded / flat;
    let worst = |scale: f64| {
        pairs
            .iter()
            .map(|&(a, b)| (dist(y[a], y[b]) / scale - dist(x[a], x[b])).abs())
            .fold(0.0, f64::max)
    };
    let max_length_change = worst(k);
    Isometry {
        scale: k,
        max_length_change_at_scale_1: worst(1.0),
        max_length_change,
        ok: max_length_change <= tol,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CreaseOrientation {
    pub checked: usize,
    pub violations: Vec<usize>,
}

/// Folding a crease flat turns one face over relative to the other: mountain and valley edges
/// join faces of opposite orientation, flat and join edges faces of the same orientation.
pub fn crease_orientation(fold: &Fold, orient: &[i8]) -> CreaseOrientation {
    let (ef, _) = fold.edge_faces();
    let mut out = CreaseOrientation {
        checked: 0,
        violations: Vec::new(),
    };
    for (k, a) in fold.edges_assignment.iter().enumerate() {
        let judged = matches!(
            a,
            Assignment::M | Assignment::V | Assignment::F | Assignment::J
        );
        if ef[k].len() != 2 || !judged {
            continue;
        }
        out.checked += 1;
        if (orient[ef[k][0]] != orient[ef[k][1]]) != a.is_fold() {
            out.violations.push(k);
        }
    }
    out
}

/// Each face in the folded frame, counterclockwise.
pub fn folded_polygons(fold: &Fold, orient: &[i8]) -> Vec<Vec<Point>> {
    let y = &fold.folded().vertices_coords;
    fold.faces_vertices
        .iter()
        .zip(orient)
        .map(|(f, &o)| {
            let mut p: Vec<Point> = f.iter().map(|&v| y[v]).collect();
            if o < 0 {
                p.reverse();
            }
            p
        })
        .collect()
}

/// A mountain or valley edge and its two faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Crease {
    pub edge: usize,
    pub faces: [usize; 2],
}

/// A crease that runs through the inside of another face.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tortilla {
    pub crease: Crease,
    pub face: usize,
}

/// Two creases folded onto one line, sharing no face, their faces on the same side of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Taco {
    pub first: Crease,
    pub second: Crease,
}

/// Everything the layer rules need that depends on geometry only, so swaps are re-checked
/// without recomputing it.
#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub cells: Vec<Cell>,
    /// Face pairs `(f, g)`, `f < g`, that share a cell.
    pub overlapping: BTreeSet<(usize, usize)>,
    pub creases: Vec<Crease>,
    pub tortillas: Vec<Tortilla>,
    pub tacos: Vec<Taco>,
    /// Collinear crease pairs whose faces lie on opposite sides of the line: no rule applies.
    pub opposite_tacos: usize,
    pub assignment: Vec<Assignment>,
}

/// The geometry of a fold that passed [`Fold::structure`] with convex faces.
pub fn prepare(fold: &Fold, orient: &[i8]) -> Geometry {
    let y = &fold.folded().vertices_coords;
    let polys = folded_polygons(fold, orient);
    let cells = overlay(&polys);
    let mut overlapping = BTreeSet::new();
    for c in &cells {
        for &f in &c.faces {
            for &g in c.faces.range(f + 1..) {
                overlapping.insert((f, g));
            }
        }
    }
    let (ef, _) = fold.edge_faces();
    let creases: Vec<Crease> = fold
        .edges_assignment
        .iter()
        .enumerate()
        .filter(|&(k, a)| a.is_fold() && ef[k].len() == 2)
        .map(|(k, _)| Crease {
            edge: k,
            faces: [ef[k][0], ef[k][1]],
        })
        .collect();
    let ends = |c: &Crease| {
        let [u, w] = fold.edges_vertices[c.edge];
        (y[u], y[w])
    };
    let mut tortillas = Vec::new();
    for c in &creases {
        let (p, r) = ends(c);
        for (t, q) in polys.iter().enumerate() {
            if !c.faces.contains(&t) && inside_length(p, r, q, INSIDE_TOL) > 1e-6 {
                tortillas.push(Tortilla {
                    crease: *c,
                    face: t,
                });
            }
        }
    }
    let centroid: Vec<Point> = polys
        .iter()
        .map(|p| {
            let n = p.len() as f64;
            [
                p.iter().map(|v| v[0]).sum::<f64>() / n,
                p.iter().map(|v| v[1]).sum::<f64>() / n,
            ]
        })
        .collect();
    let (mut tacos, mut opposite_tacos) = (Vec::new(), 0);
    for (i, c1) in creases.iter().enumerate() {
        let (p, q) = ends(c1);
        let len = dist(p, q);
        let d = [(q[0] - p[0]) / len, (q[1] - p[1]) / len];
        let side = |c: Point| d[0] * (c[1] - p[1]) - d[1] * (c[0] - p[0]);
        let along = |c: Point| (c[0] - p[0]) * d[0] + (c[1] - p[1]) * d[1];
        for c2 in &creases[i + 1..] {
            let (r, s) = ends(c2);
            if side(r).abs() > 1e-9 || side(s).abs() > 1e-9 {
                continue;
            }
            let (a0, a1) = (along(r).min(along(s)), along(r).max(along(s)));
            let shared = c1.faces.iter().any(|f| c2.faces.contains(f));
            if len.min(a1) - a0.max(0.0) <= 1e-6 || shared {
                continue;
            }
            if (side(centroid[c1.faces[0]]) > 0.0) == (side(centroid[c2.faces[0]]) > 0.0) {
                tacos.push(Taco {
                    first: *c1,
                    second: *c2,
                });
            } else {
                opposite_tacos += 1;
            }
        }
    }
    Geometry {
        cells,
        overlapping,
        creases,
        tortillas,
        tacos,
        opposite_tacos,
        assignment: fold.edges_assignment.clone(),
    }
}

/// Whether the stated "above" among `faces` has a cycle.
fn cyclic(faces: &BTreeSet<usize>, rel: &Relations, orient: &[i8]) -> bool {
    let faces: Vec<usize> = faces.iter().copied().collect();
    let mut g = Graph::with_nodes(faces.len());
    for (i, &f) in faces.iter().enumerate() {
        for (j, &h) in faces.iter().enumerate() {
            if i != j && above(rel, orient, f, h) == Some(true) {
                g.add_edge(i, j, 1.0);
            }
        }
    }
    g.topological_sort().is_none()
}

/// Whether face `x` lies strictly between faces `a` and `b`; `None` if an order is not stated.
/// Faces overlapping in one connected region keep one order, so this needs only the two pairs.
fn between(rel: &Relations, orient: &[i8], x: usize, a: usize, b: usize) -> Option<bool> {
    Some(above(rel, orient, x, a)? != above(rel, orient, x, b)?)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Adjacency {
    pub checked: usize,
    /// Creases whose two faces have no stated order.
    pub unstated: Vec<usize>,
    pub violations: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Cells {
    pub multi_layer: usize,
    pub max_layers: usize,
    /// Indices into [`Geometry::cells`].
    pub cyclic: Vec<usize>,
}

/// Taco-tortilla or taco-taco. A violation is `[crease edge, face]` for taco-tortilla and
/// `[crease edge, crease edge]` for taco-taco.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TacoRule {
    pub checked: usize,
    /// Items a missing pair leaves open: reported, not failed.
    pub undetermined: usize,
    pub violations: Vec<[usize; 2]>,
}

/// A layer rule, named as in the report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rule {
    Adjacency,
    Cells,
    TacoTortilla,
    TacoTaco,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LayerChecks {
    pub adjacency: Adjacency,
    pub cells: Cells,
    pub taco_tortilla: TacoRule,
    pub taco_taco: TacoRule,
}

impl LayerChecks {
    /// The rules with at least one violation, in [`Rule`] order.
    pub fn rejected_by(&self) -> Vec<Rule> {
        [
            (Rule::Adjacency, !self.adjacency.violations.is_empty()),
            (Rule::Cells, !self.cells.cyclic.is_empty()),
            (
                Rule::TacoTortilla,
                !self.taco_tortilla.violations.is_empty(),
            ),
            (Rule::TacoTaco, !self.taco_taco.violations.is_empty()),
        ]
        .into_iter()
        .filter_map(|(r, failed)| failed.then_some(r))
        .collect()
    }
}

/// The adjacency rule on every crease: `s = +1` across a valley, `-1` across a mountain, both
/// ways.
pub fn adjacency(geo: &Geometry, rel: &Relations, focus: Option<[usize; 2]>) -> Adjacency {
    let mut out = Adjacency {
        checked: 0,
        unstated: Vec::new(),
        violations: Vec::new(),
    };
    for c in &geo.creases {
        let [f1, f2] = c.faces;
        if focus.is_some_and(|[f, g]| !((f1 == f && f2 == g) || (f1 == g && f2 == f))) {
            continue;
        }
        let want = if geo.assignment[c.edge] == Assignment::V {
            1
        } else {
            -1
        };
        out.checked += 1;
        match (rel.get(f1, f2), rel.get(f2, f1)) {
            (Some(a), Some(b)) if a == want && b == want => {}
            (Some(_), Some(_)) => out.violations.push(c.edge),
            _ => out.unstated.push(c.edge),
        }
    }
    out
}

/// At every point the faces stacked there are in one order: the stated "above" has no cycle in
/// any cell.
pub fn cells_acyclic(
    geo: &Geometry,
    rel: &Relations,
    orient: &[i8],
    focus: Option<[usize; 2]>,
) -> Cells {
    let mut out = Cells {
        multi_layer: 0,
        max_layers: 0,
        cyclic: Vec::new(),
    };
    for (i, c) in geo.cells.iter().enumerate() {
        if c.faces.len() < 2 || focus.is_some_and(|p| !p.iter().all(|f| c.faces.contains(f))) {
            continue;
        }
        out.multi_layer += 1;
        out.max_layers = out.max_layers.max(c.faces.len());
        if cyclic(&c.faces, rel, orient) {
            out.cyclic.push(i);
        }
    }
    out
}

/// A face that a crease runs through cannot lie between the crease's two faces.
pub fn taco_tortilla(
    geo: &Geometry,
    rel: &Relations,
    orient: &[i8],
    focus: Option<[usize; 2]>,
) -> TacoRule {
    let mut out = TacoRule {
        checked: 0,
        undetermined: 0,
        violations: Vec::new(),
    };
    for t in &geo.tortillas {
        let [f1, f2] = t.crease.faces;
        let involved = [f1, f2, t.face];
        if focus.is_some_and(|p| !p.iter().all(|f| involved.contains(f))) {
            continue;
        }
        match between(rel, orient, t.face, f1, f2) {
            None => out.undetermined += 1,
            Some(b) => {
                out.checked += 1;
                if b {
                    out.violations.push([t.crease.edge, t.face]);
                }
            }
        }
    }
    out
}

/// Two creases folded onto one line, their faces on the same side: the two pairs nest or stack,
/// never interleave (exactly one face of one pair between the faces of the other).
pub fn taco_taco(
    geo: &Geometry,
    rel: &Relations,
    orient: &[i8],
    focus: Option<[usize; 2]>,
) -> TacoRule {
    let mut out = TacoRule {
        checked: 0,
        undetermined: 0,
        violations: Vec::new(),
    };
    for t in &geo.tacos {
        let ([f1, f2], [g1, g2]) = (t.first.faces, t.second.faces);
        let involved = [f1, f2, g1, g2];
        if focus.is_some_and(|p| !p.iter().all(|f| involved.contains(f))) {
            continue;
        }
        let b1 = between(rel, orient, g1, f1, f2);
        let b2 = between(rel, orient, g2, f1, f2);
        match (b1, b2) {
            (Some(b1), Some(b2)) => {
                out.checked += 1;
                if b1 != b2 {
                    out.violations.push([t.first.edge, t.second.edge]);
                }
            }
            _ => out.undetermined += 1,
        }
    }
    out
}

/// The four layer rules. With `focus = Some([f, g])`, only the items that involve both faces:
/// after swapping one pair, nothing else can change.
pub fn layer_checks(
    geo: &Geometry,
    rel: &Relations,
    orient: &[i8],
    focus: Option<[usize; 2]>,
) -> LayerChecks {
    LayerChecks {
        adjacency: adjacency(geo, rel, focus),
        cells: cells_acyclic(geo, rel, orient, focus),
        taco_tortilla: taco_tortilla(geo, rel, orient, focus),
        taco_taco: taco_taco(geo, rel, orient, focus),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Counts {
    pub vertices: usize,
    pub edges: usize,
    pub faces: usize,
    pub assignment: BTreeMap<Assignment, usize>,
    #[serde(rename = "faceOrders")]
    pub face_orders: usize,
}

/// How many faces keep (`up`) or reverse (`down`) their crease-pattern orientation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Orientation {
    pub up: usize,
    pub down: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Overlap {
    pub overlapping_pairs: usize,
    pub stated_and_overlapping: usize,
    pub overlapping_unstated: usize,
    pub stated_not_overlapping: usize,
    pub cells: usize,
}

/// Every check on one fold. `ok` needs all of them; unstated or undetermined items are
/// reported, not failed.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub counts: Counts,
    pub orientation: Orientation,
    pub converse_conflicts: Vec<(usize, usize)>,
    pub isometry: Isometry,
    pub crease_orientation: CreaseOrientation,
    pub local_theorems: LocalTheorems,
    pub overlap: Overlap,
    pub layers: LayerChecks,
    pub tacos_on_opposite_sides: usize,
    pub rejected_by: Vec<Rule>,
    pub ok: bool,
}

/// [`Fold::structure`], then convex faces in both frames: what the geometry needs.
fn usable(fold: &Fold) -> Result<(), Vec<String>> {
    let problems = fold.structure();
    if !problems.is_empty() {
        return Err(problems);
    }
    let non_convex = |coords: &[Point]| {
        fold.faces_vertices
            .iter()
            .filter(|f| !is_convex(&f.iter().map(|&v| coords[v]).collect::<Vec<_>>()))
            .count()
    };
    let (cp, folded) = (
        non_convex(&fold.vertices_coords),
        non_convex(&fold.folded().vertices_coords),
    );
    if cp + folded > 0 {
        return Err(vec![format!(
            "non-convex faces: {cp} in the crease pattern, {folded} folded"
        )]);
    }
    Ok(())
}

/// Orientation, relations and geometry of one fold, for swap experiments.
#[derive(Debug, Clone, PartialEq)]
pub struct Context {
    pub orient: Vec<i8>,
    pub rel: Relations,
    pub geo: Geometry,
}

impl Context {
    /// The fold's structure problems, or non-convex faces, if it has any.
    pub fn new(fold: &Fold) -> Result<Self, Vec<String>> {
        usable(fold)?;
        let orient = orientation(fold);
        let (rel, _) = relations(fold, &orient);
        let geo = prepare(fold, &orient);
        Ok(Self { orient, rel, geo })
    }
}

/// Every check on one fold; `Err` with the structure problems (or non-convex faces) that stop
/// the checks from running.
pub fn analyse(fold: &Fold) -> Result<Report, Vec<String>> {
    usable(fold)?;
    let orient = orientation(fold);
    let (rel, conflicts) = relations(fold, &orient);
    let geo = prepare(fold, &orient);
    let layers = layer_checks(&geo, &rel, &orient, None);
    let stated = rel.stated_pairs();
    let mut assignment = BTreeMap::new();
    for a in &fold.edges_assignment {
        *assignment.entry(*a).or_insert(0) += 1;
    }
    let up = orient.iter().filter(|&&o| o > 0).count();
    let isometry = face_isometry(fold, LEN_TOL);
    let crease_orientation = crease_orientation(fold, &orient);
    let local_theorems = local_theorems(fold, 1e-9);
    let rejected_by = layers.rejected_by();
    let ok = conflicts.is_empty()
        && isometry.ok
        && crease_orientation.violations.is_empty()
        && local_theorems.kawasaki_failing.is_empty()
        && local_theorems.maekawa_failing.is_empty()
        && rejected_by.is_empty();
    Ok(Report {
        counts: Counts {
            vertices: fold.vertices_coords.len(),
            edges: fold.edges_vertices.len(),
            faces: fold.faces_vertices.len(),
            assignment,
            face_orders: fold.folded().face_orders.len(),
        },
        orientation: Orientation {
            up,
            down: orient.len() - up,
        },
        converse_conflicts: conflicts,
        isometry,
        crease_orientation,
        local_theorems,
        overlap: Overlap {
            overlapping_pairs: geo.overlapping.len(),
            stated_and_overlapping: stated.intersection(&geo.overlapping).count(),
            overlapping_unstated: geo.overlapping.difference(&stated).count(),
            stated_not_overlapping: stated.difference(&geo.overlapping).count(),
            cells: geo.cells.len(),
        },
        layers,
        tacos_on_opposite_sides: geo.opposite_tacos,
        rejected_by,
        ok,
    })
}

/// The same fold with the stated order of faces `f` and `g` reversed: the faulty swap.
pub fn swap(fold: &Fold, f: usize, g: usize) -> Fold {
    let mut out = fold.clone();
    let (f, g) = (f as i64, g as i64);
    for t in &mut out.folded_mut().face_orders {
        if (t[0] == f && t[1] == g) || (t[0] == g && t[1] == f) {
            t[2] = -t[2];
        }
    }
    out
}

/// One stated pair flipped alone, and the rules that reject the flip.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CensusRow {
    pub pair: [usize; 2],
    /// The two faces share a crease.
    pub adjacent: bool,
    pub rejected_by: Vec<Rule>,
}

/// Flips each stated pair alone, in file order, and records which rules reject it. A rule that
/// rejects no flip cannot fail, and proves nothing.
// @ai:invariant on the crane every one of the 838 single-pair flips is rejected by at least one layer rule, so each rule's check can fail [T:test conf:0.9 src:crane::every_single_swap_of_the_crane_is_rejected]
pub fn swap_census(fold: &Fold, ctx: &Context) -> Vec<CensusRow> {
    let adjacent: BTreeSet<[usize; 2]> = ctx
        .geo
        .creases
        .iter()
        .map(|c| [c.faces[0].min(c.faces[1]), c.faces[0].max(c.faces[1])])
        .collect();
    fold.folded()
        .face_orders
        .iter()
        .filter(|t| t[2] != 0)
        .map(|t| {
            let (f, g) = (t[0] as usize, t[1] as usize);
            let rel = ctx.rel.flipped(f, g);
            let layers = layer_checks(&ctx.geo, &rel, &ctx.orient, Some([f, g]));
            CensusRow {
                pair: [f, g],
                adjacent: adjacent.contains(&[f.min(g), f.max(g)]),
                rejected_by: layers.rejected_by(),
            }
        })
        .collect()
}

/// The first stated pair, in file order, whose flip `rule` rejects, among adjacent or
/// non-adjacent pairs.
pub fn first_swap(fold: &Fold, ctx: &Context, rule: Rule, adjacent: bool) -> Option<[usize; 2]> {
    swap_census(fold, ctx)
        .into_iter()
        .find(|row| row.adjacent == adjacent && row.rejected_by.contains(&rule))
        .map(|row| row.pair)
}
