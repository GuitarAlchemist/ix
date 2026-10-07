//! The stack a thickness implies: the crane, small folds built by hand, the reduction at zero
//! thickness, and the controls that must leave it unchanged.

use ix_origami::controls::{move_folded, renumber, turn_over};
use ix_origami::layers::above;
use ix_origami::{analyse, stack, BadThickness, Context, Fold, Stack};
use serde_json::{json, Value};

fn crane() -> Fold {
    Fold::from_json_str(include_str!("fixtures/crane-f325f3fd8a.fold")).unwrap()
}

fn stacked(fold: &Fold, t: f64) -> Stack {
    stack(&Context::new(fold).expect("passes the structure checks"), t).unwrap()
}

/// A crease pattern and its folded frame, with faces 0.. in the order given.
fn fold(
    vertices: &[[f64; 2]],
    folded: &[[f64; 2]],
    edges: &[([usize; 2], &str)],
    faces: &[&[usize]],
    face_orders: &[[i64; 3]],
) -> Fold {
    Fold::from_value(&json!({
        "file_spec": 1.2,
        "vertices_coords": vertices,
        "edges_vertices": edges.iter().map(|(e, _)| e).collect::<Vec<_>>(),
        "edges_assignment": edges.iter().map(|(_, a)| a).collect::<Vec<_>>(),
        "faces_vertices": faces,
        "file_frames": [{
            "frame_classes": ["foldedForm"],
            "frame_parent": 0,
            "frame_inherit": true,
            "vertices_coords": folded,
            "faceOrders": face_orders
        }]
    }))
    .unwrap()
}

/// Separate sheets, one rectangle each, every side a border edge: `[x0, y0, x1, y1]` in the
/// crease pattern, and the folded positions of its four corners.
fn sheets(rects: &[([f64; 4], [[f64; 2]; 4])], face_orders: &[[i64; 3]]) -> Fold {
    let (mut vertices, mut folded, mut edges, mut faces) = (vec![], vec![], vec![], vec![]);
    for (i, &([x0, y0, x1, y1], corners)) in rects.iter().enumerate() {
        vertices.extend([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
        folded.extend(corners);
        let o = 4 * i;
        edges.extend((0..4).map(|k| ([o + k, o + (k + 1) % 4], "B")));
        faces.push([o, o + 1, o + 2, o + 3]);
    }
    let faces: Vec<&[usize]> = faces.iter().map(|f| &f[..]).collect();
    fold(&vertices, &folded, &edges, &faces, face_orders)
}

fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> [[f64; 2]; 4] {
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

/// Four 3×1 strips woven so that each lies over the next: A over B over C over D over A. Each
/// pair overlaps in its own corner, so no overlay cell holds a cycle.
fn weave() -> Fold {
    sheets(
        &[
            ([0.0, 0.0, 3.0, 1.0], square(0.0, 0.0, 3.0, 1.0)),
            // Turned a quarter: (x, y) -> (3 - y, x - 4).
            (
                [4.0, 0.0, 7.0, 1.0],
                [[3.0, 0.0], [3.0, 3.0], [2.0, 3.0], [2.0, 0.0]],
            ),
            ([8.0, 0.0, 11.0, 1.0], square(0.0, 2.0, 3.0, 3.0)),
            // (x, y) -> (1 - y, x - 12).
            (
                [12.0, 0.0, 15.0, 1.0],
                [[1.0, 0.0], [1.0, 3.0], [0.0, 3.0], [0.0, 0.0]],
            ),
        ],
        &[[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 0, 1]],
    )
}

/// A 2×1 sheet cut at x = 1 by a flat joint (faces 0 and 1), and a separate 1×1 sheet (face 2)
/// laid across the joint, stated over face 0 and under face 1: it would pierce the sheet.
fn pierce() -> Fold {
    let vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [3.0, 0.0],
        [4.0, 0.0],
        [4.0, 1.0],
        [3.0, 1.0],
    ];
    let folded: Vec<[f64; 2]> = vertices
        .iter()
        .enumerate()
        .map(|(v, &[x, y])| if v < 6 { [x, y] } else { [x - 2.5, y] })
        .collect();
    let mut edges: Vec<([usize; 2], &str)> = (0..6).map(|k| ([k, (k + 1) % 6], "B")).collect();
    edges.push(([1, 4], "F"));
    edges.extend((0..4).map(|k| ([6 + k, 6 + (k + 1) % 4], "B")));
    fold(
        &vertices,
        &folded,
        &edges,
        &[&[0, 1, 4, 5], &[1, 2, 3, 4], &[6, 7, 8, 9]],
        &[[2, 0, 1], [1, 2, 1]],
    )
}

// --- the crane ------------------------------------------------------------------------------

#[test]
fn the_crane_is_28_layers_at_its_thickest_and_36_as_rigid_panels() {
    let fold = crane();
    let ctx = Context::new(&fold).unwrap();
    // Paperboard 0.4 mm thick on a 150 mm sheet: the crease pattern is the unit square.
    let t = 0.4 / 150.0;
    let s = stack(&ctx, t).unwrap();
    assert_eq!((s.ply, s.ply_height), (28, 28.0 * t));
    let r = &s.rigid;
    assert!(r.acyclic && !r.refused && r.cycle.is_none());
    assert_eq!(
        (r.levels, r.levels_exact, r.height),
        (Some(36), true, Some(36.0 * t))
    );
    assert_eq!(r.undetermined_pairs, 0);
    // The levels respect every stated pair.
    let level = r.face_levels.as_ref().unwrap();
    assert_eq!(level.len(), 59);
    for &(f, g) in &ctx.geo.overlapping {
        let f_above = above(&ctx.rel, &ctx.orient, f, g).unwrap();
        assert_eq!(level[f] > level[g], f_above, "{f} {g}");
    }
    // A chain of 36 faces, each stated above the one before: no stack has fewer levels.
    let chain = [
        39, 43, 35, 29, 3, 0, 4, 27, 22, 16, 20, 19, 55, 54, 51, 58, 57, 52, 53, 56, 18, 21, 17,
        23, 25, 5, 24, 31, 34, 42, 45, 40, 47, 50, 7, 1,
    ];
    for w in chain.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        assert!(
            ctx.geo.overlapping.contains(&(lo.min(hi), lo.max(hi))),
            "{lo} {hi}"
        );
        assert_eq!(
            above(&ctx.rel, &ctx.orient, hi, lo),
            Some(true),
            "{lo} {hi}"
        );
    }
}

#[test]
fn at_zero_thickness_nothing_changes() {
    let fold = crane();
    let s = stacked(&fold, 0.0);
    assert_eq!((s.ply, s.ply_height), (28, 0.0));
    assert_eq!((s.rigid.levels, s.rigid.height), (Some(36), Some(0.0)));
    assert!(!s.rigid.refused);
    // A cycle is reported but not refused: at zero thickness the faces share one plane.
    let w = stacked(&weave(), 0.0);
    assert!(!w.rigid.acyclic && !w.rigid.refused);
    // The checks themselves do not depend on a thickness.
    assert!(analyse(&weave()).unwrap().ok);
}

#[test]
fn heights_scale_with_the_thickness() {
    let fold = crane();
    let (a, b) = (stacked(&fold, 0.25), stacked(&fold, 0.5));
    assert_eq!(b.ply_height, 2.0 * a.ply_height);
    assert_eq!(b.rigid.height.unwrap(), 2.0 * a.rigid.height.unwrap());
    assert_eq!(a.rigid.face_levels, b.rigid.face_levels);
}

#[test]
fn controls_leave_the_stack_unchanged() {
    let (fold, t) = (crane(), 0.01);
    let base = stacked(&fold, t);
    for seed in 1..=5 {
        let (other, perm) = renumber(&fold, seed);
        let s = stacked(&other, t);
        assert_eq!((s.ply, s.rigid.levels), (28, Some(36)), "seed {seed}");
        let (old, new) = (
            base.rigid.face_levels.as_ref(),
            s.rigid.face_levels.as_ref(),
        );
        for (f, &l) in old.unwrap().iter().enumerate() {
            assert_eq!(new.unwrap()[perm.faces[f]], l, "seed {seed}, face {f}");
        }
    }
    assert_eq!(stacked(&move_folded(&fold, 0.7, [3.0, -2.0]), t), base);
    // Turned over, the order is read from the other side: the same levels in number, the same
    // heights. Which face is at which level is not asserted.
    let over = stacked(&turn_over(&fold), t);
    assert_eq!((over.ply, over.ply_height), (base.ply, base.ply_height));
    assert_eq!(
        (over.rigid.levels, over.rigid.height),
        (base.rigid.levels, base.rigid.height)
    );
}

// --- small folds ----------------------------------------------------------------------------

#[test]
fn a_square_folded_in_half_is_two_layers() {
    // The right half folds over the left along a valley at x = 1.
    let vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ];
    let folded = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ];
    let mut edges: Vec<([usize; 2], &str)> = (0..6).map(|k| ([k, (k + 1) % 6], "B")).collect();
    edges.push(([1, 4], "V"));
    let f = fold(
        &vertices,
        &folded,
        &edges,
        &[&[0, 1, 4, 5], &[1, 2, 3, 4]],
        &[[0, 1, 1]],
    );
    assert!(analyse(&f).unwrap().ok);
    let s = stacked(&f, 0.5);
    assert_eq!(
        (s.ply, s.rigid.levels, s.rigid.height),
        (2, Some(2), Some(1.0))
    );
    assert_eq!(s.rigid.face_levels, Some(vec![0, 1]));
}

#[test]
fn an_unfolded_square_is_one_layer() {
    let s = stacked(
        &sheets(&[([0.0, 0.0, 1.0, 1.0], square(0.0, 0.0, 1.0, 1.0))], &[]),
        0.5,
    );
    assert_eq!(
        (s.ply, s.rigid.levels, s.rigid.height),
        (1, Some(1), Some(0.5))
    );
    assert_eq!(s.rigid.face_levels, Some(vec![0]));
}

#[test]
fn an_unstated_overlap_gives_a_lower_bound() {
    let s = stacked(
        &sheets(
            &[
                ([0.0, 0.0, 1.0, 1.0], square(0.0, 0.0, 1.0, 1.0)),
                ([2.0, 0.0, 3.0, 1.0], square(0.0, 0.0, 1.0, 1.0)),
            ],
            &[],
        ),
        0.5,
    );
    let r = &s.rigid;
    assert_eq!((s.ply, r.undetermined_pairs), (2, 1));
    assert_eq!(
        (r.levels, r.levels_exact, r.height),
        (Some(2), false, Some(1.0))
    );
    assert_eq!(r.face_levels, None);
}

#[test]
fn a_weave_passes_the_layer_rules_but_has_no_rigid_stack() {
    assert!(analyse(&weave()).unwrap().ok);
    let cycle = json!([[0, 3], [3, 2], [2, 1], [1, 0]]);
    let refused = |t: f64| -> Value {
        json!({
            "t": t,
            "ply": 2,
            "ply_height": 2.0 * t,
            "rigid": {
                "acyclic": false,
                "cycle": cycle,
                "levels": null,
                "levels_exact": false,
                "height": null,
                "face_levels": null,
                "undetermined_pairs": 0,
                "refused": t > 0.0
            }
        })
    };
    for t in [0.0, 1e-300, 0.5] {
        let s = serde_json::to_value(stacked(&weave(), t)).unwrap();
        assert_eq!(s, refused(t), "t = {t}");
    }
}

#[test]
fn a_sheet_pierced_across_a_flat_joint_has_no_rigid_stack() {
    // The layer rules accept it: no rule yet checks a face against both faces of a flat joint.
    assert!(analyse(&pierce()).unwrap().ok);
    let r = stacked(&pierce(), 0.1).rigid;
    assert!(!r.acyclic && r.refused);
    assert_eq!(r.cycle, Some(vec![[0, 2], [2, 1]]));
    assert!(!stacked(&pierce(), 0.0).rigid.refused);
}

#[test]
fn a_thickness_that_is_not_a_finite_non_negative_number_is_refused() {
    let ctx = Context::new(&weave()).unwrap();
    for t in [-1.0, -1e-300, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = stack(&ctx, t).unwrap_err();
        assert!(matches!(err, BadThickness(_)), "{t}");
        assert!(err.to_string().starts_with("thickness.t must be"), "{t}");
    }
}
