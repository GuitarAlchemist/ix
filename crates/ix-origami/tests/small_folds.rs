//! Small folds built by hand, each holding one situation the crane does not have: an unassigned
//! crease, a flat joint along a crease, two flat joints on one line, and the structure problems
//! that stop the checks. Each comes with a stated order the rules must reject and one they must
//! accept.

use ix_origami::{analyse, Fold, Report, Rule, Unchecked};
use serde_json::{json, Value};

/// A crease pattern and its folded frame, with faces 0.. in the order given.
fn fold(
    vertices: &[[f64; 2]],
    folded: &[[f64; 2]],
    edges: &[([usize; 2], &str)],
    faces: &[&[usize]],
    face_orders: &[[i64; 3]],
) -> Value {
    json!({
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
    })
}

fn analysed(v: &Value) -> Report {
    analyse(&Fold::from_value(v).unwrap()).expect("passes the structure checks")
}

fn refused(v: &Value) -> Vec<String> {
    match analyse(&Fold::from_value(v).unwrap()) {
        Err(Unchecked::Structure(p)) => p,
        other => panic!("expected a structure refusal, got {other:?}"),
    }
}

/// Faces 0 (A) and 1 (B): a 2×1 strip folded in half along x = 1, B over A. Then a 1×1 sheet
/// at x 3..4, folded to x 0.5..1.5 so the crease line runs through it: face 2 (E), or faces 2
/// (C) and 3 (D) when `split` cuts it with a flat joint that folds onto the crease line.
fn strip_and_sheet(crease: &str, split: bool, face_orders: &[[i64; 3]]) -> Value {
    let mut vertices = vec![
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
    let mut edges = vec![
        ([0, 1], "B"),
        ([1, 2], "B"),
        ([2, 3], "B"),
        ([3, 4], "B"),
        ([4, 5], "B"),
        ([5, 0], "B"),
        ([1, 4], crease),
    ];
    let mut faces: Vec<&[usize]> = vec![&[0, 1, 4, 5][..], &[1, 2, 3, 4]];
    if split {
        vertices.extend([[3.5, 0.0], [3.5, 1.0]]);
        edges.extend([
            ([6, 10], "B"),
            ([10, 7], "B"),
            ([7, 8], "B"),
            ([8, 11], "B"),
            ([11, 9], "B"),
            ([9, 6], "B"),
            ([10, 11], "F"),
        ]);
        faces.extend([&[6, 10, 11, 9][..], &[10, 7, 8, 11]]);
    } else {
        edges.extend([([6, 7], "B"), ([7, 8], "B"), ([8, 9], "B"), ([9, 6], "B")]);
        faces.push(&[6, 7, 8, 9]);
    }
    // B folds over by x -> 2 - x; the sheet moves by -2.5.
    let folded: Vec<[f64; 2]> = vertices
        .iter()
        .enumerate()
        .map(|(v, &[x, y])| match v {
            2 | 3 => [2.0 - x, y],
            0 | 1 | 4 | 5 => [x, y],
            _ => [x - 2.5, y],
        })
        .collect();
    fold(&vertices, &folded, &edges, &faces, face_orders)
}

/// A over B over nothing: face 2 stated between A and B, then above B.
const BETWEEN: [[i64; 3]; 3] = [[0, 2, -1], [2, 1, 1], [0, 1, 1]];
const ON_TOP: [[i64; 3]; 3] = [[0, 2, -1], [2, 1, -1], [0, 1, 1]];

#[test]
fn a_face_between_a_valley_it_runs_through_is_rejected() {
    let r = analysed(&strip_and_sheet("V", false, &BETWEEN));
    assert_eq!(r.rejected_by, [Rule::TacoTortilla]);
    assert_eq!(r.layers.taco_tortilla.violations, [[6, 2]]);
    assert!(analysed(&strip_and_sheet("V", false, &ON_TOP)).ok);
}

#[test]
fn an_unassigned_crease_folded_flat_enters_the_layer_rules() {
    let r = analysed(&strip_and_sheet("U", false, &BETWEEN));
    assert_eq!(r.rejected_by, [Rule::TacoTortilla]);
    // It states no direction, so the adjacency rule has nothing to check.
    assert_eq!(r.layers.adjacency.checked, 0);
    let ok = analysed(&strip_and_sheet("U", false, &ON_TOP));
    assert!(ok.ok, "{:?}", ok.rejected_by);
    assert_eq!(ok.layers.taco_tortilla.checked, 1);
}

#[test]
fn a_flat_joint_along_a_crease_carries_its_sheet_across() {
    // The crease line runs along the joint, through no face's inside: only the joint shows
    // that the sheet C-D crosses it, so C cannot lie between A and B.
    let r = analysed(&strip_and_sheet("V", true, &BETWEEN));
    assert_eq!(r.rejected_by, [Rule::TacoTortilla]);
    assert_eq!(r.layers.taco_tortilla.violations, [[6, 2]]);
    let ok = analysed(&strip_and_sheet("V", true, &ON_TOP));
    assert!(ok.ok, "{:?}", ok.rejected_by);
    assert_eq!(ok.layers.taco_tortilla.checked, 1);
}

/// Two 2×1 strips, each cut at x = 1 by a flat joint and laid flat on the same spot: faces
/// 0 (C1) and 1 (D1), 2 (C2) and 3 (D2).
fn two_sheets(face_orders: &[[i64; 3]]) -> Value {
    let strip = |x: f64| {
        [
            [x, 0.0],
            [x + 1.0, 0.0],
            [x + 2.0, 0.0],
            [x + 2.0, 1.0],
            [x + 1.0, 1.0],
            [x, 1.0],
        ]
    };
    let vertices: Vec<[f64; 2]> = strip(0.0).into_iter().chain(strip(3.0)).collect();
    let folded: Vec<[f64; 2]> = strip(0.0).into_iter().chain(strip(0.0)).collect();
    let mut edges = Vec::new();
    for o in [0, 6] {
        for i in 0..6 {
            edges.push(([o + i, o + (i + 1) % 6], "B"));
        }
        edges.push(([o + 1, o + 4], "F"));
    }
    let faces: [&[usize]; 4] = [
        &[0, 1, 4, 5],
        &[1, 2, 3, 4],
        &[6, 7, 10, 11],
        &[7, 8, 9, 10],
    ];
    fold(&vertices, &folded, &edges, &faces, face_orders)
}

#[test]
fn two_sheets_crossing_one_line_keep_one_order_on_both_sides() {
    // C1 over C2 on the left, D1 under D2 on the right: the sheets pass through each other.
    let r = analysed(&two_sheets(&[[0, 2, 1], [1, 3, -1]]));
    assert_eq!(r.rejected_by, [Rule::TortillaTortilla]);
    assert_eq!(r.layers.tortilla_tortilla.violations, [[6, 13]]);
    let ok = analysed(&two_sheets(&[[0, 2, 1], [1, 3, 1]]));
    assert!(ok.ok, "{:?}", ok.rejected_by);
    assert_eq!(ok.layers.tortilla_tortilla.checked, 1);
    // One side unstated: reported, not failed.
    let open = analysed(&two_sheets(&[[0, 2, 1]]));
    assert_eq!(open.layers.tortilla_tortilla.undetermined, 1);
    assert!(open.ok);
}

// --- structure ------------------------------------------------------------------------------

#[test]
fn a_fold_angle_on_a_flat_edge_is_refused() {
    let mut v = two_sheets(&[[0, 2, 1], [1, 3, 1]]);
    let mut angles = vec![0.0; 14];
    angles[6] = 10.0;
    v["file_frames"][0]["edges_foldAngle"] = json!(angles);
    let p = refused(&v);
    assert_eq!(
        p,
        ["edge 6 (F): fold angle +10.000, where the spec sets 0 on flat, unassigned and border edges"]
    );
    angles[6] = 0.0;
    v["file_frames"][0]["edges_foldAngle"] = json!(angles);
    assert!(analysed(&v).ok);
}

#[test]
fn a_folded_frame_that_overrides_the_crease_pattern_is_refused() {
    let base = two_sheets(&[[0, 2, 1], [1, 3, 1]]);
    for key in ["edges_assignment", "faces_vertices", "edges_vertices"] {
        let mut v = base.clone();
        v["file_frames"][0][key] = base[key].clone();
        let p = refused(&v);
        assert!(
            p.iter().any(|s| s.contains(&format!("overrides {key}"))),
            "{p:?}"
        );
    }
    let mut v = base.clone();
    v["faceOrders"] = json!([[0, 2, -1]]);
    assert!(refused(&v)
        .iter()
        .any(|s| s.starts_with("faceOrders on the crease pattern")));
}

#[test]
fn an_isolated_vertex_is_skipped_not_failed() {
    let mut v = strip_and_sheet("V", false, &ON_TOP);
    v["vertices_coords"]
        .as_array_mut()
        .unwrap()
        .push(json!([1.5, 0.5]));
    let frame = &mut v["file_frames"][0]["vertices_coords"];
    frame.as_array_mut().unwrap().push(json!([1.5, 0.5]));
    let r = analysed(&v);
    assert_eq!(r.local_theorems.skipped_isolated, 1);
    assert_eq!(r.local_theorems.applicable, 0);
    assert!(r.ok);
}

#[test]
fn a_face_of_zero_area_is_refused() {
    // Face E folded flat onto a sliver 1e-12 high: still convex, but it has no orientation to
    // read and covers nothing.
    let mut v = strip_and_sheet("V", false, &ON_TOP);
    for i in [8, 9] {
        v["file_frames"][0]["vertices_coords"][i][1] = json!(1e-12);
    }
    assert_eq!(
        refused(&v),
        ["faces of zero area: 0 in the crease pattern, 1 folded"]
    );
}

#[test]
fn a_star_shaped_face_is_refused_as_non_convex() {
    // A pentagram turns the same way at every vertex but winds twice.
    let star: Vec<[f64; 2]> = (0..5)
        .map(|k| {
            let a = std::f64::consts::FRAC_PI_2 + f64::from(2 * k) * std::f64::consts::TAU / 5.0;
            [a.cos(), a.sin()]
        })
        .collect();
    let edges: Vec<([usize; 2], &str)> = (0..5).map(|i| ([i, (i + 1) % 5], "B")).collect();
    let p = refused(&fold(&star, &star, &edges, &[&[0, 1, 2, 3, 4]], &[]));
    assert!(
        p.iter()
            .any(|s| s.starts_with("non-convex faces: 1 in the crease pattern")),
        "{p:?}"
    );
}
