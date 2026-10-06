//! The traditional crane (see `tests/fixtures/NOTICE.md`). Each check holds on the sourced crane
//! (the positive control), flips on a semantic mutant of the same object, and holds on
//! invariance controls of the same object: renumbering, a rigid motion of the folded frame, the
//! folded model turned over.
//!
//! The exact numbers come from a separate Python implementation of the same checks run on the
//! same file; the two agree on every one of them.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use ix_origami::controls::{move_folded, renumber, turn_over};
use ix_origami::layers::{crease_orientation, face_isometry, orientation};
use ix_origami::local::{interior_vertices, local_theorems};
use ix_origami::{
    analyse, analyse_within, first_swap, swap, swap_census, swap_census_within, Assignment,
    CensusRow, Context, Fold, FoldError, Limits, Report, Rule, Unchecked,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE: &[u8] = include_bytes!("fixtures/crane-f325f3fd8a.fold");
const FIXTURE_SHA256: &str = "e8c2423eb0e8e99ad36b7725a0a89b9946fb97a9fff18ef8edd2975518663ea3";
const LEN_TOL: f64 = 1e-9;

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The fixture, refused unless its bytes are the pinned ones.
fn load(bytes: &[u8], expected: &str) -> Result<Fold, String> {
    let got = sha256(bytes);
    if got != expected {
        return Err(format!("sha256 {got} is not the pinned {expected}"));
    }
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    Fold::from_json_str(text).map_err(|e| e.to_string())
}

fn crane() -> &'static Fold {
    static FOLD: OnceLock<Fold> = OnceLock::new();
    FOLD.get_or_init(|| load(FIXTURE, FIXTURE_SHA256).expect("the pinned crane loads"))
}

fn report() -> &'static Report {
    static REPORT: OnceLock<Report> = OnceLock::new();
    REPORT.get_or_init(|| analyse(crane()).expect("the crane passes the structure checks"))
}

fn context() -> &'static Context {
    static CONTEXT: OnceLock<Context> = OnceLock::new();
    CONTEXT.get_or_init(|| Context::new(crane()).expect("the crane passes the structure checks"))
}

fn census() -> &'static [CensusRow] {
    static CENSUS: OnceLock<Vec<CensusRow>> = OnceLock::new();
    CENSUS.get_or_init(|| swap_census(crane(), context()))
}

fn analysed(fold: &Fold) -> Report {
    analyse(fold).expect("passes the structure checks")
}

/// What must not change under renumbering or rigid motion. The number of overlay cells is left
/// out: the overlay's convex pieces depend on the order the faces are cut in.
fn fingerprint(r: &Report) -> [usize; 11] {
    let (o, l) = (&r.overlap, &r.layers);
    [
        o.overlapping_pairs,
        o.stated_and_overlapping,
        o.overlapping_unstated,
        o.stated_not_overlapping,
        r.crease_orientation.checked,
        r.local_theorems.applicable,
        r.local_theorems.skipped_border,
        l.adjacency.checked,
        l.cells.max_layers,
        l.taco_tortilla.checked,
        l.taco_taco.checked,
    ]
}

/// The first mountain crease with both ends interior.
fn interior_crease(fold: &Fold) -> usize {
    let interior = interior_vertices(fold);
    fold.edges_vertices
        .iter()
        .zip(&fold.edges_assignment)
        .position(|(&[u, w], &a)| {
            a == Assignment::M && interior.contains(&u) && interior.contains(&w)
        })
        .expect("the crane has an interior mountain crease")
}

fn name(rule: Rule) -> String {
    serde_json::to_value(rule)
        .expect("a rule serializes")
        .as_str()
        .expect("as a string")
        .to_string()
}

// --- sources --------------------------------------------------------------------------------

#[test]
fn fixture_is_pinned() {
    assert_eq!(sha256(FIXTURE), FIXTURE_SHA256);
}

#[test]
fn a_fixture_with_another_hash_is_refused() {
    assert!(load(FIXTURE, &"0".repeat(64)).is_err());
}

#[test]
fn fixture_is_fold_1_1_and_loads_in_the_1_2_layout() {
    let raw: Value = serde_json::from_slice(FIXTURE).unwrap();
    assert_eq!(raw["file_spec"], 1.1);
    assert_eq!(raw["frame_classes"], serde_json::json!(["foldedForm"]));
    assert_eq!(raw["faceOrders"].as_array().map(Vec::len), Some(838));

    let fold = crane();
    assert_eq!(fold.file_spec, Some(1.2));
    assert_eq!(fold.frame_classes, ["creasePattern"]);
    assert!(!fold.other.contains_key("faceOrders"));
    assert_eq!(fold.folded().face_orders.len(), 838);
    // Kept keys survive, and the 1.2 object reads back to the same fold.
    assert_eq!(fold.other["file_title"], "crane");
    assert_eq!(Fold::from_value(&fold.to_value()).as_ref(), Ok(fold));

    let mut bad = raw.clone();
    bad["file_frames"][0]["faceOrders"] = serde_json::json!([]);
    assert_eq!(Fold::from_value(&bad), Err(FoldError::Layout));
}

// --- structure ------------------------------------------------------------------------------

#[test]
fn structure_positive_control() {
    assert_eq!(crane().structure(), Vec::<String>::new());
}

#[test]
fn structure_mutant_face_order_out_of_range() {
    let mut bad = crane().clone();
    bad.folded_mut().face_orders[0][1] = bad.faces_vertices.len() as i64;
    assert!(!bad.structure().is_empty());
    assert!(analyse(&bad).is_err());
}

#[test]
fn structure_control_renumbered() {
    assert_eq!(renumber(crane(), 1).0.structure(), Vec::<String>::new());
}

// --- geometry -------------------------------------------------------------------------------

#[test]
fn geometry_positive_control() {
    let iso = face_isometry(crane(), LEN_TOL);
    assert!(iso.ok);
    // The file's folded frame is scaled.
    assert!((iso.scale - 1.4211008987678813).abs() < 1e-9);
    let orient = orientation(crane());
    assert!(crease_orientation(crane(), &orient).violations.is_empty());
}

#[test]
fn geometry_mutant_folded_vertex_moved() {
    let mut bad = crane().clone();
    bad.folded_mut().vertices_coords[4][0] += 0.01;
    assert!(!face_isometry(&bad, LEN_TOL).ok);
}

#[test]
fn geometry_mutant_folded_frame_collapsed() {
    // Every folded vertex on one point: the scale is 0 and every length change NaN.
    let mut bad = crane().clone();
    for p in &mut bad.folded_mut().vertices_coords {
        *p = [0.0, 0.0];
    }
    let iso = face_isometry(&bad, LEN_TOL);
    assert_eq!(iso.scale, 0.0);
    assert!(iso.max_length_change.is_nan());
    assert!(!iso.ok);
    let r = analysed(&bad);
    assert!(!r.ok && !r.isometry.ok);
}

#[test]
fn geometry_mutant_no_faces() {
    // Nothing to compare: the scale is 0/0, and an empty fold is not ok.
    let empty = serde_json::json!({
        "file_spec": 1.2,
        "vertices_coords": [],
        "edges_vertices": [],
        "edges_assignment": [],
        "faces_vertices": [],
        "file_frames": [{
            "frame_classes": ["foldedForm"],
            "frame_parent": 0,
            "frame_inherit": true,
            "vertices_coords": []
        }]
    });
    let r = analysed(&Fold::from_value(&empty).unwrap());
    assert!(r.isometry.scale.is_nan());
    assert!(!r.isometry.ok && !r.ok);
}

#[test]
fn geometry_mutant_crease_unfolded() {
    let mut bad = crane().clone();
    let k = bad
        .edges_assignment
        .iter()
        .position(|&a| a == Assignment::M)
        .unwrap();
    bad.edges_assignment[k] = Assignment::F;
    let orient = orientation(&bad);
    assert!(crease_orientation(&bad, &orient).violations.contains(&k));
}

#[test]
fn geometry_controls_renumbered_moved_turned_over() {
    let others = [
        renumber(crane(), 2).0,
        move_folded(crane(), 0.7, [3.0, -2.0]),
        turn_over(crane()),
    ];
    for other in &others {
        assert!(face_isometry(other, LEN_TOL).ok);
        let orient = orientation(other);
        assert!(crease_orientation(other, &orient).violations.is_empty());
    }
}

// --- Kawasaki and Maekawa -------------------------------------------------------------------

#[test]
fn local_positive_control() {
    let r = local_theorems(crane(), 1e-9);
    assert_eq!(
        (r.applicable, r.skipped_border, r.skipped_other_creases),
        (44, 12, 0)
    );
    assert!(r.kawasaki_failing.is_empty() && r.maekawa_failing.is_empty());
}

#[test]
fn local_mutant_one_crease_flipped() {
    let mut bad = crane().clone();
    let k = interior_crease(&bad);
    bad.edges_assignment[k] = Assignment::V;
    let mut ends = bad.edges_vertices[k].to_vec();
    ends.sort_unstable();
    let mut failing = local_theorems(&bad, 1e-9).maekawa_failing;
    failing.sort_unstable();
    assert_eq!(failing, ends);
}

#[test]
fn local_mutant_pattern_vertex_moved() {
    let mut bad = crane().clone();
    let v = interior_vertices(&bad)[0];
    bad.vertices_coords[v][0] += 0.01;
    assert!(local_theorems(&bad, 1e-9).kawasaki_failing.contains(&v));
}

#[test]
fn local_control_renumbered() {
    let a = local_theorems(crane(), 1e-9);
    let b = local_theorems(&renumber(crane(), 3).0, 1e-9);
    assert_eq!(b.applicable, a.applicable);
    assert!(b.kawasaki_failing.is_empty() && b.maekawa_failing.is_empty());
    assert_eq!(b.maekawa_m_minus_v, a.maekawa_m_minus_v);
}

// --- layer order ----------------------------------------------------------------------------

#[test]
fn layers_positive_control() {
    let r = report();
    assert!(r.ok);
    assert!(r.rejected_by.is_empty());
    let o = &r.overlap;
    assert_eq!(
        (
            o.overlapping_pairs,
            o.stated_and_overlapping,
            o.overlapping_unstated,
            o.stated_not_overlapping
        ),
        (838, 838, 0, 0)
    );
    assert!(r.layers.adjacency.unstated.is_empty());
}

#[test]
fn layers_faulty_swap_adjacent_faces() {
    let [f, g] = first_swap(crane(), context(), Rule::Adjacency, true).unwrap();
    let r = analysed(&swap(crane(), f, g));
    assert!(!r.ok);
    assert!(r.rejected_by.contains(&Rule::Adjacency));
}

#[test]
fn layers_faulty_swap_makes_a_cycle() {
    let [f, g] = first_swap(crane(), context(), Rule::Cells, false).unwrap();
    assert!(analysed(&swap(crane(), f, g))
        .rejected_by
        .contains(&Rule::Cells));
}

#[test]
fn layers_faulty_swaps_caught_by_one_taco_rule_only() {
    for rule in [Rule::TacoTortilla, Rule::TacoTaco] {
        let row = census()
            .iter()
            .find(|r| r.rejected_by == [rule])
            .unwrap_or_else(|| panic!("no flip is caught by {rule:?} alone"));
        let [f, g] = row.pair;
        assert_eq!(analysed(&swap(crane(), f, g)).rejected_by, [rule]);
    }
}

#[test]
fn every_single_swap_of_the_crane_is_rejected() {
    assert_eq!(census().len(), 838);
    let accepted: Vec<[usize; 2]> = census()
        .iter()
        .filter(|r| r.rejected_by.is_empty())
        .map(|r| r.pair)
        .collect();
    assert_eq!(accepted, Vec::<[usize; 2]>::new());
}

#[test]
fn layers_control_renumbered() {
    let (other, perm) = renumber(crane(), 4);
    let r = analysed(&other);
    assert!(r.ok);
    assert_eq!(fingerprint(&r), fingerprint(report()));
    let [f, g] = first_swap(crane(), context(), Rule::Adjacency, true).unwrap();
    let swapped = analysed(&swap(&other, perm.faces[f], perm.faces[g]));
    assert!(swapped.rejected_by.contains(&Rule::Adjacency));
}

#[test]
fn layers_controls_moved_and_turned_over() {
    for other in [move_folded(crane(), 0.7, [3.0, -2.0]), turn_over(crane())] {
        let r = analysed(&other);
        assert!(r.ok);
        assert_eq!(fingerprint(&r), fingerprint(report()));
    }
}

// --- agreement with the reference implementation --------------------------------------------

#[test]
fn layers_controls_rescaled() {
    // The paper's size and the folded frame's scale, as (crease pattern, folded frame) factors:
    // the checks rescale by powers of two first, so neither changes a verdict.
    for (cp, folded) in [(400.0, 400.0), (1.0, 1e-6), (1.0, 1e4), (3.0, 1e-5)] {
        let mut other = crane().clone();
        for p in &mut other.vertices_coords {
            *p = [p[0] * cp, p[1] * cp];
        }
        for p in &mut other.folded_mut().vertices_coords {
            *p = [p[0] * folded, p[1] * folded];
        }
        let r = analysed(&other);
        assert!(r.ok, "{cp} {folded}: {:?}", r.rejected_by);
        assert_eq!(fingerprint(&r), fingerprint(report()), "{cp} {folded}");
        let scale = 1.4211008987678813 * folded / cp;
        assert!(
            (r.isometry.scale / scale - 1.0).abs() < 1e-9,
            "{cp} {folded}"
        );
        assert!(!analysed(&swap(&other, 4, 2)).ok, "{cp} {folded}");
    }
    let mut tiny = crane().clone();
    for p in &mut tiny.folded_mut().vertices_coords {
        *p = [p[0] * 1e-6, p[1] * 1e-6];
    }
    let ctx = Context::new(&tiny).unwrap();
    assert!(swap_census(&tiny, &ctx)
        .iter()
        .all(|row| !row.rejected_by.is_empty()));
}

#[test]
fn the_crane_passes_every_check() {
    let r = report();
    let c = &r.counts;
    assert_eq!(
        (c.vertices, c.edges, c.faces, c.face_orders),
        (56, 114, 59, 838)
    );
    let assignment: Vec<(Assignment, usize)> = c.assignment.iter().map(|(&a, &n)| (a, n)).collect();
    assert_eq!(
        assignment,
        [
            (Assignment::B, 12),
            (Assignment::M, 61),
            (Assignment::V, 41)
        ]
    );
    assert_eq!((r.orientation.up, r.orientation.down), (29, 30));
    assert!(r.converse_conflicts.is_empty());

    let iso = &r.isometry;
    assert!(iso.ok);
    assert!((iso.scale - 1.4211008987678813).abs() < 1e-12);
    assert!((iso.max_length_change_at_scale_1 - 0.22394538226501237).abs() < 1e-12);
    assert!(iso.max_length_change < 1e-11);

    assert_eq!(r.crease_orientation.checked, 102);
    assert!(r.crease_orientation.violations.is_empty());

    let lt = &r.local_theorems;
    assert_eq!(
        (lt.applicable, lt.skipped_border, lt.skipped_other_creases),
        (44, 12, 0)
    );
    assert!(lt.kawasaki_max_abs_alt_sum < 1e-9);
    assert_eq!(lt.maekawa_m_minus_v, BTreeMap::from([(-2, 14), (2, 30)]));

    assert_eq!(r.overlap.cells, 83);
    let l = &r.layers;
    assert_eq!(
        (l.adjacency.checked, l.adjacency.violations.len()),
        (102, 0)
    );
    assert_eq!(
        (
            l.cells.multi_layer,
            l.cells.max_layers,
            l.cells.cyclic.len()
        ),
        (83, 28, 0)
    );
    assert_eq!(
        (l.taco_tortilla.checked, l.taco_tortilla.undetermined),
        (1049, 0)
    );
    assert_eq!((l.taco_taco.checked, l.taco_taco.undetermined), (196, 0));
    assert_eq!(r.tacos_on_opposite_sides, 9);
    assert!(r.ok);
}

#[test]
fn the_faulty_swaps_match_the_reference() {
    type Row = (
        [usize; 2],
        &'static [Rule],
        &'static [usize],
        usize,
        &'static [[usize; 2]],
        &'static [[usize; 2]],
    );
    use Rule::*;
    let expected: [Row; 4] = [
        (
            [0, 2],
            &[Adjacency, TacoTortilla, TacoTaco],
            &[81],
            0,
            &[[22, 2]],
            &[[80, 86]],
        ),
        (
            [4, 2],
            &[Cells, TacoTortilla, TacoTaco],
            &[],
            17,
            &[[9, 2], [22, 2]],
            &[[10, 80]],
        ),
        (
            [48, 38],
            &[TacoTortilla],
            &[],
            0,
            &[[12, 48], [28, 48], [62, 38]],
            &[],
        ),
        ([14, 8], &[TacoTaco], &[], 0, &[], &[[19, 34], [20, 49]]),
    ];
    let pick = |only: Rule| {
        census()
            .iter()
            .find(|r| r.rejected_by == [only])
            .unwrap()
            .pair
    };
    let pairs = [
        first_swap(crane(), context(), Adjacency, true).unwrap(),
        first_swap(crane(), context(), Cells, false).unwrap(),
        pick(TacoTortilla),
        pick(TacoTaco),
    ];
    for (pair, (want_pair, rules, adjacency, cyclic, tortilla, taco)) in pairs.iter().zip(expected)
    {
        assert_eq!(*pair, want_pair);
        let r = analysed(&swap(crane(), pair[0], pair[1]));
        assert!(!r.ok);
        assert_eq!(r.rejected_by, rules, "{pair:?}");
        assert_eq!(r.layers.adjacency.violations, adjacency, "{pair:?}");
        assert_eq!(r.layers.cells.cyclic.len(), cyclic, "{pair:?}");
        assert_eq!(r.layers.taco_tortilla.violations, tortilla, "{pair:?}");
        assert_eq!(r.layers.taco_taco.violations, taco, "{pair:?}");
        assert!(r.isometry.ok && r.crease_orientation.violations.is_empty());
    }
}

#[test]
fn the_swap_census_matches_the_reference() {
    let mut tally: BTreeMap<String, usize> = BTreeMap::new();
    for row in census() {
        let rules: Vec<String> = row.rejected_by.iter().map(|&r| name(r)).collect();
        let kind = if row.adjacent {
            "adjacent"
        } else {
            "non-adjacent"
        };
        *tally
            .entry(format!("{kind}: {}", rules.join(" + ")))
            .or_insert(0) += 1;
    }
    let expected = BTreeMap::from(
        [
            ("non-adjacent: cells + taco_tortilla + taco_taco", 381),
            ("non-adjacent: cells + taco_tortilla", 283),
            ("non-adjacent: cells + taco_taco", 61),
            ("adjacent: adjacency + cells + taco_tortilla", 31),
            ("adjacent: adjacency + taco_tortilla + taco_taco", 26),
            ("adjacent: adjacency + taco_taco", 17),
            (
                "adjacent: adjacency + cells + taco_tortilla + taco_taco",
                12,
            ),
            ("adjacent: adjacency + cells + taco_taco", 10),
            ("non-adjacent: taco_taco", 7),
            ("adjacent: adjacency + taco_tortilla", 6),
            ("non-adjacent: taco_tortilla + taco_taco", 2),
            ("non-adjacent: taco_tortilla", 2),
        ]
        .map(|(k, n)| (k.to_string(), n)),
    );
    assert_eq!(tally, expected);
}

#[test]
fn the_report_serializes_with_the_reference_keys() {
    let v = serde_json::to_value(report()).unwrap();
    assert_eq!(v["counts"]["faceOrders"], 838);
    assert_eq!(v["counts"]["assignment"]["M"], 61);
    assert_eq!(v["layers"]["taco_tortilla"]["checked"], 1049);
    assert_eq!(v["rejected_by"], serde_json::json!([]));
    let flipped = census().iter().find(|r| r.adjacent).unwrap();
    assert_eq!(
        serde_json::to_value(flipped).unwrap()["rejected_by"][0],
        "adjacency"
    );
}

// --- limits ---------------------------------------------------------------------------------

#[test]
fn a_fold_over_any_limit_is_refused_not_checked_in_part() {
    let none = Limits::NONE;
    let refused = |limits: Limits| match analyse_within(crane(), &limits) {
        Err(Unchecked::OverLimit(e)) => e.what,
        other => panic!("expected a refusal, got {other:?}"),
    };
    // The crane has 59 faces, 216 face-vertex incidences, 83 covered cells whose faces² sum to
    // 16580, and 1049 tortillas plus 196 tacos.
    assert_eq!(refused(Limits { faces: 58, ..none }), "faces");
    let incidences = Limits {
        face_vertices: 215,
        ..none
    };
    assert_eq!(refused(incidences), "face-vertex incidences");
    assert_eq!(refused(Limits { cells: 82, ..none }), "overlay cells");
    let pairs = Limits {
        cell_pairs: 16_579,
        ..none
    };
    assert_eq!(refused(pairs), "overlay cell face pairs");
    let tacos = Limits {
        tacos: 1_244,
        ..none
    };
    assert_eq!(refused(tacos), "tortillas and tacos");
    let at_its_size = Limits {
        faces: 59,
        face_vertices: 216,
        cell_pairs: 16_580,
        tacos: 1_245,
        ..none
    };
    assert_eq!(analyse_within(crane(), &at_its_size).as_ref(), Ok(report()));
}

#[test]
fn a_census_over_its_step_bound_is_refused_before_any_flip() {
    // 838 flips, each scanning 102 creases, 83 cells, 1049 tortillas and 196 tacos, plus at
    // most the 16580 cell face pairs.
    let steps = 838 * (102 + 83 + 1049 + 196 + 16_580);
    let at = |census_steps| Limits {
        census_steps,
        ..Limits::NONE
    };
    let full = swap_census_within(crane(), context(), &at(steps)).unwrap();
    assert_eq!(full, census());
    let err = swap_census_within(crane(), context(), &at(steps - 1)).unwrap_err();
    assert_eq!(err.what, "census steps");
}
