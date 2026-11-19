//! The `.cp` reader: one line per segment, `code x1 y1 x2 y2`.
//!
//! The code table is read from the two programs' own code, not from a description of it
//! (read on 2026-11-18):
//!
//! - ORIPA `LoaderCP` / `OriLine.Type`: 0 aux, 1 cut (the paper's edge), 2 mountain, 3 valley,
//!   4 unassigned, 9 cut model; any code other than 1, 2, 3 is loaded as aux. Its exporter
//!   writes 1, 2, 3 only and skips aux lines.
//! - Oriedita `fold.io.CreasePatternReader`: 1 border, 2 mountain, 3 valley, every other code
//!   (4 included) flat/aux.
//!
//! They agree on 1, 2, 3 and both load every other code as an auxiliary line.
//! No file from either program is committed (ORIPA is GPL-3.0, Oriedita declares no licence):
//! the inputs here are written by hand in the same layout as Oriedita's `square.cp` test file.

use ix_origami::cp::{read_cp, CpError, MAX_LINES};
use ix_origami::local::local_theorems;
use ix_origami::{Assignment, Fold};

const TOL: f64 = 1e-6;

/// A square, and four creases from its centre to its corners: one interior vertex.
fn square_with(creases: [u8; 4]) -> String {
    let c = creases;
    format!(
        "1 1 1 1 -1\n1 1 -1 -1 -1\n1 -1 -1 -1 1\n1 -1 1 1 1\n\
         {} 0 0 1 1\n{} 0 0 1 -1\n{} 0 0 -1 -1\n{} 0 0 -1 1\n",
        c[0], c[1], c[2], c[3]
    )
}

fn centre_of(fold: &Fold) -> usize {
    fold.vertices_coords
        .iter()
        .position(|p| p[0] == 0.0 && p[1] == 0.0)
        .expect("a vertex at the centre")
}

#[test]
fn codes_1_2_3_are_border_mountain_valley() {
    let r = read_cp(&square_with([2, 3, 2, 2]), TOL).unwrap();
    let f = &r.fold;
    assert_eq!(f.vertices_coords.len(), 5);
    assert_eq!(f.edges_vertices.len(), 8);
    let count = |a| f.edges_assignment.iter().filter(|&&x| x == a).count();
    assert_eq!(count(Assignment::B), 4);
    assert_eq!(count(Assignment::M), 3);
    assert_eq!(count(Assignment::V), 1);
    assert_eq!(r.skipped_aux, 0);
}

#[test]
fn every_other_code_is_an_auxiliary_line_left_out_and_counted() {
    let mut text = square_with([2, 3, 2, 2]);
    // 0 and 4: aux in both programs; 9: ORIPA's cut model; 7: unknown.
    text.push_str("0 0 0 0.5 0\n4 0 0 0 0.5\n9 0 0 -0.5 0\n7 0 0 0 -0.5\n");
    let r = read_cp(&text, TOL).unwrap();
    assert_eq!(r.skipped_aux, 4);
    assert_eq!(r.fold.edges_vertices.len(), 8);
    assert_eq!(r.fold.vertices_coords.len(), 5);
}

#[test]
fn the_local_theorems_run_on_a_crease_pattern_alone() {
    let r = read_cp(&square_with([2, 3, 2, 2]), TOL).unwrap();
    let t = local_theorems(&r.fold, 1e-9);
    assert_eq!(t.applicable, 1);
    assert!(t.kawasaki_failing.is_empty());
    assert!(t.maekawa_failing.is_empty());
    assert_eq!(t.maekawa_m_minus_v.get(&2), Some(&1));
}

#[test]
fn maekawa_fails_where_mountains_and_valleys_balance() {
    let r = read_cp(&square_with([2, 3, 2, 3]), TOL).unwrap();
    let t = local_theorems(&r.fold, 1e-9);
    let centre = centre_of(&r.fold);
    assert_eq!(t.maekawa_failing, vec![centre]);
    assert!(
        t.kawasaki_failing.is_empty(),
        "four right angles satisfy Kawasaki"
    );
}

#[test]
fn endpoints_within_the_tolerance_are_one_vertex() {
    let text = "1 0 0 1 0\n2 1.0000000001 0 1 1\n";
    assert_eq!(read_cp(text, 1e-6).unwrap().fold.vertices_coords.len(), 3);
    assert_eq!(read_cp(text, 1e-12).unwrap().fold.vertices_coords.len(), 4);
}

#[test]
fn blank_lines_tabs_and_crlf_are_read() {
    let text = "1 0 0 1 0\r\n\r\n\t2   1 0\t1 1\r\n";
    assert_eq!(read_cp(text, TOL).unwrap().fold.edges_vertices.len(), 2);
}

#[test]
fn a_malformed_line_is_refused_with_its_number() {
    for (text, line) in [
        ("2 0 0 1\n", 1),
        ("1 0 0 1 0\n2 0 0 1 1 9\n", 2),
        ("1 0 0 1 0\nx 0 0 1 1\n", 2),
        ("2.5 0 0 1 1\n", 1),
        ("2 0 0 NaN 1\n", 1),
        ("2 0 0 inf 1\n", 1),
        ("2 0 0 1e400 1\n", 1),
        ("-1 0 0 1 1\n1 0 0 1 0\n", 1),
    ] {
        match read_cp(text, TOL) {
            Err(CpError::Line { line: l, .. }) => assert_eq!(l, line, "{text:?}"),
            other => panic!("{text:?}: {other:?}"),
        }
    }
}

#[test]
fn a_bad_tolerance_is_refused() {
    for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            read_cp("1 0 0 1 0\n", tol),
            Err(CpError::Tolerance),
            "tol {tol}"
        );
    }
}

#[test]
fn a_file_with_no_crease_is_refused() {
    assert_eq!(read_cp("", TOL), Err(CpError::Empty));
    assert_eq!(read_cp("\n\n", TOL), Err(CpError::Empty));
    assert_eq!(read_cp("0 0 0 1 0\n4 0 0 0 1\n", TOL), Err(CpError::Empty));
}

#[test]
fn a_zero_length_segment_is_refused() {
    assert_eq!(
        read_cp("1 0 0 1 0\n2 0.5 0.5 0.5 0.5\n", TOL),
        Err(CpError::ZeroLength { line: 2 })
    );
    assert_eq!(
        read_cp("2 0 0 0.0000000001 0\n", TOL),
        Err(CpError::ZeroLength { line: 1 })
    );
}

#[test]
fn a_segment_given_twice_is_refused() {
    assert_eq!(
        read_cp("2 0 0 1 0\n3 1 0 0 0\n", TOL),
        Err(CpError::Duplicate { lines: [1, 2] })
    );
}

#[test]
fn segments_not_split_at_their_meeting_point_are_refused() {
    for text in [
        // a proper crossing
        "2 -1 0 1 0\n3 0 -1 0 1\n",
        // a T: the end of one lies inside the other
        "2 -1 0 1 0\n3 0 0 0 1\n",
        // an end a hair from the inside of the other
        "2 -1 0 1 0\n3 0 0.0000000001 0 1\n",
        // collinear overlap sharing no vertex
        "2 0 0 2 0\n3 1 0 3 0\n",
        // collinear overlap sharing one vertex
        "2 0 0 2 0\n3 0 0 1 0\n",
    ] {
        assert_eq!(
            read_cp(text, TOL),
            Err(CpError::NotSplit { lines: [1, 2] }),
            "{text:?}"
        );
    }
}

#[test]
fn segments_meeting_at_a_shared_vertex_or_apart_are_fine() {
    for text in [
        "2 0 0 1 0\n3 0 0 0 1\n",
        "2 0 0 1 0\n3 2 0 3 0\n",
        "2 0 0 1 0\n3 0 1 1 1\n",
    ] {
        assert!(read_cp(text, TOL).is_ok(), "{text:?}");
    }
}

#[test]
fn more_lines_than_the_cap_are_refused_before_any_pairing() {
    let text: String = (0..=MAX_LINES)
        .map(|i| format!("2 {i} 0 {i} 1\n"))
        .collect();
    assert_eq!(
        read_cp(&text, TOL),
        Err(CpError::TooMany { limit: MAX_LINES })
    );
}

#[test]
fn the_result_is_a_valid_fold_that_round_trips() {
    let r = read_cp(&square_with([2, 3, 2, 2]), TOL).unwrap();
    assert!(r.fold.validate().is_empty(), "{:?}", r.fold.validate());
    let back = Fold::from_value(&r.fold.to_value()).unwrap();
    assert_eq!(back, r.fold);
}

#[test]
fn the_folded_state_checks_refuse_a_crease_pattern_without_a_folded_frame() {
    let r = read_cp(&square_with([2, 3, 2, 2]), TOL).unwrap();
    let problems = r.fold.structure();
    assert!(
        problems.iter().any(|p| p.contains("folded frame")),
        "{problems:?}"
    );
}
