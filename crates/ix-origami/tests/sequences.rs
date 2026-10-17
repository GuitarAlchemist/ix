//! `.origami` files: the examples in `sequences/`, the 2026-10-16 prototype's results, and every
//! refusal with its line.

use std::collections::BTreeSet;

use ix_origami::sequence::{Expect, Sequence, GRAMMAR, MAX_FACES};
use ix_origami::{analyse, swap_census, Context};

const EXAMPLES: [(&str, &str); 5] = [
    ("half", include_str!("../sequences/half.origami")),
    ("quarters", include_str!("../sequences/quarters.origami")),
    (
        "triangle-twice",
        include_str!("../sequences/triangle-twice.origami"),
    ),
    ("kite", include_str!("../sequences/kite.origami")),
    ("letter", include_str!("../sequences/letter.origami")),
];

fn parse(text: &str) -> Sequence {
    text.parse().unwrap_or_else(|e| panic!("{e}\n{text}"))
}

#[test]
fn every_example_replays_passes_every_check_and_holds_its_expectations() {
    for (name, text) in EXAMPLES {
        let sequence = parse(text);
        assert_eq!(sequence.name, name);
        assert!(!sequence.expectations.is_empty(), "{name} expects nothing");
        let checked = sequence.check().unwrap();
        let report = checked.report.as_ref().expect("the replay has a structure");
        assert!(report.ok, "{name}: rejected by {:?}", report.rejected_by);
        for v in &checked.verdicts {
            assert!(
                v.holds,
                "{name} line {}: `{}` got {}",
                v.line, v.text, v.got
            );
        }
        // The replay goes through FOLD JSON and back, and the stated folded state it writes is
        // the one checked.
        assert_eq!(analyse(&checked.fold).unwrap(), *report);
    }
}

/// The seven sequences of the Python prototype (`simple_folds.py`, 2026-10-16), written as
/// `.origami` text: the faces and stated pairs it wrote, and its swap census.
#[test]
fn the_replay_reproduces_the_prototype() {
    let t = (22.5f64).to_radians().tan();
    let kite_then_more = format!(
        "origami kite-then-more
         valley through A (1, {t}) moving (1, 0)
         valley through A ({t}, 1) moving (0, 1)
         valley through (0.2, 1) (1, 0.2) moving C
         mountain through A C moving (1, 0)"
    );
    let cases: Vec<(String, usize, usize, usize)> = vec![
        // (text, faces, stated pairs, accepted single swaps)
        ("origami half\nvalley B to A".into(), 2, 1, 0),
        (
            "origami quarters\nvalley B to A\nvalley D to A".into(),
            4,
            6,
            0,
        ),
        (
            "origami half-then-mountain\nvalley B to A\nmountain D to A".into(),
            4,
            6,
            0,
        ),
        (
            "origami letter
             valley through (1/3, 0) (1/3, 1) moving A
             valley through (2/3, 0) (2/3, 1) moving B"
                .into(),
            3,
            3,
            1,
        ),
        (
            "origami triangle-twice\nvalley A to C\nvalley B to D".into(),
            4,
            6,
            0,
        ),
        (kite_then_more, 8, 28, 0),
        (
            "origami six-oblique
             valley through (0.5, 0) (0.55, 1) moving (1, 0.5)
             mountain through (0, 0.45) (1, 0.6) moving (0.2, 1)
             valley through (0.1, 0) (0.6, 0.7) moving (0, 0.3)
             valley through (0, 0.2) (0.6, 0.1) moving (0.3, 0)
             mountain through (0.3, 0) (0.35, 1) moving (0.5, 0.3)
             valley through (0, 0.35) (1, 0.3) moving (0.4, 0.6)"
                .into(),
            49,
            945,
            0,
        ),
    ];
    for (text, faces, pairs, accepted) in cases {
        let sequence = parse(&text);
        let fold = sequence.replay().unwrap();
        let report = analyse(&fold).unwrap();
        let name = &sequence.name;
        assert!(report.ok, "{name}: rejected by {:?}", report.rejected_by);
        assert_eq!(report.counts.faces, faces, "{name}");
        assert_eq!(report.counts.face_orders, pairs, "{name}");
        let census = swap_census(&fold, &Context::new(&fold).unwrap());
        assert_eq!(census.len(), pairs, "{name}");
        let kept = census.iter().filter(|r| r.rejected_by.is_empty()).count();
        assert_eq!(kept, accepted, "{name}: single swaps no rule rejects");
    }
}

#[test]
fn a_failed_expectation_is_answered_not_refused() {
    let checked = parse("origami half\nvalley B to A\nexpect faces 3\nexpect ply 2\nexpect ok")
        .check()
        .unwrap();
    let got: Vec<_> = checked
        .verdicts
        .iter()
        .map(|v| (v.line, v.got.as_str(), v.holds))
        .collect();
    assert_eq!(got, [(3, "2", false), (4, "2", true), (5, "ok", true)]);
    assert!(!checked.holds());
}

/// `(text, line, part of the reason)`.
const REFUSALS: &[(&str, usize, &str)] = &[
    // The first line that says anything must name the model.
    ("# a crane\n\nvalley A to C", 3, "starts with `origami"),
    ("", 0, "starts with `origami"),
    ("origami nothing\nexpect ok", 0, "folds at least once"),
    ("origami twice\norigami again", 2, "given twice"),
    (
        "origami late\nvalley A to C\nsheet square",
        3,
        "before the first fold",
    ),
    ("origami round\nsheet circle", 2, "sheet square"),
    ("origami 2fold\nvalley A to C", 1, "a name is one word"),
    ("origami x\nfold A to C", 2, "not a statement"),
    // A fold through two points names no side.
    ("origami x\nvalley through mAB mCD", 2, "moving"),
    (
        "origami x\nvalley A to C moving B",
        2,
        "`moving` goes with `through` only",
    ),
    ("origami x\nvalley A to AC", 2, "not both"),
    ("origami x\nvalley E to A", 2, "not both"),
    ("origami x\nvalley mAC to A", 2, "not both"),
    (
        "origami x\nvalley through E A moving B",
        2,
        "`E` is not a point",
    ),
    ("origami x\nvalley through (1, 2 (3, 4) moving A", 2, "`(`"),
    ("origami x\nexpect faces many", 2, "whole number"),
    ("origami x\nexpect flat", 2, "an expectation is"),
    // Points that coincide where the step names them: B lies on A once the sheet is halved.
    ("origami x\nvalley B to A\nvalley B to A", 3, "same place"),
    ("origami x\nvalley through A A moving B", 2, "same place"),
    // The moving point is on the fold line: the first kite fold lays B on the diagonal.
    (
        "origami x\nvalley AB to AC\nvalley through A C moving B",
        3,
        "on the fold line",
    ),
    (
        "origami x\nvalley through A C moving O",
        2,
        "on the fold line",
    ),
    // Antiparallel lines: no fold lands one direction on the other.
    ("origami x\nmountain AB to CD", 2, "write AB to DC"),
    (
        "origami x\nmountain mDA-mBC to C-D",
        2,
        "write mDA-mBC to D-C",
    ),
    // A line that crosses the fold line has no side.
    ("origami x\nvalley mDA-mBC to mAB-mCD", 2, "ambiguous"),
    ("origami x\nvalley AB to AB", 2, "the same line"),
    // The fold line misses the paper, or runs along its edge: nothing, or everything, moves.
    (
        "origami x\nvalley through (2, 0) (2, 1) moving (3, 0)",
        2,
        "does not cross the paper",
    ),
    (
        "origami x\nvalley through A B moving C",
        2,
        "does not cross the paper",
    ),
];

#[test]
fn every_refusal_names_its_line() {
    for &(text, line, reason) in REFUSALS {
        let err = text
            .parse::<Sequence>()
            .and_then(|s| s.check().map(|_| ()))
            .expect_err(text);
        assert_eq!(err.line, line, "{text}: {err}");
        assert!(err.reason.contains(reason), "{text}: {err}");
    }
}

#[test]
fn a_replay_over_the_face_cap_is_refused() {
    // Each fold halves the strip and doubles the faces: 2^10 = 1024.
    let mut text = String::from("origami strips\n");
    let mut x = 1.0f64;
    for _ in 0..10 {
        let half = x / 2.0;
        text.push_str(&format!(
            "valley through ({half}, 0) ({half}, 1) moving ({x}, 0)\n"
        ));
        x = half;
    }
    let err = parse(&text).replay().unwrap_err();
    assert_eq!(err.line, 11, "{err}");
    assert!(err.reason.contains(&MAX_FACES.to_string()), "{err}");
}

#[test]
fn a_file_over_the_size_cap_is_refused_before_it_is_read() {
    let text = format!("origami big\n{}", "# padding\n".repeat(10_000));
    assert_eq!(text.parse::<Sequence>().unwrap_err().line, 0);
}

/// The terminals of [`GRAMMAR`], and one file that uses every one of them and parses: the
/// grammar names what the parser reads, and the parser reads what the grammar names.
#[test]
fn the_grammar_and_the_parser_agree() {
    let mut terminals = BTreeSet::new();
    for (i, part) in GRAMMAR.split('"').enumerate() {
        if i % 2 == 1 {
            terminals.insert(part);
        }
    }
    let expected: BTreeSet<&str> = [
        "origami", "sheet", "square", "valley", "mountain", "to", "through", "moving", "expect",
        "ok", "faces", "ply", "m", "O", "A", "B", "C", "D", "(", ",", ")", "-", ".", "/", "#",
    ]
    .into_iter()
    .collect();
    assert_eq!(terminals, expected);

    let every_word = "# every terminal of the grammar
origami every-word
sheet square
valley A to C                                  # corners, O2
mountain mDA-mBC to (0, -1/2)-(1, -0.5)        # midpoints, lines with `-`, fractions, decimals
valley through O (0.5, 0.25) moving B          # the centre, O1
mountain AB to AD                              # two corners, O3
expect ok
expect faces 1
expect ply 1
";
    let sequence = parse(every_word);
    assert_eq!(sequence.steps.len(), 4);
    assert_eq!(
        sequence
            .expectations
            .iter()
            .map(|e| e.expect)
            .collect::<Vec<_>>(),
        [Expect::Ok, Expect::Faces(1), Expect::Ply(1)]
    );
}
