//! `assess`: one call that runs the checks and the optional readings, and applies the verdict
//! rules of ADR-0007 itself: a stack is reported beside the checks and never changes their
//! verdict, and `rigid_ok` is the top-level `ok` and not a refused stack.

use ix_origami::{assess, AssessError, Fold, Limits, Options, Unchecked};
use serde_json::json;

/// The right half of a 2 x 1 sheet folds over the left along a valley at x = 1; `order` is the
/// stated sign of the one face pair: +1 agrees with the valley, -1 contradicts it.
fn half_fold(order: i64) -> Fold {
    half_fold_scaled(order, 1.0)
}

/// [`half_fold`] with every coordinate multiplied by `scale`.
fn half_fold_scaled(order: i64, scale: f64) -> Fold {
    let vertices = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
    .map(|[x, y]| [x * scale, y * scale]);
    let folded = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
    .map(|[x, y]| [x * scale, y * scale]);
    let mut edges: Vec<([usize; 2], &str)> = (0..6).map(|k| ([k, (k + 1) % 6], "B")).collect();
    edges.push(([1, 4], "V"));
    Fold::from_value(&json!({
        "file_spec": 1.2,
        "vertices_coords": vertices,
        "edges_vertices": edges.iter().map(|(e, _)| e).collect::<Vec<_>>(),
        "edges_assignment": edges.iter().map(|(_, a)| a).collect::<Vec<_>>(),
        "faces_vertices": [[0, 1, 4, 5], [1, 2, 3, 4]],
        "file_frames": [{
            "frame_classes": ["foldedForm"],
            "frame_parent": 0,
            "frame_inherit": true,
            "vertices_coords": folded,
            "faceOrders": [[0, 1, order]]
        }]
    }))
    .unwrap()
}

fn with_thickness(t: f64) -> Options {
    Options {
        thickness: Some(t),
        census: false,
    }
}

#[test]
fn a_fold_the_checks_accept_is_rigid_ok_when_its_stack_is_not_refused() {
    let a = assess(&half_fold(1), &Limits::NONE, &with_thickness(0.5)).unwrap();
    assert!(a.report.ok);
    assert_eq!(a.rigid_ok(), Some(true));
}

#[test]
fn a_fold_the_checks_reject_is_never_rigid_ok() {
    let a = assess(&half_fold(-1), &Limits::NONE, &with_thickness(0.5)).unwrap();
    assert!(!a.report.ok);
    let stack = a.stack.as_ref().expect("a thickness was asked for");
    assert!(!stack.rigid.refused, "the stack alone would pass");
    assert_eq!(a.rigid_ok(), Some(false));
}

#[test]
fn a_thickness_never_changes_the_top_level_verdict() {
    for order in [1, -1] {
        let with = assess(&half_fold(order), &Limits::NONE, &with_thickness(0.5)).unwrap();
        let without = assess(&half_fold(order), &Limits::NONE, &Options::default()).unwrap();
        assert_eq!(with.report, without.report);
    }
}

#[test]
fn nothing_is_read_that_was_not_asked_for() {
    let a = assess(&half_fold(1), &Limits::NONE, &Options::default()).unwrap();
    assert!(a.stack.is_none() && a.census.is_none());
    assert_eq!(a.rigid_ok(), None);
}

#[test]
fn a_census_is_read_on_request() {
    let options = Options {
        thickness: None,
        census: true,
    };
    let a = assess(&half_fold(1), &Limits::NONE, &options).unwrap();
    assert_eq!(a.census.expect("a census was asked for").len(), 1);
}

#[test]
fn a_thickness_that_is_not_finite_and_non_negative_is_refused() {
    for t in [-1.0, f64::NAN, f64::INFINITY] {
        let e = assess(&half_fold(1), &Limits::NONE, &with_thickness(t)).unwrap_err();
        assert!(matches!(e, AssessError::Thickness(_)), "t = {t}: {e:?}");
    }
}

#[test]
fn a_fold_whose_structure_stops_the_checks_is_unchecked_not_assessed() {
    let mut value = half_fold(1).to_value();
    value["faces_vertices"] = json!([[0, 1, 4, 99], [1, 2, 3, 4]]);
    let broken = Fold::from_value(&value).unwrap();
    let e = assess(&broken, &Limits::NONE, &with_thickness(0.5)).unwrap_err();
    assert!(
        matches!(e, AssessError::Unchecked(Unchecked::Structure(_))),
        "{e:?}"
    );
}

#[test]
fn the_verdict_is_the_same_at_any_scale() {
    // The tolerances are absolute, so the checks rescale first: a sheet 2^-30 or 2^30 times
    // larger is judged as the sheet is.
    for order in [1, -1] {
        let at_one = assess(&half_fold(order), &Limits::NONE, &Options::default()).unwrap();
        for k in [-30, -10, 10, 30] {
            let scaled = half_fold_scaled(order, 2f64.powi(k));
            let a = assess(&scaled, &Limits::NONE, &Options::default()).unwrap();
            assert_eq!(
                (a.report.ok, &a.report.rejected_by),
                (at_one.report.ok, &at_one.report.rejected_by),
                "order {order}, scale 2^{k}"
            );
        }
    }
}
