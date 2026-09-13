//! Tests for the long-form Pareto frontier pipeline (issue #294).
//!
//! Every test here is written so that it *can* fail. In particular the
//! order-independence tests permute a table that is deliberately not sorted and
//! not grouped, and the mechanism-revert control asserts that a weakened
//! dominance rule gives a *different* answer on the same fixture — a fixture
//! that could not tell the two rules apart would prove nothing.

use ix_evolution::frontier::{frontier, to_csv, FrontierError, ObjectiveRow};

const REV: &str = "rev-a";

fn row(
    task_class: &str,
    candidate: &str,
    metric: &str,
    direction: &str,
    value: f64,
) -> ObjectiveRow {
    ObjectiveRow::new(REV, task_class, candidate, metric, direction, value)
}

fn ids(rows: &[ObjectiveRow]) -> Vec<String> {
    frontier(rows)
        .expect("valid objective table")
        .into_iter()
        .map(|found| format!("{}/{}", found.task_class, found.candidate_id))
        .collect()
}

// --- the frontier itself -----------------------------------------------------

#[test]
fn minimization_keeps_the_cheapest_tradeoffs_and_drops_the_worse_on_both() {
    let rows = [
        row("t", "cheap", "cost", "MIN", 1.0),
        row("t", "cheap", "latency", "MIN", 9.0),
        row("t", "fast", "cost", "MIN", 9.0),
        row("t", "fast", "latency", "MIN", 1.0),
        row("t", "loser", "cost", "MIN", 9.0),
        row("t", "loser", "latency", "MIN", 9.0),
    ];

    assert_eq!(ids(&rows), vec!["t/cheap".to_owned(), "t/fast".to_owned()]);
}

#[test]
fn maximization_keeps_the_highest_tradeoffs_and_drops_the_worse_on_both() {
    let rows = [
        row("t", "accurate", "accuracy", "MAX", 0.9),
        row("t", "accurate", "recall", "MAX", 0.1),
        row("t", "broad", "accuracy", "MAX", 0.1),
        row("t", "broad", "recall", "MAX", 0.9),
        row("t", "loser", "accuracy", "MAX", 0.1),
        row("t", "loser", "recall", "MAX", 0.1),
    ];

    assert_eq!(
        ids(&rows),
        vec!["t/accurate".to_owned(), "t/broad".to_owned()]
    );
}

#[test]
fn a_mixed_direction_table_reads_each_metric_the_way_it_declares() {
    // `thrifty` is cheaper; `sharp` is better. Neither dominates. `waste` costs
    // more than `thrifty` AND scores worse than `sharp`, and is dominated by
    // both — but ONLY if `cost` is read as MIN while `quality` is read as MAX.
    // Flip either direction and `waste` survives, so this test pins the
    // per-metric direction, not just "some ordering happened".
    let rows = [
        row("t", "thrifty", "cost", "MIN", 1.0),
        row("t", "thrifty", "quality", "MAX", 0.5),
        row("t", "sharp", "cost", "MIN", 5.0),
        row("t", "sharp", "quality", "MAX", 0.9),
        row("t", "waste", "cost", "MIN", 5.0),
        row("t", "waste", "quality", "MAX", 0.4),
    ];

    assert_eq!(
        ids(&rows),
        vec!["t/sharp".to_owned(), "t/thrifty".to_owned()]
    );
}

#[test]
fn identical_objective_vectors_both_stay_on_the_frontier() {
    // The tie rule: equal vectors are mutually NON-dominated because neither is
    // strictly better anywhere. Both survive; `candidate_id` separates them.
    // This is the test that breaks if strict dominance is ever weakened to
    // "no worse", which would make each twin eliminate the other.
    let rows = [
        row("t", "twin_b", "cost", "MIN", 2.0),
        row("t", "twin_b", "quality", "MAX", 0.7),
        row("t", "twin_a", "cost", "MIN", 2.0),
        row("t", "twin_a", "quality", "MAX", 0.7),
        row("t", "loser", "cost", "MIN", 3.0),
        row("t", "loser", "quality", "MAX", 0.6),
    ];

    assert_eq!(
        ids(&rows),
        vec!["t/twin_a".to_owned(), "t/twin_b".to_owned()]
    );
}

#[test]
fn incomparable_candidates_are_all_retained() {
    let rows = [
        row("t", "a", "x", "MIN", 1.0),
        row("t", "a", "y", "MIN", 3.0),
        row("t", "b", "x", "MIN", 2.0),
        row("t", "b", "y", "MIN", 2.0),
        row("t", "c", "x", "MIN", 3.0),
        row("t", "c", "y", "MIN", 1.0),
    ];

    assert_eq!(
        ids(&rows),
        vec!["t/a".to_owned(), "t/b".to_owned(), "t/c".to_owned()]
    );
}

#[test]
fn task_classes_are_isolated() {
    // `shared` is ON the `alpha` frontier and OFF the `beta` frontier in the
    // same revision. If grouping leaked across task classes, `shared` would be
    // compared against `beta` candidates on metrics `beta` does not even
    // declare, and this exact pair of memberships could not both hold.
    let rows = [
        row("alpha", "shared", "cost", "MIN", 1.0),
        row("alpha", "other", "cost", "MIN", 5.0),
        row("beta", "shared", "accuracy", "MAX", 0.1),
        row("beta", "rival", "accuracy", "MAX", 0.9),
    ];

    assert_eq!(
        ids(&rows),
        vec!["alpha/shared".to_owned(), "beta/rival".to_owned()]
    );
}

#[test]
fn revisions_are_isolated() {
    // Same task class, same candidate ids, different revisions. `slow` loses in
    // its own revision; `slow` in the other revision is the only candidate and
    // survives. A pipeline that ignored `subject_revision` would instead see a
    // duplicate (class, candidate, metric) and fail closed, so this test
    // distinguishes "grouped by revision" from "revision ignored".
    let rows = [
        ObjectiveRow::new("rev-1", "t", "slow", "cost", "MIN", 9.0),
        ObjectiveRow::new("rev-1", "t", "quick", "cost", "MIN", 1.0),
        ObjectiveRow::new("rev-2", "t", "slow", "cost", "MIN", 9.0),
    ];

    let found = frontier(&rows).expect("valid objective table");
    let labelled: Vec<String> = found
        .iter()
        .map(|item| format!("{}/{}", item.subject_revision, item.candidate_id))
        .collect();
    assert_eq!(
        labelled,
        vec!["rev-1/quick".to_owned(), "rev-2/slow".to_owned()]
    );
    // The output echoes the exact input revision, not a normalised one.
    assert_eq!(found[0].subject_revision, "rev-1");
    assert_eq!(found[1].subject_revision, "rev-2");
}

#[test]
fn objectives_are_emitted_in_ascending_metric_order_whatever_order_they_arrived_in() {
    let rows = [
        row("t", "only", "zeta", "MIN", 1.0),
        row("t", "only", "alpha", "MAX", 2.0),
        row("t", "only", "mid", "MIN", 3.0),
    ];

    let found = frontier(&rows).expect("valid objective table");
    let metrics: Vec<&str> = found[0]
        .objectives
        .iter()
        .map(|objective| objective.metric.as_str())
        .collect();
    assert_eq!(metrics, vec!["alpha", "mid", "zeta"]);
}

// --- determinism -------------------------------------------------------------

/// A table whose row order carries no useful signal: revisions interleaved,
/// task classes interleaved, metrics out of order, dominated rows in the middle.
fn scrambled_table() -> Vec<ObjectiveRow> {
    vec![
        ObjectiveRow::new("rev-b", "t", "beta", "quality", "MAX", 0.9),
        ObjectiveRow::new("rev-a", "t", "twin_b", "cost", "MIN", 2.0),
        ObjectiveRow::new("rev-a", "u", "solo", "score", "MAX", 0.4),
        ObjectiveRow::new("rev-a", "t", "loser", "quality", "MAX", 0.1),
        ObjectiveRow::new("rev-b", "t", "beta", "cost", "MIN", 4.0),
        ObjectiveRow::new("rev-a", "t", "twin_a", "quality", "MAX", 0.7),
        ObjectiveRow::new("rev-a", "t", "loser", "cost", "MIN", 8.0),
        ObjectiveRow::new("rev-a", "t", "twin_b", "quality", "MAX", 0.7),
        ObjectiveRow::new("rev-a", "t", "cheap", "cost", "MIN", 1.0),
        ObjectiveRow::new("rev-a", "t", "twin_a", "cost", "MIN", 2.0),
        ObjectiveRow::new("rev-a", "t", "cheap", "quality", "MAX", 0.2),
        ObjectiveRow::new("rev-b", "t", "alpha", "cost", "MIN", 1.0),
        ObjectiveRow::new("rev-b", "t", "alpha", "quality", "MAX", 0.3),
    ]
}

/// Deterministic Fisher-Yates using a fixed linear congruential generator, so
/// the permutations exercised are the same on every machine and every run. A
/// randomly seeded shuffle would make a failure unreproducible.
fn shuffled(rows: &[ObjectiveRow], seed: u64) -> Vec<ObjectiveRow> {
    let mut out = rows.to_vec();
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    for index in (1..out.len()).rev() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let pick = (state >> 33) as usize % (index + 1);
        out.swap(index, pick);
    }
    out
}

#[test]
fn output_is_byte_identical_under_reversal_rotation_and_two_hundred_shuffles() {
    let rows = scrambled_table();
    let expected = to_csv(&frontier(&rows).expect("valid objective table"));

    // The fixture has to actually contain a frontier, otherwise "identical
    // under permutation" would be a statement about the empty string.
    assert!(
        expected.lines().count() > 3,
        "fixture must produce several frontier rows, got:\n{expected}"
    );

    let mut reversed = rows.clone();
    reversed.reverse();
    assert_eq!(
        to_csv(&frontier(&reversed).expect("valid objective table")),
        expected,
        "reversing the input changed the output"
    );

    for offset in 1..rows.len() {
        let mut rotated = rows.clone();
        rotated.rotate_left(offset);
        assert_eq!(
            to_csv(&frontier(&rotated).expect("valid objective table")),
            expected,
            "rotating the input by {offset} changed the output"
        );
    }

    for seed in 0..200u64 {
        let permuted = shuffled(&rows, seed);
        assert_eq!(
            to_csv(&frontier(&permuted).expect("valid objective table")),
            expected,
            "shuffle seed {seed} changed the output"
        );
    }
}

#[test]
fn repeated_runs_on_the_same_input_are_byte_identical() {
    let rows = scrambled_table();
    let first = to_csv(&frontier(&rows).expect("valid objective table"));
    for _ in 0..16 {
        assert_eq!(
            to_csv(&frontier(&rows).expect("valid objective table")),
            first
        );
    }
}

#[test]
fn the_shuffle_helper_actually_shuffles() {
    // Guards the test above: if `shuffled` were the identity, the permutation
    // loop would assert nothing at all.
    let rows = scrambled_table();
    let distinct = (0..200u64)
        .filter(|seed| shuffled(&rows, *seed) != rows)
        .count();
    assert!(
        distinct > 190,
        "expected nearly every seed to permute the table, got {distinct}/200"
    );
}

// --- mechanism-revert control ------------------------------------------------

/// The frontier as it would be if strict dominance were reverted to mere
/// "no worse on every objective". This is NOT a second implementation to trust;
/// it exists only so a test can assert that the real rule and the broken rule
/// disagree on the fixture.
fn weak_frontier_ids(rows: &[ObjectiveRow]) -> Vec<String> {
    use std::collections::BTreeMap;

    type Key<'a> = (&'a str, &'a str, &'a str);
    type Objectives<'a> = BTreeMap<&'a str, (&'a str, f64)>;

    let mut wide: BTreeMap<Key, Objectives> = BTreeMap::new();
    for entry in rows {
        wide.entry((
            &entry.subject_revision,
            &entry.task_class,
            &entry.candidate_id,
        ))
        .or_default()
        .insert(&entry.metric, (&entry.direction, entry.value));
    }

    let keys: Vec<_> = wide.keys().copied().collect();
    let mut survivors = Vec::new();
    for &key in &keys {
        let dominated = keys.iter().any(|&other| {
            other != key
                && other.0 == key.0
                && other.1 == key.1
                // The mechanism under test is deleted here: "no worse" only,
                // with no requirement of being strictly better anywhere.
                && wide[&other].iter().all(|(metric, (direction, value))| {
                    let mine = wide[&key][metric].1;
                    if *direction == "MIN" {
                        *value <= mine
                    } else {
                        *value >= mine
                    }
                })
        });
        if !dominated {
            survivors.push(format!("{}/{}", key.1, key.2));
        }
    }
    survivors
}

#[test]
fn mechanism_revert_control_weakened_dominance_gives_a_different_answer() {
    // If someone deletes the strict-improvement requirement from the real
    // implementation, `frontier` starts agreeing with `weak_frontier_ids` and
    // this assertion fires. That is the point: it is a control that fails when
    // the mechanism is removed, not a restatement of the expected output.
    let rows = [
        row("t", "twin_a", "cost", "MIN", 2.0),
        row("t", "twin_a", "quality", "MAX", 0.7),
        row("t", "twin_b", "cost", "MIN", 2.0),
        row("t", "twin_b", "quality", "MAX", 0.7),
        row("t", "loser", "cost", "MIN", 3.0),
        row("t", "loser", "quality", "MAX", 0.6),
    ];

    let real = ids(&rows);
    let weak = weak_frontier_ids(&rows);

    assert_eq!(
        real,
        vec!["t/twin_a".to_owned(), "t/twin_b".to_owned()],
        "strict dominance must keep both twins"
    );
    assert!(
        weak.is_empty(),
        "weakened dominance should annihilate the twins and everything they \
         dominate, got {weak:?}"
    );
    assert_ne!(
        real, weak,
        "the fixture does not discriminate between strict and weak dominance, \
         so no test built on it could detect the mechanism being removed"
    );
}

// --- fail-closed validation --------------------------------------------------

#[test]
fn an_empty_table_is_rejected() {
    assert_eq!(frontier(&[]), Err(FrontierError::EmptyInput));
}

#[test]
fn an_empty_identifier_is_rejected() {
    let rows = [row("t", "  ", "cost", "MIN", 1.0)];
    assert_eq!(
        frontier(&rows),
        Err(FrontierError::EmptyField {
            field: "candidate_id"
        })
    );
}

#[test]
fn an_unknown_direction_is_rejected_rather_than_guessed() {
    for spelling in ["min", "MINIMIZE", "", "ASC", "MAXX"] {
        let rows = [row("t", "a", "cost", spelling, 1.0)];
        assert!(
            matches!(
                frontier(&rows),
                Err(FrontierError::UnknownDirection { .. }) | Err(FrontierError::EmptyField { .. })
            ),
            "direction {spelling:?} should have been rejected"
        );
    }
}

#[test]
fn a_duplicated_metric_is_rejected() {
    // This is the rule that makes (revision, class, candidate) a key, which is
    // what makes the output order total. Without it two rows could collide.
    let rows = [
        row("t", "a", "cost", "MIN", 1.0),
        row("t", "a", "cost", "MIN", 2.0),
    ];
    assert_eq!(
        frontier(&rows),
        Err(FrontierError::DuplicateMetric {
            subject_revision: REV.to_owned(),
            task_class: "t".to_owned(),
            candidate_id: "a".to_owned(),
            metric: "cost".to_owned(),
        })
    );
}

#[test]
fn a_duplicated_metric_is_rejected_even_when_the_values_agree() {
    let rows = [
        row("t", "a", "cost", "MIN", 1.0),
        row("t", "a", "cost", "MIN", 1.0),
    ];
    assert!(matches!(
        frontier(&rows),
        Err(FrontierError::DuplicateMetric { .. })
    ));
}

#[test]
fn one_metric_declaring_two_directions_is_rejected() {
    let rows = [
        row("t", "a", "cost", "MIN", 1.0),
        row("t", "b", "cost", "MAX", 2.0),
    ];
    assert_eq!(
        frontier(&rows),
        Err(FrontierError::MixedDirection {
            subject_revision: REV.to_owned(),
            task_class: "t".to_owned(),
            metric: "cost".to_owned(),
            first: "MIN".to_owned(),
            second: "MAX".to_owned(),
        })
    );
}

#[test]
fn the_same_metric_may_carry_different_directions_in_different_task_classes() {
    // Mixed-direction is scoped to (revision, task class, metric). Two task
    // classes are independent contracts, so this must NOT be rejected.
    let rows = [
        row("alpha", "a", "score", "MIN", 1.0),
        row("beta", "b", "score", "MAX", 2.0),
    ];
    assert!(frontier(&rows).is_ok());
}

#[test]
fn a_candidate_missing_one_of_its_task_classes_metrics_is_rejected() {
    let rows = [
        row("t", "complete", "cost", "MIN", 1.0),
        row("t", "complete", "quality", "MAX", 0.5),
        row("t", "partial", "cost", "MIN", 2.0),
    ];
    assert_eq!(
        frontier(&rows),
        Err(FrontierError::MissingMetric {
            subject_revision: REV.to_owned(),
            task_class: "t".to_owned(),
            candidate_id: "partial".to_owned(),
            metric: "quality".to_owned(),
        })
    );
}

#[test]
fn non_finite_values_are_rejected() {
    for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let rows = [row("t", "a", "cost", "MIN", bad)];
        assert_eq!(
            frontier(&rows),
            Err(FrontierError::NonFiniteValue {
                candidate_id: "a".to_owned(),
                metric: "cost".to_owned(),
            }),
            "value {bad} should have been rejected"
        );
    }
}

#[test]
fn the_reported_violation_does_not_depend_on_input_order() {
    // A table with several defects at once. Whichever one the fixed check
    // sequence reports, it must report the SAME one for every permutation —
    // otherwise "fails closed" would be order-dependent even if the success
    // path is not.
    let rows = vec![
        row("t", "a", "cost", "MIN", f64::NAN),
        row("t", "a", "cost", "MIN", 1.0),
        row("t", "b", "cost", "MAX", 2.0),
        row("t", "b", "quality", "MAX", 2.0),
    ];
    let expected = frontier(&rows).expect_err("table has several defects");

    let mut reversed = rows.clone();
    reversed.reverse();
    assert_eq!(frontier(&reversed).expect_err("still invalid"), expected);

    for seed in 0..100u64 {
        assert_eq!(
            frontier(&shuffled(&rows, seed)).expect_err("still invalid"),
            expected,
            "shuffle seed {seed} changed which violation is reported"
        );
    }
}

// --- serialisation -----------------------------------------------------------

#[test]
fn csv_rendering_is_lf_only_and_fixed_precision() {
    let rows = [
        row("t", "a", "cost", "MIN", 0.1),
        row("t", "a", "latency", "MIN", 400.0),
    ];
    let csv = to_csv(&frontier(&rows).expect("valid objective table"));

    assert_eq!(
        csv,
        "subject_revision,task_class,candidate_id,objectives\n\
         rev-a,t,a,cost:MIN=0.100000;latency:MIN=400.000000\n"
    );
    assert!(
        !csv.contains('\r'),
        "a CR would break byte-identical output"
    );
}
