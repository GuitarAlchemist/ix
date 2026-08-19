use ix_evolution::pareto::{rank, Candidate, Objective, ParetoError};

#[test]
fn ranks_the_known_nondominated_front() {
    let candidates = [
        Candidate::new("balanced", [2.0, 2.0]),
        Candidate::new("light", [1.0, 4.0]),
        Candidate::new("stiff", [4.0, 1.0]),
        Candidate::new("dominated", [3.0, 3.0]),
    ];

    let archive = rank(&candidates, &[Objective::Minimize, Objective::Minimize])
        .expect("valid candidate table");

    assert_eq!(
        archive.front(0),
        Some(
            &[
                "balanced".to_owned(),
                "light".to_owned(),
                "stiff".to_owned(),
            ][..],
        ),
    );
    assert_eq!(archive.rank_of("dominated"), Some(1));
}

#[test]
fn result_is_independent_of_candidate_row_order() {
    let forward = [
        Candidate::new("a", [1.0, 4.0]),
        Candidate::new("b", [2.0, 2.0]),
        Candidate::new("c", [4.0, 1.0]),
        Candidate::new("d", [3.0, 3.0]),
    ];
    let reversed = [
        Candidate::new("d", [3.0, 3.0]),
        Candidate::new("c", [4.0, 1.0]),
        Candidate::new("b", [2.0, 2.0]),
        Candidate::new("a", [1.0, 4.0]),
    ];
    let objectives = [Objective::Minimize, Objective::Minimize];

    assert_eq!(
        rank(&forward, &objectives).expect("forward table"),
        rank(&reversed, &objectives).expect("reversed table"),
    );
}

#[test]
fn respects_mixed_objective_directions() {
    let candidates = [
        Candidate::new("efficient", [1.0, 8.0]),
        Candidate::new("powerful", [3.0, 10.0]),
        Candidate::new("worse", [2.0, 7.0]),
    ];

    let archive = rank(&candidates, &[Objective::Minimize, Objective::Maximize])
        .expect("mixed objective table");

    assert_eq!(archive.rank_of("efficient"), Some(0));
    assert_eq!(archive.rank_of("powerful"), Some(0));
    assert_eq!(archive.rank_of("worse"), Some(1));
}

#[test]
fn refuses_non_finite_objectives() {
    let candidates = [Candidate::new("unsafe", [1.0, f64::NAN])];

    assert_eq!(
        rank(&candidates, &[Objective::Minimize, Objective::Minimize]),
        Err(ParetoError::NonFiniteValue {
            id: "unsafe".to_owned(),
            objective: 1,
        }),
    );
}

#[test]
fn rejects_empty_inputs_and_malformed_candidates() {
    assert_eq!(
        rank(&[], &[Objective::Minimize]),
        Err(ParetoError::EmptyCandidates)
    );
    assert_eq!(
        rank(&[Candidate::new("a", [1.0])], &[]),
        Err(ParetoError::EmptyObjectives),
    );
    assert_eq!(
        rank(&[Candidate::new("  ", [1.0])], &[Objective::Minimize]),
        Err(ParetoError::EmptyCandidateId),
    );
    assert_eq!(
        rank(
            &[Candidate::new("a", [1.0]), Candidate::new("a", [2.0])],
            &[Objective::Minimize],
        ),
        Err(ParetoError::DuplicateCandidateId("a".to_owned())),
    );
    assert_eq!(
        rank(&[Candidate::new("a", [1.0, 2.0])], &[Objective::Minimize],),
        Err(ParetoError::ObjectiveCount {
            id: "a".to_owned(),
            expected: 1,
            actual: 2,
        }),
    );
}

#[test]
fn equal_vectors_are_co_nondominated() {
    let candidates = [
        Candidate::new("a", [1.0, 2.0]),
        Candidate::new("b", [1.0, 2.0]),
        Candidate::new("c", [2.0, 3.0]),
    ];

    let archive = rank(&candidates, &[Objective::Minimize, Objective::Minimize])
        .expect("equal vectors are valid");

    assert_eq!(
        archive.front(0),
        Some(&["a".to_owned(), "b".to_owned()][..])
    );
    assert_eq!(archive.front(1), Some(&["c".to_owned()][..]));
}

#[test]
fn signed_zeroes_are_numerically_equal() {
    let candidates = [
        Candidate::new("negative-zero", [-0.0]),
        Candidate::new("positive-zero", [0.0]),
    ];

    let archive = rank(&candidates, &[Objective::Minimize])
        .expect("signed zeroes are finite objective values");

    assert_eq!(
        archive.front(0),
        Some(&["negative-zero".to_owned(), "positive-zero".to_owned(),][..],),
    );
    assert_eq!(archive.front(1), None);
}

#[test]
fn computes_multiple_dominance_fronts() {
    let candidates = [
        Candidate::new("best", [1.0, 1.0]),
        Candidate::new("middle-a", [2.0, 2.0]),
        Candidate::new("middle-b", [1.0, 3.0]),
        Candidate::new("worst", [4.0, 4.0]),
    ];

    let archive = rank(&candidates, &[Objective::Minimize, Objective::Minimize])
        .expect("valid dominance chain");

    assert_eq!(archive.front(0), Some(&["best".to_owned()][..]));
    assert_eq!(
        archive.front(1),
        Some(&["middle-a".to_owned(), "middle-b".to_owned()][..]),
    );
    assert_eq!(archive.front(2), Some(&["worst".to_owned()][..]));
    assert_eq!(archive.front(3), None);
}

#[test]
fn refuses_positive_and_negative_infinity() {
    for value in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            rank(&[Candidate::new("unsafe", [value])], &[Objective::Minimize]),
            Err(ParetoError::NonFiniteValue {
                id: "unsafe".to_owned(),
                objective: 0,
            }),
        );
    }
}
