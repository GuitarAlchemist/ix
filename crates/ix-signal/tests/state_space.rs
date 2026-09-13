//! Integration tests for `ix_signal::state_space`.
//!
//! The oracle is exact: every `(A, B, C)` triple below has a rank that can be
//! read off by hand, and the cases that matter carry a *seeded* rank
//! deficiency — one that has been hidden behind a similarity transform, so a
//! rank test that merely looks for structural zeros fails them.

use ix_math::linalg::inverse;
use ix_signal::kalman::{constant_velocity_1d, KalmanFilter};
use ix_signal::state_space::{StateSpaceError, StateSpaceModel};
use ndarray::{array, Array1, Array2};

/// Discrete double integrator with `dt = 1`: `x = [position, velocity]`,
/// force input, position output. Controllable and observable in every
/// textbook.
///
/// Note `A = [[1, 1], [0, 1]]`, not the nilpotent `[[0, 1], [0, 0]]` — the
/// latter is the *continuous*-time double integrator and, read as a discrete
/// update, resets velocity to zero every step.
fn double_integrator() -> StateSpaceModel {
    StateSpaceModel::new(
        array![[1.0, 1.0], [0.0, 1.0]],
        array![[0.0], [1.0]],
        array![[1.0, 0.0]],
    )
    .expect("double integrator dimensions conform")
}

/// Modal system with three distinct eigenvalues where the third mode is
/// touched by neither `B` nor `C`. Rank of both tests is 2, not 3.
fn modal_with_dark_third_mode() -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let a = array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
    let b = array![[1.0], [1.0], [0.0]];
    let c = array![[1.0, 1.0, 0.0]];
    (a, b, c)
}

/// Apply the similarity transform `x -> T x`: `(A, B, C)` becomes
/// `(T A T^-1, T B, C T^-1)`. Controllability and observability ranks are
/// invariant under this, but the zeros that made the deficiency obvious are
/// destroyed.
fn similarity_transform(
    t: &Array2<f64>,
    a: &Array2<f64>,
    b: &Array2<f64>,
    c: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    let t_inv = inverse(t).expect("T is invertible");
    (t.dot(a).dot(&t_inv), t.dot(b), c.dot(&t_inv))
}

/// A fixed, non-orthogonal, well-conditioned change of basis. Hard-coded so
/// the test is deterministic; `det = -3`.
fn seed_transform() -> Array2<f64> {
    array![[1.0, 2.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 2.0]]
}

// ---------------------------------------------------------------------------
// Construction and dimension validation
// ---------------------------------------------------------------------------

#[test]
fn rejects_non_square_transition() {
    let err = StateSpaceModel::new(
        Array2::zeros((2, 3)),
        Array2::zeros((2, 1)),
        Array2::zeros((1, 2)),
    )
    .unwrap_err();
    assert_eq!(
        err,
        StateSpaceError::NonSquareTransition { rows: 2, cols: 3 }
    );
}

#[test]
fn rejects_empty_state() {
    let err = StateSpaceModel::new(
        Array2::zeros((0, 0)),
        Array2::zeros((0, 1)),
        Array2::zeros((1, 0)),
    )
    .unwrap_err();
    assert_eq!(err, StateSpaceError::EmptyState);
}

#[test]
fn rejects_input_matrix_with_wrong_row_count() {
    let err = StateSpaceModel::new(Array2::eye(2), Array2::zeros((3, 1)), Array2::zeros((1, 2)))
        .unwrap_err();
    assert_eq!(
        err,
        StateSpaceError::DimensionMismatch {
            what: "input matrix B row count",
            expected: 2,
            got: 3,
        }
    );
}

#[test]
fn rejects_output_matrix_with_wrong_column_count() {
    let err = StateSpaceModel::new(Array2::eye(2), Array2::zeros((2, 1)), Array2::zeros((1, 5)))
        .unwrap_err();
    assert_eq!(
        err,
        StateSpaceError::DimensionMismatch {
            what: "output matrix C column count",
            expected: 2,
            got: 5,
        }
    );
}

#[test]
fn reports_dimensions() {
    let m =
        StateSpaceModel::new(Array2::eye(4), Array2::zeros((4, 2)), Array2::zeros((3, 4))).unwrap();
    assert_eq!((m.state_dim(), m.input_dim(), m.output_dim()), (4, 2, 3));
}

// ---------------------------------------------------------------------------
// Dynamics: x_{k+1} = A x_k + B u_k + w_k
// ---------------------------------------------------------------------------

#[test]
fn step_applies_state_and_input_matrices() {
    let m = double_integrator();
    // x = [1, 2], u = [3]  =>  A x = [3, 2], B u = [0, 3]  =>  [3, 5].
    let next = m.step(&array![1.0, 2.0], &array![3.0]).unwrap();
    assert_eq!(next, array![3.0, 5.0]);
}

#[test]
fn output_applies_output_matrix() {
    let m = double_integrator();
    assert_eq!(m.output(&array![7.0, -1.0]).unwrap(), array![7.0]);
}

#[test]
fn step_rejects_wrong_input_length() {
    let m = double_integrator();
    let err = m.step(&array![1.0, 2.0], &array![1.0, 1.0]).unwrap_err();
    assert_eq!(
        err,
        StateSpaceError::DimensionMismatch {
            what: "input vector",
            expected: 1,
            got: 2,
        }
    );
}

#[test]
fn simulate_matches_hand_computed_trajectory() {
    let m = double_integrator();
    let inputs = vec![array![1.0], array![0.0], array![0.0]];
    let traj = m.simulate(&array![0.0, 0.0], &inputs, None).unwrap();

    // x0 = [0,0]; u=1 -> x1 = [0,1]; u=0 -> x2 = [1,1]; u=0 -> x3 = [2,1].
    assert_eq!(traj.len(), 4, "trajectory includes the initial state");
    assert_eq!(traj[0], array![0.0, 0.0]);
    assert_eq!(traj[1], array![0.0, 1.0]);
    assert_eq!(traj[2], array![1.0, 1.0]);
    assert_eq!(traj[3], array![2.0, 1.0]);
}

#[test]
fn simulate_adds_the_process_noise_term() {
    let m = double_integrator();
    let inputs = vec![array![0.0], array![0.0]];
    let noise = vec![array![0.5, 0.0], array![0.0, -0.25]];

    let clean = m.simulate(&array![1.0, 1.0], &inputs, None).unwrap();
    let noisy = m
        .simulate(&array![1.0, 1.0], &inputs, Some(&noise))
        .unwrap();

    // Clean: x1 = [2,1], x2 = [3,1].
    assert_eq!(clean[1], array![2.0, 1.0]);
    assert_eq!(clean[2], array![3.0, 1.0]);
    // Noisy: x1 = [2,1] + [0.5,0] = [2.5,1]; x2 = A x1 = [3.5,1] + [0,-0.25].
    assert_eq!(noisy[1], array![2.5, 1.0]);
    assert_eq!(noisy[2], array![3.5, 0.75]);
    assert_ne!(clean[2], noisy[2], "w_k must actually move the state");
}

#[test]
fn simulate_rejects_noise_sequence_of_the_wrong_length() {
    let m = double_integrator();
    let err = m
        .simulate(&array![0.0, 0.0], &[array![0.0]], Some(&[]))
        .unwrap_err();
    assert_eq!(
        err,
        StateSpaceError::DimensionMismatch {
            what: "process-noise sequence length",
            expected: 1,
            got: 0,
        }
    );
}

#[test]
fn simulate_rejects_noise_vector_of_the_wrong_width() {
    let m = double_integrator();
    let err = m
        .simulate(&array![0.0, 0.0], &[array![0.0]], Some(&[array![1.0]]))
        .unwrap_err();
    assert_eq!(
        err,
        StateSpaceError::DimensionMismatch {
            what: "process-noise vector",
            expected: 2,
            got: 1,
        }
    );
}

// ---------------------------------------------------------------------------
// A5: controllability
// ---------------------------------------------------------------------------

#[test]
fn controllability_matrix_has_the_textbook_layout() {
    let m = double_integrator();
    // [B, AB] with B = [0; 1], AB = [1; 1].
    assert_eq!(m.controllability_matrix(), array![[0.0, 1.0], [1.0, 1.0]]);
}

#[test]
fn double_integrator_is_controllable() {
    let r = double_integrator().controllability().unwrap();
    assert!(r.full_rank, "double integrator is controllable: {r:?}");
    assert_eq!(r.rank, 2);
    assert_eq!(r.deficiency, 0);
    assert!(r.margin > 0.5, "well-conditioned, margin = {}", r.margin);
}

#[test]
fn modal_system_with_a_dark_mode_is_uncontrollable() {
    let (a, b, c) = modal_with_dark_third_mode();
    let r = StateSpaceModel::new(a, b, c)
        .unwrap()
        .controllability()
        .unwrap();
    assert_eq!(r.rank, 2, "third mode is unreachable: {r:?}");
    assert_eq!(r.required_rank, 3);
    assert_eq!(r.deficiency, 1);
    assert!(!r.full_rank);
}

/// The load-bearing test: the same rank-2 system, but rotated into a basis
/// where no matrix entry is zero. Nothing about the deficiency is visible by
/// inspection, so only an actual rank computation can find it.
#[test]
fn seeded_rank_deficiency_survives_a_similarity_transform() {
    let (a, b, c) = modal_with_dark_third_mode();
    let (at, bt, ct) = similarity_transform(&seed_transform(), &a, &b, &c);

    assert!(
        at.iter().all(|v| v.abs() > 1e-12) || bt.iter().all(|v| v.abs() > 1e-12),
        "the transform must destroy the obvious structural zeros"
    );

    let m = StateSpaceModel::new(at, bt, ct).unwrap();

    let ctrl = m.controllability().unwrap();
    assert_eq!(ctrl.rank, 2, "rank is a similarity invariant: {ctrl:?}");
    assert_eq!(ctrl.deficiency, 1);
    assert!(!ctrl.full_rank);

    let obs = m.observability().unwrap();
    assert_eq!(obs.rank, 2, "rank is a similarity invariant: {obs:?}");
    assert_eq!(obs.deficiency, 1);
    assert!(!obs.full_rank);
}

#[test]
fn autonomous_system_is_uncontrollable() {
    let m = StateSpaceModel::new(Array2::eye(2), Array2::zeros((2, 0)), Array2::eye(2)).unwrap();
    let r = m.controllability().unwrap();
    assert_eq!(r.rank, 0);
    assert_eq!(r.deficiency, 2);
    assert!(!r.full_rank);
}

// ---------------------------------------------------------------------------
// A5: observability
// ---------------------------------------------------------------------------

#[test]
fn observability_matrix_has_the_textbook_layout() {
    let m = double_integrator();
    // [C; CA] with C = [1, 0], CA = [1, 1].
    assert_eq!(m.observability_matrix(), array![[1.0, 0.0], [1.0, 1.0]]);
}

#[test]
fn double_integrator_is_observable_from_position_alone() {
    let r = double_integrator().observability().unwrap();
    assert!(r.full_rank, "velocity is inferable from position: {r:?}");
    assert_eq!(r.rank, 2);
}

#[test]
fn modal_system_with_a_dark_mode_is_unobservable() {
    let (a, b, c) = modal_with_dark_third_mode();
    let r = StateSpaceModel::new(a, b, c)
        .unwrap()
        .observability()
        .unwrap();
    assert_eq!(r.rank, 2);
    assert_eq!(r.deficiency, 1);
    assert!(!r.full_rank);
}

#[test]
fn unmeasured_system_is_unobservable() {
    let m = StateSpaceModel::new(Array2::eye(2), Array2::eye(2), Array2::zeros((0, 2))).unwrap();
    let r = m.observability().unwrap();
    assert_eq!(r.rank, 0);
    assert_eq!(r.deficiency, 2);
}

// ---------------------------------------------------------------------------
// Structural properties that catch transposition and truncation bugs
// ---------------------------------------------------------------------------

/// Observability of `(A, C)` is controllability of `(A^T, C^T)`. A swapped
/// or mis-transposed block in either builder breaks this identity.
#[test]
fn observability_is_dual_to_controllability_of_the_transpose() {
    let (a, _b, c) = modal_with_dark_third_mode();
    let (at, bt, ct) = similarity_transform(&seed_transform(), &a, &Array2::eye(3), &c);
    let _ = bt;

    let primal = StateSpaceModel::new(at.clone(), Array2::eye(3), ct.clone()).unwrap();
    let dual = StateSpaceModel::new(at.t().to_owned(), ct.t().to_owned(), Array2::eye(3)).unwrap();

    let obs = primal.observability().unwrap();
    let ctrl = dual.controllability().unwrap();
    assert_eq!(obs.rank, ctrl.rank, "duality: {obs:?} vs {ctrl:?}");
    assert!(
        (obs.margin - ctrl.margin).abs() < 1e-9,
        "the dual matrices are transposes, so their singular values agree: {} vs {}",
        obs.margin,
        ctrl.margin
    );
}

/// Cayley-Hamilton: `A^n B` lies in the span of `[B, ..., A^(n-1) B]`, so
/// appending it must not raise the rank. This is what makes `n` blocks exact
/// rather than an arbitrary truncation.
#[test]
fn extra_powers_beyond_n_minus_one_add_no_rank() {
    let (a, b, c) = modal_with_dark_third_mode();
    let (at, bt, ct) = similarity_transform(&seed_transform(), &a, &b, &c);
    let m = StateSpaceModel::new(at.clone(), bt.clone(), ct).unwrap();

    let base = m.controllability_matrix();
    let n = m.state_dim();

    // Extend with A^n B and A^(n+1) B.
    let mut block = bt.clone();
    for _ in 0..n {
        block = at.dot(&block);
    }
    let extended = ndarray::concatenate(
        ndarray::Axis(1),
        &[base.view(), block.view(), at.dot(&block).view()],
    )
    .unwrap();

    let extended_rank = ix_math::svd::svd(&extended)
        .unwrap()
        .rank(1e-9 * extended.iter().fold(0.0f64, |m, v| m.max(v.abs())));

    assert_eq!(
        extended_rank,
        m.controllability().unwrap().rank,
        "Cayley-Hamilton: extra powers add nothing"
    );
}

// ---------------------------------------------------------------------------
// Margin: full rank is not the same as usefully full rank
// ---------------------------------------------------------------------------

#[test]
fn near_unobservable_system_is_full_rank_with_a_tiny_margin() {
    // Two almost-identical modes, observed only through their sum. Formally
    // observable; numerically a hair away from not being.
    let near = StateSpaceModel::new(
        array![[1.0, 0.0], [0.0, 1.0 + 1e-10]],
        array![[1.0], [1.0]],
        array![[1.0, 1.0]],
    )
    .unwrap()
    .observability()
    .unwrap();

    let well_separated = StateSpaceModel::new(
        array![[1.0, 0.0], [0.0, 3.0]],
        array![[1.0], [1.0]],
        array![[1.0, 1.0]],
    )
    .unwrap()
    .observability()
    .unwrap();

    assert!(near.full_rank, "still formally observable: {near:?}");
    assert!(well_separated.full_rank);
    assert!(
        near.margin < 1e-8,
        "near-degenerate margin should be tiny, got {}",
        near.margin
    );
    assert!(
        well_separated.margin > 0.1,
        "healthy margin should be O(1), got {}",
        well_separated.margin
    );
    assert!(
        near.margin < well_separated.margin / 1e6,
        "margin must separate the two cases by orders of magnitude"
    );
}

#[test]
fn reports_expose_the_singular_values_and_tolerance() {
    let r = double_integrator().observability().unwrap();
    assert_eq!(r.singular_values.len(), 2);
    assert!(
        r.singular_values.windows(2).all(|w| w[0] >= w[1]),
        "singular values are descending: {:?}",
        r.singular_values
    );
    assert!(r.tolerance > 0.0 && r.tolerance < 1e-12);
}

// ---------------------------------------------------------------------------
// Bridge to the existing Kalman filter
// ---------------------------------------------------------------------------

#[test]
fn from_kalman_extracts_the_plant_matrices() {
    let kf = constant_velocity_1d(0.1, 1.0, 0.5);
    let m = StateSpaceModel::from_kalman(&kf).unwrap();

    assert_eq!(m.a(), &kf.transition);
    assert_eq!(m.b(), &kf.control);
    assert_eq!(m.c(), &kf.observation);
    assert_eq!(m.state_dim(), 2);
    assert_eq!(m.output_dim(), 1);
}

/// `constant_velocity_1d` measures position only, yet velocity is recoverable
/// because it drives position one step later. The rank test says so without
/// anyone reasoning about it.
#[test]
fn constant_velocity_filter_is_observable() {
    let kf = constant_velocity_1d(0.1, 1.0, 0.5);
    let m = StateSpaceModel::from_kalman(&kf).unwrap();
    let r = m.observability().unwrap();
    assert!(r.full_rank, "velocity is observable through dt: {r:?}");
    assert_eq!(r.rank, 2);
    assert!(r.margin > 0.0);
}

/// A filter whose `control` matrix was never set has no input path at all.
/// `KalmanFilter::new` leaves it as an `n x 1` zero block, and the rank test
/// reports the consequence instead of letting it pass silently.
#[test]
fn kalman_filter_with_unset_control_matrix_is_uncontrollable() {
    let kf = constant_velocity_1d(0.1, 1.0, 0.5);
    assert!(
        kf.control.iter().all(|v| *v == 0.0),
        "precondition: constant_velocity_1d never sets B"
    );

    let r = StateSpaceModel::from_kalman(&kf)
        .unwrap()
        .controllability()
        .unwrap();
    assert_eq!(r.rank, 0);
    assert_eq!(r.deficiency, 2);
    assert!(!r.full_rank);
}

/// Giving that same filter a real input path makes it controllable — so the
/// previous test is measuring `B`, not an artefact of the bridge.
#[test]
fn kalman_filter_with_a_force_input_is_controllable() {
    let mut kf = constant_velocity_1d(0.1, 1.0, 0.5);
    kf.control = array![[0.0], [1.0]];

    let r = StateSpaceModel::from_kalman(&kf)
        .unwrap()
        .controllability()
        .unwrap();
    assert!(r.full_rank, "force input reaches both states: {r:?}");
    assert_eq!(r.rank, 2);
}

/// The extracted model must reproduce the filter's own noise-free prediction.
/// If `from_kalman` mapped a matrix to the wrong slot, these diverge.
#[test]
fn extracted_model_reproduces_the_filters_noise_free_prediction() {
    let mut kf = KalmanFilter::new(2, 1);
    kf.transition = array![[1.0, 0.25], [0.0, 1.0]];
    kf.control = array![[0.0], [2.0]];
    kf.observation = array![[1.0, 0.0]];
    kf.state = array![3.0, -1.0];

    let m = StateSpaceModel::from_kalman(&kf).unwrap();
    let u: Array1<f64> = array![0.5];
    let expected = m.step(&kf.state, &u).unwrap();

    kf.predict(Some(&u));
    assert_eq!(kf.state, expected);
}
