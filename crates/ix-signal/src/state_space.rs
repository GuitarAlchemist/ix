//! Discrete-time linear state-space models and structural analysis.
//!
//! ```text
//! x_{k+1} = A x_k + B u_k + w_k     (state equation)
//! y_k     = C x_k                   (output equation)
//! ```
//!
//! `A` (n*n) is the state-transition matrix, `B` (n*m) the input matrix,
//! `C` (p*n) the output matrix, and `w_k` an additive process-noise sequence
//! supplied by the caller.
//!
//! ## Not to be confused with `ix_graph::state_space`
//!
//! `ix-graph` also has a `state_space` module. It is a *discrete search
//! space* — `trait State`, `astar`, `beam_search` — and has nothing to do
//! with linear time-invariant systems. This module is the control-theory
//! meaning of the term.
//!
//! ## Relationship to [`crate::kalman::KalmanFilter`]
//!
//! A Kalman filter *is* a linear state-space model with an estimator
//! attached: it already carries `A`, `B`, `C`, `Q` and `R` internally. What
//! it does not do is let you hand those matrices to another algorithm
//! without also dragging along the estimator's mutable `state` and
//! `covariance`. [`StateSpaceModel::from_kalman`] performs that extraction,
//! so the plant description becomes reusable by the structural analysis
//! below, and later by control synthesis.
//!
//! The filter also never *simulates* the plant. [`KalmanFilter::predict`]
//! propagates the state *estimate* and its covariance; `Q` is used as an
//! uncertainty, never sampled. [`StateSpaceModel::simulate`] runs the plant
//! itself, with the `w_k` term explicit.
//!
//! ## Structural analysis
//!
//! - **Controllability**: can the input `u` steer the state anywhere?
//!   `rank([B, AB, ..., A^(n-1) B]) == n`.
//! - **Observability**: can the output `y` reveal the whole state?
//!   `rank([C; CA; ...; C A^(n-1)]) == n`.
//!
//! Both are Kalman rank tests, evaluated numerically through the singular
//! values of the respective matrix. Cayley-Hamilton guarantees that powers
//! beyond `A^(n-1)` add nothing, so `n` blocks is exact rather than a
//! truncation.
//!
//! ## Why an agent loop should care
//!
//! Treat an agent loop as a plant: `x` is the loop's internal state (context
//! occupancy, retry depth, tool-error backlog, budget burned), `u` the
//! levers a supervisor can pull, `y` the telemetry actually recorded.
//!
//! - An **unobservable** mode is a part of the loop's state that no amount
//!   of recorded telemetry can pin down. A loop whose state is not
//!   observable cannot be diagnosed, and [`StateSpaceModel::observability`]
//!   says so mechanically instead of heuristically: `deficiency` counts how
//!   many modes are dark, and `margin` says how close to dark the rest are.
//!   That is a guardrail on the metric set itself — it fails when someone
//!   proposes a dashboard that cannot see the thing it claims to measure.
//! - An **uncontrollable** mode is a part of the loop's state that no
//!   supervisor lever can move. Governance that gates on such a mode is
//!   green-but-dead by construction.
//!
//! Both are structural: they depend only on `(A, B, C)`, not on the data,
//! so they can be checked before a single trace is collected.
//!
//! ## Numerical caveat
//!
//! The rank tests are formed from powers of `A`. For an `A` with large
//! spectral radius those powers grow geometrically and the test matrix
//! becomes badly conditioned, so `rank` may be reported low for a system
//! that is controllable in exact arithmetic. Read `margin` and
//! `singular_values` alongside `rank`, and prefer a balanced or scaled `A`.
//! A staircase (orthogonal-transformation) algorithm would be the robust
//! fix; it is not implemented here.

use ndarray::{Array1, Array2};
use thiserror::Error;

use crate::kalman::KalmanFilter;

/// Errors produced when building or exercising a [`StateSpaceModel`].
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StateSpaceError {
    /// `A` must be square; it defines the state dimension.
    #[error("state matrix A must be square, got {rows}x{cols}")]
    NonSquareTransition {
        /// Row count of the offending `A`.
        rows: usize,
        /// Column count of the offending `A`.
        cols: usize,
    },

    /// A matrix or vector did not conform to the state dimension.
    #[error("{what}: expected {expected}, got {got}")]
    DimensionMismatch {
        /// Which quantity was wrong.
        what: &'static str,
        /// Dimension implied by the model.
        expected: usize,
        /// Dimension actually supplied.
        got: usize,
    },

    /// The state dimension is zero — there is no system to analyse.
    #[error("state dimension must be non-zero")]
    EmptyState,

    /// The singular-value decomposition backing a rank test failed.
    #[error("singular value decomposition failed: {0}")]
    Svd(String),
}

/// A discrete-time linear time-invariant system `(A, B, C)`.
///
/// Fields are private so that the dimension invariant established by
/// [`StateSpaceModel::new`] cannot be broken afterwards: every rank test and
/// simulation in this module assumes `A` is `n*n`, `B` is `n*m` and `C` is
/// `p*n` for a single `n`.
// @ai:invariant every StateSpaceModel satisfies A:n*n, B:n*m, C:p*n for one n — `new` is the only constructor, it rejects every non-conforming triple, and the fields are private with borrow-only accessors so no later mutation can violate it [T:test conf:0.95 src:state_space::rejects_input_matrix_with_wrong_row_count]
#[derive(Debug, Clone, PartialEq)]
pub struct StateSpaceModel {
    a: Array2<f64>,
    b: Array2<f64>,
    c: Array2<f64>,
}

impl StateSpaceModel {
    /// Build a model from `A` (n*n), `B` (n*m) and `C` (p*n).
    ///
    /// `m` or `p` may be zero — an autonomous system (`m == 0`) is trivially
    /// uncontrollable and an unmeasured system (`p == 0`) trivially
    /// unobservable, which the rank reports state rather than reject.
    pub fn new(a: Array2<f64>, b: Array2<f64>, c: Array2<f64>) -> Result<Self, StateSpaceError> {
        let (rows, cols) = (a.nrows(), a.ncols());
        if rows != cols {
            return Err(StateSpaceError::NonSquareTransition { rows, cols });
        }
        if rows == 0 {
            return Err(StateSpaceError::EmptyState);
        }
        if b.nrows() != rows {
            return Err(StateSpaceError::DimensionMismatch {
                what: "input matrix B row count",
                expected: rows,
                got: b.nrows(),
            });
        }
        if c.ncols() != rows {
            return Err(StateSpaceError::DimensionMismatch {
                what: "output matrix C column count",
                expected: rows,
                got: c.ncols(),
            });
        }
        Ok(Self { a, b, c })
    }

    /// Extract the plant `(A, B, C)` from a configured Kalman filter.
    ///
    /// Maps `transition -> A`, `control -> B`, `observation -> C`. The
    /// filter's estimate, covariance and noise covariances are deliberately
    /// dropped: they describe the *estimator*, not the plant.
    ///
    /// Note that [`KalmanFilter::new`] leaves `control` as an `n*1` zero
    /// matrix, so a filter whose control matrix was never set yields an
    /// uncontrollable model — correctly, since it has no input path.
    // @ai:invariant the extracted (A,B,C) is the same plant the filter itself runs on — stepping the model reproduces KalmanFilter::predict's noise-free state update exactly, so a mis-mapped slot cannot pass silently [T:test conf:0.9 src:state_space::extracted_model_reproduces_the_filters_noise_free_prediction]
    pub fn from_kalman(kf: &KalmanFilter) -> Result<Self, StateSpaceError> {
        Self::new(
            kf.transition.clone(),
            kf.control.clone(),
            kf.observation.clone(),
        )
    }

    /// State dimension `n`.
    pub fn state_dim(&self) -> usize {
        self.a.nrows()
    }

    /// Input dimension `m`.
    pub fn input_dim(&self) -> usize {
        self.b.ncols()
    }

    /// Output dimension `p`.
    pub fn output_dim(&self) -> usize {
        self.c.nrows()
    }

    /// State-transition matrix `A`.
    pub fn a(&self) -> &Array2<f64> {
        &self.a
    }

    /// Input matrix `B`.
    pub fn b(&self) -> &Array2<f64> {
        &self.b
    }

    /// Output matrix `C`.
    pub fn c(&self) -> &Array2<f64> {
        &self.c
    }

    /// One noise-free state update: `A x + B u`.
    pub fn step(&self, x: &Array1<f64>, u: &Array1<f64>) -> Result<Array1<f64>, StateSpaceError> {
        self.check_state(x, "state vector")?;
        self.check_input(u)?;
        Ok(self.a.dot(x) + self.b.dot(u))
    }

    /// Output equation `y = C x`.
    pub fn output(&self, x: &Array1<f64>) -> Result<Array1<f64>, StateSpaceError> {
        self.check_state(x, "state vector")?;
        Ok(self.c.dot(x))
    }

    /// Roll the plant forward over an input sequence.
    ///
    /// Returns the trajectory `[x_0, x_1, ..., x_N]` where `N = inputs.len()`,
    /// so the result is one longer than `inputs` and starts with `x0`.
    ///
    /// `noise` supplies the `w_k` term. It is an explicit caller-provided
    /// sequence rather than an internally sampled one: `ix-signal` promises
    /// that nothing on its public surface uses an RNG, and honouring that is
    /// worth more than the convenience. Pass `None` for the noise-free
    /// rollout. When `Some`, `noise` must have exactly `inputs.len()`
    /// entries, each of length `n`.
    // @ai:assumption keeping w_k caller-supplied is what upholds ix-signal's "no RNG anywhere on the public surface" determinism contract; nothing mechanical enforces that contract, so a future edit adding a sampler here would break it silently [P:manually-reviewed conf:0.8 src:crates/ix-signal/CONTRACTS.md]
    pub fn simulate(
        &self,
        x0: &Array1<f64>,
        inputs: &[Array1<f64>],
        noise: Option<&[Array1<f64>]>,
    ) -> Result<Vec<Array1<f64>>, StateSpaceError> {
        self.check_state(x0, "initial state vector")?;
        if let Some(w) = noise {
            if w.len() != inputs.len() {
                return Err(StateSpaceError::DimensionMismatch {
                    what: "process-noise sequence length",
                    expected: inputs.len(),
                    got: w.len(),
                });
            }
        }

        let mut trajectory = Vec::with_capacity(inputs.len() + 1);
        trajectory.push(x0.clone());
        let mut x = x0.clone();
        for (k, u) in inputs.iter().enumerate() {
            x = self.step(&x, u)?;
            if let Some(w) = noise {
                self.check_state(&w[k], "process-noise vector")?;
                x = &x + &w[k];
            }
            trajectory.push(x.clone());
        }
        Ok(trajectory)
    }

    /// Controllability matrix `[B, AB, A^2 B, ..., A^(n-1) B]`, shape
    /// `n * (n*m)`.
    // @ai:invariant n Krylov blocks is EXACT, not a truncation: by Cayley-Hamilton A^n B lies in the span of [B..A^(n-1)B], so appending further powers cannot raise the rank [T:test conf:0.9 src:state_space::extra_powers_beyond_n_minus_one_add_no_rank]
    pub fn controllability_matrix(&self) -> Array2<f64> {
        let n = self.state_dim();
        let m = self.input_dim();
        let mut out = Array2::zeros((n, n * m));
        if m == 0 {
            return out;
        }
        let mut block = self.b.clone();
        for i in 0..n {
            out.slice_mut(ndarray::s![.., i * m..(i + 1) * m])
                .assign(&block);
            block = self.a.dot(&block);
        }
        out
    }

    /// Observability matrix `[C; CA; ...; C A^(n-1)]`, shape `(n*p) * n`.
    pub fn observability_matrix(&self) -> Array2<f64> {
        let n = self.state_dim();
        let p = self.output_dim();
        let mut out = Array2::zeros((n * p, n));
        if p == 0 {
            return out;
        }
        let mut block = self.c.clone();
        for i in 0..n {
            out.slice_mut(ndarray::s![i * p..(i + 1) * p, ..])
                .assign(&block);
            block = block.dot(&self.a);
        }
        out
    }

    /// Kalman controllability rank test.
    ///
    /// `full_rank` is true iff every state direction is reachable from the
    /// input.
    pub fn controllability(&self) -> Result<RankReport, StateSpaceError> {
        rank_report(
            &self.controllability_matrix(),
            self.state_dim(),
            self.input_dim() == 0,
        )
    }

    /// Kalman observability rank test.
    ///
    /// `full_rank` is true iff every state direction leaves a signature in
    /// the output.
    pub fn observability(&self) -> Result<RankReport, StateSpaceError> {
        rank_report(
            &self.observability_matrix(),
            self.state_dim(),
            self.output_dim() == 0,
        )
    }

    fn check_state(&self, x: &Array1<f64>, what: &'static str) -> Result<(), StateSpaceError> {
        if x.len() != self.state_dim() {
            return Err(StateSpaceError::DimensionMismatch {
                what,
                expected: self.state_dim(),
                got: x.len(),
            });
        }
        Ok(())
    }

    fn check_input(&self, u: &Array1<f64>) -> Result<(), StateSpaceError> {
        if u.len() != self.input_dim() {
            return Err(StateSpaceError::DimensionMismatch {
                what: "input vector",
                expected: self.input_dim(),
                got: u.len(),
            });
        }
        Ok(())
    }
}

/// Outcome of a Kalman rank test.
#[derive(Debug, Clone, PartialEq)]
pub struct RankReport {
    /// Numerical rank of the test matrix.
    pub rank: usize,
    /// State dimension `n` — the rank required for the property to hold.
    pub required_rank: usize,
    /// `rank == required_rank`.
    pub full_rank: bool,
    /// `required_rank - rank`: how many state directions are dark (for
    /// observability) or unreachable (for controllability).
    pub deficiency: usize,
    /// Singular values of the test matrix, descending.
    pub singular_values: Vec<f64>,
    /// Smallest singular value — the 2-norm distance to rank deficiency.
    ///
    /// A full-rank report with a tiny `margin` describes a system that is
    /// technically observable and practically not.
    pub margin: f64,
    /// Threshold below which a singular value was treated as zero.
    pub tolerance: f64,
}

/// Numerical rank of `matrix`, against a state dimension of `required_rank`.
///
/// `degenerate` short-circuits the zero-column / zero-row cases the SVD
/// cannot accept.
fn rank_report(
    matrix: &Array2<f64>,
    required_rank: usize,
    degenerate: bool,
) -> Result<RankReport, StateSpaceError> {
    if degenerate {
        return Ok(RankReport {
            rank: 0,
            required_rank,
            full_rank: false,
            deficiency: required_rank,
            singular_values: Vec::new(),
            margin: 0.0,
            tolerance: 0.0,
        });
    }

    let svd = ix_math::svd::svd(matrix).map_err(|e| StateSpaceError::Svd(e.to_string()))?;
    let singular_values: Vec<f64> = svd.singular_values.to_vec();
    let sigma_max = singular_values.first().copied().unwrap_or(0.0);

    // Relative rank tolerance, the LAPACK / NumPy `matrix_rank` convention:
    // tol = max(rows, cols) * sigma_max * eps.
    // @ai:invariant a relative tolerance is REQUIRED here, not decoration: ix_math's one-sided Jacobi SVD leaves a mathematically-zero singular value near 1e-16*sigma_max rather than exactly 0, so comparing against 0.0 would over-report the rank of a genuinely deficient system [T:test conf:0.9 src:state_space::seeded_rank_deficiency_survives_a_similarity_transform]
    let scale = matrix.nrows().max(matrix.ncols()) as f64;
    let tolerance = scale * sigma_max * f64::EPSILON;

    let rank = svd.rank(tolerance);
    let margin = singular_values.last().copied().unwrap_or(0.0);

    Ok(RankReport {
        rank,
        required_rank,
        full_rank: rank == required_rank,
        deficiency: required_rank.saturating_sub(rank),
        singular_values,
        margin,
        tolerance,
    })
}
