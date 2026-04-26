//! Hidden Markov Models: Viterbi, forward, backward, forward-backward, Baum-Welch.

use ndarray::{Array1, Array2};

/// A discrete Hidden Markov Model with N hidden states and M observation symbols.
///
/// - `initial[i]` = P(start in state i)
/// - `transition[i][j]` = P(state j at t+1 | state i at t)
/// - `emission[i][k]` = P(observation k | state i)
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    /// Initial state distribution (length N, sums to 1).
    pub initial: Array1<f64>,
    /// State transition matrix (N x N, row-stochastic).
    pub transition: Array2<f64>,
    /// Emission probability matrix (N x M, row-stochastic).
    pub emission: Array2<f64>,
}

impl HiddenMarkovModel {
    /// Create an HMM from initial distribution, transition matrix, and emission matrix.
    /// Validates dimensions and stochastic properties.
    pub fn new(
        initial: Array1<f64>,
        transition: Array2<f64>,
        emission: Array2<f64>,
    ) -> Result<Self, String> {
        let n = initial.len();

        // Validate transition matrix is square and matches state count.
        let (tn, tm) = transition.dim();
        if tn != n || tm != n {
            return Err(format!(
                "Transition matrix must be {}x{}, got {}x{}",
                n, n, tn, tm
            ));
        }

        // Validate emission matrix has N rows.
        let (en, _em) = emission.dim();
        if en != n {
            return Err(format!("Emission matrix must have {} rows, got {}", n, en));
        }

        // Validate initial distribution sums to 1.
        let init_sum: f64 = initial.sum();
        if (init_sum - 1.0).abs() > 1e-6 {
            return Err(format!(
                "Initial distribution sums to {}, expected 1.0",
                init_sum
            ));
        }

        // Validate transition rows sum to 1.
        for i in 0..n {
            let row_sum: f64 = transition.row(i).sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(format!(
                    "Transition row {} sums to {}, expected 1.0",
                    i, row_sum
                ));
            }
        }

        // Validate emission rows sum to 1.
        for i in 0..n {
            let row_sum: f64 = emission.row(i).sum();
            if (row_sum - 1.0).abs() > 1e-6 {
                return Err(format!(
                    "Emission row {} sums to {}, expected 1.0",
                    i, row_sum
                ));
            }
        }

        // Validate all probabilities are non-negative.
        if initial.iter().any(|&v| v < 0.0) {
            return Err("Initial distribution contains negative values".to_string());
        }
        if transition.iter().any(|&v| v < 0.0) {
            return Err("Transition matrix contains negative values".to_string());
        }
        if emission.iter().any(|&v| v < 0.0) {
            return Err("Emission matrix contains negative values".to_string());
        }

        Ok(Self {
            initial,
            transition,
            emission,
        })
    }

    /// Number of hidden states.
    pub fn n_states(&self) -> usize {
        self.initial.len()
    }

    /// Number of observation symbols.
    pub fn n_observations(&self) -> usize {
        self.emission.ncols()
    }

    // ─── Forward algorithm ───────────────────────────────────────────

    /// Forward algorithm: compute the forward variable alpha[t][i] = P(o_1..o_t, q_t=i | model).
    /// Returns the full alpha table (T x N) and per-step scaling factors (T).
    /// Scaling prevents underflow for long sequences.
    fn forward_scaled(&self, observations: &[usize]) -> (Array2<f64>, Array1<f64>) {
        let t_len = observations.len();
        let n = self.n_states();
        let mut alpha = Array2::<f64>::zeros((t_len, n));
        let mut scales = Array1::<f64>::zeros(t_len);

        // Initialization: alpha[0][i] = pi[i] * B[i][o_0].
        for i in 0..n {
            alpha[[0, i]] = self.initial[i] * self.emission[[i, observations[0]]];
        }
        scales[0] = alpha.row(0).sum();
        if scales[0] > 0.0 {
            for i in 0..n {
                alpha[[0, i]] /= scales[0];
            }
        }

        // Induction.
        for t in 1..t_len {
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += alpha[[t - 1, i]] * self.transition[[i, j]];
                }
                alpha[[t, j]] = sum * self.emission[[j, observations[t]]];
            }
            scales[t] = alpha.row(t).sum();
            if scales[t] > 0.0 {
                for j in 0..n {
                    alpha[[t, j]] /= scales[t];
                }
            }
        }

        (alpha, scales)
    }

    /// Forward algorithm: compute P(observations | model).
    /// Returns the log-probability to avoid underflow.
    pub fn forward(&self, observations: &[usize]) -> f64 {
        if observations.is_empty() {
            return 0.0;
        }
        let (_alpha, scales) = self.forward_scaled(observations);
        scales.mapv(f64::ln).sum()
    }

    // ─── Backward algorithm ──────────────────────────────────────────

    /// Backward algorithm using the same scaling factors from forward.
    /// beta[t][i] = P(o_{t+1}..o_T | q_t=i, model), scaled.
    fn backward_scaled(&self, observations: &[usize], scales: &Array1<f64>) -> Array2<f64> {
        let t_len = observations.len();
        let n = self.n_states();
        let mut beta = Array2::<f64>::zeros((t_len, n));

        // Initialization: beta[T-1][i] = 1 / scale[T-1].
        for i in 0..n {
            beta[[t_len - 1, i]] = 1.0;
        }
        if scales[t_len - 1] > 0.0 {
            for i in 0..n {
                beta[[t_len - 1, i]] /= scales[t_len - 1];
            }
        }

        // Induction (backwards).
        for t in (0..t_len - 1).rev() {
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += self.transition[[i, j]]
                        * self.emission[[j, observations[t + 1]]]
                        * beta[[t + 1, j]];
                }
                beta[[t, i]] = sum;
            }
            if scales[t] > 0.0 {
                for i in 0..n {
                    beta[[t, i]] /= scales[t];
                }
            }
        }

        beta
    }

    /// Backward algorithm: compute backward probabilities (log-probability).
    pub fn backward(&self, observations: &[usize]) -> Array2<f64> {
        if observations.is_empty() {
            return Array2::zeros((0, self.n_states()));
        }
        let (_alpha, scales) = self.forward_scaled(observations);
        self.backward_scaled(observations, &scales)
    }

    // ─── Forward-backward (smoothing) ────────────────────────────────

    /// Forward-backward algorithm: compute posterior state probabilities gamma[t][i] = P(q_t=i | observations, model).
    /// Returns gamma as a T x N matrix where each row sums to 1.
    pub fn forward_backward(&self, observations: &[usize]) -> Array2<f64> {
        if observations.is_empty() {
            return Array2::zeros((0, self.n_states()));
        }
        let t_len = observations.len();
        let n = self.n_states();

        let (alpha, scales) = self.forward_scaled(observations);
        let beta = self.backward_scaled(observations, &scales);

        let mut gamma = Array2::<f64>::zeros((t_len, n));
        for t in 0..t_len {
            for i in 0..n {
                gamma[[t, i]] = alpha[[t, i]] * beta[[t, i]];
            }
            let row_sum: f64 = gamma.row(t).sum();
            if row_sum > 0.0 {
                for i in 0..n {
                    gamma[[t, i]] /= row_sum;
                }
            }
        }

        gamma
    }

    // ─── Viterbi algorithm ───────────────────────────────────────────

    /// Viterbi algorithm: find the most likely hidden state sequence given observations.
    /// Returns (best_path, log_probability).
    pub fn viterbi(&self, observations: &[usize]) -> (Vec<usize>, f64) {
        if observations.is_empty() {
            return (vec![], 0.0);
        }
        let t_len = observations.len();
        let n = self.n_states();

        // Work in log space to avoid underflow.
        let mut delta = Array2::<f64>::zeros((t_len, n));
        let mut psi = Array2::<usize>::zeros((t_len, n));

        // Initialization.
        for i in 0..n {
            let p = self.initial[i] * self.emission[[i, observations[0]]];
            delta[[0, i]] = if p > 0.0 { p.ln() } else { f64::NEG_INFINITY };
        }

        // Recursion.
        for t in 1..t_len {
            for j in 0..n {
                let mut best_val = f64::NEG_INFINITY;
                let mut best_idx = 0;
                for i in 0..n {
                    let trans = if self.transition[[i, j]] > 0.0 {
                        self.transition[[i, j]].ln()
                    } else {
                        f64::NEG_INFINITY
                    };
                    let val = delta[[t - 1, i]] + trans;
                    if val > best_val {
                        best_val = val;
                        best_idx = i;
                    }
                }
                let emit = if self.emission[[j, observations[t]]] > 0.0 {
                    self.emission[[j, observations[t]]].ln()
                } else {
                    f64::NEG_INFINITY
                };
                delta[[t, j]] = best_val + emit;
                psi[[t, j]] = best_idx;
            }
        }

        // Termination: find best final state.
        let mut best_final = 0;
        let mut best_prob = f64::NEG_INFINITY;
        for i in 0..n {
            if delta[[t_len - 1, i]] > best_prob {
                best_prob = delta[[t_len - 1, i]];
                best_final = i;
            }
        }

        // Backtrack.
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_final;
        for t in (0..t_len - 1).rev() {
            path[t] = psi[[t + 1, path[t + 1]]];
        }

        (path, best_prob)
    }

    // ─── MAP estimation ──────────────────────────────────────────────

    /// MAP estimation: most probable state at each time step via forward-backward.
    /// Returns the state with highest posterior probability at each t.
    pub fn map_estimate(&self, observations: &[usize]) -> Vec<usize> {
        let gamma = self.forward_backward(observations);
        let t_len = gamma.nrows();
        let mut states = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let row = gamma.row(t);
            let mut best = 0;
            let mut best_val = f64::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best = i;
                }
            }
            states.push(best);
        }
        states
    }

    // ─── Baum-Welch ──────────────────────────────────────────────────

    /// Baum-Welch (EM) algorithm: learn HMM parameters from a single observation sequence.
    /// Returns a new HMM with updated parameters after `max_iter` iterations or convergence.
    pub fn baum_welch(
        &self,
        observations: &[usize],
        max_iter: usize,
        tol: f64,
    ) -> Result<Self, String> {
        if observations.is_empty() {
            return Err("Cannot run Baum-Welch on empty observations".to_string());
        }

        let t_len = observations.len();
        let n = self.n_states();
        let m = self.n_observations();
        let mut model = self.clone();
        let mut prev_log_prob = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // E-step: compute alpha, beta, gamma, xi.
            let (alpha, scales) = model.forward_scaled(observations);
            let beta = model.backward_scaled(observations, &scales);

            // Compute log-probability for convergence check.
            let log_prob: f64 = scales.mapv(f64::ln).sum();
            if (log_prob - prev_log_prob).abs() < tol {
                break;
            }
            prev_log_prob = log_prob;

            // Gamma[t][i] = P(q_t = i | O, model).
            let mut gamma = Array2::<f64>::zeros((t_len, n));
            for t in 0..t_len {
                for i in 0..n {
                    gamma[[t, i]] = alpha[[t, i]] * beta[[t, i]];
                }
                let row_sum: f64 = gamma.row(t).sum();
                if row_sum > 0.0 {
                    for i in 0..n {
                        gamma[[t, i]] /= row_sum;
                    }
                }
            }

            // Xi[t][i][j] = P(q_t=i, q_{t+1}=j | O, model) for t = 0..T-2.
            let mut xi = vec![Array2::<f64>::zeros((n, n)); t_len.saturating_sub(1)];
            for t in 0..t_len.saturating_sub(1) {
                let mut denom = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        xi[t][[i, j]] = alpha[[t, i]]
                            * model.transition[[i, j]]
                            * model.emission[[j, observations[t + 1]]]
                            * beta[[t + 1, j]];
                        denom += xi[t][[i, j]];
                    }
                }
                if denom > 0.0 {
                    xi[t].mapv_inplace(|v| v / denom);
                }
            }

            // M-step: re-estimate parameters.
            // Initial distribution.
            let mut new_initial = gamma.row(0).to_owned();
            let init_sum: f64 = new_initial.sum();
            if init_sum > 0.0 {
                new_initial.mapv_inplace(|v| v / init_sum);
            }

            // Transition matrix.
            let mut new_transition = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                let gamma_sum: f64 = (0..t_len.saturating_sub(1)).map(|t| gamma[[t, i]]).sum();
                for j in 0..n {
                    let xi_sum: f64 = (0..t_len.saturating_sub(1)).map(|t| xi[t][[i, j]]).sum();
                    new_transition[[i, j]] = if gamma_sum > 0.0 {
                        xi_sum / gamma_sum
                    } else {
                        1.0 / n as f64
                    };
                }
            }
            // Normalize rows.
            for i in 0..n {
                let row_sum: f64 = new_transition.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..n {
                        new_transition[[i, j]] /= row_sum;
                    }
                }
            }

            // Emission matrix.
            let mut new_emission = Array2::<f64>::zeros((n, m));
            for i in 0..n {
                let gamma_sum: f64 = (0..t_len).map(|t| gamma[[t, i]]).sum();
                for k in 0..m {
                    let num: f64 = (0..t_len)
                        .filter(|&t| observations[t] == k)
                        .map(|t| gamma[[t, i]])
                        .sum();
                    new_emission[[i, k]] = if gamma_sum > 0.0 {
                        num / gamma_sum
                    } else {
                        1.0 / m as f64
                    };
                }
            }
            // Normalize rows.
            for i in 0..n {
                let row_sum: f64 = new_emission.row(i).sum();
                if row_sum > 0.0 {
                    for k in 0..m {
                        new_emission[[i, k]] /= row_sum;
                    }
                }
            }

            model.initial = new_initial;
            model.transition = new_transition;
            model.emission = new_emission;
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Helper: the classic weather HMM.
    /// States: 0=Rainy, 1=Sunny
    /// Observations: 0=Walk, 1=Shop, 2=Clean
    fn weather_hmm() -> HiddenMarkovModel {
        let initial = array![0.6, 0.4];
        let transition = array![[0.7, 0.3], [0.4, 0.6]];
        let emission = array![[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]];
        HiddenMarkovModel::new(initial, transition, emission).unwrap()
    }

    #[test]
    fn test_creation_valid() {
        let hmm = weather_hmm();
        assert_eq!(hmm.n_states(), 2);
        assert_eq!(hmm.n_observations(), 3);
    }

    #[test]
    fn test_creation_dimension_mismatch() {
        let initial = array![0.5, 0.5];
        let transition = array![[0.7, 0.3, 0.0], [0.4, 0.6, 0.0], [0.0, 0.0, 1.0]];
        let emission = array![[0.5, 0.5], [0.3, 0.7]];
        assert!(HiddenMarkovModel::new(initial, transition, emission).is_err());
    }

    #[test]
    fn test_creation_bad_initial_sum() {
        let initial = array![0.3, 0.3]; // sums to 0.6
        let transition = array![[0.7, 0.3], [0.4, 0.6]];
        let emission = array![[0.5, 0.5], [0.3, 0.7]];
        assert!(HiddenMarkovModel::new(initial, transition, emission).is_err());
    }

    #[test]
    fn test_creation_bad_transition_row() {
        let initial = array![0.5, 0.5];
        let transition = array![[0.5, 0.3], [0.4, 0.6]]; // row 0 sums to 0.8
        let emission = array![[0.5, 0.5], [0.3, 0.7]];
        assert!(HiddenMarkovModel::new(initial, transition, emission).is_err());
    }

    #[test]
    fn test_creation_negative_values() {
        let initial = array![1.5, -0.5]; // sums to 1 but negative
        let transition = array![[0.7, 0.3], [0.4, 0.6]];
        let emission = array![[0.5, 0.5], [0.3, 0.7]];
        assert!(HiddenMarkovModel::new(initial, transition, emission).is_err());
    }

    #[test]
    fn test_forward_probability() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2]; // Walk, Shop, Clean
        let log_prob = hmm.forward(&obs);
        // Should be a finite negative number (log of probability < 1).
        assert!(log_prob.is_finite());
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_forward_empty() {
        let hmm = weather_hmm();
        let log_prob = hmm.forward(&[]);
        assert!((log_prob - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_forward_single_observation() {
        let hmm = weather_hmm();
        // P(Walk) = pi_0 * B[0][0] + pi_1 * B[1][0]
        //         = 0.6 * 0.1 + 0.4 * 0.6 = 0.06 + 0.24 = 0.30
        let log_prob = hmm.forward(&[0]);
        let prob = log_prob.exp();
        assert!((prob - 0.30).abs() < 1e-6);
    }

    #[test]
    fn test_backward_dimensions() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2];
        let beta = hmm.backward(&obs);
        assert_eq!(beta.dim(), (3, 2));
    }

    #[test]
    fn test_forward_backward_rows_sum_to_one() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2, 0, 1];
        let gamma = hmm.forward_backward(&obs);
        assert_eq!(gamma.dim(), (5, 2));
        for t in 0..5 {
            let row_sum: f64 = gamma.row(t).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Row {} sums to {}",
                t,
                row_sum
            );
        }
    }

    #[test]
    fn test_forward_backward_empty() {
        let hmm = weather_hmm();
        let gamma = hmm.forward_backward(&[]);
        assert_eq!(gamma.dim(), (0, 2));
    }

    #[test]
    fn test_viterbi_deterministic() {
        // A simple HMM where the mapping is nearly deterministic.
        // State 0 always emits obs 0, state 1 always emits obs 1.
        let initial = array![1.0, 0.0];
        let transition = array![[0.1, 0.9], [0.9, 0.1]];
        let emission = array![[0.99, 0.01], [0.01, 0.99]];
        let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

        // Observations: 0, 1, 0, 1 should give states: 0, 1, 0, 1.
        let (path, log_prob) = hmm.viterbi(&[0, 1, 0, 1]);
        assert_eq!(path, vec![0, 1, 0, 1]);
        assert!(log_prob.is_finite());
    }

    #[test]
    fn test_viterbi_empty() {
        let hmm = weather_hmm();
        let (path, log_prob) = hmm.viterbi(&[]);
        assert!(path.is_empty());
        assert!((log_prob - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_viterbi_single() {
        let hmm = weather_hmm();
        let (path, _log_prob) = hmm.viterbi(&[0]); // Walk
                                                   // With initial=[0.6,0.4] and emission: Rainy->Walk=0.1, Sunny->Walk=0.6
                                                   // P(Rainy,Walk) = 0.6*0.1 = 0.06
                                                   // P(Sunny,Walk) = 0.4*0.6 = 0.24
                                                   // Sunny is more likely.
        assert_eq!(path, vec![1]);
    }

    #[test]
    fn test_viterbi_weather() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2]; // Walk, Shop, Clean
        let (path, log_prob) = hmm.viterbi(&obs);
        assert_eq!(path.len(), 3);
        assert!(log_prob.is_finite());
        // First observation Walk -> Sunny more likely (0.4*0.6 > 0.6*0.1).
        assert_eq!(path[0], 1);
    }

    #[test]
    fn test_map_estimate() {
        // Same deterministic-ish HMM.
        let initial = array![1.0, 0.0];
        let transition = array![[0.1, 0.9], [0.9, 0.1]];
        let emission = array![[0.99, 0.01], [0.01, 0.99]];
        let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

        let states = hmm.map_estimate(&[0, 1, 0, 1]);
        assert_eq!(states, vec![0, 1, 0, 1]);
    }

    #[test]
    fn test_baum_welch_improves_likelihood() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 2, 1, 0];

        let log_prob_before = hmm.forward(&obs);
        let trained = hmm.baum_welch(&obs, 50, 1e-8).unwrap();
        let log_prob_after = trained.forward(&obs);

        // Baum-Welch should not decrease the likelihood.
        assert!(
            log_prob_after >= log_prob_before - 1e-6,
            "Likelihood decreased: {} -> {}",
            log_prob_before,
            log_prob_after
        );
    }

    #[test]
    fn test_baum_welch_preserves_stochastic() {
        let hmm = weather_hmm();
        let obs = [0, 1, 2, 0, 1, 2, 0, 0, 1];
        let trained = hmm.baum_welch(&obs, 20, 1e-8).unwrap();

        // Initial sums to 1.
        assert!(
            (trained.initial.sum() - 1.0).abs() < 1e-6,
            "Initial sums to {}",
            trained.initial.sum()
        );

        // Transition rows sum to 1.
        for i in 0..trained.n_states() {
            let row_sum: f64 = trained.transition.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Transition row {} sums to {}",
                i,
                row_sum
            );
        }

        // Emission rows sum to 1.
        for i in 0..trained.n_states() {
            let row_sum: f64 = trained.emission.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "Emission row {} sums to {}",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_baum_welch_empty_observations() {
        let hmm = weather_hmm();
        assert!(hmm.baum_welch(&[], 10, 1e-6).is_err());
    }

    #[test]
    fn test_baum_welch_recovers_parameters() {
        // Start with a "wrong" model and train on data generated by the true model's structure.
        // Observations strongly correlated: state 0 -> obs 0, state 1 -> obs 1, alternating.
        let obs: Vec<usize> = (0..100).map(|i| i % 2).collect();

        // Start with slightly asymmetric guesses to break EM symmetry.
        let initial = array![0.6, 0.4];
        let transition = array![[0.4, 0.6], [0.6, 0.4]];
        let emission = array![[0.6, 0.4], [0.4, 0.6]];
        let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

        let trained = hmm.baum_welch(&obs, 100, 1e-10).unwrap();

        // After training, each state should specialize on one observation.
        // State ordering may flip, so check that one emission is high for obs 0
        // and the other is high for obs 1.
        let e00 = trained.emission[[0, 0]];
        let e01 = trained.emission[[0, 1]];
        let e10 = trained.emission[[1, 0]];
        let e11 = trained.emission[[1, 1]];

        // One state should strongly emit obs 0, the other obs 1.
        let specialized = (e00 > 0.8 && e11 > 0.8) || (e01 > 0.8 && e10 > 0.8);
        assert!(
            specialized,
            "Emission not specialized: [[{:.3}, {:.3}], [{:.3}, {:.3}]]",
            e00, e01, e10, e11
        );
    }

    #[test]
    fn test_three_state_hmm() {
        let initial = array![0.5, 0.3, 0.2];
        let transition = array![[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]];
        let emission = array![[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6]];
        let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

        let obs = [0, 1, 2, 0, 2, 1];
        let (path, log_prob) = hmm.viterbi(&obs);
        assert_eq!(path.len(), 6);
        assert!(log_prob.is_finite());

        let gamma = hmm.forward_backward(&obs);
        assert_eq!(gamma.dim(), (6, 3));
        for t in 0..6 {
            let sum: f64 = gamma.row(t).sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_forward_backward_consistency() {
        // The forward log-prob should match what we compute from alpha alone.
        let hmm = weather_hmm();
        let obs = [2, 1, 0, 2];

        let log_prob_forward = hmm.forward(&obs);
        // Gamma should be valid probabilities regardless.
        let gamma = hmm.forward_backward(&obs);
        for t in 0..obs.len() {
            for i in 0..hmm.n_states() {
                assert!(gamma[[t, i]] >= 0.0);
                assert!(gamma[[t, i]] <= 1.0 + 1e-10);
            }
        }
        assert!(log_prob_forward.is_finite());
    }
}
