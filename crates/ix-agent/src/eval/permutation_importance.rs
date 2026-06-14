//! Permutation feature importance — a model-agnostic explainability metric.
//!
//! For a fitted classifier, `importance[j]` = baseline accuracy − mean accuracy
//! with feature column `j` randomly permuted. A feature the model relies on
//! shows a large accuracy drop when scrambled (high importance); an irrelevant
//! feature shows ≈0 (occasionally slightly negative). Model-agnostic: it takes a
//! prediction closure, so it explains any classifier — not just random forests
//! (Breiman 2001). It is the standard alternative to Gini/MDI and avoids MDI's
//! bias toward high-cardinality features, *and* it needs only the model's
//! existing `predict()` — so no stable library crate has to grow its public API.
//!
//! Deterministic given `seed`: the per-(feature, repeat) shuffle uses an inline
//! splitmix64 PRNG, so there is no external `rand` dependency and the result is
//! reproducible across runs.

use ndarray::Array2;

/// Permutation importance for each of the `p` feature columns of `x`.
///
/// `predict` maps a feature matrix to predicted class labels; `y` are the true
/// labels (`y.len() == x.nrows()`). Each feature is permuted `n_repeats` times
/// (at least once) and its importance is the mean accuracy drop. Returns a
/// length-`p` vector aligned to the feature columns. Returns all-zeros for empty
/// input or a label/row mismatch.
pub fn permutation_importance<F>(
    predict: F,
    x: &Array2<f64>,
    y: &[usize],
    n_repeats: usize,
    seed: u64,
) -> Vec<f64>
where
    F: Fn(&Array2<f64>) -> Vec<usize>,
{
    let (n, p) = x.dim();
    if n == 0 || y.len() != n {
        return vec![0.0; p];
    }
    let reps = n_repeats.max(1);
    let baseline = accuracy(&predict(x), y);

    let mut importances = vec![0.0; p];
    for (j, imp) in importances.iter_mut().enumerate() {
        let mut drop_sum = 0.0;
        for r in 0..reps {
            // Distinct PRNG stream per (feature, repeat) — so columns don't share
            // a permutation — but fully seed-derived, so the whole run is
            // reproducible.
            let mut rng = SplitMix64::new(
                seed ^ (j as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (r as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
            );
            let mut xp = x.clone();
            permute_column(&mut xp, j, &mut rng);
            drop_sum += baseline - accuracy(&predict(&xp), y);
        }
        *imp = drop_sum / reps as f64;
    }
    importances
}

fn accuracy(pred: &[usize], y: &[usize]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let correct = pred.iter().zip(y).filter(|(a, b)| a == b).count();
    correct as f64 / y.len() as f64
}

/// Fisher–Yates shuffle of column `j` of `x`, in place.
fn permute_column(x: &mut Array2<f64>, j: usize, rng: &mut SplitMix64) {
    let n = x.nrows();
    for i in (1..n).rev() {
        let k = (rng.next_u64() % (i as u64 + 1)) as usize;
        let tmp = x[[i, j]];
        x[[i, j]] = x[[k, j]];
        x[[k, j]] = tmp;
    }
}

/// Minimal splitmix64 PRNG — deterministic, no external dependency.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // A controlled predictor that keys ONLY on feature 0 (threshold at 0.5):
    // permuting column 0 must destroy accuracy (high importance) while permuting
    // the noise columns 1,2 must leave it ~unchanged (≈0). This proves the metric
    // attributes importance to the feature the model actually uses.
    #[test]
    fn informative_feature_ranks_above_noise() {
        let x = array![
            [1.0, 0.3, 9.0],
            [1.0, 0.9, 1.0],
            [1.0, 0.1, 4.0],
            [1.0, 0.7, 7.0],
            [0.0, 0.5, 2.0],
            [0.0, 0.2, 8.0],
            [0.0, 0.4, 3.0],
            [0.0, 0.8, 6.0]
        ];
        let y = vec![1, 1, 1, 1, 0, 0, 0, 0];
        let predict = |m: &Array2<f64>| -> Vec<usize> {
            m.column(0)
                .iter()
                .map(|&v| if v > 0.5 { 1 } else { 0 })
                .collect()
        };

        let imp = permutation_importance(predict, &x, &y, 5, 7);
        assert_eq!(imp.len(), 3);
        assert!(
            imp[0] > 0.3,
            "scrambling the determining feature should drop accuracy a lot, got {imp:?}"
        );
        assert!(
            imp[0] > imp[1] && imp[0] > imp[2],
            "feature 0 must outrank the noise features, got {imp:?}"
        );
        assert!(
            imp[1].abs() < 0.15 && imp[2].abs() < 0.15,
            "noise features should be ≈0, got {imp:?}"
        );
    }

    #[test]
    fn deterministic_given_seed() {
        let x = array![[1.0, 5.0], [0.0, 6.0], [1.0, 7.0], [0.0, 8.0]];
        let y = vec![1, 0, 1, 0];
        let predict = |m: &Array2<f64>| -> Vec<usize> {
            m.column(0)
                .iter()
                .map(|&v| if v > 0.5 { 1 } else { 0 })
                .collect()
        };
        let a = permutation_importance(predict, &x, &y, 4, 123);
        let b = permutation_importance(predict, &x, &y, 4, 123);
        assert_eq!(a, b, "same seed → identical importances");
    }

    #[test]
    fn empty_input_returns_zeros() {
        let x = Array2::<f64>::zeros((0, 3));
        let predict = |_: &Array2<f64>| -> Vec<usize> { vec![] };
        assert_eq!(permutation_importance(predict, &x, &[], 3, 1), vec![0.0; 3]);
    }
}
