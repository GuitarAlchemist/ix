//! Silhouette coefficient — an internal clustering-quality metric.
//!
//! For each sample `i`: `a(i)` = mean distance to the other points in its own
//! cluster, `b(i)` = the minimum over other clusters of the mean distance to
//! that cluster. `s(i) = (b - a) / max(a, b)`. The silhouette score is the mean
//! of `s(i)` over all samples, in `[-1, 1]`; higher means denser, better-
//! separated clusters. This is the exact O(n²) computation (Rousseeuw 1987),
//! matching sklearn's `silhouette_score` on its valid input domain
//! (`2..=n-1` clusters); on degenerate inputs it returns `0.0` rather than
//! raising as sklearn does.
//!
//! Lives in the agent layer rather than a stable library crate so the catalog
//! can expose it without growing a stable public surface (the workspace shares
//! one `0.1.0` version, so a `pub fn` in a stable crate trips the stable-surface
//! gate). Promotable to `ix-unsupervised::metrics` if a second consumer appears.

use ndarray::Array2;
use std::collections::{BTreeSet, HashMap};

/// Mean silhouette coefficient over all samples.
///
/// Returns `0.0` for the degenerate cases where it is undefined: fewer than 2
/// samples, a label/row count mismatch, or fewer than 2 distinct clusters (no
/// "other cluster" for `b(i)` to measure against). A point alone in its cluster
/// contributes `s(i) = 0` by the Rousseeuw convention.
pub fn silhouette_score(data: &Array2<f64>, labels: &[usize]) -> f64 {
    let n = data.nrows();
    if n < 2 || labels.len() != n {
        return 0.0;
    }
    // b(i) is undefined without at least one *other* cluster.
    let distinct: BTreeSet<usize> = labels.iter().copied().collect();
    if distinct.len() < 2 {
        return 0.0;
    }

    let mut total = 0.0;
    for i in 0..n {
        let ci = labels[i];
        let mut a_sum = 0.0;
        let mut a_count = 0usize;
        // mean distance from i to each *other* cluster, accumulated as (sum, count)
        let mut other: HashMap<usize, (f64, usize)> = HashMap::new();

        for (j, &lj) in labels.iter().enumerate() {
            if i == j {
                continue;
            }
            let d = euclidean(data, i, j);
            if lj == ci {
                a_sum += d;
                a_count += 1;
            } else {
                let e = other.entry(lj).or_insert((0.0, 0));
                e.0 += d;
                e.1 += 1;
            }
        }

        // Singleton cluster (no same-cluster neighbours): s(i) = 0 by convention.
        let s = if a_count == 0 {
            0.0
        } else {
            let a = a_sum / a_count as f64;
            let b = other
                .values()
                .map(|(sum, cnt)| sum / *cnt as f64)
                .fold(f64::INFINITY, f64::min);
            if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            }
        };
        total += s;
    }
    total / n as f64
}

fn euclidean(data: &Array2<f64>, i: usize, j: usize) -> f64 {
    data.row(i)
        .iter()
        .zip(data.row(j).iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn well_separated_clusters_score_near_one() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1]
        ];
        let labels = [0, 0, 0, 1, 1, 1];
        let s = silhouette_score(&data, &labels);
        assert!(s > 0.9, "tight, far-apart clusters → ~1, got {s}");
    }

    // The metric must actually measure separation: the correct labelling has to
    // score strictly higher than one that interleaves the two physical groups.
    #[test]
    fn well_separated_scores_higher_than_interleaved() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1]
        ];
        let good = silhouette_score(&data, &[0, 0, 0, 1, 1, 1]);
        let interleaved = silhouette_score(&data, &[0, 1, 0, 1, 0, 1]);
        assert!(
            good > interleaved,
            "correct labels must beat interleaved: {good} vs {interleaved}"
        );
        assert!(
            interleaved < 0.0,
            "interleaving far-apart points should be negative, got {interleaved}"
        );
    }

    #[test]
    fn single_cluster_is_zero() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        assert_eq!(silhouette_score(&data, &[0, 0, 0]), 0.0);
    }

    #[test]
    fn fewer_than_two_samples_is_zero() {
        let data = array![[0.0, 0.0]];
        assert_eq!(silhouette_score(&data, &[0]), 0.0);
    }

    #[test]
    fn label_row_mismatch_is_zero() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        assert_eq!(silhouette_score(&data, &[0]), 0.0);
    }
}
