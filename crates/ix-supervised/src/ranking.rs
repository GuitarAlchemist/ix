//! Information-retrieval / ranking metrics over a ranked list of relevance scores.
//!
//! `rels[i]` is the (graded) relevance of the item at rank `i` (0-based, so rank 1 is
//! `rels[0]`); higher means more relevant, and `rel > 0` counts as "relevant". These
//! are the per-query metrics behind retrieval and intent-routing evaluation (nDCG,
//! MRR, precision@k, recall@k). Mean over queries gives the aggregate (e.g. MRR =
//! mean of [`reciprocal_rank`]).

/// Discounted Cumulative Gain at `k`: `Σ_{i<k} rel_i / log2(i + 2)`.
pub fn dcg_at_k(rels: &[f64], k: usize) -> f64 {
    rels.iter()
        .take(k)
        .enumerate()
        .map(|(i, &r)| r / ((i + 2) as f64).log2())
        .sum()
}

/// Normalized DCG at `k`: `dcg / ideal_dcg` where ideal sorts relevances descending.
/// Returns 0 when the ideal DCG is 0 (no relevant items). Range `[0, 1]`.
pub fn ndcg_at_k(rels: &[f64], k: usize) -> f64 {
    let dcg = dcg_at_k(rels, k);
    let mut ideal = rels.to_vec();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idcg = dcg_at_k(&ideal, k);
    if idcg <= 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Reciprocal rank: `1 / (rank of the first relevant item)`, rank 1-based; 0 if none.
/// Mean over queries is MRR.
pub fn reciprocal_rank(rels: &[f64]) -> f64 {
    rels.iter()
        .position(|&r| r > 0.0)
        .map(|i| 1.0 / (i + 1) as f64)
        .unwrap_or(0.0)
}

/// Precision@k: fraction of the top `k` that are relevant. Denominator is `k`
/// (the requested cutoff); 0 if `k == 0`.
pub fn precision_at_k(rels: &[f64], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits = rels.iter().take(k).filter(|&&r| r > 0.0).count();
    hits as f64 / k as f64
}

/// Recall@k: fraction of all relevant items recovered in the top `k`.
/// `total_relevant` is supplied by the caller (the ranked list may be truncated).
/// 0 if `total_relevant == 0`.
pub fn recall_at_k(rels: &[f64], k: usize, total_relevant: usize) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }
    let hits = rels.iter().take(k).filter(|&&r| r > 0.0).count();
    hits as f64 / total_relevant as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ndcg_perfect_ranking_is_one() {
        // Already in ideal (descending) order → nDCG = 1.
        assert!((ndcg_at_k(&[3.0, 2.0, 1.0, 0.0], 4) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ndcg_worse_ranking_below_one() {
        // Reversed order → strictly worse than ideal.
        let bad = ndcg_at_k(&[0.0, 1.0, 2.0, 3.0], 4);
        assert!(bad < 1.0 && bad > 0.0, "got {bad}");
    }

    #[test]
    fn ndcg_no_relevant_is_zero() {
        assert_eq!(ndcg_at_k(&[0.0, 0.0, 0.0], 3), 0.0);
    }

    #[test]
    fn reciprocal_rank_first_relevant() {
        assert!((reciprocal_rank(&[0.0, 0.0, 1.0]) - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(reciprocal_rank(&[0.0, 0.0]), 0.0);
        assert_eq!(reciprocal_rank(&[1.0, 0.0]), 1.0);
    }

    #[test]
    fn precision_and_recall_at_k() {
        // [1,0,1,0]: top-2 has 1 relevant → P@2 = 0.5.
        assert!((precision_at_k(&[1.0, 0.0, 1.0, 0.0], 2) - 0.5).abs() < 1e-12);
        // 2 of 4 total-relevant recovered in top-3 → R@3 = 0.5.
        assert!((recall_at_k(&[1.0, 0.0, 1.0], 3, 4) - 0.5).abs() < 1e-12);
        assert_eq!(recall_at_k(&[1.0], 1, 0), 0.0);
    }
}
