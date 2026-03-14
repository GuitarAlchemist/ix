//! Mayer-Vietoris consistency checks for distributed data.
//!
//! Uses the Mayer-Vietoris exact sequence from algebraic topology:
//! K₀(A) + K₀(B) - K₀(A ∩ B) = K₀(A ∪ B)
//!
//! This provides a consistency check for distributed/sharded data:
//! if two shards (A, B) overlap, the K₀ invariants must satisfy
//! the Mayer-Vietoris relation.
//!
//! # Examples
//!
//! ```
//! use machin_ktheory::mayer_vietoris;
//!
//! // Two shards with overlap
//! let shard_a: Vec<u64> = vec![1, 2, 3, 4, 5];
//! let shard_b: Vec<u64> = vec![4, 5, 6, 7, 8];
//! let overlap: Vec<u64> = vec![4, 5];
//!
//! // Check consistency: |A| + |B| - |A∩B| = |A∪B|
//! assert!(mayer_vietoris::consistency_check(&shard_a, &shard_b, &overlap));
//! ```

/// Check Mayer-Vietoris consistency for two shards with known overlap.
///
/// Verifies: |A| + |B| - |A ∩ B| = |A ∪ B|
///
/// This is the simplest form of the Mayer-Vietoris principle applied
/// to cardinality (K₀ = rank = count of distinct elements).
pub fn consistency_check<T: Eq + std::hash::Hash + Clone>(
    shard_a: &[T],
    shard_b: &[T],
    overlap: &[T],
) -> bool {
    use std::collections::HashSet;

    let set_a: HashSet<&T> = shard_a.iter().collect();
    let set_b: HashSet<&T> = shard_b.iter().collect();

    // Verify overlap is actually a subset of both A and B
    let overlap_set: HashSet<&T> = overlap.iter().collect();
    let overlap_valid = overlap_set.iter().all(|x| set_a.contains(x) && set_b.contains(x));

    if !overlap_valid {
        return false;
    }

    // Compute actual intersection
    let actual_intersection: HashSet<&T> = set_a.intersection(&set_b).copied().collect();

    // Check that claimed overlap matches actual intersection
    if overlap_set != actual_intersection {
        return false;
    }

    // Mayer-Vietoris: |A| + |B| - |A∩B| = |A∪B|
    let union: HashSet<&T> = set_a.union(&set_b).copied().collect();
    let lhs = set_a.len() + set_b.len() - overlap_set.len();
    let rhs = union.len();

    lhs == rhs
}

/// Verify consistency of K₀ invariants across shards.
///
/// Given K₀ ranks for shard_a, shard_b, their overlap, and their union,
/// checks the Mayer-Vietoris relation: k0_a + k0_b - k0_overlap = k0_union.
pub fn k0_consistency(k0_a: i64, k0_b: i64, k0_overlap: i64, k0_union: i64) -> bool {
    k0_a + k0_b - k0_overlap == k0_union
}

/// Compute the expected K₀ of a union given individual K₀ values.
///
/// Returns k0_a + k0_b - k0_overlap.
pub fn expected_union_k0(k0_a: i64, k0_b: i64, k0_overlap: i64) -> i64 {
    k0_a + k0_b - k0_overlap
}

/// Detect inconsistency between shards.
///
/// Returns `Some(delta)` if the K₀ relation is violated, where delta
/// is the discrepancy. Returns `None` if consistent.
pub fn detect_inconsistency(k0_a: i64, k0_b: i64, k0_overlap: i64, k0_union: i64) -> Option<i64> {
    let expected = k0_a + k0_b - k0_overlap;
    let delta = k0_union - expected;
    if delta != 0 {
        Some(delta)
    } else {
        None
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consistency_check_valid() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![4, 5, 6, 7, 8];
        let overlap = vec![4, 5];
        assert!(consistency_check(&a, &b, &overlap));
    }

    #[test]
    fn test_consistency_check_no_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let overlap: Vec<i32> = vec![];
        assert!(consistency_check(&a, &b, &overlap));
    }

    #[test]
    fn test_consistency_check_full_overlap() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        let overlap = vec![1, 2, 3];
        assert!(consistency_check(&a, &b, &overlap));
    }

    #[test]
    fn test_consistency_check_wrong_overlap() {
        // Claimed overlap doesn't match actual intersection
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![4, 5, 6, 7, 8];
        let overlap = vec![4]; // Missing 5
        assert!(!consistency_check(&a, &b, &overlap));
    }

    #[test]
    fn test_consistency_check_invalid_overlap() {
        // Overlap element not in both shards
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let overlap = vec![1]; // 1 is not in B
        assert!(!consistency_check(&a, &b, &overlap));
    }

    #[test]
    fn test_k0_consistency_valid() {
        assert!(k0_consistency(5, 5, 2, 8));
    }

    #[test]
    fn test_k0_consistency_invalid() {
        assert!(!k0_consistency(5, 5, 2, 7));
    }

    #[test]
    fn test_expected_union_k0() {
        assert_eq!(expected_union_k0(10, 8, 3), 15);
    }

    #[test]
    fn test_detect_inconsistency_none() {
        assert!(detect_inconsistency(5, 5, 2, 8).is_none());
    }

    #[test]
    fn test_detect_inconsistency_some() {
        let delta = detect_inconsistency(5, 5, 2, 10).unwrap();
        assert_eq!(delta, 2); // 10 - 8 = 2
    }

    #[test]
    fn test_consistency_with_strings() {
        let a = vec!["foo", "bar", "baz"];
        let b = vec!["baz", "qux"];
        let overlap = vec!["baz"];
        assert!(consistency_check(&a, &b, &overlap));
    }
}
