//! Per-variant fuzzy operations: AND, OR, NOT.
//!
//! The Demerzel spec
//! (`governance/demerzel/logic/fuzzy-membership.md`) defines:
//!
//! - **AND** — per-variant `min`, then renormalize
//! - **OR**  — per-variant `max`, then renormalize
//! - **NOT (generic)** — per-variant `1.0 - v`, then renormalize
//!
//! The hexavalent-specific NOT (`T↔F`, `P↔D`, `U`/`C` preserved) is
//! provided in [`crate::hexavalent`] as it depends on the concrete
//! `Hexavalent` enum.
//!
//! All operations take the union of variants from both inputs, so
//! distributions do NOT need to be defined over the same support.

use std::collections::BTreeSet;

use crate::distribution::FuzzyDistribution;
use crate::error::FuzzyError;

impl<T> FuzzyDistribution<T>
where
    T: Ord + Clone,
{
    /// Per-variant `min(a, b)` followed by renormalization.
    ///
    /// The union of both distributions' supports is considered;
    /// variants absent from either distribution contribute `0.0`
    /// and therefore drop out after the `min`.
    pub fn and(&self, other: &Self) -> Result<Self, FuzzyError> {
        let keys = union_keys(self, other);
        let pairs: Vec<(T, f64)> = keys
            .into_iter()
            .map(|k| {
                let m = self.get(&k).min(other.get(&k));
                (k, m)
            })
            .collect();
        finalize_after_combine(pairs)
    }

    /// Per-variant `max(a, b)` followed by renormalization.
    pub fn or(&self, other: &Self) -> Result<Self, FuzzyError> {
        let keys = union_keys(self, other);
        let pairs: Vec<(T, f64)> = keys
            .into_iter()
            .map(|k| {
                let m = self.get(&k).max(other.get(&k));
                (k, m)
            })
            .collect();
        finalize_after_combine(pairs)
    }

    /// Generic (non-hexavalent) NOT: replace every membership with
    /// `1.0 - v` and renormalize. Callers that want the
    /// hexavalent-specific `T↔F` / `P↔D` swap should use
    /// [`crate::hexavalent::hexavalent_not`].
    pub fn not_generic(&self) -> Result<Self, FuzzyError> {
        let pairs: Vec<(T, f64)> = self
            .iter()
            .map(|(k, v)| (k.clone(), (1.0 - v).max(0.0)))
            .collect();
        finalize_after_combine(pairs)
    }
}

fn union_keys<T: Ord + Clone>(a: &FuzzyDistribution<T>, b: &FuzzyDistribution<T>) -> BTreeSet<T> {
    let mut keys: BTreeSet<T> = BTreeSet::new();
    keys.extend(a.memberships().keys().cloned());
    keys.extend(b.memberships().keys().cloned());
    keys
}

/// Turn a pair list (possibly unnormalized) into a fresh
/// [`FuzzyDistribution`]. If every mass is zero, returns
/// [`FuzzyError::BadSum`] — AND of two disjoint supports is
/// genuinely empty and the caller must decide how to handle it.
fn finalize_after_combine<T: Ord + Clone>(
    pairs: Vec<(T, f64)>,
) -> Result<FuzzyDistribution<T>, FuzzyError> {
    let sum: f64 = pairs.iter().map(|(_, v)| *v).sum();
    if sum <= 0.0 {
        return Err(FuzzyError::BadSum { sum });
    }
    let normalized: Vec<(T, f64)> = pairs.into_iter().map(|(k, v)| (k, v / sum)).collect();
    FuzzyDistribution::new(normalized)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn d(pairs: &[(&'static str, f64)]) -> FuzzyDistribution<&'static str> {
        FuzzyDistribution::new(pairs.iter().copied()).unwrap()
    }

    #[test]
    fn and_is_min_then_renormalize() {
        let a = d(&[("x", 0.8), ("y", 0.2)]);
        let b = d(&[("x", 0.4), ("y", 0.6)]);
        let c = a.and(&b).unwrap();
        // min(0.8,0.4)=0.4; min(0.2,0.6)=0.2; sum 0.6 → 0.667/0.333
        assert!((c.get(&"x") - (0.4 / 0.6)).abs() < 1e-9);
        assert!((c.get(&"y") - (0.2 / 0.6)).abs() < 1e-9);
    }

    #[test]
    fn or_is_max_then_renormalize() {
        let a = d(&[("x", 0.8), ("y", 0.2)]);
        let b = d(&[("x", 0.4), ("y", 0.6)]);
        let c = a.or(&b).unwrap();
        // max(0.8,0.4)=0.8; max(0.2,0.6)=0.6; sum 1.4 → ≈0.571/0.429
        assert!((c.get(&"x") - (0.8 / 1.4)).abs() < 1e-9);
        assert!((c.get(&"y") - (0.6 / 1.4)).abs() < 1e-9);
    }

    #[test]
    fn not_generic_inverts_and_renormalizes() {
        let a = d(&[("x", 0.8), ("y", 0.2)]);
        let n = a.not_generic().unwrap();
        // 1-0.8=0.2; 1-0.2=0.8; sum 1.0 → identical after renorm.
        assert!((n.get(&"x") - 0.2).abs() < 1e-9);
        assert!((n.get(&"y") - 0.8).abs() < 1e-9);
    }

    #[test]
    fn and_on_disjoint_supports_errors() {
        let a = d(&[("x", 1.0)]);
        let b = d(&[("y", 1.0)]);
        // min(x: 1.0, 0.0) = 0; min(y: 0.0, 1.0) = 0; empty → BadSum
        let err = a.and(&b).unwrap_err();
        assert!(matches!(err, FuzzyError::BadSum { .. }));
    }

    #[test]
    fn or_on_disjoint_supports_combines() {
        let a = d(&[("x", 1.0)]);
        let b = d(&[("y", 1.0)]);
        let c = a.or(&b).unwrap();
        assert!((c.get(&"x") - 0.5).abs() < 1e-9);
        assert!((c.get(&"y") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn ops_preserve_determinism_across_input_order() {
        let a = d(&[("a", 0.5), ("b", 0.5)]);
        let b1 = d(&[("a", 0.3), ("b", 0.7)]);
        let b2 = d(&[("b", 0.7), ("a", 0.3)]);
        let ab1 = a.and(&b1).unwrap();
        let ab2 = a.and(&b2).unwrap();
        assert_eq!(
            serde_json::to_string(&ab1).unwrap(),
            serde_json::to_string(&ab2).unwrap()
        );
    }
}
