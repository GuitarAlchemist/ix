use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Not;

/// A six-valued (hexavalent) truth value for agent reasoning under uncertainty.
///
/// Extends the original tetravalent system (T/F/U/C) with two epistemic
/// intermediate values:
///
/// - **Probable (P)**: evidence leans true but is not conclusive.
/// - **Doubtful (D)**: evidence leans false but is not conclusive — the
///   symmetric mirror of Probable (distinct from Contradictory, which is a
///   logical inconsistency, not a mere lean).
///
/// Ordering on the "truth axis": T > P > U > D > F.
/// C (Contradictory) is orthogonal — it indicates a logical inconsistency
/// that cannot resolve with more evidence.
///
/// This enum is the **governance wire adapter** (its `Display`/`"D"` symbols
/// flow through the `ix_governance_belief` MCP tool). Its *algebra*
/// (`and`/`or`/`not`/`implies`/`xor`/`equiv`) delegates to the canonical
/// [`ix_types::Hexavalent`] so there is exactly one hexavalent truth table in
/// the workspace — see [`TruthValue::to_hex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TruthValue {
    /// Verified with sufficient evidence.
    True,
    /// Evidence leans true but not conclusive.
    Probable,
    /// Insufficient evidence to determine.
    Unknown,
    /// Evidence leans false but not conclusive (mirror of Probable).
    Doubtful,
    /// Refuted with sufficient evidence.
    False,
    /// Evidence supports both True and False (logical inconsistency).
    Contradictory,
}

impl fmt::Display for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.to_hex().as_str())
    }
}

impl Not for TruthValue {
    type Output = Self;

    /// Hexavalent NOT: T↔F, P↔D, U→U, C→C (delegates to canonical).
    fn not(self) -> Self {
        Self::from_hex(self.to_hex().not())
    }
}

impl TruthValue {
    /// Lower this governance wire value to the canonical [`ix_types::Hexavalent`]
    /// (lossless, total). The algebra lives there; this enum is the adapter.
    pub const fn to_hex(self) -> ix_types::Hexavalent {
        match self {
            TruthValue::True => ix_types::Hexavalent::True,
            TruthValue::Probable => ix_types::Hexavalent::Probable,
            TruthValue::Unknown => ix_types::Hexavalent::Unknown,
            TruthValue::Doubtful => ix_types::Hexavalent::Doubtful,
            TruthValue::False => ix_types::Hexavalent::False,
            TruthValue::Contradictory => ix_types::Hexavalent::Contradictory,
        }
    }

    /// Lift a canonical [`ix_types::Hexavalent`] back to this wire enum.
    pub const fn from_hex(h: ix_types::Hexavalent) -> Self {
        match h {
            ix_types::Hexavalent::True => TruthValue::True,
            ix_types::Hexavalent::Probable => TruthValue::Probable,
            ix_types::Hexavalent::Unknown => TruthValue::Unknown,
            ix_types::Hexavalent::Doubtful => TruthValue::Doubtful,
            ix_types::Hexavalent::False => TruthValue::False,
            ix_types::Hexavalent::Contradictory => TruthValue::Contradictory,
        }
    }

    /// Hexavalent AND (delegates to [`ix_types::Hexavalent::and`]).
    pub fn and(self, other: Self) -> Self {
        Self::from_hex(self.to_hex().and(other.to_hex()))
    }

    /// Hexavalent OR (delegates to [`ix_types::Hexavalent::or`], the De
    /// Morgan-derived canonical table).
    pub fn or(self, other: Self) -> Self {
        Self::from_hex(self.to_hex().or(other.to_hex()))
    }

    /// Hexavalent implication: A -> B = (NOT A) OR B (delegates).
    pub fn implies(self, other: Self) -> Self {
        Self::from_hex(self.to_hex().implies(other.to_hex()))
    }

    /// Hexavalent XOR (delegates to [`ix_types::Hexavalent::xor`]).
    pub fn xor(self, other: Self) -> Self {
        Self::from_hex(self.to_hex().xor(other.to_hex()))
    }

    /// Hexavalent equivalence: A <-> B = (A -> B) AND (B -> A) (delegates).
    pub fn equiv(self, other: Self) -> Self {
        Self::from_hex(self.to_hex().equiv(other.to_hex()))
    }

    /// Returns true if this value is definite (True or False).
    pub fn is_definite(self) -> bool {
        matches!(self, TruthValue::True | TruthValue::False)
    }

    /// Returns true if this value is indefinite (not True or False).
    pub fn is_indefinite(self) -> bool {
        !self.is_definite()
    }

    /// All six values in canonical order (T > P > U > D > F, then C).
    pub fn all() -> [TruthValue; 6] {
        [
            TruthValue::True,
            TruthValue::Probable,
            TruthValue::Unknown,
            TruthValue::Doubtful,
            TruthValue::False,
            TruthValue::Contradictory,
        ]
    }
}

/// A piece of evidence for or against a proposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// Where the evidence comes from.
    pub source: String,
    /// What the evidence claims.
    pub claim: String,
}

/// A belief held by an agent, combining a proposition with its truth value,
/// confidence, and supporting/contradicting evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    /// The proposition being evaluated.
    pub proposition: String,
    /// The current truth value of the belief.
    pub truth_value: TruthValue,
    /// Confidence level in the assessment (0.0–1.0).
    pub confidence: f64,
    /// Evidence supporting the proposition.
    pub supporting: Vec<EvidenceItem>,
    /// Evidence contradicting the proposition.
    pub contradicting: Vec<EvidenceItem>,
}

/// The suggested action an agent should take given a belief state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedAction {
    /// Proceed — the belief is verified (T).
    Proceed,
    /// Proceed with caveat — evidence leans true but is not conclusive (P).
    ProceedWithCaveat,
    /// The belief is refuted; do not proceed (F).
    DoNotProceed,
    /// Evidence conflicts, likely should not proceed (D).
    LikelyDoNotProceed,
    /// Gather more evidence before deciding (U).
    GatherEvidence,
    /// Escalate to a human or deeper investigation (C).
    Escalate,
}

impl BeliefState {
    /// Create a new belief state.
    pub fn new(proposition: impl Into<String>, truth_value: TruthValue, confidence: f64) -> Self {
        Self {
            proposition: proposition.into(),
            truth_value,
            confidence: confidence.clamp(0.0, 1.0),
            supporting: Vec::new(),
            contradicting: Vec::new(),
        }
    }

    /// Add supporting evidence and potentially update the truth value.
    pub fn add_supporting(&mut self, item: EvidenceItem) {
        self.supporting.push(item);
        self.recompute();
    }

    /// Add contradicting evidence and potentially update the truth value.
    pub fn add_contradicting(&mut self, item: EvidenceItem) {
        self.contradicting.push(item);
        self.recompute();
    }

    /// Update the truth value based on current evidence balance.
    pub fn update(&mut self) {
        self.recompute();
    }

    /// Suggest an action based on the current truth value.
    pub fn resolve(&self) -> ResolvedAction {
        match self.truth_value {
            TruthValue::True => ResolvedAction::Proceed,
            TruthValue::Probable => ResolvedAction::ProceedWithCaveat,
            TruthValue::Unknown => ResolvedAction::GatherEvidence,
            TruthValue::Doubtful => ResolvedAction::LikelyDoNotProceed,
            TruthValue::False => ResolvedAction::DoNotProceed,
            TruthValue::Contradictory => ResolvedAction::Escalate,
        }
    }

    // Internal: recompute truth value from evidence counts.
    fn recompute(&mut self) {
        let s = self.supporting.len();
        let c = self.contradicting.len();
        if s == 0 && c == 0 {
            // No evidence — leave as-is (caller set the initial value).
            return;
        }
        if s > 0 && c > 0 {
            self.truth_value = TruthValue::Contradictory;
        } else if s > 0 {
            self.truth_value = TruthValue::True;
        } else {
            self.truth_value = TruthValue::False;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TruthValue::*;

    // The full 6×6 AND/OR conformance tables live once, on the canonical
    // `ix_types::Hexavalent` (see `ix_types::hexavalent_tests`). These tests
    // cover only what is governance-specific: the wire mapping, the
    // BeliefState/ResolvedAction behaviour, and the guarantee that this
    // adapter's algebra *is* the canonical one (no second truth table).

    // ── Display / wire symbol ────────────────────────────────────────
    #[test]
    fn display_uses_single_letter_symbols() {
        assert_eq!(format!("{}", True), "T");
        assert_eq!(format!("{}", Probable), "P");
        assert_eq!(format!("{}", Unknown), "U");
        assert_eq!(format!("{}", Doubtful), "D");
        assert_eq!(format!("{}", False), "F");
        assert_eq!(format!("{}", Contradictory), "C");
    }

    // ── Algebra delegates to the canonical ix_types::Hexavalent ──────
    #[test]
    fn algebra_matches_canonical_for_every_pair() {
        for &a in TruthValue::all().iter() {
            for &b in TruthValue::all().iter() {
                assert_eq!(a.and(b).to_hex(), a.to_hex().and(b.to_hex()));
                assert_eq!(a.or(b).to_hex(), a.to_hex().or(b.to_hex()));
                assert_eq!(a.implies(b).to_hex(), a.to_hex().implies(b.to_hex()));
            }
            assert_eq!((!a).to_hex(), a.to_hex().not());
        }
    }

    /// Regression guard for the drift this unification fixed: the old
    /// hand-written governance OR table gave `P∨U = P` and `P∨D = U`; the
    /// canonical De Morgan table gives `P∨U = U` and `P∨D = P`.
    #[test]
    fn or_table_is_the_canonical_de_morgan_one() {
        assert_eq!(Probable.or(Unknown), Unknown);
        assert_eq!(Probable.or(Doubtful), Probable);
        assert_eq!(!Probable, Doubtful);
        assert_eq!(!Doubtful, Probable);
    }

    #[test]
    fn not_is_an_involution() {
        for &v in TruthValue::all().iter() {
            assert_eq!(!!v, v, "!!{v} should be {v}");
        }
    }

    #[test]
    fn all_returns_six_in_canonical_order() {
        assert_eq!(
            TruthValue::all(),
            [True, Probable, Unknown, Doubtful, False, Contradictory]
        );
    }

    // ── BeliefState ──────────────────────────────────────────────────
    #[test]
    fn belief_new() {
        let b = BeliefState::new("test prop", Unknown, 0.5);
        assert_eq!(b.proposition, "test prop");
        assert_eq!(b.truth_value, Unknown);
        assert!((b.confidence - 0.5).abs() < f64::EPSILON);
        assert!(b.supporting.is_empty());
        assert!(b.contradicting.is_empty());
    }

    #[test]
    fn belief_confidence_clamped() {
        let b = BeliefState::new("x", True, 1.5);
        assert!((b.confidence - 1.0).abs() < f64::EPSILON);
        let b2 = BeliefState::new("x", True, -0.5);
        assert!((b2.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn belief_update_supporting() {
        let mut b = BeliefState::new("api is stable", Unknown, 0.5);
        b.add_supporting(EvidenceItem {
            source: "docs".into(),
            claim: "stable since v2".into(),
        });
        assert_eq!(b.truth_value, True);
    }

    #[test]
    fn belief_update_contradicting() {
        let mut b = BeliefState::new("api is stable", Unknown, 0.5);
        b.add_contradicting(EvidenceItem {
            source: "tests".into(),
            claim: "3 endpoints broken".into(),
        });
        assert_eq!(b.truth_value, False);
    }

    #[test]
    fn belief_update_contradictory() {
        let mut b = BeliefState::new("api is stable", Unknown, 0.5);
        b.add_supporting(EvidenceItem {
            source: "docs".into(),
            claim: "stable since v2".into(),
        });
        b.add_contradicting(EvidenceItem {
            source: "tests".into(),
            claim: "3 endpoints broken".into(),
        });
        assert_eq!(b.truth_value, Contradictory);
    }

    #[test]
    fn belief_resolve_covers_all_six() {
        assert_eq!(
            BeliefState::new("x", True, 0.9).resolve(),
            ResolvedAction::Proceed
        );
        assert_eq!(
            BeliefState::new("x", Probable, 0.8).resolve(),
            ResolvedAction::ProceedWithCaveat
        );
        assert_eq!(
            BeliefState::new("x", Unknown, 0.5).resolve(),
            ResolvedAction::GatherEvidence
        );
        assert_eq!(
            BeliefState::new("x", Doubtful, 0.5).resolve(),
            ResolvedAction::LikelyDoNotProceed
        );
        assert_eq!(
            BeliefState::new("x", False, 0.9).resolve(),
            ResolvedAction::DoNotProceed
        );
        assert_eq!(
            BeliefState::new("x", Contradictory, 0.6).resolve(),
            ResolvedAction::Escalate
        );
    }

    /// Serde derive round-trips by variant name (now `Doubtful`, formerly
    /// `Disputed`). The single-letter wire form is a separate concern
    /// (see `feedback`'s custom serializer + the `Display` impl).
    #[test]
    fn serde_derive_round_trips_by_variant_name() {
        assert_eq!(serde_json::to_string(&Probable).unwrap(), "\"Probable\"");
        assert_eq!(serde_json::to_string(&Doubtful).unwrap(), "\"Doubtful\"");
        let back: TruthValue = serde_json::from_str("\"Doubtful\"").unwrap();
        assert_eq!(back, Doubtful);
    }
}
