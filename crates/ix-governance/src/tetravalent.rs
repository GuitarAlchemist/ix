use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Not;

/// A six-valued (hexavalent) truth value for agent reasoning under uncertainty.
///
/// Extends the original tetravalent system (T/F/U/C) with two epistemic
/// intermediate values:
///
/// - **Probable (P)**: evidence leans true but is not conclusive.
/// - **Disputed (D)**: credible evidence on both sides, actively contested
///   (epistemic disagreement — unlike Contradictory which is logical).
///
/// Ordering on the "truth axis": T > P > U > D > F.
/// C (Contradictory) is orthogonal — it indicates a logical inconsistency
/// that cannot resolve with more evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TruthValue {
    /// Verified with sufficient evidence.
    True,
    /// Evidence leans true but not conclusive.
    Probable,
    /// Insufficient evidence to determine.
    Unknown,
    /// Credible evidence on both sides, actively contested (epistemic).
    Disputed,
    /// Refuted with sufficient evidence.
    False,
    /// Evidence supports both True and False (logical inconsistency).
    Contradictory,
}

impl fmt::Display for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TruthValue::True => write!(f, "T"),
            TruthValue::Probable => write!(f, "P"),
            TruthValue::Unknown => write!(f, "U"),
            TruthValue::Disputed => write!(f, "D"),
            TruthValue::False => write!(f, "F"),
            TruthValue::Contradictory => write!(f, "C"),
        }
    }
}

impl Not for TruthValue {
    type Output = Self;

    /// Hexavalent NOT: T↔F, P↔D, U→U, C→C.
    fn not(self) -> Self {
        match self {
            TruthValue::True => TruthValue::False,
            TruthValue::Probable => TruthValue::Disputed,
            TruthValue::Unknown => TruthValue::Unknown,
            TruthValue::Disputed => TruthValue::Probable,
            TruthValue::False => TruthValue::True,
            TruthValue::Contradictory => TruthValue::Contradictory,
        }
    }
}

impl TruthValue {
    /// Hexavalent AND truth table.
    ///
    /// Design principles:
    /// - F absorbs everything (F AND x = F).
    /// - C absorbs everything except F (logical inconsistency propagates).
    /// - T is the identity (T AND x = x).
    /// - P AND P = P (probable stays probable).
    /// - P AND U = U (unknown weakens probable).
    /// - P AND D = D (dispute weakens probable).
    /// - D AND D = D, D AND U = U.
    pub fn and(self, other: Self) -> Self {
        use TruthValue::*;
        match (self, other) {
            // F absorbs everything
            (False, _) | (_, False) => False,
            // C absorbs everything except F
            (Contradictory, _) | (_, Contradictory) => Contradictory,
            // T is identity
            (True, x) | (x, True) => x,
            // P interactions
            (Probable, Probable) => Probable,
            (Probable, Unknown) | (Unknown, Probable) => Unknown,
            (Probable, Disputed) | (Disputed, Probable) => Disputed,
            // D interactions
            (Disputed, Disputed) => Disputed,
            (Disputed, Unknown) | (Unknown, Disputed) => Unknown,
            // U with U
            (Unknown, Unknown) => Unknown,
        }
    }

    /// Hexavalent OR truth table.
    ///
    /// Design principles:
    /// - T absorbs everything (T OR x = T).
    /// - C absorbs everything except T (logical inconsistency propagates).
    /// - F is the identity (F OR x = x).
    /// - P OR P = P (probable stays probable).
    /// - P OR U = P (probable strengthens unknown).
    /// - P OR D = U (dispute + probable = unresolved).
    /// - D OR D = D, D OR U = U.
    pub fn or(self, other: Self) -> Self {
        use TruthValue::*;
        match (self, other) {
            // T absorbs everything
            (True, _) | (_, True) => True,
            // C absorbs everything except T
            (Contradictory, _) | (_, Contradictory) => Contradictory,
            // F is identity
            (False, x) | (x, False) => x,
            // P interactions
            (Probable, Probable) => Probable,
            (Probable, Unknown) | (Unknown, Probable) => Probable,
            (Probable, Disputed) | (Disputed, Probable) => Unknown,
            // D interactions
            (Disputed, Disputed) => Disputed,
            (Disputed, Unknown) | (Unknown, Disputed) => Unknown,
            // U with U
            (Unknown, Unknown) => Unknown,
        }
    }

    /// Hexavalent implication: A -> B = (NOT A) OR B.
    pub fn implies(self, other: Self) -> Self {
        (!self).or(other)
    }

    /// Hexavalent XOR: XOR(A, B) = (A OR B) AND NOT(A AND B).
    pub fn xor(self, other: Self) -> Self {
        self.or(other).and(!(self.and(other)))
    }

    /// Hexavalent equivalence: A <-> B = (A -> B) AND (B -> A).
    pub fn equiv(self, other: Self) -> Self {
        self.implies(other).and(other.implies(self))
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
            TruthValue::Disputed,
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
            TruthValue::Disputed => ResolvedAction::LikelyDoNotProceed,
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

    // ── NOT truth table (original 4 + hexavalent extensions) ─────────
    #[test]
    fn not_true() {
        assert_eq!(!True, False);
    }
    #[test]
    fn not_false() {
        assert_eq!(!False, True);
    }
    #[test]
    fn not_unknown() {
        assert_eq!(!Unknown, Unknown);
    }
    #[test]
    fn not_contradictory() {
        assert_eq!(!Contradictory, Contradictory);
    }

    // ── AND truth table (original 4×4 subset — backward compat) ─────
    #[test]
    fn and_truth_table() {
        let expected: [[TruthValue; 4]; 4] = [
            // T AND {T,F,U,C}
            [True, False, Unknown, Contradictory],
            // F AND {T,F,U,C}
            [False, False, False, False],
            // U AND {T,F,U,C}
            [Unknown, False, Unknown, Contradictory],
            // C AND {T,F,U,C}
            [Contradictory, False, Contradictory, Contradictory],
        ];
        let values = [True, False, Unknown, Contradictory];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                assert_eq!(
                    a.and(b),
                    expected[i][j],
                    "AND({}, {}) should be {} but got {}",
                    a,
                    b,
                    expected[i][j],
                    a.and(b)
                );
            }
        }
    }

    // ── OR truth table (original 4×4 subset — backward compat) ──────
    #[test]
    fn or_truth_table() {
        let expected: [[TruthValue; 4]; 4] = [
            // T OR {T,F,U,C}
            [True, True, True, True],
            // F OR {T,F,U,C}
            [True, False, Unknown, Contradictory],
            // U OR {T,F,U,C}
            [True, Unknown, Unknown, Contradictory],
            // C OR {T,F,U,C}
            [True, Contradictory, Contradictory, Contradictory],
        ];
        let values = [True, False, Unknown, Contradictory];
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                assert_eq!(
                    a.or(b),
                    expected[i][j],
                    "OR({}, {}) should be {} but got {}",
                    a,
                    b,
                    expected[i][j],
                    a.or(b)
                );
            }
        }
    }

    // ── Display ──────────────────────────────────────────────────────
    #[test]
    fn display() {
        assert_eq!(format!("{}", True), "T");
        assert_eq!(format!("{}", Probable), "P");
        assert_eq!(format!("{}", Unknown), "U");
        assert_eq!(format!("{}", Disputed), "D");
        assert_eq!(format!("{}", False), "F");
        assert_eq!(format!("{}", Contradictory), "C");
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
    fn belief_resolve() {
        assert_eq!(
            BeliefState::new("x", True, 0.9).resolve(),
            ResolvedAction::Proceed
        );
        assert_eq!(
            BeliefState::new("x", False, 0.9).resolve(),
            ResolvedAction::DoNotProceed
        );
        assert_eq!(
            BeliefState::new("x", Unknown, 0.5).resolve(),
            ResolvedAction::GatherEvidence
        );
        assert_eq!(
            BeliefState::new("x", Contradictory, 0.6).resolve(),
            ResolvedAction::Escalate
        );
    }

    // ══════════════════════════════════════════════════════════════════
    // Hexavalent extension tests (P and D)
    // ══════════════════════════════════════════════════════════════════

    #[test]
    fn hexavalent_not() {
        assert_eq!(!Probable, Disputed);
        assert_eq!(!Disputed, Probable);
    }

    /// Full 6x6 AND truth table covering all P and D combinations.
    #[test]
    fn hexavalent_and_truth_table() {
        // Columns: T, P, U, D, F, C
        let expected: [[TruthValue; 6]; 6] = [
            // T AND {T, P, U, D, F, C}
            [True, Probable, Unknown, Disputed, False, Contradictory],
            // P AND {T, P, U, D, F, C}
            [Probable, Probable, Unknown, Disputed, False, Contradictory],
            // U AND {T, P, U, D, F, C}
            [Unknown, Unknown, Unknown, Unknown, False, Contradictory],
            // D AND {T, P, U, D, F, C}
            [Disputed, Disputed, Unknown, Disputed, False, Contradictory],
            // F AND {T, P, U, D, F, C}
            [False, False, False, False, False, False],
            // C AND {T, P, U, D, F, C}
            [
                Contradictory,
                Contradictory,
                Contradictory,
                Contradictory,
                False,
                Contradictory,
            ],
        ];
        let values = TruthValue::all();
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                assert_eq!(
                    a.and(b),
                    expected[i][j],
                    "AND({}, {}) should be {} but got {}",
                    a,
                    b,
                    expected[i][j],
                    a.and(b)
                );
            }
        }
    }

    /// Full 6x6 OR truth table covering all P and D combinations.
    #[test]
    fn hexavalent_or_truth_table() {
        // Columns: T, P, U, D, F, C
        let expected: [[TruthValue; 6]; 6] = [
            // T OR {T, P, U, D, F, C}
            [True, True, True, True, True, True],
            // P OR {T, P, U, D, F, C}
            [True, Probable, Probable, Unknown, Probable, Contradictory],
            // U OR {T, P, U, D, F, C}
            [True, Probable, Unknown, Unknown, Unknown, Contradictory],
            // D OR {T, P, U, D, F, C}
            [True, Unknown, Unknown, Disputed, Disputed, Contradictory],
            // F OR {T, P, U, D, F, C}
            [True, Probable, Unknown, Disputed, False, Contradictory],
            // C OR {T, P, U, D, F, C}
            [
                True,
                Contradictory,
                Contradictory,
                Contradictory,
                Contradictory,
                Contradictory,
            ],
        ];
        let values = TruthValue::all();
        for (i, &a) in values.iter().enumerate() {
            for (j, &b) in values.iter().enumerate() {
                assert_eq!(
                    a.or(b),
                    expected[i][j],
                    "OR({}, {}) should be {} but got {}",
                    a,
                    b,
                    expected[i][j],
                    a.or(b)
                );
            }
        }
    }

    /// P resolves to ProceedWithCaveat, D resolves to LikelyDoNotProceed.
    #[test]
    fn hexavalent_resolve() {
        assert_eq!(
            BeliefState::new("x", Probable, 0.8).resolve(),
            ResolvedAction::ProceedWithCaveat
        );
        assert_eq!(
            BeliefState::new("x", Disputed, 0.5).resolve(),
            ResolvedAction::LikelyDoNotProceed
        );
    }

    /// Serde round-trip for P and D values.
    #[test]
    fn hexavalent_serde_roundtrip() {
        let p = Probable;
        let d = Disputed;
        let p_json = serde_json::to_string(&p).unwrap();
        let d_json = serde_json::to_string(&d).unwrap();
        assert_eq!(p_json, "\"Probable\"");
        assert_eq!(d_json, "\"Disputed\"");
        let p_back: TruthValue = serde_json::from_str(&p_json).unwrap();
        let d_back: TruthValue = serde_json::from_str(&d_json).unwrap();
        assert_eq!(p_back, Probable);
        assert_eq!(d_back, Disputed);
    }

    /// Verify all() returns 6 values in canonical order.
    #[test]
    fn hexavalent_all_returns_six() {
        let all = TruthValue::all();
        assert_eq!(all.len(), 6);
        assert_eq!(
            all,
            [True, Probable, Unknown, Disputed, False, Contradictory]
        );
    }

    /// NOT is an involution: !!x == x for all 6 values.
    #[test]
    fn hexavalent_not_involution() {
        for &v in TruthValue::all().iter() {
            assert_eq!(!!v, v, "!!{} should be {}", v, v);
        }
    }
}
