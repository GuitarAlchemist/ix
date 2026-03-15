use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::Not;

/// A four-valued truth value for agent reasoning under uncertainty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TruthValue {
    /// Verified with sufficient evidence.
    True,
    /// Refuted with sufficient evidence.
    False,
    /// Insufficient evidence to determine.
    Unknown,
    /// Evidence supports both True and False.
    Contradictory,
}

impl fmt::Display for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TruthValue::True => write!(f, "T"),
            TruthValue::False => write!(f, "F"),
            TruthValue::Unknown => write!(f, "U"),
            TruthValue::Contradictory => write!(f, "C"),
        }
    }
}

impl Not for TruthValue {
    type Output = Self;

    /// Tetravalent NOT: T↔F, U→U, C→C.
    fn not(self) -> Self {
        match self {
            TruthValue::True => TruthValue::False,
            TruthValue::False => TruthValue::True,
            TruthValue::Unknown => TruthValue::Unknown,
            TruthValue::Contradictory => TruthValue::Contradictory,
        }
    }
}

impl TruthValue {
    /// Tetravalent AND: F absorbs everything; among {T,U,C}, U and C propagate.
    pub fn and(self, other: Self) -> Self {
        use TruthValue::*;
        match (self, other) {
            (False, _) | (_, False) => False,
            (True, x) | (x, True) => x,
            (Contradictory, _) | (_, Contradictory) => Contradictory,
            (Unknown, Unknown) => Unknown,
        }
    }

    /// Tetravalent OR: T absorbs everything; among {F,U,C}, U and C propagate.
    pub fn or(self, other: Self) -> Self {
        use TruthValue::*;
        match (self, other) {
            (True, _) | (_, True) => True,
            (False, x) | (x, False) => x,
            (Contradictory, _) | (_, Contradictory) => Contradictory,
            (Unknown, Unknown) => Unknown,
        }
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
    /// Proceed — the belief is verified.
    Proceed,
    /// The belief is refuted; do not proceed.
    DoNotProceed,
    /// Gather more evidence before deciding.
    GatherEvidence,
    /// Escalate to a human or deeper investigation.
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
            TruthValue::False => ResolvedAction::DoNotProceed,
            TruthValue::Unknown => ResolvedAction::GatherEvidence,
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

    // ── NOT truth table ──────────────────────────────────────────────
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

    // ── AND truth table (4×4) ────────────────────────────────────────
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

    // ── OR truth table (4×4) ─────────────────────────────────────────
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
        assert_eq!(format!("{}", False), "F");
        assert_eq!(format!("{}", Unknown), "U");
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
}
