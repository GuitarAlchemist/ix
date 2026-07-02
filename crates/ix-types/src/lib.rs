//! Shared type lattice for the ix skill registry.
//!
//! Every `#[ix_skill]`-annotated function accepts and returns values that
//! serialize through this crate's [`Value`] enum. Socket compatibility between
//! pipeline nodes is checked via [`SocketType`], and governance verdicts flow
//! through [`Tetravalent`]. The [`FromValue`] / [`IntoValue`] trait pair is the
//! glue between native Rust types and the universal `Value`.

use ndarray::{Array1, Array2};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod traits;
pub use traits::{FromValue, IntoValue};

/// Six-valued truth for governance beliefs — the ecosystem-wide standard
/// defined in `governance/demerzel/logic/hexavalent-logic.md`.
///
/// Extends classical tetravalent logic (T/F/U/C) with two evidential gradient
/// values that separate **direction of evidence** from **sufficiency**:
///
/// | Symbol | Name          | Meaning                                 |
/// |--------|---------------|-----------------------------------------|
/// | T      | True          | Verified with sufficient evidence       |
/// | P      | Probable      | Evidence leans true, not yet verified   |
/// | U      | Unknown       | Insufficient evidence to determine      |
/// | D      | Doubtful      | Evidence leans false, not yet refuted   |
/// | F      | False         | Refuted with sufficient evidence        |
/// | C      | Contradictory | Evidence supports both true and false   |
///
/// Serialized as the single-letter symbol (`"T"` / `"P"` / …) for cross-repo
/// wire compatibility with Demerzel's `hexavalent-state.schema.json`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
pub enum Hexavalent {
    #[serde(rename = "T")]
    True,
    #[serde(rename = "P")]
    Probable,
    #[serde(rename = "U")]
    Unknown,
    #[serde(rename = "D")]
    Doubtful,
    #[serde(rename = "F")]
    False,
    #[serde(rename = "C")]
    Contradictory,
}

impl Hexavalent {
    /// Single-letter symbol (`T`/`P`/`U`/`D`/`F`/`C`) — wire format.
    pub const fn symbol(self) -> char {
        match self {
            Hexavalent::True => 'T',
            Hexavalent::Probable => 'P',
            Hexavalent::Unknown => 'U',
            Hexavalent::Doubtful => 'D',
            Hexavalent::False => 'F',
            Hexavalent::Contradictory => 'C',
        }
    }

    /// Hexavalent NOT: T↔F, P↔D, U→U, C→C.
    pub const fn not(self) -> Self {
        use Hexavalent::*;
        match self {
            True => False,
            Probable => Doubtful,
            Unknown => Unknown,
            Doubtful => Probable,
            False => True,
            Contradictory => Contradictory,
        }
    }

    /// Hexavalent AND — per `hexavalent-logic.md` truth table.
    /// F absorbs everything; P demotes T to P; D demotes T/P to D.
    pub const fn and(self, other: Self) -> Self {
        use Hexavalent::*;
        match (self, other) {
            // F is absorbing
            (False, _) | (_, False) => False,
            // Anything AND T = itself
            (x, True) => x,
            (True, x) => x,
            // P row/col
            (Probable, Probable) => Probable,
            (Probable, Unknown) | (Unknown, Probable) => Unknown,
            (Probable, Doubtful) | (Doubtful, Probable) => Doubtful,
            (Probable, Contradictory) | (Contradictory, Probable) => Contradictory,
            // U row/col (excluding pairs handled above)
            (Unknown, Unknown) => Unknown,
            (Unknown, Doubtful) | (Doubtful, Unknown) => Unknown,
            (Unknown, Contradictory) | (Contradictory, Unknown) => Contradictory,
            // D row/col
            (Doubtful, Doubtful) => Doubtful,
            (Doubtful, Contradictory) | (Contradictory, Doubtful) => Contradictory,
            // C row/col
            (Contradictory, Contradictory) => Contradictory,
        }
    }

    /// Hexavalent OR — per `hexavalent-logic.md` truth table (derived from
    /// AND via De Morgan: `a OR b = NOT(NOT a AND NOT b)`).
    pub const fn or(self, other: Self) -> Self {
        self.not().and(other.not()).not()
    }

    /// All six values in canonical lattice order (T, P, U, D, F, C).
    pub const fn all() -> [Hexavalent; 6] {
        [
            Hexavalent::True,
            Hexavalent::Probable,
            Hexavalent::Unknown,
            Hexavalent::Doubtful,
            Hexavalent::False,
            Hexavalent::Contradictory,
        ]
    }

    /// Is this a "definite" value (T or F)?
    pub const fn is_definite(self) -> bool {
        matches!(self, Hexavalent::True | Hexavalent::False)
    }

    /// Is this value *not* definite (P, U, D, C)? The complement of
    /// [`is_definite`](Hexavalent::is_definite).
    pub const fn is_indefinite(self) -> bool {
        !self.is_definite()
    }

    /// Does this value carry evidential direction (P, D, T, F)?
    pub const fn is_directed(self) -> bool {
        matches!(
            self,
            Hexavalent::True | Hexavalent::Probable | Hexavalent::Doubtful | Hexavalent::False
        )
    }

    /// Parse from the single-letter wire symbol (`T`/`P`/`U`/`D`/`F`/`C`).
    pub const fn from_char(c: char) -> Option<Self> {
        Some(match c {
            'T' => Hexavalent::True,
            'P' => Hexavalent::Probable,
            'U' => Hexavalent::Unknown,
            'D' => Hexavalent::Doubtful,
            'F' => Hexavalent::False,
            'C' => Hexavalent::Contradictory,
            _ => return None,
        })
    }

    /// Single-letter symbol as a `&'static str` (`"T"`/`"P"`/…). Mirrors
    /// [`symbol`](Hexavalent::symbol) for callers that want a string slice.
    pub const fn as_str(self) -> &'static str {
        match self {
            Hexavalent::True => "T",
            Hexavalent::Probable => "P",
            Hexavalent::Unknown => "U",
            Hexavalent::Doubtful => "D",
            Hexavalent::False => "F",
            Hexavalent::Contradictory => "C",
        }
    }

    /// Hexavalent implication: `a → b = (¬a) ∨ b`.
    pub const fn implies(self, other: Self) -> Self {
        self.not().or(other)
    }

    /// Hexavalent XOR: `a ⊕ b = (a ∨ b) ∧ ¬(a ∧ b)`.
    pub const fn xor(self, other: Self) -> Self {
        self.or(other).and(self.and(other).not())
    }

    /// Hexavalent equivalence: `a ↔ b = (a → b) ∧ (b → a)`.
    pub const fn equiv(self, other: Self) -> Self {
        self.implies(other).and(other.implies(self))
    }

    /// This value's evidential [`Polarity`] — the direction of evidence,
    /// independent of sufficiency. `T`/`P` lean positive, `D`/`F` lean
    /// negative, `U` is neutral, `C` is already contradictory.
    pub const fn polarity(self) -> Polarity {
        match self {
            Hexavalent::True | Hexavalent::Probable => Polarity::Positive,
            Hexavalent::Doubtful | Hexavalent::False => Polarity::Negative,
            Hexavalent::Unknown => Polarity::Neutral,
            Hexavalent::Contradictory => Polarity::Contradictory,
        }
    }

    /// Two values *conflict* iff one leans true and the other leans false.
    /// Neutral (`U`) and already-contradictory (`C`) never pairwise-conflict.
    pub const fn conflicts(self, other: Self) -> bool {
        matches!(
            (self.polarity(), other.polarity()),
            (Polarity::Positive, Polarity::Negative) | (Polarity::Negative, Polarity::Positive)
        )
    }
}

/// A truth value's evidential direction — the sign of the evidence, separated
/// from its sufficiency. Used by contradiction detection and confidence-weighted
/// voting (see [`weighted`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum Polarity {
    /// Leans true: `T` (True) or `P` (Probable).
    Positive,
    /// Leans false: `D` (Doubtful) or `F` (False).
    Negative,
    /// No evidential direction: `U` (Unknown).
    Neutral,
    /// Already contradictory: `C` — flagged at the source, not re-derived pairwise.
    Contradictory,
}

/// Confidence-weighted hexavalent vote over `(value, confidence)` pairs.
///
/// Returns `(argmax_value, avg_confidence)`: the value carrying the greatest
/// summed confidence, and the mean confidence across all votes. Ties break by an
/// **escalation-favoring** order `C > U > F > D > T > P`, so an unresolved or
/// contradictory reading wins over a confident-but-tied positive. Empty input
/// resolves to `U` (no evidence).
pub fn weighted(votes: &[(Hexavalent, f64)]) -> (Hexavalent, f64) {
    use std::collections::HashMap;
    if votes.is_empty() {
        return (Hexavalent::Unknown, 0.0);
    }
    let mut buckets: HashMap<Hexavalent, f64> = HashMap::new();
    let mut total = 0.0;
    for (tv, conf) in votes {
        *buckets.entry(*tv).or_insert(0.0) += *conf;
        total += *conf;
    }
    let avg = total / votes.len() as f64;
    // All-zero confidence carries no evidence: without this guard the argmax
    // scan below would match the first tie-break entry (`C`) at weight 0.0 and
    // report Contradictory from votes that assert nothing.
    if total == 0.0 {
        return (Hexavalent::Unknown, 0.0);
    }
    let order = [
        Hexavalent::Contradictory,
        Hexavalent::Unknown,
        Hexavalent::False,
        Hexavalent::Doubtful,
        Hexavalent::True,
        Hexavalent::Probable,
    ];
    let max_weight = buckets.values().copied().fold(0.0_f64, f64::max);
    let winner = order
        .into_iter()
        .find(|tv| buckets.get(tv).copied().unwrap_or(0.0) == max_weight)
        .unwrap_or(Hexavalent::Unknown);
    (winner, avg)
}

impl std::fmt::Display for Hexavalent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Hexavalent::True => "T",
            Hexavalent::Probable => "P",
            Hexavalent::Unknown => "U",
            Hexavalent::Doubtful => "D",
            Hexavalent::False => "F",
            Hexavalent::Contradictory => "C",
        })
    }
}

/// Universal value lattice. All skill inputs and outputs are carried through
/// this enum when they cross the registry boundary.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "data")]
pub enum Value {
    Null,
    Scalar(f64),
    Integer(i64),
    Bool(bool),
    Text(String),
    Bytes(Vec<u8>),
    Vector(Vec<f64>),
    Matrix {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
    Belief(Hexavalent),
    Json(serde_json::Value),
}

/// Static socket-type tag compared at both registration and runtime for edge
/// compatibility in the visual pipeline editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum SocketType {
    Any,
    Scalar,
    Integer,
    Bool,
    Text,
    Bytes,
    Vector,
    Matrix,
    Belief,
    Json,
}

impl SocketType {
    /// Can a socket of type `self` feed into a socket of type `other`?
    ///
    /// `Any` matches everything. Exact matches pass. Safe widening is
    /// allowed: `Scalar → Vector` (broadcast), `Vector → Matrix` (row).
    pub const fn compatible_with(self, other: SocketType) -> bool {
        use SocketType::*;
        if matches!(self, Any) || matches!(other, Any) {
            return true;
        }
        if self as u8 == other as u8 {
            return true;
        }
        matches!(
            (self, other),
            (Scalar, Vector) | (Vector, Matrix) | (Integer, Scalar)
        )
    }
}

/// Type-mismatch error raised when a `Value` cannot be decoded as the expected
/// native type by a skill adapter.
#[derive(Debug, Error)]
#[error("type mismatch: expected {expected:?}, got {actual}")]
pub struct TypeError {
    pub expected: SocketType,
    pub actual: &'static str,
}

impl Value {
    /// Human-readable tag of the active variant — used in error messages.
    pub fn tag(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Scalar(_) => "scalar",
            Value::Integer(_) => "integer",
            Value::Bool(_) => "bool",
            Value::Text(_) => "text",
            Value::Bytes(_) => "bytes",
            Value::Vector(_) => "vector",
            Value::Matrix { .. } => "matrix",
            Value::Belief(_) => "belief",
            Value::Json(_) => "json",
        }
    }
}

// ---------------------------------------------------------------------------
// Newtypes wrapping ndarray for pipeline transport. Skills should use these
// in their signatures rather than raw ndarray types so the macro can emit
// concrete `FromValue`/`IntoValue` calls without orphan-rule issues.
// ---------------------------------------------------------------------------

/// Owned 1-D numeric vector wrapping `ndarray::Array1<f64>`.
#[derive(Debug, Clone)]
pub struct IxVector(pub Array1<f64>);

impl IxVector {
    pub fn new(data: Vec<f64>) -> Self {
        Self(Array1::from_vec(data))
    }
    pub fn into_inner(self) -> Array1<f64> {
        self.0
    }
}

impl From<Array1<f64>> for IxVector {
    fn from(a: Array1<f64>) -> Self {
        Self(a)
    }
}

impl From<Vec<f64>> for IxVector {
    fn from(v: Vec<f64>) -> Self {
        Self::new(v)
    }
}

/// Owned 2-D numeric matrix wrapping `ndarray::Array2<f64>`.
#[derive(Debug, Clone)]
pub struct IxMatrix(pub Array2<f64>);

impl IxMatrix {
    /// Construct from row-major flat data.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, TypeError> {
        Array2::from_shape_vec((rows, cols), data)
            .map(Self)
            .map_err(|_| TypeError {
                expected: SocketType::Matrix,
                actual: "matrix: row×col != data.len()",
            })
    }
    pub fn into_inner(self) -> Array2<f64> {
        self.0
    }
}

impl From<Array2<f64>> for IxMatrix {
    fn from(a: Array2<f64>) -> Self {
        Self(a)
    }
}

#[cfg(test)]
mod hexavalent_tests {
    use super::Hexavalent::*;
    use super::{weighted, Hexavalent, Polarity};

    /// Canonical 6×6 AND table (rows/cols in lattice order T,P,U,D,F,C).
    /// F is absorbing; T is identity; C absorbs everything except F.
    #[test]
    fn and_truth_table() {
        let expected: [[Hexavalent; 6]; 6] = [
            [True, Probable, Unknown, Doubtful, False, Contradictory],
            [Probable, Probable, Unknown, Doubtful, False, Contradictory],
            [Unknown, Unknown, Unknown, Unknown, False, Contradictory],
            [Doubtful, Doubtful, Unknown, Doubtful, False, Contradictory],
            [False, False, False, False, False, False],
            [Contradictory, Contradictory, Contradictory, Contradictory, False, Contradictory],
        ];
        for (i, &a) in Hexavalent::all().iter().enumerate() {
            for (j, &b) in Hexavalent::all().iter().enumerate() {
                assert_eq!(a.and(b), expected[i][j], "AND({a}, {b})");
            }
        }
    }

    /// Canonical 6×6 OR table — derived from AND via De Morgan, so it can never
    /// silently drift from AND. Note the De Morgan results at the P/D cells
    /// (`P∨U = U`, `P∨D = P`) that a hand-written table historically got wrong.
    #[test]
    fn or_truth_table() {
        let expected: [[Hexavalent; 6]; 6] = [
            [True, True, True, True, True, True],
            [True, Probable, Unknown, Probable, Probable, Contradictory],
            [True, Unknown, Unknown, Unknown, Unknown, Contradictory],
            [True, Probable, Unknown, Doubtful, Doubtful, Contradictory],
            [True, Probable, Unknown, Doubtful, False, Contradictory],
            [True, Contradictory, Contradictory, Contradictory, Contradictory, Contradictory],
        ];
        for (i, &a) in Hexavalent::all().iter().enumerate() {
            for (j, &b) in Hexavalent::all().iter().enumerate() {
                assert_eq!(a.or(b), expected[i][j], "OR({a}, {b})");
            }
        }
    }

    #[test]
    fn not_is_an_involution() {
        for &v in Hexavalent::all().iter() {
            assert_eq!(v.not().not(), v, "¬¬{v}");
        }
    }

    #[test]
    fn and_or_are_commutative() {
        for &a in Hexavalent::all().iter() {
            for &b in Hexavalent::all().iter() {
                assert_eq!(a.and(b), b.and(a), "AND comm {a},{b}");
                assert_eq!(a.or(b), b.or(a), "OR comm {a},{b}");
            }
        }
    }

    #[test]
    fn implies_xor_equiv_on_classical_subset() {
        // Classical T/F/U cases the governance Karnaugh minimizer relies on.
        assert_eq!(True.implies(False), False);
        assert_eq!(False.implies(True), True);
        assert_eq!(Unknown.implies(False), Unknown);
        assert_eq!(True.xor(False), True);
        assert_eq!(False.xor(False), False);
        assert_eq!(True.equiv(False), False);
        assert_eq!(Unknown.equiv(Unknown), Unknown);
    }

    #[test]
    fn from_char_round_trips_symbol() {
        for &v in Hexavalent::all().iter() {
            assert_eq!(Hexavalent::from_char(v.symbol()), Some(v));
            assert_eq!(v.as_str(), &v.symbol().to_string());
        }
        assert_eq!(Hexavalent::from_char('Z'), None);
    }

    #[test]
    fn polarity_and_conflicts() {
        assert_eq!(True.polarity(), Polarity::Positive);
        assert_eq!(Probable.polarity(), Polarity::Positive);
        assert_eq!(Doubtful.polarity(), Polarity::Negative);
        assert_eq!(Unknown.polarity(), Polarity::Neutral);
        assert_eq!(Contradictory.polarity(), Polarity::Contradictory);
        assert!(True.conflicts(False));
        assert!(Probable.conflicts(Doubtful));
        assert!(!Probable.conflicts(True));
        assert!(!Unknown.conflicts(False));
        assert!(!Contradictory.conflicts(True));
        // Symmetric.
        for &a in Hexavalent::all().iter() {
            for &b in Hexavalent::all().iter() {
                assert_eq!(a.conflicts(b), b.conflicts(a));
            }
        }
    }

    #[test]
    fn weighted_picks_argmax_and_breaks_toward_escalation() {
        let (tv, avg) = weighted(&[(True, 0.9), (True, 0.8), (False, 0.5)]);
        assert_eq!(tv, True);
        assert!((avg - (2.2 / 3.0)).abs() < 1e-9);
        // Tie T vs F → escalation order prefers F.
        assert_eq!(weighted(&[(True, 0.5), (False, 0.5)]).0, False);
        // C beats everything on a tie.
        assert_eq!(weighted(&[(Contradictory, 0.5), (Unknown, 0.5)]).0, Contradictory);
        // Empty → Unknown.
        assert_eq!(weighted(&[]), (Unknown, 0.0));
        // All-zero confidence asserts nothing → Unknown, never Contradictory.
        assert_eq!(weighted(&[(True, 0.0), (False, 0.0)]), (Unknown, 0.0));
        assert_eq!(weighted(&[(True, 0.0)]), (Unknown, 0.0));
    }
}
