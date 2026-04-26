//! `FromValue` / `IntoValue` — the glue between native Rust types and [`Value`].
//!
//! Every skill argument type must implement [`FromValue`] (Value → T decode)
//! and every skill return type must implement [`IntoValue`] (T → Value encode).
//! Blanket impls are provided for the universal numeric/text scalars plus the
//! `IxVector` / `IxMatrix` newtypes that carry ndarray data.

use crate::{Hexavalent, IxMatrix, IxVector, SocketType, TypeError, Value};

/// Decode a [`Value`] into a concrete Rust type.
pub trait FromValue: Sized {
    const SOCKET: SocketType;
    fn from_value(v: &Value) -> Result<Self, TypeError>;
}

/// Encode a concrete Rust type back into a [`Value`].
pub trait IntoValue {
    const SOCKET: SocketType;
    fn into_value(self) -> Value;
}

// ---------------------------------------------------------------------------
// Scalar impls
// ---------------------------------------------------------------------------

impl FromValue for f64 {
    const SOCKET: SocketType = SocketType::Scalar;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Scalar(x) => Ok(*x),
            Value::Integer(n) => Ok(*n as f64),
            other => Err(TypeError {
                expected: SocketType::Scalar,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for f64 {
    const SOCKET: SocketType = SocketType::Scalar;
    fn into_value(self) -> Value {
        Value::Scalar(self)
    }
}

impl FromValue for i64 {
    const SOCKET: SocketType = SocketType::Integer;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Integer(n) => Ok(*n),
            Value::Scalar(x) => Ok(*x as i64),
            other => Err(TypeError {
                expected: SocketType::Integer,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for i64 {
    const SOCKET: SocketType = SocketType::Integer;
    fn into_value(self) -> Value {
        Value::Integer(self)
    }
}

impl FromValue for usize {
    const SOCKET: SocketType = SocketType::Integer;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Integer(n) if *n >= 0 => Ok(*n as usize),
            Value::Scalar(x) if *x >= 0.0 => Ok(*x as usize),
            other => Err(TypeError {
                expected: SocketType::Integer,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for usize {
    const SOCKET: SocketType = SocketType::Integer;
    fn into_value(self) -> Value {
        Value::Integer(self as i64)
    }
}

impl FromValue for u64 {
    const SOCKET: SocketType = SocketType::Integer;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Integer(n) if *n >= 0 => Ok(*n as u64),
            Value::Scalar(x) if *x >= 0.0 => Ok(*x as u64),
            other => Err(TypeError {
                expected: SocketType::Integer,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for u64 {
    const SOCKET: SocketType = SocketType::Integer;
    fn into_value(self) -> Value {
        Value::Integer(self as i64)
    }
}

impl FromValue for bool {
    const SOCKET: SocketType = SocketType::Bool;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Bool(b) => Ok(*b),
            other => Err(TypeError {
                expected: SocketType::Bool,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for bool {
    const SOCKET: SocketType = SocketType::Bool;
    fn into_value(self) -> Value {
        Value::Bool(self)
    }
}

impl FromValue for String {
    const SOCKET: SocketType = SocketType::Text;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Text(s) => Ok(s.clone()),
            other => Err(TypeError {
                expected: SocketType::Text,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for String {
    const SOCKET: SocketType = SocketType::Text;
    fn into_value(self) -> Value {
        Value::Text(self)
    }
}

impl FromValue for Vec<u8> {
    const SOCKET: SocketType = SocketType::Bytes;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Bytes(b) => Ok(b.clone()),
            other => Err(TypeError {
                expected: SocketType::Bytes,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for Vec<u8> {
    const SOCKET: SocketType = SocketType::Bytes;
    fn into_value(self) -> Value {
        Value::Bytes(self)
    }
}

impl FromValue for Hexavalent {
    const SOCKET: SocketType = SocketType::Belief;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Belief(t) => Ok(*t),
            other => Err(TypeError {
                expected: SocketType::Belief,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for Hexavalent {
    const SOCKET: SocketType = SocketType::Belief;
    fn into_value(self) -> Value {
        Value::Belief(self)
    }
}

impl FromValue for serde_json::Value {
    const SOCKET: SocketType = SocketType::Json;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Json(j) => Ok(j.clone()),
            // Allow any Value → Json by round-tripping through serde.
            other => serde_json::to_value(other).map_err(|_| TypeError {
                expected: SocketType::Json,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for serde_json::Value {
    const SOCKET: SocketType = SocketType::Json;
    fn into_value(self) -> Value {
        Value::Json(self)
    }
}

// ---------------------------------------------------------------------------
// ndarray newtypes
// ---------------------------------------------------------------------------

impl FromValue for IxVector {
    const SOCKET: SocketType = SocketType::Vector;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Vector(data) => Ok(IxVector::new(data.clone())),
            other => Err(TypeError {
                expected: SocketType::Vector,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for IxVector {
    const SOCKET: SocketType = SocketType::Vector;
    fn into_value(self) -> Value {
        Value::Vector(self.0.to_vec())
    }
}

impl FromValue for IxMatrix {
    const SOCKET: SocketType = SocketType::Matrix;
    fn from_value(v: &Value) -> Result<Self, TypeError> {
        match v {
            Value::Matrix { rows, cols, data } => IxMatrix::new(*rows, *cols, data.clone()),
            other => Err(TypeError {
                expected: SocketType::Matrix,
                actual: other.tag(),
            }),
        }
    }
}
impl IntoValue for IxMatrix {
    const SOCKET: SocketType = SocketType::Matrix;
    fn into_value(self) -> Value {
        let (rows, cols) = self.0.dim();
        let data = self.0.into_raw_vec_and_offset().0;
        Value::Matrix { rows, cols, data }
    }
}

// ---------------------------------------------------------------------------
// Unit return (skills with no meaningful output value)
// ---------------------------------------------------------------------------

impl IntoValue for () {
    const SOCKET: SocketType = SocketType::Any;
    fn into_value(self) -> Value {
        Value::Null
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_roundtrip() {
        let v = 3.125_f64.into_value();
        assert_eq!(f64::from_value(&v).unwrap(), 3.125);
    }

    #[test]
    fn integer_to_scalar_widening() {
        // An Integer Value can be decoded as f64 — needed for lenient CLI parsing.
        let v = Value::Integer(42);
        assert_eq!(f64::from_value(&v).unwrap(), 42.0);
    }

    #[test]
    fn vector_roundtrip() {
        let v = IxVector::new(vec![1.0, 2.0, 3.0]).into_value();
        let back = IxVector::from_value(&v).unwrap();
        assert_eq!(back.0.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn matrix_roundtrip() {
        let m = IxMatrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = m.into_value();
        let back = IxMatrix::from_value(&v).unwrap();
        assert_eq!(back.0.dim(), (2, 3));
        assert_eq!(back.0[[0, 1]], 2.0);
        assert_eq!(back.0[[1, 2]], 6.0);
    }

    #[test]
    fn hexavalent_roundtrip() {
        for t in Hexavalent::all() {
            let v = t.into_value();
            assert_eq!(Hexavalent::from_value(&v).unwrap(), t);
        }
    }

    #[test]
    fn hexavalent_not_is_involutive() {
        for t in Hexavalent::all() {
            assert_eq!(t.not().not(), t, "NOT(NOT({t})) should equal {t}");
        }
    }

    #[test]
    fn hexavalent_not_swaps_p_and_d() {
        assert_eq!(Hexavalent::True.not(), Hexavalent::False);
        assert_eq!(Hexavalent::False.not(), Hexavalent::True);
        assert_eq!(Hexavalent::Probable.not(), Hexavalent::Doubtful);
        assert_eq!(Hexavalent::Doubtful.not(), Hexavalent::Probable);
        assert_eq!(Hexavalent::Unknown.not(), Hexavalent::Unknown);
        assert_eq!(Hexavalent::Contradictory.not(), Hexavalent::Contradictory);
    }

    #[test]
    fn hexavalent_and_false_absorbs() {
        for t in Hexavalent::all() {
            assert_eq!(t.and(Hexavalent::False), Hexavalent::False);
            assert_eq!(Hexavalent::False.and(t), Hexavalent::False);
        }
    }

    #[test]
    fn hexavalent_or_true_absorbs() {
        for t in Hexavalent::all() {
            assert_eq!(t.or(Hexavalent::True), Hexavalent::True);
            assert_eq!(Hexavalent::True.or(t), Hexavalent::True);
        }
    }

    #[test]
    fn hexavalent_serialize_as_symbol() {
        let j = serde_json::to_string(&Hexavalent::Probable).unwrap();
        assert_eq!(j, "\"P\"");
        let back: Hexavalent = serde_json::from_str("\"D\"").unwrap();
        assert_eq!(back, Hexavalent::Doubtful);
    }

    #[test]
    fn socket_compatibility() {
        use SocketType::*;
        assert!(Scalar.compatible_with(Scalar));
        assert!(Any.compatible_with(Matrix));
        assert!(Matrix.compatible_with(Any));
        assert!(Scalar.compatible_with(Vector)); // widening
        assert!(Vector.compatible_with(Matrix)); // widening
        assert!(!Matrix.compatible_with(Vector)); // narrowing refused
        assert!(!Text.compatible_with(Scalar));
    }

    #[test]
    fn type_mismatch_reports_actual_tag() {
        let err = f64::from_value(&Value::Text("nope".into())).unwrap_err();
        assert_eq!(err.actual, "text");
        assert_eq!(err.expected, SocketType::Scalar);
    }
}
