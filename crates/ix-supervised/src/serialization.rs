//! Model serialization: versioned envelopes and state structs for persistence.

use serde::{Deserialize, Serialize};

/// Versioned model state envelope for cache persistence.
///
/// Wraps any algorithm's serialized state with metadata for safe
/// deserialization and provenance tracking.
///
/// # Example
///
/// ```
/// use ix_supervised::serialization::ModelEnvelope;
///
/// let envelope = ModelEnvelope {
///     version: "0.1.0".to_string(),
///     algorithm: "LinearRegression".to_string(),
///     params: serde_json::json!({"weights": [2.0], "bias": 1.0}),
///     preprocessing: None,
///     feature_names: Some(vec!["x1".to_string()]),
///     trained_at: "2026-03-15T00:00:00Z".to_string(),
/// };
///
/// let json = serde_json::to_string(&envelope).unwrap();
/// let restored: ModelEnvelope = serde_json::from_str(&json).unwrap();
/// assert_eq!(restored.algorithm, "LinearRegression");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEnvelope {
    /// Semantic version of the serialization format.
    pub version: String,
    /// Algorithm name (e.g. "LinearRegression", "KMeans").
    pub algorithm: String,
    /// Algorithm-specific serialized parameters / trained state.
    pub params: serde_json::Value,
    /// Optional preprocessing configuration (scaling, encoding, etc.).
    pub preprocessing: Option<serde_json::Value>,
    /// Optional feature names for interpretability.
    pub feature_names: Option<Vec<String>>,
    /// ISO-8601 timestamp when the model was trained.
    pub trained_at: String,
}
