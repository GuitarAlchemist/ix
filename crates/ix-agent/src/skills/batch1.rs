//! Batch 1 — canary migration of 6 MCP tools to the capability registry.
//!
//! These wrappers delegate to the pre-existing `handlers::*` functions to
//! preserve exact MCP behavior; only the *registration surface* changes.
//! Each wrapper carries its hand-written JSON schema via `schema_fn = ...`.

use crate::handlers;
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

// --- ix_stats --------------------------------------------------------------

fn stats_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": { "type": "number" },
                "description": "List of numbers to compute statistics on"
            }
        },
        "required": ["data"]
    })
}

/// Compute statistics (mean, std, min, max, median) on a list of numbers.
#[ix_skill(
    domain = "math",
    name = "stats",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::stats_schema"
)]
pub fn stats(params: Value) -> Result<Value, String> {
    handlers::stats(params)
}

// --- ix_distance -----------------------------------------------------------

fn distance_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "a": { "type": "array", "items": { "type": "number" }, "description": "First vector" },
            "b": { "type": "array", "items": { "type": "number" }, "description": "Second vector" },
            "metric": {
                "type": "string",
                "enum": ["euclidean", "cosine", "manhattan"],
                "description": "Distance metric"
            }
        },
        "required": ["a", "b", "metric"]
    })
}

/// Compute distance between two vectors (euclidean, cosine, or manhattan).
#[ix_skill(
    domain = "math",
    name = "distance",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::distance_schema"
)]
pub fn distance(params: Value) -> Result<Value, String> {
    handlers::distance(params)
}

// --- ix_fft ----------------------------------------------------------------

fn fft_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "signal": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Real-valued input signal (length must be power of 2)"
            },
            "inverse": {
                "type": "boolean",
                "description": "If true, compute the inverse FFT",
                "default": false
            }
        },
        "required": ["signal"]
    })
}

/// Compute the Fast Fourier Transform of a real-valued signal.
#[ix_skill(
    domain = "signal",
    name = "fft",
    governance = "deterministic",
    schema_fn = "crate::skills::batch1::fft_schema"
)]
pub fn fft(params: Value) -> Result<Value, String> {
    handlers::fft(params)
}

// --- ix_kmeans -------------------------------------------------------------

fn kmeans_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Matrix where each row is a data point"
            },
            "k": { "type": "integer", "minimum": 1, "description": "Number of clusters" },
            "max_iter": { "type": "integer", "default": 100, "description": "Max iterations" },
            "seed": { "type": "integer", "default": 42, "description": "RNG seed" }
        },
        "required": ["data", "k"]
    })
}

fn kmeans_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Cluster index assigned to each input row"
            },
            "centroids": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "The k cluster-center vectors"
            },
            "inertia": { "type": "number", "description": "Sum of squared distances to centroids" },
            "k": { "type": "integer", "description": "Number of clusters" }
        }
    })
}

/// Cluster points using K-Means.
#[ix_skill(
    domain = "unsupervised",
    name = "kmeans",
    governance = "empirical",
    schema_fn = "crate::skills::batch1::kmeans_schema",
    output_schema_fn = "crate::skills::batch1::kmeans_output_schema"
)]
pub fn kmeans(params: Value) -> Result<Value, String> {
    handlers::kmeans(params)
}

// --- ix_pca ----------------------------------------------------------------

fn pca_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Matrix where each row is an observation and each column a feature"
            },
            "n_components": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of principal components to keep"
            }
        },
        "required": ["data", "n_components"]
    })
}

fn pca_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "transformed": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Input projected onto the top n_components axes (one row per observation)"
            },
            "explained_variance_ratio": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Fraction of total variance carried by each component"
            },
            "components": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "The principal-axis vectors (n_components rows)"
            },
            "n_components": { "type": "integer", "description": "Number of components kept" }
        }
    })
}

/// Reduce dimensionality with Principal Component Analysis: project the data
/// onto its top `n_components` principal axes and report the explained-variance
/// ratio per component.
#[ix_skill(
    domain = "unsupervised",
    name = "pca",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::pca_schema",
    output_schema_fn = "crate::skills::batch1::pca_output_schema"
)]
pub fn pca(params: Value) -> Result<Value, String> {
    handlers::pca(params)
}

// --- ix_dbscan -------------------------------------------------------------

fn dbscan_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Matrix where each row is a data point"
            },
            "eps": {
                "type": "number",
                "exclusiveMinimum": 0,
                "description": "Neighborhood radius: two points are neighbors when within this Euclidean distance"
            },
            "min_points": {
                "type": "integer",
                "minimum": 1,
                "default": 4,
                "description": "Minimum neighbors (including the point itself) for a core point; defaults to 4 when omitted"
            }
        },
        "required": ["data", "eps"]
    })
}

fn dbscan_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": { "type": "integer" },
                "description": "Cluster id per input row; 0 = noise, clusters numbered from 1"
            },
            "n_clusters": { "type": "integer", "description": "Number of clusters found (noise excluded)" },
            "n_noise": { "type": "integer", "description": "Number of points labeled noise (label 0)" },
            "eps": { "type": "number", "description": "Neighborhood radius used" },
            "min_points": { "type": "integer", "description": "Core-point threshold used" }
        }
    })
}

/// Cluster points by density with DBSCAN. Unlike k-means there is no preset
/// cluster count: dense regions become clusters (numbered from 1) and points in
/// sparse regions are labeled noise (0).
#[ix_skill(
    domain = "unsupervised",
    name = "dbscan",
    governance = "empirical",
    schema_fn = "crate::skills::batch1::dbscan_schema",
    output_schema_fn = "crate::skills::batch1::dbscan_output_schema"
)]
pub fn dbscan(params: Value) -> Result<Value, String> {
    handlers::dbscan(params)
}

// --- ix_linear_regression --------------------------------------------------

fn linear_regression_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "X": {
                "type": "array",
                "items": { "type": "array", "items": { "type": "number" } },
                "description": "Feature matrix (each row an observation)"
            },
            "y": {
                "type": "array",
                "items": { "type": "number" },
                "description": "Target vector"
            }
        },
        "required": ["X", "y"]
    })
}

/// Fit an ordinary least-squares linear regression model.
#[ix_skill(
    domain = "supervised",
    name = "linear_regression",
    governance = "empirical,deterministic",
    schema_fn = "crate::skills::batch1::linear_regression_schema"
)]
pub fn linear_regression(params: Value) -> Result<Value, String> {
    handlers::linear_regression(params)
}

// --- ix_governance_belief --------------------------------------------------

fn governance_belief_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "proposition": { "type": "string", "description": "Belief proposition text" },
            "truth_value": {
                "type": "string",
                "enum": ["T", "F", "U", "C"],
                "description": "Tetravalent truth (legacy). Hexavalent values P/D supported when present."
            },
            "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "supporting": { "type": "array", "items": { "type": "object" } },
            "contradicting": { "type": "array", "items": { "type": "object" } }
        },
        "required": ["proposition", "truth_value", "confidence"]
    })
}

/// Query the Demerzel belief engine with a belief state and receive a
/// resolved action recommendation.
#[ix_skill(
    domain = "governance",
    name = "governance.belief",
    governance = "safety,reversible",
    schema_fn = "crate::skills::batch1::governance_belief_schema"
)]
pub fn governance_belief(params: Value) -> Result<Value, String> {
    handlers::governance_belief(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Executes-test for the new `pca` skill: the catalog entry must EXECUTE,
    // not merely register (no green-but-dead). Classic Lindsay-Smith 2D PCA
    // example — its first principal component carries ~96% of the variance.
    #[test]
    fn pca_skill_executes_and_reduces_dimensions() {
        let out = pca(json!({
            "data": [
                [2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
                [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]
            ],
            "n_components": 1
        }))
        .expect("pca skill runs");

        let transformed = out["transformed"].as_array().expect("transformed array");
        assert_eq!(
            transformed.len(),
            10,
            "one projected row per input observation"
        );
        assert_eq!(
            transformed[0].as_array().unwrap().len(),
            1,
            "reduced from 2 features to 1 component"
        );

        let evr = out["explained_variance_ratio"].as_array().unwrap();
        assert_eq!(evr.len(), 1);
        assert!(
            evr[0].as_f64().unwrap() > 0.9,
            "the single retained PC should explain most of the variance, got {}",
            evr[0].as_f64().unwrap()
        );
    }

    // The skill must be registered AND pipeline-callable (arity-1), so the
    // NL→pipeline proposer can actually see it (the dogfood gap this closes).
    #[test]
    fn pca_skill_is_registered_and_pipeline_callable() {
        let desc = ix_registry::all()
            .find(|s| s.name == "pca")
            .expect("pca skill registered in the capability registry");
        assert_eq!(
            desc.inputs.len(),
            1,
            "must be arity-1 to be pipeline-callable"
        );
    }

    // Honest boundary: more components than features is rejected, not silently
    // clamped — the proposer/caller gets a clear error instead of a wrong shape.
    #[test]
    fn pca_rejects_too_many_components() {
        let err = pca(json!({ "data": [[1.0, 2.0], [3.0, 4.0]], "n_components": 5 })).unwrap_err();
        assert!(err.contains("exceeds the feature count"), "got: {err}");
    }

    // Executes-test for the new `dbscan` skill: two well-separated dense blobs
    // plus a far outlier → exactly two clusters and one noise point. The catalog
    // entry must EXECUTE, not merely register (no green-but-dead).
    #[test]
    fn dbscan_skill_executes_and_separates_clusters_from_noise() {
        let out = dbscan(json!({
            "data": [
                [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
                [5.0, 5.0], [5.1, 5.1], [5.2, 5.0], [5.0, 5.2],
                [50.0, 50.0]
            ],
            "eps": 0.5,
            "min_points": 2
        }))
        .expect("dbscan skill runs");

        assert_eq!(out["n_clusters"].as_u64().unwrap(), 2, "two dense blobs");
        assert_eq!(
            out["n_noise"].as_u64().unwrap(),
            1,
            "the far point is noise"
        );
        let labels = out["labels"].as_array().unwrap();
        assert_eq!(labels.len(), 9, "one label per input row");
        assert_eq!(
            labels[8].as_u64().unwrap(),
            0,
            "the outlier is labeled noise"
        );
        assert_eq!(labels[0], labels[1], "the first blob shares a cluster");
        assert_ne!(labels[0], labels[4], "the two blobs are different clusters");
    }

    // The skill must be registered AND pipeline-callable (arity-1), so the
    // NL→pipeline proposer can actually see it.
    #[test]
    fn dbscan_skill_is_registered_and_pipeline_callable() {
        let desc = ix_registry::all()
            .find(|s| s.name == "dbscan")
            .expect("dbscan skill registered in the capability registry");
        assert_eq!(
            desc.inputs.len(),
            1,
            "must be arity-1 to be pipeline-callable"
        );
    }

    // Honest boundary: `eps` is required — there is no data-blind default radius,
    // so an omitted `eps` returns a clear error instead of a guessed clustering.
    #[test]
    fn dbscan_requires_eps() {
        let err = dbscan(json!({ "data": [[0.0, 0.0], [1.0, 1.0]] })).unwrap_err();
        assert!(err.contains("eps"), "got: {err}");
    }

    // The canonical-label fix must propagate through the SKILL, not just the
    // crate: a point density-unreachable from any core (within eps of a border
    // point only) is reported as noise, and n_noise counts it. With the default
    // Clusterer fit_predict (fit+predict) this point would be absorbed and
    // n_noise would read 0.
    #[test]
    fn dbscan_skill_labels_border_adjacent_point_as_noise() {
        let out = dbscan(json!({
            "data": [
                [0.0, 0.0], [0.2, 0.0], [0.4, 0.0], [0.2, 0.2],
                [1.3, 0.0],
                [2.2, 0.0]
            ],
            "eps": 1.0,
            "min_points": 4
        }))
        .expect("dbscan skill runs");
        let labels = out["labels"].as_array().unwrap();
        assert_eq!(
            labels[5].as_u64().unwrap(),
            0,
            "unreachable point must be noise"
        );
        assert_eq!(out["n_noise"].as_u64().unwrap(), 1);
        assert_eq!(out["n_clusters"].as_u64().unwrap(), 1);
    }

    // Honest boundary: min_points < 1 is rejected — under it nothing can be noise,
    // contradicting the advertised contract.
    #[test]
    fn dbscan_rejects_min_points_below_one() {
        let err = dbscan(json!({ "data": [[0.0, 0.0]], "eps": 1.0, "min_points": 0 })).unwrap_err();
        assert!(err.contains("min_points"), "got: {err}");
    }
}
