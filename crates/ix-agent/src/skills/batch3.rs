//! Batch 3 — final 9 composite / bridge MCP tools.
//!
//! Completes the 43-tool migration (6+28+9 = 43). These tools orchestrate
//! multiple underlying crates or bridge to external services (TARS, GA),
//! so their schemas are denser than batch1/batch2's primitives.

use crate::handlers;
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

// ---- pipeline ------------------------------------------------------------
fn pipeline_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["info"]},
            "steps": {
                "type": "array",
                "items": {"type": "object", "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}}
                }, "required": ["id"]}
            }
        },
        "required": ["operation", "steps"]
    })
}
/// DAG pipeline analysis: toposort, parallel levels, critical path.
#[ix_skill(domain = "pipeline", name = "pipeline", governance = "deterministic", schema_fn = "crate::skills::batch3::pipeline_schema")]
pub fn pipeline(p: Value) -> Result<Value, String> { handlers::pipeline_exec(p) }

// ---- cache ---------------------------------------------------------------
fn cache_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["set", "get", "delete", "keys"]},
            "key": {"type": "string"},
            "value": {}
        },
        "required": ["operation"]
    })
}
/// In-memory cache: set/get/delete/list operations.
#[ix_skill(domain = "cache", name = "cache", governance = "deterministic", schema_fn = "crate::skills::batch3::cache_schema")]
pub fn cache(p: Value) -> Result<Value, String> { handlers::cache_op(p) }

// ---- federation.discover -------------------------------------------------
fn federation_discover_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "query": {"type": "string"}
        }
    })
}
/// Discover capabilities across ix / tars / ga ecosystems.
#[ix_skill(domain = "federation", name = "federation.discover", governance = "safety", schema_fn = "crate::skills::batch3::federation_discover_schema")]
pub fn federation_discover(p: Value) -> Result<Value, String> { handlers::federation_discover(p) }

// ---- trace.ingest --------------------------------------------------------
fn trace_ingest_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "dir": {"type": "string"}
        }
    })
}
/// Ingest GA trace files and compute summary statistics.
#[ix_skill(domain = "trace", name = "trace.ingest", governance = "empirical", schema_fn = "crate::skills::batch3::trace_ingest_schema")]
pub fn trace_ingest(p: Value) -> Result<Value, String> { handlers::trace_ingest(p) }

// ---- ml_pipeline ---------------------------------------------------------
fn ml_pipeline_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "source": {"type": "object", "properties": {
                "type": {"type": "string", "enum": ["csv", "json", "inline"]},
                "path": {"type": "string"},
                "data": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "has_header": {"type": "boolean"},
                "target_column": {}
            }, "required": ["type"]},
            "task": {"type": "string", "enum": ["classify", "regress", "cluster", "auto"]},
            "model": {"type": "string"},
            "model_params": {"type": "object"},
            "preprocess": {"type": "object", "properties": {
                "normalize": {"type": "boolean"},
                "drop_nan": {"type": "boolean"},
                "pca_components": {"type": "integer"}
            }},
            "split": {"type": "object", "properties": {
                "test_ratio": {"type": "number"},
                "seed": {"type": "integer"}
            }},
            "persist": {"type": "boolean"},
            "persist_key": {"type": "string"},
            "return_predictions": {"type": "boolean"},
            "max_rows": {"type": "integer"},
            "max_features": {"type": "integer"}
        },
        "required": ["source"]
    })
}
/// End-to-end ML pipeline: load → preprocess → train → evaluate → persist.
#[ix_skill(domain = "ml_pipeline", name = "ml_pipeline", governance = "empirical", schema_fn = "crate::skills::batch3::ml_pipeline_schema")]
pub fn ml_pipeline(p: Value) -> Result<Value, String> { handlers::ml_pipeline(p) }

// ---- ml_predict ----------------------------------------------------------
fn ml_predict_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "persist_key": {"type": "string"},
            "data": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
        },
        "required": ["persist_key", "data"]
    })
}
/// Predict using a previously-persisted ML model (by persist_key).
#[ix_skill(domain = "ml_pipeline", name = "ml_predict", governance = "empirical", schema_fn = "crate::skills::batch3::ml_predict_schema")]
pub fn ml_predict(p: Value) -> Result<Value, String> { handlers::ml_predict(p) }

// ---- code_analyze --------------------------------------------------------
fn code_analyze_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "source": {"type": "string"},
            "language": {"type": "string", "enum": ["rust", "python", "javascript", "typescript", "cpp", "java", "go", "csharp", "fsharp", "php", "ruby"]},
            "path": {"type": "string"}
        }
    })
}
/// Code complexity analysis: cyclomatic, cognitive, Halstead, SLOC, MI.
#[ix_skill(domain = "code", name = "code_analyze", governance = "deterministic", schema_fn = "crate::skills::batch3::code_analyze_schema")]
pub fn code_analyze(p: Value) -> Result<Value, String> { handlers::code_analyze(p) }

// ---- tars_bridge ---------------------------------------------------------
fn tars_bridge_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["prepare_traces", "prepare_patterns", "export_grammar"]},
            "trace_dir": {"type": "string"},
            "min_frequency": {"type": "integer"}
        },
        "required": ["action"]
    })
}
/// Prepare ix results for TARS ingestion (traces / patterns / grammar).
#[ix_skill(domain = "federation", name = "tars_bridge", governance = "safety", schema_fn = "crate::skills::batch3::tars_bridge_schema")]
pub fn tars_bridge(p: Value) -> Result<Value, String> { handlers::tars_bridge(p) }

// ---- ga_bridge -----------------------------------------------------------
fn ga_bridge_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["chord_features", "progression_features", "scale_features", "workflow_guide"]},
            "chords": {"type": "array", "items": {"type": "string"}},
            "progression": {"type": "string"}
        },
        "required": ["action"]
    })
}
/// Convert GA music theory data into ML-ready feature matrices.
#[ix_skill(domain = "federation", name = "ga_bridge", governance = "safety", schema_fn = "crate::skills::batch3::ga_bridge_schema")]
pub fn ga_bridge(p: Value) -> Result<Value, String> { handlers::ga_bridge(p) }
