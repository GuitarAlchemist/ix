//! Batch 3 — final 9 composite / bridge MCP tools.
//!
//! Completes the 43-tool migration (6+28+9 = 43). These tools orchestrate
//! multiple underlying crates or bridge to external services (TARS, GA),
//! so their schemas are denser than batch1/batch2's primitives.

use crate::handlers;
use crate::schema::{object, Prop};
use ix_skill_macros::ix_skill;
use serde_json::Value;

// ---- pipeline ------------------------------------------------------------
fn pipeline_schema() -> Value {
    object(
        vec![
            ("operation", Prop::string().enum_of(&["info"])),
            (
                "steps",
                Prop::array_of(Prop::object(
                    vec![
                        ("id", Prop::string()),
                        ("description", Prop::string()),
                        ("depends_on", Prop::str_array()),
                    ],
                    &["id"],
                )),
            ),
        ],
        &["operation", "steps"],
    )
}
/// DAG pipeline analysis: toposort, parallel levels, critical path.
#[ix_skill(
    domain = "pipeline",
    name = "pipeline",
    governance = "deterministic",
    schema_fn = "crate::skills::batch3::pipeline_schema"
)]
pub fn pipeline(p: Value) -> Result<Value, String> {
    handlers::pipeline_exec(p)
}

// ---- cache ---------------------------------------------------------------
fn cache_schema() -> Value {
    object(
        vec![
            ("operation", Prop::string().enum_of(&["set", "get", "delete", "keys"])),
            ("key", Prop::string()),
            ("value", Prop::any()),
        ],
        &["operation"],
    )
}
/// In-memory cache: set/get/delete/list operations.
#[ix_skill(
    domain = "cache",
    name = "cache",
    governance = "deterministic",
    schema_fn = "crate::skills::batch3::cache_schema"
)]
pub fn cache(p: Value) -> Result<Value, String> {
    handlers::cache_op(p)
}

// ---- federation.discover -------------------------------------------------
fn federation_discover_schema() -> Value {
    object(
        vec![("domain", Prop::string()), ("query", Prop::string())],
        &[],
    )
}
/// Discover capabilities across ix / tars / ga ecosystems.
#[ix_skill(
    domain = "federation",
    name = "federation.discover",
    governance = "safety",
    schema_fn = "crate::skills::batch3::federation_discover_schema"
)]
pub fn federation_discover(p: Value) -> Result<Value, String> {
    handlers::federation_discover(p)
}

// ---- trace.ingest --------------------------------------------------------
fn trace_ingest_schema() -> Value {
    object(vec![("dir", Prop::string())], &[])
}
/// Ingest GA trace files and compute summary statistics.
#[ix_skill(
    domain = "trace",
    name = "trace.ingest",
    governance = "empirical",
    schema_fn = "crate::skills::batch3::trace_ingest_schema"
)]
pub fn trace_ingest(p: Value) -> Result<Value, String> {
    handlers::trace_ingest(p)
}

// ---- fuzzy.eval (primitive #5) -----------------------------------------
fn fuzzy_eval_schema() -> Value {
    let distribution = || {
        Prop::object(
            vec![
                ("T", Prop::number()),
                ("P", Prop::number()),
                ("U", Prop::number()),
                ("D", Prop::number()),
                ("F", Prop::number()),
                ("C", Prop::number()),
            ],
            &[],
        )
    };
    object(
        vec![
            ("operation", Prop::string().enum_of(&["info", "not", "and", "or"])),
            ("distribution", distribution()),
            (
                "other",
                distribution().desc("Second distribution for and/or"),
            ),
        ],
        &["distribution"],
    )
}
/// Evaluate hexavalent fuzzy distribution ops: info / not / and / or.
#[ix_skill(
    domain = "fuzzy",
    name = "fuzzy.eval",
    governance = "deterministic",
    schema_fn = "crate::skills::batch3::fuzzy_eval_schema"
)]
pub fn fuzzy_eval(p: Value) -> Result<Value, String> {
    handlers::fuzzy_eval(p)
}

// ---- session.flywheel_export (primitive #6) -----------------------------
fn session_flywheel_export_schema() -> Value {
    object(
        vec![
            ("session_log", Prop::string().desc("Path to the JSONL session log")),
            (
                "trace_dir",
                Prop::string().desc("Destination directory (default ~/.ga/traces)"),
            ),
            (
                "trace_id",
                Prop::string().desc("Explicit trace id (default: log filename stem)"),
            ),
        ],
        &["session_log"],
    )
}
/// Convert a persisted SessionLog to a GA trace file consumable by ix_trace_ingest.
#[ix_skill(
    domain = "session",
    name = "session.flywheel_export",
    governance = "deterministic",
    schema_fn = "crate::skills::batch3::session_flywheel_export_schema"
)]
pub fn session_flywheel_export(p: Value) -> Result<Value, String> {
    handlers::session_flywheel_export(p)
}

// ---- ml_pipeline ---------------------------------------------------------
fn ml_pipeline_schema() -> Value {
    object(
        vec![
            (
                "source",
                Prop::object(
                    vec![
                        ("type", Prop::string().enum_of(&["csv", "json", "inline"])),
                        ("path", Prop::string()),
                        ("data", Prop::num_matrix()),
                        ("has_header", Prop::boolean()),
                        ("target_column", Prop::any()),
                    ],
                    &["type"],
                ),
            ),
            ("task", Prop::string().enum_of(&["classify", "regress", "cluster", "auto"])),
            ("model", Prop::string()),
            ("model_params", Prop::object_any()),
            (
                "preprocess",
                Prop::object(
                    vec![
                        ("normalize", Prop::boolean()),
                        ("drop_nan", Prop::boolean()),
                        ("pca_components", Prop::integer()),
                    ],
                    &[],
                ),
            ),
            (
                "split",
                Prop::object(
                    vec![("test_ratio", Prop::number()), ("seed", Prop::integer())],
                    &[],
                ),
            ),
            ("persist", Prop::boolean()),
            ("persist_key", Prop::string()),
            ("return_predictions", Prop::boolean()),
            ("max_rows", Prop::integer()),
            ("max_features", Prop::integer()),
        ],
        &["source"],
    )
}
/// End-to-end ML pipeline: load → preprocess → train → evaluate → persist.
#[ix_skill(
    domain = "ml_pipeline",
    name = "ml_pipeline",
    governance = "empirical",
    schema_fn = "crate::skills::batch3::ml_pipeline_schema"
)]
pub fn ml_pipeline(p: Value) -> Result<Value, String> {
    handlers::ml_pipeline(p)
}

// ---- ml_predict ----------------------------------------------------------
fn ml_predict_schema() -> Value {
    object(
        vec![
            ("persist_key", Prop::string()),
            ("data", Prop::num_matrix()),
        ],
        &["persist_key", "data"],
    )
}
/// Predict using a previously-persisted ML model (by persist_key).
#[ix_skill(
    domain = "ml_pipeline",
    name = "ml_predict",
    governance = "empirical",
    schema_fn = "crate::skills::batch3::ml_predict_schema"
)]
pub fn ml_predict(p: Value) -> Result<Value, String> {
    handlers::ml_predict(p)
}

// ---- code_analyze --------------------------------------------------------
fn code_analyze_schema() -> Value {
    object(
        vec![
            ("source", Prop::string()),
            (
                "language",
                Prop::string().enum_of(&[
                    "rust",
                    "python",
                    "javascript",
                    "typescript",
                    "cpp",
                    "java",
                    "go",
                    "csharp",
                    "fsharp",
                    "php",
                    "ruby",
                ]),
            ),
            ("path", Prop::string()),
        ],
        &[],
    )
}
/// Code complexity analysis: cyclomatic, cognitive, Halstead, SLOC, MI.
#[ix_skill(
    domain = "code",
    name = "code_analyze",
    governance = "deterministic",
    schema_fn = "crate::skills::batch3::code_analyze_schema"
)]
pub fn code_analyze(p: Value) -> Result<Value, String> {
    handlers::code_analyze(p)
}

// ---- tars_bridge ---------------------------------------------------------
fn tars_bridge_schema() -> Value {
    object(
        vec![
            (
                "action",
                Prop::string().enum_of(&["prepare_traces", "prepare_patterns", "export_grammar"]),
            ),
            ("trace_dir", Prop::string()),
            ("min_frequency", Prop::integer()),
        ],
        &["action"],
    )
}
/// Prepare ix results for TARS ingestion (traces / patterns / grammar).
#[ix_skill(
    domain = "federation",
    name = "tars_bridge",
    governance = "safety",
    schema_fn = "crate::skills::batch3::tars_bridge_schema"
)]
pub fn tars_bridge(p: Value) -> Result<Value, String> {
    handlers::tars_bridge(p)
}

// ---- ga_bridge -----------------------------------------------------------
fn ga_bridge_schema() -> Value {
    object(
        vec![
            (
                "action",
                Prop::string().enum_of(&[
                    "chord_features",
                    "progression_features",
                    "scale_features",
                    "workflow_guide",
                ]),
            ),
            ("chords", Prop::str_array()),
            ("progression", Prop::string()),
        ],
        &["action"],
    )
}
/// Convert GA music theory data into ML-ready feature matrices.
#[ix_skill(
    domain = "federation",
    name = "ga_bridge",
    governance = "safety",
    schema_fn = "crate::skills::batch3::ga_bridge_schema"
)]
pub fn ga_bridge(p: Value) -> Result<Value, String> {
    handlers::ga_bridge(p)
}

// ---- context.walk --------------------------------------------------------
//
// Deterministic structural retrieval over a Rust workspace via the
// ix-context crate. See docs/brainstorms/2026-04-10-context-dag.md and
// crates/ix-context/src/lib.rs for the full design.
//
// Stateless handler: builds a fresh ProjectIndex from the current working
// directory on every call. This is slower than caching the index across
// calls but keeps the skill pattern uniform with every other batch entry.
// A future optimization can introduce a shared index cache at the
// ix-agent level.
fn context_walk_schema() -> Value {
    object(
        vec![
            (
                "target",
                Prop::string().desc(
                    "Fully-qualified free-function path to walk from, e.g. \"ix_math::eigen::symmetric_eigen\"",
                ),
            ),
            (
                "strategy",
                Prop::string()
                    .enum_of(&[
                        "callers",
                        "callees",
                        "siblings",
                        "cochange",
                        "callers_transitive",
                        "callees_transitive",
                        "module_siblings",
                        "git_cochange",
                    ])
                    .desc("Walk strategy. Short and long names accepted."),
            ),
            (
                "strategy_params",
                Prop::object(
                    vec![
                        (
                            "max_depth",
                            Prop::integer().minimum(1).maximum(64).default(3),
                        ),
                        (
                            "min_commits_shared",
                            Prop::integer().minimum(1).default(2),
                        ),
                    ],
                    &[],
                ),
            ),
            (
                "budget",
                Prop::object(
                    vec![
                        ("max_nodes", Prop::integer().minimum(1).default(1024)),
                        ("max_edges", Prop::integer().minimum(1).default(4096)),
                        ("timeout_ms", Prop::integer().minimum(1).default(30000)),
                    ],
                    &[],
                ),
            ),
            (
                "workspace_root",
                Prop::string().desc(
                    "Optional absolute path to the workspace root. Defaults to the current working directory.",
                ),
            ),
        ],
        &["target", "strategy"],
    )
}

/// Deterministic structural context DAG walker over a Rust workspace.
/// Returns a replayable ContextBundle with nodes, edges, and a walk_trace
/// that reconstructs the walker's exact informational state.
#[ix_skill(
    domain = "context",
    name = "context.walk",
    governance = "deterministic",
    schema_fn = "crate::skills::batch3::context_walk_schema"
)]
pub fn context_walk(p: Value) -> Result<Value, String> {
    // Optional workspace_root override: pluck it out of the params before
    // handing the rest to ix-context's handler.
    let workspace_root = match p.get("workspace_root").and_then(|v| v.as_str()) {
        Some(path) => std::path::PathBuf::from(path),
        None => std::env::current_dir().map_err(|e| format!("failed to read current dir: {e}"))?,
    };

    let index = ix_context::index::ProjectIndex::build(&workspace_root).map_err(|e| {
        format!(
            "failed to build ProjectIndex at {}: {e}",
            workspace_root.display()
        )
    })?;

    // Strip workspace_root from the params before forwarding — the ix-context
    // WalkRequest schema doesn't know about it.
    let mut forwarded = p;
    if let Some(obj) = forwarded.as_object_mut() {
        obj.remove("workspace_root");
    }

    match ix_context::mcp::handle_json_request(&index, forwarded) {
        Ok(v) => Ok(v),
        Err(e) => Err(format!("{e}")),
    }
}
