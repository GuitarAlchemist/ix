//! `ix pipeline {new,validate,dag,run}` — DAG pipelines over `ix.yaml`.

use crate::output::{self, Format};
use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lock::LockFile;
use ix_pipeline::lower::lower;
use ix_pipeline::spec::PipelineSpec;
use serde_json::{json, Value};
use std::io::{self, Write};
use std::path::Path;

const DEFAULT_FILE: &str = "ix.yaml";

/// `ix pipeline new <name>` — scaffold an `ix.yaml` if it doesn't exist.
pub fn new(name: &str, format: Format) -> Result<(), String> {
    let path = Path::new(DEFAULT_FILE);
    if path.exists() {
        return Err(format!(
            "{} already exists (refusing to overwrite)",
            path.display()
        ));
    }
    let spec = PipelineSpec::scaffold(name);
    let yaml = spec.to_yaml_string().map_err(|e| format!("{e}"))?;
    std::fs::write(path, &yaml).map_err(|e| format!("writing {}: {e}", path.display()))?;
    output::emit(
        &json!({
            "action": "created",
            "path": path.display().to_string(),
            "stages": spec.stages.len(),
        }),
        format,
    )
    .map_err(|e| format!("{e}"))?;
    Ok(())
}

/// `ix pipeline validate [-f FILE]` — parse + lower to catch cycles /
/// unknown skills / unknown stage refs, without executing.
pub fn validate(file: Option<&str>, format: Format) -> Result<(), String> {
    let path: &str = file.unwrap_or(DEFAULT_FILE);
    let spec = PipelineSpec::from_file(path).map_err(|e| format!("loading {path}: {e}"))?;
    let _dag = lower(&spec).map_err(|e| format!("lowering {path}: {e}"))?;
    output::emit(
        &json!({
            "path": path,
            "version": spec.version,
            "stages": spec.stages.len(),
            "status": "valid",
        }),
        format,
    )
    .map_err(|e| format!("{e}"))?;
    Ok(())
}

/// `ix pipeline dag [-f FILE]` — render parallel execution levels.
pub fn dag(file: Option<&str>, format: Format) -> Result<(), String> {
    let path: &str = file.unwrap_or(DEFAULT_FILE);
    let spec = PipelineSpec::from_file(path).map_err(|e| format!("loading {path}: {e}"))?;
    let dag = lower(&spec).map_err(|e| format!("lowering {path}: {e}"))?;
    let levels = dag.parallel_levels();
    let levels_json: Vec<Vec<String>> = levels
        .iter()
        .map(|lvl| lvl.iter().map(|id| (*id).clone()).collect())
        .collect();
    output::emit(
        &json!({
            "path": path,
            "levels": levels_json,
            "total_stages": spec.stages.len(),
            "parallel_depth": levels_json.len(),
        }),
        format,
    )
    .map_err(|e| format!("{e}"))?;
    Ok(())
}

/// `ix pipeline run [-f FILE] [--json]` — execute the pipeline.
///
/// When `stream_ndjson` is true, emits NDJSON events (`start`, `stage_start`,
/// `stage_complete`, `done`) to stdout in real time as the DAG runs.
pub fn run(file: Option<&str>, stream_ndjson: bool, format: Format) -> Result<(), String> {
    let path: &str = file.unwrap_or(DEFAULT_FILE);
    let spec = PipelineSpec::from_file(path).map_err(|e| format!("loading {path}: {e}"))?;
    let dag = lower(&spec).map_err(|e| format!("lowering {path}: {e}"))?;
    let levels = dag.parallel_levels();
    let total_stages = spec.stages.len();

    if stream_ndjson {
        emit_event(&json!({
            "event": "start",
            "path": path,
            "stages": total_stages,
            "parallel_depth": levels.len(),
        }));
    }

    // The current executor doesn't expose per-stage callbacks, so we emit
    // start+done around the whole run for phase 1. Per-stage events arrive
    // in a future iteration that adds a callback hook to `execute()`.
    let result =
        execute(&dag, &Default::default(), &NoCache).map_err(|e| format!("execution: {e}"))?;

    // Write ix.lock alongside the spec (phase 1: write-only, no enforcement).
    let lock = LockFile::from_run(&spec, &result);
    let lock_yaml = lock
        .to_yaml_string()
        .map_err(|e| format!("lock yaml: {e}"))?;
    let lock_path = Path::new(path)
        .parent()
        .unwrap_or(Path::new("."))
        .join("ix.lock");
    std::fs::write(&lock_path, &lock_yaml)
        .map_err(|e| format!("writing {}: {e}", lock_path.display()))?;

    if stream_ndjson {
        for (id, node_result) in &result.node_results {
            emit_event(&json!({
                "event": "stage_complete",
                "stage": id,
                "duration_ms": node_result.duration.as_millis() as u64,
                "cache_hit": node_result.cache_hit,
            }));
        }
        emit_event(&json!({
            "event": "done",
            "duration_ms": result.total_duration.as_millis() as u64,
            "cache_hits": result.cache_hits,
        }));
        return Ok(());
    }

    // Non-streaming: emit a single summary document with per-stage outputs.
    let mut stages_out = serde_json::Map::new();
    for (id, node_result) in &result.node_results {
        stages_out.insert(
            id.clone(),
            json!({
                "duration_ms": node_result.duration.as_millis() as u64,
                "cache_hit": node_result.cache_hit,
                "output": node_result.output,
            }),
        );
    }
    output::emit(
        &json!({
            "path": path,
            "total_duration_ms": result.total_duration.as_millis() as u64,
            "cache_hits": result.cache_hits,
            "stages": stages_out,
        }),
        format,
    )
    .map_err(|e| format!("{e}"))?;
    Ok(())
}

fn emit_event(value: &Value) {
    let mut out = io::stdout().lock();
    let _ = serde_json::to_writer(&mut out, value);
    let _ = writeln!(out);
}
