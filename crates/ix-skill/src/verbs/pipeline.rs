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

/// `ix pipeline schema` — emit the JSON Schema for a `PipelineSpec`.
///
/// This is the *target language* an NL→pipeline generator must produce. The
/// `skill` enum is drawn live from `ix-registry` so the schema can only name
/// real skills — a generator constrained to this schema can never emit an
/// `UnknownSkill`. Kept INTERNAL + draft (`$id` is a `urn:`, not the public
/// `https://ix.guitaralchemist.com/...` URL) until the shape is frozen; see
/// `docs/plans/2026-06-06-ix-thinking-machine.md` (one-way doors).
pub fn schema(format: Format) -> Result<(), String> {
    output::emit(&build_schema(), format).map_err(|e| format!("{e}"))?;
    Ok(())
}

/// Build the `PipelineSpec` JSON Schema as a `Value` (split out for testing).
pub(crate) fn build_schema() -> Value {
    let mut skills: Vec<&'static str> = ix_registry::all().map(|s| s.name).collect();
    skills.sort_unstable();

    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:ix:pipeline:v1-draft",
        "title": "IX PipelineSpec",
        "description": "An ix.yaml pipeline: a DAG of skill stages. \
            Inter-stage data flow uses {\"from\": \"stage_id[.dotted.key]\"} \
            references anywhere inside a stage's args.",
        "type": "object",
        "required": ["stages"],
        "additionalProperties": false,
        "properties": {
            "version": { "const": "1", "default": "1" },
            "params": {
                "type": "object",
                "description": "Named parameter bag (string expansion only; \
                    inert in the current executor — prefer inlining values)."
            },
            "stages": {
                "type": "object",
                "minProperties": 1,
                "description": "Stages keyed by a unique id. Order is decided by \
                    deps + from-refs, not by key order.",
                "additionalProperties": { "$ref": "#/$defs/stage" }
            }
        },
        "$defs": {
            "stage": {
                "type": "object",
                "required": ["skill"],
                "additionalProperties": false,
                "properties": {
                    "skill": {
                        "type": "string",
                        "description": "Dotted registered-skill name (see `ix list skills --schemas`).",
                        "enum": skills
                    },
                    "args": {
                        "type": "object",
                        "description": "Static JSON input for the skill. May embed \
                            {\"from\": \"upstream_stage[.key]\"} references."
                    },
                    "deps": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Explicit upstream stage ids. Refs found in \
                            args are merged automatically; list any not expressed via `from`."
                    },
                    "cache": { "type": ["boolean", "null"] }
                }
            }
        }
    })
}

/// Build the natural-language action string the constitution checks for a
/// pipeline stage. Deliberately based on the *requested operation* — the skill
/// name, the stage's actual `args`, and its curated risk tags — and **not** the
/// skill's capability docs. A skill's doc may enumerate operations it *can*
/// perform (e.g. `cache`'s "set/get/delete/list"); the constitution's keyword
/// scan would then falsely reject a benign stage whose doc merely mentions
/// "delete". The args reflect what *this* stage does, so a genuinely
/// irreversible request is still caught while a benign one passes.
/// (Codex PR #77 P2.)
fn governance_action(skill: &str, args: &Value, tags: &[&str]) -> String {
    let args_str = serde_json::to_string(args).unwrap_or_default();
    format!(
        "execute skill '{skill}' with args {args_str} and governance tags [{}]",
        tags.join(", ")
    )
}

/// Fail-closed constitutional review of a spec before it executes. Every
/// stage's skill must (a) carry governance tags and (b) pass
/// `Constitution::check_action`. A missing constitution, an untagged skill, or
/// any non-compliant stage → reject. Shared by `ix pipeline run` and
/// `ix pipeline compile` so governance is unbypassable on the canonical path.
pub(crate) fn governance_gate(spec: &PipelineSpec) -> Result<Value, String> {
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let const_path = format!("{gov_dir}/constitutions/default.constitution.md");
    let constitution = ix_governance::Constitution::load(std::path::Path::new(&const_path))
        .map_err(|e| {
            format!(
                "cannot verify governance (constitution load failed: {e}) — refusing to execute"
            )
        })?;

    let mut cited: Vec<Value> = Vec::new();
    for (id, stage) in &spec.stages {
        let desc = ix_registry::by_name(&stage.skill)
            .ok_or_else(|| format!("stage '{id}': skill '{}' not in registry", stage.skill))?;
        if desc.governance_tags.is_empty() {
            return Err(format!(
                "stage '{id}' uses untagged skill '{}' (unknown blast radius) — fail-closed reject",
                stage.skill
            ));
        }
        let action = format!(
            "{} as pipeline stage '{id}'",
            governance_action(&stage.skill, &stage.args, desc.governance_tags)
        );
        let result = constitution.check_action(&action);
        if !result.compliant {
            return Err(format!(
                "stage '{id}' (skill '{}') is non-compliant: {}",
                stage.skill,
                result.warnings.join("; ")
            ));
        }
        for art in &result.relevant_articles {
            cited.push(json!({ "stage": id, "article": art.number }));
        }
    }
    Ok(json!({ "verdict": "compliant", "articles_cited": cited }))
}

/// `ix pipeline run [-f FILE] [--json]` — execute the pipeline.
///
/// When `stream_ndjson` is true, emits NDJSON events (`start`, `stage_start`,
/// `stage_complete`, `done`) to stdout in real time as the DAG runs.
pub fn run(file: Option<&str>, stream_ndjson: bool, format: Format) -> Result<(), String> {
    let path: &str = file.unwrap_or(DEFAULT_FILE);
    let spec = PipelineSpec::from_file(path).map_err(|e| format!("loading {path}: {e}"))?;
    let dag = lower(&spec).map_err(|e| format!("lowering {path}: {e}"))?;
    // Fail-closed governance gate — a pipeline never executes unreviewed.
    governance_gate(&spec).map_err(|e| format!("governance gate: {e}"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_skill_enum_matches_registry() {
        let schema = build_schema();
        let enum_vals = schema["$defs"]["stage"]["properties"]["skill"]["enum"]
            .as_array()
            .expect("skill enum is an array");
        let from_schema: std::collections::BTreeSet<&str> =
            enum_vals.iter().map(|v| v.as_str().unwrap()).collect();
        let from_registry: std::collections::BTreeSet<&str> =
            ix_registry::all().map(|s| s.name).collect();
        assert_eq!(
            from_schema, from_registry,
            "pipeline schema skill enum must equal ix_registry::all()"
        );
    }

    #[test]
    fn schema_accepts_the_showcase_shape() {
        // The one checked-in real spec must be expressible under the schema:
        // its skills are in the enum, and `stages` is a non-empty object.
        let schema = build_schema();
        assert_eq!(schema["properties"]["stages"]["minProperties"], json!(1));
        let enum_vals = schema["$defs"]["stage"]["properties"]["skill"]["enum"]
            .as_array()
            .unwrap();
        for skill in ["stats", "fft", "number_theory", "governance.check"] {
            assert!(
                enum_vals.iter().any(|v| v == skill),
                "showcase skill '{skill}' missing from schema enum"
            );
        }
    }

    fn default_constitution() -> ix_governance::Constitution {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../governance/demerzel/constitutions/default.constitution.md");
        ix_governance::Constitution::load(&path).expect("load default constitution")
    }

    #[test]
    fn governance_action_excludes_skill_docs() {
        // Regression for Codex PR #77 P2: the constitutional action string must
        // carry the requested args + tags, never the skill's capability docs
        // (which may list "delete" and trip Article 3 on a benign stage).
        let action = governance_action("cache", &json!({ "operation": "keys" }), &["safety"]);
        assert!(
            action.contains("\"operation\":\"keys\""),
            "requested args must drive the check: {action}"
        );
        assert!(
            !action.contains("delete"),
            "capability-doc words must not leak into the action: {action}"
        );
    }

    #[test]
    fn benign_cache_stage_is_compliant() {
        // The `cache` skill's docs mention "delete"; a benign `keys` operation
        // must still pass the constitution. Pre-fix this falsely tripped
        // Article 3 (Reversibility) and refused a legitimate compiled pipeline.
        let c = default_constitution();
        let action = governance_action("cache", &json!({ "operation": "keys" }), &["safety"]);
        assert!(
            c.check_action(&action).compliant,
            "benign cache stage must be compliant: {action}"
        );
    }

    #[test]
    fn destructive_request_is_still_caught() {
        // The fix must not neuter the gate: a stage that actually *requests* a
        // delete is still non-compliant via Article 3.
        let c = default_constitution();
        let action = governance_action(
            "cache",
            &json!({ "operation": "delete", "key": "x" }),
            &["safety"],
        );
        assert!(
            !c.check_action(&action).compliant,
            "a real delete request must be caught: {action}"
        );
    }
}
