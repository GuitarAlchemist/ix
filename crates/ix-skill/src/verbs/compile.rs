//! `ix pipeline compile "<natural language>"` — the IX "thinking machine".
//!
//! Translates a natural-language request into a canonical [`PipelineSpec`]
//! (`ix.yaml`), validates it with `lower()`, gates it through the Demerzel
//! constitution, executes it, and narrates the result back in prose.
//!
//! # Architecture (see `docs/brainstorms/2026-06-06-ix-thinking-machine-architecture.md`)
//!
//! - **Proposer = direct Anthropic Messages API.** MCP server-initiated
//!   `sampling/createMessage` is deprecated (SEP-2577) and unsupported by
//!   Claude Code, so we call the provider API directly rather than relying on
//!   a host that won't answer. The LLM is a *proposer*, never the judge.
//! - **Oracle = `lower()`** (`ix-pipeline`): the deterministic accept/reject
//!   authority. Unknown skill / unresolved `from`-ref / cycle → typed error
//!   fed back verbatim to the proposer for bounded self-repair (≤ N rounds).
//! - **Gate = `governance_gate`**: fail-closed constitutional review BEFORE
//!   execution — a generated pipeline never runs unreviewed.
//! - **Dogfood loop:** every translation/validation/governance failure is
//!   appended to `state/thinking-machine/gaps.jsonl` — that backlog IS the
//!   feedback that drives IX's own improvement.

use crate::output::{self, Format};
use crate::verbs::pipeline::build_schema;
use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lower::lower;
use ix_pipeline::spec::PipelineSpec;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const DEFAULT_MODEL: &str = "claude-opus-4-8";
const GEN_FILE: &str = "ix.compiled.yaml";

/// `ix pipeline compile "<sentence>"` entry point.
pub fn compile(
    sentence: &str,
    max_rounds: u32,
    run_it: bool,
    format: Format,
) -> Result<(), String> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| "ANTHROPIC_API_KEY not set (the proposer calls the Anthropic Messages API directly; MCP sampling is deprecated)".to_string())?;
    let model = std::env::var("IX_THINKER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());

    let catalog = pipeline_callable_catalog();
    let schema = build_schema();
    let system_prompt = build_system_prompt(&schema, &catalog);

    // Conversation history for the bounded repair loop.
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": sentence })];

    let mut last_error = String::new();
    let mut spec: Option<PipelineSpec> = None;
    let mut rounds_used = 0u32;

    for round in 0..=max_rounds {
        rounds_used = round;
        let reply = call_anthropic(&api_key, &model, &system_prompt, &messages)?;
        let yaml = strip_fences(&reply);

        match parse_and_lower(&yaml) {
            Ok(parsed) => {
                spec = Some(parsed);
                break;
            }
            Err(e) => {
                last_error = e;
                if round == max_rounds {
                    break;
                }
                // Feed the verbatim error back for self-repair.
                messages.push(json!({ "role": "assistant", "content": reply }));
                messages.push(json!({
                    "role": "user",
                    "content": format!(
                        "That spec failed validation:\n\n{last_error}\n\n\
                         Emit ONLY a corrected `ix.yaml` PipelineSpec (no prose, no fences). \
                         Use the map-based `stages:` shape and only skills from the catalog."
                    )
                }));
            }
        }
    }

    let Some(spec) = spec else {
        // Translation failure → dogfood backlog row.
        log_gap(sentence, "translate", &last_error, rounds_used);
        return emit_result(
            format,
            json!({
                "status": "translate_failed",
                "sentence": sentence,
                "rounds": rounds_used,
                "error": last_error,
                "logged_to": gap_log_path(),
            }),
        );
    };

    // ── Fail-closed governance gate (before any execution) ───────────────
    match governance_gate(&spec) {
        Ok(gate) => {
            // Persist the compiled spec for inspection / `ix pipeline run`.
            let yaml = spec.to_yaml_string().map_err(|e| format!("{e}"))?;
            std::fs::write(GEN_FILE, &yaml)
                .map_err(|e| format!("writing {GEN_FILE}: {e}"))?;

            if !run_it {
                return emit_result(
                    format,
                    json!({
                        "status": "compiled",
                        "sentence": sentence,
                        "rounds": rounds_used,
                        "stages": spec.stages.len(),
                        "governance": gate,
                        "path": GEN_FILE,
                        "note": "governance PASS; spec written. Re-run with --run to execute.",
                    }),
                );
            }
            execute_and_narrate(&spec, sentence, rounds_used, gate, format)
        }
        Err(rejection) => {
            log_gap(sentence, "governance", &rejection, rounds_used);
            emit_result(
                format,
                json!({
                    "status": "governance_rejected",
                    "sentence": sentence,
                    "rounds": rounds_used,
                    "reason": rejection,
                    "logged_to": gap_log_path(),
                    "note": "fail-closed: generated pipeline refused before execution.",
                }),
            )
        }
    }
}

/// Parse YAML → `PipelineSpec` and lower it. Returns the spec or a verbatim
/// error string (the exact text the proposer must repair against).
fn parse_and_lower(yaml: &str) -> Result<PipelineSpec, String> {
    let spec = PipelineSpec::from_yaml_str(yaml).map_err(|e| format!("parse/shape error: {e}"))?;
    lower(&spec).map_err(|e| format!("lowering error: {e}"))?;
    Ok(spec)
}

/// Fail-closed constitutional review. Every stage's skill must (a) carry
/// governance tags and (b) pass `Constitution::check_action`. Missing
/// constitution, an untagged skill, or any non-compliant stage → reject.
fn governance_gate(spec: &PipelineSpec) -> Result<Value, String> {
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let const_path = format!("{gov_dir}/constitutions/default.constitution.md");
    let constitution = ix_governance::Constitution::load(std::path::Path::new(&const_path))
        .map_err(|e| format!("cannot verify governance (constitution load failed: {e}) — refusing to execute"))?;

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
            "execute skill '{}' ({}) with governance tags [{}] as pipeline stage '{id}'",
            stage.skill,
            desc.doc,
            desc.governance_tags.join(", ")
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

/// Execute the lowered pipeline and narrate the run back in prose.
fn execute_and_narrate(
    spec: &PipelineSpec,
    sentence: &str,
    rounds: u32,
    gate: Value,
    format: Format,
) -> Result<(), String> {
    let dag = lower(spec).map_err(|e| format!("re-lower: {e}"))?;
    let levels = dag.parallel_levels();
    let result = execute(&dag, &Default::default(), &NoCache)
        .map_err(|e| format!("execution: {e}"))?;

    // ── Deterministic NL narration grounded in the typed DAG ─────────────
    let mut prose = String::new();
    prose.push_str(&format!(
        "Compiled \"{sentence}\" into a {}-stage pipeline ({} parallel level{}). Governance: PASS.\n",
        spec.stages.len(),
        levels.len(),
        if levels.len() == 1 { "" } else { "s" }
    ));
    for (i, level) in levels.iter().enumerate() {
        for id in level {
            let skill = spec.stages.get(*id).map(|s| s.skill.as_str()).unwrap_or("?");
            let out = result
                .node_results
                .get(*id)
                .map(|r| compact(&r.output))
                .unwrap_or_else(|| "—".into());
            prose.push_str(&format!("  L{i}: stage '{id}' ran '{skill}' → {out}\n"));
        }
    }
    prose.push_str(&format!(
        "Completed in {} ms.",
        result.total_duration.as_millis()
    ));

    emit_result(
        format,
        json!({
            "status": "ok",
            "sentence": sentence,
            "rounds": rounds,
            "stages": spec.stages.len(),
            "governance": gate,
            "narration": prose,
            "path": GEN_FILE,
        }),
    )
}

// ── Anthropic Messages API (direct provider call) ───────────────────────

fn call_anthropic(
    api_key: &str,
    model: &str,
    system_prompt: &str,
    messages: &[Value],
) -> Result<String, String> {
    let body = json!({
        "model": model,
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": messages,
    });
    let client = reqwest::blocking::Client::new();
    let resp = client
        .post(ANTHROPIC_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .map_err(|e| format!("anthropic request failed: {e}"))?;

    let status = resp.status();
    let v: Value = resp
        .json()
        .map_err(|e| format!("anthropic response not JSON: {e}"))?;
    if !status.is_success() {
        return Err(format!("anthropic API error {status}: {v}"));
    }
    // content: [{type:"text", text:"..."}]
    let text = v
        .get("content")
        .and_then(|c| c.as_array())
        .map(|arr| {
            arr.iter()
                .filter(|b| b.get("type").and_then(|t| t.as_str()) == Some("text"))
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("no text content in anthropic response: {v}"))?;
    Ok(text)
}

// ── Prompt construction ─────────────────────────────────────────────────

/// Skills callable from a pipeline stage (arity-1, single JSON arg), with the
/// doc, governance tags, and arg schema the proposer needs.
fn pipeline_callable_catalog() -> Value {
    let mut skills: Vec<&'static ix_registry::SkillDescriptor> = ix_registry::all()
        .filter(|s| s.inputs.len() == 1)
        .collect();
    skills.sort_by_key(|s| s.name);
    let rows: Vec<Value> = skills
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "doc": s.doc,
                "governance_tags": s.governance_tags,
                "args_schema": (s.json_schema)(),
            })
        })
        .collect();
    json!(rows)
}

fn build_system_prompt(schema: &Value, catalog: &Value) -> String {
    // The showcase is the only real checked-in spec — use it as the few-shot
    // exemplar to fight "instructional drift" toward simpler list shapes.
    let exemplar = r#"version: "1"
stages:
  baseline_stats:
    skill: stats
    args:
      data: [0.1, 0.4, 0.8, 1.2, 1.7]
  audit:
    skill: governance.check
    args:
      action: "review computed statistics"
    deps: [baseline_stats]"#;

    format!(
        "You compile a natural-language request into a canonical IX pipeline \
         (`ix.yaml`), and emit ONLY the YAML — no prose, no markdown fences.\n\n\
         CRITICAL SHAPE: a PipelineSpec is a MAP of stages keyed by id (NOT a \
         list of steps). Each stage names a `skill` from the catalog and a JSON \
         `args` object. Express data flow with `{{\"from\": \"stage_id.key\"}}` \
         inside args, and/or explicit `deps: [stage_id]`. Use ONLY skills that \
         appear in the catalog below, and match each skill's args_schema.\n\n\
         JSON SCHEMA for the document you must emit:\n{}\n\n\
         SKILL CATALOG (pipeline-callable skills only):\n{}\n\n\
         EXAMPLE of a valid PipelineSpec:\n{}\n",
        serde_json::to_string_pretty(schema).unwrap_or_default(),
        serde_json::to_string_pretty(catalog).unwrap_or_default(),
        exemplar,
    )
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Strip ```yaml / ``` fences and surrounding prose if the model added them.
fn strip_fences(s: &str) -> String {
    let t = s.trim();
    if let Some(start) = t.find("```") {
        let after = &t[start + 3..];
        let after = after.strip_prefix("yaml").unwrap_or(after);
        let after = after.strip_prefix("yml").unwrap_or(after);
        if let Some(end) = after.find("```") {
            return after[..end].trim().to_string();
        }
    }
    t.to_string()
}

/// Compact a stage output Value to a short one-line string for narration.
fn compact(v: &Value) -> String {
    let s = v.to_string();
    if s.len() > 80 {
        format!("{}…", &s[..80])
    } else {
        s
    }
}

fn gap_log_path() -> String {
    "state/thinking-machine/gaps.jsonl".to_string()
}

/// Append a dogfood-backlog row: a failure here is a concrete IX gap.
fn log_gap(sentence: &str, kind: &str, error: &str, rounds: u32) {
    let dir = std::path::Path::new("state/thinking-machine");
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let row = json!({
        "ts_ms": ts,
        "sentence": sentence,
        "kind": kind,
        "rounds": rounds,
        "error": error,
    });
    if let Ok(line) = serde_json::to_string(&row) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("gaps.jsonl"))
        {
            let _ = writeln!(f, "{line}");
        }
    }
}

fn emit_result(format: Format, payload: Value) -> Result<(), String> {
    output::emit(&payload, format).map_err(|e| format!("{e}"))
}
