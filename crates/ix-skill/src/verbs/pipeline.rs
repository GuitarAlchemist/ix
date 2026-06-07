//! `ix pipeline {new,validate,dag,run}` — DAG pipelines over `ix.yaml`.

use crate::output::{self, Format};
use ix_pipeline::executor::{execute, NoCache};
use ix_pipeline::lock::LockFile;
use ix_pipeline::lower::{lower, lower_with_gate};
use ix_pipeline::spec::PipelineSpec;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
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

/// Collect `{"from": "ref"}` reference strings anywhere inside a stage's args,
/// using the same single-key-object rule the executor applies
/// (`ix_pipeline::lower::resolve_from_refs`). These resolve to upstream outputs
/// at execution time, so a *template-time* gate cannot vet their values — it
/// surfaces them in the verdict instead of pretending it checked them.
fn collect_from_refs(args: &Value) -> Vec<String> {
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(map) => {
                if map.len() == 1 {
                    if let Some(Value::String(r)) = map.get("from") {
                        out.push(r.clone());
                        return; // a ref node has no further children to vet
                    }
                }
                for child in map.values() {
                    walk(child, out);
                }
            }
            Value::Array(arr) => arr.iter().for_each(|c| walk(c, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(args, &mut out);
    out
}

// ── Constitution tamper-evidence (RSI safe-loop research, 2026-06-07) ────────
//
// A silent edit to the constitution the gate enforces would change what counts
// as a violation without anyone noticing — the gate would keep returning
// "compliant" against moved goalposts. We hash the bytes actually loaded and
// compare to a pin committed in IX's OWN tree (state/governance/, NOT the
// Demerzel submodule: IX asserts which constitution it was reviewed against),
// failing CLOSED on mismatch. sha256 so the pin is independently verifiable.
// A legitimate constitution change is a deliberate governance act that re-pins;
// an undeclared edit is exactly what this catches. (reference_safe_rsi_loop_validated)

/// Result of checking a loaded constitution against its committed integrity pin.
#[derive(Debug, PartialEq, Eq)]
enum PinStatus {
    /// Loaded bytes match the committed pin.
    Verified,
    /// No pin is committed for this constitution — cannot verify. Reported in
    /// the verdict (never silently trusted) so the gap is visible.
    Unpinned,
}

impl PinStatus {
    fn as_str(&self) -> &'static str {
        match self {
            PinStatus::Verified => "verified",
            PinStatus::Unpinned => "unpinned",
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Strip CR bytes so the pin is over the constitution's CONTENT, not its
/// platform line-ending encoding. The gate reads raw file bytes, so a Windows
/// (CRLF) checkout and a Linux/CI (LF) checkout of the *same* constitution would
/// otherwise hash differently and a single committed pin could never match both.
/// A CRLF↔LF flip is benign — only real content changes must trip tamper-
/// evidence. (CI caught this: local CRLF hash ≠ committed LF-blob hash.)
fn normalize_newlines(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().copied().filter(|&b| b != b'\r').collect()
}

/// Compare a loaded constitution's bytes against an expected `<algo>:<hex>` pin.
/// Pure (no I/O) so the fail-closed branch is unit-testable. `expected = None`
/// → `Unpinned`. A present-but-mismatched pin → `Err` (fail-CLOSED). Only
/// `sha256:` pins are understood; an unknown algorithm prefix is a hard error,
/// never a silent pass. Hashes line-ending-normalized content so the pin is
/// reproducible across platforms.
fn check_pin(expected: Option<&str>, bytes: &[u8]) -> Result<PinStatus, String> {
    let Some(expected) = expected else {
        return Ok(PinStatus::Unpinned);
    };
    let Some(hex) = expected.strip_prefix("sha256:") else {
        return Err(format!(
            "constitution pin '{expected}' uses an unsupported algorithm \
             (only sha256: is understood) — refusing to execute"
        ));
    };
    let actual = sha256_hex(&normalize_newlines(bytes));
    if actual.eq_ignore_ascii_case(hex) {
        Ok(PinStatus::Verified)
    } else {
        Err(format!(
            "constitution integrity check FAILED: pinned sha256:{hex}, loaded sha256:{actual}. \
             The constitution was modified without updating its pin \
             (state/governance/constitution-pins.json). Refusing to execute — \
             re-pin deliberately if this change is intended."
        ))
    }
}

/// Locate the integrity-pin file. Resolution order: `IX_CONSTITUTION_PIN_FILE`
/// override (tests/CI) → walk UP from the constitution toward the repo root
/// where IX keeps `state/` (CWD-independent, so it works when the gate is
/// reached with an absolute `IX_GOVERNANCE_DIR`, e.g. the MCP path) → finally
/// the CWD-relative default (the common repo-root invocation).
fn find_pins_file(const_path: &Path) -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("IX_CONSTITUTION_PIN_FILE") {
        return Some(std::path::PathBuf::from(p));
    }
    let mut dir = const_path.parent();
    while let Some(d) = dir {
        let candidate = d.join("state/governance/constitution-pins.json");
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    let cwd_default = std::path::PathBuf::from("state/governance/constitution-pins.json");
    cwd_default.is_file().then_some(cwd_default)
}

/// Extract the `<algo>:<hex>` pin for `filename` from already-read pin-file
/// text. A genuinely missing `constitutions[filename]` entry → `Ok(None)`
/// (Unpinned is legitimate — not every constitution must be pinned). But a
/// MALFORMED file is an `Err` (fail-CLOSED): a present-but-corrupt pin must not
/// silently degrade to "unpinned", or an attacker could *truncate* the pin file
/// (rather than match it) to bypass the gate. (Codex #80 P2.) Pure for testing.
fn pin_lookup(pins_text: &str, filename: &str) -> Result<Option<String>, String> {
    let pins: Value = serde_json::from_str(pins_text).map_err(|e| {
        format!(
            "constitution pin file is present but malformed JSON: {e} — refusing to execute (fail-closed)"
        )
    })?;
    Ok(pins
        .get("constitutions")
        .and_then(|c| c.get(filename))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string()))
}

/// The committed `<algo>:<hex>` pin for a constitution filename. Distinguishes
/// three cases: no pin file anywhere → `Ok(None)` (legitimately unpinned); pin
/// file present but unreadable/malformed → `Err` (fail-CLOSED); file readable
/// but no entry for this constitution → `Ok(None)` (legitimately unpinned).
fn pin_for(filename: &str, const_path: &Path) -> Result<Option<String>, String> {
    let Some(pins_file) = find_pins_file(const_path) else {
        return Ok(None); // no pin file → unpinned (legitimately absent)
    };
    let txt = std::fs::read_to_string(&pins_file).map_err(|e| {
        format!(
            "constitution pin file present but unreadable ({}): {e} — refusing to execute (fail-closed)",
            pins_file.display()
        )
    })?;
    pin_lookup(&txt, filename)
}

/// Verify the constitution the gate is about to trust against its committed pin.
/// Fail-CLOSED on mismatch OR on a present-but-corrupt pin file; `Unpinned` only
/// when no pin is genuinely committed for this constitution.
fn verify_constitution_pin(const_path: &Path, bytes: &[u8]) -> Result<PinStatus, String> {
    let filename = const_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    check_pin(pin_for(filename, const_path)?.as_deref(), bytes)
}

/// Load + integrity-check the constitution at `{gov_dir}/constitutions/
/// default.constitution.md`. Returns the parsed constitution alongside its pin
/// status, or a fail-closed error (unreadable, or tampered against its pin).
/// Shared by the template-time `governance_gate` and the execution-time
/// `ConstitutionGate` so both enforce the same tamper-evidence.
fn load_checked_constitution() -> Result<(ix_governance::Constitution, PinStatus), String> {
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let const_path = format!("{gov_dir}/constitutions/default.constitution.md");
    let bytes = std::fs::read(&const_path).map_err(|e| {
        format!("cannot read constitution for integrity check ({const_path}): {e} — refusing to execute")
    })?;
    let status = verify_constitution_pin(Path::new(&const_path), &bytes)?;
    // Parse the SAME bytes we just hashed — never re-read the file. A second
    // `Constitution::load(path)` would open a time-of-check/time-of-use gap: an
    // attacker could swap the file between the pin check and the load, so the
    // verified bytes and the executed constitution would differ. (P0, review.)
    let content = String::from_utf8(bytes)
        .map_err(|e| format!("constitution is not valid UTF-8: {e} — refusing to execute"))?;
    let constitution = ix_governance::Constitution::parse_str(&content).map_err(|e| {
        format!("cannot verify governance (constitution parse failed: {e}) — refusing to execute")
    })?;
    Ok((constitution, status))
}

/// Fail-closed constitutional review of a spec before it executes. Every
/// stage's skill must (a) carry governance tags and (b) pass
/// `Constitution::check_action`. A missing constitution, an untagged skill, or
/// any non-compliant stage → reject. Shared by `ix pipeline run` and
/// `ix pipeline compile` so every compiled spec is reviewed on the canonical
/// path.
///
/// SCOPE — this is a *template-time* gate: it inspects the spec as written. A
/// stage arg supplied by a `{"from": "upstream[.key]"}` ref is only resolved at
/// execution time, so its runtime value is NOT vetted here; such refs are
/// surfaced in the verdict as `unvetted_runtime_inputs`. Gating the *resolved*
/// args inside the executor is tracked as a gap (gaps.jsonl:
/// governance-resolved-args) — the driver for pushing this check into
/// `ix-pipeline::execute()`.
pub(crate) fn governance_gate(spec: &PipelineSpec) -> Result<Value, String> {
    let (constitution, integrity) = load_checked_constitution()?;

    let mut cited: Vec<Value> = Vec::new();
    let mut unvetted: Vec<Value> = Vec::new();
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
        // Surface runtime-resolved inputs the template-time gate could not vet.
        let refs = collect_from_refs(&stage.args);
        if !refs.is_empty() {
            unvetted.push(json!({ "stage": id, "from_refs": refs }));
        }
    }
    Ok(json!({
        "verdict": "compliant",
        "articles_cited": cited,
        "unvetted_runtime_inputs": unvetted,
        "constitution_integrity": integrity.as_str(),
    }))
}

/// A [`StageGate`](ix_pipeline::gate::StageGate) backed by the Demerzel
/// constitution. Where `governance_gate` is the *template-time* pre-flight,
/// this runs at *execution time* on each stage's RESOLVED args — so a
/// `{"from"}` ref that supplied a destructive operation is finally vetted with
/// the real value the skill is about to run with. The durable closure of the
/// PR #77 review P1. Shares `governance_action` with the pre-flight so both
/// speak the same constitutional vocabulary.
pub(crate) struct ConstitutionGate {
    constitution: ix_governance::Constitution,
}

impl ConstitutionGate {
    /// Load + integrity-check the default constitution (honoring
    /// `IX_GOVERNANCE_DIR`). Fail-closed: an unreadable constitution OR one that
    /// fails its committed integrity pin is an error, never a silent allow.
    pub(crate) fn load() -> Result<std::sync::Arc<Self>, String> {
        let (constitution, _integrity) = load_checked_constitution()?;
        Ok(std::sync::Arc::new(Self { constitution }))
    }
}

impl ix_pipeline::gate::StageGate for ConstitutionGate {
    fn check(&self, stage_id: &str, skill: &str, resolved_args: &Value) -> Result<(), String> {
        let tags = ix_registry::by_name(skill)
            .map(|d| d.governance_tags)
            .unwrap_or(&[]);
        let action = format!(
            "{} as pipeline stage '{stage_id}'",
            governance_action(skill, resolved_args, tags)
        );
        let result = self.constitution.check_action(&action);
        if !result.compliant {
            return Err(format!(
                "skill '{skill}' is non-compliant on resolved args: {}",
                result.warnings.join("; ")
            ));
        }
        Ok(())
    }
}

/// `ix pipeline run [-f FILE] [--json]` — execute the pipeline.
///
/// When `stream_ndjson` is true, emits NDJSON events (`start`, `stage_start`,
/// `stage_complete`, `done`) to stdout in real time as the DAG runs.
pub fn run(file: Option<&str>, stream_ndjson: bool, format: Format) -> Result<(), String> {
    let path: &str = file.unwrap_or(DEFAULT_FILE);
    let spec = PipelineSpec::from_file(path).map_err(|e| format!("loading {path}: {e}"))?;
    // Template-time pre-flight: fail fast on literal violations before ANY
    // stage runs (prevents partial side effects from a multi-stage pipeline).
    governance_gate(&spec).map_err(|e| format!("governance gate: {e}"))?;
    // Execution-time gate: lower WITH a constitution-backed StageGate so each
    // stage's RESOLVED args (post-`{"from"}`-resolution) are vetted just before
    // its skill runs — closes the template-only gate's ref blind spot (PR #77
    // review P1). A `{"from"}` ref that supplies a destructive op is caught here.
    let gate = ConstitutionGate::load().map_err(|e| format!("governance gate: {e}"))?;
    let dag = lower_with_gate(&spec, gate).map_err(|e| format!("lowering {path}: {e}"))?;
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

    // ── Execution-time resolved-args gate (PR #77 review P1 closure) ─────
    //
    // A single literal end-to-end "{from}-ref resolves to delete → rejected"
    // test isn't constructible: no registry skill emits a constitutional
    // trigger word as output without a literal that an *earlier* gate already
    // catches — which is exactly why the P1 is bounded. So closure is proven
    // compositionally: (A) the gate receives RESOLVED args, (B) ConstitutionGate
    // rejects a resolved destructive op, (C) a rejection aborts before the skill
    // runs. A ∘ B ∘ C ⇒ a from-ref-supplied destructive op is caught at exec.

    #[test]
    fn resolved_gate_sees_post_resolution_args_not_the_template() {
        use ix_pipeline::executor::{execute, NoCache};
        use ix_pipeline::gate::StageGate;
        use ix_pipeline::lower::lower_with_gate;
        use std::sync::{Arc, Mutex};

        struct Spy(Arc<Mutex<Vec<(String, Value)>>>);
        impl StageGate for Spy {
            fn check(&self, stage_id: &str, _skill: &str, resolved: &Value) -> Result<(), String> {
                self.0
                    .lock()
                    .unwrap()
                    .push((stage_id.to_string(), resolved.clone()));
                Ok(())
            }
        }

        let seen = Arc::new(Mutex::new(Vec::new()));
        let gate: Arc<dyn StageGate> = Arc::new(Spy(seen.clone()));

        // `sink`'s data is a {from: "src.mean"} ref → resolves to src's mean (4.0).
        let yaml = r#"
version: "1"
stages:
  src:
    skill: stats
    args: { data: [2.0, 4.0, 6.0] }
  sink:
    skill: stats
    args: { data: { from: "src.mean" } }
"#;
        let spec = PipelineSpec::from_yaml_str(yaml).expect("parse spec");
        let dag = lower_with_gate(&spec, gate).expect("lower");
        // Execution itself may error (a scalar isn't a valid stats input); we
        // assert only what the gate SAW before `sink`'s skill ran.
        let _ = execute(&dag, &Default::default(), &NoCache);

        let recorded = seen.lock().unwrap();
        let sink = recorded
            .iter()
            .find(|(id, _)| id == "sink")
            .expect("gate was consulted for sink");
        assert_eq!(
            sink.1,
            json!({ "data": 4.0 }),
            "gate must see RESOLVED args (src.mean=4.0), not the {{from}} template: {:?}",
            sink.1
        );
    }

    #[test]
    fn constitution_gate_rejects_resolved_destructive_op() {
        use ix_pipeline::gate::StageGate;
        let gate = ConstitutionGate {
            constitution: default_constitution(),
        };
        assert!(
            gate.check("s", "cache", &json!({ "operation": "delete", "key": "x" }))
                .is_err(),
            "a resolved delete must be rejected at execution time (Article 3)"
        );
        assert!(
            gate.check("s", "cache", &json!({ "operation": "keys" }))
                .is_ok(),
            "a benign resolved op must pass"
        );
    }

    #[test]
    fn rejecting_gate_aborts_stage_before_skill_runs() {
        use ix_pipeline::executor::{execute, NoCache};
        use ix_pipeline::gate::StageGate;
        use ix_pipeline::lower::lower_with_gate;
        use std::sync::Arc;

        struct DenyAll;
        impl StageGate for DenyAll {
            fn check(&self, stage_id: &str, _skill: &str, _r: &Value) -> Result<(), String> {
                Err(format!("denied {stage_id}"))
            }
        }
        let yaml = r#"
version: "1"
stages:
  only:
    skill: stats
    args: { data: [1.0, 2.0, 3.0] }
"#;
        let spec = PipelineSpec::from_yaml_str(yaml).unwrap();
        let dag = lower_with_gate(&spec, Arc::new(DenyAll)).unwrap();
        let err = execute(&dag, &Default::default(), &NoCache).unwrap_err();
        assert!(
            format!("{err}").contains("governance: denied only"),
            "a gate rejection must abort execution with the governance reason: {err}"
        );
    }

    #[test]
    fn collect_from_refs_matches_executor_single_key_rule() {
        // The gate surfaces {"from": "ref"} inputs it cannot vet at template
        // time. Detection must mirror the executor (single-key {from:string});
        // a multi-key object that happens to carry a `from` field is NOT a ref.
        let args = json!({
            "operation": { "from": "prep.op" },
            "data": [ { "from": "src.values" }, 1, 2 ],
            "not_a_ref": { "from": "x", "extra": 1 },
            "plain": "delete"
        });
        let mut refs = collect_from_refs(&args);
        refs.sort();
        assert_eq!(
            refs,
            vec!["prep.op".to_string(), "src.values".to_string()],
            "only single-key {{from:string}} nodes are refs; got {refs:?}"
        );
    }

    // ── Constitution tamper-evidence (integrity pin) ────────────────────────

    #[test]
    fn check_pin_verifies_match_and_fails_closed_on_tamper() {
        let bytes = b"WE THE GOVERNED, in order to form a more perfect alignment...";
        let good = format!("sha256:{}", sha256_hex(bytes));
        assert_eq!(
            check_pin(Some(&good), bytes).unwrap(),
            PinStatus::Verified,
            "matching bytes verify"
        );
        // A single flipped byte must fail CLOSED (Err), not pass.
        assert!(
            check_pin(
                Some(&good),
                b"WE THE GOVERNED, in order to form a more perfect alignment!"
            )
            .is_err(),
            "tampered bytes must fail closed"
        );
        // Case-insensitive hex comparison.
        assert_eq!(
            check_pin(
                Some(&good.to_uppercase().replace("SHA256:", "sha256:")),
                bytes
            )
            .unwrap(),
            PinStatus::Verified
        );
        // An unknown algorithm prefix is a hard error, never a silent pass.
        assert!(
            check_pin(Some("md5:abc123"), bytes).is_err(),
            "unsupported algo must error, not silently allow"
        );
        // No pin committed → Unpinned (reported, never silently trusted).
        assert_eq!(check_pin(None, bytes).unwrap(), PinStatus::Unpinned);
    }

    #[test]
    fn check_pin_is_line_ending_independent() {
        // The gate reads raw file bytes; a Windows (CRLF) and a Linux/CI (LF)
        // checkout of the SAME constitution must verify against ONE committed
        // pin. Regression for the CI failure where the local CRLF hash differed
        // from the committed LF-blob hash.
        let lf = b"Article 1: Truth\nArticle 2: Reversibility\n".to_vec();
        let crlf = b"Article 1: Truth\r\nArticle 2: Reversibility\r\n".to_vec();
        let pin = format!("sha256:{}", sha256_hex(&normalize_newlines(&lf)));
        assert_eq!(check_pin(Some(&pin), &lf).unwrap(), PinStatus::Verified);
        assert_eq!(
            check_pin(Some(&pin), &crlf).unwrap(),
            PinStatus::Verified,
            "CRLF and LF of identical content must verify against the same pin"
        );
    }

    #[test]
    fn default_constitution_matches_committed_pin() {
        // Binds the committed pin to the ACTUAL constitution file: editing the
        // constitution without re-pinning fails HERE (in CI), beyond the
        // runtime gate. The pin lives in IX's tree; the constitution in the
        // Demerzel submodule — skip loudly if tested outside the workspace.
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let const_path = root.join("governance/demerzel/constitutions/default.constitution.md");
        let pins_path = root.join("state/governance/constitution-pins.json");
        let (Ok(bytes), Ok(pins_txt)) = (
            std::fs::read(&const_path),
            std::fs::read_to_string(&pins_path),
        ) else {
            eprintln!(
                "SKIP default_constitution_matches_committed_pin: files absent (outside workspace)"
            );
            return;
        };
        let pins: Value = serde_json::from_str(&pins_txt).expect("pins file is valid JSON");
        let expected = pins["constitutions"]["default.constitution.md"].as_str();
        assert!(
            expected.is_some(),
            "default.constitution.md must carry a committed pin"
        );
        assert_eq!(
            check_pin(expected, &bytes).unwrap(),
            PinStatus::Verified,
            "committed pin must match the actual constitution — re-pin deliberately if the change is intended"
        );
    }

    #[test]
    fn pin_lookup_fails_closed_on_corrupt_file_but_not_on_missing_entry() {
        // A present-but-corrupt pin file must ERROR (fail-closed): an attacker
        // who can truncate the pin file must not be able to downgrade the gate
        // to "unpinned". (Codex #80 P2.)
        assert!(
            pin_lookup("{\"constitutions\": {", "default.constitution.md").is_err(),
            "malformed JSON must fail closed, not degrade to unpinned"
        );
        assert!(
            pin_lookup("not json at all", "default.constitution.md").is_err(),
            "garbage must fail closed"
        );
        // A well-formed file that simply has no entry for THIS constitution is a
        // legitimate "unpinned" — not every constitution must be pinned.
        assert_eq!(
            pin_lookup(
                "{\"constitutions\": {\"other.md\": \"sha256:abc\"}}",
                "default.constitution.md"
            )
            .unwrap(),
            None
        );
        // A present entry is returned verbatim.
        assert_eq!(
            pin_lookup(
                "{\"constitutions\": {\"default.constitution.md\": \"sha256:deadbeef\"}}",
                "default.constitution.md"
            )
            .unwrap(),
            Some("sha256:deadbeef".to_string())
        );
    }
}
