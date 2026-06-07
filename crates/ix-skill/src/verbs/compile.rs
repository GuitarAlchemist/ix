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
/// Hard cap on propose→repair rounds. Each round is a fresh blocking POST and
/// `max_rounds` is caller-supplied (the MCP schema sets no maximum), so clamp
/// it to bound total provider calls to `MAX_REPAIR_ROUNDS + 1`. (PR #77 review P2.)
const MAX_REPAIR_ROUNDS: u32 = 5;

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
    // Bound total provider calls regardless of caller-supplied value. (PR #77 review P2.)
    let max_rounds = max_rounds.min(MAX_REPAIR_ROUNDS);

    let skills = pipeline_callable_skills();

    // ── Semantic-coverage gate (executable, pre-LLM) ─────────────────────
    // A capable model will confabulate a structurally-valid spec from
    // unrelated skills for an out-of-domain request (first dogfood finding,
    // 2026-06-06). `lower()` + governance check STRUCTURE, not coverage, so
    // we score intent↔catalog relevance with IX's own TF-IDF *before* calling
    // the LLM, and log a capability gap instead of confabulating.
    let cov_min: f64 = std::env::var("IX_THINKER_COVERAGE_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.08);
    let (cov_max, cov_top) = coverage_score(sentence, &skills);
    if cov_max < cov_min {
        log_gap(
            sentence,
            "coverage",
            &format!("no IX skill covers this intent (max TF-IDF cosine {cov_max:.3} < {cov_min}); nearest: {cov_top:?}"),
            0,
        );
        return emit_result(
            format,
            json!({
                "status": "out_of_domain",
                "sentence": sentence,
                "coverage_max": cov_max,
                "coverage_threshold": cov_min,
                "nearest_skills": cov_top.iter().map(|(n, s)| json!({ "skill": n, "score": s })).collect::<Vec<_>>(),
                "logged_to": gap_log_path(),
                "note": "No IX skill plausibly serves this request — logged as a capability gap instead of confabulating a spec.",
            }),
        );
    }

    let catalog = catalog_value(&skills);
    let schema = build_schema();
    let system_prompt = build_system_prompt(&schema, &catalog);

    // The proposer is a closure over the live Anthropic call; `resolve_spec`
    // drives the propose → validate → repair sub-loop and is unit-tested with a
    // scripted proposer (see tests::repair_*).
    let mut proposer = |msgs: &[Value]| call_anthropic(&api_key, &model, &system_prompt, msgs);

    let (spec, rounds_used) = match resolve_spec(&mut proposer, sentence, max_rounds)? {
        ResolveOutcome::Compiled { spec, rounds } => (spec, rounds),
        ResolveOutcome::NoCoverage { reason, rounds } => {
            // Best-effort LLM relevance (fail-OPEN): refuses only when the model
            // voluntarily emits the NO_COVERAGE sentinel, catching lexical-
            // collision cases the TF-IDF pre-gate cannot. A confabulated
            // structurally-valid spec for an out-of-domain request still passes
            // here (see gaps.jsonl + plan D4); real embeddings coverage is the
            // durable fix. Routing role, not a truth oracle (LLM self-judge
            // <25% TNR — cf. feedback_llm_judge_panel_failclosed).
            log_gap(sentence, "semantic-noncoverage", &reason, rounds);
            return emit_result(
                format,
                json!({
                    "status": "out_of_domain",
                    "sentence": sentence,
                    "reason": reason,
                    "detected_by": "llm-relevance",
                    "rounds": rounds,
                    "logged_to": gap_log_path(),
                    "note": "No IX skill serves this request (semantic relevance) — refused instead of confabulating.",
                }),
            );
        }
        ResolveOutcome::TranslateFailed { error, rounds } => {
            log_gap(sentence, "translate", &error, rounds);
            return emit_result(
                format,
                json!({
                    "status": "translate_failed",
                    "sentence": sentence,
                    "rounds": rounds,
                    "error": error,
                    "logged_to": gap_log_path(),
                }),
            );
        }
    };

    // ── Fail-closed governance gate (template-time, before any execution) ─
    // Shared with `ix pipeline run` so every compiled spec is reviewed on the
    // canonical path. NOTE: this gate inspects the *unresolved* spec; a stage
    // whose operation is supplied by a {"from": "upstream"} ref is only vetted
    // at execution time once the ref resolves — resolved-value gating in the
    // executor is tracked as a gap (see gaps.jsonl: governance-resolved-args).
    match crate::verbs::pipeline::governance_gate(&spec) {
        Ok(gate) => {
            // Persist the compiled spec for inspection / `ix pipeline run`.
            let yaml = spec.to_yaml_string().map_err(|e| format!("{e}"))?;
            std::fs::write(GEN_FILE, &yaml).map_err(|e| format!("writing {GEN_FILE}: {e}"))?;

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

/// Outcome of the propose → validate → repair sub-loop.
#[derive(Debug)]
enum ResolveOutcome {
    Compiled { spec: PipelineSpec, rounds: u32 },
    NoCoverage { reason: String, rounds: u32 },
    TranslateFailed { error: String, rounds: u32 },
}

/// Drive the bounded self-repair loop: call the proposer, honor a `NO_COVERAGE`
/// sentinel, parse + lower, and on a typed validation error feed it back
/// verbatim for up to `max_rounds` repair attempts. The proposer is injected
/// (a closure over the live API call in `compile`, a scripted fn in tests) so
/// the loop is deterministically testable without a network call. `Err` is
/// reserved for hard proposer failures (e.g. HTTP), not validation failures.
fn resolve_spec(
    proposer: &mut dyn FnMut(&[Value]) -> Result<String, String>,
    sentence: &str,
    max_rounds: u32,
) -> Result<ResolveOutcome, String> {
    let mut messages: Vec<Value> = vec![json!({ "role": "user", "content": sentence })];
    let mut last_error = String::new();

    for round in 0..=max_rounds {
        let reply = proposer(&messages)?;

        if let Some(rest) = reply.trim().strip_prefix("NO_COVERAGE") {
            let reason = rest.trim_start_matches([':', ' ', '\n']).trim().to_string();
            return Ok(ResolveOutcome::NoCoverage {
                reason,
                rounds: round,
            });
        }

        match parse_and_lower(&strip_fences(&reply)) {
            Ok(spec) => {
                return Ok(ResolveOutcome::Compiled {
                    spec,
                    rounds: round,
                })
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
    Ok(ResolveOutcome::TranslateFailed {
        error: last_error,
        rounds: max_rounds,
    })
}

/// Parse YAML → `PipelineSpec` and lower it. Returns the spec or a verbatim
/// error string (the exact text the proposer must repair against).
fn parse_and_lower(yaml: &str) -> Result<PipelineSpec, String> {
    let spec = PipelineSpec::from_yaml_str(yaml).map_err(|e| format!("parse/shape error: {e}"))?;
    lower(&spec).map_err(|e| format!("lowering error: {e}"))?;
    Ok(spec)
}

/// A resolved-args `StageGate` rejection surfaces from `execute()` as a
/// `ComputeError` carrying the `: governance: ` marker that `lower_with_gate`
/// injects (see `ix_pipeline::lower`). This distinguishes it from an ordinary
/// skill/execution failure so the caller emits a structured
/// `governance_rejected` verdict instead of a bare stderr error. (Codex #78 P2.)
fn is_resolved_governance_rejection(execute_err: &str) -> bool {
    execute_err.contains(": governance: ")
}

/// Execute the lowered pipeline and narrate the run back in prose.
fn execute_and_narrate(
    spec: &PipelineSpec,
    sentence: &str,
    rounds: u32,
    gate: Value,
    format: Format,
) -> Result<(), String> {
    // Execution-time governance on RESOLVED args (PR #77 review P1): the same
    // constitution-backed gate `ix pipeline run` uses, so executing a compiled
    // pipeline cannot smuggle a destructive op through a `{"from"}` ref.
    let gate_impl = crate::verbs::pipeline::ConstitutionGate::load()?;
    let dag = ix_pipeline::lower::lower_with_gate(spec, gate_impl)
        .map_err(|e| format!("re-lower: {e}"))?;
    let levels = dag.parallel_levels();
    let result = match execute(&dag, &Default::default(), &NoCache) {
        Ok(r) => r,
        Err(e) => {
            let msg = format!("{e}");
            // A resolved-args StageGate rejection (e.g. a `{from}` ref resolved
            // to a destructive op) must be reported with the SAME structured
            // `governance_rejected` shape as the template-time gate — otherwise
            // the JSON consumer (the `ix_nl_to_pipeline` MCP wrapper) sees empty
            // stdout instead of a verdict. (Codex #78 P2.)
            if is_resolved_governance_rejection(&msg) {
                log_gap(sentence, "governance-resolved", &msg, rounds);
                return emit_result(
                    format,
                    json!({
                        "status": "governance_rejected",
                        "sentence": sentence,
                        "rounds": rounds,
                        "reason": msg,
                        "detected_by": "resolved-args-gate",
                        "logged_to": gap_log_path(),
                        "note": "fail-closed: a stage's resolved args were refused mid-execution.",
                    }),
                );
            }
            return Err(format!("execution: {msg}"));
        }
    };

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
            let skill = spec
                .stages
                .get(*id)
                .map(|s| s.skill.as_str())
                .unwrap_or("?");
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
    // Explicit timeouts: reqwest's default blocking client has none, so a
    // stalled connection would hang the call (and the MCP handler that waits on
    // the child `ix`) unboundedly. (PR #77 review P2.)
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .map_err(|e| format!("building HTTP client: {e}"))?;
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

/// Skills callable from a pipeline stage (arity-1, single JSON arg).
fn pipeline_callable_skills() -> Vec<&'static ix_registry::SkillDescriptor> {
    let mut skills: Vec<&'static ix_registry::SkillDescriptor> =
        ix_registry::all().filter(|s| s.inputs.len() == 1).collect();
    skills.sort_by_key(|s| s.name);
    skills
}

/// The catalog the proposer sees: name, doc, governance tags, and arg schema.
fn catalog_value(skills: &[&'static ix_registry::SkillDescriptor]) -> Value {
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

/// Executable semantic-coverage score: the max TF-IDF cosine between the
/// request and any pipeline-callable skill's "name + doc". Dogfoods
/// `ix-supervised`'s own `TfidfVectorizer` — deterministic, NOT an LLM judging
/// its own output. Returns `(max_score, top-3 matches)` so a rejection is
/// auditable rather than a silent block.
fn coverage_score(
    sentence: &str,
    skills: &[&'static ix_registry::SkillDescriptor],
) -> (f64, Vec<(String, f64)>) {
    use ix_supervised::text::TfidfVectorizer;
    // Strip stopwords first: raw lexical TF-IDF lets "the/and/to/a" match many
    // skill docs and floats out-of-domain requests up to ~0.25 (measured). With
    // stopwords gone, only CONTENT words count, so an out-of-domain query whose
    // nouns/verbs appear in no skill doc collapses toward 0.
    let corpus: Vec<String> = skills
        .iter()
        .map(|s| content_terms(&format!("{} {}", s.name.replace(['.', '_'], " "), s.doc)))
        .collect();
    let corpus_refs: Vec<&str> = corpus.iter().map(|s| s.as_str()).collect();
    let query = content_terms(sentence);
    let mut vectorizer = TfidfVectorizer::new();
    let m = vectorizer.fit_transform(&corpus_refs); // n × vocab
    let q = vectorizer.transform(&[query.as_str()]); // 1 × vocab
    let qrow = q.row(0);
    let mut scored: Vec<(String, f64)> = skills
        .iter()
        .enumerate()
        .map(|(i, s)| (s.name.to_string(), cosine(qrow, m.row(i))))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max = scored.first().map(|x| x.1).unwrap_or(0.0);
    scored.truncate(3);
    (max, scored)
}

/// Lowercase, split on non-alphanumerics, drop short tokens + English
/// stopwords. The stopword set is deliberately generic (no domain verbs like
/// "compute"/"analyze"/"run") so content signal survives.
fn content_terms(s: &str) -> String {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "that", "this", "from", "into", "over", "are", "was", "your",
        "you", "our", "their", "its", "it", "of", "to", "in", "on", "at", "by", "as", "is", "be",
        "an", "or", "me", "my", "we", "us", "then", "them", "these", "those", "some", "any", "all",
        "a", "i",
    ];
    s.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 3 && !STOP.contains(t))
        .collect::<Vec<_>>()
        .join(" ")
}

fn cosine(a: ndarray::ArrayView1<f64>, b: ndarray::ArrayView1<f64>) -> f64 {
    let na = a.dot(&a).sqrt();
    let nb = b.dot(&b).sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        a.dot(&b) / (na * nb)
    }
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
         COVERAGE FIRST: you may use ONLY skills from the catalog below. If the \
         request requires a capability that NO catalog skill provides (e.g. web \
         scraping, sending email, reading external files/URLs, network I/O, \
         image generation), do NOT substitute unrelated skills to seem helpful. \
         Instead reply with EXACTLY `NO_COVERAGE: <one line naming the missing \
         capability>` and nothing else.\n\n\
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
    // Truncate on a CHAR boundary. `s.len()`/`&s[..80]` are byte-indexed and
    // panic when byte 80 splits a multibyte UTF-8 sequence — skill outputs carry
    // non-ASCII (♭ ♯ … accents), so byte-slicing is a latent crash on the
    // `--run` narration path. (PR #77 review P2.)
    const MAX_CHARS: usize = 80;
    if s.chars().count() > MAX_CHARS {
        let head: String = s.chars().take(MAX_CHARS).collect();
        format!("{head}…")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coverage_ranks_in_domain_above_out_of_domain() {
        let skills = pipeline_callable_skills();
        assert!(!skills.is_empty(), "registry has pipeline-callable skills");
        let (in_dom, in_top) = coverage_score(
            "compute the mean and standard deviation of these numbers",
            &skills,
        );
        let (out_dom, _) =
            coverage_score("teleport to the moon and bake a chocolate cake", &skills);
        let (collision, ctop) = coverage_score(
            "scrape the front page of a news website and email me a summary",
            &skills,
        );
        eprintln!("coverage: in_domain={in_dom:.4} (top={in_top:?}) out_domain={out_dom:.4} lexical_collision={collision:.4} (top={ctop:?})");
        assert!(
            in_dom > out_dom,
            "in-domain {in_dom} must exceed out-of-domain {out_dom}"
        );
        // After stopword stripping, an out-of-domain query whose content words
        // appear in no skill doc must collapse near zero.
        assert!(
            out_dom < 0.05,
            "stopword-stripped out-of-domain should be ~0, got {out_dom}"
        );
        assert!(
            in_dom > 0.15,
            "in-domain should be clearly positive, got {in_dom}"
        );
        // KNOWN LIMITATION (pinned): lexical coverage canNOT catch partial
        // content-word collisions — "summary of a website" collides with skill
        // docs ("summary statistics", "ingest…page"), scoring well above the
        // gate. The fix is semantic-embedding coverage, not a higher lexical
        // threshold (which would false-block thin in-domain requests). Logged
        // to state/thinking-machine/gaps.jsonl as the next increment.
        assert!(
            collision > 0.1,
            "documents that lexical coverage misses the collision case; got {collision}"
        );
    }

    #[test]
    fn repair_loop_recovers_from_unknown_skill() {
        // Round 0: a spec naming a skill `lower()` rejects (UnknownSkill).
        // Round 1: a valid spec. The loop must feed the error back and recover.
        let replies = [
            "version: \"1\"\nstages:\n  s:\n    skill: no.such.skill\n    args: {}\n",
            "version: \"1\"\nstages:\n  s:\n    skill: stats\n    args: { data: [1.0, 2.0, 3.0] }\n",
        ];
        let mut calls = 0usize;
        let mut proposer = |_m: &[Value]| -> Result<String, String> {
            let r = replies[calls].to_string();
            calls += 1;
            Ok(r)
        };
        match resolve_spec(&mut proposer, "compute stats", 3).unwrap() {
            ResolveOutcome::Compiled { spec, rounds } => {
                assert_eq!(rounds, 1, "must repair on the 2nd attempt");
                assert!(spec.stages.contains_key("s"));
            }
            other => panic!("expected Compiled, got {other:?}"),
        }
        assert_eq!(calls, 2, "proposer called exactly twice (bad, then good)");
    }

    #[test]
    fn repair_loop_gives_up_after_max_rounds() {
        let mut calls = 0usize;
        let mut proposer = |_m: &[Value]| -> Result<String, String> {
            calls += 1;
            Ok(
                "version: \"1\"\nstages:\n  s:\n    skill: no.such.skill\n    args: {}\n"
                    .to_string(),
            )
        };
        match resolve_spec(&mut proposer, "x", 2).unwrap() {
            ResolveOutcome::TranslateFailed { rounds, error } => {
                assert_eq!(rounds, 2);
                assert!(
                    error.contains("no.such.skill"),
                    "verbatim error preserved: {error}"
                );
            }
            other => panic!("expected TranslateFailed, got {other:?}"),
        }
        assert_eq!(calls, 3, "1 initial attempt + 2 repair attempts");
    }

    #[test]
    fn resolve_honors_no_coverage_sentinel() {
        let mut proposer = |_m: &[Value]| -> Result<String, String> {
            Ok("NO_COVERAGE: web scraping not available".to_string())
        };
        match resolve_spec(&mut proposer, "scrape a site", 3).unwrap() {
            ResolveOutcome::NoCoverage { reason, rounds } => {
                assert_eq!(rounds, 0);
                assert!(reason.contains("web scraping"));
            }
            other => panic!("expected NoCoverage, got {other:?}"),
        }
    }

    #[test]
    fn compact_truncates_on_char_boundary_without_panicking() {
        // Regression for PR #77 review P2: a stage output whose JSON
        // stringification puts a multibyte char straddling byte 80 must not
        // panic (byte-indexed `&s[..80]` did). Build a string so the 80th byte
        // lands inside a multibyte char: a leading `"` + 78 ASCII + '♭' means
        // the '♭' occupies bytes 79..82, i.e. byte 80 is mid-char.
        let s = format!("{}{}", "a".repeat(78), "♭♯…tail");
        let v = Value::String(s);
        assert!(
            !v.to_string().is_char_boundary(80),
            "test precondition: byte 80 must split a multibyte char"
        );
        let out = compact(&v); // must not panic
        assert!(out.ends_with('…'), "truncated output is ellipsized: {out}");
        assert!(out.chars().count() <= 81, "≤80 chars + ellipsis: {out}");
    }

    #[test]
    fn resolved_governance_rejection_classified_from_real_executor_error() {
        // Codex #78 P2: a `--run` resolved-args gate rejection must be reported
        // as structured `governance_rejected` JSON, not a bare error. The
        // classifier keys on the marker `lower_with_gate` injects — bind it to
        // the REAL executor error so a marker change in ix-pipeline fails here.
        use ix_pipeline::executor::{execute, NoCache};
        use ix_pipeline::gate::StageGate;
        use ix_pipeline::lower::lower_with_gate;
        use std::sync::Arc;

        struct Deny;
        impl StageGate for Deny {
            fn check(&self, id: &str, _s: &str, _r: &Value) -> Result<(), String> {
                Err(format!("blocked {id}"))
            }
        }
        let spec = PipelineSpec::from_yaml_str(
            "version: \"1\"\nstages:\n  s:\n    skill: stats\n    args: { data: [1.0] }\n",
        )
        .unwrap();
        let dag = lower_with_gate(&spec, Arc::new(Deny)).unwrap();
        let err = format!(
            "{}",
            execute(&dag, &Default::default(), &NoCache).unwrap_err()
        );
        assert!(
            is_resolved_governance_rejection(&err),
            "a real gate rejection must classify as governance: {err}"
        );
        assert!(
            !is_resolved_governance_rejection("s: stats: invalid input shape"),
            "an ordinary execution error must NOT be misclassified as governance"
        );
    }

    // ── Embeddings front (D4): baseline of the CURRENT lexical TF-IDF gate ──
    //
    // Measures the live `coverage_score` against the validated probe corpus
    // (state/thinking-machine/coverage-probes.jsonl) at the production 0.08
    // threshold. This is the instrument: it quantifies the gap an embedding
    // gate must close — specifically that TF-IDF refuses far-OOD well but lets
    // NEAR-MISS out-of-domain requests (shared content words) through. The
    // asserted bounds lock that baseline; an embedding replacement is only
    // worth shipping if it raises near_miss TNR without dropping in-domain
    // recall below this line.
    #[test]
    fn coverage_baseline_tfidf_over_probe_corpus() {
        use std::collections::BTreeMap;
        use std::io::BufRead;

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../state/thinking-machine/coverage-probes.jsonl");
        // The corpus is a workspace-level state/ artifact, not packaged inside
        // the crate. In the normal workspace checkout it is always present; skip
        // loudly (rather than panic) if the crate is tested in isolation outside
        // the workspace tree. (Codex #79 P2.)
        let Ok(file) = std::fs::File::open(&path) else {
            eprintln!(
                "SKIP coverage_baseline_tfidf: probe corpus not found at {} (crate tested outside the workspace)",
                path.display()
            );
            return;
        };
        let skills = pipeline_callable_skills();
        let threshold = 0.08_f64; // the live IX_THINKER_COVERAGE_MIN default

        let (mut covered, mut false_reject) = (0u32, 0u32); // in_domain outcomes
        let mut by_kind: BTreeMap<String, (u32, u32)> = BTreeMap::new(); // ood kind -> (refused, confabulated)
        let (mut n_in, mut n_out) = (0u32, 0u32);

        for line in std::io::BufReader::new(file).lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let p: Value = serde_json::from_str(&line).unwrap();
            let sentence = p["sentence"].as_str().unwrap();
            let label = p["label"].as_str().unwrap();
            let kind = p["kind"].as_str().unwrap().to_string();
            let (score, _) = coverage_score(sentence, &skills);
            let predicted_in = score >= threshold;
            if label == "in_domain" {
                n_in += 1;
                if predicted_in {
                    covered += 1;
                } else {
                    false_reject += 1;
                }
            } else {
                n_out += 1;
                let e = by_kind.entry(kind).or_insert((0, 0));
                if predicted_in {
                    e.1 += 1; // confabulated: gate let an OOD request through
                } else {
                    e.0 += 1; // correctly refused
                }
            }
        }

        let recall = covered as f64 / n_in.max(1) as f64;
        let (refused_total, confab_total): (u32, u32) = by_kind
            .values()
            .fold((0, 0), |(a, b), (c, f)| (a + c, b + f));
        let tnr = refused_total as f64 / n_out.max(1) as f64;
        let kind_tnr = |k: &str| -> f64 {
            by_kind
                .get(k)
                .map(|(c, f)| *c as f64 / (*c + *f).max(1) as f64)
                .unwrap_or(0.0)
        };
        let near = kind_tnr("near_miss_ood");
        let far = kind_tnr("far_ood");

        eprintln!(
            "=== TF-IDF coverage baseline @ threshold {threshold} (n={}) ===",
            n_in + n_out
        );
        eprintln!(
            "in_domain   n={n_in:3} recall(pass)={recall:.3} false_reject={:.3}",
            false_reject as f64 / n_in.max(1) as f64
        );
        eprintln!("out_domain  n={n_out:3} TNR(refuse)={tnr:.3} confabulated={confab_total}");
        for (kind, (c, f)) in &by_kind {
            eprintln!(
                "  {kind:14} TNR={:.3} ({c} refused / {} total)",
                *c as f64 / (*c + *f).max(1) as f64,
                c + f
            );
        }
        eprintln!("HEADLINE gap: far_ood TNR={far:.3}  vs  near_miss TNR={near:.3}");

        // Corpus is real and non-trivial.
        assert!(
            n_in >= 80 && n_out >= 60,
            "corpus has both classes: in={n_in} out={n_out}"
        );
        // Measured baseline (2026-06-07, threshold 0.08): recall≈0.99,
        // overall TNR≈0.16, far_ood≈0.46, near_miss≈0.04 — the lexical pre-gate
        // confabulates ~84% of out-of-domain requests. Guardrails (NOT targets):
        // in-domain recall must stay high, and near-miss must remain the weakest
        // slice (the embedding gate's primary target). The poor absolute TNR is
        // recorded in the plan doc as the baseline the embedding work must beat;
        // raise these into TNR targets once embeddings land.
        assert!(
            recall >= 0.90,
            "in-domain recall must stay high — a coverage gate that false-blocks real work is worse than useless: recall={recall:.3}"
        );
        assert!(
            near < far,
            "near-miss OOD must be the weakest slice (the embedding gate's target): near={near:.3} far={far:.3}"
        );
        assert!(
            tnr < 0.50,
            "documents the baseline: the lexical pre-gate is a weak refuser (raise/flip this assertion when the embedding gate lands): TNR={tnr:.3}"
        );
    }
}
