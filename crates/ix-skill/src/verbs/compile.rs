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
///
/// Thin wrapper over [`compile_inner`] that guarantees **no dark outcomes**: a
/// translation that began but ended in a propagated `Err` (missing key, spec
/// serialization, disk write, an execution-time gate/lower/skill failure) is
/// still a terminal outcome. We log it to the ledger as `"error"` so it counts
/// in the yield denominator and can't silently inflate `yield_rate` by vanishing
/// — then re-propagate the `Err` so the CLI still exits non-zero with the
/// reason on stderr. The happy/refusal paths already logged via `emit_outcome`
/// and returned `Ok`, so this only fires on un-instrumented errors. (Review P1:
/// the metric+guardrail pair is only honest if the denominator is complete.)
pub fn compile(
    sentence: &str,
    max_rounds: u32,
    run_it: bool,
    format: Format,
) -> Result<(), String> {
    let outcome = compile_inner(sentence, max_rounds, run_it, format);
    if outcome.is_err() {
        // coverage_max is unknown/irrelevant for an operational error → NaN,
        // which serializes to JSON null (summarize_hits ignores the field).
        log_hit(sentence, "error", Some("operational"), f64::NAN, 0, None);
    }
    outcome
}

fn compile_inner(
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
    let (cov_max, cov_top, detector, cov_min) = score_coverage(sentence, &skills);
    if cov_max < cov_min {
        log_gap(
            sentence,
            "coverage",
            &format!("no IX skill covers this intent ({detector} score {cov_max:.3} < {cov_min}); nearest: {cov_top:?}"),
            0,
        );
        return emit_outcome(
            format,
            sentence,
            cov_max,
            json!({
                "status": "out_of_domain",
                "detected_by": detector,
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
            return emit_outcome(
                format,
                sentence,
                cov_max,
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
            return emit_outcome(
                format,
                sentence,
                cov_max,
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
                return emit_outcome(
                    format,
                    sentence,
                    cov_max,
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
            execute_and_narrate(&spec, sentence, rounds_used, cov_max, gate, format)
        }
        Err(rejection) => {
            log_gap(sentence, "governance", &rejection, rounds_used);
            emit_outcome(
                format,
                sentence,
                cov_max,
                json!({
                    "status": "governance_rejected",
                    "detected_by": "template-gate",
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
    coverage_max: f64,
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
                return emit_outcome(
                    format,
                    sentence,
                    coverage_max,
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

    emit_outcome(
        format,
        sentence,
        coverage_max,
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
pub(crate) fn pipeline_callable_skills() -> Vec<&'static ix_registry::SkillDescriptor> {
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

/// Coverage dispatcher: the in-process embedding scorer (bge-base-en-v1.5) when
/// the `embeddings` feature is built AND the model loads, else the lexical
/// TF-IDF pre-gate. The embedding tier raises OOD refusal ~4× at equal recall
/// (`state/thinking-machine/embedding-sweep-rust-results.md`) but is OPTIONAL —
/// TF-IDF is the always-available default + graceful fallback (set
/// `IX_THINKER_DISABLE_EMBED` to force it). Thresholds: embedding ≈ 0.45
/// (`IX_THINKER_EMBED_COVERAGE_MIN`), TF-IDF 0.08 (`IX_THINKER_COVERAGE_MIN`).
/// Returns `(score, top-3 nearest, detector tag, threshold)`.
fn score_coverage(
    sentence: &str,
    skills: &[&'static ix_registry::SkillDescriptor],
) -> (f64, Vec<(String, f64)>, &'static str, f64) {
    #[cfg(feature = "embeddings")]
    {
        if std::env::var_os("IX_THINKER_DISABLE_EMBED").is_none() {
            if let Some(mut ec) = crate::verbs::embed_coverage::EmbeddingCoverage::load(skills) {
                let (max, top) = ec.score(sentence);
                let min: f64 = std::env::var("IX_THINKER_EMBED_COVERAGE_MIN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0.45);
                return (max, top, "embedding-bge-base-en", min);
            }
            // model unavailable (offline / no ONNX) → fall through to TF-IDF.
        }
    }
    let (max, top) = coverage_score(sentence, skills);
    let min: f64 = std::env::var("IX_THINKER_COVERAGE_MIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.08);
    (max, top, "tfidf-coverage", min)
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

fn hits_log_path() -> String {
    "state/thinking-machine/hits.jsonl".to_string()
}

/// Build one `hits.jsonl` row. Pure (timestamp injected) so the writer's shape
/// stays unit-testable against the reader (`summarize_hits`).
fn hit_row(
    sentence: &str,
    outcome: &str,
    detected_by: Option<&str>,
    coverage_max: f64,
    rounds: u32,
    stages: Option<usize>,
    ts_ms: u128,
) -> Value {
    json!({
        "ts_ms": ts_ms,
        "sentence": sentence,
        "outcome": outcome,
        "detected_by": detected_by,
        "coverage_max": coverage_max,
        "rounds": rounds,
        "stages": stages,
    })
}

/// Append a translation-outcome row to the hits ledger. Where `gaps.jsonl`
/// records only FAILURES, `hits.jsonl` records EVERY terminal outcome —
/// success, refusal, and failure — so `summarize_hits` can report translation
/// *yield* alongside its refusal *guardrails* as a pair. A bare success
/// denominator would be Goodhart-bait: a loop maximizing "% compiled" games it
/// by confabulating specs for out-of-domain requests. Pairing yield with the
/// refusal rates makes that gaming visible (coverage-refusal collapses as yield
/// is inflated). Best-effort: instrumentation must never block or fail a
/// translation. (RSI safe-loop research 2026-06-07; the metric+guardrail PAIR
/// discipline — reference_safe_rsi_loop_validated.)
fn log_hit(
    sentence: &str,
    outcome: &str,
    detected_by: Option<&str>,
    coverage_max: f64,
    rounds: u32,
    stages: Option<usize>,
) {
    let dir = std::path::Path::new("state/thinking-machine");
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let row = hit_row(
        sentence,
        outcome,
        detected_by,
        coverage_max,
        rounds,
        stages,
        ts,
    );
    if let Ok(mut line) = serde_json::to_string(&row) {
        use std::io::Write;
        line.push('\n');
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("hits.jsonl"))
        {
            // ONE write_all of the complete line+newline. On an O_APPEND handle
            // a single write minimizes interleaving when two concurrent
            // compile() processes append at once (vs writeln!'s separate
            // content/newline writes, which can tear). The reader
            // (summarize_hits) additionally tolerates any torn line. (Review P1.)
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// The single choke point every terminal `compile` outcome flows through: log
/// the hit (derived from the SAME payload the caller returns, so the instrument
/// can't drift out of sync with the emitted verdict), then emit the result.
fn emit_outcome(
    format: Format,
    sentence: &str,
    coverage_max: f64,
    payload: Value,
) -> Result<(), String> {
    let outcome = payload
        .get("status")
        .and_then(|s| s.as_str())
        .unwrap_or("unknown");
    let detected_by = payload.get("detected_by").and_then(|s| s.as_str());
    let rounds = payload.get("rounds").and_then(|r| r.as_u64()).unwrap_or(0) as u32;
    let stages = payload
        .get("stages")
        .and_then(|s| s.as_u64())
        .map(|n| n as usize);
    log_hit(sentence, outcome, detected_by, coverage_max, rounds, stages);
    emit_result(format, payload)
}

/// `ix pipeline hits` — aggregate the translation ledger into a yield metric
/// paired with its refusal guardrails. A rising yield with a FALLING coverage-
/// refusal rate is the signature of the gate being loosened or the proposer
/// confabulating — the pair surfaces it. Rates are over UNLABELED production
/// outcomes (a drift signal); the calibrated, labelled version is the
/// coverage-baseline test over the probe corpus.
pub fn hits(format: Format) -> Result<(), String> {
    let summary = summarize_hits(std::path::Path::new(&hits_log_path()))?;
    emit_result(format, summary)
}

/// Read the hits ledger and compute the paired yield/guardrail summary.
/// Split from the verb so it is testable over a fixture path.
fn summarize_hits(path: &std::path::Path) -> Result<Value, String> {
    use std::io::BufRead;
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => {
            return Ok(json!({
                "status": "no_hits",
                "path": path.display().to_string(),
                "note": "no translations recorded yet (hits.jsonl absent).",
            }));
        }
    };
    let mut by_outcome: std::collections::BTreeMap<String, u32> = std::collections::BTreeMap::new();
    let mut total = 0u32;
    let mut skipped = 0u32;
    for line in std::io::BufReader::new(file).lines() {
        // Tolerate a torn/corrupt line (e.g. an interleaved append from a
        // concurrent compile, or invalid UTF-8) — skip it rather than failing
        // the whole query. "Best-effort instrumentation must never block": a
        // single bad line must not crash `ix pipeline hits` or the
        // ix_thinker_hits MCP tool. Skips are counted + surfaced. (Review P1.)
        let Ok(line) = line else {
            skipped += 1;
            continue;
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(row) = serde_json::from_str::<Value>(&line) else {
            skipped += 1;
            continue;
        };
        let outcome = row
            .get("outcome")
            .and_then(|s| s.as_str())
            .unwrap_or("unknown")
            .to_string();
        *by_outcome.entry(outcome).or_insert(0) += 1;
        total += 1;
    }
    let count = |k: &str| *by_outcome.get(k).unwrap_or(&0);
    let yield_n = count("ok") + count("compiled");
    let denom = total.max(1) as f64;
    Ok(json!({
        "status": "ok",
        "path": path.display().to_string(),
        "total": total,
        "skipped_lines": skipped,
        "by_outcome": by_outcome,
        // METRIC (maximize): fraction of requests that produced a runnable spec.
        "yield_rate": yield_n as f64 / denom,
        // GUARDRAILS (must NOT collapse as yield climbs — Goodhart tripwires):
        "coverage_refusal_rate": count("out_of_domain") as f64 / denom,
        "governance_refusal_rate": count("governance_rejected") as f64 / denom,
        "translate_fail_rate": count("translate_failed") as f64 / denom,
        "note": "Paired instrument: yield_rate is the metric; the *_refusal_rate fields are guardrails. A yield gain paired with a coverage-refusal drop is gate-loosening/confabulation, not improvement. Rates over UNLABELED outcomes — the calibrated metric is the coverage-baseline test.",
    }))
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

    // ── hits.jsonl — translation yield instrumented as a metric+guardrail PAIR ──

    /// Write rows in the EXACT shape `log_hit`/`hit_row` emit, so these tests
    /// bind the writer's format to the reader (`summarize_hits`).
    fn write_hits(rows: &[(&str, Option<&str>)]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for (i, (outcome, detected_by)) in rows.iter().enumerate() {
            let row = hit_row("req", outcome, *detected_by, 0.12, 0, None, i as u128);
            writeln!(f, "{}", serde_json::to_string(&row).unwrap()).unwrap();
        }
        f.flush().unwrap();
        f
    }

    #[test]
    fn summarize_hits_reports_yield_paired_with_guardrails() {
        // A healthy mix: 5 runnable (ok/compiled), 3 coverage refusals, 1
        // governance refusal, 1 translate failure → 10 total.
        let f = write_hits(&[
            ("ok", None),
            ("ok", None),
            ("compiled", None),
            ("compiled", None),
            ("compiled", None),
            ("out_of_domain", Some("tfidf-coverage")),
            ("out_of_domain", Some("llm-relevance")),
            ("out_of_domain", Some("tfidf-coverage")),
            ("governance_rejected", Some("template-gate")),
            ("translate_failed", None),
        ]);
        let s = summarize_hits(f.path()).unwrap();
        assert_eq!(s["total"], 10);
        assert_eq!(s["status"], "ok");
        assert!(
            (s["yield_rate"].as_f64().unwrap() - 0.5).abs() < 1e-9,
            "5/10 runnable: {s}"
        );
        assert!(
            (s["coverage_refusal_rate"].as_f64().unwrap() - 0.3).abs() < 1e-9,
            "3/10 OOD: {s}"
        );
        assert!((s["governance_refusal_rate"].as_f64().unwrap() - 0.1).abs() < 1e-9);
        assert!((s["translate_fail_rate"].as_f64().unwrap() - 0.1).abs() < 1e-9);
        // by_outcome is the raw census the rates derive from.
        assert_eq!(s["by_outcome"]["out_of_domain"], 3);
        assert_eq!(s["by_outcome"]["compiled"], 3);
    }

    #[test]
    fn summarize_hits_surfaces_confabulation_gaming() {
        // The whole point of the PAIR: a loop that games "% compiled" by
        // confabulating a spec for EVERY request (never refusing) shows a
        // perfect yield_rate=1.0 AND a coverage_refusal_rate that has collapsed
        // to 0.0 — the guardrail is the tripwire that exposes the gaming a bare
        // success denominator would hide.
        let gamed = write_hits(&[
            ("ok", None),
            ("compiled", None),
            ("ok", None),
            ("compiled", None),
        ]);
        let s = summarize_hits(gamed.path()).unwrap();
        assert!(
            (s["yield_rate"].as_f64().unwrap() - 1.0).abs() < 1e-9,
            "gamed yield looks perfect: {s}"
        );
        assert_eq!(
            s["coverage_refusal_rate"].as_f64().unwrap(),
            0.0,
            "guardrail collapsed to zero — the tripwire that catches confabulation: {s}"
        );
    }

    #[test]
    fn summarize_hits_absent_file_is_no_hits_not_error() {
        let missing = std::path::Path::new("definitely/not/here/hits.jsonl");
        let s = summarize_hits(missing).unwrap();
        assert_eq!(
            s["status"], "no_hits",
            "absent ledger is a state, not an error: {s}"
        );
    }

    #[test]
    fn summarize_hits_skips_torn_lines_without_crashing() {
        // Concurrent appends can interleave into a torn line. The reader must
        // skip it (counting the skip) rather than fail the whole query — else a
        // single bad line crashes `ix pipeline hits` / the MCP tool. (Review P1.)
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        let good =
            |o: &str| serde_json::to_string(&hit_row("r", o, None, 0.1, 0, None, 0)).unwrap();
        writeln!(f, "{}", good("ok")).unwrap();
        writeln!(f, "{{\"ts_ms\":123,\"outcome\":\"compi").unwrap(); // torn mid-write
        writeln!(f, "{}", good("out_of_domain")).unwrap();
        writeln!(f, "not json at all").unwrap();
        f.flush().unwrap();
        let s = summarize_hits(f.path()).unwrap();
        assert_eq!(
            s["status"], "ok",
            "a torn line must not fail the query: {s}"
        );
        assert_eq!(s["total"], 2, "only the 2 well-formed rows count: {s}");
        assert_eq!(
            s["skipped_lines"], 2,
            "both torn lines are counted as skipped: {s}"
        );
    }

    #[test]
    fn summarize_hits_error_outcomes_dilute_yield_not_vanish() {
        // The no-dark-outcomes guarantee: an operational error is logged as
        // "error" and counts in the denominator, so it lowers yield_rate rather
        // than silently inflating it by disappearing. (Review P1.)
        let f = write_hits(&[
            ("ok", None),
            ("compiled", None),
            ("error", Some("operational")),
        ]);
        let s = summarize_hits(f.path()).unwrap();
        assert_eq!(s["total"], 3);
        assert_eq!(
            s["by_outcome"]["error"], 1,
            "the error is visible in the census: {s}"
        );
        assert!(
            (s["yield_rate"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-9,
            "error sits in the denominator, diluting yield (2/3), not vanishing: {s}"
        );
    }

    // ── Embedding-gate validation SPIKE: which English embedder beats TF-IDF?
    //    (run: cargo test -p ix-skill --features embeddings embedding_sweep
    //     -- --nocapture) ──────────────────────────────────────────────────
    //
    // Feature-gated so CI (`--workspace`, default features) never pulls ONNX.
    // Proves fastembed-rs/ort builds + runs here, and sweeps several built-in
    // English retrieval embedders — each with ITS correct query/passage prefix
    // scheme — over the 184-probe corpus, reporting AUC + the operating point at
    // in-domain recall ≥ 0.99 (same methodology as the Python sweep). Go/no-go:
    // at least one must out-refuse the lexical baseline (recall 0.99 / TNR 0.163)
    // at equal recall. First run downloads each model (~270–670 MB).
    #[cfg(feature = "embeddings")]
    #[test]
    fn embedding_sweep_english_models_over_probe_corpus() {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        use std::io::BufRead;

        fn cosine_f32(a: &[f32], b: &[f32]) -> f64 {
            let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for (x, y) in a.iter().zip(b) {
                let (x, y) = (*x as f64, *y as f64);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot / (na.sqrt() * nb.sqrt())
            }
        }

        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../state/thinking-machine/coverage-probes.jsonl");
        let Ok(file) = std::fs::File::open(&path) else {
            eprintln!(
                "SKIP embedding_sweep: probe corpus not found at {}",
                path.display()
            );
            return;
        };
        let mut sentences = Vec::new();
        let mut labels: Vec<(bool, String)> = Vec::new(); // (is_in_domain, kind)
        for line in std::io::BufReader::new(file).lines() {
            let line = line.unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let p: Value = serde_json::from_str(&line).unwrap();
            sentences.push(p["sentence"].as_str().unwrap().to_string());
            labels.push((
                p["label"].as_str().unwrap() == "in_domain",
                p["kind"].as_str().unwrap().to_string(),
            ));
        }
        let skills = pipeline_callable_skills();
        let docs: Vec<String> = skills
            .iter()
            .map(|s| format!("{} {}", s.name.replace(['.', '_'], " "), s.doc))
            .collect();

        // Given per-query scores, compute AUC + the operating point at recall≥0.99.
        let eval = |scores: &[f64]| -> (f64, f64, f64, f64, f64, f64) {
            let pos: Vec<f64> = scores
                .iter()
                .zip(&labels)
                .filter(|(_, (i, _))| *i)
                .map(|(s, _)| *s)
                .collect();
            let neg: Vec<f64> = scores
                .iter()
                .zip(&labels)
                .filter(|(_, (i, _))| !*i)
                .map(|(s, _)| *s)
                .collect();
            let (mut wins, mut total) = (0.0f64, 0.0f64);
            for p in &pos {
                for n in &neg {
                    total += 1.0;
                    if p > n {
                        wins += 1.0;
                    } else if (p - n).abs() < 1e-12 {
                        wins += 0.5;
                    }
                }
            }
            let auc = if total > 0.0 { wins / total } else { 0.0 };
            let mut ps = pos.clone();
            ps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let drop = ((1.0 - 0.99) * ps.len() as f64).floor() as usize;
            let thr = ps[drop.min(ps.len().saturating_sub(1))];
            let recall = pos.iter().filter(|s| **s >= thr).count() as f64 / pos.len().max(1) as f64;
            let tnr = neg.iter().filter(|s| **s < thr).count() as f64 / neg.len().max(1) as f64;
            let kt = |k: &str| -> f64 {
                let t: Vec<f64> = scores
                    .iter()
                    .zip(&labels)
                    .filter(|(_, (i, kk))| !*i && kk == k)
                    .map(|(s, _)| *s)
                    .collect();
                t.iter().filter(|s| **s < thr).count() as f64 / t.len().max(1) as f64
            };
            (auc, thr, recall, tnr, kt("near_miss_ood"), kt("far_ood"))
        };

        // (name, model, query-prefix, passage-prefix). e5 uses query:/passage:;
        // bge/mxbai/arctic use a query instruction + bare passages; gte uses none.
        let instr = "Represent this sentence for searching relevant passages: ";
        let configs: Vec<(&str, EmbeddingModel, String, String)> = vec![
            (
                "multilingual-e5-base",
                EmbeddingModel::MultilingualE5Base,
                "query: ".into(),
                "passage: ".into(),
            ),
            (
                "bge-base-en-v1.5",
                EmbeddingModel::BGEBaseENV15,
                instr.into(),
                String::new(),
            ),
            (
                "gte-base-en-v1.5",
                EmbeddingModel::GTEBaseENV15,
                String::new(),
                String::new(),
            ),
            (
                "snowflake-arctic-embed-m",
                EmbeddingModel::SnowflakeArcticEmbedM,
                instr.into(),
                String::new(),
            ),
        ];

        eprintln!(
            "{:28} {:>5} {:>7} {:>6} {:>6} {:>6} {:>6}",
            "model (mean-top-3 cosine)", "AUC", "thr", "recall", "TNR", "near", "far"
        );
        let (mut best_tnr, mut best_name) = (0.0f64, "none");
        for (name, model_kind, qp, pp) in configs {
            let mut model = match TextEmbedding::try_new(
                InitOptions::new(model_kind).with_show_download_progress(true),
            ) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("{name:28} load FAILED: {e}");
                    continue;
                }
            };
            let passages: Vec<String> = docs.iter().map(|d| format!("{pp}{d}")).collect();
            let queries: Vec<String> = sentences.iter().map(|s| format!("{qp}{s}")).collect();
            let cat = model.embed(passages, None).expect("embed catalog");
            let qry = model.embed(queries, None).expect("embed queries");
            let scores: Vec<f64> = qry
                .iter()
                .map(|q| {
                    let mut c: Vec<f64> = cat.iter().map(|p| cosine_f32(q, p)).collect();
                    c.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    c.iter().take(3).sum::<f64>() / 3.0
                })
                .collect();
            let (auc, thr, recall, tnr, near, far) = eval(&scores);
            eprintln!("{name:28} {auc:.3} {thr:.4} {recall:.3} {tnr:.3} {near:.3} {far:.3}");
            if tnr > best_tnr {
                best_tnr = tnr;
                best_name = name;
            }
        }
        eprintln!(
            "{:28} {:>5} {:>7} {:>6} {:>6} {:>6} {:>6}",
            "TF-IDF baseline", "~.78", "-", "0.990", "0.163", "0.036", "0.458"
        );
        eprintln!("BEST embedder: {best_name} @ TNR {best_tnr:.3} (recall≈0.99)");

        // Go/no-go bar: at least one English embedder must out-refuse the lexical
        // baseline at equal recall — else embeddings aren't worth the ONNX dep.
        assert!(
            best_tnr > 0.163,
            "no English embedder out-refused the TF-IDF baseline (TNR 0.163) at recall≈0.99; best={best_name} {best_tnr:.3}"
        );
    }
}
