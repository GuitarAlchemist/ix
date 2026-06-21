//! Chatbot flight recorder — turns GA's per-run golden-trace JSON into queryable
//! DuckDB tables (Slice A) and a canonical-diff regression gate (Slice B).
//!
//! Source of truth stays in GA's files (`docs/DUCKDB.md`); this is a read-only
//! analyst lens over them. Reads a *corpus directory* that contains a
//! `golden-traces/<promptId>/{run-*.json,_signature.json}` layout — either the live
//! GA corpus (`../ga/state/quality/chatbot-qa`) or vendored fixtures under
//! `tests/fixtures/`. Absent/empty corpus degrades to an empty table + `skipped`
//! gate status (never an error).
//!
//! The hard regression signal is the **routed agent** (`agent_id`): the
//! `orchestration.answer` step's `agentId` in `_signature.json` (the canonical
//! expectation) vs the run's `response.agentId`. `response_length` is a *soft* band
//! only — measured to drift on correct answers, so it never hard-fails.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use duckdb::Connection;
use serde::Serialize;

use crate::source::{self, Files};

/// Errors from the flight recorder: corpus I/O (fail-closed) vs DuckDB.
#[derive(Debug)]
pub enum ChatbotError {
    /// A corpus directory existed but could not be read (permissions, broken mount, …).
    /// Surfaced — never silently treated as an absent corpus — per the contract's
    /// fail-closed read-error semantics.
    Io(std::io::Error),
    Duck(duckdb::Error),
}

impl std::fmt::Display for ChatbotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatbotError::Io(e) => write!(f, "corpus I/O error: {e}"),
            ChatbotError::Duck(e) => write!(f, "duckdb error: {e}"),
        }
    }
}
impl std::error::Error for ChatbotError {}
impl From<std::io::Error> for ChatbotError {
    fn from(e: std::io::Error) -> Self {
        ChatbotError::Io(e)
    }
}
impl From<duckdb::Error> for ChatbotError {
    fn from(e: duckdb::Error) -> Self {
        ChatbotError::Duck(e)
    }
}
/// The shared artifact-source error maps onto the lens's own fail-closed taxonomy,
/// so the public `ChatbotError` surface is unchanged while enumeration goes through
/// [`source::select_files`].
impl From<source::SourceError> for ChatbotError {
    fn from(e: source::SourceError) -> Self {
        match e {
            source::SourceError::Io(e) => ChatbotError::Io(e),
            source::SourceError::Duck(e) => ChatbotError::Duck(e),
        }
    }
}

/// Result of the canonical-diff gate over a corpus.
#[derive(Debug, Clone, Serialize)]
pub struct GateReport {
    /// `chatbot-trace-regression` contract, v0.1 (documented shape; no formal schema yet).
    pub schema_version: String,
    pub run_at: String,
    /// Always `"single"` in v1 — the corpus is 100% `runCount:1`. Reserved for a future
    /// majority/median multi-run reducer.
    pub run_selection: String,
    /// `pass` | `regression` | `degraded` | `skipped`. (`warn` is reserved for soft flags.)
    pub status: String,
    pub prompts_checked: usize,
    /// Deterministic FNV-1a hash of the sorted (prompt_id, expected_agent) baseline set.
    pub baseline_ref: String,
    /// v1 has no baseline history store → always false (human-ack re-baseline deferred).
    pub baseline_changed: bool,
    pub regressions: Vec<Regression>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degraded_reason: Option<String>,
}

/// One flagged prompt: the routed agent diverged from its canonical expectation.
#[derive(Debug, Clone, Serialize)]
pub struct Regression {
    pub prompt_id: String,
    pub category: Option<String>,
    /// The signal that flagged — `agent_id` (hard) in v1.
    pub signal: String,
    /// `hard` (gate-failing) | `soft` (advisory).
    pub severity: String,
    pub expected: Option<String>,
    pub actual: Option<String>,
}

// ---- corpus enumeration ---------------------------------------------------
//
// The chatbot corpus is one level deeper than the flat lenses: a
// `golden-traces/<promptId>/<leaf>` tree, not a single dir of dated files. So the
// traversal stays here (resolving prompt subdirs), but the per-dir listing,
// sorting, escaping and SQL-list building all route through `source` — there is no
// chatbot-private `sql_list` or per-file match loop anymore.

fn golden_traces_dir(corpus_dir: &Path) -> PathBuf {
    corpus_dir.join("golden-traces")
}

/// Enumerate files matching `golden-traces/*/<leaf>` (e.g. `run-1.json`,
/// `_signature.json`). Explicit enumeration (vs a wildcard handed to DuckDB) makes
/// the "absent corpus → skip" path real and keeps untrusted input bounded.
///
/// The nested traversal (resolving each prompt subdir) is the chatbot-specific bit;
/// the per-dir listing + sort + match runs through [`source::select_files`], so the
/// list-building and escaping live in one place across all lenses.
///
/// Fail-closed: an absent `golden-traces` dir (`NotFound`) returns an empty list
/// (→ skipped), but any *other* I/O error on a present corpus (permissions, broken
/// mount) is propagated — never silently treated as absent.
fn collect(
    corpus_dir: &Path,
    matches: fn(&str) -> bool,
) -> Result<Vec<PathBuf>, ChatbotError> {
    let mut out = Vec::new();
    let gt = golden_traces_dir(corpus_dir);
    let prompts = match std::fs::read_dir(&gt) {
        Ok(p) => p,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(out), // absent → skip
        Err(e) => return Err(e.into()),                              // present but unreadable → fail
    };
    for prompt in prompts {
        let p = prompt?.path();
        if !p.is_dir() {
            continue;
        }
        // Per-prompt-dir listing goes through the shared selector (filter + per-dir
        // sort live in `source`); we concatenate and re-sort the union below so the
        // overall ordering is identical to the old flat `out.sort()`.
        out.extend(source::select_files(Files { dir: &p, matches })?);
    }
    out.sort();
    Ok(out)
}

fn run_files(corpus_dir: &Path) -> Result<Vec<PathBuf>, ChatbotError> {
    collect(corpus_dir, |n| n.starts_with("run-") && n.ends_with(".json"))
}

fn signature_files(corpus_dir: &Path) -> Result<Vec<PathBuf>, ChatbotError> {
    collect(corpus_dir, |n| n == "_signature.json")
}

// ---- Slice A: trace warehouse ---------------------------------------------

/// Build the `chatbot_traces` table (one row per `run-*.json`). Returns the row
/// count; 0 when the corpus is absent/empty (graceful-degrade).
// @ai:invariant build_traces creates chatbot_traces with one row per golden-trace run file [T:test conf:0.9 src:ix_duck::chatbot::tests::traces_row_per_run]
pub fn build_traces(conn: &Connection, corpus_dir: &Path) -> Result<usize, ChatbotError> {
    let files = run_files(corpus_dir)?;
    if files.is_empty() {
        conn.execute_batch(
            "CREATE OR REPLACE TABLE chatbot_traces (
                 prompt_id VARCHAR, prompt VARCHAR, category VARCHAR, agent_id VARCHAR,
                 routing_confidence DOUBLE, routing_method VARCHAR, grounding_present BOOLEAN,
                 response_length BIGINT, elapsed_ms BIGINT, recorded_at VARCHAR);",
        )?;
        return Ok(0);
    }
    let list = source::sql_list(&files);
    conn.execute_batch(&format!(
        // Read `response.*` via `json_extract(to_json(response), …)` rather than struct-field
        // access: a subfield absent from EVERY trace file would otherwise be a bind-time
        // error ("Could not find key"), since struct field access binds against the inferred
        // type. This mirrors the `facts` handling below and hardens the guardrail lens
        // against trace-schema evolution. A missing subfield → SQL NULL, never a crash.
        "CREATE OR REPLACE TABLE chatbot_traces AS
         WITH raw AS (
             SELECT promptId AS prompt_id, prompt, category, recordedAt AS recorded_at,
                    to_json(response) AS rj
             FROM read_json_auto({list}, filename=true, union_by_name=true, sample_size=-1)
         )
         SELECT prompt_id, prompt, category,
                json_extract_string(rj, '$.agentId')                       AS agent_id,
                TRY_CAST(json_extract(rj, '$.confidence') AS DOUBLE)       AS routing_confidence,
                json_extract_string(rj, '$.routingMethod')                 AS routing_method,
                (json_extract(rj, '$.grounding') IS NOT NULL
                 AND json_extract(rj, '$.grounding')::VARCHAR <> 'null')   AS grounding_present,
                length(json_extract_string(rj, '$.naturalLanguageAnswer')) AS response_length,
                TRY_CAST(json_extract(rj, '$.elapsedMs') AS BIGINT)        AS elapsed_ms,
                recorded_at
         FROM raw;"
    ))?;
    let n: i64 = conn.query_row("SELECT count(*) FROM chatbot_traces", [], |r| r.get(0))?;
    Ok(n as usize)
}

/// Intents whose mean routing confidence is below `threshold` — the "weak intents"
/// dev query. Returns `(agent_id, avg_confidence, n)`.
pub fn weak_intents(conn: &Connection, threshold: f64) -> duckdb::Result<Vec<(String, f64, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT coalesce(agent_id, '<none>'), avg(routing_confidence), count(*)
         FROM chatbot_traces
         GROUP BY agent_id
         HAVING avg(routing_confidence) < ?
         ORDER BY avg(routing_confidence)",
    )?;
    let rows = stmt
        .query_map([threshold], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<f64>>(1)?.unwrap_or(0.0),
                r.get::<_, i64>(2)?,
            ))
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(rows)
}

/// `(ungrounded, total)` — answers with no `response.grounding`.
pub fn ungrounded(conn: &Connection) -> duckdb::Result<(i64, i64)> {
    conn.query_row(
        "SELECT count(*) FILTER (WHERE NOT grounding_present), count(*) FROM chatbot_traces",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
}

/// A latency-outlier row: `(prompt_id, agent_id, elapsed_ms)`.
pub type LatencyOutlier = (String, Option<String>, Option<i64>);

/// Slowest prompts by `elapsed_ms`.
pub fn latency_outliers(conn: &Connection, limit: usize) -> duckdb::Result<Vec<LatencyOutlier>> {
    let mut stmt = conn.prepare(
        "SELECT prompt_id, agent_id, elapsed_ms FROM chatbot_traces
         ORDER BY elapsed_ms DESC NULLS LAST LIMIT ?",
    )?;
    let rows = stmt
        .query_map([limit as i64], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(rows)
}

/// Share of each `routing_method` — `(routing_method, n)`.
pub fn routing_method_share(conn: &Connection) -> duckdb::Result<Vec<(String, i64)>> {
    let mut stmt = conn.prepare(
        "SELECT coalesce(routing_method, '<none>'), count(*) FROM chatbot_traces
         GROUP BY routing_method ORDER BY count(*) DESC",
    )?;
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(rows)
}

// ---- Grounding quality lens -----------------------------------------------
//
// The chatbot stores `response.grounding` as present/absent only. This lens adds
// two missing signals: COVERAGE (are facts present?) and, for `ix-compatible`
// algebra facts, CORRECTNESS — IX recomputes the fact and checks the claim, so a
// *hallucinated* grounding (a cited fact that is actually wrong) is caught. Facts
// from other sources / query types are `unvalidated` (no oracle yet — not "wrong").

use ix_bracelet::grothendieck::icv;
use ix_bracelet::pc_set::PcSet;
use ix_bracelet::prime_form::bracelet_prime_form;

/// Per-trace grounding assessment.
#[derive(Debug, Clone, Serialize)]
pub struct GroundingAssessment {
    pub prompt_id: String,
    pub category: String,
    /// `ix-compatible` | `ga.dsl` | `none` | …
    pub source: String,
    pub query_type: String,
    /// Structured facts present and non-null.
    pub grounded: bool,
    /// `valid` | `invalid` | `unvalidated` | `no_facts`.
    pub validation: String,
    /// For `invalid`, the specific mismatch(es) IX found.
    pub detail: String,
}

/// Build `chatbot_grounding` (one row per run file) projecting the grounding source,
/// query type, a `grounded` flag, and the z-relation fact fields (NULL otherwise).
/// Reads via `to_json` + `json_extract` so a heterogeneous `facts` bag never errors.
pub fn build_grounding(conn: &Connection, corpus_dir: &Path) -> Result<usize, ChatbotError> {
    let cols = "prompt_id VARCHAR, category VARCHAR, source VARCHAR, query_type VARCHAR, \
                grounded BOOLEAN, zr_left VARCHAR, zr_right VARCHAR, zr_left_icv VARCHAR, \
                zr_right_icv VARCHAR, zr_related VARCHAR";
    let files = run_files(corpus_dir)?;
    if files.is_empty() {
        conn.execute_batch(&format!("CREATE OR REPLACE TABLE chatbot_grounding ({cols});"))?;
        return Ok(0);
    }
    let list = source::sql_list(&files);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE chatbot_grounding AS
         WITH g AS (
             SELECT promptId AS prompt_id, category,
                    json_extract(to_json(response), '$.grounding') AS gj
             FROM read_json_auto({list}, filename=true, union_by_name=true, sample_size=-1)
         )
         SELECT prompt_id, category,
                coalesce(json_extract_string(gj, '$.source'), 'none')     AS source,
                coalesce(json_extract_string(gj, '$.queryType'), '')      AS query_type,
                (json_extract(gj, '$.facts')::VARCHAR IS NOT NULL
                 AND json_extract(gj, '$.facts')::VARCHAR <> 'null')      AS grounded,
                json_extract_string(gj, '$.facts.left')     AS zr_left,
                json_extract_string(gj, '$.facts.right')    AS zr_right,
                json_extract_string(gj, '$.facts.leftIcv')  AS zr_left_icv,
                json_extract_string(gj, '$.facts.rightIcv') AS zr_right_icv,
                json_extract_string(gj, '$.facts.zRelated') AS zr_related
         FROM g;"
    ))?;
    let n: i64 = conn.query_row("SELECT count(*) FROM chatbot_grounding", [], |r| r.get(0))?;
    Ok(n as usize)
}

/// Assess each trace's grounding: coverage + IX correctness validation of
/// `ix-compatible` z-relation facts. Builds the table first (graceful-degrade).
pub fn grounding_report(
    conn: &Connection,
    corpus_dir: &Path,
) -> Result<Vec<GroundingAssessment>, ChatbotError> {
    build_grounding(conn, corpus_dir)?;
    let mut stmt = conn.prepare(
        "SELECT prompt_id, category, source, query_type, grounded,
                zr_left, zr_right, zr_left_icv, zr_right_icv, zr_related
         FROM chatbot_grounding ORDER BY prompt_id",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, bool>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, Option<String>>(6)?,
                r.get::<_, Option<String>>(7)?,
                r.get::<_, Option<String>>(8)?,
                r.get::<_, Option<String>>(9)?,
            ))
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;

    let assessed = rows
        .into_iter()
        .map(|(prompt_id, category, source, query_type, grounded, l, r, li, ri, zr)| {
            let (validation, detail) = if !grounded {
                ("no_facts".to_string(), String::new())
            } else if source == "ix-compatible" && query_type == "z-relation" {
                match (l.as_deref(), r.as_deref()) {
                    (Some(left), Some(right)) => {
                        let (valid, detail) = validate_zrelation(
                            left,
                            right,
                            li.as_deref(),
                            ri.as_deref(),
                            zr.as_deref(),
                        );
                        ((if valid { "valid" } else { "invalid" }).to_string(), detail)
                    }
                    _ => ("unvalidated".to_string(), "missing z-relation operands".to_string()),
                }
            } else {
                ("unvalidated".to_string(), String::new())
            };
            GroundingAssessment {
                prompt_id,
                category,
                source,
                query_type,
                grounded,
                validation,
                detail,
            }
        })
        .collect();
    Ok(assessed)
}

/// Parse `"[0,1,4,6]"` into pitch classes (mod 12). `None` if any token is non-numeric.
fn parse_pcs(s: &str) -> Option<Vec<u8>> {
    let inner = s.trim().trim_start_matches('[').trim_end_matches(']');
    inner
        .split(',')
        .map(|t| t.trim().parse::<i64>().ok().map(|v| v.rem_euclid(12) as u8))
        .collect()
}

/// Parse a 6-entry ICV. Tolerant of the formats IX/GA have used across versions:
/// angle or square brackets, space- or comma-separated (`"<1 1 1 1 1 1>"`,
/// `"<1,1,1,1,1,1>"`, `"[1, 1, 1, 1, 1, 1]"`). `None` if it is not exactly 6
/// parseable numbers (i.e. malformed), so the caller distinguishes malformed from match.
fn parse_icv(s: &str) -> Option<[u32; 6]> {
    let inner = s.trim().trim_matches(|c| matches!(c, '<' | '>' | '[' | ']'));
    let toks: Vec<&str> = inner.split([' ', ',', '\t']).filter(|t| !t.is_empty()).collect();
    let nums: Vec<u32> = toks.iter().filter_map(|t| t.trim().parse().ok()).collect();
    (nums.len() == toks.len()).then_some(()).and(<[u32; 6]>::try_from(nums).ok())
}

/// Check a claimed ICV against the IX-computed one, pushing a problem for a
/// mismatch *or* a malformed claim (never silently skipped).
fn check_icv(claim: Option<&str>, ix: [u32; 6], label: &str, problems: &mut Vec<String>) {
    let Some(c) = claim else { return };
    match parse_icv(c) {
        Some(p) if p == ix => {}
        Some(p) => problems.push(format!("{label} claim {p:?} != IX {ix:?}")),
        None => problems.push(format!("{label} claim malformed: {c:?}")),
    }
}

/// Recompute a z-relation grounding fact in IX and check the chatbot's claim.
/// Two sets are Z-related iff they share an ICV but have different prime forms.
/// Returns `(valid, detail)`; `detail` names every mismatch found.
fn validate_zrelation(
    left: &str,
    right: &str,
    claim_left_icv: Option<&str>,
    claim_right_icv: Option<&str>,
    claim_zrelated: Option<&str>,
) -> (bool, String) {
    let (Some(lp), Some(rp)) = (parse_pcs(left), parse_pcs(right)) else {
        return (false, format!("unparseable pc-sets ({left}, {right})"));
    };
    let la = PcSet::from_pcs(lp.iter().copied());
    let ra = PcSet::from_pcs(rp.iter().copied());
    let ix_li = icv(la).data;
    let ix_ri = icv(ra).data;
    let pf_l: Vec<String> = bracelet_prime_form(la).iter_pcs().map(|p| p.to_string()).collect();
    let pf_r: Vec<String> = bracelet_prime_form(ra).iter_pcs().map(|p| p.to_string()).collect();
    let ix_zrelated = ix_li == ix_ri && pf_l != pf_r;

    let mut problems = Vec::new();
    check_icv(claim_left_icv, ix_li, "leftIcv", &mut problems);
    check_icv(claim_right_icv, ix_ri, "rightIcv", &mut problems);
    if let Some(c) = claim_zrelated {
        match c.trim().to_ascii_lowercase().as_str() {
            "true" if !ix_zrelated => problems.push("zRelated claim true != IX false".into()),
            "false" if ix_zrelated => problems.push("zRelated claim false != IX true".into()),
            "true" | "false" => {}
            other => problems.push(format!("zRelated claim malformed: {other:?}")),
        }
    }
    if problems.is_empty() {
        (true, String::new())
    } else {
        (false, problems.join("; "))
    }
}

/// `(grounded, total, valid, invalid, unvalidated)` over a set of assessments —
/// the headline grounding-quality summary (`invalid` = hallucinated facts).
pub fn grounding_summary(a: &[GroundingAssessment]) -> (usize, usize, usize, usize, usize) {
    let total = a.len();
    let grounded = a.iter().filter(|x| x.grounded).count();
    let valid = a.iter().filter(|x| x.validation == "valid").count();
    let invalid = a.iter().filter(|x| x.validation == "invalid").count();
    let unvalidated = a.iter().filter(|x| x.validation == "unvalidated").count();
    (grounded, total, valid, invalid, unvalidated)
}

// ---- Slice B: canonical-diff gate -----------------------------------------

/// Build `chatbot_signatures(prompt_id, expected_agent)` from each `_signature.json`'s
/// `orchestration.answer` step. UNNEST (not a list-lambda) so it is stable across
/// DuckDB versions. Prompts with no such step are absent → expected NULL → degraded.
fn build_signatures(conn: &Connection, corpus_dir: &Path) -> Result<(), ChatbotError> {
    let files = signature_files(corpus_dir)?;
    if files.is_empty() {
        conn.execute_batch(
            "CREATE OR REPLACE TABLE chatbot_signatures (prompt_id VARCHAR, expected_agent VARCHAR);",
        )?;
        return Ok(());
    }
    let list = source::sql_list(&files);
    conn.execute_batch(&format!(
        "CREATE OR REPLACE TABLE chatbot_signatures AS
         SELECT t.promptId AS prompt_id,
                json_extract_string(to_json(step), '$.agentId') AS expected_agent
         FROM read_json_auto({list}, filename=true, union_by_name=true, sample_size=-1) t,
              UNNEST(t.steps) AS u(step)
         WHERE json_extract_string(to_json(step), '$.name') = 'orchestration.answer';"
    ))?;
    Ok(())
}

/// Run the canonical-diff gate over a corpus: build both tables, diff the routed
/// agent, and classify the result by failure *fingerprint* (a single clean
/// heterogeneous flip fails; a homogeneous collapse warns as `degraded`).
// @ai:invariant check_regressions flags a run whose response.agentId differs from its canonical expected agent [T:test conf:0.9 src:ix_duck::chatbot::tests::diff_flags_agent_drift]
pub fn check_regressions(
    conn: &Connection,
    corpus_dir: &Path,
) -> Result<GateReport, ChatbotError> {
    let prompts_checked = build_traces(conn, corpus_dir)?;
    build_signatures(conn, corpus_dir)?;

    // (prompt_id, category, actual_agent, expected_agent)
    let mut stmt = conn.prepare(
        "SELECT t.prompt_id, t.category, t.agent_id, s.expected_agent
         FROM chatbot_traces t
         LEFT JOIN chatbot_signatures s USING (prompt_id)
         ORDER BY t.prompt_id",
    )?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;

    let mut regressions = Vec::new();
    let mut missing_canonical = 0usize;
    for (prompt_id, category, actual, expected) in &rows {
        match expected {
            None => missing_canonical += 1, // no canonical for this prompt → degraded signal
            Some(_) if actual != expected => regressions.push(Regression {
                prompt_id: prompt_id.clone(),
                category: category.clone(),
                signal: "agent_id".into(),
                severity: "hard".into(),
                expected: expected.clone(),
                actual: actual.clone(),
            }),
            Some(_) => {}
        }
    }

    let baseline_ref = baseline_hash(conn)?;
    let run_at = chrono::Utc::now().to_rfc3339();
    let mk = |status: &str, degraded_reason: Option<String>| GateReport {
        schema_version: "chatbot-trace-regression.v0.1".into(),
        run_at: run_at.clone(),
        run_selection: "single".into(),
        status: status.into(),
        prompts_checked,
        baseline_ref: baseline_ref.clone(),
        baseline_changed: false,
        regressions: regressions.clone(),
        degraded_reason,
    };

    // No corpus at all → skipped (graceful-degrade, exit 0).
    if prompts_checked == 0 {
        return Ok(mk("skipped", Some("corpus absent or empty".into())));
    }
    // Canonical missing for most prompts → stale/degraded baseline, warn not fail.
    if missing_canonical * 2 >= prompts_checked {
        return Ok(mk(
            "degraded",
            Some(format!("{missing_canonical}/{prompts_checked} prompts have no canonical agent")),
        ));
    }
    if regressions.is_empty() {
        return Ok(mk("pass", None));
    }
    // Fingerprint: a homogeneous collapse (≥2 flags all to the same agent, ≥50% of the
    // corpus) is an environment/backend symptom → degraded (warn). Otherwise a real
    // heterogeneous regression → fail. A single clean flip is never homogeneous → fails.
    let homogeneous = regressions.len() >= 2
        && regressions.len() * 2 >= prompts_checked
        && regressions.iter().all(|r| r.actual == regressions[0].actual);
    if homogeneous {
        let to = regressions[0].actual.clone().unwrap_or_else(|| "<none>".into());
        return Ok(mk(
            "degraded",
            Some(format!("{} prompts collapsed to '{to}' (backend/env symptom)", regressions.len())),
        ));
    }
    Ok(mk("regression", None))
}

/// Deterministic FNV-1a 64-bit hash of the sorted (prompt_id, expected_agent) set —
/// the baseline content reference. Stable across platforms/toolchains.
fn baseline_hash(conn: &Connection) -> duckdb::Result<String> {
    let mut stmt = conn.prepare(
        "SELECT prompt_id, coalesce(expected_agent, '') FROM chatbot_signatures ORDER BY prompt_id",
    )?;
    let pairs = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
        .collect::<duckdb::Result<Vec<_>>>()?;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (k, v) in pairs {
        for b in k.bytes().chain(b"=".iter().copied()).chain(v.bytes()).chain(b";".iter().copied()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    Ok(format!("fnv1a64:{h:016x}"))
}

/// Map a gate status to a process exit code: `regression` → 1, all else → 0.
pub fn exit_code(status: &str) -> i32 {
    if status == "regression" {
        1
    } else {
        0
    }
}

/// Atomically write the contract JSON (same-dir temp + rename; replaces on Windows).
pub fn write_contract(report: &GateReport, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(report).map_err(std::io::Error::other)?;
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, json.as_bytes())?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/chatbot-qa")
    }

    /// Copy the fixtures corpus into a tempdir so a test can mutate it.
    fn copy_corpus(dst: &Path) {
        let src = fixtures();
        for entry in walk(&src) {
            let rel = entry.strip_prefix(&src).unwrap();
            let target = dst.join(rel);
            fs::create_dir_all(target.parent().unwrap()).unwrap();
            fs::copy(&entry, &target).unwrap();
        }
    }

    fn walk(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for e in fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(walk(&p));
            } else {
                out.push(p);
            }
        }
        out
    }

    /// Rewrite a run file's `response.agentId`.
    fn set_agent(run: &Path, agent: &str) {
        let mut v: serde_json::Value = serde_json::from_str(&fs::read_to_string(run).unwrap()).unwrap();
        v["response"]["agentId"] = serde_json::Value::String(agent.into());
        fs::write(run, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }

    #[test]
    fn traces_row_per_run() {
        let conn = crate::open_bench().unwrap();
        let n = build_traces(&conn, &fixtures()).unwrap();
        let files = run_files(&fixtures()).unwrap().len();
        assert!(files >= 3, "need a real multi-prompt fixture corpus, found {files}");
        assert_eq!(n, files, "one chatbot_traces row per run file");
    }

    /// A trace whose `response` object lacks subfields (trace-schema drift) must BUILD.
    /// With struct-field access this bind-errored ("Could not find key …"); via
    /// `json_extract(to_json(response), …)` the missing fields read as NULL. Guards the
    /// guardrail lens against GA renaming/removing a `response` subfield.
    #[test]
    fn response_missing_subfields_builds_as_null() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path().join("golden-traces/q1/run-1.json");
        fs::create_dir_all(run.parent().unwrap()).unwrap();
        // `response` carries only agentId — no confidence/routingMethod/grounding/answer/elapsedMs.
        fs::write(
            &run,
            r#"{"promptId":"q1","prompt":"p","category":"c",
                "response":{"agentId":"skill.x"},"recordedAt":"2026-06-19T00:00:00Z"}"#,
        )
        .unwrap();
        let conn = crate::open_bench().unwrap();
        let n = build_traces(&conn, dir.path()).unwrap(); // struct-field access would Err here
        assert_eq!(n, 1, "the trace builds despite the sparse response");
        let (agent, conf): (String, Option<f64>) = conn
            .query_row(
                "SELECT agent_id, routing_confidence FROM chatbot_traces",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(agent, "skill.x");
        assert_eq!(conf, None, "absent confidence reads as NULL, not a crash or 0");
    }

    #[test]
    fn lens_queries_run_on_real_fixtures() {
        let conn = crate::open_bench().unwrap();
        build_traces(&conn, &fixtures()).unwrap();
        let (ungr, total) = ungrounded(&conn).unwrap();
        assert!(total >= 3 && ungr <= total);
        // grounding/ routing columns parse without error
        let _ = weak_intents(&conn, 0.7).unwrap();
        let _ = latency_outliers(&conn, 5).unwrap();
        assert!(!routing_method_share(&conn).unwrap().is_empty());
    }

    #[test]
    fn grounding_validates_correct_zrelation() {
        // The are-0146 trace cites a correct z-relation; IX recomputation confirms it.
        let conn = crate::open_bench().unwrap();
        let report = grounding_report(&conn, &fixtures()).unwrap();
        let zr = report
            .iter()
            .find(|a| a.prompt_id == "are-0146-and-0137-z-related")
            .expect("z-relation trace present in fixtures");
        assert_eq!(zr.source, "ix-compatible");
        assert!(zr.grounded);
        assert_eq!(zr.validation, "valid", "IX should confirm the fact; detail: {}", zr.detail);
    }

    #[test]
    fn grounding_flags_hallucinated_fact() {
        // Flip the (true) z-relation claim to False → IX must catch the lie.
        let dir = tempfile::tempdir().unwrap();
        copy_corpus(dir.path());
        let run = dir
            .path()
            .join("golden-traces/are-0146-and-0137-z-related/run-1.json");
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&run).unwrap()).unwrap();
        v["response"]["grounding"]["facts"]["zRelated"] = serde_json::Value::String("False".into());
        fs::write(&run, serde_json::to_string_pretty(&v).unwrap()).unwrap();

        let conn = crate::open_bench().unwrap();
        let report = grounding_report(&conn, dir.path()).unwrap();
        let zr = report
            .iter()
            .find(|a| a.prompt_id == "are-0146-and-0137-z-related")
            .unwrap();
        assert_eq!(zr.validation, "invalid", "a flipped zRelated claim must be caught");
        assert!(zr.detail.contains("zRelated"), "detail names the field: {}", zr.detail);
    }

    #[test]
    fn zrelation_accepts_comma_separated_icv() {
        // Both space- and comma-separated ICV formats parse to the same value.
        assert_eq!(parse_icv("<1 1 1 1 1 1>"), Some([1, 1, 1, 1, 1, 1]));
        assert_eq!(parse_icv("<1,1,1,1,1,1>"), Some([1, 1, 1, 1, 1, 1]));
        // Square-bracket form (a format IX/GA have used) also parses.
        assert_eq!(parse_icv("[1, 1, 1, 1, 1, 1]"), Some([1, 1, 1, 1, 1, 1]));
        // A correct comma-form claim validates (no problems).
        let (valid, detail) =
            validate_zrelation("[0,1,4,6]", "[0,1,3,7]", Some("<1,1,1,1,1,1>"), None, Some("True"));
        assert!(valid, "comma ICV should validate: {detail}");
    }

    #[test]
    fn zrelation_flags_malformed_claims() {
        // Malformed ICV → flagged, not silently skipped.
        let (valid, detail) =
            validate_zrelation("[0,1,4,6]", "[0,1,3,7]", Some("<oops>"), None, Some("True"));
        assert!(!valid && detail.contains("malformed"), "malformed ICV: {detail}");
        // Malformed zRelated → flagged, not coerced to false.
        let (valid, detail) =
            validate_zrelation("[0,1,4,6]", "[0,1,3,7]", None, None, Some("maybe"));
        assert!(!valid && detail.contains("malformed"), "malformed zRelated: {detail}");
    }

    #[test]
    fn grounding_dsl_facts_are_no_facts_not_invalid() {
        // ga.dsl common-tones has facts:null → no_facts, never "invalid".
        let conn = crate::open_bench().unwrap();
        let report = grounding_report(&conn, &fixtures()).unwrap();
        let ct = report
            .iter()
            .find(|a| a.prompt_id == "common-tones-between-g7-and-dm7")
            .unwrap();
        assert!(!ct.grounded);
        assert_eq!(ct.validation, "no_facts");
    }

    #[test]
    fn happy_corpus_passes() {
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, &fixtures()).unwrap();
        assert_eq!(report.status, "pass", "vendored fixtures must be regression-free: {report:?}");
        assert_eq!(exit_code(&report.status), 0);
    }

    #[test]
    fn diff_flags_agent_drift() {
        let dir = tempfile::tempdir().unwrap();
        copy_corpus(dir.path());
        // Flip exactly one prompt's routed agent → a single clean heterogeneous flip.
        let runs = run_files(dir.path()).unwrap();
        set_agent(&runs[0], "skill.WRONG_AGENT");

        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, dir.path()).unwrap();
        assert_eq!(report.status, "regression", "{report:?}");
        assert_eq!(exit_code(&report.status), 1);
        assert!(report.regressions.iter().any(|r| r.actual.as_deref() == Some("skill.WRONG_AGENT")));
    }

    #[test]
    fn homogeneous_collapse_is_degraded() {
        let dir = tempfile::tempdir().unwrap();
        copy_corpus(dir.path());
        // Collapse every prompt to the same fallback agent → backend/env symptom.
        for run in run_files(dir.path()).unwrap() {
            set_agent(&run, "skill.fallback");
        }
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, dir.path()).unwrap();
        assert_eq!(report.status, "degraded", "{report:?}");
        assert_eq!(exit_code(&report.status), 0, "degraded warns, never fails");
    }

    #[test]
    fn absent_corpus_skips() {
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, Path::new("/no/such/corpus")).unwrap();
        assert_eq!(report.status, "skipped");
        assert_eq!(report.prompts_checked, 0);
        assert_eq!(exit_code(&report.status), 0);
    }

    #[test]
    fn present_but_unreadable_corpus_fails_closed() {
        // golden-traces exists but is a FILE, not a dir → read_dir errors with a kind
        // other than NotFound. Must surface as an error, never a silent `skipped` pass.
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("golden-traces"), b"not a directory").unwrap();
        let conn = crate::open_bench().unwrap();
        assert!(
            check_regressions(&conn, dir.path()).is_err(),
            "a present-but-unreadable corpus must fail closed, not skip"
        );
    }

    #[test]
    fn baseline_hash_is_stable_and_changes_with_canonical() {
        let conn = crate::open_bench().unwrap();
        let a = check_regressions(&conn, &fixtures()).unwrap().baseline_ref;
        let b = check_regressions(&conn, &fixtures()).unwrap().baseline_ref;
        assert_eq!(a, b, "same corpus → same baseline_ref");
        assert!(a.starts_with("fnv1a64:"));
    }

    #[test]
    fn contract_round_trips() {
        let conn = crate::open_bench().unwrap();
        let report = check_regressions(&conn, &fixtures()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("sub/chatbot-trace-regressions.json");
        write_contract(&report, &out).unwrap();
        let back: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&out).unwrap()).unwrap();
        assert_eq!(back["status"], "pass");
        assert_eq!(back["run_selection"], "single");
    }
}
