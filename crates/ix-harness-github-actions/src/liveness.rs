//! Workflow liveness — scheduled-run history → one hexavalent verdict
//! per scheduled workflow.
//!
//! The run-level adapter in the crate root projects ONE run. A loop
//! dies across many runs, so that projection cannot see it: a nightly
//! that fails every night, a schedule GitHub silently stopped firing,
//! or a producer whose runs are cancelled at the timeout wall (reads
//! grey, never red). This module looks at the scheduled-run history of
//! a workflow and emits two independent observations on the same
//! claim, then merges them with `ix_fuzzy::observations::merge`:
//!
//! | source                     | signal                                         |
//! |----------------------------|------------------------------------------------|
//! | `github-actions-outcome`   | trailing non-success streak of scheduled runs  |
//! | `github-actions-schedule`  | scheduled runs owed by the cron since the last one |
//!
//! Missed fires are counted against the workflow file's own `cron`
//! entries (see [`crate::cron`]), so a biweekly or weekday-only schedule
//! is judged by its real calendar. Without the file, the median gap
//! between past scheduled runs stands in for the cadence.
//!
//! Only `event == "schedule"` runs count. Pull-request runs of the same
//! workflow often take a shorter path and pass, which is exactly how a
//! nightly that never succeeds can look green in the checks list.
//!
//! Because both observations share one claim key, the merge synthesizes
//! a `Contradictory` observation when they disagree. The disagreement
//! that escalates is "last run succeeded, but the schedule stopped":
//! green but dead.
//!
//! Not covered: whether a run produced an artifact, or whether anything
//! reads it. The run history carries neither.

use std::collections::BTreeMap;

use ix_agent_core::SessionEvent;
use ix_fuzzy::observations::{merge_all, HexObservation};
use ix_fuzzy::{escalation_triggered, HexavalentDistribution};
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};

use crate::cron::Cron;
use crate::{emit, parse_epoch, sanitize, sha256_hex, AdapterError};

pub const OUTCOME_SOURCE: &str = "github-actions-outcome";
pub const SCHEDULE_SOURCE: &str = "github-actions-schedule";

/// Cadence assumed when fewer than two scheduled runs exist.
const DEFAULT_CADENCE_HOURS: f64 = 24.0;

/// GitHub starts scheduled runs late under load, sometimes by hours.
/// Fires this recent are not yet counted as missed.
const GRACE_SECS: i64 = 3 * 3600;

/// Upper bound on the fires-per-run throttle factor, so a cron change or
/// a long gap in history cannot excuse a long silence.
const MAX_FIRES_PER_RUN: f64 = 12.0;

/// One workflow and its run history, as returned by
/// `gh workflow list --json name,path,state` joined with
/// the Actions API's `workflows/<file>/runs?event=schedule`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRuns {
    /// `owner/name`.
    pub repo: String,
    /// Display name of the workflow.
    pub workflow: String,
    /// `.github/workflows/<file>.yml`.
    #[serde(default)]
    pub path: Option<String>,
    /// `active`, `disabled_inactivity`, `disabled_manually`, ...
    #[serde(default)]
    pub state: Option<String>,
    /// `on.schedule` cron expressions from the workflow file on the default
    /// branch. `None` when the file was not read; `Some(vec![])` when it
    /// has no schedule (event-triggered, or the cron was removed).
    #[serde(default)]
    pub crons: Option<Vec<String>>,
    /// When the workflow file last changed on the default branch
    /// (`YYYY-MM-DDTHH:MM:SSZ`). No run is owed from before it: a new,
    /// renamed or re-scheduled file starts with a clean slate.
    #[serde(default)]
    pub schedule_since: Option<String>,
    /// Run rows, any order, any event (non-schedule events are ignored).
    #[serde(default)]
    pub runs: Vec<RunRecord>,
    /// The fetch hit its run limit, so older history was not read.
    #[serde(default)]
    pub runs_truncated: bool,
    /// Reading this workflow (or the repo's workflow list) failed.
    #[serde(default)]
    pub fetch_error: Option<String>,
}

/// One run: `gh run list --json conclusion,status,createdAt,event` shape.
/// Extra fields are ignored.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunRecord {
    #[serde(default)]
    pub conclusion: Option<String>,
    #[serde(default)]
    pub status: String,
    pub created_at: String,
    #[serde(default)]
    pub event: String,
}

/// Liveness verdict for one scheduled workflow.
#[derive(Debug, Clone, Serialize)]
pub struct LivenessReport {
    pub repo: String,
    pub workflow: String,
    pub path: Option<String>,
    pub claim_key: String,
    pub verdict: Hexavalent,
    /// Scheduled runs seen in the window.
    pub scheduled_runs: usize,
    /// Consecutive most-recent completed scheduled runs that did not succeed.
    pub failure_streak: usize,
    /// `true` when no scheduled run in the window succeeded.
    pub no_success_in_window: bool,
    pub last_conclusion: Option<String>,
    pub last_success_age_hours: Option<f64>,
    pub last_run_age_hours: Option<f64>,
    /// Median gap between scheduled runs.
    pub cadence_hours: f64,
    /// Scheduled runs the cron owes since the latest one (fires older than
    /// the grace period, per historically delivered fires-per-run). `None`
    /// when the workflow file was not read.
    pub missed_runs: Option<usize>,
    /// Older runs were not read, so the streak is a lower bound when no
    /// run in the window succeeded.
    pub window_truncated: bool,
    pub fetch_error: Option<String>,
    /// Merged distribution over `T P U D F C`.
    pub distribution: BTreeMap<String, f64>,
    #[serde(skip)]
    pub observations: Vec<SessionEvent>,
}

impl LivenessReport {
    /// `Doubtful`, `False` or `Contradictory`.
    pub fn is_unhealthy(&self) -> bool {
        matches!(
            self.verdict,
            Hexavalent::Doubtful | Hexavalent::False | Hexavalent::Contradictory
        )
    }
}

/// Assess one workflow. Returns `None` when there is no schedule to
/// judge: the workflow file has no `cron` (event-triggered, or the cron
/// was removed on purpose), or neither crons nor scheduled runs are known.
/// A workflow GitHub disabled for inactivity is always assessed, and a
/// workflow that could not be read is reported as `Unknown`.
///
/// `now` is `YYYY-MM-DDTHH:MM:SSZ`, injected so runs are reproducible.
pub fn assess(
    wf: &WorkflowRuns,
    now: &str,
    round: u32,
) -> Result<Option<LivenessReport>, AdapterError> {
    let now_s =
        parse_epoch(now).ok_or_else(|| AdapterError::Parse(format!("invalid --now: {now}")))?;
    if let Some(error) = &wf.fetch_error {
        return Ok(Some(unreadable(wf, error)));
    }

    // @ai:invariant only schedule-event runs count toward liveness, so passing PR runs never mask a failing nightly [T:test conf:0.9 src:test_pr_successes_do_not_mask_scheduled_failures]
    let mut scheduled: Vec<(f64, &RunRecord)> = wf
        .runs
        .iter()
        .filter(|r| r.event == "schedule")
        .filter_map(|r| parse_epoch(&r.created_at).map(|t| (t, r)))
        .collect();
    scheduled.sort_by(|a, b| b.0.total_cmp(&a.0));

    let disabled_inactivity = wf.state.as_deref() == Some("disabled_inactivity");
    // Any unparseable entry drops back to the median-gap heuristic.
    let crons: Option<Vec<Cron>> = wf
        .crons
        .as_ref()
        .and_then(|cs| cs.iter().map(|c| Cron::parse(c)).collect());
    let fires = |after: i64, until: i64, cs: &[Cron]| -> usize {
        cs.iter()
            .map(|c| c.fires_between(after, until, usize::MAX))
            .sum()
    };
    // Runs are owed from the later of the last scheduled run and the last
    // change to the workflow file. With neither, nothing can be owed.
    let since = wf.schedule_since.as_deref().and_then(parse_epoch);
    let owed_from = match (scheduled.first().map(|(t, _)| *t), since) {
        (Some(last), Some(since)) => Some(last.max(since)),
        (last, since) => last.or(since),
    };
    // Scheduled runs owed since then. GitHub throttles frequent crons (a
    // `*/30` cron may run every few hours), so raw missed fires are divided
    // by the MEDIAN fires per gap between past runs, capped: a mean would let
    // one past outage excuse the next death.
    let missed_runs = crons
        .as_ref()
        .filter(|cs| !cs.is_empty())
        .zip(owed_from)
        .map(|(cs, from)| {
            let missed = fires(from as i64, now_s as i64 - GRACE_SECS, cs);
            let mut per_gap: Vec<usize> = scheduled
                .windows(2)
                .map(|w| fires(w[1].0 as i64, w[0].0 as i64, cs))
                .collect();
            per_gap.sort_unstable();
            let fires_per_run = per_gap
                .get(per_gap.len() / 2)
                .map_or(1.0, |m| (*m as f64).clamp(1.0, MAX_FIRES_PER_RUN));
            (missed as f64 / fires_per_run).floor() as usize
        });
    if !disabled_inactivity {
        match &wf.crons {
            Some(cs) if cs.is_empty() => return Ok(None),
            None if scheduled.is_empty() => return Ok(None),
            _ => {}
        }
    }

    let completed: Vec<&(f64, &RunRecord)> = scheduled
        .iter()
        .filter(|(_, r)| r.status == "completed")
        .filter(|(_, r)| !matches!(r.conclusion.as_deref(), Some("skipped" | "neutral")))
        .collect();
    let failure_streak = completed
        .iter()
        .take_while(|(_, r)| r.conclusion.as_deref() != Some("success"))
        .count();
    let last_success = completed
        .iter()
        .find(|(_, r)| r.conclusion.as_deref() == Some("success"));
    let last_success_age_hours = last_success.map(|(t, _)| (now_s - t) / 3600.0);
    let last_run_age_hours = scheduled.first().map(|(t, _)| (now_s - t) / 3600.0);
    let cadence_hours = median_gap_hours(&scheduled).unwrap_or(DEFAULT_CADENCE_HOURS);

    let claim_key = claim_key(wf);

    let diagnosis_id = sha256_hex(
        serde_json::to_string(wf)
            .map_err(|e| AdapterError::Parse(e.to_string()))?
            .as_bytes(),
    );
    let mut ordinal = 0u64;
    let mut observations = Vec::new();

    let outcome_signal = if !completed.is_empty() {
        let (variant, weight) = match failure_streak {
            0 => (Hexavalent::True, 0.8),
            1 => (Hexavalent::Doubtful, 0.5),
            2 => (Hexavalent::Doubtful, 0.8),
            _ => (Hexavalent::False, 1.0),
        };
        Some((
            variant,
            weight,
            format!(
                "{failure_streak} consecutive non-success scheduled runs of {}",
                completed.len()
            ),
        ))
    } else {
        // No run yet, or every run skipped or in progress: the outcome is
        // unknown, which also keeps a lone on-time T@0.3 from standing.
        Some((
            Hexavalent::Unknown,
            0.5,
            format!("no conclusive scheduled run of {}", scheduled.len()),
        ))
    };
    if let Some((variant, weight, evidence)) = outcome_signal {
        observations.push(with_source(
            emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                &claim_key,
                variant,
                weight,
                evidence,
            ),
            OUTCOME_SOURCE,
        ));
    }

    // Being on time is weak evidence of life; stopping is strong evidence
    // of death. Hence T@0.3 against D/F@0.6-1.0.
    let schedule_signal = if disabled_inactivity {
        Some((
            Hexavalent::False,
            1.0,
            "workflow disabled_inactivity: GitHub stopped the schedule".to_string(),
        ))
    } else if let Some(missed) = missed_runs {
        // One owed run is tolerated: GitHub drops scheduled runs under load.
        let evidence = format!(
            "{missed} scheduled runs owed by the cron since the last one ({}) or the last workflow file change",
            last_run_age_hours
                .map(|h| format!("{h:.1}h ago"))
                .unwrap_or_else(|| "none in window".to_string())
        );
        Some(match missed {
            0 | 1 => (Hexavalent::True, 0.3, evidence),
            2 => (Hexavalent::Doubtful, 0.6, evidence),
            _ => (Hexavalent::False, 0.9, evidence),
        })
    } else {
        last_run_age_hours.map(|age| {
            let ratio = age / cadence_hours;
            let evidence = format!(
                "last scheduled run {age:.1}h ago, cadence {cadence_hours:.1}h (x{ratio:.1})"
            );
            if ratio <= 2.0 {
                (Hexavalent::True, 0.3, evidence)
            } else if ratio <= 4.0 {
                (Hexavalent::Doubtful, 0.6, evidence)
            } else {
                (Hexavalent::False, 0.9, evidence)
            }
        })
    };
    if let Some((variant, weight, evidence)) = schedule_signal {
        observations.push(with_source(
            emit(
                &mut ordinal,
                &diagnosis_id,
                round,
                &claim_key,
                variant,
                weight,
                evidence,
            ),
            SCHEDULE_SOURCE,
        ));
    }

    let merged = merge_all(
        &observations
            .iter()
            .filter_map(to_hex_observation)
            .collect::<Vec<_>>(),
    )
    .map_err(|e| AdapterError::Parse(e.to_string()))?;

    Ok(Some(LivenessReport {
        repo: wf.repo.clone(),
        workflow: wf.workflow.clone(),
        path: wf.path.clone(),
        claim_key,
        verdict: verdict(&merged.distribution),
        scheduled_runs: scheduled.len(),
        failure_streak,
        no_success_in_window: last_success.is_none(),
        last_conclusion: completed.first().and_then(|(_, r)| r.conclusion.clone()),
        last_success_age_hours,
        last_run_age_hours,
        cadence_hours,
        missed_runs,
        window_truncated: wf.runs_truncated,
        fetch_error: None,
        distribution: HEX_ORDER
            .iter()
            .map(|v| (letter(*v).to_string(), merged.distribution.get(v)))
            .collect(),
        observations,
    }))
}

/// A workflow the sweep could not read: `Unknown`, with no observations,
/// so one API error neither aborts the sweep nor reads as a dead loop.
fn unreadable(wf: &WorkflowRuns, error: &str) -> LivenessReport {
    LivenessReport {
        repo: wf.repo.clone(),
        workflow: wf.workflow.clone(),
        path: wf.path.clone(),
        claim_key: claim_key(wf),
        verdict: Hexavalent::Unknown,
        scheduled_runs: 0,
        failure_streak: 0,
        no_success_in_window: false,
        last_conclusion: None,
        last_success_age_hours: None,
        last_run_age_hours: None,
        cadence_hours: DEFAULT_CADENCE_HOURS,
        missed_runs: None,
        window_truncated: false,
        fetch_error: Some(error.to_string()),
        distribution: HEX_ORDER
            .iter()
            .map(|v| (letter(*v).to_string(), f64::from(*v == Hexavalent::Unknown)))
            .collect(),
        observations: Vec::new(),
    }
}

fn claim_key(wf: &WorkflowRuns) -> String {
    let workflow_key = sanitize(
        wf.path
            .as_deref()
            .and_then(|p| p.rsplit('/').next())
            .map(|f| f.trim_end_matches(".yml").trim_end_matches(".yaml"))
            .unwrap_or(&wf.workflow),
    );
    let repo_key = sanitize(wf.repo.rsplit('/').next().unwrap_or(&wf.repo));
    format!("gha_loop:{repo_key}:{workflow_key}::reliable")
}

const HEX_ORDER: [Hexavalent; 6] = [
    Hexavalent::True,
    Hexavalent::Probable,
    Hexavalent::Unknown,
    Hexavalent::Doubtful,
    Hexavalent::False,
    Hexavalent::Contradictory,
];

/// Escalation (C share of informative mass > 0.3) wins; otherwise the
/// heaviest variant, ties broken pessimistically (F before D before U
/// before P before T) because this is a health monitor.
fn verdict(dist: &HexavalentDistribution) -> Hexavalent {
    if escalation_triggered(dist) {
        return Hexavalent::Contradictory;
    }
    let mut best = Hexavalent::Unknown;
    let mut best_mass = f64::NEG_INFINITY;
    for v in [
        Hexavalent::False,
        Hexavalent::Doubtful,
        Hexavalent::Unknown,
        Hexavalent::Probable,
        Hexavalent::True,
    ] {
        let mass = dist.get(&v);
        if mass > best_mass + 1e-12 {
            best = v;
            best_mass = mass;
        }
    }
    best
}

fn letter(v: Hexavalent) -> &'static str {
    match v {
        Hexavalent::True => "T",
        Hexavalent::Probable => "P",
        Hexavalent::Unknown => "U",
        Hexavalent::Doubtful => "D",
        Hexavalent::False => "F",
        Hexavalent::Contradictory => "C",
    }
}

fn median_gap_hours(sorted_desc: &[(f64, &RunRecord)]) -> Option<f64> {
    let mut gaps: Vec<f64> = sorted_desc
        .windows(2)
        .map(|w| (w[0].0 - w[1].0) / 3600.0)
        .filter(|g| *g > 0.0)
        .collect();
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_by(f64::total_cmp);
    Some(gaps[gaps.len() / 2])
}

fn with_source(mut event: SessionEvent, new_source: &str) -> SessionEvent {
    if let SessionEvent::ObservationAdded { source, .. } = &mut event {
        *source = new_source.to_string();
    }
    event
}

fn to_hex_observation(event: &SessionEvent) -> Option<HexObservation> {
    match event {
        SessionEvent::ObservationAdded {
            ordinal,
            source,
            diagnosis_id,
            round,
            claim_key,
            variant,
            weight,
            evidence,
        } => Some(HexObservation {
            source: source.clone(),
            diagnosis_id: diagnosis_id.clone(),
            round: *round,
            ordinal: *ordinal as u32,
            claim_key: claim_key.clone(),
            variant: *variant,
            weight: *weight,
            evidence: evidence.clone(),
        }),
        _ => None,
    }
}

/// Hari Phase-6 session JSONL (`hari-core replay --session <file>`):
/// an `open` header, one `belief_update` per report, a `close` trailer.
pub fn to_hari_session(reports: &[LivenessReport], round: u32) -> Vec<serde_json::Value> {
    let mut lines = vec![serde_json::json!({"op": "open", "config": {}})];
    // An unreadable workflow is a sweep failure, not evidence about the loop.
    for r in reports.iter().filter(|r| r.fetch_error.is_none()) {
        lines.push(serde_json::json!({
            "op": "event",
            "event": {
                "cycle": round,
                "source": crate::SOURCE,
                "payload": {
                    "type": "belief_update",
                    "proposition": r.claim_key,
                    "value": hari_value(r.verdict),
                    "evidence": {
                        "repo": r.repo,
                        "workflow": r.workflow,
                        "failure_streak": r.failure_streak,
                        "no_success_in_window": r.no_success_in_window,
                        "last_success_age_hours": r.last_success_age_hours,
                        "last_run_age_hours": r.last_run_age_hours,
                        "cadence_hours": r.cadence_hours,
                        "missed_runs": r.missed_runs,
                        "distribution": r.distribution,
                    }
                }
            }
        }));
    }
    lines.push(serde_json::json!({"op": "close"}));
    lines
}

/// Hari's `HexValue` serializes long-form.
fn hari_value(v: Hexavalent) -> &'static str {
    match v {
        Hexavalent::True => "True",
        Hexavalent::Probable => "Probable",
        Hexavalent::Unknown => "Unknown",
        Hexavalent::Doubtful => "Doubtful",
        Hexavalent::False => "False",
        Hexavalent::Contradictory => "Contradictory",
    }
}

/// Markdown table of the unhealthy reports, worst first.
pub fn to_markdown(reports: &[LivenessReport]) -> String {
    let mut unhealthy: Vec<&LivenessReport> = reports.iter().filter(|r| r.is_unhealthy()).collect();
    unhealthy.sort_by(|a, b| {
        severity(a.verdict)
            .cmp(&severity(b.verdict))
            .then(b.failure_streak.cmp(&a.failure_streak))
            .then(a.repo.cmp(&b.repo))
            .then(a.workflow.cmp(&b.workflow))
    });
    let mut out = format!(
        "{} of {} scheduled workflows unhealthy (verdict D, F or C).\n\n\
         | Repo | Workflow | Verdict | Failing runs in a row | Last scheduled success | Last scheduled run | Scheduled runs missed |\n\
         |---|---|---|---|---|---|---|\n",
        unhealthy.len(),
        reports.iter().filter(|r| r.fetch_error.is_none()).count()
    );
    for r in unhealthy {
        // Older runs were not read: the streak may be longer.
        let streak_bound = if r.window_truncated && r.no_success_in_window {
            "≥"
        } else {
            ""
        };
        let success = match r.last_success_age_hours {
            Some(h) => days(h),
            None if r.window_truncated => format!("none in last {}", r.scheduled_runs),
            None if r.scheduled_runs == 0 => "no scheduled run yet".to_string(),
            None => format!("never ({} scheduled runs)", r.scheduled_runs),
        };
        let last_run = r.last_run_age_hours.map(days).unwrap_or_else(|| "-".into());
        let missed = r
            .missed_runs
            .map(|n| n.to_string())
            .unwrap_or_else(|| format!("? (~{:.0}h cadence)", r.cadence_hours));
        out.push_str(&format!(
            "| {} | {} | {} | {streak_bound}{} | {} | {} | {} |\n",
            md_cell(r.repo.rsplit('/').next().unwrap_or(&r.repo)),
            md_cell(&r.workflow),
            letter(r.verdict),
            r.failure_streak,
            success,
            last_run,
            missed
        ));
    }
    let errors: Vec<&LivenessReport> = reports.iter().filter(|r| r.fetch_error.is_some()).collect();
    if !errors.is_empty() {
        out.push_str(&format!(
            "\nNot assessed, read failed ({}):\n\n",
            errors.len()
        ));
        for r in errors {
            out.push_str(&format!(
                "- {} / {}: {}\n",
                md_cell(&r.repo),
                md_cell(&r.workflow),
                md_cell(r.fetch_error.as_deref().unwrap_or_default())
            ));
        }
    }
    out
}

/// Names and messages come from other repos: keep them from breaking the
/// table (`|`, newlines) or pinging people (`@`).
fn md_cell(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('@', "&#64;")
        .replace(['\r', '\n'], " ")
}

fn severity(v: Hexavalent) -> u8 {
    match v {
        Hexavalent::Contradictory => 0,
        Hexavalent::False => 1,
        Hexavalent::Doubtful => 2,
        _ => 3,
    }
}

fn days(hours: f64) -> String {
    format!("{:.1}d ago", hours / 24.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: &str = "2026-09-14T12:00:00Z";

    fn run(created_at: &str, event: &str, conclusion: &str) -> RunRecord {
        RunRecord {
            conclusion: Some(conclusion.to_string()),
            status: "completed".to_string(),
            created_at: created_at.to_string(),
            event: event.to_string(),
        }
    }

    fn nightly(days_and_conclusions: &[(u32, &str)]) -> WorkflowRuns {
        WorkflowRuns {
            repo: "GuitarAlchemist/ix".to_string(),
            workflow: "Nightly".to_string(),
            path: Some(".github/workflows/nightly-thing.yml".to_string()),
            state: Some("active".to_string()),
            crons: None,
            schedule_since: None,
            runs: days_and_conclusions
                .iter()
                .map(|(d, c)| run(&format!("2026-09-{d:02}T09:00:00Z"), "schedule", c))
                .collect(),
            runs_truncated: false,
            fetch_error: None,
        }
    }

    fn scheduled(cron: &str, since: &str) -> WorkflowRuns {
        let mut wf = nightly(&[]);
        wf.crons = Some(vec![cron.to_string()]);
        wf.schedule_since = Some(since.to_string());
        wf
    }

    fn schedule_variant(r: &LivenessReport) -> Hexavalent {
        let json = r
            .observations
            .iter()
            .map(|e| serde_json::to_value(e).unwrap())
            .find(|j| j["source"] == SCHEDULE_SOURCE)
            .expect("a schedule observation");
        serde_json::from_value(json["variant"].clone()).unwrap()
    }

    #[test]
    fn healthy_nightly_is_true() {
        let wf = nightly(&[(14, "success"), (13, "success"), (12, "failure")]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.verdict, Hexavalent::True);
        assert_eq!(r.failure_streak, 0);
        assert!((r.cadence_hours - 24.0).abs() < 1e-9);
        assert!(!r.is_unhealthy());
    }

    #[test]
    fn one_failed_night_is_doubtful() {
        let wf = nightly(&[(14, "failure"), (13, "success"), (12, "success")]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.verdict, Hexavalent::Doubtful);
        assert_eq!(r.failure_streak, 1);
    }

    #[test]
    fn cancelled_runs_count_toward_the_streak() {
        // chatbot-qa died a month "cancelled" at the timeout wall: grey, not red.
        let wf = nightly(&[
            (14, "cancelled"),
            (13, "cancelled"),
            (12, "timed_out"),
            (11, "success"),
        ]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.failure_streak, 3);
        assert_eq!(r.verdict, Hexavalent::False);
    }

    #[test]
    fn test_pr_successes_do_not_mask_scheduled_failures() {
        let mut wf = nightly(&[(14, "failure"), (13, "failure"), (12, "failure")]);
        wf.runs
            .insert(0, run("2026-09-14T11:00:00Z", "pull_request", "success"));
        wf.runs
            .insert(2, run("2026-09-13T15:00:00Z", "pull_request", "success"));
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.failure_streak, 3);
        assert_eq!(r.scheduled_runs, 3);
        assert_eq!(r.verdict, Hexavalent::False);
    }

    #[test]
    fn green_but_dead_escalates_to_contradictory() {
        // Last scheduled run succeeded, but it was ten days ago on a daily cron.
        let wf = nightly(&[(4, "success"), (3, "success"), (2, "success")]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.failure_streak, 0);
        assert_eq!(r.verdict, Hexavalent::Contradictory);
    }

    #[test]
    fn disabled_for_inactivity_is_false_even_without_runs() {
        let mut wf = nightly(&[]);
        wf.state = Some("disabled_inactivity".to_string());
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.verdict, Hexavalent::False);
        assert_eq!(schedule_variant(&r), Hexavalent::False);
    }

    #[test]
    fn event_triggered_workflow_is_not_assessed() {
        let mut wf = nightly(&[]);
        wf.runs
            .push(run("2026-09-14T11:00:00Z", "pull_request", "failure"));
        assert!(assess(&wf, NOW, 0).unwrap().is_none());
    }

    #[test]
    fn workflow_without_cron_is_not_assessed_even_with_old_scheduled_runs() {
        // Demerzel Auto-Remediation: the cron was removed in March; its old
        // scheduled runs are history, not a dead loop.
        let mut wf = nightly(&[(4, "success"), (3, "success")]);
        wf.crons = Some(vec![]);
        assert!(assess(&wf, NOW, 0).unwrap().is_none());
    }

    #[test]
    fn irregular_cron_is_judged_by_its_calendar_not_the_median_gap() {
        // Demerzel Show & Tell: daily on days 1-7 and 15-21, or any Monday.
        // Silent from Tuesday the 8th to Monday the 14th before 14:07 is correct.
        let mut wf = nightly(&[]);
        wf.runs = (4..=7)
            .map(|d| run(&format!("2026-09-{d:02}T16:30:00Z"), "schedule", "success"))
            .collect();
        wf.crons = Some(vec!["7 14 1-7,15-21 * 1".to_string()]);
        let r = assess(&wf, "2026-09-14T13:00:00Z", 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(0));
        assert_eq!(r.verdict, Hexavalent::True);
        // The median-gap fallback would have called it dead.
        wf.crons = None;
        let fallback = assess(&wf, "2026-09-14T13:00:00Z", 0).unwrap().unwrap();
        assert!(fallback.is_unhealthy());
    }

    #[test]
    fn throttled_frequent_cron_is_normalized_by_delivered_fires_per_run() {
        // A `*/30` cron that GitHub delivers roughly every 3h: 7h of silence
        // owes ~2 runs at that rate, not 14.
        let mut wf = nightly(&[]);
        wf.crons = Some(vec!["*/30 * * * *".to_string()]);
        wf.runs = ["00", "03", "06"]
            .iter()
            .map(|h| run(&format!("2026-09-14T{h}:00:00Z"), "schedule", "success"))
            .collect();
        let r = assess(&wf, "2026-09-14T13:00:00Z", 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(1));
        assert_eq!(r.verdict, Hexavalent::True);
    }

    #[test]
    fn new_or_renamed_workflow_owes_nothing_before_its_file_changed() {
        // Added (or moved, which gives a new workflow id) this morning,
        // after its 05:00 fire: Unknown, not dead.
        let wf = scheduled("0 5 * * *", "2026-09-14T08:00:00Z");
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(0));
        assert_eq!(r.verdict, Hexavalent::Unknown);
        assert!(!r.is_unhealthy());
    }

    #[test]
    fn monthly_cron_that_never_ran_is_not_reliable_and_stays_visible() {
        // Two monthly fires (08-01, 09-01) owed since mid-July, no run.
        let wf = scheduled("0 6 1 * *", "2026-07-15T00:00:00Z");
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(2));
        assert_eq!(r.verdict, Hexavalent::Doubtful);
        // No fire falls between 09-14 and 09-20: still assessed.
        let later = assess(&wf, "2026-09-20T12:00:00Z", 0).unwrap().unwrap();
        assert_eq!(later.verdict, Hexavalent::Doubtful);
        // One owed fire and no run is not T either.
        let once = scheduled("0 6 1 * *", "2026-08-15T00:00:00Z");
        let r = assess(&once, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(1));
        assert_eq!(r.verdict, Hexavalent::Unknown);
    }

    #[test]
    fn weekly_cron_that_never_ran_for_a_month_is_false() {
        // Mondays 08-17 .. 09-14 at 06:00.
        let wf = scheduled("0 6 * * 1", "2026-08-15T00:00:00Z");
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(5));
        assert_eq!(r.verdict, Hexavalent::False);
    }

    #[test]
    fn file_change_after_the_last_run_resets_what_is_owed() {
        let mut wf = nightly(&[(4, "success"), (3, "success")]);
        wf.crons = Some(vec!["0 9 * * *".to_string()]);
        wf.schedule_since = Some("2026-09-14T10:00:00Z".to_string());
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(0));
        assert_eq!(r.verdict, Hexavalent::True);
    }

    #[test]
    fn past_outage_does_not_excuse_a_later_death() {
        // Daily runs 08-01..08-10, a month-long outage, runs 09-09..09-12,
        // then six silent nights. A window-mean throttle factor read this as T.
        let mut wf = nightly(&[]);
        wf.crons = Some(vec!["0 9 * * *".to_string()]);
        wf.runs = (1..=10)
            .map(|d| format!("2026-08-{d:02}T09:00:00Z"))
            .chain((9..=12).map(|d| format!("2026-09-{d:02}T09:00:00Z")))
            .map(|t| run(&t, "schedule", "success"))
            .collect();
        let r = assess(&wf, "2026-09-18T12:00:00Z", 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(6));
        assert_eq!(r.verdict, Hexavalent::Contradictory);
    }

    #[test]
    fn cron_change_cannot_inflate_the_throttle_factor_past_its_cap() {
        // Daily history, cron now hourly, silent for two days: 48 fires owed,
        // 24 per past gap capped at 12, so 4 runs owed.
        let mut wf = nightly(&[(10, "success"), (11, "success"), (12, "success")]);
        wf.crons = Some(vec!["0 * * * *".to_string()]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(4));
        assert_eq!(r.verdict, Hexavalent::Contradictory);
    }

    #[test]
    fn throttle_factor_never_drops_below_one_run_per_fire() {
        // Daily history, cron now weekly: past gaps hold no fire (median 0).
        let mut wf = nightly(&[]);
        wf.crons = Some(vec!["0 9 * * 1".to_string()]);
        wf.runs = (1..=5)
            .map(|d| run(&format!("2026-09-{d:02}T10:00:00Z"), "schedule", "success"))
            .collect();
        let r = assess(&wf, "2026-09-28T12:00:00Z", 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(4));
    }

    #[test]
    fn one_owed_run_is_tolerated() {
        let mut wf = nightly(&[(12, "skipped"), (13, "skipped")]);
        wf.crons = Some(vec!["0 9 * * *".to_string()]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(1));
        assert_eq!(schedule_variant(&r), Hexavalent::True);
        assert_eq!(r.verdict, Hexavalent::Unknown);
    }

    #[test]
    fn verdict_ties_break_pessimistically() {
        use ix_fuzzy::hexavalent::hexavalent_from_tpudfc;
        let tie = hexavalent_from_tpudfc(0.5, 0.0, 0.0, 0.0, 0.5, 0.0).unwrap();
        assert_eq!(verdict(&tie), Hexavalent::False);
        let tie = hexavalent_from_tpudfc(0.5, 0.0, 0.0, 0.5, 0.0, 0.0).unwrap();
        assert_eq!(verdict(&tie), Hexavalent::Doubtful);
    }

    #[test]
    fn unreadable_workflow_is_unknown_and_listed_not_believed() {
        let mut wf = nightly(&[]);
        wf.fetch_error = Some("gh api: HTTP 502".to_string());
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.verdict, Hexavalent::Unknown);
        assert!(r.observations.is_empty());
        let md = to_markdown(std::slice::from_ref(&r));
        assert!(
            md.starts_with("0 of 0 scheduled workflows unhealthy"),
            "{md}"
        );
        assert!(md.contains("read failed (1)"), "{md}");
        assert_eq!(to_hari_session(&[r], 0).len(), 2, "open + close only");
    }

    #[test]
    fn markdown_escapes_names_and_marks_truncated_streaks() {
        let mut wf = nightly(&[(14, "failure"), (13, "failure"), (12, "failure")]);
        wf.workflow = "Build | deploy @owner".to_string();
        wf.runs_truncated = true;
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        let md = to_markdown(&[r]);
        assert!(md.contains("Build \\| deploy &#64;owner"), "{md}");
        assert!(md.contains("| ≥3 |"), "{md}");
        assert!(md.contains("none in last 3"), "{md}");

        wf.runs_truncated = false;
        let md = to_markdown(&[assess(&wf, NOW, 0).unwrap().unwrap()]);
        assert!(md.contains("| 3 |"), "{md}");
        assert!(md.contains("never (3 scheduled runs)"), "{md}");
    }

    #[test]
    fn green_but_dead_with_cron_escalates_to_contradictory() {
        let mut wf = nightly(&[(4, "success"), (3, "success"), (2, "success")]);
        wf.crons = Some(vec!["0 9 * * *".to_string()]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.missed_runs, Some(10));
        assert_eq!(r.verdict, Hexavalent::Contradictory);
    }

    #[test]
    fn all_skipped_runs_are_unknown_not_unhealthy() {
        let mut wf = nightly(&[(14, "skipped"), (13, "skipped")]);
        wf.crons = Some(vec!["0 9 * * *".to_string()]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.verdict, Hexavalent::Unknown);
        assert!(!r.is_unhealthy());
    }

    #[test]
    fn claim_key_and_sources_match_session_event_schema() {
        // session-event.schema.json: claim_key ^[a-z][a-zA-Z0-9_:]*::[a-z_]+$,
        // source ^[a-z][a-z0-9-]*$.
        let wf = nightly(&[(14, "failure"), (13, "success")]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        assert_eq!(r.claim_key, "gha_loop:ix:nightly_thing::reliable");
        let (action, aspect) = r.claim_key.rsplit_once("::").unwrap();
        assert!(action.starts_with(|c: char| c.is_ascii_lowercase()));
        assert!(action
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':'));
        assert!(aspect.chars().all(|c| c.is_ascii_lowercase() || c == '_'));
        for e in &r.observations {
            let json = serde_json::to_value(e).unwrap();
            let source = json["source"].as_str().unwrap();
            assert!(source
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'));
            assert_eq!(json["diagnosis_id"].as_str().unwrap().len(), 64);
        }
    }

    #[test]
    fn assessment_is_deterministic_for_fixed_now() {
        let wf = nightly(&[(14, "failure"), (13, "success")]);
        let a = assess(&wf, NOW, 3).unwrap().unwrap();
        let b = assess(&wf, NOW, 3).unwrap().unwrap();
        assert_eq!(a.observations, b.observations);
        assert_eq!(a.distribution, b.distribution);
    }

    #[test]
    fn invalid_now_is_a_parse_error() {
        let wf = nightly(&[(14, "success")]);
        assert!(matches!(
            assess(&wf, "yesterday", 0),
            Err(AdapterError::Parse(_))
        ));
    }

    #[test]
    fn hari_session_has_open_events_close() {
        let wf = nightly(&[(14, "failure"), (13, "failure"), (12, "failure")]);
        let r = assess(&wf, NOW, 0).unwrap().unwrap();
        let lines = to_hari_session(&[r], 7);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["op"], "open");
        assert_eq!(lines[1]["event"]["cycle"], 7);
        assert_eq!(lines[1]["event"]["payload"]["type"], "belief_update");
        assert_eq!(lines[1]["event"]["payload"]["value"], "False");
        assert_eq!(lines[2]["op"], "close");
    }

    #[test]
    fn markdown_lists_only_unhealthy_worst_first() {
        let healthy = assess(&nightly(&[(14, "success"), (13, "success")]), NOW, 0)
            .unwrap()
            .unwrap();
        let doubtful = assess(&nightly(&[(14, "failure"), (13, "success")]), NOW, 0)
            .unwrap()
            .unwrap();
        let dead = assess(
            &nightly(&[(14, "failure"), (13, "failure"), (12, "failure")]),
            NOW,
            0,
        )
        .unwrap()
        .unwrap();
        let md = to_markdown(&[healthy, doubtful, dead]);
        assert!(md.starts_with("2 of 3 scheduled workflows unhealthy"));
        let f = md.find("| F |").unwrap();
        let d = md.find("| D |").unwrap();
        assert!(f < d, "F rows sort before D rows:\n{md}");
    }
}
