//! CLI wrapper for ix-harness-github-actions.

use std::fs;
use std::io::{self, Read, Write};
use std::process::{Command, ExitCode};
use std::time::{SystemTime, UNIX_EPOCH};

use ix_harness_github_actions::cron::crons_in_workflow;
use ix_harness_github_actions::github_actions_to_observations;
use ix_harness_github_actions::liveness::{self, RunRecord, WorkflowRuns};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("ix-harness-github-actions: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("liveness") {
        return run_liveness(&args[2..]);
    }
    let mut round: Option<u32> = None;
    let mut input_path: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--round" => {
                i += 1;
                round = Some(
                    args[i]
                        .parse()
                        .map_err(|_| format!("invalid round: {}", args[i]))?,
                );
            }
            "--input" => {
                i += 1;
                input_path = Some(args[i].clone());
            }
            "--output" => {
                i += 1;
                output_path = Some(args[i].clone());
            }
            "-h" | "--help" => {
                eprintln!(
                    "ix-harness-github-actions — GitHub Actions run JSON → SessionEvent JSONL\n\
                     \n\
                     Usage:\n\
                     \x20   ix-harness-github-actions --round <N> [--input <path>] [--output <path>]\n\
                     \x20   ix-harness-github-actions liveness --help\n\
                     \n\
                     Input format: combined run + jobs JSON\n\
                     \x20   {{\"run\": {{<workflow run fields>}}, \"jobs\": [<job objects>]}}\n\
                     \n\
                     See demerzel/logic/harness-github-actions.md for projection rules.\n"
                );
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}")),
        }
        i += 1;
    }
    let round = round.ok_or_else(|| "missing required --round <N>".to_string())?;

    let input_bytes: Vec<u8> = match input_path {
        Some(p) => fs::read(&p).map_err(|e| format!("read {p}: {e}"))?,
        None => {
            let mut buf = Vec::new();
            io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| format!("read stdin: {e}"))?;
            buf
        }
    };

    let events = github_actions_to_observations(&input_bytes, round).map_err(|e| e.to_string())?;

    let mut out: Box<dyn Write> = match output_path {
        Some(p) => Box::new(fs::File::create(&p).map_err(|e| format!("create {p}: {e}"))?),
        None => Box::new(io::stdout()),
    };
    for event in &events {
        let line = serde_json::to_string(event).map_err(|e| format!("serialize: {e}"))?;
        writeln!(out, "{line}").map_err(|e| format!("write: {e}"))?;
    }
    Ok(())
}

fn run_liveness(args: &[String]) -> Result<(), String> {
    let mut repos: Vec<String> = Vec::new();
    let mut input_path: Option<String> = None;
    let mut dump_input: Option<String> = None;
    let mut output_path: Option<String> = None;
    let mut now: Option<String> = None;
    let mut round: u32 = 0;
    let mut limit: u32 = 30;
    let mut format = "observations".to_string();
    let value = |i: usize| -> Result<String, String> {
        args.get(i)
            .cloned()
            .ok_or_else(|| format!("missing value for {}", args[i - 1]))
    };
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--repo" => {
                i += 1;
                repos.push(value(i)?);
            }
            "--input" => {
                i += 1;
                input_path = Some(value(i)?);
            }
            "--dump-input" => {
                i += 1;
                dump_input = Some(value(i)?);
            }
            "--output" => {
                i += 1;
                output_path = Some(value(i)?);
            }
            "--now" => {
                i += 1;
                now = Some(value(i)?);
            }
            "--round" => {
                i += 1;
                round = value(i)?
                    .parse()
                    .map_err(|_| format!("invalid round: {}", args[i]))?;
            }
            "--limit" => {
                i += 1;
                limit = value(i)?
                    .parse()
                    .map_err(|_| format!("invalid limit: {}", args[i]))?;
            }
            "--format" => {
                i += 1;
                format = value(i)?;
            }
            "-h" | "--help" => {
                eprintln!(
                    "ix-harness-github-actions liveness — scheduled-workflow liveness verdicts\n\
                     \n\
                     Usage:\n\
                     \x20   ix-harness-github-actions liveness --repo <owner/name> [--repo ...] [--limit 30]\n\
                     \x20   ix-harness-github-actions liveness --input <workflows.json>\n\
                     \n\
                     Options:\n\
                     \x20   --format observations|hari-session|json|markdown   (default observations)\n\
                     \x20   --now <YYYY-MM-DDTHH:MM:SSZ>   clock for age computations (default: now)\n\
                     \x20   --round <N>                    observation round / hari cycle (default 0)\n\
                     \x20   --dump-input <path>            save the fetched history for offline replay\n\
                     \x20   --output <path>\n\
                     \n\
                     --repo makes read-only gh calls: `workflow list`, the workflow file (for its crons)\n\
                     and the Actions API's scheduled runs of each workflow.\n\
                     --input takes a JSON array of {{repo, workflow, path, state, crons, schedule_since, runs: [{{conclusion, status, createdAt, event}}]}}.\n"
                );
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}")),
        }
        i += 1;
    }

    let workflows: Vec<WorkflowRuns> = match (&input_path, repos.is_empty()) {
        (Some(p), true) => {
            let bytes = fs::read(p).map_err(|e| format!("read {p}: {e}"))?;
            serde_json::from_slice(&bytes).map_err(|e| format!("parse {p}: {e}"))?
        }
        (None, false) => {
            let mut all = Vec::new();
            for repo in &repos {
                all.extend(fetch_repo(repo, limit));
            }
            all
        }
        _ => return Err("pass either --input <path> or one or more --repo".to_string()),
    };
    if let Some(p) = &dump_input {
        let json = serde_json::to_string_pretty(&workflows).map_err(|e| e.to_string())?;
        fs::write(p, json).map_err(|e| format!("write {p}: {e}"))?;
    }

    let now = now.unwrap_or_else(utc_now);
    let mut reports = Vec::new();
    for wf in &workflows {
        if let Some(report) = liveness::assess(wf, &now, round).map_err(|e| e.to_string())? {
            reports.push(report);
        }
    }

    let mut out: Box<dyn Write> = match output_path {
        Some(p) => Box::new(fs::File::create(&p).map_err(|e| format!("create {p}: {e}"))?),
        None => Box::new(io::stdout()),
    };
    let write_err = |e: io::Error| format!("write: {e}");
    match format.as_str() {
        "observations" => {
            for event in reports.iter().flat_map(|r| &r.observations) {
                let line = serde_json::to_string(event).map_err(|e| e.to_string())?;
                writeln!(out, "{line}").map_err(write_err)?;
            }
        }
        "hari-session" => {
            for line in liveness::to_hari_session(&reports, round) {
                writeln!(out, "{line}").map_err(write_err)?;
            }
        }
        "json" => {
            let json = serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?;
            writeln!(out, "{json}").map_err(write_err)?;
        }
        "markdown" => {
            write!(out, "{}", liveness::to_markdown(&reports)).map_err(write_err)?;
        }
        other => return Err(format!("unknown --format: {other}")),
    }
    Ok(())
}

/// `gh workflow list`, then per workflow file its crons and last change
/// (contents + commits API) and, when it has a schedule,
/// its scheduled runs (Actions API). Manually disabled workflows are a
/// deliberate kill, not a dead loop: skipped. Read errors never abort the
/// sweep; they come back as entries carrying `fetch_error`.
fn fetch_repo(repo: &str, limit: u32) -> Vec<WorkflowRuns> {
    #[derive(serde::Deserialize)]
    struct WorkflowRow {
        name: String,
        path: String,
        state: String,
    }
    let rows: Vec<WorkflowRow> = match gh(&[
        "workflow",
        "list",
        "-R",
        repo,
        "--all",
        "--limit",
        "500",
        "--json",
        "name,path,state",
    ]) {
        Ok(rows) => rows,
        Err(e) => return vec![read_failure(repo, "(workflow list)", None, e)],
    };
    rows.into_iter()
        .filter(|row| {
            row.path.starts_with(".github/workflows/") && row.state != "disabled_manually"
        })
        .map(|row| {
            fetch_workflow(repo, &row.name, &row.path, &row.state, limit)
                .unwrap_or_else(|e| read_failure(repo, &row.name, Some(row.path.clone()), e))
        })
        .collect()
}

fn fetch_workflow(
    repo: &str,
    name: &str,
    path: &str,
    state: &str,
    limit: u32,
) -> Result<WorkflowRuns, String> {
    let file = path.rsplit('/').next().unwrap_or(path);
    // A registered workflow whose file was deleted can no longer run: no schedule.
    let crons = match gh_raw(&[
        "api",
        "-H",
        "Accept: application/vnd.github.raw+json",
        &format!("repos/{repo}/contents/{path}"),
    ]) {
        Ok(yaml) => crons_in_workflow(&yaml),
        Err(e) if e.contains("HTTP 404") => Some(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut workflow = WorkflowRuns {
        repo: repo.to_string(),
        workflow: name.to_string(),
        path: Some(path.to_string()),
        state: Some(state.to_string()),
        crons,
        schedule_since: None,
        runs: Vec::new(),
        runs_truncated: false,
        fetch_error: None,
    };
    if workflow.crons.as_ref().is_some_and(Vec::is_empty) && state != "disabled_inactivity" {
        return Ok(workflow);
    }
    let since = gh_raw(&[
        "api",
        &format!("repos/{repo}/commits?path={path}&per_page=1"),
        "--jq",
        ".[0].commit.committer.date // empty",
    ])?;
    workflow.schedule_since = Some(since.trim().to_string()).filter(|s| !s.is_empty());
    // The Actions API, not `gh run list --event schedule`: on 2026-09-14 the
    // latter returned runs a week stale for ga's hourly Gemini triage
    // (6211 runs), which read as a dead schedule.
    let runs = |limit: u32| -> Result<Vec<RunRecord>, String> {
        let per_page = limit.min(100);
        let mut out: Vec<RunRecord> = Vec::new();
        for page in 1.. {
            let body = gh_raw(&[
                "api",
                &format!(
                    "repos/{repo}/actions/workflows/{file}/runs?event=schedule&per_page={per_page}&page={page}"
                ),
                "--jq",
                ".workflow_runs[] | {conclusion, status, createdAt: .created_at, event}",
            ])?;
            let before = out.len();
            for line in body.lines().filter(|l| !l.trim().is_empty()) {
                out.push(serde_json::from_str(line).map_err(|e| format!("parse run: {e}"))?);
            }
            let got = out.len() - before;
            if got < per_page as usize || out.len() >= limit as usize {
                break;
            }
        }
        out.truncate(limit as usize);
        Ok(out)
    };
    let mut fetched = runs(limit)?;
    let mut used = limit;
    // A window with no success hides how long the streak really is.
    if fetched.len() == limit as usize
        && fetched
            .iter()
            .all(|r| r.conclusion.as_deref() != Some("success"))
    {
        used = limit.max(1000);
        fetched = runs(used)?;
    }
    workflow.runs_truncated = fetched.len() == used as usize;
    workflow.runs = fetched;
    Ok(workflow)
}

fn read_failure(repo: &str, name: &str, path: Option<String>, error: String) -> WorkflowRuns {
    eprintln!("ix-harness-github-actions: {repo} {name}: {error}");
    WorkflowRuns {
        repo: repo.to_string(),
        workflow: name.to_string(),
        path,
        state: None,
        crons: None,
        schedule_since: None,
        runs: Vec::new(),
        runs_truncated: false,
        fetch_error: Some(error),
    }
}

fn gh<T: serde::de::DeserializeOwned>(args: &[&str]) -> Result<T, String> {
    serde_json::from_str(&gh_raw(args)?).map_err(|e| format!("parse gh {}: {e}", args.join(" ")))
}

fn gh_raw(args: &[&str]) -> Result<String, String> {
    let output = Command::new("gh")
        .args(args)
        .output()
        .map_err(|e| format!("spawn gh: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "gh {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("gh {}: {e}", args.join(" ")))
}

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ` (Hinnant's civil-from-days).
fn utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}
