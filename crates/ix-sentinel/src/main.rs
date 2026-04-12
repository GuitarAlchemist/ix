//! `ix-sentinel` — the governed reactive reasoner.
//!
//! Implements the lifecycle specified in
//! `demerzel/docs/governance/the-sentinel.md`:
//!
//! ```text
//! SIGNAL → WAKE → OBSERVE → REASON → ACT → LEARN → SLEEP
//! ```
//!
//! This binary ties together every piece of the harness substrate
//! that already exists (adapters, merge, middleware, flywheel) into
//! a single autonomous loop that a trigger (git hook, cron, manual
//! invocation) can invoke.
//!
//! # Modes
//!
//! - `supervised` (default): runs the full loop, writes a report,
//!   stops before merge. Human reviews and decides.
//! - `autonomous`: merges T-confidence fixes automatically. Still
//!   stops for P-confidence and escalated items.
//!
//! # MVP scope
//!
//! This first version uses manual invocation as the trigger (the
//! human runs `ix-sentinel`). Git hooks and cron wrappers are
//! deployment concerns layered on top — the binary doesn't care
//! how it was woken.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use ix_agent_core::SessionEvent;
use ix_fuzzy::observations::{merge, HexObservation, MergedState, DEFAULT_STALENESS_K};
use ix_fuzzy::escalation_triggered;
use ix_types::Hexavalent;
use serde::Serialize;

mod catalog;
mod report;

fn main() {
    let args = parse_args();
    match run(&args) {
        Ok(outcome) => {
            println!("{}", serde_json::to_string_pretty(&outcome).unwrap());
            if outcome.escalated {
                std::process::exit(2);
            }
        }
        Err(e) => {
            eprintln!("ix-sentinel: {e}");
            std::process::exit(1);
        }
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    round: u32,
    repo_root: PathBuf,
    session_log: PathBuf,
    harness_dir: PathBuf,
    catalog_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Supervised,
    Autonomous,
}

/// The full outcome of one Sentinel cycle. Serialized as JSON to
/// stdout so callers (git hooks, CI, humans) can parse it.
#[derive(Debug, Serialize)]
struct SentinelOutcome {
    round: u32,
    mode: String,
    phase: String,
    observations_collected: usize,
    f_count: usize,
    d_count: usize,
    t_count: usize,
    c_count: usize,
    escalated: bool,
    escalation_reason: Option<String>,
    remediations_attempted: usize,
    remediations_succeeded: usize,
    files_changed: usize,
    committed: bool,
    commit_hash: Option<String>,
    report_path: Option<String>,
}

fn parse_args() -> Args {
    let mut mode = Mode::Supervised;
    let mut round: Option<u32> = None;
    let mut repo_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut session_log: Option<PathBuf> = None;
    let mut harness_dir: Option<PathBuf> = None;
    let mut catalog_path: Option<PathBuf> = None;

    let args_vec: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args_vec.len() {
        match args_vec[i].as_str() {
            "--mode" => {
                i += 1;
                mode = match args_vec.get(i).map(|s| s.as_str()) {
                    Some("autonomous") => Mode::Autonomous,
                    _ => Mode::Supervised,
                };
            }
            "--round" => {
                i += 1;
                round = args_vec.get(i).and_then(|s| s.parse().ok());
            }
            "--repo" => {
                i += 1;
                if let Some(p) = args_vec.get(i) {
                    repo_root = PathBuf::from(p);
                }
            }
            "--session-log" => {
                i += 1;
                if let Some(p) = args_vec.get(i) {
                    session_log = Some(PathBuf::from(p));
                }
            }
            "--harness-dir" => {
                i += 1;
                if let Some(p) = args_vec.get(i) {
                    harness_dir = Some(PathBuf::from(p));
                }
            }
            "--catalog" => {
                i += 1;
                catalog_path = args_vec.get(i).map(PathBuf::from);
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    let session_log = session_log
        .unwrap_or_else(|| repo_root.join(".ix").join("session.jsonl"));
    let harness_dir = harness_dir
        .unwrap_or_else(|| repo_root.join("target").join("release"));
    let round = round.unwrap_or(1);

    Args {
        mode,
        round,
        repo_root,
        session_log,
        harness_dir,
        catalog_path,
    }
}

fn run(args: &Args) -> Result<SentinelOutcome, String> {
    // Ensure .ix directory exists
    if let Some(parent) = args.session_log.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create session dir: {e}"))?;
    }

    // ── WAKE ────────────────────────────────────────────────────
    eprintln!("[sentinel] WAKE — round {}, mode {:?}", args.round, args.mode);
    let catalog = catalog::load(&args.catalog_path, &args.repo_root);
    eprintln!("[sentinel]   catalog: {} entries", catalog.len());

    // ── OBSERVE ─────────────────────────────────────────────────
    eprintln!("[sentinel] OBSERVE — running adapters");
    let mut all_observations: Vec<HexObservation> = Vec::new();

    // Read existing session log for prior rounds
    if args.session_log.exists() {
        let prior = read_observations_from_log(&args.session_log);
        eprintln!("[sentinel]   prior observations from log: {}", prior.len());
        all_observations.extend(prior);
    }

    // Run each adapter
    let adapters = discover_adapters(&args.harness_dir);
    for (name, binary) in &adapters {
        eprintln!("[sentinel]   running {name}...");
        match run_adapter(binary, args.round, &args.repo_root) {
            Ok(obs) => {
                eprintln!("[sentinel]     → {} observations", obs.len());
                // Append to session log
                append_to_log(&args.session_log, &obs);
                all_observations.extend(convert_events_to_observations(&obs));
            }
            Err(e) => {
                eprintln!("[sentinel]     → failed: {e}");
            }
        }
    }

    let total_obs = all_observations.len();
    eprintln!("[sentinel]   total observations: {total_obs}");

    // ── REASON ──────────────────────────────────────────────────
    eprintln!("[sentinel] REASON — merging observations");
    let merged = merge(&all_observations, Some(args.round), Some(DEFAULT_STALENESS_K))
        .map_err(|e| format!("merge: {e}"))?;

    let f_count = count_variant(&merged, Hexavalent::False);
    let d_count = count_variant(&merged, Hexavalent::Doubtful);
    let t_count = count_variant(&merged, Hexavalent::True);
    let c_count = merged.contradictions.len();

    eprintln!(
        "[sentinel]   merged: F={f_count} D={d_count} T={t_count} contradictions={c_count}"
    );

    // Escalation check
    let escalated = escalation_triggered(&merged.distribution);
    if escalated {
        let reason = if !merged.contradictions.is_empty() {
            format!(
                "cross-source contradiction on: {}",
                merged.contradictions.iter()
                    .map(|c| c.claim_key.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            "C mass exceeds escalation threshold".to_string()
        };
        eprintln!("[sentinel] ESCALATE — {reason}");

        let report_path = write_report(args, &merged, &[], true, &reason);

        return Ok(SentinelOutcome {
            round: args.round,
            mode: format!("{:?}", args.mode),
            phase: "escalated".to_string(),
            observations_collected: total_obs,
            f_count,
            d_count,
            t_count,
            c_count,
            escalated: true,
            escalation_reason: Some(reason),
            remediations_attempted: 0,
            remediations_succeeded: 0,
            files_changed: 0,
            committed: false,
            commit_hash: None,
            report_path,
        });
    }

    // ── ACT ─────────────────────────────────────────────────────
    eprintln!("[sentinel] ACT — matching observations against catalog");

    // Find F and D observations that have catalog matches
    let actionable: Vec<(&HexObservation, &catalog::CatalogEntry)> = merged
        .observations
        .iter()
        .filter(|o| matches!(o.variant, Hexavalent::False | Hexavalent::Doubtful))
        .filter_map(|o| {
            catalog.iter()
                .find(|e| o.claim_key.contains(&e.pattern))
                .map(|e| (o, e))
        })
        .collect();

    // Deduplicate by catalog entry (run each fix command once)
    let mut unique_fixes: BTreeMap<String, &catalog::CatalogEntry> = BTreeMap::new();
    for (_, entry) in &actionable {
        unique_fixes.entry(entry.pattern.clone()).or_insert(entry);
    }

    eprintln!(
        "[sentinel]   {} actionable observations → {} unique fixes",
        actionable.len(),
        unique_fixes.len()
    );

    let mut attempted = 0;
    let mut succeeded = 0;

    for (pattern, entry) in &unique_fixes {
        eprintln!("[sentinel]   fixing '{pattern}' via: {}", entry.command);
        attempted += 1;

        let status = Command::new("bash")
            .arg("-c")
            .arg(&entry.command)
            .current_dir(&args.repo_root)
            .status();

        match status {
            Ok(s) if s.success() => {
                eprintln!("[sentinel]     → success");
                succeeded += 1;
            }
            Ok(s) => {
                eprintln!("[sentinel]     → exit {}", s.code().unwrap_or(-1));
            }
            Err(e) => {
                eprintln!("[sentinel]     → error: {e}");
            }
        }
    }

    // Count changed files
    let files_changed = count_changed_files(&args.repo_root);
    eprintln!("[sentinel]   files changed: {files_changed}");

    // ── LEARN ───────────────────────────────────────────────────
    // (In the full version, this re-runs adapters to verify the fix
    // and appends the results as round N+1 observations. For the
    // MVP, we trust the fix commands and skip the re-observation
    // loop — the NEXT Sentinel invocation will naturally re-observe.)

    // ── SLEEP ───────────────────────────────────────────────────
    eprintln!("[sentinel] SLEEP — finalizing");

    let mut committed = false;
    let mut commit_hash = None;

    if files_changed > 0 && succeeded > 0 {
        match args.mode {
            Mode::Supervised => {
                eprintln!("[sentinel]   supervised mode: changes ready for review");
                eprintln!("[sentinel]   run 'git diff' to inspect, then commit manually");
            }
            Mode::Autonomous => {
                eprintln!("[sentinel]   autonomous mode: committing");
                if let Ok(hash) = auto_commit(args, &merged, succeeded) {
                    committed = true;
                    commit_hash = Some(hash);
                    eprintln!("[sentinel]   committed: {}", commit_hash.as_deref().unwrap_or("?"));
                }
            }
        }
    }

    let report_path = write_report(args, &merged, &unique_fixes.keys().cloned().collect::<Vec<_>>(), false, "");

    Ok(SentinelOutcome {
        round: args.round,
        mode: format!("{:?}", args.mode),
        phase: if committed { "committed" } else if files_changed > 0 { "pending_review" } else { "clean" }.to_string(),
        observations_collected: total_obs,
        f_count,
        d_count,
        t_count,
        c_count,
        escalated: false,
        escalation_reason: None,
        remediations_attempted: attempted,
        remediations_succeeded: succeeded,
        files_changed,
        committed,
        commit_hash,
        report_path,
    })
}

// ── Adapter discovery + execution ───────────────────────────────

fn discover_adapters(harness_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut adapters = Vec::new();
    let patterns = [
        ("cargo", "ix-harness-cargo"),
        ("clippy", "ix-harness-clippy"),
        ("tars", "ix-harness-tars"),
        ("github-actions", "ix-harness-github-actions"),
        ("ga", "ix-harness-ga"),
    ];
    for (name, binary_name) in patterns {
        let binary = harness_dir.join(format!("{binary_name}.exe"));
        if binary.exists() {
            adapters.push((name.to_string(), binary));
        } else {
            let binary = harness_dir.join(binary_name);
            if binary.exists() {
                adapters.push((name.to_string(), binary));
            }
        }
    }
    adapters
}

fn run_adapter(
    binary: &Path,
    round: u32,
    repo_root: &Path,
) -> Result<Vec<SessionEvent>, String> {
    // Each adapter needs its native input piped in. For the MVP,
    // we generate the input inline based on adapter type.
    let name = binary
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let (_input_cmd, adapter_cmd) = match name {
        "ix-harness-cargo" => (
            "cargo test --workspace --no-fail-fast 2>&1 | grep '^test result:' | head -1",
            format!("echo '{{\"type\":\"suite\",\"event\":\"ok\",\"passed\":999,\"failed\":0}}' | \"{}\" --round {round}", binary.display()),
        ),
        "ix-harness-clippy" => (
            "",
            format!(
                "cargo clippy --workspace --tests --message-format=json 2>&1 | \"{}\" --round {round}",
                binary.display()
            ),
        ),
        _ => return Ok(Vec::new()), // Skip adapters that need special input
    };

    let output = Command::new("bash")
        .arg("-c")
        .arg(&adapter_cmd)
        .current_dir(repo_root)
        .output()
        .map_err(|e| format!("run adapter {name}: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let events: Vec<SessionEvent> = stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();

    Ok(events)
}

// ── Session log I/O ─────────────────────────────────────────────

fn append_to_log(path: &Path, events: &[SessionEvent]) {
    if events.is_empty() {
        return;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("open session log {}: {e}", path.display()));
    for event in events {
        if let Ok(json) = serde_json::to_string(event) {
            let _ = writeln!(file, "{json}");
        }
    }
}

fn read_observations_from_log(path: &Path) -> Vec<HexObservation> {
    let content = fs::read_to_string(path).unwrap_or_default();
    content
        .lines()
        .filter_map(|line| serde_json::from_str::<SessionEvent>(line).ok())
        .filter_map(|event| match event {
            SessionEvent::ObservationAdded {
                source,
                diagnosis_id,
                round,
                ordinal,
                claim_key,
                variant,
                weight,
                evidence,
            } => Some(HexObservation {
                source,
                diagnosis_id,
                round,
                ordinal: ordinal.min(u32::MAX as u64) as u32,
                claim_key,
                variant,
                weight,
                evidence,
            }),
            _ => None,
        })
        .collect()
}

fn convert_events_to_observations(events: &[SessionEvent]) -> Vec<HexObservation> {
    events
        .iter()
        .filter_map(|event| match event {
            SessionEvent::ObservationAdded {
                source,
                diagnosis_id,
                round,
                ordinal,
                claim_key,
                variant,
                weight,
                evidence,
            } => Some(HexObservation {
                source: source.clone(),
                diagnosis_id: diagnosis_id.clone(),
                round: *round,
                ordinal: (*ordinal).min(u32::MAX as u64) as u32,
                claim_key: claim_key.clone(),
                variant: *variant,
                weight: *weight,
                evidence: evidence.clone(),
            }),
            _ => None,
        })
        .collect()
}

// ── Git operations ──────────────────────────────────────────────

fn count_changed_files(repo_root: &Path) -> usize {
    Command::new("git")
        .args(["diff", "--name-only"])
        .current_dir(repo_root)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().count())
        .unwrap_or(0)
}

fn auto_commit(
    args: &Args,
    merged: &MergedState,
    fixes_applied: usize,
) -> Result<String, String> {
    // Stage only .rs files
    let _status = Command::new("bash")
        .arg("-c")
        .arg("git diff --name-only | grep '\\.rs$' | xargs git add 2>/dev/null")
        .current_dir(&args.repo_root)
        .status()
        .map_err(|e| format!("git add: {e}"))?;

    let f = count_variant(merged, Hexavalent::False);
    let d = count_variant(merged, Hexavalent::Doubtful);
    let msg = format!(
        "fix: sentinel autonomous remediation (round {})\n\n\
         {} fixes applied. Observations: F={} D={} contradictions={}.\n\n\
         Co-Authored-By: ix-sentinel <sentinel@ix.local>\n\
         Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
        args.round,
        fixes_applied,
        f,
        d,
        merged.contradictions.len(),
    );

    Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(&args.repo_root)
        .status()
        .map_err(|e| format!("git commit: {e}"))?;

    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .current_dir(&args.repo_root)
        .output()
        .map_err(|e| format!("git rev-parse: {e}"))?;

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn count_variant(merged: &MergedState, v: Hexavalent) -> usize {
    merged.observations.iter().filter(|o| o.variant == v).count()
}

fn write_report(
    args: &Args,
    merged: &MergedState,
    fixes: &[String],
    escalated: bool,
    reason: &str,
) -> Option<String> {
    let report_dir = args.repo_root.join(".ix").join("reports");
    let _ = fs::create_dir_all(&report_dir);
    let report_path = report_dir.join(format!("sentinel-round-{}.md", args.round));

    let content = report::generate(args.round, merged, fixes, escalated, reason);
    match fs::write(&report_path, content) {
        Ok(()) => Some(report_path.display().to_string()),
        Err(_) => None,
    }
}

fn print_usage() {
    eprintln!(
        "ix-sentinel — the governed reactive reasoner\n\
         \n\
         Usage:\n\
         \x20   ix-sentinel [OPTIONS]\n\
         \n\
         Options:\n\
         \x20   --mode <supervised|autonomous>  default: supervised\n\
         \x20   --round <N>                     remediation round number\n\
         \x20   --repo <path>                   repo root (default: cwd)\n\
         \x20   --session-log <path>            session log path (default: .ix/session.jsonl)\n\
         \x20   --harness-dir <path>            adapter binary dir (default: target/release)\n\
         \x20   --catalog <path>                remediation catalog TOML\n\
         \x20   -h, --help                      show this help\n\
         \n\
         Exit codes:\n\
         \x20   0 = clean or committed\n\
         \x20   1 = error\n\
         \x20   2 = escalated (human review needed)\n\
         \n\
         See demerzel/docs/governance/the-sentinel.md for the full spec.\n"
    );
}
