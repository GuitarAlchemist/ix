//! ix-sentrux-gate-writer — wraps `sentrux gate` and appends a v1 ledger row.
//!
//! Usage:
//!
//!     ix-sentrux-gate-writer \
//!         --ledger state/quality/gate-ledger.jsonl \
//!         --path . \
//!         [--save] [--metric-name <name>]
//!
//! Runs `sentrux gate [PATH] [--save]`, captures the exit code, parses the
//! first numeric line from stdout as the metric value (sentrux's gate command
//! prints structural quality numbers — we capture the headline number).
//! On stdout-without-numbers we still emit a row with `decision=skip` and
//! `metric.value=NaN` so the producer is visible in the ledger.
//!
//! Exit code mirrors `sentrux gate`'s exit code (so CI gates still pass/fail).

use std::path::PathBuf;
use std::process::{Command, ExitCode};

use clap::Parser;
use ix_quality_trend::{
    append_entry, EvidenceKind, GateDecision, GateEvidence, GateLedgerEntry, GateMetric,
};

#[derive(Parser, Debug)]
#[command(version, about = "Wrap `sentrux gate` and append a v1 quality-gate-ledger row.")]
struct Args {
    /// Path to the ledger JSONL file.
    #[arg(long, default_value = "state/quality/gate-ledger.jsonl")]
    ledger: PathBuf,

    /// Directory to gate (passed to `sentrux gate`).
    #[arg(long, default_value = ".")]
    path: PathBuf,

    /// Pass `--save` through to sentrux (rebaselines).
    #[arg(long)]
    save: bool,

    /// Metric name to record. Sentrux's headline metric is the structural
    /// quality signal, but the name is producer-supplied so consumers can
    /// re-bind without code changes.
    #[arg(long, default_value = "structural_quality_signal")]
    metric_name: String,

    /// Override the sentrux binary path. Defaults to `sentrux` on PATH.
    #[arg(long, default_value = "sentrux")]
    sentrux_bin: String,

    /// Print the row to stdout instead of appending. Useful for dry runs.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    let mut cmd = Command::new(&args.sentrux_bin);
    cmd.arg("gate").arg(&args.path);
    if args.save {
        cmd.arg("--save");
    }

    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "ix-sentrux-gate-writer: failed to spawn `{}`: {}",
                args.sentrux_bin, e
            );
            // Emit a skip row so the failure is visible in the ledger.
            let mut entry = make_entry(
                GateDecision::Skip,
                0.0,
                &args.metric_name,
                &args.path,
                Some(format!("sentrux spawn failed: {}", e)),
            );
            // Mark the value as unknown (vs. a real zero) so consumers
            // don't treat a spawn failure as "metric measured at 0".
            if let Some(extra) = entry.extra.as_mut().and_then(|v| v.as_object_mut()) {
                extra.insert("value_unknown".into(), serde_json::Value::Bool(true));
            }
            emit(&args.ledger, &entry, args.dry_run);
            return ExitCode::from(2);
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let exit_code = output.status.code().unwrap_or(1) as u8;

    // Echo sentrux's own output so CI logs still show it.
    print!("{}", stdout);
    eprint!("{}", stderr);

    let (decision, value_opt) = interpret(&stdout, output.status.success());

    // Schema requires `metric.value` to be a number. When sentrux didn't
    // print a parseable value we record 0.0 plus `extra.value_unknown=true`
    // so consumers can distinguish "0 means zero" from "we didn't measure".
    let (recorded_value, value_unknown) = match value_opt {
        Some(v) => (v, false),
        None => (0.0, true),
    };

    let mut entry = make_entry(
        decision,
        recorded_value,
        &args.metric_name,
        &args.path,
        None,
    );
    let mut extra = serde_json::Map::new();
    if value_unknown {
        extra.insert("value_unknown".into(), serde_json::Value::Bool(true));
    }
    if !output.status.success() {
        extra.insert(
            "sentrux_exit_code".into(),
            serde_json::json!(output.status.code()),
        );
        extra.insert(
            "sentrux_stderr_tail".into(),
            serde_json::json!(tail(&stderr, 4)),
        );
    }
    if !extra.is_empty() {
        entry.extra = Some(serde_json::Value::Object(extra));
    }
    emit(&args.ledger, &entry, args.dry_run);

    ExitCode::from(exit_code)
}

fn make_entry(
    decision: GateDecision,
    value: f64,
    metric_name: &str,
    path: &std::path::Path,
    note: Option<String>,
) -> GateLedgerEntry {
    let mut e = GateLedgerEntry::new(
        "sentrux",
        "structural",
        decision,
        GateMetric {
            name: metric_name.to_string(),
            value,
            threshold: None,
            trend: None,
        },
    );
    e.evidence = Some(GateEvidence::new(
        EvidenceKind::File,
        path.display().to_string(),
    ));
    if let Some(n) = note {
        e.extra = Some(serde_json::json!({ "note": n }));
    }
    e
}

fn interpret(stdout: &str, success: bool) -> (GateDecision, Option<f64>) {
    // Sentrux prints lines like "Quality signal: 3015" — pull the first
    // number we see. If we can't find one, return None.
    let value = stdout.lines().find_map(|line| {
        line.split(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
            .find(|s| !s.is_empty())
            .and_then(|s| s.parse::<f64>().ok())
    });

    let decision = if success {
        GateDecision::Pass
    } else {
        GateDecision::Fail
    };
    (decision, value)
}

fn tail(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}

fn emit(ledger: &std::path::Path, entry: &GateLedgerEntry, dry_run: bool) {
    if dry_run {
        match serde_json::to_string_pretty(entry) {
            Ok(s) => println!("{}", s),
            Err(e) => eprintln!("ix-sentrux-gate-writer: dry-run serialize failed: {}", e),
        }
        return;
    }
    if let Err(e) = append_entry(ledger, entry) {
        eprintln!("ix-sentrux-gate-writer: append failed: {}", e);
    } else {
        eprintln!(
            "ix-sentrux-gate-writer: appended {} row to {}",
            entry.source,
            ledger.display()
        );
    }
}
