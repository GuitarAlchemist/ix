//! `ix-sentrux-annotations` — bridge sentrux structural findings into the
//! `ai-annotation-v1` JSONL stream consumed by the reconciler.
//!
//! Usage:
//!
//! ```text
//! ix-sentrux-annotations --workspace . --sidecar
//! ix-sentrux-annotations --workspace . --inline --dry-run
//! ix-sentrux-annotations --workspace . --from-fixture report.json
//! ```

use clap::Parser;
use ix_sentrux_annotations::{
    convert::violation_to_annotation,
    emit_inline, emit_sidecar,
    mcp_bridge::{run_sentrux_check, SentruxConfig},
    rules_response::RulesReport,
};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "ix-sentrux-annotations",
    about = "Bridge sentrux structural-rule findings into ai-annotation-v1 JSONL"
)]
struct Args {
    /// Workspace root to scan. Passed to sentrux and used to resolve
    /// repo-relative paths in violations.
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    /// Emit mode (default: sidecar). Mutually exclusive with `--inline`.
    #[arg(long, default_value_t = false)]
    sidecar: bool,

    /// Emit annotations as in-source `// @ai:smell …` comments above the
    /// violating line. Skips files that already carry the same sentrux
    /// annotation at that location.
    #[arg(long, default_value_t = false, conflicts_with = "sidecar")]
    inline: bool,

    /// Dry run: do everything except actually mutate files / write JSONL.
    /// In sidecar mode this also skips the write; the summary still
    /// reports what would have been written.
    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Override sentrux executable path. Default:
    /// `C:/Users/spare/bin/sentrux.exe` (or `$SENTRUX_EXE` if set).
    #[arg(long)]
    sentrux_exe: Option<PathBuf>,

    /// Override the JSONL output path. Default:
    /// `<workspace>/state/quality/ai-annotations-sentrux.jsonl`.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Read a captured `check_rules` payload from a JSON file instead of
    /// spawning sentrux. The file may either be the `RulesReport` itself
    /// (`{ "violations": [...] }`) or the full MCP tool-call envelope
    /// (`{ "result": { "content": [{ "text": "..." }] } }`).
    ///
    /// Useful for CI where sentrux is captured by a separate step, or for
    /// reproducible tests.
    #[arg(long)]
    from_fixture: Option<PathBuf>,

    /// Sentrux call timeout in seconds.
    #[arg(long, default_value_t = 60)]
    timeout_secs: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let report = if let Some(fixture) = &args.from_fixture {
        load_fixture(fixture)?
    } else {
        let sentrux_exe = args
            .sentrux_exe
            .clone()
            .or_else(|| std::env::var_os("SENTRUX_EXE").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from(ix_sentrux_annotations::DEFAULT_SENTRUX_EXE));
        let cfg = SentruxConfig {
            sentrux_exe,
            workspace: args.workspace.clone(),
            timeout: Duration::from_secs(args.timeout_secs),
        };
        run_sentrux_check(&cfg)?
    };

    let now = chrono::Utc::now().to_rfc3339();
    let mut annotations = Vec::new();
    for v in &report.violations {
        annotations.extend(violation_to_annotation(&args.workspace, v, &now));
    }

    // Choose mode. Default is sidecar when neither flag is set.
    let inline_mode = args.inline;
    let outcome = if inline_mode {
        emit_inline(&args.workspace, &annotations, args.dry_run)?
    } else if args.dry_run {
        // sidecar + dry-run: report counts without writing.
        ix_sentrux_annotations::EmitOutcome {
            written: annotations.len(),
            skipped: 0,
        }
    } else {
        emit_sidecar(&args.workspace, &annotations, args.out.as_deref())?
    };

    let mode = if inline_mode { "inline" } else { "sidecar" };
    let dry = if args.dry_run { " (dry-run)" } else { "" };
    eprintln!(
        "ix-sentrux-annotations: {} violation(s) -> {} annotation(s); mode={mode} written={written} skipped={skipped}{dry}",
        report.violations.len(),
        annotations.len(),
        written = outcome.written,
        skipped = outcome.skipped,
    );

    Ok(())
}

fn load_fixture(path: &PathBuf) -> Result<RulesReport, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    // Try the MCP envelope shape first.
    if v.get("result").and_then(|r| r.get("content")).is_some() {
        return ix_sentrux_annotations::rules_response::parse_check_rules_response(&v)
            .map_err(Into::into);
    }
    // Otherwise treat the file as a bare RulesReport.
    let report: RulesReport = serde_json::from_value(v)?;
    Ok(report)
}
