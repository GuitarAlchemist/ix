//! `ix-ai-annotations-reconcile` CLI — read the extractor JSONL, run
//! the reconciler, and write a structured report.

use clap::Parser;
use ix_ai_annotations::{reconcile, Annotation, ReconcilerConfig};
use std::fs::{create_dir_all, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "ix-ai-annotations-reconcile",
    about = "Reconcile extracted AI annotations against tests and contradictions"
)]
struct Args {
    /// Workspace root (used to resolve relative paths).
    #[arg(long, default_value = ".")]
    workspace: PathBuf,

    /// Input JSONL (output of the extractor).
    /// Default: <workspace>/state/quality/ai-annotations.jsonl
    #[arg(long)]
    input: Option<PathBuf>,

    /// Output JSON report.
    /// Default: <workspace>/state/quality/ai-annotations-reconciliation.json
    #[arg(long)]
    out: Option<PathBuf>,

    /// Stale threshold in days. Default 7.
    #[arg(long, default_value_t = 7)]
    stale_days: i64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let input = args
        .input
        .unwrap_or_else(|| args.workspace.join("state/quality/ai-annotations.jsonl"));
    let out = args
        .out
        .unwrap_or_else(|| args.workspace.join("state/quality/ai-annotations-reconciliation.json"));

    let annotations = load_jsonl(&input)?;
    let test_files = discover_test_files(&args.workspace)?;

    let cfg = ReconcilerConfig::new(&args.workspace)
        .with_test_files(test_files);
    let cfg = ReconcilerConfig {
        stale_threshold_days: args.stale_days,
        ..cfg
    };

    let report = reconcile(annotations, &cfg);

    if let Some(parent) = out.parent() {
        create_dir_all(parent)?;
    }
    let mut f = BufWriter::new(File::create(&out)?);
    serde_json::to_writer_pretty(&mut f, &report)?;
    f.flush()?;

    eprintln!(
        "ix-ai-annotations-reconcile: {} annotations, {} verified-by-test, {} stale, {} contradictory -> {}",
        report.total_annotations,
        report.verified_by_test,
        report.stale,
        report.contradictory,
        out.display()
    );
    Ok(())
}

fn load_jsonl(path: &Path) -> std::io::Result<Vec<Annotation>> {
    let f = File::open(path)?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Annotation>(&line) {
            Ok(a) => out.push(a),
            Err(e) => eprintln!("skip malformed line: {}", e),
        }
    }
    Ok(out)
}

fn discover_test_files(workspace: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut paths: Vec<PathBuf> = Vec::new();
    use ignore::WalkBuilder;
    let mut wb = WalkBuilder::new(workspace);
    wb.filter_entry(|e| {
        let n = e.file_name().to_string_lossy();
        !matches!(
            n.as_ref(),
            "target" | "node_modules" | ".git" | "dist" | "build"
        )
    });
    for ent in wb.build().flatten() {
        if !ent.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let p = ent.path();
        let rel = p.strip_prefix(workspace).unwrap_or(p).to_path_buf();
        let s = rel.to_string_lossy().to_lowercase().replace('\\', "/");
        let stem = p
            .file_stem()
            .map(|st| st.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        // Heuristics: under tests/, ends in _test.<ext>, or test_*.py / *.spec.ts.
        let looks_like_test = s.contains("/tests/")
            || s.starts_with("tests/")
            || stem.ends_with("_test")
            || stem.ends_with("_tests")
            || stem.starts_with("test_")
            || stem.ends_with(".test")
            || stem.ends_with(".spec");
        if looks_like_test {
            paths.push(rel);
        }
    }
    Ok(paths)
}
