use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::{Duration, NaiveDate, Utc};
use clap::Parser;
use serde_json::{json, Value};

const CATEGORIES: [&str; 3] = ["embeddings", "voicing-analysis", "chatbot-qa"];

#[derive(Debug, Parser)]
#[command(
    name = "ix-quality-trend-seed-history",
    about = "Backfill a contiguous baseline history by cloning the latest snapshots into prior dates"
)]
struct Cli {
    /// Root snapshots directory containing embeddings/, voicing-analysis/,
    /// chatbot-qa/ subdirectories.
    #[arg(long)]
    snapshots_dir: PathBuf,

    /// Number of prior days to create before the anchor snapshot date.
    #[arg(long, default_value_t = 13)]
    days: usize,

    /// Optional anchor date. When omitted, each category uses its latest
    /// existing snapshot as the source and anchor.
    #[arg(long)]
    anchor_date: Option<NaiveDate>,

    /// Replace existing dated files in the seeded window.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(&cli) {
        Ok(created) => {
            eprintln!(
                "ix-quality-trend-seed-history: wrote {created} snapshot(s) under {}",
                cli.snapshots_dir.display()
            );
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("ix-quality-trend-seed-history: {err}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: &Cli) -> Result<usize, String> {
    let mut created = 0usize;

    for category in CATEGORIES {
        let dir = cli.snapshots_dir.join(category);
        let (source_date, source_path) = latest_snapshot_path(&dir, cli.anchor_date)?;
        let source_bytes =
            fs::read(&source_path).map_err(|e| format!("read {:?}: {e}", source_path.display()))?;
        let source_json: Value = serde_json::from_slice(&source_bytes)
            .map_err(|e| format!("parse {:?}: {e}", source_path.display()))?;

        for offset in 1..=cli.days {
            let seeded_date = source_date - Duration::days(offset as i64);
            let out_path = dir.join(format!("{seeded_date}.json"));
            if out_path.exists() && !cli.overwrite {
                continue;
            }

            let seeded = seed_snapshot(&source_json, source_date, seeded_date, &source_path);
            let bytes = serde_json::to_vec_pretty(&seeded)
                .map_err(|e| format!("serialize {:?}: {e}", out_path.display()))?;
            fs::write(&out_path, bytes)
                .map_err(|e| format!("write {:?}: {e}", out_path.display()))?;
            created += 1;
        }
    }

    Ok(created)
}

fn latest_snapshot_path(
    dir: &Path,
    anchor_date: Option<NaiveDate>,
) -> Result<(NaiveDate, PathBuf), String> {
    let read_dir = fs::read_dir(dir).map_err(|e| format!("read_dir {:?}: {e}", dir.display()))?;
    let mut latest: Option<(NaiveDate, PathBuf)> = None;

    for entry in read_dir {
        let entry = entry.map_err(|e| format!("iterate {:?}: {e}", dir.display()))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(date) = NaiveDate::parse_from_str(stem, "%Y-%m-%d") else {
            continue;
        };
        if let Some(anchor) = anchor_date {
            if date != anchor {
                continue;
            }
        }

        if latest.as_ref().map(|(d, _)| date > *d).unwrap_or(true) {
            latest = Some((date, path));
        }
    }

    latest.ok_or_else(|| {
        if let Some(anchor) = anchor_date {
            format!(
                "no dated snapshot found in {} for anchor {}",
                dir.display(),
                anchor
            )
        } else {
            format!("no dated snapshots found in {}", dir.display())
        }
    })
}

fn seed_snapshot(
    source: &Value,
    source_date: NaiveDate,
    seeded_date: NaiveDate,
    source_path: &Path,
) -> Value {
    let mut seeded = source.clone();
    let timestamp = format!("{seeded_date}T00:00:00Z");

    if let Some(obj) = seeded.as_object_mut() {
        if obj.contains_key("timestamp") {
            obj.insert("timestamp".into(), Value::String(timestamp.clone()));
        }
        if obj.contains_key("Timestamp") {
            obj.insert("Timestamp".into(), Value::String(timestamp));
        }
        obj.insert(
            "_seeded_baseline".into(),
            json!({
                "source_date": source_date,
                "seeded_date": seeded_date,
                "source_file": source_path.file_name().and_then(|s| s.to_str()).unwrap_or_default(),
                "seeded_on": Utc::now().date_naive(),
            }),
        );
    }

    seeded
}
