use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use chrono::Utc;
use clap::Parser;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Parser)]
#[command(
    name = "ix-quality-trend-bootstrap",
    about = "Generate CI-friendly quality snapshots from checked-in GuitarAlchemist state"
)]
struct Cli {
    /// Root of the checked-in GA voicing state directory.
    #[arg(long)]
    state_dir: PathBuf,

    /// Deterministic ga-chatbot QA findings JSONL file.
    #[arg(long)]
    qa_results: PathBuf,

    /// Destination snapshots root containing embeddings/, voicing-analysis/,
    /// chatbot-qa/ subdirectories.
    #[arg(long)]
    out_dir: PathBuf,

    /// Optional embedding diagnostics JSON report to snapshot directly.
    #[arg(long)]
    embeddings_report: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct RawVoicing {
    #[serde(default)]
    frets: Vec<String>,
    #[serde(default, rename = "midiNotes")]
    midi_notes: Vec<i32>,
}

#[derive(Debug, Deserialize)]
struct QaResultLine {
    #[serde(default, rename = "prompt_id")]
    prompt_id: String,
    #[serde(default, rename = "deterministic_verdict")]
    deterministic_verdict: Option<char>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let date = Utc::now().date_naive();

    if let Err(err) = run(&cli, date) {
        eprintln!("ix-quality-trend-bootstrap: {err}");
        return ExitCode::from(1);
    }

    eprintln!(
        "ix-quality-trend-bootstrap: wrote snapshots for {} under {}",
        date,
        cli.out_dir.display()
    );
    ExitCode::SUCCESS
}

fn run(cli: &Cli, date: chrono::NaiveDate) -> Result<(), String> {
    fs::create_dir_all(&cli.out_dir).map_err(|e| format!("create {:?}: {e}", cli.out_dir))?;

    let voicing = build_voicing_snapshot(&cli.state_dir)?;
    let chatbot = build_chatbot_snapshot(&cli.qa_results)?;
    let embeddings = build_embeddings_snapshot(&cli.state_dir, cli.embeddings_report.as_deref())?;

    write_snapshot(
        &cli.out_dir.join("voicing-analysis"),
        date,
        &serde_json::to_vec_pretty(&voicing).map_err(|e| format!("serialize voicing: {e}"))?,
    )?;
    write_snapshot(
        &cli.out_dir.join("chatbot-qa"),
        date,
        &serde_json::to_vec_pretty(&chatbot).map_err(|e| format!("serialize chatbot: {e}"))?,
    )?;
    write_snapshot(
        &cli.out_dir.join("embeddings"),
        date,
        &serde_json::to_vec_pretty(&embeddings)
            .map_err(|e| format!("serialize embeddings: {e}"))?,
    )?;

    Ok(())
}

fn write_snapshot(dir: &Path, date: chrono::NaiveDate, bytes: &[u8]) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| format!("create {:?}: {e}", dir))?;
    let path = dir.join(format!("{date}.json"));
    fs::write(&path, bytes).map_err(|e| format!("write {:?}: {e}", path))
}

fn build_embeddings_snapshot(
    state_dir: &Path,
    embeddings_report: Option<&Path>,
) -> Result<serde_json::Value, String> {
    if let Some(path) = embeddings_report {
        let bytes = fs::read(path).map_err(|e| format!("read {:?}: {e}", path))?;
        return serde_json::from_slice(&bytes).map_err(|e| format!("parse {:?}: {e}", path));
    }

    let mut total = 0u64;
    for instrument in ["guitar", "bass", "ukulele"] {
        let count = count_jsonl_lines(&state_dir.join("raw").join(format!("{instrument}.jsonl")))?;
        total += count as u64;
    }

    Ok(json!({
        "timestamp": Utc::now().to_rfc3339(),
        "corpus": {
            "count": total,
        },
        "notes": [
            "CI bootstrap snapshot derived from checked-in corpus state.",
            "OPTIC-K index-dependent leak and retrieval metrics are unavailable in GitHub-hosted CI because optick.index is not present in the repository."
        ]
    }))
}

fn build_voicing_snapshot(state_dir: &Path) -> Result<serde_json::Value, String> {
    let mut corpus_counts: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut cardinality_distribution: BTreeMap<String, u64> = BTreeMap::new();
    let mut pcset_mask: BTreeMap<String, u8> = BTreeMap::new();
    let mut midi_notes_mismatch = 0u64;
    let mut null_pitch_class_set = 0u64;
    let mut negative_physical_layout = 0u64;
    let mut interval_spread_invariant = 0u64;

    for (instrument_idx, instrument) in ["guitar", "bass", "ukulele"].iter().enumerate() {
        let path = state_dir.join("raw").join(format!("{instrument}.jsonl"));
        let file = File::open(&path).map_err(|e| format!("open {:?}: {e}", path))?;
        let reader = BufReader::new(file);
        let mut count = 0u64;

        for line in reader.lines() {
            let line = line.map_err(|e| format!("read {:?}: {e}", path))?;
            if line.trim().is_empty() {
                continue;
            }
            let voicing: RawVoicing =
                serde_json::from_str(&line).map_err(|e| format!("parse {:?}: {e}", path))?;
            count += 1;

            let played_strings = voicing.frets.iter().filter(|f| f.as_str() != "x").count();
            if played_strings != voicing.midi_notes.len() {
                midi_notes_mismatch += 1;
            }

            let mut pcs = BTreeSet::new();
            for midi in &voicing.midi_notes {
                pcs.insert(midi.rem_euclid(12));
            }
            if pcs.is_empty() {
                null_pitch_class_set += 1;
            } else {
                *cardinality_distribution
                    .entry(pcs.len().to_string())
                    .or_insert(0) += 1;
                let key = pcs
                    .iter()
                    .map(|pc| pc.to_string())
                    .collect::<Vec<_>>()
                    .join("-");
                let mask = pcset_mask.entry(key).or_insert(0);
                *mask |= 1 << instrument_idx;
            }

            if voicing
                .frets
                .iter()
                .any(|f| f.parse::<i32>().map(|n| n < 0).unwrap_or(false))
            {
                negative_physical_layout += 1;
            }

            let numeric_frets: Vec<i32> = voicing
                .frets
                .iter()
                .filter_map(|f| f.parse::<i32>().ok())
                .collect();
            if let (Some(min), Some(max)) = (numeric_frets.iter().min(), numeric_frets.iter().max())
            {
                if max - min > 4 {
                    interval_spread_invariant += 1;
                }
            }
        }

        corpus_counts.insert(*instrument, count);
    }

    let total = corpus_counts.values().sum::<u64>();
    let shared_sets = pcset_mask
        .values()
        .filter(|mask| mask.count_ones() >= 2)
        .count() as u64;
    let consistent = pcset_mask.values().filter(|mask| **mask == 0b111).count() as u64;
    let consistency_pct = if shared_sets > 0 {
        consistent as f64 / shared_sets as f64 * 100.0
    } else {
        0.0
    };

    Ok(json!({
        "Timestamp": Utc::now().to_rfc3339(),
        "Corpus": {
            "Guitar": corpus_counts.get("guitar").copied(),
            "Bass": corpus_counts.get("bass").copied(),
            "Ukulele": corpus_counts.get("ukulele").copied(),
            "Total": total,
        },
        "CrossInstrumentConsistency": {
            "SharedSets": shared_sets,
            "Consistent": consistent,
            "Pct": consistency_pct,
        },
        "CardinalityDistribution": cardinality_distribution,
        "InvariantFailures": {
            "MidiNotesMismatch": midi_notes_mismatch,
            "NullPitchClassSet": null_pitch_class_set,
            "NegativePhysicalLayout": negative_physical_layout,
            "IntervalSpreadInvariant": interval_spread_invariant,
        },
        "Performance": {
            "RuntimeSeconds": serde_json::Value::Null,
            "VoicingsPerSec": serde_json::Value::Null,
        }
    }))
}

fn build_chatbot_snapshot(qa_results: &Path) -> Result<serde_json::Value, String> {
    let file = File::open(qa_results).map_err(|e| format!("open {:?}: {e}", qa_results))?;
    let reader = BufReader::new(file);

    let mut total = 0u64;
    let mut pass = 0u64;
    let mut category_totals: BTreeMap<String, (u64, u64)> = BTreeMap::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("read {:?}: {e}", qa_results))?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(result) = serde_json::from_str::<QaResultLine>(&line) else {
            continue;
        };
        if result.prompt_id.is_empty() {
            continue;
        }

        total += 1;
        let passed = matches!(result.deterministic_verdict, Some('T') | Some('P'));
        if passed {
            pass += 1;
        }

        let category = result
            .prompt_id
            .split_once('-')
            .map(|(prefix, _)| prefix.to_string())
            .unwrap_or_else(|| "uncategorized".to_string());
        let entry = category_totals.entry(category).or_insert((0, 0));
        entry.0 += 1;
        if passed {
            entry.1 += 1;
        }
    }

    let by_category: BTreeMap<String, serde_json::Value> = category_totals
        .into_iter()
        .map(|(category, (count, pass_count))| {
            let pass_pct = if count > 0 {
                pass_count as f64 / count as f64 * 100.0
            } else {
                0.0
            };
            (
                category,
                json!({
                    "pass_pct": pass_pct,
                    "total": count,
                }),
            )
        })
        .collect();

    let pass_pct = if total > 0 {
        pass as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    Ok(json!({
        "timestamp": Utc::now().to_rfc3339(),
        "total_prompts": total,
        "pass_pct": pass_pct,
        "avg_response_ms": serde_json::Value::Null,
        "by_category": by_category,
        "mode": "deterministic-fixture-ci",
    }))
}

fn count_jsonl_lines(path: &Path) -> Result<usize, String> {
    let file = File::open(path).map_err(|e| format!("open {:?}: {e}", path))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read {:?}: {e}", path))?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}
