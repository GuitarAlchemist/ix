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

    /// Adversarial prompt corpus directory the QA run read (`*.jsonl`).
    /// Hashed into the chatbot population id.
    #[arg(long)]
    qa_corpus: PathBuf,

    /// Stub response fixtures file the QA run read. Hashed into the chatbot
    /// population id.
    #[arg(long)]
    qa_fixtures: PathBuf,

    /// Destination snapshots root containing embeddings/, voicing-analysis/,
    /// chatbot-qa/ subdirectories.
    #[arg(long)]
    out_dir: PathBuf,

    /// Optional embedding diagnostics JSON report to snapshot directly.
    #[arg(long)]
    embeddings_report: Option<PathBuf>,
}

const INSTRUMENTS: [&str; 3] = ["guitar", "bass", "ukulele"];

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

    let population = voicing_population_id(&cli.state_dir)?;
    let mut voicing = build_voicing_snapshot(&cli.state_dir)?;
    voicing["PopulationId"] = json!(population);
    let chatbot_population =
        chatbot_population_id(&cli.qa_corpus, &cli.qa_fixtures, &cli.state_dir)?;
    let chatbot = build_chatbot_snapshot(&cli.qa_results, &chatbot_population)?;
    let mut embeddings =
        build_embeddings_snapshot(&cli.state_dir, cli.embeddings_report.as_deref())?;
    // The embeddings report shares the voicing population only if its index
    // was built from these inputs: check what the report says it indexed.
    if cli.embeddings_report.is_some() {
        check_embeddings_population(&embeddings, &voicing)?;
    }
    embeddings
        .as_object_mut()
        .ok_or("embeddings report is not a JSON object")?
        .insert("population_id".into(), json!(population));

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
    for instrument in INSTRUMENTS {
        let count = count_voicings(state_dir, instrument)?;
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

    for (instrument_idx, instrument) in INSTRUMENTS.iter().enumerate() {
        let voicings = read_voicings(state_dir, instrument)?;
        let mut count = 0u64;

        for voicing in voicings {
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

fn build_chatbot_snapshot(
    qa_results: &Path,
    population_id: &str,
) -> Result<serde_json::Value, String> {
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
        "population_id": population_id,
    }))
}

/// Input file `read_voicings` uses for an instrument, relative to the state
/// dir: the raw GA dump when present, else the tracked fixture, else `None`.
fn voicing_source(state_dir: &Path, instrument: &str) -> Option<String> {
    [
        format!("raw/{instrument}.jsonl"),
        format!("{instrument}-corpus.json"),
    ]
    .into_iter()
    .find(|rel| state_dir.join(rel).exists())
}

/// Population identity of the voicing inputs:
/// `voicings:guitar=raw/guitar.jsonl,bass=raw/bass.jsonl,ukulele=raw/ukulele.jsonl`
/// (full GA dump) or `voicings:guitar=guitar-corpus.json,...` (CI fixture).
///
/// It names the source, deliberately not a content hash: the raw dump's
/// content changes exactly when the upstream producer changes, so hashing it
/// would turn a producer regression into a free rebaseline. A count or content
/// change within the same source must stay comparable and alert.
///
/// Fails closed on a partial dump. The fixtures are always tracked, so a GA
/// export that failed to write one `raw/*.jsonl` would otherwise fall back to
/// that instrument's fixture, yield a new mixed id, and reset the history of
/// what is really a producer failure (ix#342 re-review).
fn voicing_population_id(state_dir: &Path) -> Result<String, String> {
    let mut sources = Vec::new();
    for instrument in INSTRUMENTS {
        let source = voicing_source(state_dir, instrument).ok_or_else(|| {
            format!(
                "no voicing input for '{instrument}' under {state_dir:?} (looked for \
                 raw/{instrument}.jsonl and {instrument}-corpus.json); refusing to write \
                 snapshots for a partial corpus"
            )
        })?;
        sources.push(format!("{instrument}={source}"));
    }
    let raw = sources.iter().filter(|s| s.contains("=raw/")).count();
    if raw != 0 && raw != INSTRUMENTS.len() {
        return Err(format!(
            "mixed voicing sources ({}): a partial raw dump would be measured as a \
             different population; restore the missing raw/*.jsonl or remove all of them",
            sources.join(", ")
        ));
    }
    Ok(format!("voicings:{}", sources.join(",")))
}

/// Fails unless the embeddings report indexed exactly the voicings read for
/// the voicing snapshot, instrument by instrument (`corpus.instruments`).
fn check_embeddings_population(
    report: &serde_json::Value,
    voicing: &serde_json::Value,
) -> Result<(), String> {
    for (instrument, key) in INSTRUMENTS.iter().zip(["Guitar", "Bass", "Ukulele"]) {
        let indexed = report
            .pointer(&format!("/corpus/instruments/{instrument}"))
            .and_then(serde_json::Value::as_u64);
        let read = voicing["Corpus"][key].as_u64();
        if indexed.is_none() || indexed != read {
            return Err(format!(
                "embeddings report corpus.instruments.{instrument} is {indexed:?} but the \
                 voicing inputs hold {read:?}: the report was not built from this population"
            ));
        }
    }
    Ok(())
}

/// Population identity of the deterministic chatbot QA harness:
/// `chatbot-qa:deterministic-fixture-ci@<hash>`, where the hash covers every
/// input the grading reads from the repo: the prompt corpus (`*.jsonl`), the
/// stub responses, and the voicing fixtures used for grounding.
///
/// These are reviewed, repo-controlled fixtures, so a change to them is a
/// declared harness change (a new category, new stubs) and rebaselines. A QA
/// run that crashes part-way reads the same fixtures and keeps the id, so its
/// shorter findings file still compares (and trips the corpus-shrink alert).
/// Line endings are normalized so a Windows checkout hashes like CI.
fn chatbot_population_id(
    qa_corpus: &Path,
    qa_fixtures: &Path,
    state_dir: &Path,
) -> Result<String, String> {
    let mut prompt_files: Vec<PathBuf> = fs::read_dir(qa_corpus)
        .map_err(|e| format!("read {qa_corpus:?}: {e}"))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
        .collect();
    if prompt_files.is_empty() {
        return Err(format!("no *.jsonl prompt files in {qa_corpus:?}"));
    }
    prompt_files.sort();

    let mut inputs: Vec<(String, PathBuf)> = prompt_files
        .into_iter()
        .map(|path| {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            (format!("corpus/{name}"), path.clone())
        })
        .collect();
    inputs.push(("fixtures".to_string(), qa_fixtures.to_path_buf()));
    for instrument in INSTRUMENTS {
        let name = format!("{instrument}-corpus.json");
        inputs.push((format!("voicings/{name}"), state_dir.join(name)));
    }

    let mut hasher = blake3::Hasher::new();
    for (label, path) in inputs {
        let bytes = fs::read(&path).map_err(|e| format!("read {path:?}: {e}"))?;
        let text = String::from_utf8_lossy(&bytes).replace("\r\n", "\n");
        hasher.update(label.as_bytes());
        hasher.update(&[0]);
        hasher.update(&(text.len() as u64).to_le_bytes());
        hasher.update(text.as_bytes());
    }
    let hash = hasher.finalize().to_hex();
    Ok(format!(
        "chatbot-qa:deterministic-fixture-ci@{}",
        &hash[..16]
    ))
}

/// Read voicings for an instrument from raw/{name}.jsonl if present, falling
/// back to {name}-corpus.json (JSON array of the same RawVoicing shape).
/// Missing instruments return an empty Vec — partial coverage > total failure.
fn read_voicings(state_dir: &Path, instrument: &str) -> Result<Vec<RawVoicing>, String> {
    let raw_path = state_dir.join("raw").join(format!("{instrument}.jsonl"));
    let corpus_path = state_dir.join(format!("{instrument}-corpus.json"));

    if raw_path.exists() {
        let file = File::open(&raw_path).map_err(|e| format!("open {:?}: {e}", raw_path))?;
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| format!("read {:?}: {e}", raw_path))?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(
                serde_json::from_str(&line).map_err(|e| format!("parse {:?}: {e}", raw_path))?,
            );
        }
        Ok(out)
    } else if corpus_path.exists() {
        let bytes = fs::read(&corpus_path).map_err(|e| format!("read {:?}: {e}", corpus_path))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {:?}: {e}", corpus_path))
    } else {
        eprintln!(
            "ix-quality-trend-bootstrap: no input for '{instrument}' \
             (looked for {raw_path:?} and {corpus_path:?}), skipping"
        );
        Ok(Vec::new())
    }
}

/// Count voicings for an instrument via `read_voicings`. Cheap because the
/// fallback parses the whole file anyway; consistency with `read_voicings`
/// is more valuable than the avoided allocation here.
fn count_voicings(state_dir: &Path, instrument: &str) -> Result<usize, String> {
    Ok(read_voicings(state_dir, instrument)?.len())
}

#[allow(dead_code)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_state_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("raw")).unwrap();
        for instrument in INSTRUMENTS {
            fs::write(dir.path().join(format!("{instrument}-corpus.json")), "[]").unwrap();
        }
        dir
    }

    #[test]
    fn population_id_names_one_source_for_all_instruments() {
        let dir = fixture_state_dir();
        assert_eq!(
            voicing_population_id(dir.path()).unwrap(),
            "voicings:guitar=guitar-corpus.json,bass=bass-corpus.json,ukulele=ukulele-corpus.json"
        );

        // A full raw dump wins over the fixtures, exactly as read_voicings does.
        for instrument in INSTRUMENTS {
            fs::write(dir.path().join(format!("raw/{instrument}.jsonl")), "").unwrap();
        }
        assert_eq!(
            voicing_population_id(dir.path()).unwrap(),
            "voicings:guitar=raw/guitar.jsonl,bass=raw/bass.jsonl,ukulele=raw/ukulele.jsonl"
        );
    }

    #[test]
    fn partial_raw_dump_refuses_to_produce_a_population() {
        // Re-review P1: one missing raw file must not become a new, mixed id.
        let dir = fixture_state_dir();
        fs::write(dir.path().join("raw/guitar.jsonl"), "").unwrap();
        fs::write(dir.path().join("raw/bass.jsonl"), "").unwrap();
        let err = voicing_population_id(dir.path()).unwrap_err();
        assert!(err.contains("mixed voicing sources"), "{err}");

        // Nor may an instrument with no input at all be silently skipped.
        let dir = fixture_state_dir();
        fs::remove_file(dir.path().join("ukulele-corpus.json")).unwrap();
        let err = voicing_population_id(dir.path()).unwrap_err();
        assert!(err.contains("no voicing input for 'ukulele'"), "{err}");
    }

    #[test]
    fn embeddings_report_must_index_the_voicing_population() {
        let voicing = json!({"Corpus": {"Guitar": 500, "Bass": 12614, "Ukulele": 8612}});
        let matching =
            json!({"corpus": {"instruments": {"guitar": 500, "bass": 12614, "ukulele": 8612}}});
        assert!(check_embeddings_population(&matching, &voicing).is_ok());

        // A report over another index (e.g. GA's full optick.index).
        let other =
            json!({"corpus": {"instruments": {"guitar": 667125, "bass": 12614, "ukulele": 8612}}});
        let err = check_embeddings_population(&other, &voicing).unwrap_err();
        assert!(err.contains("guitar"), "{err}");

        let unlabelled = json!({"corpus": {"count": 21726}});
        assert!(check_embeddings_population(&unlabelled, &voicing).is_err());
    }

    #[test]
    fn chatbot_population_id_tracks_the_harness_inputs() {
        let state = fixture_state_dir();
        let qa = tempfile::tempdir().unwrap();
        let corpus = qa.path().join("corpus");
        fs::create_dir_all(&corpus).unwrap();
        fs::write(
            corpus.join("grounding.jsonl"),
            "{\"id\":\"grounding-001\"}\n",
        )
        .unwrap();
        fs::write(corpus.join("README.md"), "not hashed").unwrap();
        let stubs = qa.path().join("stub-responses.jsonl");
        fs::write(&stubs, "{}\n").unwrap();

        let id = |_: &str| chatbot_population_id(&corpus, &stubs, state.path()).unwrap();
        let base = id("base");
        assert!(
            base.starts_with("chatbot-qa:deterministic-fixture-ci@"),
            "{base}"
        );

        // CRLF checkouts hash like LF ones; non-prompt files are ignored.
        fs::write(
            corpus.join("grounding.jsonl"),
            "{\"id\":\"grounding-001\"}\r\n",
        )
        .unwrap();
        fs::write(corpus.join("README.md"), "edited").unwrap();
        assert_eq!(id("crlf"), base);

        // A new prompt category, new stubs, or new grounding fixtures are a
        // declared harness change.
        fs::write(corpus.join("algebra.jsonl"), "{\"id\":\"algebra-001\"}\n").unwrap();
        let with_algebra = id("algebra");
        assert_ne!(with_algebra, base);
        fs::write(&stubs, "{\"prompt_prefix\":\"x\"}\n").unwrap();
        let with_stubs = id("stubs");
        assert_ne!(with_stubs, with_algebra);
        fs::write(state.path().join("bass-corpus.json"), "[{}]").unwrap();
        assert_ne!(id("voicings"), with_stubs);

        let empty = tempfile::tempdir().unwrap();
        assert!(chatbot_population_id(empty.path(), &stubs, state.path()).is_err());
    }
}
