use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser;
use ix_optick::compute_schema_hash;
use serde::{Deserialize, Serialize};

const DIM: usize = 124;
const INSTRUMENTS: [&str; 3] = ["guitar", "bass", "ukulele"];

#[derive(Debug, Parser)]
#[command(
    name = "ix-quality-trend-build-ci-index",
    about = "Build a CI-safe reduced OPTIC-K index from checked-in GuitarAlchemist JSONL state"
)]
struct Cli {
    /// Root of the checked-in GA voicing state directory.
    #[arg(long)]
    state_dir: PathBuf,

    /// Destination OPTK file path.
    #[arg(long)]
    out: PathBuf,
}

#[derive(Debug, Deserialize)]
struct RawVoicing {
    instrument: String,
    #[serde(rename = "stringCount")]
    _string_count: usize,
    diagram: String,
    #[serde(default)]
    frets: Vec<String>,
    #[serde(default, rename = "midiNotes")]
    midi_notes: Vec<i32>,
    #[serde(default, rename = "minFret")]
    min_fret: Option<i32>,
    #[serde(default, rename = "maxFret")]
    max_fret: Option<i32>,
    #[serde(default, rename = "fretSpan")]
    fret_span: Option<i32>,
}

#[derive(Debug, Serialize)]
struct VoicingMetadata {
    diagram: String,
    instrument: String,
    #[serde(rename = "midiNotes")]
    midi_notes: Vec<i32>,
    quality_inferred: Option<String>,
}

struct IndexedVoicing {
    vector: [f32; DIM],
    metadata: VoicingMetadata,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Err(err) = run(&cli) {
        eprintln!("ix-quality-trend-build-ci-index: {err}");
        return ExitCode::from(1);
    }
    eprintln!(
        "ix-quality-trend-build-ci-index: wrote {}",
        cli.out.display()
    );
    ExitCode::SUCCESS
}

fn run(cli: &Cli) -> Result<(), String> {
    let mut by_instrument: [Vec<IndexedVoicing>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for (instrument_idx, instrument) in INSTRUMENTS.iter().enumerate() {
        let raw_path = cli
            .state_dir
            .join("raw")
            .join(format!("{instrument}.jsonl"));
        let corpus_path = cli.state_dir.join(format!("{instrument}-corpus.json"));

        let voicings = if raw_path.exists() {
            read_jsonl(&raw_path)?
        } else if corpus_path.exists() {
            read_corpus_json(&corpus_path)?
        } else {
            // Production has raw dumps; CI has only corpus.json fixtures;
            // a fresh clone with neither just skips the instrument so the
            // index still builds (partial coverage > total failure).
            eprintln!(
                "ix-quality-trend-build-ci-index: no input for '{instrument}' — \
                 looked for {raw_path:?} and {corpus_path:?}, skipping"
            );
            continue;
        };

        for raw in voicings {
            let quality_inferred = infer_quality_label(&raw);
            by_instrument[instrument_idx].push(IndexedVoicing {
                vector: build_vector(&raw),
                metadata: VoicingMetadata {
                    diagram: raw.diagram,
                    instrument: raw.instrument,
                    midi_notes: raw.midi_notes,
                    quality_inferred,
                },
            });
        }
    }

    write_index(&cli.out, &by_instrument)
}

/// Read a `raw/{instrument}.jsonl` file (one RawVoicing per line).
fn read_jsonl(path: &Path) -> Result<Vec<RawVoicing>, String> {
    let file = File::open(path).map_err(|e| format!("open {:?}: {e}", path))?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for line in reader.lines() {
        let line = line.map_err(|e| format!("read {:?}: {e}", path))?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).map_err(|e| format!("parse {:?}: {e}", path))?);
    }
    Ok(out)
}

/// Read a `{instrument}-corpus.json` file (JSON array of RawVoicing).
fn read_corpus_json(path: &Path) -> Result<Vec<RawVoicing>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("open {:?}: {e}", path))?;
    serde_json::from_str(&content).map_err(|e| format!("parse {:?}: {e}", path))
}

fn build_vector(raw: &RawVoicing) -> [f32; DIM] {
    let mut vector = [0.0f32; DIM];

    let pcs = pitch_class_presence(&raw.midi_notes);
    let root_pc = raw
        .midi_notes
        .iter()
        .min()
        .map(|m| m.rem_euclid(12) as usize)
        .unwrap_or(0);
    let played_strings = raw.frets.iter().filter(|f| f.as_str() != "x").count();
    let min_midi = raw.midi_notes.iter().min().copied().unwrap_or(0);
    let max_midi = raw.midi_notes.iter().max().copied().unwrap_or(min_midi);
    let range = (max_midi - min_midi).max(0) as usize;
    let min_fret = raw.min_fret.unwrap_or(0).max(0) as usize;
    let max_fret = raw.max_fret.unwrap_or(min_fret as i32).max(0) as usize;
    let fret_span = raw
        .fret_span
        .unwrap_or((max_fret as i32 - min_fret as i32).max(0)) as usize;

    // STRUCTURE 0..24
    let pc_count = pcs.iter().filter(|&&v| v > 0.0).count().max(1) as f32;
    for i in 0..12 {
        let exact_membership = if pcs[i] > 0.0 { 2.0 } else { -2.0 };
        vector[i] = exact_membership;
        vector[12 + i] = exact_membership;
    }

    // MORPHOLOGY 24..48
    for (i, fret) in raw.frets.iter().take(12).enumerate() {
        if fret != "x" {
            vector[24 + i] = 1.0;
        }
    }
    vector[36 + fret_span.min(5)] = 1.0;
    vector[42 + played_strings.saturating_sub(1).min(5)] = 1.0;

    // CONTEXT 48..60
    vector[48 + raw.midi_notes.len().saturating_sub(1).min(5)] = 1.0;
    vector[54 + range.min(5)] = 1.0;

    // SYMBOLIC 60..72
    let quality_idx = infer_quality_index(&pcs, root_pc);
    vector[60 + quality_idx] = 1.0;

    // MODAL 72..112
    for i in 0..12 {
        vector[72 + i] = pcs[i] / pc_count;
        vector[84 + i] = pcs[(12 + i - root_pc) % 12] / pc_count;
    }
    vector[96 + (range / 2).min(7)] = 1.0;
    vector[104 + (min_fret / 2).min(7)] = 1.0;

    // ROOT 112..124
    vector[112 + root_pc] = 1.0;

    normalize(&mut vector);
    vector
}

fn pitch_class_presence(midi_notes: &[i32]) -> [f32; 12] {
    let mut pcs = [0.0f32; 12];
    for &midi in midi_notes {
        pcs[midi.rem_euclid(12) as usize] = 1.0;
    }
    pcs
}

fn infer_quality_label(raw: &RawVoicing) -> Option<String> {
    let pcs = pitch_class_presence(&raw.midi_notes);
    let root_pc = raw
        .midi_notes
        .iter()
        .min()
        .map(|m| m.rem_euclid(12) as usize)
        .unwrap_or(0);
    let label = match infer_quality_index(&pcs, root_pc) {
        0 => "dyad",
        1 => "maj",
        2 => "min",
        3 => "dim",
        4 => "aug",
        5 => "sus",
        6 => "maj7",
        7 => "dom7",
        8 => "min7",
        9 => "min7b5",
        10 => "dim7",
        _ => "other",
    };
    Some(label.to_string())
}

fn infer_quality_index(pcs: &[f32; 12], root_pc: usize) -> usize {
    let rel = |interval: usize| pcs[(root_pc + interval) % 12] > 0.0;
    let note_count = pcs.iter().filter(|&&v| v > 0.0).count();
    if note_count <= 2 {
        return 0;
    }
    if rel(0) && rel(4) && rel(7) && rel(11) {
        return 6;
    }
    if rel(0) && rel(4) && rel(7) && rel(10) {
        return 7;
    }
    if rel(0) && rel(3) && rel(7) && rel(10) {
        return 8;
    }
    if rel(0) && rel(3) && rel(6) && rel(10) {
        return 9;
    }
    if rel(0) && rel(3) && rel(6) && rel(9) {
        return 10;
    }
    if rel(0) && rel(4) && rel(7) {
        return 1;
    }
    if rel(0) && rel(3) && rel(7) {
        return 2;
    }
    if rel(0) && rel(3) && rel(6) {
        return 3;
    }
    if rel(0) && rel(4) && rel(8) {
        return 4;
    }
    if rel(0) && (rel(5) || rel(2)) {
        return 5;
    }
    11
}

fn normalize(vector: &mut [f32; DIM]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in vector {
            *value /= norm;
        }
    }
}

fn write_index(path: &Path, by_instrument: &[Vec<IndexedVoicing>; 3]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {:?}: {e}", parent))?;
    }

    let mut buf = Vec::new();
    let count = by_instrument.iter().map(Vec::len).sum::<usize>();

    buf.extend_from_slice(b"OPTK");
    buf.extend_from_slice(&4u32.to_le_bytes());
    let header_size_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&compute_schema_hash().to_le_bytes());
    buf.extend_from_slice(&0xFEFFu16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&(DIM as u32).to_le_bytes());
    buf.extend_from_slice(&(count as u64).to_le_bytes());
    buf.push(INSTRUMENTS.len() as u8);
    buf.extend_from_slice(&[0u8; 7]);

    let inst_offsets_pos = buf.len();
    for _ in 0..INSTRUMENTS.len() {
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
    }

    let metadata_offsets_offset_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes());
    let vectors_offset_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes());
    let metadata_offset_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes());
    let metadata_length_pos = buf.len();
    buf.extend_from_slice(&0u64.to_le_bytes());

    for _ in 0..DIM {
        buf.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let header_size = buf.len() as u32;
    buf[header_size_pos..header_size_pos + 4].copy_from_slice(&header_size.to_le_bytes());

    let metadata_offsets_offset = buf.len() as u64;
    buf[metadata_offsets_offset_pos..metadata_offsets_offset_pos + 8]
        .copy_from_slice(&metadata_offsets_offset.to_le_bytes());
    let offsets_table_start = buf.len();
    for _ in 0..count {
        buf.extend_from_slice(&0u64.to_le_bytes());
    }

    let vectors_offset = buf.len() as u64;
    buf[vectors_offset_pos..vectors_offset_pos + 8].copy_from_slice(&vectors_offset.to_le_bytes());

    for bucket in by_instrument {
        for voicing in bucket {
            for value in voicing.vector {
                buf.extend_from_slice(&value.to_le_bytes());
            }
        }
    }

    let mut running = 0u64;
    let mut pos = inst_offsets_pos;
    for bucket in by_instrument {
        let byte_offset = vectors_offset + running;
        buf[pos..pos + 8].copy_from_slice(&byte_offset.to_le_bytes());
        buf[pos + 8..pos + 16].copy_from_slice(&(bucket.len() as u64).to_le_bytes());
        running += (bucket.len() * DIM * 4) as u64;
        pos += 16;
    }

    let metadata_offset = buf.len() as u64;
    buf[metadata_offset_pos..metadata_offset_pos + 8]
        .copy_from_slice(&metadata_offset.to_le_bytes());

    let mut record_rel_offsets = Vec::with_capacity(count);
    for bucket in by_instrument {
        for voicing in bucket {
            let rel = (buf.len() as u64) - metadata_offset;
            record_rel_offsets.push(rel);
            let packed = rmp_serde::to_vec(&voicing.metadata)
                .map_err(|e| format!("pack metadata {}: {e}", voicing.metadata.diagram))?;
            buf.extend_from_slice(&packed);
        }
    }

    let metadata_length = (buf.len() as u64) - metadata_offset;
    buf[metadata_length_pos..metadata_length_pos + 8]
        .copy_from_slice(&metadata_length.to_le_bytes());

    for (i, rel) in record_rel_offsets.iter().enumerate() {
        let p = offsets_table_start + i * 8;
        buf[p..p + 8].copy_from_slice(&rel.to_le_bytes());
    }

    fs::write(path, buf).map_err(|e| format!("write {:?}: {e}", path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_is_normalized_and_root_encoded() {
        let raw = RawVoicing {
            instrument: "guitar".to_string(),
            _string_count: 6,
            diagram: "3-2-0-0-0-3".to_string(),
            frets: vec![
                "3".to_string(),
                "2".to_string(),
                "0".to_string(),
                "0".to_string(),
                "0".to_string(),
                "3".to_string(),
            ],
            midi_notes: vec![55, 59, 62, 67, 71, 79],
            min_fret: Some(0),
            max_fret: Some(3),
            fret_span: Some(3),
        };

        let vector = build_vector(&raw);
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(vector[119] > 0.0);
    }

    #[test]
    fn quality_inference_recognizes_major_triad() {
        let pcs = pitch_class_presence(&[60, 64, 67]);
        assert_eq!(infer_quality_index(&pcs, 0), 1);
    }
}
