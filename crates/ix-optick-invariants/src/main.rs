//! ix-optick-invariants — corpus-reading invariant checker
//!
//! Companion to `ix-invariant-produce` (which runs against synthetic exemplars).
//! This binary opens the real `optick.index` mmap, groups voicings by pitch-class
//! set, and emits firings.json compatible with `ix-invariant-coverage`.
//!
//! **Phase 1** — invariant #25 (cross-instrument STRUCTURE equality). Under
//! OPTIC-K v4-pp per-partition normalization, voicings with identical PC-set
//! across instruments MUST have bit-identical STRUCTURE slices.
//!
//! **Phase 2** — invariant #32 (same PC-set across octaves: cosine ≈ 1.0).
//! Stricter than #25's cross-instrument check: ALL voicings sharing a PC-set
//! (any instrument, any octave realization) must have STRUCTURE slices that
//! are pairwise cosine-similar to 1.0. STRUCTURE is computed from the PC-set
//! bitmask alone, so this is required by construction; failures indicate
//! octave/position information has leaked into STRUCTURE.

use clap::Parser;
use ix_invariant_coverage::coverage::{Exemplar, Firings};
use ix_optick::OptickIndex;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

// Compact v4-pp-r layout: STRUCTURE is dims 0..24.
const STRUCTURE_OFFSET: usize = 0;
const STRUCTURE_DIM: usize = 24;

#[derive(Parser, Debug)]
#[command(
    name = "ix-optick-invariants",
    about = "Corpus-reading checker for OPTIC-K embedding invariants",
    long_about = "Reads a v4-pp optick.index mmap, enumerates voicings by pitch-class set, and \
                  emits firings.json in the schema ix-invariant-coverage ingests. Implements \
                  invariant #25 (cross-instrument STRUCTURE equality, byte-tolerant) and #32 \
                  (same PC-set across octaves, STRUCTURE pairwise cosine ≈ 1.0). Both invariants \
                  MUST hold under v4-pp-r; failures indicate octave/position information has \
                  leaked into the supposedly O+P+T+I-invariant STRUCTURE partition."
)]
struct Args {
    /// Path to optick.index (v4-pp-r format required)
    #[arg(long)]
    index: PathBuf,

    /// Output path for firings.json; stdout if omitted
    #[arg(long)]
    out: Option<PathBuf>,

    /// Pretty-print JSON output
    #[arg(long)]
    pretty: bool,

    /// Tolerance for STRUCTURE-slice equality, invariant #25 (float abs diff)
    #[arg(long, default_value = "1e-4")]
    tolerance: f32,

    /// Tolerance for STRUCTURE cosine deviation from 1.0, invariant #32
    #[arg(long, default_value = "1e-4")]
    cosine_tolerance: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("[1/4] Opening OPTIC-K index: {}", args.index.display());
    let index = OptickIndex::open(&args.index)?;
    let count = index.count() as usize;
    eprintln!("      loaded {} voicings, dim={}", count, index.dimension());

    // ── Single pass: build both groupings simultaneously ──────────────────────
    eprintln!("[2/4] Grouping voicings by pitch-class set...");
    // For #25: pc_set → instrument → representative voicing index.
    let mut by_pcs_inst: BTreeMap<u16, BTreeMap<String, usize>> = BTreeMap::new();
    // For #32: pc_set → all voicing indices (any instrument, any octave realization).
    let mut by_pcs_all: BTreeMap<u16, Vec<usize>> = BTreeMap::new();

    for i in 0..count {
        let meta = match index.metadata(i) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let pcs = midi_to_pc_set(&meta.midi_notes);
        by_pcs_inst
            .entry(pcs)
            .or_default()
            .entry(meta.instrument)
            .or_insert(i);
        by_pcs_all.entry(pcs).or_default().push(i);
    }
    eprintln!("      {} distinct PC-sets", by_pcs_all.len());

    let mut all_exemplars: Vec<Exemplar> = Vec::new();
    let mut fired: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();

    // ── Invariant #25: cross-instrument STRUCTURE equality ────────────────────
    eprintln!("[3/4] Testing invariant #25 (cross-instrument STRUCTURE equality)...");
    let (ex25, fired25, viol25) = check_invariant_25(&index, &by_pcs_inst, args.tolerance);
    let tested25 = ex25.len();
    let passed25 = fired25.len();
    eprintln!(
        "      invariant #25: {}/{} multi-instrument PC-sets PASS, {} FAIL",
        passed25,
        tested25,
        tested25 - passed25
    );
    print_violations("#25", &viol25);
    all_exemplars.extend(ex25);
    fired.insert(25u32, fired25);

    // ── Invariant #32: same PC-set across octaves, cosine ≈ 1.0 ──────────────
    eprintln!("[4/4] Testing invariant #32 (same PC-set across octaves, cosine ≈ 1.0)...");
    let (ex32, fired32, viol32) = check_invariant_32(&index, &by_pcs_all, args.cosine_tolerance);
    let tested32 = ex32.len();
    let passed32 = fired32.len();
    eprintln!(
        "      invariant #32: {}/{} PC-sets PASS, {} FAIL",
        passed32,
        tested32,
        tested32 - passed32
    );
    print_violations("#32", &viol32);
    all_exemplars.extend(ex32);
    fired.insert(32u32, fired32);

    // ── Emit firings.json ─────────────────────────────────────────────────────
    let firings = Firings {
        exemplars: all_exemplars,
        fired,
    };
    let json = if args.pretty {
        serde_json::to_string_pretty(&firings)?
    } else {
        serde_json::to_string(&firings)?
    };

    match args.out {
        Some(path) => {
            std::fs::write(&path, &json)?;
            eprintln!("      wrote firings → {}", path.display());
        }
        None => println!("{}", json),
    }

    // Exit non-zero if any invariant failed — lets CI gate on regressions.
    let any_failed = (tested25 - passed25) > 0 || (tested32 - passed32) > 0;
    if any_failed {
        std::process::exit(1);
    }
    Ok(())
}

/// Invariant #25: voicings with identical PC-set across instruments must have
/// bit-identical STRUCTURE slices (within `tolerance` to absorb float jitter).
fn check_invariant_25(
    index: &OptickIndex,
    by_pcs_inst: &BTreeMap<u16, BTreeMap<String, usize>>,
    tolerance: f32,
) -> (Vec<Exemplar>, BTreeSet<String>, Vec<String>) {
    let multi_instrument: Vec<(&u16, &BTreeMap<String, usize>)> = by_pcs_inst
        .iter()
        .filter(|(_, inst_map)| inst_map.len() >= 2)
        .collect();

    let mut exemplars = Vec::new();
    let mut fired = BTreeSet::new();
    let mut violations = Vec::new();

    for (pcs, inst_map) in &multi_instrument {
        let exemplar_id = format!("pcs-0x{:03X}-n{}", pcs, inst_map.len());
        let instruments_list = inst_map.keys().cloned().collect::<Vec<_>>().join(",");
        let description = format!(
            "PC-set 0x{:03X} (cardinality {}) across instruments: {}",
            pcs,
            pcs.count_ones(),
            instruments_list
        );
        exemplars.push(Exemplar {
            id: exemplar_id.clone(),
            description: description.clone(),
            kind: "embedding-invariant".to_string(),
        });

        let slices: Vec<Vec<f32>> = inst_map
            .values()
            .filter_map(|&vidx| {
                index
                    .vector(vidx)
                    .map(|v| v[STRUCTURE_OFFSET..STRUCTURE_OFFSET + STRUCTURE_DIM].to_vec())
            })
            .collect();
        if slices.len() < 2 {
            continue;
        }

        let reference = &slices[0];
        let mut max_diff = 0.0f32;
        let mut all_equal = true;
        for other in &slices[1..] {
            for (a, b) in other.iter().zip(reference.iter()) {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                }
                if d > tolerance {
                    all_equal = false;
                }
            }
        }

        if all_equal {
            fired.insert(exemplar_id);
        } else if violations.len() < 5 {
            violations.push(format!("  {} — max_diff={:.6}", description, max_diff));
        }
    }

    (exemplars, fired, violations)
}

/// Invariant #32: STRUCTURE is computed from PC-set alone, so any two voicings
/// sharing a PC-set must have STRUCTURE cosine ≈ 1.0 — independent of instrument
/// or MIDI octave realization. Stricter than #25 because it spans intra-instrument
/// octave shifts as well as cross-instrument groupings.
fn check_invariant_32(
    index: &OptickIndex,
    by_pcs_all: &BTreeMap<u16, Vec<usize>>,
    cosine_tolerance: f32,
) -> (Vec<Exemplar>, BTreeSet<String>, Vec<String>) {
    let multi_voicing: Vec<(&u16, &Vec<usize>)> = by_pcs_all
        .iter()
        .filter(|(_, vs)| vs.len() >= 2)
        .collect();

    let mut exemplars = Vec::new();
    let mut fired = BTreeSet::new();
    let mut violations = Vec::new();

    for (pcs, voicings) in &multi_voicing {
        let exemplar_id = format!("pcs-0x{:03X}-v{}", pcs, voicings.len());
        let description = format!(
            "PC-set 0x{:03X} (cardinality {}) across {} voicing realizations",
            pcs,
            pcs.count_ones(),
            voicings.len()
        );
        exemplars.push(Exemplar {
            id: exemplar_id.clone(),
            description: description.clone(),
            kind: "embedding-invariant".to_string(),
        });

        let Some(reference) = index
            .vector(voicings[0])
            .map(|v| v[STRUCTURE_OFFSET..STRUCTURE_OFFSET + STRUCTURE_DIM].to_vec())
        else {
            continue;
        };

        let mut min_cos = 1.0f32;
        let mut all_pass = true;
        for &vidx in voicings.iter().skip(1) {
            let Some(other) = index.vector(vidx) else {
                continue;
            };
            let other_slice = &other[STRUCTURE_OFFSET..STRUCTURE_OFFSET + STRUCTURE_DIM];
            let cos = cosine(&reference, other_slice);
            if cos < min_cos {
                min_cos = cos;
            }
            if (1.0 - cos).abs() > cosine_tolerance {
                all_pass = false;
            }
        }

        if all_pass {
            fired.insert(exemplar_id);
        } else if violations.len() < 5 {
            violations.push(format!("  {} — min_cos={:.6}", description, min_cos));
        }
    }

    (exemplars, fired, violations)
}

fn print_violations(label: &str, violations: &[String]) {
    if violations.is_empty() {
        return;
    }
    eprintln!(
        "      {} first violations (max {} shown):",
        label,
        violations.len()
    );
    for v in violations {
        eprintln!("{}", v);
    }
}

/// Cosine similarity between two equal-length f32 slices. Returns 0.0 if either
/// vector has zero norm — invariant #32 expects unit-normalized partitions, so
/// a zero-norm slice is itself a violation worth surfacing as min_cos = 0.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Convert a slice of MIDI note numbers into a 12-bit pitch-class bitmask.
fn midi_to_pc_set(midi: &[i32]) -> u16 {
    let mut m = 0u16;
    for &n in midi {
        let pc = ((n.rem_euclid(12)) as u16) & 0xFFF;
        m |= 1 << pc;
    }
    m & 0xFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_returns_one() {
        let v = [0.5_f32, 0.5, 0.5, 0.5];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_vectors_returns_zero() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_norm_returns_zero_safely() {
        let z = [0.0_f32, 0.0, 0.0];
        let v = [1.0_f32, 0.0, 0.0];
        assert_eq!(cosine(&z, &v), 0.0);
    }

    #[test]
    fn cosine_anti_parallel_returns_negative_one() {
        let a = [1.0_f32, 0.0];
        let b = [-1.0_f32, 0.0];
        assert!((cosine(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn midi_to_pc_set_c_major_triad() {
        // C-E-G in two octaves → bits 0, 4, 7 set
        let midi = [60, 64, 67, 72, 76, 79];
        let pcs = midi_to_pc_set(&midi);
        assert_eq!(pcs, 0b0000_1001_0001);
    }

    #[test]
    fn midi_to_pc_set_octave_invariant() {
        // Same chord in two different octaves → identical PC-set
        let low = [48, 52, 55];
        let high = [72, 76, 79];
        assert_eq!(midi_to_pc_set(&low), midi_to_pc_set(&high));
    }
}
