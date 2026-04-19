//! ix-optick-invariants — corpus-reading invariant checker
//!
//! Companion to `ix-invariant-produce` (which runs against synthetic exemplars).
//! This binary opens the real `optick.index` mmap, groups voicings by pitch-class
//! set, and emits firings.json compatible with `ix-invariant-coverage`.
//!
//! **Phase 1** implements invariant #25 (cross-instrument STRUCTURE equality).
//! Under OPTIC-K v4-pp per-partition normalization, voicings with identical
//! pitch-class set across instruments MUST have bit-identical STRUCTURE slices.
//! This checker verifies that at the data level.

use clap::Parser;
use ix_invariant_coverage::coverage::{Exemplar, Firings};
use ix_optick::OptickIndex;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "ix-optick-invariants",
    about = "Corpus-reading checker for OPTIC-K embedding invariants",
    long_about = "Reads a v4-pp optick.index mmap, enumerates voicings by pitch-class set across \
                  instruments, and emits firings.json in the schema ix-invariant-coverage ingests. \
                  Phase 1 implements invariant #25 (cross-instrument STRUCTURE equality). Under v4-pp \
                  per-partition normalization this invariant MUST hold; failures indicate the \
                  normalization contract is broken."
)]
struct Args {
    /// Path to optick.index (v4-pp format required)
    #[arg(long)]
    index: PathBuf,

    /// Output path for firings.json; stdout if omitted
    #[arg(long)]
    out: Option<PathBuf>,

    /// Pretty-print JSON output
    #[arg(long)]
    pretty: bool,

    /// Tolerance for STRUCTURE-slice equality (float abs diff)
    #[arg(long, default_value = "1e-4")]
    tolerance: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("[1/3] Opening OPTIC-K index: {}", args.index.display());
    let index = OptickIndex::open(&args.index)?;
    let count = index.count() as usize;
    eprintln!("      loaded {} voicings, dim={}", count, index.dimension());

    // ── Group voicings by pitch-class set, collect one representative per instrument ──
    eprintln!("[2/3] Grouping voicings by pitch-class set...");
    // pc_set (u16 bitmask) -> instrument -> representative voicing index
    let mut by_pcs: BTreeMap<u16, BTreeMap<String, usize>> = BTreeMap::new();

    for i in 0..count {
        let meta = match index.metadata(i) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let pcs = midi_to_pc_set(&meta.midi_notes);
        by_pcs
            .entry(pcs)
            .or_default()
            .entry(meta.instrument)
            .or_insert(i);
    }

    let multi_instrument_pcs: Vec<(&u16, &BTreeMap<String, usize>)> = by_pcs
        .iter()
        .filter(|(_, inst_map)| inst_map.len() >= 2)
        .collect();

    eprintln!(
        "      {} distinct PC-sets; {} appear in ≥2 instruments",
        by_pcs.len(),
        multi_instrument_pcs.len()
    );

    // ── Test invariant #25 per multi-instrument PC-set ────────────────────────────
    eprintln!("[3/3] Testing invariant #25 (cross-instrument STRUCTURE equality)...");

    // Compact v4-pp layout: STRUCTURE is dims 0..24.
    const STRUCTURE_OFFSET: usize = 0;
    const STRUCTURE_DIM: usize = 24;

    let mut exemplars: Vec<Exemplar> = Vec::new();
    let mut fired_25: BTreeSet<String> = BTreeSet::new();
    let mut violations: Vec<String> = Vec::new();

    for (pcs, inst_map) in &multi_instrument_pcs {
        let exemplar_id = format!("pcs-0x{:03X}-n{}", pcs, inst_map.len());
        let instruments_list = inst_map
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(",");
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

        // Collect STRUCTURE slice per instrument
        let slices: Vec<(&String, Vec<f32>)> = inst_map
            .iter()
            .filter_map(|(instrument, &vidx)| {
                index
                    .vector(vidx)
                    .map(|v| (instrument, v[STRUCTURE_OFFSET..STRUCTURE_OFFSET + STRUCTURE_DIM].to_vec()))
            })
            .collect();

        if slices.len() < 2 {
            continue;
        }

        // All slices must match the first within tolerance
        let (_first_instr, reference) = &slices[0];
        let mut all_equal = true;
        let mut max_diff = 0.0f32;

        for (_, other) in &slices[1..] {
            for (a, b) in other.iter().zip(reference.iter()) {
                let d = (a - b).abs();
                if d > max_diff {
                    max_diff = d;
                }
                if d > args.tolerance {
                    all_equal = false;
                }
            }
        }

        if all_equal {
            fired_25.insert(exemplar_id);
        } else if violations.len() < 5 {
            // Capture first few violations for diagnostics
            violations.push(format!("  {} — max_diff={:.6}", description, max_diff));
        }
    }

    let tested = exemplars.len();
    let passed = fired_25.len();
    let failed = tested - passed;

    eprintln!(
        "      invariant #25: {}/{} PC-sets PASS, {} FAIL",
        passed, tested, failed
    );
    if !violations.is_empty() {
        eprintln!("      first violations (max {} shown):", violations.len());
        for v in &violations {
            eprintln!("{}", v);
        }
    }

    // ── Emit firings.json ─────────────────────────────────────────────────────────
    let mut fired: BTreeMap<u32, BTreeSet<String>> = BTreeMap::new();
    fired.insert(25u32, fired_25);

    let firings = Firings { exemplars, fired };
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

    // Exit non-zero if any invariant failed — lets CI gate on regressions
    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
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
