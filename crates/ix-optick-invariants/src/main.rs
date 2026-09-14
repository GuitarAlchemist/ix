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
//!
//! **Phase 3** — invariant #36 (Z-pair STRUCTURE separation). Z-related set
//! classes share an interval-class vector but live in different D₁₂ orbits
//! (e.g. Forte 4-Z15 ↔ 4-Z29, both with ICV [1,1,1,1,1,1]). If STRUCTURE
//! merely re-encoded ICV, Z-pairs would have cosine 1.0 — collapsing two
//! distinct set classes into one point. So for every Z-pair represented in
//! the corpus, STRUCTURE cosine MUST be strictly less than 1.0. This check
//! catches the failure mode "STRUCTURE accidentally degenerated to ICV".
//!
//! **Phase 4** — invariant #37 (no dead dimension). A compact dimension whose
//! value is constant across the whole index (always zero, or the same non-zero
//! value everywhere) carries no information: it adds the same term to every
//! dot product. Exemplars are the 124 compact dims; a dim fires when it varies.
//! See GuitarAlchemist/ga#552.
//!
//! **Phase 5** — invariant #38 (no dead weighted partition). A partition whose
//! dims are all dead but whose header weight is > 0 spends similarity weight on
//! a constant. Exemplars are the 6 partitions. See GuitarAlchemist/ga#616.
//!
//! #37 and #38 only change the exit code with `--fail-on-dead`: the current
//! corpus has known dead dims, and `ix-autoresearch` treats a non-zero exit as
//! an eval failure.

use clap::Parser;
use ix_invariant_coverage::coverage::{Exemplar, Firings};
use ix_optick::OptickIndex;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

// Compact v4-pp-r layout: STRUCTURE is dims 0..24.
const STRUCTURE_OFFSET: usize = 0;
const STRUCTURE_DIM: usize = 24;

// Compact v4-pp-r partition layout: (name, offset, width).
// @ai:invariant PARTITIONS mirrors the ix-optick SCHEMA_SEED layout byte-for-byte [T:test conf:0.95 src:tests::partition_layout_matches_ix_optick_schema_hash]
const PARTITIONS: [(&str, usize, usize); 6] = [
    ("STRUCTURE", 0, 24),
    ("MORPHOLOGY", 24, 24),
    ("CONTEXT", 48, 12),
    ("SYMBOLIC", 60, 12),
    ("MODAL", 72, 40),
    ("ROOT", 112, 12),
];

#[derive(Parser, Debug)]
#[command(
    name = "ix-optick-invariants",
    about = "Corpus-reading checker for OPTIC-K embedding invariants",
    long_about = "Reads a v4-pp optick.index mmap, enumerates voicings by pitch-class set, and \
                  emits firings.json in the schema ix-invariant-coverage ingests. Implements \
                  invariant #25 (cross-instrument STRUCTURE equality, byte-tolerant) and #32 \
                  (same PC-set across octaves, STRUCTURE pairwise cosine ≈ 1.0). Both invariants \
                  MUST hold under v4-pp-r; failures indicate octave/position information has \
                  leaked into the supposedly O+P+T+I-invariant STRUCTURE partition. Also reports \
                  #36 (Z-pair STRUCTURE separation), #37 (no dead dimension: every compact dim \
                  varies across the index) and #38 (no partition with weight > 0 whose dims are \
                  all dead)."
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

    /// A dim is dead when max - min across the index is at most this, invariants #37/#38
    #[arg(long, default_value = "1e-6")]
    dead_tolerance: f32,

    /// Exit non-zero when #37 or #38 fail (off by default: the current corpus has known dead dims)
    #[arg(long)]
    fail_on_dead: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("[1/7] Opening OPTIC-K index: {}", args.index.display());
    let index = OptickIndex::open(&args.index)?;
    let count = index.count() as usize;
    eprintln!("      loaded {} voicings, dim={}", count, index.dimension());

    // ── Single pass: build both groupings simultaneously ──────────────────────
    eprintln!("[2/7] Grouping voicings by pitch-class set...");
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
    eprintln!("[3/7] Testing invariant #25 (cross-instrument STRUCTURE equality)...");
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
    eprintln!("[4/7] Testing invariant #32 (same PC-set across octaves, cosine ≈ 1.0)...");
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

    // ── Invariant #36: Z-pair STRUCTURE separation ────────────────────────────
    eprintln!("[5/7] Testing invariant #36 (Z-pair STRUCTURE separation, cosine < 1)...");
    let (ex36, fired36, viol36) =
        check_invariant_36_z_pair_separation(&index, &by_pcs_all, args.cosine_tolerance);
    let tested36 = ex36.len();
    let passed36 = fired36.len();
    eprintln!(
        "      invariant #36: {}/{} corpus-represented Z-pairs PASS, {} FAIL",
        passed36,
        tested36,
        tested36 - passed36
    );
    print_violations("#36", &viol36);
    all_exemplars.extend(ex36);
    fired.insert(36u32, fired36);

    // ── Invariant #37: no dead dimension ──────────────────────────────────────
    eprintln!("[6/7] Testing invariant #37 (no dead dimension, max - min > tolerance)...");
    let dim = index.dimension() as usize;
    let ranges = dimension_ranges(index.vectors(), dim);
    let (ex37, fired37, viol37) = check_invariant_37_dead_dimensions(&ranges, args.dead_tolerance);
    let tested37 = ex37.len();
    let passed37 = fired37.len();
    let always_zero = ranges
        .iter()
        .filter(|&&(min, max)| min == 0.0 && max == 0.0)
        .count();
    eprintln!(
        "      invariant #37: {}/{} compact dims PASS, {} FAIL ({} always zero, {} constant non-zero)",
        passed37,
        tested37,
        tested37 - passed37,
        always_zero,
        (tested37 - passed37) - always_zero
    );
    print_violations("#37", &viol37);
    all_exemplars.extend(ex37);
    fired.insert(37u32, fired37);

    // ── Invariant #38: no dead weighted partition ─────────────────────────────
    eprintln!("[7/7] Testing invariant #38 (no dead partition with weight > 0)...");
    let (ex38, fired38, viol38) = check_invariant_38_dead_weighted_partitions(
        &ranges,
        &index.header().partition_weights,
        args.dead_tolerance,
    );
    let tested38 = ex38.len();
    let passed38 = fired38.len();
    eprintln!(
        "      invariant #38: {}/{} partitions PASS, {} FAIL",
        passed38,
        tested38,
        tested38 - passed38
    );
    print_violations("#38", &viol38);
    all_exemplars.extend(ex38);
    fired.insert(38u32, fired38);

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
    // #37/#38 count only with --fail-on-dead.
    let any_failed = (tested25 - passed25) > 0
        || (tested32 - passed32) > 0
        || (tested36 - passed36) > 0
        || (args.fail_on_dead && ((tested37 - passed37) > 0 || (tested38 - passed38) > 0));
    if any_failed {
        std::process::exit(1);
    }
    Ok(())
}

/// Invariant #36: every Z-related orbit pair represented in the corpus must
/// have STRUCTURE cosine STRICTLY LESS than 1.0. Z-pairs share an ICV; if
/// STRUCTURE merely re-encoded ICV they would have cosine == 1.0 and the
/// embedding would silently merge two distinct Forte classes. This check
/// catches that failure mode.
///
/// For each Z-pair `(rep_a, rep_b)` from `ix_bracelet::z_related_pairs`, we
/// expand each rep to its full D₁₂ orbit and look up any voicing whose
/// PC-set is in either side. If both sides are corpus-represented, we pick
/// one voicing per side and check `cosine(STRUCTURE_a, STRUCTURE_b)` is
/// at most `1.0 - cosine_tolerance`.
fn check_invariant_36_z_pair_separation(
    index: &OptickIndex,
    by_pcs_all: &BTreeMap<u16, Vec<usize>>,
    cosine_tolerance: f32,
) -> (Vec<Exemplar>, BTreeSet<String>, Vec<String>) {
    let mut exemplars = Vec::new();
    let mut fired = BTreeSet::new();
    let mut violations = Vec::new();

    for (rep_a, rep_b) in ix_bracelet::z_related_pairs() {
        // Expand each rep to its full D₁₂ orbit and find any corpus voicing
        // landing on it. Pick the first voicing index found for each side.
        let voicing_a = ix_bracelet::orbit_unique(rep_a)
            .into_iter()
            .find_map(|s| by_pcs_all.get(&s.raw()).and_then(|vs| vs.first().copied()));
        let voicing_b = ix_bracelet::orbit_unique(rep_b)
            .into_iter()
            .find_map(|s| by_pcs_all.get(&s.raw()).and_then(|vs| vs.first().copied()));

        let exemplar_id = format!("z-pair-0x{:03X}-0x{:03X}", rep_a.raw(), rep_b.raw());
        let card = rep_a.cardinality();
        let description = format!(
            "Z-pair card={} {{ {} }} ↔ {{ {} }}",
            card,
            pcs_string(rep_a),
            pcs_string(rep_b)
        );

        let (vidx_a, vidx_b) = match (voicing_a, voicing_b) {
            (Some(a), Some(b)) => (a, b),
            // If either side isn't represented in the corpus, skip — the test
            // is vacuous (no two voicings to compare). Don't add an exemplar
            // for a vacuous case so the pass/fail rate is meaningful.
            _ => continue,
        };

        exemplars.push(Exemplar {
            id: exemplar_id.clone(),
            description: description.clone(),
            kind: "embedding-invariant".to_string(),
        });

        let Some(vec_a) = index.vector(vidx_a) else {
            continue;
        };
        let Some(vec_b) = index.vector(vidx_b) else {
            continue;
        };
        let slice_a = &vec_a[STRUCTURE_OFFSET..STRUCTURE_OFFSET + STRUCTURE_DIM];
        let slice_b = &vec_b[STRUCTURE_OFFSET..STRUCTURE_OFFSET + STRUCTURE_DIM];
        let cos = cosine(slice_a, slice_b);

        // PASS iff cos is meaningfully less than 1.0 (i.e. STRUCTURE
        // distinguishes the two orbits). Use the same cosine_tolerance as #32
        // — anything within tolerance of 1.0 is treated as "STRUCTURE
        // collapsed to ICV", which is the failure we want to catch.
        if cos <= 1.0 - cosine_tolerance {
            fired.insert(exemplar_id);
        } else if violations.len() < 5 {
            violations.push(format!("  {} — cos={:.6}", description, cos));
        }
    }

    (exemplars, fired, violations)
}

/// Per-dimension `(min, max)` over a flat `count * dim` vector slice.
/// An empty slice yields `(0.0, 0.0)` for every dim.
fn dimension_ranges(vectors: &[f32], dim: usize) -> Vec<(f32, f32)> {
    let mut ranges = vec![(f32::INFINITY, f32::NEG_INFINITY); dim];
    for row in vectors.chunks_exact(dim) {
        for (range, &x) in ranges.iter_mut().zip(row) {
            range.0 = range.0.min(x);
            range.1 = range.1.max(x);
        }
    }
    for range in &mut ranges {
        if range.0 > range.1 {
            *range = (0.0, 0.0);
        }
    }
    ranges
}

fn is_dead(range: (f32, f32), tolerance: f32) -> bool {
    range.1 - range.0 <= tolerance
}

/// Name and local index of a compact dim, e.g. `("CONTEXT", 0)` for dim 48.
fn partition_of(dim: usize) -> Option<(&'static str, usize)> {
    PARTITIONS
        .iter()
        .find(|&&(_, offset, width)| dim >= offset && dim < offset + width)
        .map(|&(name, offset, _)| (name, dim - offset))
}

/// Invariant #37: every compact dim must vary across the index. A dim whose
/// `max - min` is within `tolerance` is dead (always zero, or constant).
/// Violations are one line per partition listing its dead dims.
fn check_invariant_37_dead_dimensions(
    ranges: &[(f32, f32)],
    tolerance: f32,
) -> (Vec<Exemplar>, BTreeSet<String>, Vec<String>) {
    let mut exemplars = Vec::new();
    let mut fired = BTreeSet::new();

    for (dim, &range) in ranges.iter().enumerate() {
        let exemplar_id = format!("optick-dim-{:03}", dim);
        let location = match partition_of(dim) {
            Some((name, local)) => format!("{} local {}", name, local),
            None => "no partition".to_string(),
        };
        exemplars.push(Exemplar {
            id: exemplar_id.clone(),
            description: format!(
                "compact dim {} ({}), range [{:.6}, {:.6}]",
                dim, location, range.0, range.1
            ),
            kind: "embedding-invariant".to_string(),
        });
        if !is_dead(range, tolerance) {
            fired.insert(exemplar_id);
        }
    }

    let violations = PARTITIONS
        .iter()
        .filter_map(|&(name, offset, width)| {
            let dead: Vec<String> = (offset..(offset + width).min(ranges.len()))
                .filter(|&d| is_dead(ranges[d], tolerance))
                .map(|d| d.to_string())
                .collect();
            (!dead.is_empty()).then(|| {
                format!(
                    "  {} {}/{} dead: {}",
                    name,
                    dead.len(),
                    width,
                    dead.join(",")
                )
            })
        })
        .collect();

    (exemplars, fired, violations)
}

/// Invariant #38: a partition whose weight is > 0 must have at least one live
/// dim. The header stores the per-dim sqrt-weight scale (GA `OptickIndexWriter`),
/// so the partition weight is the square of the largest scale in its range.
fn check_invariant_38_dead_weighted_partitions(
    ranges: &[(f32, f32)],
    scales: &[f32],
    tolerance: f32,
) -> (Vec<Exemplar>, BTreeSet<String>, Vec<String>) {
    let mut exemplars = Vec::new();
    let mut fired = BTreeSet::new();
    let mut violations = Vec::new();

    for &(name, offset, width) in &PARTITIONS {
        let end = offset + width;
        if end > ranges.len() || end > scales.len() {
            continue;
        }
        let scale = scales[offset..end].iter().copied().fold(0.0f32, f32::max);
        let weight = scale * scale;
        let live = ranges[offset..end]
            .iter()
            .filter(|&&r| !is_dead(r, tolerance))
            .count();
        let exemplar_id = format!("optick-partition-{}", name);
        let description = format!(
            "partition {} (dims {}-{}), weight {:.4}, {}/{} live dims",
            name,
            offset,
            end - 1,
            weight,
            live,
            width
        );
        exemplars.push(Exemplar {
            id: exemplar_id.clone(),
            description: description.clone(),
            kind: "embedding-invariant".to_string(),
        });
        if live > 0 || weight <= 0.0 {
            fired.insert(exemplar_id);
        } else {
            violations.push(format!("  {}", description));
        }
    }

    (exemplars, fired, violations)
}

fn pcs_string(set: ix_bracelet::PcSet) -> String {
    set.iter_pcs()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(",")
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
    let multi_voicing: Vec<(&u16, &Vec<usize>)> =
        by_pcs_all.iter().filter(|(_, vs)| vs.len() >= 2).collect();

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

    #[test]
    fn partition_layout_matches_ix_optick_schema_hash() {
        let layout = PARTITIONS
            .iter()
            .map(|&(name, offset, width)| format!("{}:{}-{}", name, offset, offset + width - 1))
            .collect::<Vec<_>>()
            .join(",");
        let seed = format!("optk-v4-pp-r:{}", layout);
        assert_eq!(
            crc32fast::hash(seed.as_bytes()),
            ix_optick::compute_schema_hash()
        );
    }

    /// Synthetic 124-dim index of 3 voicings: CONTEXT (48..60) always zero,
    /// SYMBOLIC dim 60 constant non-zero, everything else varies.
    fn synthetic_index() -> Vec<f32> {
        let dim = 124;
        let mut vectors = vec![0.0f32; 3 * dim];
        for (row, chunk) in vectors.chunks_exact_mut(dim).enumerate() {
            for (d, x) in chunk.iter_mut().enumerate() {
                *x = match d {
                    48..=59 => 0.0,
                    60 => 0.5,
                    _ => row as f32 * 0.1 + 0.01,
                };
            }
        }
        vectors
    }

    #[test]
    fn dimension_ranges_tracks_min_and_max() {
        let vectors = [1.0_f32, -2.0, 3.0, 0.5];
        assert_eq!(dimension_ranges(&vectors, 2), vec![(1.0, 3.0), (-2.0, 0.5)]);
    }

    #[test]
    fn dimension_ranges_empty_index_is_all_zero() {
        assert_eq!(dimension_ranges(&[], 3), vec![(0.0, 0.0); 3]);
    }

    #[test]
    fn invariant_37_flags_zero_and_constant_dims() {
        let ranges = dimension_ranges(&synthetic_index(), 124);
        let (exemplars, fired, violations) = check_invariant_37_dead_dimensions(&ranges, 1e-6);
        assert_eq!(exemplars.len(), 124);
        assert_eq!(fired.len(), 124 - 13);
        assert!(!fired.contains("optick-dim-048"));
        assert!(!fired.contains("optick-dim-060"));
        assert!(fired.contains("optick-dim-061"));
        assert_eq!(
            violations,
            vec![
                "  CONTEXT 12/12 dead: 48,49,50,51,52,53,54,55,56,57,58,59".to_string(),
                "  SYMBOLIC 1/12 dead: 60".to_string(),
            ]
        );
    }

    #[test]
    fn invariant_37_tolerance_absorbs_jitter() {
        let ranges = [(0.5_f32, 0.5 + 1e-7), (0.0, 0.1)];
        let (_, fired, _) = check_invariant_37_dead_dimensions(&ranges, 1e-6);
        assert_eq!(fired.len(), 1);
        assert!(fired.contains("optick-dim-001"));
    }

    #[test]
    fn invariant_38_flags_dead_partition_only_when_weighted() {
        let ranges = dimension_ranges(&synthetic_index(), 124);
        let mut scales = vec![0.1_f32; 124];
        let (exemplars, fired, violations) =
            check_invariant_38_dead_weighted_partitions(&ranges, &scales, 1e-6);
        assert_eq!(exemplars.len(), 6);
        assert_eq!(fired.len(), 5);
        assert!(!fired.contains("optick-partition-CONTEXT"));
        // SYMBOLIC has one dead dim but 11 live ones: it passes.
        assert!(fired.contains("optick-partition-SYMBOLIC"));
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("CONTEXT"));

        // Zero weight: a dead partition costs nothing, so it passes.
        scales[48..60].fill(0.0);
        let (_, fired, violations) =
            check_invariant_38_dead_weighted_partitions(&ranges, &scales, 1e-6);
        assert_eq!(fired.len(), 6);
        assert!(violations.is_empty());
    }
}
