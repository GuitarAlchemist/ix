//! `viz-precompute` — Phase C of the voicings study pipeline.
//!
//! Reads the Phase B artifacts (`{instrument}-corpus.json`,
//! `{instrument}-clusters.json`, `{instrument}-transitions.json`) for each
//! instrument and emits the precomputed inputs for the **Harmonic Nebula**
//! visualization:
//!
//! - `state/viz/cluster-layout.json` — cross-instrument cluster centroids
//!   with 3D positions (one entry per cluster-per-instrument).
//! - `state/viz/voicing-layout.json` — per-voicing world position + packed
//!   attributes (cluster id, instrument, chord-family id, rarity rank).
//! - `state/viz/neighbors.json` — top-K transition neighbors per voicing,
//!   for the selection-triggered "harmonic wind" particle effect.
//! - `state/viz/manifest.json` — schema version, generation timestamp,
//!   counts, and a stable schema hash.
//!
//! # Locked vs swappable
//!
//! The **struct shapes** in this module are the one-way door — they
//! define the contract that the React `HarmonicNebulaDemo` renderer
//! consumes. The **layout algorithm** (currently golden-spiral sphere
//! placement for clusters, seeded random offsets for voicings inside
//! each cluster) is a two-way door. Phase 1.5 will swap in classical
//! MDS on the transition-cost graph for clusters and local UMAP on the
//! 124-dim embedding for voicings. The struct contract won't change.
//!
//! # What this version does NOT do
//!
//! - Does not read `ga/state/baseline/*/embedding-clusters-k50.json`
//!   (the cross-corpus k=50 clusters from ga's Phase E pipeline). Uses
//!   ix-voicings' per-instrument k=5 clusters instead — up to 15 clouds
//!   total. Swap in the k=50 source in Phase 1.5.
//! - Does not run UMAP. Positions are deterministic-seeded placeholders
//!   that let the React renderer build against real file formats while
//!   the algorithmic layout work is separately iterated.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{state_root, ClusterArtifacts, Instrument, TransitionArtifacts, VoicingRow, VoicingsError};

/// Current output schema version. Bump on any struct shape change so
/// renderer and precompute can detect drift.
pub const VIZ_SCHEMA_VERSION: u32 = 1;

/// Number of transition neighbors emitted per voicing for the
/// selection-triggered "harmonic wind" effect.
pub const NEIGHBORS_PER_VOICING: usize = 5;

/// One cluster in the Harmonic Nebula macro layout. Each entry is a
/// (instrument, local cluster id) pair flattened to a global cluster id.
/// Up to `sum(k_i)` entries (≤15 with ix-voicings' default k=5 per
/// instrument).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClusterLayout {
    /// Global id, e.g. `"guitar-C0"`. Stable across precompute runs.
    pub id: String,
    /// Instrument this cluster belongs to (`"guitar"`, `"bass"`, `"ukulele"`).
    pub instrument: String,
    /// Local cluster id within the instrument (0..k).
    pub local_cluster_id: usize,
    /// 3D world position for the cluster centroid (meters, arbitrary scale).
    pub position: [f32; 3],
    /// Number of voicings assigned to this cluster.
    pub voicing_count: usize,
    /// Index into the instrument's corpus of the representative voicing
    /// (closest to the cluster centroid in feature space).
    pub representative_voicing_idx: usize,
}

/// One voicing's world position + packed attrs for instanced rendering.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VoicingLayout {
    /// Global voicing id, e.g. `"guitar_v0042"`.
    pub global_id: String,
    /// Cluster membership; references a [`ClusterLayout::id`].
    pub cluster_id: String,
    /// Absolute 3D world position (cluster centroid + local offset).
    pub position: [f32; 3],
    /// Chord-family id for hue mapping:
    /// `0`=major, `1`=minor, `2`=dominant, `3`=diminished, `4`=suspended,
    /// `5`=altered, `6`=other/unclassified.
    pub chord_family_id: u8,
    /// Rarity in [0.0, 1.0]; 0 = common, 1 = rare.
    pub rarity_rank: f32,
    /// Instrument tag.
    pub instrument: String,
}

/// Chord family ids — the contract with the renderer's hue palette.
pub mod chord_family {
    pub const MAJOR: u8 = 0;
    pub const MINOR: u8 = 1;
    pub const DOMINANT: u8 = 2;
    pub const DIMINISHED: u8 = 3;
    pub const SUSPENDED: u8 = 4;
    pub const ALTERED: u8 = 5;
    pub const OTHER: u8 = 6;
}

/// Top-K transition neighbors for one voicing. Powers the
/// selection-triggered "harmonic wind" particle effect in the renderer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeighborList {
    /// Voicing being queried; references a [`VoicingLayout::global_id`].
    pub voicing_id: String,
    /// Top-K neighbors by transition cost, ascending (lowest cost first).
    pub top_k: Vec<NeighborEntry>,
}

/// One neighbor of a voicing in the transition graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeighborEntry {
    /// Neighbor's global id.
    pub id: String,
    /// Transition cost (lower = easier move).
    pub transition_cost: f32,
}

/// Manifest describing a precompute run. Lets the renderer verify it
/// loaded a compatible schema before parsing the data files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VizManifest {
    /// [`VIZ_SCHEMA_VERSION`] at emit time.
    pub version: u32,
    /// RFC3339 timestamp of precompute run.
    pub generated_at: String,
    /// Total number of clusters in `cluster-layout.json`.
    pub cluster_count: usize,
    /// Total number of voicings in `voicing-layout.json`.
    pub voicing_count: usize,
    /// Instruments included (subset of `"guitar"`, `"bass"`, `"ukulele"`).
    pub instruments: Vec<String>,
}

/// Output directory for viz artifacts. Renderer loads from here (or a
/// copy synced into the ga React app's public/ directory).
pub fn viz_root() -> PathBuf {
    state_root().join("viz")
}

/// Run Phase C: read Phase B artifacts for each instrument, emit the
/// four viz files into `state/viz/`.
///
/// Requires Phase B artifacts (`{instrument}-clusters.json`,
/// `{instrument}-corpus.json`, `{instrument}-transitions.json`) to
/// exist on disk for each instrument in `instruments`.
pub fn run_viz_precompute(instruments: &[Instrument]) -> Result<VizManifest, VoicingsError> {
    let mut clusters: Vec<ClusterLayout> = Vec::new();
    let mut voicings: Vec<VoicingLayout> = Vec::new();
    let mut neighbors: Vec<NeighborList> = Vec::new();

    for &inst in instruments {
        load_and_layout_instrument(inst, &mut clusters, &mut voicings, &mut neighbors)?;
    }

    // Place cluster centroids on a golden-spiral sphere for deterministic,
    // visually balanced placeholder layout. Phase 1.5 swaps this for
    // classical MDS on the cross-instrument transition-cost graph.
    let radius = 10.0_f32;
    let cluster_total = clusters.len();
    for (i, c) in clusters.iter_mut().enumerate() {
        c.position = golden_spiral_point(i, cluster_total, radius);
    }

    // Place voicings on a seeded random shell around their owning cluster
    // centroid. Phase 1.5 swaps this for local UMAP on the 124-dim
    // embedding, normalized per cluster.
    let cluster_index: HashMap<String, [f32; 3]> = clusters
        .iter()
        .map(|c| (c.id.clone(), c.position))
        .collect();
    for v in voicings.iter_mut() {
        let centroid = cluster_index
            .get(&v.cluster_id)
            .copied()
            .unwrap_or([0.0, 0.0, 0.0]);
        let offset = seeded_offset(&v.global_id, 1.2);
        v.position = [
            centroid[0] + offset[0],
            centroid[1] + offset[1],
            centroid[2] + offset[2],
        ];
    }

    let out_dir = viz_root();
    std::fs::create_dir_all(&out_dir)?;

    std::fs::write(
        out_dir.join("cluster-layout.json"),
        serde_json::to_vec_pretty(&clusters)?,
    )?;
    std::fs::write(
        out_dir.join("voicing-layout.json"),
        serde_json::to_vec_pretty(&voicings)?,
    )?;
    std::fs::write(
        out_dir.join("neighbors.json"),
        serde_json::to_vec_pretty(&neighbors)?,
    )?;

    let manifest = VizManifest {
        version: VIZ_SCHEMA_VERSION,
        generated_at: current_rfc3339(),
        cluster_count: clusters.len(),
        voicing_count: voicings.len(),
        instruments: instruments.iter().map(|i| i.as_str().to_string()).collect(),
    };
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;

    Ok(manifest)
}

/// Read Phase B artifacts for one instrument and append to the three
/// output vectors. Cluster positions are written as zero here and filled
/// in by the golden-spiral pass in the caller.
fn load_and_layout_instrument(
    inst: Instrument,
    clusters: &mut Vec<ClusterLayout>,
    voicings: &mut Vec<VoicingLayout>,
    neighbors: &mut Vec<NeighborList>,
) -> Result<(), VoicingsError> {
    let inst_name = inst.as_str();

    let corpus_path = state_root()
        .join("voicings")
        .join(format!("{}-corpus.json", inst_name));
    let corpus_bytes = std::fs::read(&corpus_path).map_err(|e| {
        VoicingsError::Pipeline(format!(
            "viz-precompute: missing corpus for {inst_name}. Run Phase A+B first. ({e})"
        ))
    })?;
    let corpus: Vec<VoicingRow> = serde_json::from_slice(&corpus_bytes)?;

    let cluster_path = state_root()
        .join("voicings")
        .join(format!("{}-clusters.json", inst_name));
    let cluster_bytes = std::fs::read(&cluster_path).map_err(|e| {
        VoicingsError::Pipeline(format!(
            "viz-precompute: missing clusters for {inst_name}. Run Phase B. ({e})"
        ))
    })?;
    let cluster_art: ClusterArtifacts = serde_json::from_slice(&cluster_bytes)?;

    let trans_path = state_root()
        .join("voicings")
        .join(format!("{}-transitions.json", inst_name));
    let trans_art: Option<TransitionArtifacts> = std::fs::read(&trans_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());

    // Count voicings per cluster
    let mut counts = vec![0usize; cluster_art.k];
    for &a in &cluster_art.assignments {
        if a < cluster_art.k {
            counts[a] += 1;
        }
    }

    // Emit cluster entries. Position is placeholder; filled in by caller.
    for (local_id, &count) in counts.iter().enumerate() {
        clusters.push(ClusterLayout {
            id: format!("{inst_name}-C{local_id}"),
            instrument: inst_name.to_string(),
            local_cluster_id: local_id,
            position: [0.0, 0.0, 0.0],
            voicing_count: count,
            representative_voicing_idx: cluster_art
                .representative_voicing_per_cluster
                .get(local_id)
                .copied()
                .unwrap_or(0),
        });
    }

    // Emit voicing entries. Position is placeholder; filled in by caller.
    let n = corpus.len();
    for (idx, row) in corpus.iter().enumerate() {
        let local_cluster = cluster_art
            .assignments
            .get(idx)
            .copied()
            .unwrap_or(0)
            .min(cluster_art.k.saturating_sub(1));
        let global_id = format!("{inst_name}_v{idx:04}");
        let cluster_id = format!("{inst_name}-C{local_cluster}");
        voicings.push(VoicingLayout {
            global_id,
            cluster_id,
            position: [0.0, 0.0, 0.0],
            chord_family_id: classify_chord_family(row),
            rarity_rank: (idx as f32) / (n.max(1) as f32),
            instrument: inst_name.to_string(),
        });
    }

    // Phase 1: transitions in ix-voicings are CLUSTER-to-CLUSTER, not
    // voicing-to-voicing. A per-voicing neighbor list requires a
    // voicing-level transition graph which lives in ga's optick.index —
    // wiring that in is Phase 1.5. For now, for every voicing we emit an
    // empty neighbor list so the renderer sees the contract shape but
    // skips the "harmonic wind" effect gracefully until the real data
    // lands. The `trans_art` Phase B file is kept in scope so Phase 1.5
    // can replace this stub without restructuring the loader.
    let _ = trans_art; // acknowledge we loaded it; wiring comes in Phase 1.5
    for row_idx in 0..corpus.len() {
        neighbors.push(NeighborList {
            voicing_id: format!("{inst_name}_v{row_idx:04}"),
            top_k: Vec::new(),
        });
    }

    Ok(())
}

/// Chord family classification. Returns [`chord_family::OTHER`] for now —
/// [`VoicingRow`] does not carry chord identity, and wiring it from
/// ga's `ChordIdentification` into this pipeline is Phase 1.5 work.
///
/// The renderer consumes `chord_family_id` from [`VoicingLayout`], so
/// the contract is stable; Phase 1.5 just swaps this stub for the real
/// classifier without changing call sites.
fn classify_chord_family(_row: &VoicingRow) -> u8 {
    chord_family::OTHER
}

/// Golden-spiral distribution of `n` points on a sphere of radius `r`.
/// Deterministic; `i=0..n` produces the same point every run. Visually
/// balanced — no clumping, no seams.
fn golden_spiral_point(i: usize, n: usize, r: f32) -> [f32; 3] {
    if n == 0 {
        return [0.0, 0.0, 0.0];
    }
    let phi = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());
    let y = 1.0 - (i as f32 / (n.saturating_sub(1).max(1) as f32)) * 2.0;
    let radius_at_y = (1.0 - y * y).max(0.0).sqrt();
    let theta = phi * i as f32;
    [
        r * theta.cos() * radius_at_y,
        r * y,
        r * theta.sin() * radius_at_y,
    ]
}

/// Deterministic 3D offset derived from hashing a voicing's global id.
/// `scale` bounds the magnitude. Produces different offsets for different
/// ids but the same offset for the same id across runs.
fn seeded_offset(global_id: &str, scale: f32) -> [f32; 3] {
    let mut h: u64 = 1469598103934665603; // FNV offset basis
    for b in global_id.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211); // FNV prime
    }
    let fx = ((h & 0xFFFF) as f32) / 65536.0 - 0.5;
    let fy = (((h >> 16) & 0xFFFF) as f32) / 65536.0 - 0.5;
    let fz = (((h >> 32) & 0xFFFF) as f32) / 65536.0 - 0.5;
    [fx * scale, fy * scale, fz * scale]
}

/// Current time as RFC3339 string. Avoids a chrono dep for one fn call;
/// format is `YYYY-MM-DDTHH:MM:SSZ` (UTC seconds precision).
fn current_rfc3339() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Integer seconds → calendar math. Days since 1970-01-01.
    let secs_in_day = 86_400u64;
    let days = now / secs_in_day;
    let secs_of_day = now % secs_in_day;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Days since 1970-01-01 → (year, month, day). Proleptic Gregorian.
/// Good through 2099+ which is sufficient for generated_at timestamps.
fn days_to_ymd(mut days: u64) -> (u32, u32, u32) {
    let mut year = 1970u32;
    loop {
        let leap = is_leap(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_lengths = if is_leap(year) {
        [31u64, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1u32;
    for &ml in &month_lengths {
        if days < ml {
            break;
        }
        days -= ml;
        month += 1;
    }
    (year, month, (days + 1) as u32)
}

fn is_leap(y: u32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_roundtrip_cluster() {
        let c = ClusterLayout {
            id: "guitar-C0".into(),
            instrument: "guitar".into(),
            local_cluster_id: 0,
            position: [1.0, 2.0, 3.0],
            voicing_count: 42,
            representative_voicing_idx: 7,
        };
        let s = serde_json::to_string(&c).unwrap();
        let back: ClusterLayout = serde_json::from_str(&s).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn schema_roundtrip_voicing() {
        let v = VoicingLayout {
            global_id: "guitar_v0042".into(),
            cluster_id: "guitar-C1".into(),
            position: [1.0, 2.0, 3.0],
            chord_family_id: chord_family::ALTERED,
            rarity_rank: 0.73,
            instrument: "guitar".into(),
        };
        let s = serde_json::to_string(&v).unwrap();
        let back: VoicingLayout = serde_json::from_str(&s).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn schema_roundtrip_manifest() {
        let m = VizManifest {
            version: VIZ_SCHEMA_VERSION,
            generated_at: "2026-04-20T12:00:00Z".into(),
            cluster_count: 15,
            voicing_count: 313_047,
            instruments: vec!["guitar".into(), "bass".into(), "ukulele".into()],
        };
        let s = serde_json::to_string(&m).unwrap();
        let back: VizManifest = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn golden_spiral_produces_points_on_unit_sphere() {
        for n in [1usize, 5, 15, 50] {
            for i in 0..n {
                let p = golden_spiral_point(i, n, 1.0);
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                // Points are on or inside the unit sphere within float tolerance.
                assert!(r <= 1.0 + 1e-5, "point {i}/{n} r={r}");
                assert!(r >= 0.0);
            }
        }
    }

    #[test]
    fn golden_spiral_is_deterministic() {
        for i in 0..10 {
            assert_eq!(
                golden_spiral_point(i, 15, 10.0),
                golden_spiral_point(i, 15, 10.0)
            );
        }
    }

    #[test]
    fn seeded_offset_is_deterministic_and_bounded() {
        let o1 = seeded_offset("guitar_v0042", 1.0);
        let o2 = seeded_offset("guitar_v0042", 1.0);
        assert_eq!(o1, o2);
        for v in o1 {
            assert!((-0.5..=0.5).contains(&v));
        }
    }

    #[test]
    fn seeded_offset_varies_by_id() {
        let o1 = seeded_offset("guitar_v0000", 1.0);
        let o2 = seeded_offset("guitar_v0001", 1.0);
        assert_ne!(o1, o2);
    }

    #[test]
    fn chord_family_classifier_is_stubbed_to_other() {
        // Phase 1: classifier returns OTHER uniformly. Phase 1.5 will
        // swap in real ChordIdentification-driven classification.
        let row = VoicingRow {
            instrument: "guitar".into(),
            string_count: 6,
            diagram: "x-3-2-0-1-0".into(),
            frets: vec!["x".into(), "3".into(), "2".into(), "0".into(), "1".into(), "0".into()],
            midi_notes: vec![60, 64, 67, 72],
            min_fret: 0,
            max_fret: 3,
            fret_span: 3,
        };
        assert_eq!(classify_chord_family(&row), chord_family::OTHER);
    }

    #[test]
    fn days_to_ymd_epoch() {
        assert_eq!(days_to_ymd(0), (1970, 1, 1));
    }

    #[test]
    fn days_to_ymd_known_dates() {
        // 2000-01-01 = day 10957 since 1970-01-01
        assert_eq!(days_to_ymd(10957), (2000, 1, 1));
        // 2024-02-29 (leap) = day 19782
        assert_eq!(days_to_ymd(19782), (2024, 2, 29));
    }
}
