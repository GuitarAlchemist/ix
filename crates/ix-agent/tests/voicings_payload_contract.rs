//! Cross-repo contract test for `voicings.payload.v1` — phase 4 of
//! ix:docs/plans/2026-05-02-voicings-in-prime-radiant.md.
//!
//! Verifies two things:
//!
//! 1. **Handler shape** (always runs). The `ix_voicings_payload` MCP
//!    tool returns a payload that conforms to the published schema:
//!    correct `schema` literal, all required fields present, sane
//!    defaults, URL-shaped strings.
//!
//! 2. **On-disk data integrity** (gated on file presence). If
//!    `state/viz/voicing-positions.{bin,meta.json}` exist locally
//!    (i.e. someone has run `serve_viz` against the OPTIC-K corpus),
//!    we further verify the binary length matches the metadata —
//!    `bin_size_bytes == meta.total * 3 floats * 4 bytes` — and that
//!    per-instrument blocks tile the buffer exactly with no gaps or
//!    overlap. Without this, the consumer (GA `createVoicingCloud`)
//!    would silently render garbage.
//!
//! This is the data-layer half of the smoke test. The Three.js render
//! half (does the cloud actually draw at the scene_offset?) is left
//! to manual browser verification — see the plan doc.

use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};
use std::path::PathBuf;

const PROJECT_ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn project_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is crates/ix-agent — go up two to reach workspace root.
    PathBuf::from(PROJECT_ROOT).join("..").join("..")
}

#[test]
fn payload_handler_matches_voicings_payload_v1_schema() {
    let reg = ToolRegistry::new();
    let payload = reg
        .call("ix_voicings_payload", json!({}))
        .expect("ix_voicings_payload reachable via registry");

    assert_eq!(payload["schema"], "voicings.payload.v1");
    assert_eq!(payload["format"], "binary-positions+meta");

    for url_field in ["positions_url", "meta_url"] {
        let url = payload[url_field]
            .as_str()
            .unwrap_or_else(|| panic!("{url_field} must be a string"));
        assert!(url.starts_with("http"), "{url_field} should be an http URL, got {url}");
        assert!(!url.ends_with('/'), "{url_field} should not end with a slash");
    }

    let offset = payload["scene_offset"]
        .as_array()
        .expect("scene_offset is an array");
    assert_eq!(offset.len(), 3, "scene_offset must be [x,y,z]");
    assert_eq!(offset[0], 200.0, "default x offset clears the governance graph");

    assert!(
        payload["default_spread"].as_f64().unwrap() >= 0.0,
        "default_spread must be non-negative"
    );
    assert!(
        payload["default_point_size"].as_f64().unwrap() > 0.0,
        "default_point_size must be positive"
    );
}

/// Phase 5: serve_viz can serve a single voicing's full metadata from
/// `state/voicings/{instrument}-corpus.json` when the precomputed
/// details sample doesn't contain it. This test exercises the on-disk
/// half — does the corpus file exist and yield a parseable entry at
/// the requested index? — without speaking HTTP.
#[test]
fn corpus_lazy_lookup_resolves_when_present() {
    let bass_corpus = project_root().join("state/voicings/bass-corpus.json");
    if !bass_corpus.exists() {
        eprintln!("skipping: bass-corpus.json not present");
        return;
    }
    let bytes = std::fs::read(&bass_corpus).expect("read bass corpus");
    let entries: Vec<Value> = serde_json::from_slice(&bytes).expect("parse bass corpus");
    assert!(!entries.is_empty(), "corpus should have at least one voicing");
    let first = &entries[0];
    assert!(
        first.get("instrument").and_then(|v| v.as_str()) == Some("bass"),
        "first bass corpus entry should declare instrument=bass"
    );
    assert!(
        first.get("frets").is_some() || first.get("midiNotes").is_some(),
        "corpus entries must carry at least frets or midiNotes for the lookup to be useful"
    );
}

/// If the binary buffer + sidecar exist locally, verify they honour the
/// contract the payload promises. Skipped silently otherwise — CI
/// without OPTIC-K data shouldn't fail.
#[test]
fn binary_buffer_layout_matches_meta_when_present() {
    let bin_path = project_root().join("state/viz/voicing-positions.bin");
    let meta_path = project_root().join("state/viz/voicing-positions.meta.json");

    if !bin_path.exists() || !meta_path.exists() {
        eprintln!("skipping: voicing-positions.{{bin,meta.json}} not present (run serve_viz first)");
        return;
    }

    let meta_bytes = std::fs::read(&meta_path).expect("read meta");
    let meta: Value = serde_json::from_slice(&meta_bytes).expect("meta is JSON");
    let total = meta["total"].as_u64().expect("meta.total is integer") as usize;

    let bin_size = std::fs::metadata(&bin_path)
        .expect("stat bin")
        .len() as usize;
    let expected = total * 3 * 4;
    assert_eq!(
        bin_size, expected,
        "bin size ({bin_size}) must equal total ({total}) * 3 floats * 4 bytes ({expected})"
    );

    // Blocks must tile the buffer with no gaps or overlap.
    let blocks = meta["instruments"].as_array().expect("instruments array");
    let mut running = 0usize;
    for block in blocks {
        let name = block["name"].as_str().expect("name");
        let offset = block["offset"].as_u64().expect("offset") as usize;
        let count = block["count"].as_u64().expect("count") as usize;
        assert_eq!(
            offset, running,
            "{name} block offset ({offset}) must equal running cursor ({running}) — gap or overlap"
        );
        running += count;
    }
    assert_eq!(running, total, "sum of per-instrument counts ({running}) must equal total ({total})");
}
