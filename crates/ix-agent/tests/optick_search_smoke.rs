//! `ix_optick_search` through the registry against a small generated OPTK v4
//! index (the real ~184 MB GA index is never needed in CI).
//!
//! Guards the green-but-dead failure where the tool schema advertised a
//! hardcoded query dimension the index no longer had: the tool must report
//! the dimension it actually requires (read from the index header) and say
//! both dimensions when a query does not match.

use ix_agent::tools::ToolRegistry;
use ix_optick::OptickIndex;
use serde_json::json;
use std::path::Path;

/// Header dimension written by the fixture. `OptickIndex::open` rejects any
/// other value, so the tests below read the dimension back through the reader
/// instead of trusting this constant.
const DIM: usize = 124;

/// Write a minimal OPTK v4 index: one voicing per instrument, each a unit
/// spike on a different axis. Layout mirrors ix-optick's own test builder.
fn write_index(path: &Path) {
    let voicings: Vec<Vec<f32>> = (0..3)
        .map(|i| {
            let mut v = vec![0.0f32; DIM];
            v[i] = 1.0;
            v
        })
        .collect();
    let count = voicings.len();

    let mut buf = Vec::new();
    buf.extend_from_slice(b"OPTK");
    buf.extend_from_slice(&4u32.to_le_bytes());
    let header_size_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&ix_optick::compute_schema_hash().to_le_bytes());
    buf.extend_from_slice(&0xFEFFu16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&(DIM as u32).to_le_bytes());
    buf.extend_from_slice(&(count as u64).to_le_bytes());
    buf.push(3u8);
    buf.extend_from_slice(&[0u8; 7]);
    let inst_pos = buf.len();
    buf.extend_from_slice(&[0u8; 48]);
    let tail_pos = buf.len(); // metadata_offsets, vectors, metadata offset, metadata length
    buf.extend_from_slice(&[0u8; 32]);
    for _ in 0..DIM {
        buf.extend_from_slice(&1.0f32.to_le_bytes());
    }
    let header_size = buf.len() as u32;
    buf[header_size_pos..header_size_pos + 4].copy_from_slice(&header_size.to_le_bytes());

    let offsets_table = buf.len();
    buf.extend_from_slice(&vec![0u8; count * 8]);

    let vectors_offset = buf.len() as u64;
    for v in &voicings {
        for x in v {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    for i in 0..3 {
        let p = inst_pos + i * 16;
        let off = vectors_offset + (i * DIM * 4) as u64;
        buf[p..p + 8].copy_from_slice(&off.to_le_bytes());
        buf[p + 8..p + 16].copy_from_slice(&1u64.to_le_bytes());
    }

    let metadata_offset = buf.len() as u64;
    for (i, inst) in ["guitar", "bass", "ukulele"].iter().enumerate() {
        let rel = buf.len() as u64 - metadata_offset;
        let p = offsets_table + i * 8;
        buf[p..p + 8].copy_from_slice(&rel.to_le_bytes());
        let meta = json!({
            "diagram": format!("V{i}"),
            "instrument": inst,
            "midiNotes": [60 + i as i32],
            "quality_inferred": null,
        });
        buf.extend_from_slice(&rmp_serde::to_vec(&meta).unwrap());
    }
    let metadata_length = buf.len() as u64 - metadata_offset;

    for (k, value) in [
        offsets_table as u64,
        vectors_offset,
        metadata_offset,
        metadata_length,
    ]
    .iter()
    .enumerate()
    {
        let p = tail_pos + k * 8;
        buf[p..p + 8].copy_from_slice(&value.to_le_bytes());
    }

    std::fs::write(path, buf).unwrap();
}

#[test]
fn reported_dimension_is_the_index_header_dimension() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("optick.index");
    write_index(&path);
    let dim = OptickIndex::open(&path).unwrap().dimension() as usize;

    let mut query = vec![0.0f64; dim];
    query[1] = 1.0;
    let out = ToolRegistry::new()
        .call(
            "ix_optick_search",
            json!({ "query": query, "top_k": 2, "index_path": path.to_str().unwrap() }),
        )
        .expect("a query of the index's dimension must succeed");

    assert_eq!(out["index_dimension"], dim);
    assert_eq!(out["count"], 2);
    assert_eq!(out["results"][0]["diagram"], "V1");
    assert_eq!(out["results"][0]["instrument"], "bass");
}

#[test]
fn dimension_mismatch_error_states_both_dimensions() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("optick.index");
    write_index(&path);
    let dim = OptickIndex::open(&path).unwrap().dimension() as usize;

    // The dimension the schema used to advertise.
    let err = ToolRegistry::new()
        .call(
            "ix_optick_search",
            json!({ "query": vec![0.5f64; 228], "index_path": path.to_str().unwrap() }),
        )
        .expect_err("a wrong-length query must be rejected");

    assert!(
        err.contains(&format!("got 228, expected {dim}")),
        "error must state both dimensions, got: {err}"
    );
}

#[test]
fn schema_does_not_hardcode_a_query_dimension() {
    let listing = ToolRegistry::new().list();
    let tool = listing["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "ix_optick_search")
        .expect("ix_optick_search registered");
    let text = tool.to_string();
    assert!(
        !text.contains("-dim"),
        "schema hardcodes a dimension: {text}"
    );
}
