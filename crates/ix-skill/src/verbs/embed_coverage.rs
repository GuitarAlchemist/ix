//! Optional in-process embedding coverage scorer (Cargo feature `embeddings`).
//!
//! Scores natural-language-request ↔ skill-catalog relevance with
//! **bge-base-en-v1.5** via in-process `fastembed-rs` (ONNX Runtime), replacing
//! the lexical TF-IDF pre-gate when the `embeddings` feature is built. Chosen by
//! the 2026-06-07 validation sweep (AUC 0.936, OOD TNR 0.625 @ recall 0.99 —
//! 3.8× the TF-IDF baseline; near-miss 0.036 → 0.482). Full results:
//! `state/thinking-machine/embedding-sweep-rust-results.md`.
//!
//! Inference-only pretrained tooling behind an optional-dep boundary per the
//! CLAUDE.md ML-framework carve-out; the pure-Rust TF-IDF gate in `compile.rs`
//! stays the default + graceful fallback. This module degrades (returns
//! `None`/`0.0`) on any failure rather than hard-failing the gate.

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::cmp::Ordering;

/// bge-v1.5 s2p-retrieval query instruction (passages get no prefix).
const BGE_QUERY_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
const MODEL_TAG: &str = "bge-base-en-v1.5";

pub struct EmbeddingCoverage {
    model: TextEmbedding,
    /// (skill name, passage embedding) — cosine normalizes, so raw is fine.
    catalog: Vec<(String, Vec<f32>)>,
}

impl EmbeddingCoverage {
    /// Load bge-base-en-v1.5 + the catalog passage embeddings (disk-cached by
    /// model + catalog-content hash). Returns `None` on any failure (model
    /// download/init or embedding) so the caller falls back to TF-IDF.
    pub fn load(skills: &[&'static ix_registry::SkillDescriptor]) -> Option<Self> {
        let mut model =
            TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGEBaseENV15)).ok()?;
        let passages: Vec<String> = skills
            .iter()
            .map(|s| format!("{} {}", s.name.replace(['.', '_'], " "), s.doc))
            .collect();
        let embeddings = catalog_embeddings(&mut model, passages)?;
        if embeddings.len() != skills.len() {
            return None;
        }
        let catalog = skills
            .iter()
            .map(|s| s.name.to_string())
            .zip(embeddings)
            .collect();
        Some(Self { model, catalog })
    }

    /// Coverage score for a request: the mean of the top-3 cosine similarities
    /// to the catalog, plus the top-3 nearest skills (for an auditable verdict).
    pub fn score(&mut self, sentence: &str) -> (f64, Vec<(String, f64)>) {
        let query = format!("{BGE_QUERY_INSTRUCTION}{sentence}");
        let emb = match self.model.embed(vec![query], None) {
            Ok(mut e) if !e.is_empty() => e.remove(0),
            _ => return (0.0, Vec::new()),
        };
        let mut scored: Vec<(String, f64)> = self
            .catalog
            .iter()
            .map(|(name, p)| (name.clone(), cosine(&emb, p)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let top3 = scored.iter().take(3).map(|x| x.1).sum::<f64>() / 3.0;
        scored.truncate(3);
        (top3, scored)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Embed the catalog passages, caching to disk keyed by (model, catalog-content
/// hash) so a static catalog is embedded only once across CLI invocations.
fn catalog_embeddings(model: &mut TextEmbedding, passages: Vec<String>) -> Option<Vec<Vec<f32>>> {
    let key = blake3::hash(passages.join("\n").as_bytes()).to_hex();
    let cache = std::path::Path::new("state/thinking-machine/embed-cache")
        .join(format!("{MODEL_TAG}.{}.bin", &key[..16]));
    let n = passages.len();
    if let Some(v) = read_cache(&cache) {
        if v.len() == n {
            return Some(v);
        }
    }
    let embeddings = model.embed(passages, None).ok()?;
    write_cache(&cache, &embeddings);
    Some(embeddings)
}

/// Binary cache layout: `[count u32 LE][dim u32 LE][f32 LE × count*dim]`.
fn read_cache(path: &std::path::Path) -> Option<Vec<Vec<f32>>> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() < 8 {
        return None;
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    if dim == 0 || bytes.len() != 8 + count * dim * 4 {
        return None;
    }
    let mut out = Vec::with_capacity(count);
    let mut off = 8;
    for _ in 0..count {
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            row.push(f32::from_le_bytes(bytes[off..off + 4].try_into().ok()?));
            off += 4;
        }
        out.push(row);
    }
    Some(out)
}

fn write_cache(path: &std::path::Path, embeddings: &[Vec<f32>]) {
    let dim = embeddings.first().map(|e| e.len()).unwrap_or(0);
    if dim == 0 || embeddings.iter().any(|e| e.len() != dim) {
        return;
    }
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let mut buf = Vec::with_capacity(8 + embeddings.len() * dim * 4);
    buf.extend_from_slice(&(embeddings.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(dim as u32).to_le_bytes());
    for row in embeddings {
        for &v in row {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    let _ = std::fs::write(path, buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_round_trips_and_rejects_corrupt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.bin");
        let emb = vec![vec![0.1f32, 0.2, 0.3], vec![-0.4, 0.5, 0.6]];
        write_cache(&path, &emb);
        assert_eq!(read_cache(&path).expect("round trip"), emb);
        // A truncated cache must be rejected (→ recompute), not mis-read.
        std::fs::write(&path, b"\x02\x00\x00\x00\x03").unwrap();
        assert!(read_cache(&path).is_none());
    }

    #[test]
    fn embedding_ranks_in_domain_above_out_of_domain() {
        // Real model load (downloads bge-base-en on first run); proves the
        // scorer separates an in-domain request from an out-of-domain one.
        let skills = crate::verbs::compile::pipeline_callable_skills();
        let Some(mut ec) = EmbeddingCoverage::load(&skills) else {
            eprintln!("SKIP: bge-base-en unavailable (offline / no ONNX)");
            return;
        };
        let (in_dom, _) = ec.score("compute the mean and standard deviation of these numbers");
        let (out_dom, _) = ec.score("teleport to the moon and bake a chocolate cake");
        eprintln!("embedding coverage: in={in_dom:.3} out={out_dom:.3}");
        assert!(
            in_dom > out_dom,
            "in-domain {in_dom} must exceed OOD {out_dom}"
        );
    }
}
