//! Out-of-domain (OOD) query lens over GA chatbot **query embeddings**.
//!
//! Reads `state/quality/query-embeddings/YYYY-MM-DD.jsonl` — one row per routed
//! query, carrying the *exact* vector the router scored — into `query_embeddings`,
//! then scores each query by its **mean top-k cosine to its nearest neighbours**
//! (leave-one-out kNN density). A query that sits far from all others scores low →
//! out-of-domain. This is the raw-cosine method IX's ROC sweep validated; per-query
//! z-norm is deliberately NOT applied (it divides out the magnitude the gate keys on
//! — see `reference_ood_scoring_method_research`).
//!
//! Contract B **RATIFIED 2026-06-16**: GA's `QueryEmbeddingLog.cs` (ga #425) emits
//! snake_case keys (`[JsonPropertyName]`) matching this lens byte-for-byte —
//! `query_id, ts, query_text, intent, route_method, route_confidence, embedder, dim,
//! embedding`. `intent` is nullable (router declined); `embedder`/`dim` are dynamic
//! (today bge-large @ 1024) — the lens reads them per-row and is dimension-agnostic.
//! Producer is default-on; first rows land when the chatbot routes a live query.
//! Absent/empty directory → empty (never error).

use std::path::Path;

use duckdb::Connection;

use crate::source::{self, Col, Files};

/// Errors from the OOD lens — the shared artifact-source error
/// ([`crate::source::SourceError`]); aliased so the lens's public API keeps its name.
pub type OodError = source::SourceError;

/// `query_embeddings` column spec. Every field (including the `embedding DOUBLE[]` vector and
/// the derived `day` slice) is a flat [`Col::direct`] projection; the deep artifact-source
/// path owns the safe read + empty-fallback. `intent` is nullable per Contract B — the direct
/// `intent::VARCHAR` cast yields NULL on a JSON null, not an error.
const EMBEDDING_SPEC: &[Col] = &[
    Col::direct("query_id", "VARCHAR", "query_id::VARCHAR"),
    Col::direct("ts", "VARCHAR", "ts::VARCHAR"),
    Col::direct("day", "VARCHAR", "(ts::VARCHAR)[1:10]"),
    Col::direct("query_text", "VARCHAR", "query_text::VARCHAR"),
    Col::direct("intent", "VARCHAR", "intent::VARCHAR"),
    Col::direct("route_method", "VARCHAR", "route_method::VARCHAR"),
    Col::direct("route_confidence", "DOUBLE", "TRY_CAST(route_confidence AS DOUBLE)"),
    Col::direct("embedder", "VARCHAR", "embedder::VARCHAR"),
    Col::direct("dim", "BIGINT", "TRY_CAST(dim AS BIGINT)"),
    Col::direct("embedding", "DOUBLE[]", "embedding::DOUBLE[]"),
];

fn is_jsonl(name: &str) -> bool {
    name.ends_with(".jsonl")
}

/// Build `query_embeddings` (one row per routed query). Returns the row count;
/// 0 when the directory is absent/empty.
// @ai:invariant build_query_embeddings creates query_embeddings with one row per JSONL line, embedding as DOUBLE[] [T:test conf:0.9 src:ix_duck::ood::tests::rows_one_per_query]
pub fn build_query_embeddings(conn: &Connection, dir: &Path) -> Result<usize, OodError> {
    source::materialize(
        conn,
        "query_embeddings",
        Files { dir, matches: is_jsonl },
        EMBEDDING_SPEC,
    )
}

/// Mean top-`k` cosine to nearest neighbours, per **distinct** query embedding
/// (leave-one-out). `(query_id, intent, score)` ascending — most out-of-domain first.
///
/// **Deduplicated by embedding first** (verified on real Contract-B data: a 263-row day
/// held only 51 distinct queries — 5.2× replay). A query repeated more than `k` times is
/// otherwise its *own* nearest neighbours at cosine 1.0 → scores ~1.0 and **never flags**,
/// even when genuinely out-of-domain. Scoring over distinct embeddings makes the score
/// "distance to the distinct corpus" — the correct OOD question — and collapses the O(n²)
/// pair join to the distinct set (≈26× fewer pairs on that day). Each distinct embedding
/// keeps a representative `query_id`/`intent`.
///
/// Needs ≥2 **distinct** embeddings; fewer → empty. Uses the registered `ix_cosine` UDF,
/// so dimensions must be uniform (mismatch surfaces as a SQL error).
// @ai:invariant ood_scores dedups by embedding before scoring, so a query repeated >k times is not masked by its own copies — a duplicated out-of-domain query is still flagged [T:test conf:0.9 src:ix_duck::ood::tests::duplicates_dont_mask_ood]
pub fn ood_scores(conn: &Connection, k: i64) -> duckdb::Result<Vec<(String, String, f64)>> {
    let mut stmt = conn.prepare(
        "WITH distinct_q AS (
             SELECT any_value(query_id) AS query_id, any_value(intent) AS intent, embedding
             FROM query_embeddings GROUP BY embedding
         ),
         pairs AS (
             SELECT a.query_id, a.intent,
                    ix_cosine(a.embedding, b.embedding) AS sim
             FROM distinct_q a JOIN distinct_q b ON a.query_id <> b.query_id
         ),
         ranked AS (
             SELECT query_id, intent, sim,
                    row_number() OVER (PARTITION BY query_id ORDER BY sim DESC) AS rn
             FROM pairs
         )
         SELECT query_id, coalesce(any_value(intent), '(declined)') AS intent, avg(sim) AS score
         FROM ranked WHERE rn <= ?
         GROUP BY query_id ORDER BY score",
    )?;
    stmt.query_map([k], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect()
}

/// Queries whose mean top-`k` cosine falls **below** `threshold` — the OOD flags.
/// `(query_id, intent, score)` ascending.
pub fn flag_ood(
    conn: &Connection,
    k: i64,
    threshold: f64,
) -> duckdb::Result<Vec<(String, String, f64)>> {
    Ok(ood_scores(conn, k)?
        .into_iter()
        .filter(|(_, _, s)| *s < threshold)
        .collect())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/query-embeddings")
    }

    #[test]
    fn rows_one_per_query() {
        let conn = crate::open_bench().unwrap();
        let n = build_query_embeddings(&conn, &fixtures()).unwrap();
        // 4 in-domain (clustered) + 1 out-of-domain (orthogonal) = 5.
        assert_eq!(n, 5);
    }

    #[test]
    fn ood_query_scores_lowest() {
        let conn = crate::open_bench().unwrap();
        build_query_embeddings(&conn, &fixtures()).unwrap();
        let scores = ood_scores(&conn, 3).unwrap();
        assert_eq!(scores.len(), 5);
        // The orthogonal query sits farthest from the in-domain cluster → lowest score.
        assert_eq!(scores[0].0, "q-oos", "most-OOD query ranks first");
        assert!(
            scores[0].2 < scores[1].2,
            "OOS score strictly below the cluster"
        );
    }

    #[test]
    fn flag_ood_catches_only_the_outlier() {
        let conn = crate::open_bench().unwrap();
        build_query_embeddings(&conn, &fixtures()).unwrap();
        let flagged = flag_ood(&conn, 3, 0.5).unwrap();
        assert_eq!(flagged.len(), 1, "only the orthogonal query is below 0.5");
        assert_eq!(flagged[0].0, "q-oos");
    }

    #[test]
    fn absent_dir_degrades_to_empty() {
        let conn = crate::open_bench().unwrap();
        let n = build_query_embeddings(&conn, Path::new("/no/such/dir")).unwrap();
        assert_eq!(n, 0);
        assert!(ood_scores(&conn, 3).unwrap().is_empty());
        assert!(flag_ood(&conn, 3, 1.0).unwrap().is_empty());
    }

    #[test]
    fn single_row_has_no_neighbours() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("2026-06-16.jsonl"),
            "{\"query_id\":\"q1\",\"ts\":\"2026-06-16T00:00:00Z\",\"query_text\":\"x\",\"intent\":\"i\",\"route_method\":\"embedding\",\"route_confidence\":0.9,\"embedder\":\"bge-base-en-v1.5\",\"dim\":3,\"embedding\":[1.0,0.0,0.0]}\n",
        )
        .unwrap();
        let conn = crate::open_bench().unwrap();
        let n = build_query_embeddings(&conn, dir.path()).unwrap();
        assert_eq!(n, 1);
        assert!(
            ood_scores(&conn, 3).unwrap().is_empty(),
            "no neighbours to score against"
        );
    }

    /// Contract B emits a null `intent` when the router declines — that row must
    /// still score (not raise a DuckDB conversion error).
    #[test]
    fn null_intent_does_not_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("2026-06-16.jsonl"),
            "{\"query_id\":\"q1\",\"ts\":\"2026-06-16T00:00:00Z\",\"query_text\":\"x\",\"intent\":null,\"route_method\":\"fallback\",\"route_confidence\":0.2,\"embedder\":\"bge-large\",\"dim\":3,\"embedding\":[0.0,0.0,1.0]}\n\
             {\"query_id\":\"q2\",\"ts\":\"2026-06-16T00:01:00Z\",\"query_text\":\"y\",\"intent\":\"icv\",\"route_method\":\"embedding\",\"route_confidence\":0.9,\"embedder\":\"bge-large\",\"dim\":3,\"embedding\":[1.0,0.0,0.0]}\n",
        )
        .unwrap();
        let conn = crate::open_bench().unwrap();
        build_query_embeddings(&conn, dir.path()).unwrap();
        let scores = ood_scores(&conn, 3).unwrap();
        assert_eq!(scores.len(), 2, "both rows scored despite the null intent");
        let declined = scores.iter().find(|(id, _, _)| id == "q1").unwrap();
        assert_eq!(
            declined.1, "(declined)",
            "null intent surfaces as a label, not an error"
        );
    }

    /// A genuinely OOD query repeated more than `k` times would, without dedup, be its own
    /// nearest neighbours at cosine 1.0 and score ~1.0 — never flagging. Dedup-by-embedding
    /// fixes that: it is scored against the *distinct* corpus and flagged.
    #[test]
    fn duplicates_dont_mask_ood() {
        let dir = tempfile::tempdir().unwrap();
        let row = |id: &str, e: &str| {
            format!(
                "{{\"query_id\":\"{id}\",\"ts\":\"2026-06-16T00:00:00Z\",\"query_text\":\"{id}\",\"intent\":\"i\",\"route_method\":\"embedding\",\"route_confidence\":0.9,\"embedder\":\"x\",\"dim\":3,\"embedding\":{e}}}\n"
            )
        };
        let mut s = String::new();
        // 4 distinct in-domain queries, tightly clustered near [1,0,0].
        s.push_str(&row("a0", "[1.0,0.0,0.0]"));
        s.push_str(&row("a1", "[0.99,0.1,0.0]"));
        s.push_str(&row("a2", "[0.98,0.0,0.1]"));
        s.push_str(&row("a3", "[0.97,0.1,0.1]"));
        // One OOD query (orthogonal) repeated 4× — would self-mask without dedup.
        for i in 0..4 {
            s.push_str(&row(&format!("b{i}"), "[0.0,0.0,1.0]"));
        }
        std::fs::write(dir.path().join("2026-06-16.jsonl"), s).unwrap();

        let conn = crate::open_bench().unwrap();
        let total = build_query_embeddings(&conn, dir.path()).unwrap();
        assert_eq!(total, 8, "8 raw rows loaded");

        let scores = ood_scores(&conn, 3).unwrap();
        assert_eq!(
            scores.len(),
            5,
            "scored over 5 DISTINCT embeddings, not 8 raw rows"
        );

        // The duplicated OOD query is flagged despite its 4 identical copies.
        let flagged = flag_ood(&conn, 3, 0.5).unwrap();
        assert_eq!(flagged.len(), 1, "duplicated OOD query no longer masked");
        assert!(
            flagged[0].0.starts_with('b'),
            "the flagged query is the orthogonal one, got {}",
            flagged[0].0
        );
    }
}
