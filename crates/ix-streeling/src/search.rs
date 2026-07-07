//! BM25 full-text ranking over the learnings catalog.
//!
//! Compounding-KB Slice 1 ("retrieve-before-re-solve"): rank the ecosystem's
//! `LearningRecord`s by Okapi BM25 so the agent can *find* a prior solution
//! whose wording differs from the query, instead of re-solving it. Frontmatter
//! (`repo` / `kind` / `category` / `tags`) is applied as a **hard pre-filter**
//! (a WHERE clause), and BM25 ranks *within* the filtered set.
//!
//! Pure Rust, no dependencies — the corpus is ~90 short docs, so a hand-written
//! ranker is the right amount of machinery (see the plan's Slice 1 note:
//! "hybrid vector retrieval earns its keep" only at 500–5,000 notes).

use crate::model::{Kind, LearningRecord};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Okapi BM25 term-frequency saturation.
const K1: f64 = 1.2;
/// Okapi BM25 length-normalization.
const B: f64 = 0.75;

/// Hard pre-filter on frontmatter fields, applied BEFORE ranking. A record is
/// eligible only if it matches **every** provided constraint (`None` / empty
/// means "don't constrain on this field"). String matches are ASCII
/// case-insensitive; every requested tag must be present.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchFilter {
    /// Restrict to one originating repo (`ix`, `ga`, `tars`, `Demerzel`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Restrict to one artifact kind.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<Kind>,
    /// Restrict to one category / faculty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    /// Record must contain every one of these tags.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl SearchFilter {
    /// Whether `r` passes every constraint in this filter.
    fn matches(&self, r: &LearningRecord) -> bool {
        if let Some(repo) = &self.repo {
            if !r.repo.eq_ignore_ascii_case(repo) {
                return false;
            }
        }
        if let Some(kind) = self.kind {
            if r.kind != kind {
                return false;
            }
        }
        if let Some(cat) = &self.category {
            if !r.category.eq_ignore_ascii_case(cat) {
                return false;
            }
        }
        self.tags
            .iter()
            .all(|want| r.tags.iter().any(|t| t.eq_ignore_ascii_case(want)))
    }
}

/// Tokenize by lowercasing and splitting on any non-alphanumeric character.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// The per-record document text: `title` + `tags` + `symptom` + `root_cause` +
/// `category`, skipping absent fields. This carries the frontmatter into the
/// indexed unit (per the plan's chunking note).
fn document_tokens(r: &LearningRecord) -> Vec<String> {
    let mut text = String::new();
    text.push_str(&r.title);
    text.push(' ');
    for t in &r.tags {
        text.push_str(t);
        text.push(' ');
    }
    if let Some(s) = &r.symptom {
        text.push_str(s);
        text.push(' ');
    }
    if let Some(rc) = &r.root_cause {
        text.push_str(rc);
        text.push(' ');
    }
    text.push_str(&r.category);
    tokenize(&text)
}

/// BM25 IDF, Lucene form `ln(1 + (N - df + 0.5) / (df + 0.5))` — always
/// non-negative, even for terms appearing in more than half the corpus.
fn idf(n_docs: f64, doc_freq: f64) -> f64 {
    (1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)).ln()
}

/// Rank `records` against `query` by Okapi BM25 (k1=1.2, b=0.75).
///
/// `filter` is applied FIRST as a hard pre-filter; BM25 statistics (IDF, average
/// document length) are computed over the surviving set, which is also what gets
/// ranked. Returns up to `top_k` `(record, score)` pairs, highest score first,
/// dropping zero-score (no query-term overlap) records.
pub fn search<'a>(
    records: &'a [LearningRecord],
    query: &str,
    filter: &SearchFilter,
    top_k: usize,
) -> Vec<(&'a LearningRecord, f64)> {
    let filtered: Vec<&LearningRecord> = records.iter().filter(|r| filter.matches(r)).collect();
    if filtered.is_empty() {
        return Vec::new();
    }

    let docs: Vec<Vec<String>> = filtered.iter().map(|r| document_tokens(r)).collect();
    let n = docs.len() as f64;
    let total_len: usize = docs.iter().map(Vec::len).sum();
    let avgdl = (total_len as f64 / n).max(1.0);

    // Document frequency per term (each term counted once per doc).
    let mut df: HashMap<&str, usize> = HashMap::new();
    for doc in &docs {
        let mut seen: HashSet<&str> = HashSet::new();
        for tok in doc {
            if seen.insert(tok.as_str()) {
                *df.entry(tok.as_str()).or_insert(0) += 1;
            }
        }
    }

    let query_terms = tokenize(query);

    let mut scored: Vec<(&LearningRecord, f64)> = filtered
        .iter()
        .zip(docs.iter())
        .map(|(&r, doc)| {
            let dl = doc.len() as f64;
            let mut tf: HashMap<&str, usize> = HashMap::new();
            for tok in doc {
                *tf.entry(tok.as_str()).or_insert(0) += 1;
            }
            let mut score = 0.0;
            for term in &query_terms {
                let f = *tf.get(term.as_str()).unwrap_or(&0) as f64;
                if f == 0.0 {
                    continue;
                }
                let dfq = *df.get(term.as_str()).unwrap_or(&0) as f64;
                let denom = f + K1 * (1.0 - B + B * dl / avgdl);
                score += idf(n, dfq) * (f * (K1 + 1.0)) / denom;
            }
            (r, score)
        })
        .filter(|(_, s)| *s > 0.0)
        .collect();

    // Stable sort by descending score keeps input order as the tie-break.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

/// One labeled ground-truth pair from a `retrieval-eval.jsonl` fixture: a
/// natural-language `query` and the `expected_id` (`"{repo}:{path}"`) of the
/// prior doc it should retrieve, with an optional frontmatter `filter` applied
/// exactly as a live `streeling search` call would.
#[derive(Debug, Clone, Deserialize)]
pub struct EvalPair {
    /// Natural-language query — a paraphrase of the target doc's symptom/title.
    pub query: String,
    /// Ground-truth id the query should retrieve (`"{repo}:{path}"`).
    pub expected_id: String,
    /// Frontmatter pre-filter to apply for this query (default: none).
    #[serde(default)]
    pub filter: SearchFilter,
}

/// Per-query outcome in an [`EvalReport`].
#[derive(Debug, Clone, Serialize)]
pub struct PerQuery {
    pub query: String,
    pub expected_id: String,
    /// 1-based rank of `expected_id` in the ranked hit list, or `null` if the
    /// target never surfaced (missed).
    pub found_rank: Option<usize>,
    pub filter: SearchFilter,
}

/// A committed, re-runnable retrieval-quality report — recall@k + MRR over a
/// labeled eval set, plus per-query detail for auditing regressions.
#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    /// Report schema tag (`streeling-retrieval-eval-v1`).
    pub schema: String,
    /// Date/timestamp the run was stamped with (caller-supplied for determinism).
    pub generated_at: String,
    /// The `k` used for recall@k.
    pub k: usize,
    pub n_queries: usize,
    /// Fraction of queries whose target landed at rank ≤ k.
    pub recall_at_k: f64,
    /// Mean reciprocal rank (1/rank, or 0 for a miss), averaged over all queries.
    pub mrr: f64,
    pub per_query: Vec<PerQuery>,
}

/// The retrieval depth scanned per query when locating the expected id. Ranks
/// beyond this are treated as misses (`found_rank = None`); kept well above any
/// realistic `k` so MRR is not truncated at the recall cutoff.
const EVAL_SCAN_DEPTH: usize = 50;

/// Run the recall@k / MRR measurement of [`search`] over a labeled eval set.
///
/// For each pair, runs `search` with that pair's filter, finds the 1-based rank
/// of `expected_id` in the ranked hits (scanning up to [`EVAL_SCAN_DEPTH`]),
/// then aggregates **recall@k** (share of targets at rank ≤ `k`) and **MRR**
/// (mean of `1/rank`, 0 for a miss). `generated_at` is caller-supplied so the
/// run is deterministic/testable — this function never reads the clock.
pub fn evaluate(
    records: &[LearningRecord],
    pairs: &[EvalPair],
    k: usize,
    generated_at: &str,
) -> EvalReport {
    let scan_depth = EVAL_SCAN_DEPTH.max(k);
    let mut per_query = Vec::with_capacity(pairs.len());
    let mut hits_at_k = 0usize;
    let mut reciprocal_sum = 0.0;

    for pair in pairs {
        let hits = search(records, &pair.query, &pair.filter, scan_depth);
        let found_rank = hits
            .iter()
            .position(|(r, _)| r.id == pair.expected_id)
            .map(|i| i + 1);
        if let Some(rank) = found_rank {
            if rank <= k {
                hits_at_k += 1;
            }
            reciprocal_sum += 1.0 / rank as f64;
        }
        per_query.push(PerQuery {
            query: pair.query.clone(),
            expected_id: pair.expected_id.clone(),
            found_rank,
            filter: pair.filter.clone(),
        });
    }

    let n = pairs.len();
    let (recall_at_k, mrr) = if n == 0 {
        (0.0, 0.0)
    } else {
        (hits_at_k as f64 / n as f64, reciprocal_sum / n as f64)
    };

    EvalReport {
        schema: "streeling-retrieval-eval-v1".to_string(),
        generated_at: generated_at.to_string(),
        k,
        n_queries: n,
        recall_at_k,
        mrr,
        per_query,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test-corpus constructor mirrors LearningRecord's fields 1:1; the arg count
    // is inherent to the record shape, not a design smell.
    #[allow(clippy::too_many_arguments)]
    fn rec(
        id: &str,
        repo: &str,
        kind: Kind,
        category: &str,
        title: &str,
        tags: &[&str],
        symptom: Option<&str>,
        root_cause: Option<&str>,
    ) -> LearningRecord {
        LearningRecord {
            schema_version: crate::model::SCHEMA_VERSION.to_string(),
            id: id.to_string(),
            repo: repo.to_string(),
            kind,
            category: category.to_string(),
            title: title.to_string(),
            date: None,
            tags: tags.iter().map(|t| t.to_string()).collect(),
            symptom: symptom.map(str::to_string),
            root_cause: root_cause.map(str::to_string),
            path: id.split_once(':').map(|(_, p)| p).unwrap_or(id).to_string(),
        }
    }

    /// A hermetic mini-corpus modeled on real `state/streeling/catalog.jsonl`
    /// entries (same ids/titles/tags/symptoms), so the labeled queries below are
    /// realistic paraphrases of prior solutions.
    fn corpus() -> Vec<LearningRecord> {
        vec![
            rec(
                "ix:docs/solutions/build-errors/windows-app-control-blocks-cargo-test-binaries.md",
                "ix", Kind::Solution, "build-errors",
                "Windows Application Control blocks cargo test/build-script binaries (OS error 4551)",
                &["windows", "app-control", "wdac", "cargo", "rust", "test-execution"],
                Some("cargo test fails with 'An Application Control policy has blocked this file. (os error 4551)' on freshly compiled test binaries"),
                Some("Windows Defender Application Control (WDAC) policy blocks execution of unsigned binaries under the target directory"),
            ),
            rec(
                "ix:docs/solutions/build-errors/windows-lnk1318-pdb-size-limit.md",
                "ix", Kind::Solution, "build-errors",
                "Windows LNK1318 PDB size limit when linking test binaries with many features",
                &["windows", "msvc", "linker", "pdb", "cargo-features", "rust"],
                Some("LINK : fatal error LNK1318: Unexpected PDB error; LIMIT (12)"),
                Some("MSVC linker PDB size limit exceeded when building a test binary with all feature-gated modules"),
            ),
            rec(
                "ix:docs/solutions/math-correctness/jacobi-vs-power-iteration-repeated-eigenvalues.md",
                "ix", Kind::Solution, "math-correctness",
                "Jacobi vs power iteration for matrices with repeated eigenvalues",
                &["eigenvalue", "jacobi", "power-iteration", "linear-algebra", "deflation"],
                Some("Classical MDS of a unit square returned pairwise-distance error 1.14 instead of 1e-6"),
                Some("Power iteration plus rank-1 deflation converges to an arbitrary vector in the dominant eigenspace when eigenvalues are repeated"),
            ),
            rec(
                "ix:docs/solutions/math-correctness/lda-non-symmetric-deflation.md",
                "ix", Kind::Solution, "math-correctness",
                "LDA non-symmetric deflation produces duplicated or wrong discriminants",
                &["lda", "generalized-eigenvalue", "deflation", "symmetrization"],
                Some("LDA with three axis-aligned class clusters returned duplicated components"),
                Some("Non-symmetric deflation of the generalized eigenproblem needs a symmetrization step to avoid duplicated discriminants"),
            ),
            rec(
                "ix:docs/solutions/feature-implementations/2026-06-19-duckdb-absence-as-zero-and-struct-bind-crash.md",
                "ix", Kind::Solution, "feature-implementations",
                "DuckDB over read_json_auto: absence-as-zero and struct-field bind-crashes",
                &["duckdb", "read_json_auto", "json_extract", "null", "coalesce", "struct"],
                Some("ix-duck lenses over GA JSON artifacts reported 0 for metrics that were never measured"),
                Some("coalesce(optional_metric, 0) conflates absent with zero; use null-aware extraction instead"),
            ),
            rec(
                "ix:docs/solutions/feature-implementations/2026-06-16-flashassign-fusion-loses-at-ix-scale.md",
                "ix", Kind::Solution, "feature-implementations",
                "FlashAssign kernel fusion loses to materialize-then-argmin at IX scale",
                &["gpu", "wgpu", "kmeans", "flash-kmeans", "performance", "io-aware"],
                Some("Wanted to know if Flash-KMeans FlashAssign would speed up GPU kmeans by fusing distance and argmin"),
                Some("On a consumer GPU at IX data scale, memory bandwidth is not the bottleneck, so materializing the matrix wins"),
            ),
            rec(
                "ga:docs/solutions/architecture/2026-05-07-di-composition-root-casing-drift.md",
                "ga", Kind::Solution, "architecture",
                "Two casing-variant DI extension methods drift apart silently",
                &["csharp", "dependency-injection", "extension-methods", "composition-root", "naming-conventions"],
                Some("AddGuitarAlchemistAi and AddGuitarAlchemistAI diverged; one registration path went stale"),
                Some("Two casing-variant extension methods on the DI composition root drifted apart with no compiler error"),
            ),
            rec(
                "ga:docs/solutions/integration-issues/2026-03-07-yaml-knowledge-loader-rag-pipeline-integration.md",
                "ga", Kind::Solution, "integration-issues",
                "Connect curated music theory YAML files to the RAG pipeline via a generic knowledge loader",
                &["yaml", "rag", "knowledge-base", "fsharp", "mongodb", "embeddings"],
                Some("Roughly 20 curated music theory YAML files were never reaching the RAG pipeline"),
                Some("Only the domain-typed F# loaders were wired up; the generic YAML knowledge loader was missing"),
            ),
        ]
    }

    /// Labeled (query, expected-id) pairs — each query paraphrases a prior
    /// solution's symptom/topic without quoting the id.
    fn labeled() -> Vec<(&'static str, &'static str)> {
        vec![
            (
                "cargo test blocked by windows application control os error 4551 unsigned binary",
                "ix:docs/solutions/build-errors/windows-app-control-blocks-cargo-test-binaries.md",
            ),
            (
                "msvc linker pdb size limit lnk1318 when building test binary with many features",
                "ix:docs/solutions/build-errors/windows-lnk1318-pdb-size-limit.md",
            ),
            (
                "power iteration rank-1 deflation repeated eigenvalues wrong eigenvector",
                "ix:docs/solutions/math-correctness/jacobi-vs-power-iteration-repeated-eigenvalues.md",
            ),
            (
                "lda duplicated discriminants symmetrization generalized eigenproblem deflation",
                "ix:docs/solutions/math-correctness/lda-non-symmetric-deflation.md",
            ),
            (
                "duckdb coalesce treats absent metric as zero null-aware extraction",
                "ix:docs/solutions/feature-implementations/2026-06-19-duckdb-absence-as-zero-and-struct-bind-crash.md",
            ),
            (
                "flash kmeans gpu kernel fusion memory bandwidth not the bottleneck",
                "ix:docs/solutions/feature-implementations/2026-06-16-flashassign-fusion-loses-at-ix-scale.md",
            ),
            (
                "csharp dependency injection extension method casing drift composition root",
                "ga:docs/solutions/architecture/2026-05-07-di-composition-root-casing-drift.md",
            ),
            (
                "curated music theory yaml files not reaching rag pipeline knowledge loader",
                "ga:docs/solutions/integration-issues/2026-03-07-yaml-knowledge-loader-rag-pipeline-integration.md",
            ),
        ]
    }

    /// 1-based rank of `expected` in the hit list, or `None` if absent.
    fn rank_of(hits: &[(&LearningRecord, f64)], expected: &str) -> Option<usize> {
        hits.iter().position(|(r, _)| r.id == expected).map(|i| i + 1)
    }

    #[test]
    fn recall_at_k_and_mrr_on_labeled_fixture() {
        let corpus = corpus();
        let pairs = labeled();
        let k = 3;

        let mut hits_at_k = 0usize;
        let mut reciprocal_sum = 0.0;

        for (query, expected) in &pairs {
            let results = search(&corpus, query, &SearchFilter::default(), 10);
            let rank = rank_of(&results, expected);
            if let Some(r) = rank {
                if r <= k {
                    hits_at_k += 1;
                }
                reciprocal_sum += 1.0 / r as f64;
            }
            // Every labeled query must at least retrieve its target as rank 1.
            assert_eq!(
                rank,
                Some(1),
                "query {query:?} should retrieve {expected} at rank 1, got rank {rank:?}"
            );
        }

        let recall_at_3 = hits_at_k as f64 / pairs.len() as f64;
        let mrr = reciprocal_sum / pairs.len() as f64;

        assert_eq!(recall_at_3, 1.0, "recall@3 must be 1.0 on the labeled fixture, got {recall_at_3}");
        assert!(mrr > 0.9, "MRR must exceed 0.9 on the labeled fixture, got {mrr}");
    }

    #[test]
    fn filter_is_a_hard_pre_filter() {
        let corpus = corpus();
        // Query hits both ix build-errors docs and (weakly) nothing in ga, but
        // constraining repo=ga must exclude every ix record regardless of score.
        let filter = SearchFilter {
            repo: Some("ga".to_string()),
            ..Default::default()
        };
        let hits = search(&corpus, "windows cargo test binary linker", &filter, 10);
        assert!(
            hits.iter().all(|(r, _)| r.repo == "ga"),
            "repo filter must exclude non-ga records"
        );
    }

    #[test]
    fn tag_filter_requires_all_tags() {
        let corpus = corpus();
        let filter = SearchFilter {
            tags: vec!["deflation".to_string(), "lda".to_string()],
            ..Default::default()
        };
        let hits = search(&corpus, "eigenvalue deflation", &filter, 10);
        assert_eq!(hits.len(), 1, "only the LDA record carries both tags");
        assert!(hits[0].0.id.contains("lda-non-symmetric-deflation"));
    }

    #[test]
    fn empty_query_and_no_match_return_nothing() {
        let corpus = corpus();
        assert!(search(&corpus, "", &SearchFilter::default(), 5).is_empty());
        assert!(search(&corpus, "zzzznonexistenttoken", &SearchFilter::default(), 5).is_empty());
    }

    #[test]
    fn evaluate_computes_recall_and_mrr() {
        let corpus = corpus();
        // One pair whose target is retrievable at rank 1 (contributes 1.0 to MRR
        // and hits recall@k) and one deliberate miss (no query-term overlap AND a
        // non-existent id) that must contribute 0 to both.
        let pairs = vec![
            EvalPair {
                query: "windows application control blocked cargo test os error 4551".to_string(),
                expected_id:
                    "ix:docs/solutions/build-errors/windows-app-control-blocks-cargo-test-binaries.md"
                        .to_string(),
                filter: SearchFilter::default(),
            },
            EvalPair {
                query: "zzzznonexistenttoken".to_string(),
                expected_id: "ix:does/not/exist.md".to_string(),
                filter: SearchFilter::default(),
            },
        ];

        let report = evaluate(&corpus, &pairs, 5, "2026-07-07");

        assert_eq!(report.schema, "streeling-retrieval-eval-v1");
        assert_eq!(report.generated_at, "2026-07-07");
        assert_eq!(report.k, 5);
        assert_eq!(report.n_queries, 2);
        assert_eq!(report.per_query[0].found_rank, Some(1));
        assert_eq!(report.per_query[1].found_rank, None);
        // recall@5 = 1 hit / 2 queries; MRR = (1/1 + 0) / 2.
        assert!((report.recall_at_k - 0.5).abs() < 1e-9, "recall {}", report.recall_at_k);
        assert!((report.mrr - 0.5).abs() < 1e-9, "mrr {}", report.mrr);
    }

    #[test]
    fn evaluate_over_labeled_fixture_is_perfect() {
        // The whole hermetic labeled set should be recall@1 = 1.0, MRR = 1.0.
        let corpus = corpus();
        let pairs: Vec<EvalPair> = labeled()
            .into_iter()
            .map(|(q, id)| EvalPair {
                query: q.to_string(),
                expected_id: id.to_string(),
                filter: SearchFilter::default(),
            })
            .collect();
        let report = evaluate(&corpus, &pairs, 1, "2026-07-07");
        assert_eq!(report.recall_at_k, 1.0, "recall@1 over labeled fixture");
        assert!((report.mrr - 1.0).abs() < 1e-9, "MRR over labeled fixture = {}", report.mrr);
    }

    #[test]
    fn committed_fixture_parses_and_is_well_formed() {
        // The committed eval fixture must deserialize and every expected_id must
        // look like a "{repo}:{path}" id — guards against fixture bit-rot.
        let raw = include_str!("../tests/fixtures/retrieval-eval.jsonl");
        let pairs: Vec<EvalPair> = raw
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<EvalPair>(l).expect("fixture line parses"))
            .collect();
        assert!(pairs.len() >= 8, "expected ≥8 labeled pairs, got {}", pairs.len());
        for p in &pairs {
            assert!(!p.query.trim().is_empty(), "empty query in fixture");
            assert!(
                p.expected_id.contains(':') && p.expected_id.contains('/'),
                "expected_id {:?} is not a repo:path id",
                p.expected_id
            );
        }
    }
}
