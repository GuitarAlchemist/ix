//! `ga-chatbot` — Rust QA harness + MCP bridge for the GA chatbot.
//!
//! **Naming note:** despite the crate name, this is **not** the production
//! Guitar Alchemist chatbot. The user-facing chatbot lives in the `ga` repo
//! (`Apps/ga-server/GaApi/Controllers/NebulaChatController.cs` and friends).
//! This crate exists to *test* the GA chatbot adversarially using IX
//! primitives — `ix-sanitize` (prompt injection), `ix-bracelet` (pitch-class
//! algebra grounding), `ix-game::nash` (Shapley judge aggregation), and
//! `ix-fuzzy` (adversarial corpus). It also ships a Rust MCP bridge that
//! spawns the GA + IX MCP servers as children and proxies tool calls.
//!
//! It is named `ga-chatbot` because every test it runs targets the GA
//! chatbot. Rename to `ix-chatbot-harness` if the boundary ever causes real
//! confusion — workspace-internal name, no published consumers.
//!
//! A stub MCP server that answers grounded questions about chord voicings on
//! guitar, bass, and ukulele. Every voicing cited must resolve to a real row
//! in the corpus. The QA harness layers deterministic checks (sanitization,
//! corpus grounding, confidence thresholds) before expensive LLM judges.

pub mod aggregate;
pub mod algebra;
pub mod mcp_bridge;
pub mod qa;
pub mod shapley;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Supported instruments.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Instrument {
    Guitar,
    Bass,
    Ukulele,
}

/// A question to the chatbot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatbotRequest {
    /// Natural language question about chord voicings.
    pub question: String,
    /// Target instrument. If omitted, defaults to guitar.
    pub instrument: Option<Instrument>,
}

/// A source citation backing a voicing reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    /// File path to the corpus artifact.
    pub path: String,
    /// Row index in the corpus JSON array.
    pub row: usize,
}

/// The chatbot's response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatbotResponse {
    /// Natural language answer grounded in corpus data.
    pub answer: String,
    /// Corpus voicing IDs cited in the answer (e.g., "guitar_v042").
    pub voicing_ids: Vec<String>,
    /// Alignment-policy confidence score (0.0-1.0).
    pub confidence: f64,
    /// File paths and row numbers backing each cited voicing.
    pub sources: Vec<Source>,
}

/// A fixture entry mapping a prompt prefix to a canned response.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FixtureEntry {
    prompt_prefix: String,
    response: ChatbotResponse,
}

/// Load canned stub responses from a JSONL fixture file.
///
/// Each line is a JSON object with `prompt_prefix` and `response` fields.
/// Returns a map from lowercase prompt prefix to the canned response.
pub fn load_fixtures(path: &Path) -> HashMap<String, ChatbotResponse> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };
    let mut map = HashMap::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<FixtureEntry>(line) {
            map.insert(entry.prompt_prefix.to_lowercase(), entry.response);
        }
    }
    map
}

/// Answer a question using canned stub responses.
///
/// Looks up `req.question` by fuzzy prefix match against the fixture map.
/// If no match is found, returns a refusal with confidence 0.0.
pub fn ask_stub(
    req: &ChatbotRequest,
    fixtures: &HashMap<String, ChatbotResponse>,
) -> ChatbotResponse {
    let question_lower = req.question.to_lowercase();

    // Try fuzzy prefix match: find the longest fixture prefix that matches
    let mut best_match: Option<&ChatbotResponse> = None;
    let mut best_len = 0;

    for (prefix, response) in fixtures {
        if question_lower.starts_with(prefix) && prefix.len() > best_len {
            best_match = Some(response);
            best_len = prefix.len();
        }
    }

    match best_match {
        Some(response) => response.clone(),
        None => ChatbotResponse {
            answer: "I don't have enough information to answer that.".to_string(),
            voicing_ids: vec![],
            confidence: 0.0,
            sources: vec![],
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_fixture_file() -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        writeln!(f, r#"{{"prompt_prefix": "does voicing guitar_v042", "response": {{"answer": "Yes, guitar_v042 is a dyad voicing at frets x-10-8-x-x-x.", "voicing_ids": ["guitar_v042"], "confidence": 0.92, "sources": [{{"path": "state/voicings/guitar-corpus.json", "row": 42}}]}}}}"#).unwrap();
        writeln!(f, r#"{{"prompt_prefix": "show me voicing guitar_v000", "response": {{"answer": "guitar_v000 is a dyad at frets 8-8-x-x-x-x.", "voicing_ids": ["guitar_v000"], "confidence": 0.95, "sources": [{{"path": "state/voicings/guitar-corpus.json", "row": 0}}]}}}}"#).unwrap();
        f
    }

    #[test]
    fn stub_returns_canned_response() {
        let f = make_fixture_file();
        let fixtures = load_fixtures(f.path());
        let req = ChatbotRequest {
            question: "Does voicing guitar_v042 exist in the corpus?".to_string(),
            instrument: Some(Instrument::Guitar),
        };
        let resp = ask_stub(&req, &fixtures);
        assert!(resp.answer.contains("guitar_v042"));
        assert_eq!(resp.voicing_ids, vec!["guitar_v042"]);
        assert!(resp.confidence > 0.9);
        assert_eq!(resp.sources.len(), 1);
        assert_eq!(resp.sources[0].row, 42);
    }

    #[test]
    fn stub_refuses_unknown() {
        let f = make_fixture_file();
        let fixtures = load_fixtures(f.path());
        let req = ChatbotRequest {
            question: "What is the meaning of life?".to_string(),
            instrument: None,
        };
        let resp = ask_stub(&req, &fixtures);
        assert!(resp.confidence < f64::EPSILON);
        assert!(resp.voicing_ids.is_empty());
        assert!(resp.answer.contains("don't have enough information"));
    }

    #[test]
    fn stub_parses_fixture_file() {
        let f = make_fixture_file();
        let fixtures = load_fixtures(f.path());
        assert_eq!(fixtures.len(), 2);
        assert!(fixtures.contains_key("does voicing guitar_v042"));
        assert!(fixtures.contains_key("show me voicing guitar_v000"));
    }

    /// Every adversarial-corpus prompt that is not `expected_check == "llm"`
    /// must have a stub fixture that matches it under the same `starts_with`
    /// rule used by [`ask_stub`]. Without this, a newly added prompt silently
    /// falls back to the default refusal — the exact rot mode that reduced
    /// the regression gauge to "stubs only match 1 prompt."
    #[test]
    fn every_graded_prompt_has_fixture_coverage() {
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root from crate manifest");
        let fixtures_path = workspace.join("tests/adversarial/fixtures/stub-responses.jsonl");
        let corpus_dir = workspace.join("tests/adversarial/corpus");

        let fixtures = load_fixtures(&fixtures_path);
        assert!(
            !fixtures.is_empty(),
            "no fixtures loaded from {:?}",
            fixtures_path
        );
        let prefixes: Vec<String> = fixtures.keys().cloned().collect();

        let mut uncovered: Vec<String> = Vec::new();

        for entry in std::fs::read_dir(&corpus_dir).expect("corpus dir") {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "jsonl") {
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("bad JSON in {:?}: {}", path, e));
                let id = v["id"].as_str().unwrap_or("?");
                let prompt = v["prompt"].as_str().unwrap_or("");
                let expected_check = v["expected_check"].as_str().unwrap_or("");
                if expected_check == "llm" {
                    continue;
                }
                let prompt_lower = prompt.to_lowercase();
                let covered = prefixes.iter().any(|p| prompt_lower.starts_with(p));
                if !covered {
                    uncovered.push(format!("{} -- {:?}", id, prompt));
                }
            }
        }

        assert!(
            uncovered.is_empty(),
            "uncovered deterministic-graded prompts (add stubs to stub-responses.jsonl):\n  {}",
            uncovered.join("\n  ")
        );
    }

    /// For every corpus prompt that declares `expected_patterns`, the
    /// sanitizer's `matched_patterns` must be a superset. Weakening a pattern
    /// (e.g. tightening `imperative_override` so it no longer matches
    /// "ignore previous") would leave the verdict-only gauge green because
    /// Layer 2's confidence check still fires F — but the Layer 0 coverage
    /// silently dropped. Superset semantics allow strengthening: a new pattern
    /// that catches an additional prompt is fine.
    #[test]
    fn sanitizer_patterns_meet_expectations() {
        use ix_sanitize::Sanitizer;
        use std::collections::HashSet;

        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root from crate manifest");
        let corpus_dir = workspace.join("tests/adversarial/corpus");

        let sanitizer = Sanitizer::new();
        let mut failures: Vec<String> = Vec::new();
        let mut checked = 0;

        for entry in std::fs::read_dir(&corpus_dir).expect("corpus dir") {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "jsonl") {
                continue;
            }
            let content = std::fs::read_to_string(&path).unwrap();
            for line in content.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let v: serde_json::Value = serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("bad JSON in {:?}: {}", path, e));
                let Some(expected) = v.get("expected_patterns").and_then(|p| p.as_array()) else {
                    continue;
                };
                let id = v["id"].as_str().unwrap_or("?");
                let prompt = v["prompt"].as_str().unwrap_or("");
                let expected: HashSet<String> = expected
                    .iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect();

                let sanitized = sanitizer.sanitize(prompt);
                let actual: HashSet<String> = sanitized.matched_patterns.iter().cloned().collect();

                let missing: Vec<_> = expected.difference(&actual).cloned().collect();
                if !missing.is_empty() {
                    failures.push(format!(
                        "{}: missing patterns {:?} (actual: {:?})",
                        id, missing, actual
                    ));
                }
                checked += 1;
            }
        }

        assert!(checked > 0, "no corpus entries declared expected_patterns");
        assert!(
            failures.is_empty(),
            "sanitizer coverage regressed for {} prompt(s):\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}
