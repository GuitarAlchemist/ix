//! Telemetry-driven coverage analysis for the chatbot adversarial corpus.
//!
//! Reads GA voicing-search telemetry (JSONL files emitted by the live
//! chatbot/MCP path), classifies each real user query into an intent
//! bucket, and diffs the bucket distribution against the adversarial
//! corpus categories. Surfaces intent clusters that real users hit but
//! the corpus does not exercise, and emits proposed adversarial test
//! cases derived directly from telemetry.
//!
//! This closes the manual "telemetry sweep before design" loop into
//! deterministic automation: an autonomous agent can run `ga-chatbot
//! coverage` on a schedule, file the proposed cases as PRs, and never
//! ask the human "what queries should I add to the corpus?".

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// One telemetry record from `ga/state/telemetry/voicing-search/*.jsonl`.
///
/// Fields beyond `ts` and `q` are optional because telemetry shape has
/// evolved over time and older lines may omit later additions.
#[derive(Debug, Clone, Deserialize)]
pub struct TelemetryEntry {
    pub ts: String,
    pub q: String,
    #[serde(default)]
    pub chord: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub results: usize,
    #[serde(default)]
    pub empty: bool,
}

/// Coarse intent label assigned by the rule-based classifier.
///
/// Rule-based, not learned. Each label maps 1:1 to a recommended
/// adversarial corpus category so the proposal loop is deterministic.
/// v2 adds mood/style/famous-reference buckets surfaced by real
/// telemetry ("something warm and mellow", "Hendrix chord") that v1
/// silently routed to out_of_scope/uncategorized.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentLabel {
    ChordVoicing,
    ScaleQuery,
    VoiceLeading,
    InstrumentSpecific,
    /// Mood/affect language ("warm", "dark", "ethereal").
    MoodBased,
    /// Genre/style tags ("jazz", "blues", "bossa").
    StyleQuery,
    /// Artist or song reference ("Hendrix chord", "Beatles voicing").
    FamousReference,
    OutOfScope,
    NaturalLanguage,
    Uncategorized,
}

impl IntentLabel {
    /// Adversarial corpus category most likely to cover this intent.
    ///
    /// MoodBased and FamousReference route to `hallucination` because the
    /// risk is the chatbot fabricating a plausible-but-wrong answer; the
    /// QA harness should grade restraint, not creativity. StyleQuery
    /// goes to `graduated` because style/genre tags need stratified
    /// difficulty, not a single canonical answer.
    pub fn recommended_corpus_category(&self) -> &'static str {
        match self {
            IntentLabel::ChordVoicing => "grounding",
            IntentLabel::ScaleQuery => "grounding",
            IntentLabel::VoiceLeading => "graduated",
            IntentLabel::InstrumentSpecific => "cross-instrument",
            IntentLabel::MoodBased => "hallucination",
            IntentLabel::StyleQuery => "graduated",
            IntentLabel::FamousReference => "hallucination",
            IntentLabel::OutOfScope => "hallucination",
            IntentLabel::NaturalLanguage => "graduated",
            IntentLabel::Uncategorized => "graduated",
        }
    }
}

/// Famous artist/band references that signal a "name-the-chord" intent.
///
/// Lowercase; matched as substring after lowercasing the query. Kept
/// narrow to avoid false positives — extend by telemetry, not guess.
const FAMOUS_REFERENCES: &[&str] = &[
    "hendrix",
    "beatles",
    "pink floyd",
    "steely dan",
    "coltrane",
    "monk",
    "metheny",
    "holdsworth",
    "wes montgomery",
    "miles davis",
    "bowie",
    "zappa",
];

/// Style/genre keywords. Lowercase substring match.
const STYLE_KEYWORDS: &[&str] = &[
    "jazz",
    "blues",
    "rock",
    "metal",
    "funk",
    "bossa",
    "samba",
    "swing",
    "bebop",
    "fusion",
    "gospel",
    "country",
    "folk",
    "gypsy",
    "reggae",
    "latin",
    "comping",
];

/// Mood/affect adjectives. Picked for low overlap with voice-leading
/// vocabulary ("smooth" stays in VL, not here).
const MOOD_KEYWORDS: &[&str] = &[
    "warm",
    "dark",
    "bright",
    "mellow",
    "ethereal",
    "gritty",
    "dreamy",
    "mysterious",
    "haunting",
    "nostalgic",
    "sad",
    "happy",
    "tense",
    "open",
    "lush",
];

/// Classify a telemetry entry into an intent bucket.
///
/// Order is load-bearing: out-of-scope dominates everything; voice
/// leading wins over mood (a "smooth resolve" is VL, not affect);
/// famous-reference wins over style ("Hendrix chord" is a reference,
/// not a genre); style/instrument win over mood (more specific). Mood
/// is the catch-all for affective language before length-based
/// fallbacks.
pub fn classify_intent(entry: &TelemetryEntry) -> IntentLabel {
    if entry.empty || entry.results == 0 {
        return IntentLabel::OutOfScope;
    }
    let q = entry.q.to_lowercase();
    if q.contains("voice leading")
        || q.contains("voice-leading")
        || q.contains("smooth")
        || q.contains("transition")
        || q.contains("resolve")
    {
        return IntentLabel::VoiceLeading;
    }
    if FAMOUS_REFERENCES.iter().any(|k| q.contains(k)) {
        return IntentLabel::FamousReference;
    }
    if STYLE_KEYWORDS.iter().any(|k| q.contains(k)) {
        return IntentLabel::StyleQuery;
    }
    let instruments = [
        "bass",
        "ukulele",
        "mandolin",
        "banjo",
        "baritone",
        "7-string",
        "8-string",
        "fretless",
    ];
    if instruments.iter().any(|k| q.contains(k)) {
        return IntentLabel::InstrumentSpecific;
    }
    if q.contains("scale")
        || q.contains("mode ")
        || q.contains("ionian")
        || q.contains("dorian")
        || q.contains("phrygian")
        || q.contains("lydian")
        || q.contains("mixolydian")
        || q.contains("aeolian")
        || q.contains("locrian")
    {
        return IntentLabel::ScaleQuery;
    }
    if MOOD_KEYWORDS.iter().any(|k| q.contains(k)) {
        return IntentLabel::MoodBased;
    }
    if entry.chord.is_some() {
        return IntentLabel::ChordVoicing;
    }
    if entry.q.len() > 50 {
        return IntentLabel::NaturalLanguage;
    }
    IntentLabel::Uncategorized
}

/// Per-cluster summary in the coverage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentCluster {
    pub label: IntentLabel,
    pub telemetry_count: usize,
    pub recommended_category: String,
    pub corpus_count: usize,
    pub sample_queries: Vec<String>,
}

/// Proposed adversarial test case derived from an uncovered cluster.
///
/// Shape matches `tests/adversarial/corpus/*.jsonl` so an autonomous
/// agent can append directly with no transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedTestCase {
    pub id: String,
    pub category: String,
    pub prompt: String,
    pub expected_check: String,
    pub expected_verdict: String,
    pub source: String,
}

/// Top-level coverage report. Schema versioned so consumers can pin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub schema_version: String,
    pub produced_at: String,
    pub telemetry_query_count: usize,
    pub corpus_total: usize,
    pub corpus_by_category: BTreeMap<String, usize>,
    pub clusters: Vec<IntentCluster>,
    pub uncovered_clusters: Vec<IntentCluster>,
    pub proposed_test_cases: Vec<ProposedTestCase>,
    /// Fraction of observed intents whose recommended corpus category
    /// has at least `MIN_COVERAGE_PROMPTS` prompts.
    pub coverage_score: f64,
}

/// Minimum prompt count for a corpus category to count as "covering"
/// an intent. Three is enough for stratified sampling without being so
/// permissive that one stale prompt counts as coverage.
pub const MIN_COVERAGE_PROMPTS: usize = 3;

/// Load all `*.jsonl` telemetry files in a directory.
///
/// Malformed lines are silently skipped — telemetry is opportunistic
/// data, not a contract; one bad line should not fail the whole sweep.
pub fn load_telemetry(dir: &Path) -> Vec<TelemetryEntry> {
    let mut entries = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return entries;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(t) = serde_json::from_str::<TelemetryEntry>(line) {
                entries.push(t);
            }
        }
    }
    entries
}

/// Count adversarial corpus prompts per category.
///
/// Each `*.jsonl` file in the corpus dir is one category; the file stem
/// is the category name (e.g. `cross-instrument.jsonl` → `cross-instrument`).
pub fn load_corpus_categories(corpus_dir: &Path) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    let Ok(rd) = std::fs::read_dir(corpus_dir) else {
        return counts;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(category) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        let n = content.lines().filter(|l| !l.trim().is_empty()).count();
        counts.insert(category.to_string(), n);
    }
    counts
}

/// Run the coverage analysis end-to-end.
///
/// `produced_at` is injected so tests can pin a deterministic timestamp.
pub fn analyze(
    telemetry: Vec<TelemetryEntry>,
    corpus_counts: BTreeMap<String, usize>,
    produced_at: String,
) -> CoverageReport {
    let corpus_total: usize = corpus_counts.values().sum();
    let telemetry_query_count = telemetry.len();

    let mut by_intent: BTreeMap<IntentLabel, Vec<String>> = BTreeMap::new();
    for entry in &telemetry {
        let intent = classify_intent(entry);
        by_intent.entry(intent).or_default().push(entry.q.clone());
    }

    let mut clusters = Vec::new();
    let mut uncovered = Vec::new();
    let mut proposed = Vec::new();
    let mut covered_intents = 0usize;
    let total_intents = by_intent.len();

    for (intent, queries) in &by_intent {
        let recommended = intent.recommended_corpus_category();
        let corpus_count = corpus_counts.get(recommended).copied().unwrap_or(0);
        let is_covered = corpus_count >= MIN_COVERAGE_PROMPTS;
        if is_covered {
            covered_intents += 1;
        }

        let sample: Vec<String> = queries.iter().take(3).cloned().collect();
        let cluster = IntentCluster {
            label: intent.clone(),
            telemetry_count: queries.len(),
            recommended_category: recommended.to_string(),
            corpus_count,
            sample_queries: sample.clone(),
        };
        clusters.push(cluster.clone());

        if !is_covered {
            for (i, q) in sample.iter().enumerate() {
                let intent_slug = format!("{:?}", intent).to_lowercase();
                proposed.push(ProposedTestCase {
                    id: format!("telemetry-{}-{:03}", intent_slug, i),
                    category: recommended.to_string(),
                    prompt: q.clone(),
                    expected_check: "telemetry_derived_coverage".into(),
                    expected_verdict: "U".into(),
                    source: "telemetry-derived".into(),
                });
            }
            uncovered.push(cluster);
        }
    }

    let coverage_score = if total_intents == 0 {
        0.0
    } else {
        covered_intents as f64 / total_intents as f64
    };

    CoverageReport {
        schema_version: "0.1.0".into(),
        produced_at,
        telemetry_query_count,
        corpus_total,
        corpus_by_category: corpus_counts,
        clusters,
        uncovered_clusters: uncovered,
        proposed_test_cases: proposed,
        coverage_score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn entry(q: &str, chord: Option<&str>, results: usize, empty: bool) -> TelemetryEntry {
        TelemetryEntry {
            ts: "2026-05-04T00:00:00Z".into(),
            q: q.into(),
            chord: chord.map(String::from),
            tags: vec![],
            results,
            empty,
        }
    }

    #[test]
    fn classifier_detects_voice_leading_phrases() {
        let e = entry("smoothest transition from Dm7 to G7", None, 5, false);
        assert_eq!(classify_intent(&e), IntentLabel::VoiceLeading);
    }

    #[test]
    fn classifier_detects_instrument_specific() {
        let e = entry("bass voicing for Am7", None, 3, false);
        assert_eq!(classify_intent(&e), IntentLabel::InstrumentSpecific);
    }

    #[test]
    fn empty_results_dominate_classifier() {
        let e = entry("Cmaj7 jazz voice leading bass", None, 0, true);
        assert_eq!(classify_intent(&e), IntentLabel::OutOfScope);
    }

    #[test]
    fn chord_query_falls_through_to_chord_voicing() {
        let e = entry("Cmaj7", Some("Cmaj7"), 3, false);
        assert_eq!(classify_intent(&e), IntentLabel::ChordVoicing);
    }

    #[test]
    fn scale_keyword_routes_to_scale_query() {
        let e = entry("Lydian scale notes", None, 5, false);
        assert_eq!(classify_intent(&e), IntentLabel::ScaleQuery);
    }

    #[test]
    fn mood_keyword_routes_to_mood_based() {
        let e = entry("something warm and mellow", None, 5, false);
        assert_eq!(classify_intent(&e), IntentLabel::MoodBased);
    }

    #[test]
    fn style_keyword_routes_to_style_query() {
        let e = entry("jazz comping voicings", None, 5, false);
        assert_eq!(classify_intent(&e), IntentLabel::StyleQuery);
    }

    #[test]
    fn famous_reference_routes_to_famous_reference() {
        let e = entry("Hendrix chord", None, 5, false);
        assert_eq!(classify_intent(&e), IntentLabel::FamousReference);
    }

    #[test]
    fn voice_leading_wins_over_mood() {
        // "smooth" is a VL keyword, "warm" is mood — VL must win
        let e = entry("smooth warm transition to tonic", None, 5, false);
        assert_eq!(classify_intent(&e), IntentLabel::VoiceLeading);
    }

    #[test]
    fn famous_reference_wins_over_style() {
        // "jazz" is style, "Coltrane" is famous — famous must win
        let e = entry("Coltrane jazz changes", None, 5, false);
        assert_eq!(classify_intent(&e), IntentLabel::FamousReference);
    }

    #[test]
    fn analyze_marks_uncovered_intent_and_proposes_cases() {
        let telemetry = vec![
            entry("smoothest transition from Dm7 to G7", None, 5, false),
            entry("voice leading ii-V-I", None, 5, false),
            entry("smooth resolve to tonic", None, 5, false),
            entry("Cmaj7", Some("Cmaj7"), 3, false),
        ];
        let mut corpus = BTreeMap::new();
        // grounding category covers ChordVoicing — has enough prompts
        corpus.insert("grounding".into(), 6);
        // graduated category has zero — should mark VoiceLeading uncovered
        corpus.insert("graduated".into(), 0);

        let r = analyze(telemetry, corpus, "2026-05-04T12:00:00Z".into());

        assert_eq!(r.telemetry_query_count, 4);
        assert_eq!(r.corpus_total, 6);
        // VoiceLeading has 3 telemetry queries, 0 in corpus → uncovered
        let vl = r
            .uncovered_clusters
            .iter()
            .find(|c| c.label == IntentLabel::VoiceLeading)
            .expect("voice-leading should be uncovered");
        assert_eq!(vl.telemetry_count, 3);
        // Three sample queries → three proposed cases for this cluster
        let vl_proposed: Vec<_> = r
            .proposed_test_cases
            .iter()
            .filter(|c| c.category == "graduated")
            .collect();
        assert_eq!(vl_proposed.len(), 3);
        // ChordVoicing has corpus coverage → not in uncovered
        assert!(r
            .uncovered_clusters
            .iter()
            .all(|c| c.label != IntentLabel::ChordVoicing));
        // Coverage score: 1 of 2 intents covered = 0.5
        assert!((r.coverage_score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn load_telemetry_skips_malformed_lines() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, r#"{{"ts":"2026-05-04T00:00:00Z","q":"Cmaj7","results":3}}"#).unwrap();
        writeln!(f, "this is not json").unwrap();
        writeln!(f).unwrap();
        writeln!(f, r#"{{"ts":"2026-05-04T00:01:00Z","q":"Dm7","results":3}}"#).unwrap();
        let entries = load_telemetry(dir.path());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].q, "Cmaj7");
        assert_eq!(entries[1].q, "Dm7");
    }

    #[test]
    fn load_corpus_categories_counts_per_file() {
        let dir = tempdir().unwrap();
        let mut a = std::fs::File::create(dir.path().join("grounding.jsonl")).unwrap();
        writeln!(a, "{{}}").unwrap();
        writeln!(a, "{{}}").unwrap();
        let mut b = std::fs::File::create(dir.path().join("cross-instrument.jsonl")).unwrap();
        writeln!(b, "{{}}").unwrap();
        let counts = load_corpus_categories(dir.path());
        assert_eq!(counts.get("grounding"), Some(&2));
        assert_eq!(counts.get("cross-instrument"), Some(&1));
    }
}
