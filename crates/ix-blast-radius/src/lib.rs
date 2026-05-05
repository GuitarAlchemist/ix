//! Blast-radius analyzer for the qa-architect-cycle pipeline.
//!
//! Promotes `qa-architect-cycle.ixql` from Phase 0 (hardcoded skeleton)
//! toward Phase 1 by providing a real, deterministic blast-radius
//! analysis from a list of changed file paths. Output matches the
//! `blast_radius` field of `ga/docs/contracts/qa-verdict.schema.json`
//! v0.1.0 byte-for-byte.
//!
//! v1 scope is path-based — crate-level component identification, layer
//! mapping by directory pattern, one-way door detection by glob. Public
//! API diff and invariant impact are deliberate v2 work tracked in
//! `docs/plans/2026-05-04-chatbot-autonomy-action-layer.md`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Layer enum from `qa-verdict.schema.json` `blast_radius.layers_touched.items.enum`.
///
/// Stays in lockstep with the schema. Adding a layer here without
/// updating the contract will produce verdicts that fail validation
/// downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Layer {
    Core,
    Domain,
    Analysis,
    AiMl,
    Orchestration,
    Apps,
    Frontend,
    Infra,
    Docs,
}

/// Output shape of a blast-radius analysis.
///
/// Field order, names, and types match the GA qa-verdict v0.1.0
/// contract `blast_radius` definition. Serialized JSON drops directly
/// into the verdict's `blast_radius` field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlastRadius {
    pub layers_touched: Vec<Layer>,
    pub one_way_doors_crossed: Vec<String>,
    pub invariants_at_risk: Vec<String>,
    pub components_reached: Vec<String>,
    /// Heuristic in [0.0, 1.0]. See `score_blast` for the formula.
    pub estimated_blast_score: f64,
}

/// Crate-name → Layer mapping. Order matters: more specific crate
/// substrings come first because layer assignment uses first-match.
const CRATE_LAYER_MAP: &[(&str, Layer)] = &[
    // Domain (music/voicings/optick) — check before generic ai_ml
    ("ga-chatbot", Layer::Domain),
    ("ix-voicings", Layer::Domain),
    ("ix-optick", Layer::Domain),
    ("ix-bracelet", Layer::Domain),
    ("memristive-markov", Layer::Domain),
    // Orchestration (agent loops, session, governance integration)
    ("ix-agent", Layer::Orchestration),
    ("ix-session", Layer::Orchestration),
    ("ix-context", Layer::Orchestration),
    ("ix-approval", Layer::Orchestration),
    ("ix-fuzzy", Layer::Orchestration),
    ("ix-loop-detect", Layer::Orchestration),
    ("ix-memory", Layer::Orchestration),
    ("ix-sentinel", Layer::Orchestration),
    ("ix-registry", Layer::Orchestration),
    ("ix-skill", Layer::Orchestration),
    ("ix-autoresearch", Layer::Orchestration),
    ("ix-catalog-core", Layer::Orchestration),
    // Analysis (code observatory, QA, drift)
    ("ix-code", Layer::Analysis),
    ("ix-grammar", Layer::Analysis),
    ("ix-quality-trend", Layer::Analysis),
    ("ix-invariant-coverage", Layer::Analysis),
    ("ix-embedding-diagnostics", Layer::Analysis),
    ("ix-adversarial", Layer::Analysis),
    ("ix-blast-radius", Layer::Analysis),
    ("ix-search", Layer::Analysis),
    // AI/ML primitives
    ("ix-math", Layer::AiMl),
    ("ix-optimize", Layer::AiMl),
    ("ix-supervised", Layer::AiMl),
    ("ix-unsupervised", Layer::AiMl),
    ("ix-ensemble", Layer::AiMl),
    ("ix-nn", Layer::AiMl),
    ("ix-rl", Layer::AiMl),
    ("ix-evolution", Layer::AiMl),
    ("ix-probabilistic", Layer::AiMl),
    ("ix-signal", Layer::AiMl),
    ("ix-chaos", Layer::AiMl),
    ("ix-game", Layer::AiMl),
    ("ix-gpu", Layer::AiMl),
    ("ix-autograd", Layer::AiMl),
    ("ix-dynamics", Layer::AiMl),
    ("ix-fractal", Layer::AiMl),
    ("ix-number-theory", Layer::AiMl),
    ("ix-rotation", Layer::AiMl),
    ("ix-sedenion", Layer::AiMl),
    ("ix-topo", Layer::AiMl),
    ("ix-ktheory", Layer::AiMl),
    ("ix-category", Layer::AiMl),
    ("ix-manifold", Layer::AiMl),
    ("ix-graph", Layer::AiMl),
    // Infra (governance, harness, sanitization, types, IO, cache)
    ("ix-governance", Layer::Infra),
    ("ix-harness", Layer::Infra),
    ("ix-sanitize", Layer::Infra),
    ("ix-types", Layer::Core),
    ("ix-io", Layer::Core),
    ("ix-cache", Layer::Core),
    ("ix-pipeline", Layer::Core),
    ("ix-net", Layer::Core),
    // Apps
    ("ix-dashboard", Layer::Apps),
    ("ix-demo", Layer::Apps),
];

/// Map a single repository-relative path to a `Layer`. Returns None if
/// the path does not belong to a recognized layer (caller can decide
/// whether to default or to flag).
pub fn layer_for_path(path: &str) -> Option<Layer> {
    let normalized = path.replace('\\', "/");

    // Repo-level dirs first
    if normalized.starts_with("docs/") || normalized.ends_with(".md") {
        return Some(Layer::Docs);
    }
    if normalized.starts_with("governance/") || normalized.starts_with("state/") {
        return Some(Layer::Infra);
    }
    if normalized.starts_with(".github/") {
        return Some(Layer::Infra);
    }

    // Crate-based mapping
    if let Some(rest) = normalized.strip_prefix("crates/") {
        let crate_name = rest.split('/').next().unwrap_or("");
        for (prefix, layer) in CRATE_LAYER_MAP {
            if crate_name.starts_with(prefix) {
                return Some(*layer);
            }
        }
    }

    // Workspace files
    if matches!(
        normalized.as_str(),
        "Cargo.toml" | "Cargo.lock" | "rust-toolchain.toml"
    ) {
        return Some(Layer::Infra);
    }

    None
}

/// Extract the component name (crate name or top-level dir) from a path.
pub fn component_for_path(path: &str) -> Option<String> {
    let normalized = path.replace('\\', "/");
    if let Some(rest) = normalized.strip_prefix("crates/") {
        let name = rest.split('/').next().unwrap_or("");
        if !name.is_empty() {
            return Some(name.to_string());
        }
    }
    if let Some(top) = normalized.split('/').next() {
        if !top.is_empty() && !top.contains('/') && normalized.contains('/') {
            return Some(top.to_string());
        }
    }
    None
}

/// One-way door patterns. Substring match against the path. Each door's
/// crossing is high-friction and recorded in `one_way_doors_crossed` so
/// the verdict can flag merge risk.
pub fn one_way_doors_in(paths: &[String]) -> Vec<String> {
    let mut doors = BTreeSet::new();
    for p in paths {
        let n = p.replace('\\', "/");
        if n.contains("docs/contracts/") && n.ends_with(".schema.json") {
            doors.insert(format!("contract:{n}"));
        }
        if n.contains("governance/demerzel/articles/") {
            doors.insert(format!("constitution:{n}"));
        }
        if n.ends_with("Cargo.toml") {
            // Cargo.toml changes can break semver — flag for review
            doors.insert(format!("cargo_manifest:{n}"));
        }
        if n.contains("docs/contracts/") && n.ends_with(".contract.md") {
            doors.insert(format!("contract_md:{n}"));
        }
        if n.starts_with("governance/demerzel/policies/") && n.ends_with(".yaml") {
            doors.insert(format!("policy:{n}"));
        }
    }
    doors.into_iter().collect()
}

/// Compute the heuristic blast score in [0.0, 1.0].
///
/// Conservative weighting — layers contribute most because cross-layer
/// changes have the widest blast radius; one-way doors gate merge so
/// they push hard; component count breadth contributes a small bump
/// past five touched components.
pub fn score_blast(
    layers: &[Layer],
    one_way_doors: &[String],
    components: &[String],
) -> f64 {
    let mut score: f64 = 0.0;
    score += (layers.len() as f64 * 0.10).min(0.40);
    score += (one_way_doors.len() as f64 * 0.15).min(0.50);
    if components.len() > 5 {
        score += ((components.len() - 5) as f64 * 0.05).min(0.20);
    }
    score.clamp(0.0, 1.0)
}

/// Run the full analysis on a list of changed paths.
///
/// Paths should be repo-root-relative (the format `git diff --name-only`
/// emits). Unknown paths are silently dropped from the layer set but
/// still counted toward components if they look like top-level dirs.
pub fn analyze(paths: &[String]) -> BlastRadius {
    let mut layers: BTreeSet<Layer> = BTreeSet::new();
    let mut components: BTreeSet<String> = BTreeSet::new();
    for p in paths {
        if let Some(layer) = layer_for_path(p) {
            layers.insert(layer);
        }
        if let Some(comp) = component_for_path(p) {
            components.insert(comp);
        }
    }

    let layers_vec: Vec<Layer> = layers.into_iter().collect();
    let components_vec: Vec<String> = components.into_iter().collect();
    let doors = one_way_doors_in(paths);
    let score = score_blast(&layers_vec, &doors, &components_vec);

    BlastRadius {
        layers_touched: layers_vec,
        one_way_doors_crossed: doors,
        invariants_at_risk: Vec::new(), // v2: tie to ix-invariant-coverage
        components_reached: components_vec,
        estimated_blast_score: score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ga_chatbot_path_resolves_to_domain() {
        assert_eq!(
            layer_for_path("crates/ga-chatbot/src/coverage.rs"),
            Some(Layer::Domain)
        );
    }

    #[test]
    fn ix_nn_path_resolves_to_ai_ml() {
        assert_eq!(
            layer_for_path("crates/ix-nn/src/transformer.rs"),
            Some(Layer::AiMl)
        );
    }

    #[test]
    fn ix_governance_path_resolves_to_infra() {
        assert_eq!(
            layer_for_path("crates/ix-governance/src/karnaugh.rs"),
            Some(Layer::Infra)
        );
    }

    #[test]
    fn docs_path_resolves_to_docs() {
        assert_eq!(layer_for_path("docs/plans/some-plan.md"), Some(Layer::Docs));
    }

    #[test]
    fn windows_backslash_paths_normalize() {
        assert_eq!(
            layer_for_path(r"crates\ix-agent\src\main.rs"),
            Some(Layer::Orchestration)
        );
    }

    #[test]
    fn unknown_path_returns_none_layer() {
        assert_eq!(layer_for_path("random/unrecognized/path.txt"), None);
    }

    #[test]
    fn component_extracted_from_crate_path() {
        assert_eq!(
            component_for_path("crates/ix-blast-radius/src/lib.rs"),
            Some("ix-blast-radius".to_string())
        );
    }

    #[test]
    fn schema_change_recorded_as_one_way_door() {
        let doors =
            one_way_doors_in(&["docs/contracts/some-thing.schema.json".into()]);
        assert_eq!(doors.len(), 1);
        assert!(doors[0].starts_with("contract:"));
    }

    #[test]
    fn cargo_toml_change_recorded_as_one_way_door() {
        let doors = one_way_doors_in(&["crates/ix-blast-radius/Cargo.toml".into()]);
        assert_eq!(doors.len(), 1);
        assert!(doors[0].starts_with("cargo_manifest:"));
    }

    #[test]
    fn constitutional_article_change_recorded() {
        let doors = one_way_doors_in(&[
            "governance/demerzel/articles/article-12-structural-vocabulary.yaml".into(),
        ]);
        assert_eq!(doors.len(), 1);
        assert!(doors[0].starts_with("constitution:"));
    }

    #[test]
    fn analyze_one_crate_one_layer_no_doors() {
        let r = analyze(&["crates/ga-chatbot/src/coverage.rs".into()]);
        assert_eq!(r.layers_touched, vec![Layer::Domain]);
        assert_eq!(r.components_reached, vec!["ga-chatbot".to_string()]);
        assert!(r.one_way_doors_crossed.is_empty());
        // 1 layer * 0.10 = 0.10
        assert!((r.estimated_blast_score - 0.10).abs() < 1e-9);
    }

    #[test]
    fn analyze_cross_layer_with_door_lifts_score() {
        let paths = vec![
            "crates/ga-chatbot/src/coverage.rs".into(),
            "crates/ix-nn/src/lib.rs".into(),
            "crates/ix-governance/src/karnaugh.rs".into(),
            "docs/contracts/qa-verdict.schema.json".into(),
        ];
        let r = analyze(&paths);
        // 3 layers: Domain, AiMl, Infra (Docs is also touched? no — doc
        // path is the schema file under docs/, but our layer mapping
        // routes anything under docs/ to Docs. So 4 layers actually.)
        assert!(r.layers_touched.contains(&Layer::Domain));
        assert!(r.layers_touched.contains(&Layer::AiMl));
        assert!(r.layers_touched.contains(&Layer::Infra));
        assert!(r.layers_touched.contains(&Layer::Docs));
        assert_eq!(r.layers_touched.len(), 4);
        assert_eq!(r.one_way_doors_crossed.len(), 1);
        // 4 layers * 0.10 = 0.40 (capped) + 1 door * 0.15 = 0.55
        assert!((r.estimated_blast_score - 0.55).abs() < 1e-9);
    }

    #[test]
    fn analyze_drops_unknown_paths_from_layers_but_components_unchanged() {
        let paths = vec![
            "crates/ga-chatbot/src/lib.rs".into(),
            "totally/unknown/file.txt".into(),
        ];
        let r = analyze(&paths);
        assert_eq!(r.layers_touched, vec![Layer::Domain]);
        // 'totally' becomes a component name from the top-level dir extractor
        assert!(r.components_reached.contains(&"totally".to_string()));
    }

    #[test]
    fn json_serialization_matches_contract_field_names() {
        let r = analyze(&["crates/ga-chatbot/src/lib.rs".into()]);
        let j = serde_json::to_value(&r).unwrap();
        // Contract requires these exact field names
        assert!(j.get("layers_touched").is_some());
        assert!(j.get("one_way_doors_crossed").is_some());
        assert!(j.get("invariants_at_risk").is_some());
        assert!(j.get("components_reached").is_some());
        assert!(j.get("estimated_blast_score").is_some());
        // Layer enum serializes as snake_case
        let arr = j["layers_touched"].as_array().unwrap();
        assert_eq!(arr[0].as_str(), Some("domain"));
    }

    #[test]
    fn score_clamps_at_one() {
        // 9 layers * 0.10 = 0.90 (cap 0.40), + 10 doors * 0.15 = 1.50 (cap 0.50)
        // = 0.40 + 0.50 = 0.90 + components bonus → 0.90 + (>5 comps)
        let layers = vec![
            Layer::Core,
            Layer::Domain,
            Layer::Analysis,
            Layer::AiMl,
            Layer::Orchestration,
            Layer::Apps,
            Layer::Frontend,
            Layer::Infra,
            Layer::Docs,
        ];
        let doors: Vec<String> = (0..10).map(|i| format!("d:{i}")).collect();
        let comps: Vec<String> = (0..15).map(|i| format!("c{i}")).collect();
        let s = score_blast(&layers, &doors, &comps);
        assert!(s <= 1.0);
        assert!(s >= 0.9);
    }
}
