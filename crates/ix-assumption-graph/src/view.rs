//! Phase 4 — faceted navigation over the assumption graph.
//!
//! Answers "how is the graph structured?" along **both** axes:
//! - **namespace** (project/module) — derived from a code-anchored node's path
//!   (`crates/<crate>/…` → `<crate>`); research claims bucket under `research`;
//! - **kind** (invariant / assumption / hypothesis / … / research-claim) and
//!   **domain** (`dev` vs `research`).
//!
//! [`AssumptionGraph::view`] produces a [`FacetedView`] — the serializable data
//! model a UI (Prime Radiant, a dashboard) or the `ix_assumption_query` MCP tool
//! renders. Escalated (Contradictory) claims are surfaced separately for triage.

use std::collections::BTreeMap;

use ix_fuzzy::FuzzyError;
use ix_types::Hexavalent;
use serde::Serialize;

use crate::graph::{normalize_claim, AssumptionGraph};
use crate::node::AssumptionNode;

/// Namespace facet for a node: its crate (or top path segment) for code-anchored
/// nodes, `"research"` for research claims.
pub fn namespace_of(node: &AssumptionNode) -> String {
    match &node.location {
        Some(loc) => namespace_from_path(&loc.path),
        None => "research".to_string(),
    }
}

/// `crates/<crate>/src/…` → `<crate>`; otherwise the first path segment;
/// `(unknown)` if empty.
fn namespace_from_path(path: &str) -> String {
    let norm = path.replace('\\', "/");
    let parts: Vec<&str> = norm.split('/').filter(|s| !s.is_empty()).collect();
    if let Some(i) = parts.iter().position(|&s| s == "crates") {
        if let Some(crate_name) = parts.get(i + 1) {
            return (*crate_name).to_string();
        }
    }
    parts.first().map(|s| (*s).to_string()).unwrap_or_else(|| "(unknown)".to_string())
}

/// One claim's fused state, with its navigation facets.
#[derive(Debug, Clone, Serialize)]
pub struct ClaimView {
    pub claim: String,
    pub verdict: Hexavalent,
    pub escalated: bool,
    pub namespace: String,
    pub domains: Vec<String>,
    pub source_count: usize,
    pub contradiction_count: usize,
}

/// Faceted snapshot of the whole graph — the UI / MCP data model.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FacetedView {
    pub node_count: usize,
    pub claim_count: usize,
    pub escalated_count: usize,
    /// Claim views grouped under their primary namespace.
    pub by_namespace: BTreeMap<String, Vec<ClaimView>>,
    /// Node counts per kind (kebab name).
    pub by_kind: BTreeMap<String, usize>,
    /// Node counts per domain (`dev` / `research`).
    pub by_domain: BTreeMap<String, usize>,
    /// Just the escalated (Contradictory) claims, for quick triage.
    pub escalations: Vec<ClaimView>,
}

impl AssumptionGraph {
    /// Build the faceted navigation view. Groups nodes by normalized claim,
    /// joins each with its fused verdict, and buckets by namespace / kind /
    /// domain. A claim's *primary namespace* is the most common namespace among
    /// its contributing nodes (ties broken alphabetically; `research`-only
    /// claims land under `research`).
    pub fn view(&self) -> Result<FacetedView, FuzzyError> {
        // claim (normalized) -> contributing nodes
        let mut groups: BTreeMap<String, Vec<&AssumptionNode>> = BTreeMap::new();
        for node in self.nodes() {
            groups
                .entry(normalize_claim(&node.claim))
                .or_default()
                .push(node);
        }

        // normalized claim -> fused verdict
        let fused = self.fuse()?;
        let by_norm: BTreeMap<String, &crate::FusedClaim> = fused
            .iter()
            .map(|f| (normalize_claim(&f.claim), f))
            .collect();

        let mut view = FacetedView {
            node_count: self.node_count(),
            ..Default::default()
        };

        for node in self.nodes() {
            *view.by_kind.entry(node.kind.as_str().to_string()).or_default() += 1;
            *view.by_domain.entry(node.domain().to_string()).or_default() += 1;
        }

        for (norm, nodes) in &groups {
            let Some(f) = by_norm.get(norm) else { continue };

            let namespace = primary_namespace(nodes);
            let mut domains: Vec<String> =
                nodes.iter().map(|n| n.domain().to_string()).collect();
            domains.sort_unstable();
            domains.dedup();

            let cv = ClaimView {
                claim: f.claim.clone(),
                verdict: f.verdict,
                escalated: f.escalated,
                namespace: namespace.clone(),
                domains,
                source_count: f.source_count,
                contradiction_count: f.contradiction_count,
            };

            if cv.escalated {
                view.escalations.push(cv.clone());
            }
            view.by_namespace.entry(namespace).or_default().push(cv);
        }

        view.claim_count = groups.len();
        view.escalated_count = view.escalations.len();
        // Deterministic escalation order.
        view.escalations.sort_by(|a, b| a.claim.cmp(&b.claim));
        Ok(view)
    }
}

/// Most common namespace among a claim's nodes (ties → alphabetical first).
fn primary_namespace(nodes: &[&AssumptionNode]) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for n in nodes {
        *counts.entry(namespace_of(n)).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(ns, _)| ns)
        .unwrap_or_else(|| "(unknown)".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::ResearchClaim;
    use ix_ai_annotations::types::annotation_id;
    use ix_ai_annotations::{
        Annotation, AnnotationKind, Certainty, Location, Source, TruthValue, SCHEMA_VERSION,
    };

    fn ann(path: &str, line: u32, claim: &str, tv: TruthValue) -> Annotation {
        let kind = AnnotationKind::Invariant;
        Annotation {
            schema_version: SCHEMA_VERSION,
            id: annotation_id(path, line, kind, claim),
            kind,
            claim: claim.to_string(),
            truth_value: tv,
            certainty: Certainty::Test,
            confidence: 0.9,
            source: Source { author: "claude".to_string(), model: None, evidence: None },
            location: Location { path: path.to_string(), line_start: line, line_end: line },
            created_at: "2026-05-31T00:00:00Z".to_string(),
            updated_at: "2026-05-31T00:00:00Z".to_string(),
            stale: false,
            reconciliation: None,
        }
    }

    #[test]
    fn namespace_derives_crate_from_path() {
        assert_eq!(namespace_from_path("crates/ix-search/src/binary.rs"), "ix-search");
        assert_eq!(namespace_from_path("crates\\ix-fuzzy\\src\\ops.rs"), "ix-fuzzy");
        assert_eq!(namespace_from_path("README.md"), "README.md");
    }

    #[test]
    fn view_buckets_by_namespace_and_domain() {
        let g = AssumptionGraph::from_parts(
            vec![
                ann("crates/ix-search/src/a.rs", 1, "alpha holds", TruthValue::T),
                ann("crates/ix-fuzzy/src/b.rs", 2, "beta holds", TruthValue::T),
            ],
            vec![ResearchClaim {
                claim: "gamma is settled".to_string(),
                truth_value: TruthValue::P,
                confidence: 0.8,
                source: "deep-research".to_string(),
                evidence: None,
            }],
        )
        .unwrap();

        let view = g.view().unwrap();
        assert_eq!(view.node_count, 3);
        assert_eq!(view.claim_count, 3);
        assert!(view.by_namespace.contains_key("ix-search"));
        assert!(view.by_namespace.contains_key("ix-fuzzy"));
        assert!(view.by_namespace.contains_key("research"));
        assert_eq!(view.by_domain.get("dev"), Some(&2));
        assert_eq!(view.by_domain.get("research"), Some(&1));
        assert_eq!(view.by_kind.get("research-claim"), Some(&1));
        assert_eq!(view.escalated_count, 0);
    }

    #[test]
    fn view_surfaces_escalations() {
        // A research claim refutes a code assumption on the same claim.
        let g = AssumptionGraph::from_parts(
            vec![ann("crates/ix-x/src/a.rs", 1, "fast path is safe", TruthValue::T)],
            vec![ResearchClaim {
                claim: "fast path is safe".to_string(),
                truth_value: TruthValue::F,
                confidence: 0.9,
                source: "deep-research".to_string(),
                evidence: None,
            }],
        )
        .unwrap();

        let view = g.view().unwrap();
        assert_eq!(view.escalated_count, 1);
        assert_eq!(view.escalations[0].verdict, Hexavalent::Contradictory);
        // The claim touches both domains.
        assert_eq!(view.escalations[0].domains, vec!["dev", "research"]);
    }
}
