//! Builds a `Dag<AssumptionNode>` from `@ai:` annotations and derives
//! `contradicts` relations from opposing hexavalent truth values.

use std::collections::HashMap;
use std::path::Path;

use ix_ai_annotations::{Annotation, TruthValue};
use ix_pipeline::dag::Dag;
use serde::Serialize;

use crate::node::{AssumptionNode, ResearchClaim};

// The pairwise-conflict predicate is part of the hexavalent truth-value algebra,
// owned by `ix-ai-annotations`. Re-exported so callers keep using
// `ix_assumption_graph::{conflicts}` / `graph::conflicts`.
pub use ix_ai_annotations::truth::conflicts;

/// Errors raised while building the graph.
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("annotation extraction failed: {0}")]
    Extract(#[from] ix_ai_annotations::Error),
    #[error("dag construction failed: {0}")]
    Dag(#[from] ix_pipeline::dag::DagError),
}

/// A derived contradiction: two nodes asserting opposing polarity about the
/// same (normalized) claim. Stored outside the DAG because `contradicts` is
/// symmetric and may cycle. `a`/`b` are ordered by id for deterministic output.
#[derive(Debug, Clone, Serialize)]
pub struct Contradiction {
    pub a: String,
    pub b: String,
    pub claim: String,
    pub a_truth: TruthValue,
    pub b_truth: TruthValue,
}

/// A graph of assumptions plus the contradictions derived from them.
pub struct AssumptionGraph {
    dag: Dag<AssumptionNode>,
    contradictions: Vec<Contradiction>,
}

impl AssumptionGraph {
    /// Build from a list of annotations (each promoted to a node).
    pub fn from_annotations(annotations: Vec<Annotation>) -> Result<Self, BuildError> {
        Self::build(annotations.into_iter().map(AssumptionNode::from).collect())
    }

    /// Build the **unified** graph from both dev annotations AND research
    /// claims. A research claim sharing a normalized claim with a dev
    /// annotation participates in fusion exactly like any other source — so a
    /// `/deep-research` finding that contradicts a code assumption escalates.
    pub fn from_parts(
        annotations: Vec<Annotation>,
        research: Vec<ResearchClaim>,
    ) -> Result<Self, BuildError> {
        let mut nodes: Vec<AssumptionNode> =
            annotations.into_iter().map(AssumptionNode::from).collect();
        nodes.extend(research.into_iter().map(AssumptionNode::from));
        Self::build(nodes)
    }

    /// Core builder: de-duplicates nodes by id (first occurrence wins), then
    /// derives contradictions.
    /// Build directly from a list of nodes (e.g. mined from code — see
    /// [`crate::mine`]). De-duplicates by id; derives contradictions.
    pub fn from_nodes(nodes: Vec<AssumptionNode>) -> Result<Self, BuildError> {
        Self::build(nodes)
    }

    fn build(nodes: Vec<AssumptionNode>) -> Result<Self, BuildError> {
        let mut dag: Dag<AssumptionNode> = Dag::new();
        for node in nodes {
            if dag.get(&node.id).is_some() {
                continue;
            }
            let id = node.id.clone();
            dag.add_node(id, node)?;
        }
        let contradictions = derive_contradictions(&dag);
        Ok(Self {
            dag,
            contradictions,
        })
    }

    /// Walk a workspace, extract `@ai:` annotations, and build the graph.
    pub fn from_workspace(workspace: &Path) -> Result<Self, BuildError> {
        Self::from_parts(production_annotations(workspace)?, Vec::new())
    }

    /// Walk a workspace and combine its `@ai:` annotations with research claims.
    pub fn from_workspace_with_research(
        workspace: &Path,
        research: Vec<ResearchClaim>,
    ) -> Result<Self, BuildError> {
        Self::from_parts(production_annotations(workspace)?, research)
    }

    /// Number of assumption nodes.
    pub fn node_count(&self) -> usize {
        self.dag.node_count()
    }

    /// Look up a node by id.
    pub fn get(&self, id: &str) -> Option<&AssumptionNode> {
        self.dag.get(id)
    }

    /// The derived contradictions, in deterministic order.
    pub fn contradictions(&self) -> &[Contradiction] {
        &self.contradictions
    }

    /// The underlying node container (for future acyclic edges / traversal).
    pub fn dag(&self) -> &Dag<AssumptionNode> {
        &self.dag
    }

    /// Iterate nodes in insertion order.
    pub fn nodes(&self) -> impl Iterator<Item = &AssumptionNode> {
        self.dag.node_ids().iter().filter_map(|id| self.dag.get(id))
    }
}

/// Generic, project-independent ignore substrings: test scaffolding, build
/// output, and documentation hold `@ai:` *examples* / marker-syntax demos, not
/// claims about shipping code. Paths are normalized to forward slashes with a
/// leading `/` before matching, so `/tests/` also catches a top-level `tests/`.
const DEFAULT_IGNORE: &[&str] = &[
    "/tests/",
    "/fixtures/",
    "/benches/",
    "/docs/",
    "/target/",
    ".md",
];

/// Read project-specific ignore substrings from `<workspace>/.assumptionignore`
/// (one substring per line, `#` comments and blanks skipped). Absent file → no
/// extra patterns. This is where repo-specific policy lives — e.g. excluding
/// the marker-defining crate, which is full of examples, not claims — so the
/// library never hardcodes a sibling crate's name.
fn extra_ignore_patterns(workspace: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(workspace.join(".assumptionignore")) else {
        return Vec::new();
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// Extract a workspace's `@ai:` annotations, keeping only **production** ones —
/// dropping anything matching [`DEFAULT_IGNORE`] or a `.assumptionignore`
/// pattern (test/doc examples would otherwise inflate and pollute the graph).
pub(crate) fn production_annotations(workspace: &Path) -> Result<Vec<Annotation>, BuildError> {
    let anns = ix_ai_annotations::extract_workspace(workspace)?;
    let extra = extra_ignore_patterns(workspace);
    Ok(anns
        .into_iter()
        .filter(|a| {
            let p = format!("/{}", a.location.path.replace('\\', "/"));
            let ignored = DEFAULT_IGNORE.iter().any(|pat| p.contains(pat))
                || extra.iter().any(|pat| p.contains(pat.as_str()));
            !ignored
        })
        .collect())
}

/// Collapse whitespace and case so trivially-different claim spellings group together.
pub(crate) fn normalize_claim(claim: &str) -> String {
    claim
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn derive_contradictions(dag: &Dag<AssumptionNode>) -> Vec<Contradiction> {
    // Group node ids by normalized claim.
    let mut by_claim: HashMap<String, Vec<&str>> = HashMap::new();
    for id in dag.node_ids() {
        if let Some(node) = dag.get(id) {
            by_claim
                .entry(normalize_claim(&node.claim))
                .or_default()
                .push(id.as_str());
        }
    }

    let mut out = Vec::new();
    for ids in by_claim.values() {
        for (i, &id_a) in ids.iter().enumerate() {
            for &id_b in &ids[i + 1..] {
                let (na, nb) = (dag.get(id_a).unwrap(), dag.get(id_b).unwrap());
                if conflicts(na.truth_value, nb.truth_value) {
                    // Canonical order: smaller id first.
                    let (lo, hi) = if na.id <= nb.id { (na, nb) } else { (nb, na) };
                    out.push(Contradiction {
                        a: lo.id.clone(),
                        b: hi.id.clone(),
                        claim: lo.claim.clone(),
                        a_truth: lo.truth_value,
                        b_truth: hi.truth_value,
                    });
                }
            }
        }
    }
    // Deterministic regardless of HashMap iteration order.
    out.sort_by(|x, y| (&x.a, &x.b).cmp(&(&y.a, &y.b)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ix_ai_annotations::types::annotation_id;
    use ix_ai_annotations::{
        Annotation, AnnotationKind, Certainty, Location, Source, TruthValue, SCHEMA_VERSION,
    };

    /// Build an annotation with a realistic wire shape (matches what the
    /// extractor actually emits — see feedback on real-data-shape tests).
    fn ann(
        path: &str,
        line: u32,
        kind: AnnotationKind,
        claim: &str,
        tv: TruthValue,
        conf: f64,
    ) -> Annotation {
        Annotation {
            schema_version: SCHEMA_VERSION,
            id: annotation_id(path, line, kind, claim),
            kind,
            claim: claim.to_string(),
            truth_value: tv,
            certainty: Certainty::Assumed,
            confidence: conf,
            source: Source {
                author: "claude".to_string(),
                model: Some("claude-opus-4-8".to_string()),
                evidence: None,
            },
            location: Location {
                path: path.to_string(),
                line_start: line,
                line_end: line,
            },
            created_at: "2026-05-31T00:00:00Z".to_string(),
            updated_at: "2026-05-31T00:00:00Z".to_string(),
            stale: false,
            reconciliation: None,
        }
    }

    #[test]
    fn builds_one_node_per_distinct_annotation() {
        let g = AssumptionGraph::from_annotations(vec![
            ann(
                "a.rs",
                1,
                AnnotationKind::Invariant,
                "arr is sorted",
                TruthValue::T,
                0.95,
            ),
            ann(
                "b.rs",
                2,
                AnnotationKind::Assumption,
                "caller holds lock",
                TruthValue::P,
                0.7,
            ),
            ann(
                "c.rs",
                3,
                AnnotationKind::Hypothesis,
                "race-free",
                TruthValue::U,
                0.4,
            ),
        ])
        .unwrap();
        assert_eq!(g.node_count(), 3);
        assert!(g.contradictions().is_empty());
    }

    #[test]
    fn detects_contradiction_on_opposing_polarity() {
        let g = AssumptionGraph::from_annotations(vec![
            ann(
                "a.rs",
                1,
                AnnotationKind::Invariant,
                "buffer is flushed",
                TruthValue::T,
                0.9,
            ),
            ann(
                "b.rs",
                9,
                AnnotationKind::Invariant,
                "buffer is flushed",
                TruthValue::F,
                0.8,
            ),
        ])
        .unwrap();
        assert_eq!(g.node_count(), 2);
        let c = g.contradictions();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].claim, "buffer is flushed");
        // a/b ordered by id; truths are the opposing pair {T, F}.
        let truths = [c[0].a_truth, c[0].b_truth];
        assert!(truths.contains(&TruthValue::T) && truths.contains(&TruthValue::F));
    }

    #[test]
    fn normalizes_claim_spelling_before_grouping() {
        // Different whitespace/case, same claim — must still group + contradict.
        let g = AssumptionGraph::from_annotations(vec![
            ann(
                "a.rs",
                1,
                AnnotationKind::Assumption,
                "Caller   holds the Lock",
                TruthValue::P,
                0.7,
            ),
            ann(
                "b.rs",
                2,
                AnnotationKind::Assumption,
                "caller holds the lock",
                TruthValue::D,
                0.6,
            ),
        ])
        .unwrap();
        assert_eq!(g.contradictions().len(), 1);
    }

    #[test]
    fn same_polarity_is_not_a_contradiction() {
        let g = AssumptionGraph::from_annotations(vec![
            ann(
                "a.rs",
                1,
                AnnotationKind::Invariant,
                "x positive",
                TruthValue::T,
                0.9,
            ),
            ann(
                "b.rs",
                2,
                AnnotationKind::Invariant,
                "x positive",
                TruthValue::P,
                0.7,
            ),
        ])
        .unwrap();
        assert!(g.contradictions().is_empty());
    }

    #[test]
    fn unknown_does_not_contradict_false() {
        // U is neutral — no evidential direction, so no conflict with F.
        let g = AssumptionGraph::from_annotations(vec![
            ann(
                "a.rs",
                1,
                AnnotationKind::Hypothesis,
                "leak free",
                TruthValue::U,
                0.4,
            ),
            ann(
                "b.rs",
                2,
                AnnotationKind::Hypothesis,
                "leak free",
                TruthValue::F,
                0.8,
            ),
        ])
        .unwrap();
        assert!(g.contradictions().is_empty());
    }

    #[test]
    fn duplicate_id_is_deduped() {
        // Same path:line:kind:claim ⇒ same id ⇒ one node.
        let g = AssumptionGraph::from_annotations(vec![
            ann(
                "a.rs",
                1,
                AnnotationKind::Invariant,
                "sorted",
                TruthValue::T,
                0.9,
            ),
            ann(
                "a.rs",
                1,
                AnnotationKind::Invariant,
                "sorted",
                TruthValue::T,
                0.9,
            ),
        ])
        .unwrap();
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn research_claim_contradicting_a_dev_assumption_escalates() {
        use crate::node::ResearchClaim;
        use ix_types::Hexavalent;

        // A code assumption says latency is under budget (T); a /deep-research
        // finding refutes it (F). Distinct sources ⇒ fusion escalates to C.
        // graph-test `ann` hardcodes author "claude".
        let annotations = vec![ann(
            "perf.rs",
            10,
            AnnotationKind::Invariant,
            "p99 latency under 5ms",
            TruthValue::T,
            0.9,
        )];
        let research = vec![ResearchClaim {
            claim: "p99 latency under 5ms".to_string(),
            truth_value: TruthValue::F,
            confidence: 0.85,
            source: "deep-research".to_string(),
            evidence: Some("bench-2026-05".to_string()),
        }];

        let g = AssumptionGraph::from_parts(annotations, research).unwrap();
        assert_eq!(g.node_count(), 2);

        let fused = g.fuse().unwrap();
        let latency = fused
            .iter()
            .find(|f| f.claim.to_lowercase().contains("latency"))
            .expect("latency claim present");
        assert_eq!(latency.verdict, Hexavalent::Contradictory);
        assert!(latency.escalated);
        assert_eq!(latency.source_count, 2);
    }

    #[test]
    fn conflicts_is_symmetric() {
        use TruthValue::*;
        for &a in &[T, P, U, D, F, C] {
            for &b in &[T, P, U, D, F, C] {
                assert_eq!(conflicts(a, b), conflicts(b, a), "asymmetry at {a:?},{b:?}");
            }
        }
        // Spot-check the truth table.
        assert!(conflicts(T, F));
        assert!(conflicts(P, D));
        assert!(!conflicts(T, P));
        assert!(!conflicts(U, F));
        assert!(!conflicts(C, T)); // C is flagged at source, not re-derived pairwise
    }
}
