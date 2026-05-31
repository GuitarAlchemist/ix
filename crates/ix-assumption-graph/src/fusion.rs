//! Phase 2 — per-claim evidence fusion via `ix-fuzzy`'s CRDT observation merge.
//!
//! Each assumption node becomes a [`HexObservation`]; nodes sharing a normalized
//! claim are fused into a single verdict. This deliberately *reuses*
//! [`ix_fuzzy::observations::merge_all`] (Belnap `C` synthesis, six proven CRDT
//! obligations) rather than reimplementing fusion.
//!
//! The independence guard from contract §7.1 comes for free: `merge` only
//! synthesizes a contradiction between **distinct sources**, so two annotations
//! by the same author disagreeing about a claim do NOT escalate to `C` here
//! (that self-inconsistency is still surfaced by Phase 1's structural
//! [`crate::AssumptionGraph::contradictions`]). Phase 1 = "any opposing
//! polarity"; Phase 2 = "independent sources disagree → escalate".

use std::collections::BTreeMap;

use ix_fuzzy::hexavalent::{escalation_triggered, hexavalent_argmax};
use ix_fuzzy::observations::{merge_all, HexObservation};
use ix_fuzzy::FuzzyError;
use ix_types::Hexavalent;
use serde::Serialize;

use crate::graph::{normalize_claim, AssumptionGraph};
use crate::opinion::to_hexavalent;

/// The fused epistemic state of all assumptions sharing one normalized claim.
#[derive(Debug, Clone, Serialize)]
pub struct FusedClaim {
    /// The claim text (from the first contributing node).
    pub claim: String,
    /// Tiebreak-aware argmax of the fused distribution (priority `C>U>D>P>T>F`).
    pub verdict: Hexavalent,
    /// Mass on the winning verdict, in `[0, 1]`.
    pub confidence: f64,
    /// `true` iff `C` mass exceeds the Demerzel escalation threshold (`0.3`).
    pub escalated: bool,
    /// Distinct sources (independence classes) contributing to this claim.
    pub source_count: usize,
    /// Synthesized cross-source contradictions on this claim.
    pub contradiction_count: usize,
}

/// Floor applied to a node's confidence so a zero-confidence observation still
/// registers its variant (and the group never degenerates to an empty merge).
const WEIGHT_FLOOR: f64 = 1e-6;

impl AssumptionGraph {
    /// Fuse evidence per claim. Nodes are grouped by normalized claim (the same
    /// key Phase 1 uses), each group merged via [`ix_fuzzy::observations`].
    /// Returns one [`FusedClaim`] per distinct claim, in normalized-claim order.
    pub fn fuse(&self) -> Result<Vec<FusedClaim>, FuzzyError> {
        let mut groups: BTreeMap<String, Vec<&crate::node::AssumptionNode>> = BTreeMap::new();
        for node in self.nodes() {
            groups
                .entry(normalize_claim(&node.claim))
                .or_default()
                .push(node);
        }

        let mut out = Vec::with_capacity(groups.len());
        for (claim_key, nodes) in groups {
            let observations: Vec<HexObservation> = nodes
                .iter()
                .enumerate()
                .map(|(i, n)| HexObservation {
                    source: n.source.author.clone(),
                    diagnosis_id: n.id.clone(),
                    round: 0,
                    ordinal: i as u32,
                    claim_key: claim_key.clone(),
                    variant: to_hexavalent(n.truth_value),
                    weight: n.confidence.max(WEIGHT_FLOOR),
                    evidence: None,
                })
                .collect();

            let merged = merge_all(&observations)?;
            let escalated = escalation_triggered(&merged.distribution);
            // §7.2 discipline: an active contradiction (C above the escalation
            // threshold) IS the verdict — a higher-confidence dissenting source
            // must not let the argmax paper over it. Otherwise the tiebreak-aware
            // argmax wins.
            let verdict = if escalated {
                Hexavalent::Contradictory
            } else {
                hexavalent_argmax(&merged.distribution)
            };

            let mut sources: Vec<&str> = nodes.iter().map(|n| n.source.author.as_str()).collect();
            sources.sort_unstable();
            sources.dedup();

            out.push(FusedClaim {
                claim: nodes[0].claim.clone(),
                verdict,
                confidence: merged.distribution.get(&verdict),
                escalated,
                source_count: sources.len(),
                contradiction_count: merged.contradictions.len(),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::AssumptionGraph;
    use ix_ai_annotations::types::annotation_id;
    use ix_ai_annotations::{
        Annotation, AnnotationKind, Certainty, Location, Source, TruthValue, SCHEMA_VERSION,
    };
    use ix_types::Hexavalent;

    fn ann(
        author: &str,
        path: &str,
        line: u32,
        claim: &str,
        tv: TruthValue,
        conf: f64,
    ) -> Annotation {
        let kind = AnnotationKind::Assumption;
        Annotation {
            schema_version: SCHEMA_VERSION,
            id: annotation_id(path, line, kind, claim),
            kind,
            claim: claim.to_string(),
            truth_value: tv,
            certainty: Certainty::Assumed,
            confidence: conf,
            source: Source {
                author: author.to_string(),
                model: None,
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
    fn single_source_single_claim_fuses_to_its_value() {
        let g = AssumptionGraph::from_annotations(vec![ann(
            "claude", "a.rs", 1, "buffer flushed", TruthValue::T, 0.9,
        )])
        .unwrap();
        let fused = g.fuse().unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].verdict, Hexavalent::True);
        assert!(!fused[0].escalated);
        assert_eq!(fused[0].source_count, 1);
        assert_eq!(fused[0].contradiction_count, 0);
    }

    #[test]
    fn cross_source_disagreement_escalates_to_contradictory() {
        // Independent sources (claude vs sentrux) disagree on the same claim.
        let g = AssumptionGraph::from_annotations(vec![
            ann("claude", "a.rs", 1, "lock held on entry", TruthValue::T, 1.0),
            ann("sentrux", "b.rs", 2, "lock held on entry", TruthValue::F, 1.0),
        ])
        .unwrap();
        let fused = g.fuse().unwrap();
        assert_eq!(fused.len(), 1);
        assert_eq!(fused[0].verdict, Hexavalent::Contradictory);
        assert!(fused[0].escalated);
        assert_eq!(fused[0].source_count, 2);
        assert!(fused[0].contradiction_count >= 1);
    }

    #[test]
    fn same_source_disagreement_does_not_escalate() {
        // §7.1 independence guard: one author contradicting itself is NOT a
        // cross-source contradiction — no C synthesis, no escalation.
        let g = AssumptionGraph::from_annotations(vec![
            ann("claude", "a.rs", 1, "leak free", TruthValue::T, 1.0),
            ann("claude", "b.rs", 2, "leak free", TruthValue::F, 1.0),
        ])
        .unwrap();
        let fused = g.fuse().unwrap();
        assert_eq!(fused.len(), 1);
        assert!(!fused[0].escalated);
        assert_eq!(fused[0].contradiction_count, 0);
    }

    #[test]
    fn cross_source_agreement_does_not_escalate() {
        // claude T + human P — both lean true. No contradiction.
        let g = AssumptionGraph::from_annotations(vec![
            ann("claude", "a.rs", 1, "sorted ascending", TruthValue::T, 0.9),
            ann("human", "b.rs", 2, "sorted ascending", TruthValue::P, 0.7),
        ])
        .unwrap();
        let fused = g.fuse().unwrap();
        assert_eq!(fused.len(), 1);
        assert!(!fused[0].escalated);
        assert_eq!(fused[0].contradiction_count, 0);
        assert!(matches!(
            fused[0].verdict,
            Hexavalent::True | Hexavalent::Probable
        ));
        assert_eq!(fused[0].source_count, 2);
    }

    #[test]
    fn distinct_claims_fuse_independently() {
        let g = AssumptionGraph::from_annotations(vec![
            ann("claude", "a.rs", 1, "claim one", TruthValue::T, 0.9),
            ann("claude", "b.rs", 2, "claim two", TruthValue::F, 0.9),
        ])
        .unwrap();
        let fused = g.fuse().unwrap();
        assert_eq!(fused.len(), 2);
    }
}
