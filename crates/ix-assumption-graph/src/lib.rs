//! Temporal assumption graph — Phase 1.
//!
//! Composes existing IX primitives into a graph of *assumptions*:
//!
//! - **nodes** come from [`ix_ai_annotations`] (`@ai:invariant`, `@ai:assumption`,
//!   `@ai:hypothesis`, …) — each already carries a hexavalent truth value
//!   (T/P/U/D/F/C) and certainty, so ingest is free;
//! - the node container is [`ix_pipeline::dag::Dag`] (cycle-checked), holding the
//!   acyclic structure (`depends_on` / `refined_from` edges arrive in a later phase);
//! - **contradictions** are derived structurally — two nodes asserting *opposing
//!   polarity* about the same claim (one leans true, the other leans false). This
//!   is the reconciler's `C`-promotion signal, surfaced as a relation.
//!
//! Contradictions are deliberately stored *outside* the [`Dag`] because the
//! `contradicts` relation is symmetric and may form cycles, which the DAG forbids.
//! Acceptability semantics over this relation (ABA dispute derivations), the
//! subjective-logic `opinion`, the `belief_time` axis, and JSONL persistence are
//! defined in `docs/contracts/2026-05-31-assumption-graph.contract.md` and land in
//! Phase 2/3 — this crate is the substrate they build on.

pub mod graph;
pub mod node;

pub use graph::{conflicts, AssumptionGraph, BuildError, Contradiction};
pub use node::{polarity, AssumptionNode, Polarity};

// Re-export the reused annotation vocabulary so consumers don't need a direct
// dependency on ix-ai-annotations for the common case.
pub use ix_ai_annotations::{AnnotationKind, Certainty, Location, Source, TruthValue};
