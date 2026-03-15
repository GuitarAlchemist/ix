//! Demerzel governance: constitutions, personas, policies, and tetravalent logic.
//!
//! This crate loads and validates governance artifacts used by the Demerzel
//! agent framework, including constitutional articles, persona definitions,
//! operating policies, and a four-valued logic system for reasoning under
//! uncertainty.

pub mod constitution;
pub mod error;
pub mod persona;
pub mod policy;
pub mod tetravalent;

pub use constitution::{Article, ArticleRef, ComplianceResult, Constitution};
pub use error::{GovernanceError, Result};
pub use persona::{list_personas, Persona, Voice};
pub use policy::{AlignmentPolicy, EscalationLevel, Policy};
pub use tetravalent::{BeliefState, EvidenceItem, ResolvedAction, TruthValue};
