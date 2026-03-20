//! Demerzel governance: constitutions, personas, policies, and tetravalent logic.
//!
//! This crate loads and validates governance artifacts used by the Demerzel
//! agent framework, including constitutional articles, persona definitions,
//! operating policies, and a four-valued logic system for reasoning under
//! uncertainty.

pub mod constitution;
pub mod error;
pub mod feedback;
pub mod persona;
pub mod policy;
pub mod remediation_optimizer;
pub mod tetravalent;
pub mod karnaugh;
pub mod research_anomaly;
pub mod violation_pattern;
pub mod weight_evolution;

pub use constitution::{Article, ArticleRef, ComplianceResult, Constitution};
pub use error::{GovernanceError, Result};
pub use feedback::{
    AnomalyDetector, BeliefFile, ConfidenceCalibrator, MlFeedbackRecommendation,
    StalenessPredictor, StateReader,
};
pub use persona::{list_personas, Persona, Voice};
pub use policy::{AlignmentPolicy, EscalationLevel, Policy};
pub use remediation_optimizer::{RemediationOptimizer, RemediationPlan};
pub use research_anomaly::{AnomalyCluster, AnomalyClusterer, ParadigmAssessment, ResearchAnomaly};
pub use tetravalent::{BeliefState, EvidenceItem, ResolvedAction, TruthValue};
pub use violation_pattern::{ViolationPatternAnalyzer, ViolationPatternReport};
pub use weight_evolution::{ResearchOutcome, TransferOpportunity, WeightEvolver, WeightRecommendation};
