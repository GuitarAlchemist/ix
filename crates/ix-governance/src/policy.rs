use crate::error::{GovernanceError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A generic policy loaded from YAML.
///
/// Policies have variable structure beyond the common fields, so the
/// remaining content is stored as a `serde_json::Value`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Policy {
    /// Policy name.
    pub name: String,
    /// Policy version.
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// Remaining fields as a JSON value.
    #[serde(flatten)]
    pub extra: serde_json::Value,
}

impl Policy {
    /// Load a policy from a YAML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content).map_err(|e| GovernanceError::ParseError(e.to_string()))
    }
}

/// Confidence thresholds from the alignment policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholds {
    /// Confidence at or above which the agent may act autonomously.
    pub proceed_autonomously: f64,
    /// Confidence at or above which the agent may proceed with a note.
    pub proceed_with_note: f64,
    /// Confidence at or above which the agent should ask for confirmation.
    pub ask_for_confirmation: f64,
    /// Confidence below which the agent must escalate to a human.
    pub escalate_to_human: f64,
}

/// A strongly-typed representation of the alignment policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentPolicy {
    /// Policy name.
    pub name: String,
    /// Policy version.
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// Confidence thresholds for escalation decisions.
    pub confidence_thresholds: ConfidenceThresholds,
    /// Triggers that force escalation regardless of confidence.
    pub escalation_triggers: Vec<String>,
}

/// The escalation level recommended for a given confidence score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EscalationLevel {
    /// Agent may act on its own.
    Autonomous,
    /// Agent may proceed but should note the action.
    ProceedWithNote,
    /// Agent should ask for confirmation before proceeding.
    AskConfirmation,
    /// Agent must escalate to a human.
    Escalate,
}

impl AlignmentPolicy {
    /// Load the alignment policy from a YAML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content).map_err(|e| GovernanceError::ParseError(e.to_string()))
    }

    /// Determine the escalation level for a given confidence score.
    pub fn should_escalate(&self, confidence: f64) -> EscalationLevel {
        let t = &self.confidence_thresholds;
        if confidence >= t.proceed_autonomously {
            EscalationLevel::Autonomous
        } else if confidence >= t.proceed_with_note {
            EscalationLevel::ProceedWithNote
        } else if confidence >= t.ask_for_confirmation {
            EscalationLevel::AskConfirmation
        } else {
            EscalationLevel::Escalate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn policies_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../governance/demerzel/policies")
    }

    #[test]
    fn load_alignment_policy() {
        let p = AlignmentPolicy::load(&policies_dir().join("alignment-policy.yaml"))
            .expect("should load alignment policy");
        assert_eq!(p.name, "alignment-policy");
        assert_eq!(p.version, "1.0.0");
        assert!(!p.escalation_triggers.is_empty());
    }

    #[test]
    fn alignment_thresholds() {
        let p = AlignmentPolicy::load(&policies_dir().join("alignment-policy.yaml")).unwrap();
        let t = &p.confidence_thresholds;
        assert!((t.proceed_autonomously - 0.9).abs() < f64::EPSILON);
        assert!((t.proceed_with_note - 0.7).abs() < f64::EPSILON);
        assert!((t.ask_for_confirmation - 0.5).abs() < f64::EPSILON);
        assert!((t.escalate_to_human - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn escalation_autonomous() {
        let p = AlignmentPolicy::load(&policies_dir().join("alignment-policy.yaml")).unwrap();
        assert_eq!(p.should_escalate(0.95), EscalationLevel::Autonomous);
        assert_eq!(p.should_escalate(0.9), EscalationLevel::Autonomous);
    }

    #[test]
    fn escalation_proceed_with_note() {
        let p = AlignmentPolicy::load(&policies_dir().join("alignment-policy.yaml")).unwrap();
        assert_eq!(p.should_escalate(0.8), EscalationLevel::ProceedWithNote);
        assert_eq!(p.should_escalate(0.7), EscalationLevel::ProceedWithNote);
    }

    #[test]
    fn escalation_ask_confirmation() {
        let p = AlignmentPolicy::load(&policies_dir().join("alignment-policy.yaml")).unwrap();
        assert_eq!(p.should_escalate(0.6), EscalationLevel::AskConfirmation);
        assert_eq!(p.should_escalate(0.5), EscalationLevel::AskConfirmation);
    }

    #[test]
    fn escalation_escalate() {
        let p = AlignmentPolicy::load(&policies_dir().join("alignment-policy.yaml")).unwrap();
        assert_eq!(p.should_escalate(0.2), EscalationLevel::Escalate);
        assert_eq!(p.should_escalate(0.0), EscalationLevel::Escalate);
        assert_eq!(p.should_escalate(0.29), EscalationLevel::Escalate);
    }

    #[test]
    fn load_generic_policies() {
        let names = [
            "alignment-policy.yaml",
            "rollback-policy.yaml",
            "self-modification-policy.yaml",
        ];
        for name in &names {
            let p = Policy::load(&policies_dir().join(name));
            assert!(p.is_ok(), "failed to load policy: {}", name);
        }
    }

    #[test]
    fn generic_policy_has_extra_fields() {
        let p = Policy::load(&policies_dir().join("rollback-policy.yaml")).unwrap();
        assert_eq!(p.name, "rollback-policy");
        // The extra fields should contain "triggers", "procedure", etc.
        assert!(
            p.extra.get("triggers").is_some(),
            "should have triggers field"
        );
        assert!(
            p.extra.get("procedure").is_some(),
            "should have procedure field"
        );
    }
}
