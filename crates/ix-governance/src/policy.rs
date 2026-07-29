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
    ///
    /// Any `ref:confidence#<key>` token in the body is resolved to its number,
    /// so consumers — and `ix describe policy` — see the effective threshold
    /// rather than the indirection that produced it.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut policy: Self = serde_yaml::from_str(&content)
            .map_err(|e| GovernanceError::ParseError(e.to_string()))?;
        let ladder = load_confidence_ladder(path)?;
        resolve_confidence_refs(&mut policy.extra, "", ladder.as_ref())?;
        Ok(policy)
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

/// Prefix marking a threshold that defers to the canonical ladder.
const CONFIDENCE_REF_PREFIX: &str = "ref:confidence#";

/// Canonical ladder path, relative to the Demerzel root that contains
/// `policies/`.
const CONFIDENCE_LADDER_REL: &str = "logic/confidence-thresholds.yaml";

/// A threshold as written in a policy file.
///
/// Demerzel single-sourced the confidence ladder (2026-06-21 architecture
/// review), so policies now carry `ref:confidence#<key>` tokens instead of
/// restating numbers that had drifted across 8+ files. Literals are still
/// accepted, so a policy pinned before that change keeps loading.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ThresholdSpec {
    Literal(f64),
    Ref(String),
}

/// One rung of `logic/confidence-thresholds.yaml`. Only `value` is consumed
/// here; `operator` and `meaning` are documentation for humans.
#[derive(Debug, Clone, Deserialize)]
struct LadderRung {
    value: f64,
}

/// The canonical confidence ladder.
#[derive(Debug, Clone, Deserialize)]
struct ConfidenceLadder {
    thresholds: std::collections::BTreeMap<String, LadderRung>,
}

impl ThresholdSpec {
    /// Resolve to a number, following a `ref:` token into the ladder.
    ///
    /// A dangling key is an error rather than a default: Demerzel's
    /// `build_manifest.py` fails CI on one, and silently substituting a
    /// plausible number here would let ix act on a threshold nobody wrote.
    fn resolve(&self, field: &str, ladder: Option<&ConfidenceLadder>) -> Result<f64> {
        match self {
            ThresholdSpec::Literal(v) => Ok(*v),
            ThresholdSpec::Ref(token) => {
                let key = token.strip_prefix(CONFIDENCE_REF_PREFIX).ok_or_else(|| {
                    GovernanceError::ParseError(format!(
                        "{field}: expected a number or a `{CONFIDENCE_REF_PREFIX}<key>` token, got {token:?}"
                    ))
                })?;
                let ladder = ladder.ok_or_else(|| {
                    GovernanceError::ParseError(format!(
                        "{field} references {token:?} but {CONFIDENCE_LADDER_REL} could not be read"
                    ))
                })?;
                ladder
                    .thresholds
                    .get(key)
                    .map(|rung| rung.value)
                    .ok_or_else(|| {
                        GovernanceError::ParseError(format!(
                            "{field}: dangling confidence key {key:?} — not a rung of {CONFIDENCE_LADDER_REL}"
                        ))
                    })
            }
        }
    }
}

/// Read the canonical ladder that sits beside the policy's directory.
///
/// Returns `Ok(None)` when the file is absent, so a policy carrying only
/// literals does not start failing because of a file it never referenced. A
/// present-but-malformed ladder is still an error.
fn load_confidence_ladder(policy_path: &Path) -> Result<Option<ConfidenceLadder>> {
    let Some(root) = policy_path.parent().and_then(Path::parent) else {
        return Ok(None);
    };
    let ladder_path = root.join(CONFIDENCE_LADDER_REL);
    match std::fs::read_to_string(&ladder_path) {
        Ok(text) => serde_yaml::from_str::<ConfidenceLadder>(&text)
            .map(Some)
            .map_err(|e| GovernanceError::ParseError(format!("{}: {e}", ladder_path.display()))),
        Err(_) => Ok(None),
    }
}

/// Replace every `ref:confidence#<key>` string in `value` with its number.
///
/// Walks the whole document rather than a fixed field list: the tokens are not
/// confined to `confidence_thresholds`, and a policy that grows a new one
/// should not need a matching code change here.
fn resolve_confidence_refs(
    value: &mut serde_json::Value,
    field: &str,
    ladder: Option<&ConfidenceLadder>,
) -> Result<()> {
    match value {
        serde_json::Value::String(s) if s.starts_with(CONFIDENCE_REF_PREFIX) => {
            let resolved = ThresholdSpec::Ref(s.clone()).resolve(field, ladder)?;
            *value = serde_json::Value::from(resolved);
            Ok(())
        }
        serde_json::Value::Array(items) => {
            items.iter_mut().enumerate().try_for_each(|(i, item)| {
                resolve_confidence_refs(item, &format!("{field}[{i}]"), ladder)
            })
        }
        serde_json::Value::Object(map) => map.iter_mut().try_for_each(|(key, item)| {
            let path = if field.is_empty() {
                key.clone()
            } else {
                format!("{field}.{key}")
            };
            resolve_confidence_refs(item, &path, ladder)
        }),
        _ => Ok(()),
    }
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfidenceThresholds {
    proceed_autonomously: ThresholdSpec,
    proceed_with_note: ThresholdSpec,
    ask_for_confirmation: ThresholdSpec,
    escalate_to_human: ThresholdSpec,
}

#[derive(Debug, Clone, Deserialize)]
struct RawAlignmentPolicy {
    name: String,
    version: String,
    description: String,
    confidence_thresholds: RawConfidenceThresholds,
    escalation_triggers: Vec<String>,
}

impl AlignmentPolicy {
    /// Load the alignment policy from a YAML file.
    ///
    /// Thresholds may be literals or `ref:confidence#<key>` tokens; tokens are
    /// resolved against `<demerzel>/logic/confidence-thresholds.yaml`, located
    /// relative to the policy file so callers keep passing just the one path.
    // @ai:invariant a ref: token resolves to the ladder value, and a dangling key errors rather than defaulting [T:test conf:0.95 src:test_dangling_confidence_ref_is_an_error]
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let raw: RawAlignmentPolicy = serde_yaml::from_str(&content)
            .map_err(|e| GovernanceError::ParseError(e.to_string()))?;

        let ladder = load_confidence_ladder(path)?;

        let t = &raw.confidence_thresholds;
        Ok(Self {
            name: raw.name,
            version: raw.version,
            description: raw.description,
            confidence_thresholds: ConfidenceThresholds {
                proceed_autonomously: t
                    .proceed_autonomously
                    .resolve("proceed_autonomously", ladder.as_ref())?,
                proceed_with_note: t
                    .proceed_with_note
                    .resolve("proceed_with_note", ladder.as_ref())?,
                ask_for_confirmation: t
                    .ask_for_confirmation
                    .resolve("ask_for_confirmation", ladder.as_ref())?,
                escalate_to_human: t
                    .escalate_to_human
                    .resolve("escalate_to_human", ladder.as_ref())?,
            },
            escalation_triggers: raw.escalation_triggers,
        })
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

    /// Write a policy + ladder pair into a temp dir laid out the way Demerzel
    /// is (`policies/` and `logic/` as siblings), and return the policy path.
    fn scratch_policy(dir: &Path, thresholds: &str, ladder: Option<&str>) -> PathBuf {
        std::fs::create_dir_all(dir.join("policies")).unwrap();
        let policy = dir.join("policies/alignment-policy.yaml");
        std::fs::write(
            &policy,
            format!(
                "name: alignment-policy\nversion: 1.0.0\ndescription: test\n\
                 confidence_thresholds:\n{thresholds}\n\
                 escalation_triggers:\n  - something\n"
            ),
        )
        .unwrap();
        if let Some(l) = ladder {
            std::fs::create_dir_all(dir.join("logic")).unwrap();
            std::fs::write(dir.join(CONFIDENCE_LADDER_REL), l).unwrap();
        }
        policy
    }

    const FULL_LADDER: &str = "schema_version: 1\nthresholds:\n\
        \x20 autonomous:\n    value: 0.9\n    operator: \">=\"\n\
        \x20 with_note:\n    value: 0.7\n    operator: \">=\"\n\
        \x20 ask_confirmation:\n    value: 0.5\n    operator: \">=\"\n\
        \x20 escalate:\n    value: 0.3\n    operator: \">=\"\n";

    const REF_THRESHOLDS: &str = "  proceed_autonomously: ref:confidence#autonomous\n\
        \x20 proceed_with_note: ref:confidence#with_note\n\
        \x20 ask_for_confirmation: ref:confidence#ask_confirmation\n\
        \x20 escalate_to_human: ref:confidence#escalate";

    #[test]
    fn literal_thresholds_still_load_without_a_ladder() {
        // A policy pinned before the single-sourcing change must keep working,
        // ladder file absent and all.
        let tmp = tempfile::tempdir().unwrap();
        let p = scratch_policy(
            tmp.path(),
            "  proceed_autonomously: 0.9\n  proceed_with_note: 0.7\n\
             \x20 ask_for_confirmation: 0.5\n  escalate_to_human: 0.3",
            None,
        );
        let loaded = AlignmentPolicy::load(&p).expect("literals need no ladder");
        assert!((loaded.confidence_thresholds.proceed_autonomously - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn ref_tokens_resolve_against_the_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let p = scratch_policy(tmp.path(), REF_THRESHOLDS, Some(FULL_LADDER));
        let loaded = AlignmentPolicy::load(&p).expect("refs resolve");
        let t = &loaded.confidence_thresholds;
        assert!((t.proceed_autonomously - 0.9).abs() < f64::EPSILON);
        assert!((t.proceed_with_note - 0.7).abs() < f64::EPSILON);
        assert!((t.ask_for_confirmation - 0.5).abs() < f64::EPSILON);
        assert!((t.escalate_to_human - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dangling_confidence_ref_is_an_error() {
        // Demerzel's build_manifest fails CI on a dangling key; ix must not
        // quietly substitute a plausible number for a rung nobody wrote.
        let tmp = tempfile::tempdir().unwrap();
        let p = scratch_policy(
            tmp.path(),
            "  proceed_autonomously: ref:confidence#nonexistent\n\
             \x20 proceed_with_note: ref:confidence#with_note\n\
             \x20 ask_for_confirmation: ref:confidence#ask_confirmation\n\
             \x20 escalate_to_human: ref:confidence#escalate",
            Some(FULL_LADDER),
        );
        let err = AlignmentPolicy::load(&p).expect_err("dangling key must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("nonexistent"),
            "error should name the key: {msg}"
        );
    }

    #[test]
    fn ref_token_without_a_ladder_file_is_an_error() {
        // Failing loudly beats falling back to a stale default ladder.
        let tmp = tempfile::tempdir().unwrap();
        let p = scratch_policy(tmp.path(), REF_THRESHOLDS, None);
        assert!(AlignmentPolicy::load(&p).is_err());
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
