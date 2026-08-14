//! Effect plans and commit receipts for strict IXQL execution.

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpectedState {
    Any,
    Absent,
    Sha256(String),
}

impl ExpectedState {
    pub(crate) fn parse(value: &str) -> Result<Self, EffectError> {
        match value {
            "any" => Ok(Self::Any),
            "absent" => Ok(Self::Absent),
            value if value.starts_with("sha256:") => {
                let digest = &value[7..];
                if digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    Ok(Self::Sha256(digest.to_ascii_lowercase()))
                } else {
                    Err(EffectError::InvalidContract(
                        "expected_state sha256 digest must contain 64 hex characters".into(),
                    ))
                }
            }
            _ => Err(EffectError::InvalidContract(format!(
                "unsupported expected_state `{value}`"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Compensation {
    Delete,
    RestorePrevious,
    None,
}

impl Compensation {
    pub(crate) fn parse(value: &str) -> Result<Self, EffectError> {
        match value {
            "delete" => Ok(Self::Delete),
            "restore_previous" => Ok(Self::RestorePrevious),
            "none" => Ok(Self::None),
            _ => Err(EffectError::InvalidContract(format!(
                "unsupported compensation `{value}`"
            ))),
        }
    }
}

/// One fully evaluated mutation request.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectIntent {
    pub path: String,
    pub value: Value,
    pub idempotency_key: String,
    pub expected_state: ExpectedState,
    pub compensation: Compensation,
    pub authority: String,
}

impl EffectIntent {
    pub fn digest(&self) -> String {
        digest_bytes(&serde_json::to_vec(self).expect("effect intent is serializable"))
    }
}

/// A batch that has been evaluated but has not mutated an adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectPlan {
    source_digest: String,
    verification_policy_digest: String,
    plan_digest: String,
    effects: Vec<EffectIntent>,
}

impl EffectPlan {
    pub(crate) fn new(
        source_digest: &str,
        verification_policy_digest: &str,
        effects: Vec<EffectIntent>,
    ) -> Self {
        let mut bytes = source_digest.as_bytes().to_vec();
        bytes.extend(verification_policy_digest.as_bytes());
        bytes.extend(serde_json::to_vec(&effects).expect("effect plan is serializable"));
        let plan_digest = digest_bytes(&bytes);
        Self {
            source_digest: source_digest.into(),
            verification_policy_digest: verification_policy_digest.into(),
            plan_digest,
            effects,
        }
    }

    pub fn source_digest(&self) -> &str {
        &self.source_digest
    }

    pub fn plan_digest(&self) -> &str {
        &self.plan_digest
    }

    pub fn verification_policy_digest(&self) -> &str {
        &self.verification_policy_digest
    }

    pub fn effects(&self) -> &[EffectIntent] {
        &self.effects
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectReceipt {
    pub idempotency_key: String,
    pub intent_digest: String,
    pub path: String,
    pub before_digest: Option<String>,
    pub after_digest: String,
    pub authority: String,
    pub expected_state: ExpectedState,
    pub compensation: Compensation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionReceipt {
    pub plan_digest: String,
    pub verification_policy_digest: String,
    pub effects: Vec<EffectReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EffectError {
    #[error("invalid effect contract: {0}")]
    InvalidContract(String),
    #[error("idempotency key `{key}` was already used for a different intent")]
    IdempotencyConflict { key: String },
    #[error("expected state did not hold for `{path}`: {reason}")]
    ExpectedState { path: String, reason: String },
    #[error("adapter cannot atomically commit this plan: {0}")]
    UnsupportedAtomicCommit(String),
}

/// The authority-bearing seam.
// @ai:invariant plan_verified never calls EffectAdapter::commit; only an explicit caller can cross this seam [T:test conf:0.95 src:compiler_verifier_tests::verified_effects_are_staged_then_committed_atomically_and_idempotently]
pub trait EffectAdapter: Send + Sync {
    fn commit(&self, plan: &EffectPlan) -> Result<ExecutionReceipt, EffectError>;
}

pub(crate) fn value_digest(value: &Value) -> String {
    digest_bytes(&serde_json::to_vec(value).expect("JSON value is serializable"))
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
