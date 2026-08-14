//! Pure policy verification for compiled IXQL programs.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::compiler::{Capability, Diagnostic, TypedProgram};

/// Verification limits and the capabilities a caller is willing to grant.
#[derive(Debug, Clone)]
pub struct VerificationPolicy {
    allowed_capabilities: BTreeSet<Capability>,
    max_statements: usize,
    max_effects: usize,
    require_fresh_reads: bool,
    fresh_artifacts: BTreeMap<String, String>,
    require_declared_effects: bool,
    allowed_authorities: BTreeSet<String>,
    schema_gate_digest: Option<String>,
}

impl VerificationPolicy {
    pub fn new(capabilities: impl IntoIterator<Item = Capability>) -> Self {
        Self {
            allowed_capabilities: capabilities.into_iter().collect(),
            max_statements: usize::MAX,
            max_effects: usize::MAX,
            require_fresh_reads: false,
            fresh_artifacts: BTreeMap::new(),
            require_declared_effects: false,
            allowed_authorities: BTreeSet::new(),
            schema_gate_digest: None,
        }
    }

    pub fn with_max_statements(mut self, limit: usize) -> Self {
        self.max_statements = limit;
        self
    }

    pub fn with_max_effects(mut self, limit: usize) -> Self {
        self.max_effects = limit;
        self
    }

    pub fn require_fresh_reads(mut self) -> Self {
        self.require_fresh_reads = true;
        self
    }

    /// Bind a path's fresh status to the exact revision digest that was checked.
    pub fn with_fresh_artifact(
        mut self,
        path: impl Into<String>,
        revision_digest: impl Into<String>,
    ) -> Self {
        let path = path.into();
        let canonical_path = crate::path::normalize(&path).unwrap_or(path);
        self.fresh_artifacts
            .insert(canonical_path, revision_digest.into());
        self
    }

    pub fn require_declared_effects(mut self) -> Self {
        self.require_declared_effects = true;
        self
    }

    /// Grant one exact authority label to mutation declarations.
    pub fn allow_authority(mut self, authority: impl Into<String>) -> Self {
        self.allowed_authorities.insert(authority.into());
        self
    }

    /// Bind strict evaluation to one exact set of path/schema registrations.
    pub fn with_schema_gate_digest(mut self, digest: impl Into<String>) -> Self {
        self.schema_gate_digest = Some(digest.into());
        self
    }

    fn digest(&self) -> String {
        #[derive(Serialize)]
        struct PolicyIdentity<'a> {
            allowed_capabilities: &'a BTreeSet<Capability>,
            max_statements: usize,
            max_effects: usize,
            require_fresh_reads: bool,
            fresh_artifacts: &'a BTreeMap<String, String>,
            require_declared_effects: bool,
            allowed_authorities: &'a BTreeSet<String>,
            schema_gate_digest: &'a Option<String>,
        }

        let canonical = serde_json::to_vec(&PolicyIdentity {
            allowed_capabilities: &self.allowed_capabilities,
            max_statements: self.max_statements,
            max_effects: self.max_effects,
            require_fresh_reads: self.require_fresh_reads,
            fresh_artifacts: &self.fresh_artifacts,
            require_declared_effects: self.require_declared_effects,
            allowed_authorities: &self.allowed_authorities,
            schema_gate_digest: &self.schema_gate_digest,
        })
        .expect("verification policy identity is serializable");
        format!("{:x}", Sha256::digest(canonical))
    }
}

/// All policy failures from one verification pass.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("IXQL verification failed with {count} diagnostic(s)", count = .0.len())]
pub struct VerificationDiagnostics(pub Vec<Diagnostic>);

impl VerificationDiagnostics {
    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.0.iter()
    }
}

/// A compiled program that satisfied one explicit policy.
#[derive(Debug, Clone)]
pub struct VerifiedProgram {
    program: TypedProgram,
    verification_policy_digest: String,
    fresh_artifacts: BTreeMap<String, String>,
    schema_gate_digest: Option<String>,
}

impl VerifiedProgram {
    pub fn program(&self) -> &TypedProgram {
        &self.program
    }

    pub fn into_program(self) -> TypedProgram {
        self.program
    }

    pub fn verification_policy_digest(&self) -> &str {
        &self.verification_policy_digest
    }

    pub(crate) fn fresh_artifacts(&self) -> &BTreeMap<String, String> {
        &self.fresh_artifacts
    }

    pub(crate) fn expected_schema_gate_digest(&self) -> Option<&str> {
        self.schema_gate_digest.as_deref()
    }
}

pub struct Verifier {
    policy: VerificationPolicy,
}

impl Verifier {
    pub fn new(policy: VerificationPolicy) -> Self {
        Self { policy }
    }

    pub fn verify(
        &self,
        program: TypedProgram,
    ) -> Result<VerifiedProgram, VerificationDiagnostics> {
        let mut diagnostics = Vec::new();

        for capability in program.required_capabilities() {
            if !self.policy.allowed_capabilities.contains(capability) {
                diagnostics.push(Diagnostic {
                    code: "IXQL_FORBIDDEN_CAPABILITY",
                    message: format!("capability {capability:?} is not allowed by policy"),
                });
            }
        }
        if program.statement_count() > self.policy.max_statements {
            diagnostics.push(Diagnostic {
                code: "IXQL_STATEMENT_BUDGET",
                message: format!(
                    "program has {} statements; policy allows {}",
                    program.statement_count(),
                    self.policy.max_statements
                ),
            });
        }
        if program.effect_count() > self.policy.max_effects {
            diagnostics.push(Diagnostic {
                code: "IXQL_EFFECT_BUDGET",
                message: format!(
                    "program has {} effects; policy allows {}",
                    program.effect_count(),
                    self.policy.max_effects
                ),
            });
        }
        if self.policy.require_fresh_reads {
            for artifact in program.artifact_reads() {
                match &artifact.path {
                    None => diagnostics.push(Diagnostic {
                        code: "IXQL_UNVERIFIABLE_FRESHNESS",
                        message: "dynamic artifact path has no digest-bound freshness evidence"
                            .into(),
                    }),
                    Some(path) => match self.policy.fresh_artifacts.get(path) {
                        None => diagnostics.push(Diagnostic {
                            code: "IXQL_STALE_OR_UNKNOWN_INPUT",
                            message: format!("{path} has no fresh revision evidence"),
                        }),
                        Some(digest)
                            if digest.len() != 64
                                || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
                        {
                            diagnostics.push(Diagnostic {
                                code: "IXQL_INVALID_FRESHNESS_DIGEST",
                                message: format!(
                                    "{path} freshness evidence is not a SHA-256 digest"
                                ),
                            });
                        }
                        Some(_) => {}
                    },
                }
            }
        }
        if self.policy.require_declared_effects {
            for effect in program.write_effects() {
                for (present, code, field) in [
                    (
                        effect.idempotency_key.is_some(),
                        "IXQL_EFFECT_IDEMPOTENCY_REQUIRED",
                        "idempotency_key",
                    ),
                    (
                        effect.expected_state.is_some(),
                        "IXQL_EFFECT_EXPECTED_STATE_REQUIRED",
                        "expected_state",
                    ),
                    (
                        effect.compensation.is_some(),
                        "IXQL_EFFECT_COMPENSATION_REQUIRED",
                        "compensation",
                    ),
                    (
                        effect.authority.is_some(),
                        "IXQL_EFFECT_AUTHORITY_REQUIRED",
                        "authority",
                    ),
                ] {
                    if !present {
                        diagnostics.push(Diagnostic {
                            code,
                            message: format!("ix.io.write requires literal `{field}` metadata"),
                        });
                    }
                }
            }
            if !program.write_effects().is_empty() {
                match self.policy.schema_gate_digest.as_deref() {
                    None => diagnostics.push(Diagnostic {
                        code: "IXQL_SCHEMA_GATE_REQUIRED",
                        message: "strict mutations require an exact schema-gate digest".into(),
                    }),
                    Some(digest)
                        if digest.len() != 64
                            || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
                    {
                        diagnostics.push(Diagnostic {
                            code: "IXQL_INVALID_SCHEMA_GATE_DIGEST",
                            message: "schema-gate identity must be a 64-character SHA-256 digest"
                                .into(),
                        });
                    }
                    Some(_) => {}
                }
            }
        }

        for effect in program.write_effects() {
            match &effect.authority {
                Some(authority) if !self.policy.allowed_authorities.contains(authority) => {
                    diagnostics.push(Diagnostic {
                        code: "IXQL_FORBIDDEN_AUTHORITY",
                        message: format!("authority `{authority}` is not granted by policy"),
                    });
                }
                None if !self.policy.allowed_authorities.is_empty() => {
                    diagnostics.push(Diagnostic {
                        code: "IXQL_EFFECT_AUTHORITY_REQUIRED",
                        message:
                            "an authority allowlist requires a literal `authority` declaration"
                                .into(),
                    });
                }
                _ => {}
            }
        }

        if diagnostics.is_empty() {
            Ok(VerifiedProgram {
                program,
                verification_policy_digest: self.policy.digest(),
                fresh_artifacts: if self.policy.require_fresh_reads {
                    self.policy.fresh_artifacts.clone()
                } else {
                    BTreeMap::new()
                },
                schema_gate_digest: self.policy.schema_gate_digest.clone(),
            })
        } else {
            Err(VerificationDiagnostics(diagnostics))
        }
    }
}
