//! At-rest schema gate.
//!
//! BAML governs what an LLM hands back *in flight*. The canonical JSON Schemas
//! govern what is allowed to reach disk. This module is the second half: every
//! `ix.io.write` is matched against the registered schemas, and a write that
//! fails one is refused — the pipeline stops rather than persisting a value
//! that downstream consumers would then have to defend against.

use jsonschema::Validator;
use serde_json::Value;

use crate::path::normalize_lossy;

/// A write that a registered schema rejected.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("write to `{path}` violates schema `{schema}`: {}", errors.join("; "))]
pub struct SchemaViolation {
    pub path: String,
    pub schema: String,
    pub errors: Vec<String>,
}

struct Rule {
    prefix: String,
    name: String,
    validator: Validator,
}

/// Path-prefix → schema bindings, consulted on every write.
///
/// An empty gate lets every write through. That is the honest default for a
/// language whose pipelines write many kinds of artifact, but it means the
/// gate only protects what a caller has explicitly registered — which is why
/// [`Executor::schema_gate`](crate::Executor::schema_gate) is part of the
/// public setup surface rather than something the executor guesses at.
#[derive(Default)]
pub struct SchemaGate {
    rules: Vec<Rule>,
}

impl SchemaGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a compiled schema to every path starting with `prefix`.
    ///
    /// `name` is what shows up in the violation message, so make it the schema
    /// file's name.
    pub fn register(
        &mut self,
        prefix: impl Into<String>,
        name: impl Into<String>,
        schema: &Value,
    ) -> Result<&mut Self, String> {
        let name = name.into();
        let validator = jsonschema::draft202012::new(schema)
            .map_err(|e| format!("schema `{name}` does not compile: {e}"))?;
        self.rules.push(Rule {
            // Both sides of the comparison go through the same normalizer, so
            // a rule registered as `state\quality\` matches a path built as
            // `state/./quality/…`.
            prefix: normalize_prefix(&prefix.into()),
            name,
            validator,
        });
        Ok(self)
    }

    /// Validate a value against every schema whose prefix matches the path.
    ///
    /// All matching rules must pass; the first violation is reported. Matching
    /// several is not an error — a broad envelope schema and a narrow one can
    /// legitimately both apply to the same directory.
    pub fn check(&self, path: &str, value: &Value) -> Result<(), SchemaViolation> {
        let normalized = normalize_lossy(path);
        for rule in &self.rules {
            if !normalized.starts_with(&rule.prefix) {
                continue;
            }
            let errors: Vec<String> = rule
                .validator
                .iter_errors(value)
                .map(|e| {
                    let loc = e.instance_path.to_string();
                    if loc.is_empty() {
                        format!("{e}")
                    } else {
                        format!("{e} (at `{loc}`)")
                    }
                })
                .collect();
            if !errors.is_empty() {
                return Err(SchemaViolation {
                    path: normalized,
                    schema: rule.name.clone(),
                    errors,
                });
            }
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// A prefix is a directory stub, not a whole path, so it is normalized with the
/// trailing separator preserved — `state/quality/` must not become
/// `state/quality`, or it would also match `state/quality-archive/`.
fn normalize_prefix(prefix: &str) -> String {
    let trailing = prefix.ends_with('/') || prefix.ends_with('\\');
    let mut normalized = normalize_lossy(prefix);
    if trailing && !normalized.ends_with('/') {
        normalized.push('/');
    }
    normalized
}

impl std::fmt::Debug for SchemaGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bindings: Vec<(&str, &str)> = self
            .rules
            .iter()
            .map(|r| (r.prefix.as_str(), r.name.as_str()))
            .collect();
        f.debug_struct("SchemaGate")
            .field("bindings", &bindings)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn gate() -> SchemaGate {
        let mut gate = SchemaGate::new();
        gate.register(
            "state/quality/verdicts/",
            "qa-verdict.schema.json",
            &json!({
                "type": "object",
                "required": ["schema_version", "verdict"],
                "properties": {
                    "schema_version": {"const": 1},
                    "verdict": {"enum": ["pass", "fail", "informational"]}
                }
            }),
        )
        .unwrap();
        gate
    }

    #[test]
    fn a_conforming_write_passes() {
        assert!(gate()
            .check(
                "state/quality/verdicts/ga/main/v1.json",
                &json!({"schema_version": 1, "verdict": "informational"})
            )
            .is_ok());
    }

    #[test]
    fn a_wrong_enum_value_is_refused_and_names_the_schema() {
        let err = gate()
            .check(
                "state/quality/verdicts/ga/main/v1.json",
                &json!({"schema_version": 1, "verdict": "probably fine"}),
            )
            .unwrap_err();
        assert_eq!("qa-verdict.schema.json", err.schema);
        assert!(!err.errors.is_empty());
    }

    #[test]
    fn paths_outside_the_prefix_are_untouched() {
        assert!(gate().check("state/evolution/log.json", &json!(42)).is_ok());
    }

    #[test]
    fn dot_segments_cannot_dodge_a_registered_prefix() {
        // The bypass Codex found on #274: the filesystem resolves this into the
        // protected directory, so the gate has to see it that way too.
        assert!(gate()
            .check(
                "state/./quality//verdicts/ga/v1.json",
                &json!({"schema_version": 1, "verdict": "probably fine"})
            )
            .is_err());
    }

    #[test]
    fn a_prefix_does_not_match_a_sibling_directory_that_shares_its_name() {
        assert!(gate()
            .check("state/quality/verdicts-archive/v1.json", &json!(42))
            .is_ok());
    }

    #[test]
    fn backslash_paths_match_the_same_prefix() {
        // Windows-built interpolations arrive with backslashes; the gate must
        // not fall open just because of the separator.
        assert!(gate()
            .check(
                "state\\quality\\verdicts\\ga\\main\\v1.json",
                &json!({"schema_version": 2, "verdict": "pass"})
            )
            .is_err());
    }

    #[test]
    fn an_uncompilable_schema_is_rejected_at_registration() {
        let mut gate = SchemaGate::new();
        assert!(gate.register("x/", "bad", &json!({"type": 17})).is_err());
    }
}
