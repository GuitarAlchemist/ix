//! `ix.yaml` pipeline definition format.
//!
//! A `PipelineSpec` is the on-disk form of a DAG. Each stage references a
//! skill from `ix-registry` by name and provides an `args` JSON blob.
//! Inter-stage data flow is expressed with `{"from": "stage_id[.key]"}`
//! references anywhere inside `args`.
//!
//! Example:
//!
//! ```yaml
//! version: "1"
//! params:
//!   k: 3
//! stages:
//!   load:
//!     skill: stats
//!     args: { data: [1.0, 2.0, 3.0, 4.0, 5.0] }
//!   audit:
//!     skill: governance.check
//!     args: { action: "use computed mean" }
//!     deps: [load]
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Root document of an `ix.yaml` pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSpec {
    /// Schema version — currently always `"1"`.
    #[serde(default = "default_version")]
    pub version: String,

    /// Named parameter bag: default values for `{"param": "NAME"}` object-refs in
    /// stage args, bound at run time by `lower::bind_params` and overridable with
    /// `ix pipeline run --param NAME=<json|@file>`. A `null` default means the
    /// param MUST be supplied at run. This is how the NL compiler emits a spec for
    /// a request whose data isn't inline ("reduce this dataset"). (A separate
    /// `${params.NAME}` string-interpolation form is reserved, not yet built.)
    #[serde(default)]
    pub params: BTreeMap<String, Value>,

    /// Stages keyed by string id. Order is determined by `deps`, not by the
    /// map's iteration order.
    pub stages: BTreeMap<String, StageSpec>,

    /// Opaque editor metadata (node positions, etc.). Ignored by the executor.
    #[serde(rename = "x-editor", default)]
    pub x_editor: Value,
}

fn default_version() -> String {
    "1".into()
}

/// One stage: calls a registered skill with a blob of JSON `args`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageSpec {
    /// Dotted skill name — must exist in `ix-registry`.
    pub skill: String,

    /// Static JSON input blob. May contain `{"from": "stage_id"}` references
    /// that `lower()` resolves at build time.
    #[serde(default = "default_args")]
    pub args: Value,

    /// Explicit upstream dependencies. Extra deps implied by `from:`
    /// references are merged automatically during lowering.
    #[serde(default)]
    pub deps: Vec<String>,

    /// Override the default caching behavior for this stage.
    #[serde(default)]
    pub cache: Option<bool>,
}

fn default_args() -> Value {
    Value::Object(Default::default())
}

/// Schema / parse errors.
#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    #[error("YAML parse error: {0}")]
    Parse(#[from] serde_yaml::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported version: {0} (expected '1')")]
    UnsupportedVersion(String),

    #[error("stage '{stage}' references unknown upstream stage '{missing}'")]
    UnknownDep { stage: String, missing: String },

    #[error("stage '{0}' has empty skill name")]
    EmptySkill(String),
}

impl PipelineSpec {
    /// Parse a YAML string into a `PipelineSpec`.
    pub fn from_yaml_str(yaml: &str) -> Result<Self, SpecError> {
        let spec: PipelineSpec = serde_yaml::from_str(yaml)?;
        spec.validate_shape()?;
        Ok(spec)
    }

    /// Load a spec from a file path.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, SpecError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_yaml_str(&text)
    }

    /// Serialize to a YAML string.
    pub fn to_yaml_string(&self) -> Result<String, SpecError> {
        Ok(serde_yaml::to_string(self)?)
    }

    /// Shallow structural validation — version, non-empty skill names,
    /// deps reference existing stages. Does NOT hit the registry.
    pub fn validate_shape(&self) -> Result<(), SpecError> {
        if self.version != "1" {
            return Err(SpecError::UnsupportedVersion(self.version.clone()));
        }
        for (id, stage) in &self.stages {
            if stage.skill.trim().is_empty() {
                return Err(SpecError::EmptySkill(id.clone()));
            }
            for dep in &stage.deps {
                if !self.stages.contains_key(dep) {
                    return Err(SpecError::UnknownDep {
                        stage: id.clone(),
                        missing: dep.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// A minimal scaffold with one `stats` stage — what `ix pipeline new`
    /// writes to disk.
    pub fn scaffold(name: &str) -> Self {
        let mut stages = BTreeMap::new();
        stages.insert(
            "load".into(),
            StageSpec {
                skill: "stats".into(),
                args: serde_json::json!({ "data": [1.0, 2.0, 3.0, 4.0, 5.0] }),
                deps: vec![],
                cache: None,
            },
        );
        let mut x_editor = serde_json::Map::new();
        x_editor.insert("name".into(), Value::String(name.into()));
        PipelineSpec {
            version: "1".into(),
            params: BTreeMap::new(),
            stages,
            x_editor: Value::Object(x_editor),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_minimal_spec() {
        let yaml = r#"
version: "1"
stages:
  ingest:
    skill: stats
    args:
      data: [1.0, 2.0, 3.0]
"#;
        let spec = PipelineSpec::from_yaml_str(yaml).unwrap();
        assert_eq!(spec.stages.len(), 1);
        assert_eq!(spec.stages["ingest"].skill, "stats");
    }

    #[test]
    fn roundtrip_with_deps_and_from() {
        let yaml = r#"
version: "1"
stages:
  load:
    skill: stats
    args: { data: [1.0, 2.0, 3.0] }
  check:
    skill: governance.check
    args: { action: "process {from: load}" }
    deps: [load]
"#;
        let spec = PipelineSpec::from_yaml_str(yaml).unwrap();
        assert_eq!(spec.stages["check"].deps, vec!["load"]);
    }

    #[test]
    fn rejects_unknown_dep() {
        let yaml = r#"
version: "1"
stages:
  a:
    skill: stats
    deps: [missing]
"#;
        let err = PipelineSpec::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, SpecError::UnknownDep { .. }));
    }

    #[test]
    fn rejects_wrong_version() {
        let yaml = r#"
version: "2"
stages: {}
"#;
        let err = PipelineSpec::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, SpecError::UnsupportedVersion(_)));
    }

    #[test]
    fn rejects_empty_skill() {
        let yaml = r#"
version: "1"
stages:
  bad:
    skill: ""
"#;
        let err = PipelineSpec::from_yaml_str(yaml).unwrap_err();
        assert!(matches!(err, SpecError::EmptySkill(_)));
    }

    #[test]
    fn scaffold_is_parseable() {
        let spec = PipelineSpec::scaffold("demo");
        let yaml = spec.to_yaml_string().unwrap();
        let back = PipelineSpec::from_yaml_str(&yaml).unwrap();
        assert_eq!(back.stages.len(), 1);
        assert_eq!(back.stages["load"].skill, "stats");
    }

    #[test]
    fn x_editor_is_preserved() {
        let yaml = r#"
version: "1"
stages:
  a:
    skill: stats
x-editor:
  layout:
    a: { x: 10, y: 20 }
"#;
        let spec = PipelineSpec::from_yaml_str(yaml).unwrap();
        assert!(spec.x_editor.get("layout").is_some());
    }
}
