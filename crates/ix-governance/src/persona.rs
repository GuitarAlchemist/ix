use crate::error::{GovernanceError, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::Path;

/// Voice characteristics for a persona.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Voice {
    /// Communication tone (e.g. "direct", "cautious").
    pub tone: String,
    /// How verbose the persona is (e.g. "concise", "detailed").
    pub verbosity: String,
    /// Communication style (e.g. "technical but approachable").
    pub style: String,
}

/// Interaction patterns with humans and other agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPatterns {
    /// Patterns for interacting with human users.
    #[serde(default, deserialize_with = "deserialize_string_or_map_vec")]
    pub with_humans: Vec<String>,
    /// Patterns for interacting with other agents.
    #[serde(default, deserialize_with = "deserialize_string_or_map_vec")]
    pub with_agents: Vec<String>,
}

/// Deserialize a list where each element can be either a plain string or a
/// single-entry YAML mapping (which happens when the string contains an
/// unquoted colon). Maps are flattened to `"key: value"` strings.
fn deserialize_string_or_map_vec<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;

    struct StringOrMapVecVisitor;

    impl<'de> de::Visitor<'de> for StringOrMapVecVisitor {
        type Value = Vec<String>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a sequence of strings or single-entry maps")
        }

        fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Vec<String>, A::Error>
        where
            A: de::SeqAccess<'de>,
        {
            let mut out = Vec::new();
            while let Some(val) = seq.next_element::<serde_yaml::Value>()? {
                match val {
                    serde_yaml::Value::String(s) => out.push(s),
                    serde_yaml::Value::Mapping(map) => {
                        for (k, v) in map {
                            let key = match k {
                                serde_yaml::Value::String(s) => s,
                                other => format!("{:?}", other),
                            };
                            let value = match v {
                                serde_yaml::Value::String(s) => s,
                                other => format!("{:?}", other),
                            };
                            out.push(format!("{}: {}", key, value));
                        }
                    }
                    other => {
                        out.push(format!("{:?}", other));
                    }
                }
            }
            Ok(out)
        }
    }

    deserializer.deserialize_seq(StringOrMapVecVisitor)
}

/// Provenance information for a persona.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// The source of the persona definition.
    pub source: String,
    /// When the persona was extracted or created.
    pub extraction_date: String,
    /// Optional archetype name.
    #[serde(default)]
    pub archetype: Option<String>,
}

/// A Demerzel agent persona loaded from YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    /// Persona name (e.g. "default", "kaizen-optimizer").
    pub name: String,
    /// Persona version.
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// Role of this persona.
    pub role: String,
    /// Domain of expertise.
    pub domain: String,
    /// List of capabilities.
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// List of constraints the persona must follow.
    #[serde(default)]
    pub constraints: Vec<String>,
    /// Voice characteristics.
    pub voice: Voice,
    /// Interaction patterns (optional).
    #[serde(default)]
    pub interaction_patterns: Option<InteractionPatterns>,
    /// Provenance information (optional).
    #[serde(default)]
    pub provenance: Option<Provenance>,
}

impl Persona {
    /// Load a persona from a YAML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        serde_yaml::from_str(&content).map_err(|e| GovernanceError::ParseError(e.to_string()))
    }

    /// Load a persona by name from a directory of persona YAML files.
    ///
    /// Looks for `{name}.persona.yaml` in the given directory.
    pub fn load_by_name(dir: &Path, name: &str) -> Result<Self> {
        let filename = format!("{}.persona.yaml", name);
        let path = dir.join(filename);
        if !path.exists() {
            return Err(GovernanceError::PersonaNotFound(name.to_string()));
        }
        Self::load(&path)
    }
}

/// List the names of all available personas in a directory.
///
/// Scans for files matching `*.persona.yaml` and returns the name portion.
pub fn list_personas(dir: &Path) -> Result<Vec<String>> {
    let mut names = Vec::new();
    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let file_name = entry.file_name();
        let name_str = file_name.to_string_lossy();
        if let Some(base) = name_str.strip_suffix(".persona.yaml") {
            names.push(base.to_string());
        }
    }
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn personas_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../governance/demerzel/personas")
    }

    #[test]
    fn load_default_persona() {
        let p = Persona::load_by_name(&personas_dir(), "default").expect("should load default");
        assert_eq!(p.name, "default");
        assert_eq!(p.version, "1.0.0");
        assert_eq!(p.role, "General-purpose assistant");
        assert_eq!(p.domain, "any");
        assert!(!p.capabilities.is_empty());
        assert!(!p.constraints.is_empty());
        assert_eq!(p.voice.tone, "direct");
    }

    #[test]
    fn load_all_personas() {
        let names = ["default", "kaizen-optimizer", "reflective-architect", "skeptical-auditor", "system-integrator"];
        for name in &names {
            let p = Persona::load_by_name(&personas_dir(), name);
            assert!(p.is_ok(), "failed to load persona: {}: {:?}", name, p.err());
        }
    }

    #[test]
    fn persona_not_found() {
        let result = Persona::load_by_name(&personas_dir(), "nonexistent");
        assert!(matches!(result, Err(GovernanceError::PersonaNotFound(_))));
    }

    #[test]
    fn list_all_personas() {
        let names = list_personas(&personas_dir()).expect("should list personas");
        assert!(names.len() >= 5, "expected at least 5 personas, got {}", names.len());
        assert!(names.contains(&"default".to_string()));
        assert!(names.contains(&"kaizen-optimizer".to_string()));
        assert!(names.contains(&"reflective-architect".to_string()));
        assert!(names.contains(&"skeptical-auditor".to_string()));
        assert!(names.contains(&"system-integrator".to_string()));
    }

    #[test]
    fn persona_has_voice() {
        let p = Persona::load_by_name(&personas_dir(), "default").unwrap();
        assert!(!p.voice.tone.is_empty());
        assert!(!p.voice.verbosity.is_empty());
        assert!(!p.voice.style.is_empty());
    }

    #[test]
    fn persona_has_interaction_patterns() {
        let p = Persona::load_by_name(&personas_dir(), "default").unwrap();
        let patterns = p.interaction_patterns.expect("default should have interaction_patterns");
        assert!(!patterns.with_humans.is_empty());
        assert!(!patterns.with_agents.is_empty());
    }

    #[test]
    fn persona_has_provenance() {
        let p = Persona::load_by_name(&personas_dir(), "default").unwrap();
        let prov = p.provenance.expect("default should have provenance");
        assert!(!prov.source.is_empty());
        assert!(!prov.extraction_date.is_empty());
    }
}
