//! Lower a [`PipelineSpec`] to an executable DAG.
//!
//! For each stage:
//!  1. Look up the skill in `ix-registry` — fail if missing.
//!  2. Scan its `args` blob for `{"from": "stage_id[.key]"}` references
//!     and record implicit dependencies.
//!  3. Build a compute closure that resolves the references against
//!     upstream outputs at execution time, then calls the skill.

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use crate::dag::Dag;
use crate::executor::{PipelineError, PipelineNode};
use crate::spec::{PipelineSpec, SpecError};

/// Errors that can occur while lowering a spec to a DAG.
#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("spec error: {0}")]
    Spec(#[from] SpecError),

    #[error("stage '{stage}' references unknown skill '{skill}'")]
    UnknownSkill { stage: String, skill: String },

    #[error("stage '{stage}' `from:` reference points at unknown stage '{missing}'")]
    UnknownFromRef { stage: String, missing: String },

    #[error("DAG construction failed: {0}")]
    Dag(String),
}

/// Lower a spec to an executable DAG. The resulting DAG can be handed to
/// `executor::execute()`.
pub fn lower(spec: &PipelineSpec) -> Result<Dag<PipelineNode>, LowerError> {
    spec.validate_shape()?;

    // First pass: verify every skill exists and collect implicit deps from
    // `{"from": "..."}` references inside args.
    let stage_ids: BTreeSet<String> = spec.stages.keys().cloned().collect();
    let mut all_deps: HashMap<String, BTreeSet<String>> = HashMap::new();

    for (id, stage) in &spec.stages {
        if ix_registry::by_name(&stage.skill).is_none() {
            return Err(LowerError::UnknownSkill {
                stage: id.clone(),
                skill: stage.skill.clone(),
            });
        }

        let mut deps: BTreeSet<String> = stage.deps.iter().cloned().collect();
        collect_from_refs(&stage.args, &mut deps);
        for d in &deps {
            let base = d.split('.').next().unwrap_or(d).to_string();
            if !stage_ids.contains(&base) {
                return Err(LowerError::UnknownFromRef {
                    stage: id.clone(),
                    missing: base,
                });
            }
        }
        all_deps.insert(id.clone(), deps);
    }

    // Build the DAG.
    let mut dag: Dag<PipelineNode> = Dag::new();

    for (id, stage) in &spec.stages {
        let skill_name: String = stage.skill.clone();
        let args_template: Value = stage.args.clone();
        let stage_id_for_err = id.clone();
        let deps = &all_deps[id];

        // input_map: every dep is exposed as an input whose "source_node" is
        // the dep itself and key is "*" (full output). We also resolve
        // `{"from":"X"}` references at compute time.
        let mut input_map: HashMap<String, (String, String)> = HashMap::new();
        for d in deps {
            let base = d.split('.').next().unwrap_or(d).to_string();
            input_map.insert(base.clone(), (base.clone(), "*".into()));
        }

        let compute = Box::new(
            move |inputs: &HashMap<String, Value>| -> Result<Value, PipelineError> {
                let resolved = resolve_from_refs(&args_template, inputs)
                    .map_err(|e| PipelineError::ComputeError(format!("{stage_id_for_err}: {e}")))?;

                let desc = ix_registry::by_name(&skill_name).ok_or_else(|| {
                    PipelineError::ComputeError(format!(
                        "{stage_id_for_err}: skill '{skill_name}' vanished from registry"
                    ))
                })?;
                let args = [ix_types::Value::Json(resolved)];
                let out = (desc.fn_ptr)(&args)
                    .map_err(|e| PipelineError::ComputeError(format!("{stage_id_for_err}: {e}")))?;
                match out {
                    ix_types::Value::Json(j) => Ok(j),
                    other => serde_json::to_value(other).map_err(|e| {
                        PipelineError::ComputeError(format!("{stage_id_for_err}: encode {e}"))
                    }),
                }
            },
        );

        let node = PipelineNode {
            name: id.clone(),
            compute,
            input_map,
            cost: 1.0,
            cacheable: stage.cache.unwrap_or(true),
        };
        dag.add_node(id, node)
            .map_err(|e| LowerError::Dag(format!("{e:?}")))?;
    }

    // Second pass: wire edges.
    for (id, deps) in &all_deps {
        for d in deps {
            let base = d.split('.').next().unwrap_or(d).to_string();
            dag.add_edge(&base, id)
                .map_err(|e| LowerError::Dag(format!("{e:?}")))?;
        }
    }

    Ok(dag)
}

/// Walk an `args` JSON blob, collecting every string-id inside
/// `{"from": "stage[.key]"}` references.
fn collect_from_refs(value: &Value, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(target)) = map.get("from") {
                // `{"from": "stage[.key]"}` — a reference. Record the
                // unqualified stage id only; `resolve_from_refs` does the
                // sub-path lookup at runtime.
                if map.len() == 1 {
                    out.insert(target.clone());
                    return;
                }
            }
            for (_, v) in map {
                collect_from_refs(v, out);
            }
        }
        Value::Array(items) => {
            for v in items {
                collect_from_refs(v, out);
            }
        }
        _ => {}
    }
}

/// Clone `template` replacing every `{"from": "..."}` reference with the
/// matching upstream output. Unknown references bubble up as `Err`.
fn resolve_from_refs(template: &Value, inputs: &HashMap<String, Value>) -> Result<Value, String> {
    match template {
        Value::Object(map) => {
            if let Some(Value::String(target)) = map.get("from") {
                if map.len() == 1 {
                    let (stage, path): (&str, &[&str]) = match target.split_once('.') {
                        Some((s, rest)) => (s, &[rest][..]),
                        None => (target.as_str(), &[][..]),
                    };
                    let upstream = inputs
                        .get(stage)
                        .ok_or_else(|| format!("no output for stage '{stage}'"))?;
                    let mut cursor = upstream;
                    for seg in path {
                        // Allow dotted multi-hop: "a.b.c"
                        for part in seg.split('.') {
                            cursor = cursor
                                .get(part)
                                .ok_or_else(|| format!("field '{part}' missing in '{stage}'"))?;
                        }
                    }
                    return Ok(cursor.clone());
                }
            }
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k.clone(), resolve_from_refs(v, inputs)?);
            }
            Ok(Value::Object(out))
        }
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                out.push(resolve_from_refs(v, inputs)?);
            }
            Ok(Value::Array(out))
        }
        other => Ok(other.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn collect_from_refs_finds_references() {
        let v = json!({
            "action": "use result",
            "nested": { "from": "stage1" },
            "list": [ { "from": "stage2" }, 42 ]
        });
        let mut refs = BTreeSet::new();
        collect_from_refs(&v, &mut refs);
        assert!(refs.contains("stage1"));
        assert!(refs.contains("stage2"));
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn resolve_from_refs_substitutes() {
        let template = json!({
            "scalar": 1,
            "from_stage": { "from": "up" }
        });
        let mut inputs = HashMap::new();
        inputs.insert("up".to_string(), json!({"answer": 42}));
        let resolved = resolve_from_refs(&template, &inputs).unwrap();
        assert_eq!(resolved["from_stage"]["answer"], 42);
    }

    #[test]
    fn resolve_from_refs_walks_dotted_path() {
        let template = json!({ "data": { "from": "up.nested.value" } });
        let mut inputs = HashMap::new();
        inputs.insert(
            "up".to_string(),
            json!({ "nested": { "value": [1, 2, 3] } }),
        );
        let resolved = resolve_from_refs(&template, &inputs).unwrap();
        assert_eq!(resolved["data"], json!([1, 2, 3]));
    }
}
