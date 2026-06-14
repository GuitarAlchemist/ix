//! Lower a [`PipelineSpec`] to an executable DAG.
//!
//! For each stage:
//!  1. Look up the skill in `ix-registry` — fail if missing.
//!  2. Scan its `args` blob for `{"from": "stage_id[.key]"}` references
//!     and record implicit dependencies.
//!  3. Build a compute closure that resolves the references against
//!     upstream outputs at execution time, then calls the skill.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::Value;

use crate::dag::Dag;
use crate::executor::{PipelineError, PipelineNode};
use crate::gate::{allow_all, SharedGate};
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

/// Lower a spec to an executable DAG (no execution-time gate — every stage is
/// allowed). The resulting DAG can be handed to `executor::execute()`.
pub fn lower(spec: &PipelineSpec) -> Result<Dag<PipelineNode>, LowerError> {
    lower_with_gate(spec, allow_all())
}

/// Like [`lower`], but each stage's compute closure consults `gate` on its
/// **resolved** args (post-`{"from"}`-resolution) immediately before the skill
/// runs. This is the seam where governance that must see real runtime values —
/// not the spec template — is enforced; a `{"from": "upstream"}` ref that
/// supplies a destructive operation is vetted here, after it resolves. See
/// [`crate::gate`].
pub fn lower_with_gate(
    spec: &PipelineSpec,
    gate: SharedGate,
) -> Result<Dag<PipelineNode>, LowerError> {
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

        // One Arc handle per node — the closure is `move` and the executor runs
        // independent levels in parallel, so each closure owns its own clone.
        let gate = gate.clone();

        let compute = Box::new(
            move |inputs: &HashMap<String, Value>| -> Result<Value, PipelineError> {
                let resolved = resolve_from_refs(&args_template, inputs)
                    .map_err(|e| PipelineError::ComputeError(format!("{stage_id_for_err}: {e}")))?;

                // Execution-time governance: the gate sees the RESOLVED args, so
                // a `{"from"}` ref that supplied a destructive op is vetted here
                // — the value the skill is about to run with, not the template.
                gate.check(&stage_id_for_err, &skill_name, &resolved)
                    .map_err(|e| {
                        PipelineError::ComputeError(format!("{stage_id_for_err}: governance: {e}"))
                    })?;

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
pub(crate) fn resolve_from_refs(
    template: &Value,
    inputs: &HashMap<String, Value>,
) -> Result<Value, String> {
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

/// Bind run-time parameters into a spec: replace every `{"param": "NAME"}`
/// object-ref in a stage's args with a concrete value, *before* lowering.
///
/// Values are sourced from the spec's own `params` bag (defaults) overridden by
/// `overrides` (run-time `--param` values win). Unlike `{"from": "stage"}` —
/// which resolves against an upstream stage's OUTPUT during execution — a
/// `{"param"}` is bound up front: it carries data the NL request didn't provide
/// inline (e.g. "reduce *this dataset*" → `data: {param: "dataset"}`). A
/// referenced param with no value (no spec default and no override, or an
/// explicit `null` default meaning "must be supplied") is a hard error *before*
/// anything executes — never silently dropped.
// @ai:invariant bind_params returns an unbound {param:"X"} (missing/null) as Err, never as a substituted spec — so a caller that binds before executing cannot pass an unbound placeholder to a skill [T:test conf:0.9 src:lower::tests::bind_params_missing_param_errors]
pub fn bind_params(
    spec: &PipelineSpec,
    overrides: &BTreeMap<String, Value>,
) -> Result<PipelineSpec, String> {
    // spec defaults first, then run-time overrides win.
    let mut merged = spec.params.clone();
    for (k, v) in overrides {
        merged.insert(k.clone(), v.clone());
    }

    let mut out = spec.clone();
    for (id, stage) in out.stages.iter_mut() {
        stage.args =
            substitute_params(&stage.args, &merged).map_err(|e| format!("stage '{id}': {e}"))?;
    }
    Ok(out)
}

/// Recursively replace `{"param": "NAME"}` (a single-key object whose value is a
/// string) with `params[NAME]`. Everything else — including `{"from": ...}` — is
/// passed through untouched. A `null` param value counts as unsupplied. The
/// substituted value is NOT re-scanned for refs: a param is opaque DATA, so a
/// supplied value that is itself a `{from}`/`{param}` reference is rejected
/// rather than allowed to inject a DAG edge (data must not become control flow).
fn substitute_params(value: &Value, params: &BTreeMap<String, Value>) -> Result<Value, String> {
    match value {
        Value::Object(map) => {
            if map.len() == 1 {
                if let Some(Value::String(name)) = map.get("param") {
                    return match params.get(name) {
                        Some(Value::Null) | None => Err(format!(
                            "missing required param '{name}' (supply it with --param {name}=<json|@file> or give it a default in the spec's `params`)"
                        )),
                        Some(v) if contains_ref(v) => Err(format!(
                            "param '{name}' value must be literal data, not a {{\"from\"}}/{{\"param\"}} reference (refs are control flow, params are data)"
                        )),
                        Some(v) => Ok(v.clone()),
                    };
                }
            }
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                out.insert(k.clone(), substitute_params(v, params)?);
            }
            Ok(Value::Object(out))
        }
        Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for v in items {
                out.push(substitute_params(v, params)?);
            }
            Ok(Value::Array(out))
        }
        other => Ok(other.clone()),
    }
}

/// True if `value` contains a `{"from": "..."}` or `{"param": "..."}` reference
/// anywhere (single-key object with a string value). Used to reject a `--param`
/// value that would smuggle a control-flow ref into the DAG via substitution.
fn contains_ref(value: &Value) -> bool {
    match value {
        Value::Object(map) => {
            if map.len() == 1
                && (matches!(map.get("from"), Some(Value::String(_)))
                    || matches!(map.get("param"), Some(Value::String(_))))
            {
                return true;
            }
            map.values().any(contains_ref)
        }
        Value::Array(items) => items.iter().any(contains_ref),
        _ => false,
    }
}

/// Collect every param NAME referenced by `{"param": "NAME"}` anywhere in a
/// spec's stage args. Used to report what a template still needs.
pub fn collect_param_names(spec: &PipelineSpec) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for stage in spec.stages.values() {
        collect_param_refs(&stage.args, &mut out);
    }
    out
}

fn collect_param_refs(value: &Value, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            if map.len() == 1 {
                if let Some(Value::String(name)) = map.get("param") {
                    out.insert(name.clone());
                    return;
                }
            }
            for v in map.values() {
                collect_param_refs(v, out);
            }
        }
        Value::Array(items) => {
            for v in items {
                collect_param_refs(v, out);
            }
        }
        _ => {}
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

    // ---- bind_params (run-time data binding) ----------------------------

    fn spec_with(params: Value, args: Value) -> PipelineSpec {
        let mut stages = std::collections::BTreeMap::new();
        stages.insert(
            "s".to_string(),
            crate::spec::StageSpec {
                skill: "pca".to_string(),
                args,
                deps: vec![],
                cache: None,
            },
        );
        let params: BTreeMap<String, Value> =
            serde_json::from_value(params).expect("params object");
        PipelineSpec {
            version: "1".into(),
            params,
            stages,
            x_editor: Value::Null,
        }
    }

    fn ov(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn bind_params_substitutes_whole_value() {
        let spec = spec_with(
            json!({ "dataset": null }),
            json!({ "data": { "param": "dataset" }, "n_components": 2 }),
        );
        let bound =
            bind_params(&spec, &ov(&[("dataset", json!([[1.0, 2.0], [3.0, 4.0]]))])).unwrap();
        assert_eq!(
            bound.stages["s"].args["data"],
            json!([[1.0, 2.0], [3.0, 4.0]])
        );
        assert_eq!(bound.stages["s"].args["n_components"], json!(2));
    }

    #[test]
    fn bind_params_uses_spec_default_and_override_wins() {
        let spec = spec_with(
            json!({ "n": 2 }),
            json!({ "n_components": { "param": "n" } }),
        );
        // default
        let bound = bind_params(&spec, &ov(&[])).unwrap();
        assert_eq!(bound.stages["s"].args["n_components"], json!(2));
        // override beats default
        let bound = bind_params(&spec, &ov(&[("n", json!(5))])).unwrap();
        assert_eq!(bound.stages["s"].args["n_components"], json!(5));
    }

    #[test]
    fn bind_params_missing_param_errors() {
        let spec = spec_with(json!({}), json!({ "data": { "param": "x" } }));
        let err = bind_params(&spec, &ov(&[])).unwrap_err();
        assert!(err.contains("missing required param 'x'"), "got: {err}");
    }

    #[test]
    fn bind_params_null_default_is_unsupplied() {
        let spec = spec_with(json!({ "x": null }), json!({ "data": { "param": "x" } }));
        let err = bind_params(&spec, &ov(&[])).unwrap_err();
        assert!(err.contains("missing required param 'x'"), "got: {err}");
    }

    #[test]
    fn bind_params_leaves_from_refs_untouched() {
        let spec = spec_with(json!({}), json!({ "data": { "from": "up" } }));
        let bound = bind_params(&spec, &ov(&[])).unwrap();
        assert_eq!(bound.stages["s"].args["data"], json!({ "from": "up" }));
    }

    #[test]
    fn bind_params_rejects_ref_shaped_param_value() {
        // A --param value that is itself a {from}/{param} ref must be rejected:
        // external run-time input must not inject DAG control flow (data != control).
        let spec = spec_with(json!({}), json!({ "data": { "param": "x" } }));
        let err = bind_params(&spec, &ov(&[("x", json!({ "from": "load" }))])).unwrap_err();
        assert!(err.contains("literal data"), "top-level ref, got: {err}");
        let err =
            bind_params(&spec, &ov(&[("x", json!({ "a": { "from": "load" } }))])).unwrap_err();
        assert!(err.contains("literal data"), "nested ref, got: {err}");
        // A plain data matrix is still accepted.
        let ok = bind_params(&spec, &ov(&[("x", json!([[1.0, 2.0]]))])).unwrap();
        assert_eq!(ok.stages["s"].args["data"], json!([[1.0, 2.0]]));
    }

    #[test]
    fn collect_param_names_finds_refs() {
        let spec = spec_with(
            json!({}),
            json!({ "data": { "param": "dataset" }, "k": { "param": "k" }, "lit": 3 }),
        );
        let names = collect_param_names(&spec);
        assert!(names.contains("dataset") && names.contains("k"));
        assert_eq!(names.len(), 2);
    }
}
