//! `ix_petri_analyze` — the agent-facing surface for `ix-petri`.
//!
//! Without this an agent cannot reach the analyses at all: `ix-petri` is a
//! library, and the gap matrix (`docs/research/tars-v1-advanced-math-ix-gap-matrix.md`,
//! row A3) records exactly this failure mode — an implemented algorithm with no
//! MCP tool is "unreachable from an agent loop".
//!
//! The tool takes a net either inline (places / transitions / arcs) or as a
//! PNML Place/Transition document, and returns the deadlock / boundedness /
//! liveness report. It is a pure computation: no filesystem, no network, no
//! state. Reading a `.pnml` from disk is deliberately *not* offered — the
//! caller passes the document text, so the tool has no path handling and no
//! ambient authority.

use ix_petri::analysis::{analyze, Analysis, Limits};
use ix_petri::{PetriNet, PetriNetBuilder};
use ix_skill_macros::ix_skill;
use serde_json::{json, Value};

fn petri_analyze_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "pnml": {
                "type": "string",
                "description": "A PNML document (ISO/IEC 15909-2, Place/Transition subclass). Mutually exclusive with `places`/`transitions`/`arcs`."
            },
            "name": { "type": "string", "description": "Optional net name, for inline nets" },
            "places": {
                "type": "array",
                "description": "Inline places. Either \"id\" for an empty place, or {id, tokens, name}.",
                "items": {
                    "oneOf": [
                        { "type": "string" },
                        {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "tokens": { "type": "integer", "minimum": 0, "default": 0 },
                                "name": { "type": "string" }
                            },
                            "required": ["id"]
                        }
                    ]
                }
            },
            "transitions": {
                "type": "array",
                "description": "Inline transitions. Either \"id\", or {id, name}.",
                "items": {
                    "oneOf": [
                        { "type": "string" },
                        {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "name": { "type": "string" }
                            },
                            "required": ["id"]
                        }
                    ]
                }
            },
            "arcs": {
                "type": "array",
                "description": "Inline arcs. Each crosses a place and a transition in either direction; `weight` defaults to 1.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string" },
                        "target": { "type": "string" },
                        "weight": { "type": "integer", "minimum": 1, "default": 1 }
                    },
                    "required": ["source", "target"]
                }
            },
            "max_states": {
                "type": "integer",
                "minimum": 1,
                "default": 50000,
                "description": "Enumeration budget. Hitting it makes undecided properties `unknown` rather than guessed."
            }
        }
    })
}

fn petri_analyze_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "net": { "type": ["string", "null"], "description": "Net name, if it has one" },
            "places": { "type": "integer" },
            "transitions": { "type": "integer" },
            "states": { "type": "integer", "description": "Distinct reachable markings enumerated" },
            "transitions_fired": { "type": "integer", "description": "Edges in the reachability graph" },
            "truncated": { "type": "boolean", "description": "True when max_states stopped the search" },
            "deadlock_free": {
                "type": "object",
                "description": "verdict holds|fails|unknown; on `fails`, `detail` lists dead markings with the shortest firing sequence reaching each"
            },
            "deadlock_count": { "type": "integer" },
            "bounded": { "type": "object", "description": "verdict holds|fails|unknown; on `holds`, `detail` gives k and the per-place maxima" },
            "unbounded_witness": { "type": ["object", "null"], "description": "A marking pair m < m' proving unboundedness, and the repeatable sequence between them" },
            "quasi_live": { "type": "object", "description": "On `fails`, the transitions enabled in no reachable marking" },
            "live": { "type": "object", "description": "L4-liveness. On `fails`, transitions absent from some terminal SCC" },
            "reversible": { "type": "object", "description": "Whether the initial marking is reachable from everywhere" },
            "witness_labels": {
                "type": "object",
                "description": "Deadlock witnesses rendered with transition names instead of ids, keyed by deadlock index"
            }
        }
    })
}

/// Analyse a Place/Transition Petri net for deadlock, boundedness, dead
/// transitions, liveness and reversibility.
///
/// Supply the net inline (`places` / `transitions` / `arcs`) or as a PNML
/// document (`pnml`). Every verdict is `holds`, `fails` with a witness, or
/// `unknown` — never a guess. Firing order is deterministic, so the same net
/// yields the same report on every call.
#[ix_skill(
    domain = "petri",
    name = "petri.analyze",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::petri::petri_analyze_schema",
    output_schema_fn = "crate::skills::petri::petri_analyze_output_schema"
)]
pub fn petri_analyze(params: Value) -> Result<Value, String> {
    let net = build_net(&params)?;
    let limits = match params.get("max_states").and_then(Value::as_u64) {
        Some(0) | None => Limits::default(),
        Some(n) => Limits::with_max_states(n as usize),
    };
    let report = analyze(&net, limits);
    render(&net, &report)
}

/// Either the PNML document or the inline node lists, never both.
fn build_net(params: &Value) -> Result<PetriNet, String> {
    let has_inline = ["places", "transitions", "arcs"]
        .iter()
        .any(|k| params.get(*k).is_some());

    match params.get("pnml").and_then(Value::as_str) {
        Some(source) => {
            if has_inline {
                return Err(
                    "give either `pnml` or the inline `places`/`transitions`/`arcs`, not both"
                        .to_string(),
                );
            }
            let nets = ix_petri::read_pnml(source).map_err(|e| e.to_string())?;
            let mut nets = nets;
            if nets.len() > 1 {
                return Err(format!(
                    "the document holds {} nets; this tool analyses one at a time",
                    nets.len()
                ));
            }
            Ok(nets.remove(0).net)
        }
        None => {
            if !has_inline {
                return Err(
                    "supply a net: either `pnml` or `places`+`transitions`+`arcs`".to_string(),
                );
            }
            build_inline(params)
        }
    }
}

fn build_inline(params: &Value) -> Result<PetriNet, String> {
    let mut b = PetriNetBuilder::default();
    if let Some(name) = params.get("name").and_then(Value::as_str) {
        b = b.name(name);
    }

    for (i, place) in array(params, "places")?.iter().enumerate() {
        let (id, name) = node_id(place, "places", i)?;
        let tokens = place.get("tokens").and_then(Value::as_u64).unwrap_or(0);
        b = match name {
            Some(n) => b.named_place(id, n, tokens),
            None => b.place(id, tokens),
        };
    }
    for (i, transition) in array(params, "transitions")?.iter().enumerate() {
        let (id, name) = node_id(transition, "transitions", i)?;
        b = match name {
            Some(n) => b.named_transition(id, n),
            None => b.transition(id),
        };
    }
    for (i, arc) in array(params, "arcs")?.iter().enumerate() {
        let source = arc
            .get("source")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("arcs[{i}] has no `source`"))?;
        let target = arc
            .get("target")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("arcs[{i}] has no `target`"))?;
        let weight = arc.get("weight").and_then(Value::as_u64).unwrap_or(1);
        b = b.weighted_arc(source, target, weight);
    }

    b.build().map_err(|e| e.to_string())
}

fn array<'a>(params: &'a Value, key: &str) -> Result<&'a [Value], String> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(items)) => Ok(items),
        Some(_) => Err(format!("`{key}` must be an array")),
    }
}

/// A node is either the bare id string or an object with `id` and an optional
/// `name`.
fn node_id(value: &Value, key: &str, i: usize) -> Result<(String, Option<String>), String> {
    match value {
        Value::String(id) => Ok((id.clone(), None)),
        Value::Object(_) => {
            let id = value
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{key}[{i}] has no `id`"))?;
            let name = value
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string);
            Ok((id.to_string(), name))
        }
        _ => Err(format!("{key}[{i}] must be a string id or an object")),
    }
}

fn render(net: &PetriNet, report: &Analysis) -> Result<Value, String> {
    let mut out = serde_json::to_value(report).map_err(|e| e.to_string())?;
    let object = out
        .as_object_mut()
        .ok_or_else(|| "analysis did not serialize to an object".to_string())?;
    object.insert("places".into(), json!(net.places().len()));
    object.insert("transitions".into(), json!(net.transitions().len()));

    // Witnesses carry transition ids because those are replayable; agents also
    // want the human names, so both are available without either being lossy.
    if let ix_petri::Verdict::Fails(deadlocks) = &report.deadlock_free {
        let labels: Vec<Value> = deadlocks
            .iter()
            .map(|d| json!(net.label_sequence(&d.witness)))
            .collect();
        object.insert("witness_labels".into(), Value::Array(labels));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyses_an_inline_net_and_reports_the_deadlock_witness() {
        let out = petri_analyze(json!({
            "name": "leaked-lock",
            "places": [{ "id": "lock", "tokens": 1 }, "working"],
            "transitions": [{ "id": "acquire", "name": "take the lock" }],
            "arcs": [
                { "source": "lock", "target": "acquire" },
                { "source": "acquire", "target": "working" }
            ]
        }))
        .expect("a well-formed net analyses");

        assert_eq!(out["net"], json!("leaked-lock"));
        assert_eq!(out["places"], json!(2));
        assert_eq!(out["deadlock_free"]["verdict"], json!("fails"));
        assert_eq!(out["deadlock_count"], json!(1));
        assert_eq!(
            out["deadlock_free"]["detail"][0]["marking"],
            json!("working=1")
        );
        assert_eq!(
            out["deadlock_free"]["detail"][0]["witness"],
            json!(["acquire"])
        );
        assert_eq!(out["witness_labels"][0], json!(["take the lock"]));
        assert_eq!(out["bounded"]["verdict"], json!("holds"));
    }

    #[test]
    fn analyses_a_pnml_document() {
        let pnml = r#"<pnml xmlns="http://www.pnml.org/version-2009/grammar/pnml">
          <net id="n" type="http://www.pnml.org/version-2009/grammar/ptnet">
            <page id="p">
              <place id="a"><initialMarking><text>1</text></initialMarking></place>
              <place id="b"/>
              <transition id="t"/>
              <arc id="x" source="a" target="t"/>
              <arc id="y" source="t" target="b"/>
              <transition id="u"/>
              <arc id="z" source="b" target="u"/>
              <arc id="w" source="u" target="a"/>
            </page>
          </net></pnml>"#;
        let out = petri_analyze(json!({ "pnml": pnml })).expect("valid PNML");
        assert_eq!(out["deadlock_free"]["verdict"], json!("holds"));
        assert_eq!(out["states"], json!(2));
        assert_eq!(out["reversible"]["verdict"], json!("holds"));
    }

    #[test]
    fn a_tight_budget_yields_unknown_rather_than_a_guess() {
        let out = petri_analyze(json!({
            "places": ["queue"],
            "transitions": ["grow"],
            "arcs": [{ "source": "grow", "target": "queue" }],
            "max_states": 20
        }))
        .unwrap();
        assert_eq!(out["truncated"], json!(true));
        assert_eq!(out["bounded"]["verdict"], json!("fails"));
        assert_eq!(out["live"]["verdict"], json!("unknown"));
        assert!(out["unbounded_witness"]["pumping_sequence"]
            .as_array()
            .is_some_and(|s| s == &[json!("grow")]));
    }

    #[test]
    fn malformed_requests_are_refused_with_a_reason() {
        assert!(petri_analyze(json!({}))
            .unwrap_err()
            .contains("supply a net"));
        assert!(petri_analyze(json!({ "pnml": "<x/>", "places": [] }))
            .unwrap_err()
            .contains("not both"));
        assert!(petri_analyze(json!({
            "places": ["p", "q"],
            "arcs": [{ "source": "p", "target": "q" }]
        }))
        .unwrap_err()
        .contains("must cross place and transition"));
    }
}
