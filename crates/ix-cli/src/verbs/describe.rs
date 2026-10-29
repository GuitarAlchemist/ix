//! `ix describe <noun> <name>` — introspect a skill / persona / policy.

use crate::output::{self, Format};
use serde_json::{json, Value};
use std::path::Path;

fn governance_dir() -> String {
    std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string())
}

pub fn skill(name: &str, format: Format) -> Result<(), String> {
    let desc = ix_registry::by_name(name)
        .ok_or_else(|| format!("skill not found: {name}\n\nTry `ix list skills`."))?;

    let inputs: Vec<Value> = desc
        .inputs
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "type": format!("{:?}", s.ty),
                "optional": s.optional,
                "doc": s.doc,
            })
        })
        .collect();
    let outputs: Vec<Value> = desc
        .outputs
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "type": format!("{:?}", s.ty),
                "optional": s.optional,
                "doc": s.doc,
            })
        })
        .collect();

    let payload = json!({
        "name": desc.name,
        "domain": desc.domain,
        "crate": desc.crate_name,
        "doc": desc.doc,
        "inputs": inputs,
        "outputs": outputs,
        "governance_tags": desc.governance_tags,
        "schema": (desc.json_schema)(),
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}

pub fn persona(name: &str, format: Format) -> Result<(), String> {
    let dir = governance_dir();
    let personas_dir = format!("{dir}/personas");
    let p = ix_governance::Persona::load_by_name(Path::new(&personas_dir), name)
        .map_err(|e| format!("loading persona '{name}': {e}"))?;

    let payload = json!({
        "name": p.name,
        "version": p.version,
        "description": p.description,
        "role": p.role,
        "domain": p.domain,
        "capabilities": p.capabilities,
        "constraints": p.constraints,
        "voice": {
            "tone": p.voice.tone,
            "verbosity": p.voice.verbosity,
            "style": p.voice.style,
        },
        "interaction_patterns": p.interaction_patterns.as_ref().map(|ip| json!({
            "with_humans": ip.with_humans,
            "with_agents": ip.with_agents,
        })),
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}

pub fn policy(name: &str, format: Format) -> Result<(), String> {
    let dir = governance_dir();
    // Policies live at {gov}/policies/{name}-policy.yaml (common convention)
    // Try both `<name>.yaml` and `<name>-policy.yaml`.
    let candidates = [
        format!("{dir}/policies/{name}-policy.yaml"),
        format!("{dir}/policies/{name}.yaml"),
    ];
    let path = candidates
        .iter()
        .find(|p| Path::new(p).is_file())
        .ok_or_else(|| format!("policy '{name}' not found (tried: {candidates:?})"))?;

    let p =
        ix_governance::Policy::load(Path::new(path)).map_err(|e| format!("loading {path}: {e}"))?;
    let payload = json!({
        "name": p.name,
        "version": p.version,
        "description": p.description,
        "path": path,
        "extra": p.extra,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}
