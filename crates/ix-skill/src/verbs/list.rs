//! `ix list <noun>` — discover skills / personas / policies.

use crate::output::{self, Format};
use serde_json::{json, Value};

pub fn skills(
    domain_filter: Option<&str>,
    query: Option<&str>,
    format: Format,
) -> Result<(), String> {
    let mut entries: Vec<&'static ix_registry::SkillDescriptor> = match domain_filter {
        Some(d) => ix_registry::by_domain(d),
        None => ix_registry::all().collect(),
    };
    if let Some(q) = query {
        let q = q.to_lowercase();
        entries.retain(|s| s.name.to_lowercase().contains(&q) || s.doc.to_lowercase().contains(&q));
    }
    entries.sort_by_key(|s| s.name);

    let rows: Vec<Value> = entries
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "domain": s.domain,
                "crate": s.crate_name,
                "inputs": s.inputs.len(),
                "doc": s.doc,
            })
        })
        .collect();
    let payload = json!({ "count": rows.len(), "skills": rows });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}

pub fn personas(format: Format) -> Result<(), String> {
    // Resolve governance dir: IX_GOVERNANCE_DIR env var, else
    // ./governance/demerzel (common submodule location).
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let personas_dir = format!("{gov_dir}/personas");

    let names = ix_governance::list_personas(std::path::Path::new(&personas_dir))
        .map_err(|e| format!("listing personas in {personas_dir}: {e}"))?;

    let payload = json!({
        "count": names.len(),
        "personas": names,
        "source": personas_dir,
    });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}

pub fn domains(format: Format) -> Result<(), String> {
    use std::collections::BTreeMap;
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for s in ix_registry::all() {
        *counts.entry(s.domain).or_insert(0) += 1;
    }
    let rows: Vec<Value> = counts
        .iter()
        .map(|(d, c)| json!({ "domain": d, "skill_count": c }))
        .collect();
    let payload = json!({ "count": rows.len(), "domains": rows });
    output::emit(&payload, format).map_err(|e| format!("writing output: {e}"))?;
    Ok(())
}
