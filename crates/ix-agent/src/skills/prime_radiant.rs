//! Prime Radiant data-generation skills — scan the governance submodule
//! and emit the node+edge graph that the 3D visualization consumes.
//!
//! ## Refresh mechanism
//!
//! The `governance.graph` tool performs a full scan on every call (stateless,
//! safe to call repeatedly).  For efficient polling, callers should use
//! `governance.graph.rescan` instead:
//!
//! 1. Call `governance.graph.rescan` with `last_scan_epoch = 0` to get an
//!    initial graph.  The response includes `scanned_at` (Unix seconds).
//! 2. Every ~60 s, call again with `last_scan_epoch` set to the previous
//!    `scanned_at` value.
//! 3. If no artifact has changed since `last_scan_epoch`, the response
//!    contains `{"changed": false, "scanned_at": <now>}` — cheap, no graph
//!    payload.
//! 4. If any file mtime is newer, `{"changed": true, "scanned_at": <now>,
//!    "graph": {...}}` is returned and the caller re-renders.
//!
//! This avoids a persistent background thread or OS file-watcher while still
//! giving callers live updates within one poll interval.

use ix_skill_macros::ix_skill;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn governance_root() -> std::path::PathBuf {
    if let Ok(root) = std::env::var("IX_ROOT") {
        return Path::new(&root).join("governance/demerzel");
    }
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return Path::new(&manifest).join("../../governance/demerzel");
    }
    Path::new("governance/demerzel").to_path_buf()
}

fn scan_dir(dir: &Path, node_type: &str) -> Vec<Value> {
    let mut nodes = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_string();
                if name.is_empty() || name.starts_with('.') {
                    continue;
                }
                nodes.push(json!({
                    "id": format!("{node_type}:{name}"),
                    "type": node_type,
                    "name": name,
                    "path": path.display().to_string(),
                }));
            }
        }
    }
    nodes
}

fn governance_graph_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": "Override governance directory (default: auto-detect)"
            }
        }
    })
}

/// Scan the Demerzel governance submodule and emit a graph of all
/// artifacts — constitutions, policies, personas, schemas, tests — with
/// structural edges between type layers. Output is directly consumable
/// by Prime Radiant's Three.js force-directed layout.
#[ix_skill(
    domain = "governance",
    name = "governance.graph",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::prime_radiant::governance_graph_schema"
)]
pub fn governance_graph(params: Value) -> Result<Value, String> {
    let root = params
        .get("root")
        .and_then(|v| v.as_str())
        .map(|s| Path::new(s).to_path_buf())
        .unwrap_or_else(governance_root);

    if !root.is_dir() {
        return Err(format!(
            "governance directory not found: {}",
            root.display()
        ));
    }

    // Scan each artifact directory.
    let constitutions = scan_dir(&root.join("constitutions"), "constitution");
    let policies = scan_dir(&root.join("policies"), "policy");
    let personas = scan_dir(&root.join("personas"), "persona");
    let schemas = scan_dir(&root.join("schemas"), "schema");

    // Tests are in tests/behavioral/ (many subdirs + files).
    let mut tests = Vec::new();
    let tests_dir = root.join("tests/behavioral");
    if tests_dir.is_dir() {
        for entry in walkdir_flat(&tests_dir) {
            let name = entry
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            if !name.is_empty() {
                tests.push(json!({
                    "id": format!("test:{name}"),
                    "type": "test",
                    "name": name,
                    "path": entry.display().to_string(),
                }));
            }
        }
    }

    // Collect all nodes.
    let mut all_nodes = Vec::new();
    all_nodes.extend(constitutions.iter().cloned());
    all_nodes.extend(policies.iter().cloned());
    all_nodes.extend(personas.iter().cloned());
    all_nodes.extend(schemas.iter().cloned());
    all_nodes.extend(tests.iter().cloned());

    // Build structural edges (type-layer hierarchy).
    let mut edges: Vec<Value> = Vec::new();

    // Constitution → Policy: every policy descends from
    // default.constitution.md (the operational constitution).
    let default_const_id = "constitution:default.constitution";
    for policy in &policies {
        edges.push(json!({
            "source": default_const_id,
            "target": policy["id"],
            "relation": "governs",
        }));
    }

    // Policy → Persona: every persona is governed by the alignment policy
    // (the most general policy).
    let alignment_id = "policy:alignment-policy";
    for persona in &personas {
        edges.push(json!({
            "source": alignment_id,
            "target": persona["id"],
            "relation": "constrains",
        }));
    }

    // Persona → Test: match test filenames to personas by prefix.
    // e.g. "skeptical-auditor-cases" → linked to persona "skeptical-auditor".
    let persona_names: Vec<String> = personas
        .iter()
        .filter_map(|p| p["name"].as_str().map(String::from))
        .collect();
    for test in &tests {
        if let Some(test_name) = test["name"].as_str() {
            for pname in &persona_names {
                if test_name.starts_with(pname) {
                    edges.push(json!({
                        "source": format!("persona:{pname}"),
                        "target": test["id"],
                        "relation": "tested_by",
                    }));
                    break;
                }
            }
        }
    }

    // Asimov constitution → default constitution (root hierarchy).
    let asimov_id = "constitution:asimov.constitution";
    if constitutions.iter().any(|c| c["id"] == asimov_id) {
        edges.push(json!({
            "source": asimov_id,
            "target": default_const_id,
            "relation": "root_authority",
        }));
    }

    // Node-type summary.
    let mut type_counts = BTreeMap::new();
    for node in &all_nodes {
        let t = node["type"].as_str().unwrap_or("unknown");
        *type_counts.entry(t.to_string()).or_insert(0u64) += 1;
    }

    Ok(json!({
        "total_nodes": all_nodes.len(),
        "total_edges": edges.len(),
        "type_counts": type_counts,
        "nodes": all_nodes,
        "edges": edges,
        "root": root.display().to_string(),
        // Freshness metadata — used by governance.graph.rescan to detect
        // whether a re-render is needed without re-fetching the full graph.
        "scanned_at": now_epoch(),
        "newest_artifact_epoch": newest_mtime(&root),
    }))
}

/// Return current Unix epoch seconds (falls back to 0 on error).
fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Walk the governance root (two levels deep, matching `walkdir_flat`) and
/// return the newest mtime as Unix seconds.  Returns 0 if nothing is readable.
fn newest_mtime(root: &Path) -> u64 {
    let mut max: u64 = 0;

    // Helper to update max from a single path's metadata.
    let mut check = |p: &Path| {
        if let Ok(meta) = std::fs::metadata(p) {
            if let Ok(modified) = meta.modified() {
                if let Ok(dur) = modified.duration_since(UNIX_EPOCH) {
                    let secs = dur.as_secs();
                    if secs > max {
                        max = secs;
                    }
                }
            }
        }
    };

    if let Ok(top) = std::fs::read_dir(root) {
        for entry in top.flatten() {
            let p = entry.path();
            if p.is_file() {
                check(&p);
            } else if p.is_dir() {
                if let Ok(sub) = std::fs::read_dir(&p) {
                    for s in sub.flatten() {
                        let sp = s.path();
                        if sp.is_file() {
                            check(&sp);
                        } else if sp.is_dir() {
                            // One extra level for tests/behavioral/{subdir}/
                            if let Ok(sub2) = std::fs::read_dir(&sp) {
                                for s2 in sub2.flatten() {
                                    if s2.path().is_file() {
                                        check(&s2.path());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    max
}

fn governance_rescan_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "root": {
                "type": "string",
                "description": "Override governance directory (default: auto-detect)"
            },
            "last_scan_epoch": {
                "type": "integer",
                "description": "Unix timestamp (seconds) of the previous scan. Pass 0 for the initial call."
            }
        },
        "required": ["last_scan_epoch"]
    })
}

/// Poll-based governance graph refresh — returns the full graph only when
/// Demerzel artifacts have changed since `last_scan_epoch`.
///
/// Callers should invoke this every ~60 s, passing the `scanned_at` value
/// from the previous response as `last_scan_epoch`.  When `changed` is false
/// the payload is tiny (~50 bytes); when true the full graph is included.
///
/// ## Typical polling loop (pseudo-code)
/// ```text
/// let last = 0;
/// loop {
///     let r = governance.graph.rescan({ last_scan_epoch: last });
///     last = r.scanned_at;
///     if r.changed { render(r.graph); }
///     sleep(60s);
/// }
/// ```
#[ix_skill(
    domain = "governance",
    name = "governance.graph.rescan",
    governance = "safety,deterministic",
    schema_fn = "crate::skills::prime_radiant::governance_rescan_schema"
)]
pub fn governance_graph_rescan(params: Value) -> Result<Value, String> {
    let root = params
        .get("root")
        .and_then(|v| v.as_str())
        .map(|s| Path::new(s).to_path_buf())
        .unwrap_or_else(governance_root);

    if !root.is_dir() {
        return Err(format!(
            "governance directory not found: {}",
            root.display()
        ));
    }

    let last_scan_epoch = params
        .get("last_scan_epoch")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let scanned_at = now_epoch();
    let newest = newest_mtime(&root);

    // No changes since last scan — return lightweight response.
    if newest <= last_scan_epoch && last_scan_epoch > 0 {
        return Ok(json!({
            "changed": false,
            "scanned_at": scanned_at,
            "newest_artifact_epoch": newest,
        }));
    }

    // Changes detected (or initial call) — run full scan and return graph.
    let graph = governance_graph(json!({ "root": root.display().to_string() }))?;

    Ok(json!({
        "changed": true,
        "scanned_at": scanned_at,
        "newest_artifact_epoch": newest,
        "graph": graph,
    }))
}

/// Non-recursive file listing — only files, no subdirectories.
fn walkdir_flat(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            } else if path.is_dir() {
                // One level of subdirectory for tests/behavioral/{subdir}/*.md
                if let Ok(sub) = std::fs::read_dir(&path) {
                    for s in sub.flatten() {
                        if s.path().is_file() {
                            files.push(s.path());
                        }
                    }
                }
            }
        }
    }
    files
}
