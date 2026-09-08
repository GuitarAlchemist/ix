//! Registry-snapshot check — the skill/tool inventory, without a magic number.
//!
//! IX exposes the same capabilities through two surfaces: the link-time
//! capability registry (`ix_registry::IX_SKILLS`, populated by `#[ix_skill]`)
//! and the MCP tool registry (`ix_agent::tools::ToolRegistry`). Both used to be
//! guarded by hand-typed counts — `assert_eq!(EXPECTED.len(), 94)` — which is
//! exactly the oracle that drifted and turned `main` red when `mesh_correlate`
//! landed (ix#185).
//!
//! A snapshot replaces the count. Adding a tool now means regenerating
//! `state/registry/skills.snapshot.json` with `ix doctor --write`, which
//! produces a reviewable diff of *names* rather than a number nobody can
//! sanity-check. Removing one shows up the same way.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Path of the snapshot, relative to the repo root.
pub const SNAPSHOT_PATH: &str = "state/registry/skills.snapshot.json";

/// Schema version of the snapshot file. Bump on a breaking layout change.
pub const SCHEMA_VERSION: u32 = 1;

/// One named surface: a sorted list of capability names plus its length.
///
/// The count is stored redundantly so a human skimming the file sees the
/// number, but `names` is the oracle — the count is never compared on its own.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Surface {
    /// Number of names. Derived from `names`; informational only.
    pub count: usize,
    /// Sorted, de-duplicated capability names.
    pub names: Vec<String>,
}

impl Surface {
    /// Build a surface from any name iterator, sorting and de-duplicating.
    pub fn from_names<I: IntoIterator<Item = String>>(names: I) -> Self {
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        names.dedup();
        Self {
            count: names.len(),
            names,
        }
    }
}

/// The checked-in expectation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Snapshot {
    /// Schema version — see [`SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Free-text header explaining the file to whoever opens it next.
    #[serde(default)]
    pub note: String,
    /// Link-time capability-registry skill names.
    pub skills: Surface,
    /// MCP tool names exposed by `ix-agent`.
    pub mcp_tools: Surface,
    /// Tool names that appear only under a non-default cargo feature. These are
    /// tolerated whether present or absent, so both the default build and a
    /// `--features maintain-gate` build stay green against one snapshot.
    #[serde(default)]
    pub feature_gated_tools: Vec<String>,
}

impl Snapshot {
    /// Load the snapshot from `root`.
    pub fn load(root: &Path) -> Result<Self, String> {
        let path = root.join(SNAPSHOT_PATH);
        let raw = std::fs::read_to_string(&path).map_err(|e| {
            format!(
                "reading {}: {e} — regenerate it with `cargo run -p ix-skill --bin ix -- doctor --write`",
                path.display()
            )
        })?;
        let snap: Self = serde_json::from_str(&raw)
            .map_err(|e| format!("parsing {}: {e}", path.display()))?;
        if snap.schema_version != SCHEMA_VERSION {
            return Err(format!(
                "{}: schema_version {} but this build expects {SCHEMA_VERSION}",
                path.display(),
                snap.schema_version
            ));
        }
        Ok(snap)
    }

    /// Write the snapshot to `root`, creating parent directories.
    pub fn write(&self, root: &Path) -> Result<(), String> {
        let path = root.join(SNAPSHOT_PATH);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("creating {}: {e}", parent.display()))?;
        }
        let mut json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("serializing snapshot: {e}"))?;
        json.push('\n');
        std::fs::write(&path, json).map_err(|e| format!("writing {}: {e}", path.display()))
    }
}

/// The two live surfaces read out of the linked binary.
#[derive(Debug, Clone, Default)]
pub struct Live {
    /// Link-time capability-registry skill names.
    pub skills: Surface,
    /// MCP tool names.
    pub mcp_tools: Surface,
}

/// Read both surfaces from the current build.
pub fn live() -> Live {
    let skills = Surface::from_names(ix_registry::all().map(|s| s.name.to_string()));
    let tools = ix_agent::tools::ToolRegistry::new().list();
    let tool_names = tools["tools"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t["name"].as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Live {
        skills,
        mcp_tools: Surface::from_names(tool_names),
    }
}

/// Build the snapshot that `--write` would persist for the current build.
pub fn snapshot_of(live: &Live, feature_gated_tools: Vec<String>) -> Snapshot {
    Snapshot {
        schema_version: SCHEMA_VERSION,
        note: "Generated by `ix doctor --write`. The `names` lists are the oracle; \
               `count` is informational. Adding or removing a skill or MCP tool must \
               show up here as a reviewable name diff — do not hand-edit a count."
            .to_string(),
        skills: live.skills.clone(),
        mcp_tools: live.mcp_tools.clone(),
        feature_gated_tools,
    }
}

/// Names present on one side and not the other.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Drift {
    /// Live skills missing from the snapshot.
    pub skills_added: Vec<String>,
    /// Snapshot skills missing from the live registry.
    pub skills_removed: Vec<String>,
    /// Live MCP tools missing from the snapshot.
    pub tools_added: Vec<String>,
    /// Snapshot MCP tools missing from the live registry.
    pub tools_removed: Vec<String>,
}

impl Drift {
    /// True when the live build matches the snapshot exactly.
    pub fn is_clean(&self) -> bool {
        self.skills_added.is_empty()
            && self.skills_removed.is_empty()
            && self.tools_added.is_empty()
            && self.tools_removed.is_empty()
    }
}

/// Compare a live build against a snapshot, ignoring feature-gated tool names.
pub fn diff(snapshot: &Snapshot, live: &Live) -> Drift {
    let gated = &snapshot.feature_gated_tools;
    Drift {
        skills_added: missing(&live.skills.names, &snapshot.skills.names, &[]),
        skills_removed: missing(&snapshot.skills.names, &live.skills.names, &[]),
        tools_added: missing(&live.mcp_tools.names, &snapshot.mcp_tools.names, gated),
        tools_removed: missing(&snapshot.mcp_tools.names, &live.mcp_tools.names, gated),
    }
}

/// Names in `from` absent from `to`, skipping anything in `ignore`.
fn missing(from: &[String], to: &[String], ignore: &[String]) -> Vec<String> {
    from.iter()
        .filter(|n| !to.contains(n) && !ignore.contains(n))
        .cloned()
        .collect()
}
