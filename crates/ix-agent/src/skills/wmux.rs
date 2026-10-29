//! `ix_wmux_snapshot` — what the wmux terminal multiplexer is showing, as rows.
//!
//! It runs the wmux CLI that the operator installed (`IX_WMUX_BIN`, else `wmux` on
//! `PATH`), read-only: `sidebar-state`, one `tree --workspace <id>` per workspace,
//! `agent-state` and `agent list`. It sends no key, answers no prompt and opens
//! nothing, so it is classified as a read (Tier 1).
//!
//! Only metadata leaves it, chosen field by field: workspaces, panes and surfaces
//! with their titles and working directories, and each agent's state. Screen
//! contents, prompts, markdown bodies, file paths, URLs and command lines stay out,
//! because any of them can carry a secret. A blocked agent's reason is clipped, and
//! `key=value` secrets in it are masked.

use ix_skill_macros::ix_skill;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const TIMEOUT: Duration = Duration::from_secs(10);
const MAX_OUTPUT_BYTES: u64 = 8 * 1024 * 1024;
const MAX_WORKSPACES: usize = 64;
const MAX_REASON_CHARS: usize = 300;

fn wmux_snapshot_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": false
    })
}

fn wmux_snapshot_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "workspaces": { "type": "array", "items": { "type": "object" },
                "description": "One row per workspace: workspace_id, title, active, cwd_name (last folder of its directory), shell, panes, surfaces." },
            "surfaces": { "type": "array", "items": { "type": "object" },
                "description": "One row per surface: workspace_id, workspace_title, pane_id, surface_id, type (terminal, browser, markdown, diff), title, cwd_name (last folder of the working directory, terminals only), claude_session_id, agent_state (idle, working, blocked, or null), blocked_reason, blocked_since_ms, updated_at_ms." },
            "agents": { "type": "array", "items": { "type": "object" },
                "description": "Agents started with `wmux agent spawn`: agent_id, surface_id, workspace_id, label, status, pid, exit_code, spawn_time_ms. The command line is left out." },
            "counts": { "type": "object",
                "description": "workspaces, panes, surfaces, by_type, by_state, and orphan_states (agent states whose surface is in no workspace tree)." }
        },
        "required": ["workspaces", "surfaces", "agents", "counts"]
    })
}

#[ix_skill(
    domain = "wmux",
    name = "wmux.snapshot",
    governance = "safety",
    schema_fn = "crate::skills::wmux::wmux_snapshot_schema",
    output_schema_fn = "crate::skills::wmux::wmux_snapshot_output_schema"
)]
pub fn wmux_snapshot(params: Value) -> Result<Value, String> {
    if params.as_object().is_some_and(|o| !o.is_empty()) {
        return Err("wmux.snapshot takes no arguments".into());
    }
    let sidebar = run_json(&["sidebar-state"])?;
    let mut trees = Vec::new();
    for ws in workspaces_of(&sidebar).iter().take(MAX_WORKSPACES) {
        let Some(id) = ws.get("id").and_then(Value::as_str) else {
            continue;
        };
        trees.push((id.to_string(), run_json(&["tree", "--workspace", id])?));
    }
    let states = run_json(&["agent-state"])?;
    let agents = run_json(&["agent", "list"])?;
    Ok(normalize(&sidebar, &trees, &states, &agents))
}

fn workspaces_of(sidebar: &Value) -> Vec<Value> {
    sidebar
        .get("workspaces")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Builds the rows from what wmux printed. Pure, so the shape is tested on real,
/// anonymized output.
pub(crate) fn normalize(
    sidebar: &Value,
    trees: &[(String, Value)],
    states: &Value,
    agents: &Value,
) -> Value {
    let mut by_surface: BTreeMap<&str, &Value> = BTreeMap::new();
    for s in states
        .get("states")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(id) = s.get("surfaceId").and_then(Value::as_str) {
            by_surface.insert(id, s);
        }
    }

    let titles: BTreeMap<&str, &Value> = workspaces_of_ref(sidebar)
        .filter_map(|w| {
            Some((
                w.get("id")?.as_str()?,
                w.get("title").unwrap_or(&Value::Null),
            ))
        })
        .collect();
    let mut leaves_of: BTreeMap<&str, Vec<(&str, &Value)>> = BTreeMap::new();
    for (ws, tree) in trees {
        let mut leaves = Vec::new();
        if let Some(root) = tree.get("tree") {
            collect_surfaces(root, &mut leaves);
        }
        leaves_of.insert(ws.as_str(), leaves);
    }

    let mut surfaces = Vec::new();
    let mut by_type: BTreeMap<String, u64> = BTreeMap::new();
    let mut by_state: BTreeMap<String, u64> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut panes = 0usize;
    let mut workspaces = Vec::new();
    for w in workspaces_of_ref(sidebar) {
        let Some(ws) = w.get("id").and_then(Value::as_str) else {
            continue;
        };
        let leaves = leaves_of.get(ws).map(Vec::as_slice).unwrap_or_default();
        let ws_panes: BTreeSet<&str> = leaves.iter().map(|(pane, _)| *pane).collect();
        panes += ws_panes.len();
        for (pane, s) in leaves {
            let id = s.get("id").and_then(Value::as_str);
            let kind = s.get("type").and_then(Value::as_str);
            *by_type
                .entry(kind.unwrap_or("unknown").to_string())
                .or_default() += 1;
            let state = id.and_then(|id| by_surface.get(id)).copied();
            if let Some(id) = id {
                seen.insert(id);
            }
            if let Some(st) = state.and_then(|st| st.get("state")).and_then(Value::as_str) {
                *by_state.entry(st.to_string()).or_default() += 1;
            }
            let cwd = if kind == Some("terminal") {
                s.get("currentCwd")
                    .or_else(|| s.get("cwd"))
                    .and_then(Value::as_str)
                    .map(folder_name)
            } else {
                None
            };
            surfaces.push(json!({
                "agent_state": state.and_then(|st| st.get("state")),
                "blocked_reason": state
                    .and_then(|st| st.get("blockedReason"))
                    .and_then(Value::as_str)
                    .map(|r| clip(&mask_secrets(r), MAX_REASON_CHARS)),
                "blocked_since_ms": state.and_then(|st| st.get("blockedSince")),
                "claude_session_id": s.get("claudeSessionId"),
                "cwd_name": cwd,
                "pane_id": pane,
                "surface_id": id,
                "title": s.get("customTitle").or_else(|| s.get("title")),
                "type": kind,
                "updated_at_ms": state.and_then(|st| st.get("updatedAt")),
                "workspace_id": ws,
                "workspace_title": titles.get(ws),
            }));
        }
        workspaces.push(json!({
            "active": w.get("isActive").and_then(Value::as_bool).unwrap_or(false),
            "cwd_name": w.get("cwd").and_then(Value::as_str).map(folder_name),
            "panes": ws_panes.len(),
            "shell": w.get("shell").and_then(Value::as_str).map(file_name),
            "surfaces": leaves.len(),
            "title": w.get("title"),
            "workspace_id": ws,
        }));
    }

    let agents: Vec<Value> = agents
        .get("agents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|a| {
            json!({
                "agent_id": a.get("agentId"),
                "exit_code": a.get("exitCode"),
                "label": a.get("label"),
                "pid": a.get("pid"),
                "spawn_time_ms": a.get("spawnTime"),
                "status": a.get("status"),
                "surface_id": a.get("surfaceId"),
                "workspace_id": a.get("workspaceId"),
            })
        })
        .collect();

    let orphan_states = by_surface.keys().filter(|id| !seen.contains(*id)).count();
    json!({
        "agents": agents,
        "counts": {
            "by_state": by_state,
            "by_type": by_type,
            "orphan_states": orphan_states,
            "panes": panes,
            "surfaces": surfaces.len(),
            "workspaces": workspaces.len(),
        },
        "surfaces": surfaces,
        "workspaces": workspaces,
    })
}

fn workspaces_of_ref(sidebar: &Value) -> impl Iterator<Item = &Value> {
    sidebar
        .get("workspaces")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
}

/// Every surface under `node`, with the pane that holds it, depth first.
fn collect_surfaces<'a>(node: &'a Value, out: &mut Vec<(&'a str, &'a Value)>) {
    if let Some(pane) = node.get("paneId").and_then(Value::as_str) {
        for s in node
            .get("surfaces")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            out.push((pane, s));
        }
    }
    for child in node
        .get("children")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        collect_surfaces(child, out);
    }
}

fn file_name(path: &str) -> &str {
    path.rsplit(['\\', '/']).next().unwrap_or(path)
}

/// The last folder of a working directory: which project a pane is in, without the path
/// (and the user name) above it.
fn folder_name(path: &str) -> &str {
    file_name(path.trim_end_matches(['\\', '/']))
}

/// Masks the value of every `key=value` or `key: value` whose key names a secret.
fn mask_secrets(text: &str) -> String {
    const MARKERS: [&str; 4] = ["token", "key", "secret", "pass"];
    let is_secret = |k: &str| {
        let k = k.to_ascii_lowercase();
        MARKERS.iter().any(|m| k.contains(m))
    };
    let mut out = String::with_capacity(text.len());
    let mut mask_next = false;
    for (i, word) in text.split(' ').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        if mask_next && !word.is_empty() {
            out.push_str("***");
            mask_next = false;
            continue;
        }
        match word.find(['=', ':']) {
            Some(at) if is_secret(&word[..at]) => {
                out.push_str(&word[..=at]);
                if at + 1 < word.len() {
                    out.push_str("***");
                } else {
                    mask_next = true;
                }
            }
            _ => out.push_str(word),
        }
    }
    out
}

/// How to start the wmux CLI: the program, the arguments before wmux's own, and whether
/// the program is Electron that must run as Node.
#[derive(Debug, PartialEq)]
struct Launcher {
    program: String,
    lead: Vec<String>,
    electron: bool,
}

/// `IX_WMUX_BIN` when the operator set it; else, inside wmux, the Node client its shims
/// run (`WMUX_NODE`, else `node`, on `WMUX_CLI`), so no `.cmd` or `.ps1` shim and no shell
/// stands between the tool and wmux; else `wmux` on PATH.
fn launcher(var: &dyn Fn(&str) -> Option<String>) -> Launcher {
    if let Some(bin) = var("IX_WMUX_BIN") {
        return Launcher {
            program: bin,
            lead: vec![],
            electron: false,
        };
    }
    match var("WMUX_CLI") {
        Some(cli) => Launcher {
            program: var("WMUX_NODE").unwrap_or_else(|| "node".to_string()),
            lead: vec![cli],
            electron: var("WMUX_NODE_ELECTRON").is_some(),
        },
        None => Launcher {
            program: "wmux".to_string(),
            lead: vec![],
            electron: false,
        },
    }
}

/// Runs one wmux command and parses its JSON output.
fn run_json(args: &[&str]) -> Result<Value, String> {
    let launch = launcher(&|k: &str| std::env::var(k).ok().filter(|v| !v.is_empty()));
    let bin = &launch.program;
    let mut command = Command::new(bin);
    if launch.electron {
        command.env("ELECTRON_RUN_AS_NODE", "1");
    }
    let mut child = command
        .args(&launch.lead)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                format!("wmux CLI not found ({bin}); run inside wmux or set IX_WMUX_BIN")
            }
            _ => format!("could not start wmux ({bin}): {e}"),
        })?;
    let mut out_pipe = child.stdout.take().ok_or("no stdout for wmux")?;
    let mut err_pipe = child.stderr.take().ok_or("no stderr for wmux")?;
    let stdout = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut out_pipe)
            .take(MAX_OUTPUT_BYTES + 1)
            .read_to_end(&mut buf);
        let _ = std::io::copy(&mut out_pipe, &mut std::io::sink());
        buf
    });
    let stderr = std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut err_pipe).take(64 * 1024).read_to_end(&mut buf);
        let _ = std::io::copy(&mut err_pipe, &mut std::io::sink());
        buf
    });
    let deadline = Instant::now() + TIMEOUT;
    let status = loop {
        match child.try_wait().map_err(|e| e.to_string())? {
            Some(status) => break status,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "wmux {} took more than {} s and was stopped",
                    args[0],
                    TIMEOUT.as_secs()
                ));
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    };
    let out = stdout.join().map_err(|_| "wmux stdout reader panicked")?;
    let err = stderr.join().map_err(|_| "wmux stderr reader panicked")?;
    if !status.success() {
        let msg = String::from_utf8_lossy(&err);
        return Err(format!(
            "wmux {}: {}",
            args[0],
            clip(&mask_secrets(msg.trim()), 500)
        ));
    }
    if out.len() as u64 > MAX_OUTPUT_BYTES {
        return Err(format!(
            "wmux {} printed more than {MAX_OUTPUT_BYTES} bytes",
            args[0]
        ));
    }
    serde_json::from_slice(&out).map_err(|e| format!("wmux {}: not JSON: {e}", args[0]))
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Value {
        let text = match name {
            "sidebar-state" => include_str!("../../tests/fixtures/wmux/sidebar-state.json"),
            "agent-state" => include_str!("../../tests/fixtures/wmux/agent-state.json"),
            "agent-list" => include_str!("../../tests/fixtures/wmux/agent-list.json"),
            "tree-1" => include_str!("../../tests/fixtures/wmux/tree-1.json"),
            "tree-2" => include_str!("../../tests/fixtures/wmux/tree-2.json"),
            "tree-3" => include_str!("../../tests/fixtures/wmux/tree-3.json"),
            "tree-4" => include_str!("../../tests/fixtures/wmux/tree-4.json"),
            "tree-5" => include_str!("../../tests/fixtures/wmux/tree-5.json"),
            "tree-6" => include_str!("../../tests/fixtures/wmux/tree-6.json"),
            "tree-7" => include_str!("../../tests/fixtures/wmux/tree-7.json"),
            "tree-8" => include_str!("../../tests/fixtures/wmux/tree-8.json"),
            "tree-9" => include_str!("../../tests/fixtures/wmux/tree-9.json"),
            "tree-10" => include_str!("../../tests/fixtures/wmux/tree-10.json"),
            "tree-11" => include_str!("../../tests/fixtures/wmux/tree-11.json"),
            _ => unreachable!("{name}"),
        };
        serde_json::from_str(text).unwrap()
    }

    /// The snapshot of the machine the fixtures were captured on (2026-10-29, anonymized):
    /// tree `i` is the tree of the sidebar's workspace `i`.
    fn captured() -> Value {
        let sidebar = fixture("sidebar-state");
        let trees: Vec<(String, Value)> = workspaces_of(&sidebar)
            .iter()
            .enumerate()
            .map(|(i, ws)| {
                let id = ws["id"].as_str().unwrap().to_string();
                (id, fixture(&format!("tree-{}", i + 1)))
            })
            .collect();
        normalize(
            &sidebar,
            &trees,
            &fixture("agent-state"),
            &fixture("agent-list"),
        )
    }

    #[test]
    fn the_captured_wmux_comes_out_as_rows_with_the_counts_wmux_showed() {
        let snap = captured();
        let counts = &snap["counts"];
        assert_eq!(counts["workspaces"], 11);
        assert_eq!(counts["panes"], 18);
        assert_eq!(counts["surfaces"], 37);
        assert_eq!(
            counts["by_type"],
            json!({ "browser": 7, "diff": 10, "markdown": 7, "terminal": 13 })
        );
        assert_eq!(
            counts["by_state"],
            json!({ "blocked": 2, "idle": 9, "working": 1 })
        );
        assert_eq!(counts["orphan_states"], 0);
        assert_eq!(snap["workspaces"].as_array().unwrap().len(), 11);
        assert_eq!(snap["surfaces"].as_array().unwrap().len(), 37);
        assert_eq!(snap["agents"].as_array().unwrap().len(), 1);
        let active: Vec<_> = snap["workspaces"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|w| w["active"] == true)
            .collect();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn a_terminal_row_joins_its_tree_and_its_agent_state() {
        let snap = captured();
        // id-13 is a terminal in workspace id-1's tree, with an idle Claude session id-14.
        let row = snap["surfaces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|s| s["surface_id"] == "id-13")
            .expect("surface id-13");
        assert_eq!(row["workspace_id"], "id-1");
        assert_eq!(row["workspace_title"], "text-1");
        assert_eq!(row["pane_id"], "id-12");
        assert_eq!(row["type"], "terminal");
        assert_eq!(row["title"], "text-12");
        assert_eq!(row["cwd_name"], "proj-1");
        assert!(row.get("cwd").is_none());
        assert_eq!(row["claude_session_id"], "id-14");
        assert_eq!(row["agent_state"], "idle");
        assert_eq!(row["updated_at_ms"], 1792900997501u64);
    }

    #[test]
    fn only_listed_fields_leave_and_no_url_path_body_or_command_line() {
        let text = captured().to_string();
        for leak in [
            "<url>",
            "<cmd>",
            "<dropped>",
            "markdownContent",
            "browserHistory",
            "\"url\"",
            "\"cmd\"",
            // Every captured working directory is `<home>\proj-N`: only `proj-N` may leave.
            "<home>",
        ] {
            assert!(!text.contains(leak), "{leak} leaked");
        }
        let agent = &captured()["agents"][0];
        assert_eq!(
            agent
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            [
                "agent_id",
                "exit_code",
                "label",
                "pid",
                "spawn_time_ms",
                "status",
                "surface_id",
                "workspace_id"
            ]
        );
    }

    #[test]
    fn a_blocked_reason_is_clipped_and_its_secrets_are_masked() {
        let states = json!({ "states": [{
            "surfaceId": "s1", "state": "blocked", "updatedAt": 1, "blockedSince": 1,
            "blockedReason": format!("Run curl -H token=abc123 api_key: xyz PASSWORD=hunter2 {}", "x".repeat(400))
        }]});
        let tree = json!({ "tree": { "type": "leaf", "paneId": "p1",
            "surfaces": [{ "id": "s1", "type": "terminal", "cwd": "c" }] } });
        let sidebar = json!({ "workspaces": [{ "id": "w1", "title": "t", "isActive": true }] });
        let snap = normalize(
            &sidebar,
            &[("w1".into(), tree)],
            &states,
            &json!({ "agents": [] }),
        );
        let reason = snap["surfaces"][0]["blocked_reason"].as_str().unwrap();
        for secret in ["abc123", "xyz", "hunter2"] {
            assert!(!reason.contains(secret), "{secret} in {reason}");
        }
        assert!(reason.contains("token=***"), "{reason}");
        assert!(
            reason.chars().count() <= MAX_REASON_CHARS + 1,
            "{}",
            reason.chars().count()
        );
    }

    #[test]
    fn arguments_are_refused_before_wmux_is_started() {
        assert!(wmux_snapshot(json!({ "send": "rm -rf /" }))
            .unwrap_err()
            .contains("no arguments"));
    }

    #[test]
    fn a_working_directory_leaves_as_its_folder_name_only() {
        assert_eq!(folder_name("C:\\Users\\someone\\repos\\ix\\"), "ix");
        assert_eq!(folder_name("/home/someone/ix"), "ix");
        assert_eq!(folder_name("C:\\"), "C:");
    }

    /// On Windows the `wmux` on PATH is a `.cmd` or `.ps1` shim, which `Command` cannot
    /// start; inside wmux the tool runs the Node client the shims run, with no shell.
    #[test]
    fn inside_wmux_the_node_client_it_injected_runs_without_a_shell() {
        fn launch(vars: &[(&str, &str)]) -> Launcher {
            launcher(&|k: &str| {
                vars.iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            })
        }
        let all = [
            ("WMUX_CLI", "cli.js"),
            ("WMUX_NODE", "electron.exe"),
            ("WMUX_NODE_ELECTRON", "1"),
        ];
        assert_eq!(
            launch(&all),
            Launcher {
                program: "electron.exe".into(),
                lead: vec!["cli.js".into()],
                electron: true
            }
        );
        assert_eq!(
            launch(&[("WMUX_CLI", "cli.js")]),
            Launcher {
                program: "node".into(),
                lead: vec!["cli.js".into()],
                electron: false
            }
        );
        let mut operator = all.to_vec();
        operator.push(("IX_WMUX_BIN", "my-wmux"));
        assert_eq!(
            launch(&operator),
            Launcher {
                program: "my-wmux".into(),
                lead: vec![],
                electron: false
            }
        );
        assert_eq!(
            launch(&[]),
            Launcher {
                program: "wmux".into(),
                lead: vec![],
                electron: false
            }
        );
    }

    /// Pins the pipeline example of `docs/pipelines/dag-execution.md`: the surfaces go
    /// straight into `ix_duckdb_query` as a table. Skips where the DuckDB CLI is missing,
    /// and fails instead under `IX_REQUIRE_DUCKDB=1`, as `tests/duckdb_query.rs` does.
    #[test]
    fn the_surfaces_load_as_a_duckdb_table_that_finds_the_blocked_panes() {
        let bin = std::env::var("IX_DUCKDB_BIN").unwrap_or_else(|_| "duckdb".into());
        let found = Command::new(&bin)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        let required =
            std::env::var_os("IX_REQUIRE_DUCKDB").is_some_and(|v| !v.is_empty() && v != "0");
        assert!(
            found || !required,
            "IX_REQUIRE_DUCKDB is set but the DuckDB CLI ({bin}) cannot run"
        );
        if !found {
            eprintln!(
                "SKIPPED: DuckDB CLI ({bin}) not found; the wmux pipeline example is unchecked."
            );
            return;
        }
        let out = crate::skills::duckdb::duckdb_query(json!({
            "sql": "SELECT workspace_title, title, blocked_reason FROM surfaces WHERE agent_state = 'blocked' ORDER BY workspace_title",
            "tables": { "surfaces": captured()["surfaces"] }
        }))
        .expect("query runs");
        // Read off the raw fixtures: the two blocked states, joined by hand to their trees.
        // The second terminal has no title in wmux.
        assert_eq!(
            out["rows"],
            json!([
                { "workspace_title": "text-8", "title": "text-18", "blocked_reason": "text-20" },
                { "workspace_title": "text-9", "title": null, "blocked_reason": "text-20" }
            ])
        );
    }

    /// Against the wmux this test runs in: `cargo test -p ix-agent --lib -- --ignored wmux`.
    #[test]
    #[ignore = "needs a running wmux"]
    fn a_live_wmux_answers_with_rows_and_consistent_counts() {
        let snap = wmux_snapshot(json!({})).expect("wmux answers");
        let surfaces = snap["surfaces"].as_array().unwrap();
        assert!(snap["counts"]["workspaces"].as_u64().unwrap() >= 1);
        assert_eq!(
            snap["counts"]["surfaces"].as_u64().unwrap() as usize,
            surfaces.len()
        );
        assert!(surfaces
            .iter()
            .all(|s| s["workspace_title"] != Value::Null || s["workspace_id"].is_string()));
        let rows = surfaces
            .iter()
            .chain(snap["workspaces"].as_array().unwrap());
        for name in rows.filter_map(|r| r["cwd_name"].as_str()) {
            assert!(!name.contains(['\\', '/']), "a path left: {name}");
        }
    }
}
