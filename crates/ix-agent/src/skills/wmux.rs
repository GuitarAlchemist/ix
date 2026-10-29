//! `ix_wmux_snapshot` — what the wmux terminal multiplexer is showing, as rows.
//!
//! It runs the wmux CLI that the operator installed (`IX_WMUX_BIN`, else `wmux` on
//! `PATH`), read-only: `sidebar-state`, one `tree --workspace <id>` per workspace,
//! `agent-state` and `agent list`. It sends no key, answers no prompt and opens
//! nothing, so it is classified as a read (Tier 1).
//!
//! Only metadata leaves it, chosen field by field and checked value by value:
//! workspaces, panes and surfaces with their titles and the folder name of their
//! working directory, and each agent's state. Ids and states leave only as plain
//! identifiers, titles, labels and folder names only as plain names (no URL, path,
//! shell character or credential-shaped word), else as `null`. Screen contents,
//! prompts, markdown bodies, file paths, URLs and command lines stay out, because any
//! of them can carry a secret. A blocked agent's reason is free text the operator
//! reads: its URLs and paths become `<url>` and `<path>`, its secrets are masked and it
//! is clipped. One budget of time and bytes bounds the whole call.

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
    let launch = launcher(&|k: &str| std::env::var(k).ok().filter(|v| !v.is_empty()));
    let mut budget = Budget::new(TIMEOUT, MAX_OUTPUT_BYTES);
    let mut run = |args: &[&str]| run_json(&launch, args, &mut budget);
    let sidebar = run(&["sidebar-state"])?;
    let mut trees = Vec::new();
    for ws in workspaces_of(&sidebar).iter().take(MAX_WORKSPACES) {
        let Some(id) = ws
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| is_ident(id))
        else {
            continue;
        };
        trees.push((id.to_string(), run(&["tree", "--workspace", id])?));
    }
    let states = run(&["agent-state"])?;
    let agents = run(&["agent", "list"])?;
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

    let titles: BTreeMap<&str, Value> = workspaces_of_ref(sidebar)
        .filter_map(|w| Some((w.get("id")?.as_str()?, plain(w.get("title")))))
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
            let id = s
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| is_ident(id));
            let kind = match s.get("type").and_then(Value::as_str) {
                Some(k) if KINDS.contains(&k) => k,
                Some(_) => "other",
                None => "unknown",
            };
            *by_type.entry(kind.to_string()).or_default() += 1;
            let state = id.and_then(|id| by_surface.get(id)).copied();
            if let Some(id) = id {
                seen.insert(id);
            }
            let agent_state = state
                .and_then(|st| st.get("state"))
                .and_then(Value::as_str)
                .filter(|st| STATES.contains(st));
            if let Some(st) = agent_state {
                *by_state.entry(st.to_string()).or_default() += 1;
            }
            let cwd = if kind == "terminal" {
                s.get("currentCwd")
                    .or_else(|| s.get("cwd"))
                    .and_then(Value::as_str)
                    .map(folder_name)
                    .filter(|name| is_plain(name))
            } else {
                None
            };
            surfaces.push(json!({
                "agent_state": agent_state,
                "blocked_reason": state
                    .and_then(|st| st.get("blockedReason"))
                    .and_then(Value::as_str)
                    .map(|r| clip(&scrub(r), MAX_REASON_CHARS)),
                "blocked_since_ms": number(state.and_then(|st| st.get("blockedSince"))),
                "claude_session_id": ident(s.get("claudeSessionId")),
                "cwd_name": cwd,
                "pane_id": Some(*pane).filter(|p| is_ident(p)),
                "surface_id": id,
                "title": plain(s.get("customTitle").or_else(|| s.get("title"))),
                "type": kind,
                "updated_at_ms": number(state.and_then(|st| st.get("updatedAt"))),
                "workspace_id": Some(ws).filter(|w| is_ident(w)),
                "workspace_title": titles.get(ws).cloned().unwrap_or(Value::Null),
            }));
        }
        workspaces.push(json!({
            "active": w.get("isActive").and_then(Value::as_bool).unwrap_or(false),
            "cwd_name": w
                .get("cwd")
                .and_then(Value::as_str)
                .map(folder_name)
                .filter(|name| is_plain(name)),
            "panes": ws_panes.len(),
            "shell": w
                .get("shell")
                .and_then(Value::as_str)
                .map(file_name)
                .filter(|name| is_ident(name)),
            "surfaces": leaves.len(),
            "title": plain(w.get("title")),
            "workspace_id": Some(ws).filter(|w| is_ident(w)),
        }));
    }

    let agents: Vec<Value> = agents
        .get("agents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|a| {
            json!({
                "agent_id": ident(a.get("agentId")),
                "exit_code": number(a.get("exitCode")),
                "label": plain(a.get("label")),
                "pid": number(a.get("pid")),
                "spawn_time_ms": number(a.get("spawnTime")),
                "status": ident(a.get("status")),
                "surface_id": ident(a.get("surfaceId")),
                "workspace_id": ident(a.get("workspaceId")),
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

const KINDS: [&str; 4] = ["terminal", "browser", "markdown", "diff"];
const STATES: [&str; 3] = ["idle", "working", "blocked"];

/// An id or a short status: letters, digits and `_.:-` only, so nothing a shell, a path or
/// a URL needs.
fn is_ident(s: &str) -> bool {
    (1..=80).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
}

fn ident(v: Option<&Value>) -> Value {
    match v.and_then(Value::as_str) {
        Some(s) if is_ident(s) => Value::from(s),
        _ => Value::Null,
    }
}

fn number(v: Option<&Value>) -> Value {
    v.filter(|v| v.is_number()).cloned().unwrap_or(Value::Null)
}

/// A title, a label or a folder name leaves only when it reads as a plain name: no URL,
/// path, shell or query character, nothing that looks like a secret, at most 80 characters.
/// wmux titles are free text, so an allowlisted field name alone proves nothing.
fn is_plain(s: &str) -> bool {
    (1..=80).contains(&s.chars().count())
        && !s
            .chars()
            .any(|c| c.is_control() || "/\\?=&@%$<>|;`\"'".contains(c))
        && s.split_whitespace().all(|w| !looks_secret(w))
        && mask_secrets(s) == s
}

fn plain(v: Option<&Value>) -> Value {
    match v.and_then(Value::as_str) {
        Some(s) if is_plain(s) => Value::from(s),
        _ => Value::Null,
    }
}

/// A word shaped like a credential: a known token prefix, or a long run of letters and
/// digits.
fn looks_secret(word: &str) -> bool {
    const PREFIXES: [&str; 9] = [
        "sk-",
        "ghp_",
        "gho_",
        "ghs_",
        "ghu_",
        "github_pat_",
        "xox",
        "AKIA",
        "eyJ",
    ];
    let w = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
    PREFIXES.iter().any(|p| w.starts_with(p))
        || (w.len() >= 24
            && w.bytes().any(|b| b.is_ascii_digit())
            && w.bytes().any(|b| b.is_ascii_alphabetic())
            && w.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)))
}

/// A blocked reason is free text that the operator reads: URLs and paths become `<url>` and
/// `<path>`, credential-shaped words `***`, then [`mask_secrets`] masks named secrets.
fn scrub(text: &str) -> String {
    let words: Vec<&str> = text
        .split(' ')
        .map(|w| {
            if w.contains("://") {
                "<url>"
            } else if w.contains(['\\', '/']) {
                "<path>"
            } else if looks_secret(w) {
                "***"
            } else {
                w
            }
        })
        .collect();
    mask_secrets(&words.join(" "))
}

/// Masks the value of every `key=value`, `key: value` or `key value` whose key names a
/// secret, and the word after `Bearer`.
fn mask_secrets(text: &str) -> String {
    const MARKERS: [&str; 5] = ["token", "key", "secret", "pass", "auth"];
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
            _ => {
                out.push_str(word);
                let bare = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
                if !bare.is_empty() && (is_secret(bare) || bare.eq_ignore_ascii_case("bearer")) {
                    mask_next = true;
                }
            }
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
/// run (`WMUX_NODE`, else `node`, on `WMUX_CLI`), so no shim and no shell stands between the
/// tool and wmux; else the `wmux` found on `PATH`, which on Windows is a `.cmd` or `.ps1`
/// shim.
fn launcher(var: &dyn Fn(&str) -> Option<String>) -> Launcher {
    if let Some(bin) = var("IX_WMUX_BIN") {
        return for_program(bin);
    }
    match var("WMUX_CLI") {
        Some(cli) => Launcher {
            program: var("WMUX_NODE").unwrap_or_else(|| "node".to_string()),
            lead: vec![cli],
            electron: var("WMUX_NODE_ELECTRON").is_some(),
        },
        None => var("PATH")
            .and_then(|path| find_on_path(&path))
            .map(for_program)
            .unwrap_or_else(|| for_program("wmux".to_string())),
    }
}

/// The first `wmux` program on `path`. Only Windows needs the search: there `Command` looks
/// for `wmux.exe` alone, while an install gives `wmux.cmd` or `wmux.ps1`.
fn find_on_path(path: &str) -> Option<String> {
    if !cfg!(windows) {
        return None;
    }
    std::env::split_paths(path).find_map(|dir| {
        ["wmux.exe", "wmux.cmd", "wmux.bat", "wmux.ps1"]
            .iter()
            .map(|name| dir.join(name))
            .find(|p| p.is_file())
            .map(|p| p.display().to_string())
    })
}

/// A `.ps1` runs through `pwsh -File`. `Command` starts a `.cmd` or `.bat` through `cmd.exe`
/// and quotes its arguments, which are fixed verbs and ids that [`is_ident`] accepted.
fn for_program(program: String) -> Launcher {
    if program.to_ascii_lowercase().ends_with(".ps1") {
        Launcher {
            program: "pwsh".to_string(),
            lead: vec![
                "-NoProfile".into(),
                "-NonInteractive".into(),
                "-File".into(),
                program,
            ],
            electron: false,
        }
    } else {
        Launcher {
            program,
            lead: vec![],
            electron: false,
        }
    }
}

/// What one snapshot may spend on wmux, across all its commands.
struct Budget {
    deadline: Instant,
    bytes_left: u64,
}

impl Budget {
    fn new(time: Duration, bytes: u64) -> Self {
        Budget {
            deadline: Instant::now() + time,
            bytes_left: bytes,
        }
    }
}

/// Runs one wmux command and parses its JSON output, within what is left of `budget`. An
/// error names the program, never its path, and its stderr is scrubbed like a blocked
/// reason.
fn run_json(launch: &Launcher, args: &[&str], budget: &mut Budget) -> Result<Value, String> {
    let verb = args[0];
    if Instant::now() >= budget.deadline {
        return Err(format!(
            "wmux {verb}: the snapshot's time budget ran out before it started"
        ));
    }
    let name = file_name(&launch.program).to_string();
    let mut command = Command::new(&launch.program);
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
                format!("wmux CLI not found ({name}); run inside wmux or set IX_WMUX_BIN")
            }
            _ => format!("could not start wmux ({name}): {}", e.kind()),
        })?;
    let mut out_pipe = child.stdout.take().ok_or("no stdout for wmux")?;
    let mut err_pipe = child.stderr.take().ok_or("no stderr for wmux")?;
    // Each reader sends what it read as soon as it reaches EOF or its cap, then drains. A
    // process the CLI started can hold a pipe open after the CLI exits; the call does not
    // wait for it past the deadline, and the reader is left to finish on its own.
    let (tx, rx) = std::sync::mpsc::channel();
    let cap = budget.bytes_left;
    let out_tx = tx.clone();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut out_pipe).take(cap + 1).read_to_end(&mut buf);
        let _ = out_tx.send((true, buf));
        let _ = std::io::copy(&mut out_pipe, &mut std::io::sink());
    });
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = (&mut err_pipe).take(64 * 1024).read_to_end(&mut buf);
        let _ = tx.send((false, buf));
        let _ = std::io::copy(&mut err_pipe, &mut std::io::sink());
    });
    let stopped = |what: &str| {
        format!(
            "wmux {verb}: {what} within the snapshot's {} s budget, and was stopped",
            TIMEOUT.as_secs()
        )
    };
    let status = loop {
        match child.try_wait().map_err(|e| e.kind().to_string())? {
            Some(status) => break status,
            None if Instant::now() >= budget.deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(stopped("did not finish"));
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    };
    let (mut out, mut err) = (None, None);
    while out.is_none() || err.is_none() {
        let left = budget.deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(left) {
            Ok((true, buf)) => out = Some(buf),
            Ok((false, buf)) => err = Some(buf),
            Err(_) => return Err(stopped("exited, but its output stayed open")),
        }
    }
    let (out, err) = (out.unwrap_or_default(), err.unwrap_or_default());
    if !status.success() {
        let msg = String::from_utf8_lossy(&err);
        return Err(format!("wmux {verb}: {}", clip(&scrub(msg.trim()), 500)));
    }
    if out.len() as u64 > budget.bytes_left {
        return Err(format!(
            "wmux {verb}: the snapshot printed more than {MAX_OUTPUT_BYTES} bytes in all"
        ));
    }
    budget.bytes_left -= out.len() as u64;
    serde_json::from_slice(&out).map_err(|e| format!("wmux {verb}: not JSON: {e}"))
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

    /// A field name on the allowlist is not enough: wmux titles, labels, folder and shell
    /// names are free text, so each value leaves only if it reads as a plain name.
    #[test]
    fn free_text_with_a_url_a_path_a_command_or_a_secret_leaves_as_null() {
        let sidebar = json!({ "workspaces": [{
            "id": "w1", "title": "curl https://internal/?auth=ghp_x", "isActive": true,
            "cwd": "C:\\work\\ghp_0123456789abcdefABCDEF0123", "shell": "pwsh.exe -Command x"
        }]});
        let tree = json!({ "tree": { "type": "leaf", "paneId": "p 1", "surfaces": [
            { "id": "s1", "type": "terminal", "customTitle": "C:\\Users\\alice\\notes",
              "cwd": "C:\\work\\proj", "claudeSessionId": "id; rm -rf /" },
            { "id": "s2", "type": "<script>", "title": "Claude Code" }
        ]}});
        let states = json!({ "states": [
            { "surfaceId": "s1", "state": "rm -rf", "updatedAt": "soon" }
        ]});
        let agents = json!({ "agents": [{
            "agentId": "a1", "label": "sk-live-0123456789abcdef", "status": "running now",
            "pid": "12", "surfaceId": "s1", "workspaceId": "w1"
        }]});
        let snap = normalize(&sidebar, &[("w1".into(), tree)], &states, &agents);
        let ws = &snap["workspaces"][0];
        assert_eq!(ws["title"], Value::Null);
        assert_eq!(ws["cwd_name"], Value::Null);
        assert_eq!(ws["shell"], Value::Null);
        let s1 = &snap["surfaces"][0];
        assert_eq!(s1["title"], Value::Null);
        assert_eq!(s1["pane_id"], Value::Null);
        assert_eq!(s1["claude_session_id"], Value::Null);
        assert_eq!(s1["agent_state"], Value::Null);
        assert_eq!(s1["updated_at_ms"], Value::Null);
        assert_eq!(s1["workspace_title"], Value::Null);
        assert_eq!(s1["cwd_name"], "proj");
        let s2 = &snap["surfaces"][1];
        assert_eq!(s2["type"], "other");
        assert_eq!(s2["title"], "Claude Code");
        assert_eq!(
            snap["counts"]["by_type"],
            json!({ "other": 1, "terminal": 1 })
        );
        assert_eq!(snap["counts"]["by_state"], json!({}));
        let a = &snap["agents"][0];
        assert_eq!(a["label"], Value::Null);
        assert_eq!(a["status"], Value::Null);
        assert_eq!(a["pid"], Value::Null);
        assert_eq!(a["agent_id"], "a1");
    }

    #[test]
    fn a_blocked_reason_loses_its_urls_paths_and_bearer_secrets() {
        let reason = "Fetch https://host/x?token=abc with Authorization: Bearer sk-live-1 \
                      from C:\\Users\\alice\\.ssh\\id_rsa using password hunter2 and \
                      ghp_0123456789abcdefABCDEF0123";
        let states = json!({ "states": [{
            "surfaceId": "s1", "state": "blocked", "blockedReason": reason
        }]});
        let tree = json!({ "tree": { "type": "leaf", "paneId": "p1",
            "surfaces": [{ "id": "s1", "type": "terminal" }] } });
        let sidebar = json!({ "workspaces": [{ "id": "w1", "title": "t" }] });
        let snap = normalize(
            &sidebar,
            &[("w1".into(), tree)],
            &states,
            &json!({ "agents": [] }),
        );
        let out = snap["surfaces"][0]["blocked_reason"].as_str().unwrap();
        for leak in [
            "abc",
            "host",
            "sk-live-1",
            "alice",
            "id_rsa",
            "hunter2",
            "ghp_",
        ] {
            assert!(!out.contains(leak), "{leak} in {out}");
        }
        assert!(out.starts_with("Fetch <url> with"), "{out}");
        assert!(out.contains("from <path> using"), "{out}");
    }

    /// A launcher that runs a short script: PowerShell on Windows, `sh` elsewhere. The wmux
    /// arguments land after a `#` or in `$0`, so the script ignores them.
    fn script(windows: &str, unix: &str) -> Launcher {
        if cfg!(windows) {
            Launcher {
                program: "pwsh".into(),
                lead: vec![
                    "-NoProfile".into(),
                    "-NonInteractive".into(),
                    "-Command".into(),
                    format!("{windows} #"),
                ],
                electron: false,
            }
        } else {
            Launcher {
                program: "sh".into(),
                lead: vec!["-c".into(), unix.into()],
                electron: false,
            }
        }
    }

    fn fresh() -> Budget {
        Budget::new(TIMEOUT, MAX_OUTPUT_BYTES)
    }

    #[test]
    fn an_error_names_the_program_and_never_its_path_or_a_secret() {
        let missing = Launcher {
            program: if cfg!(windows) {
                r"C:\Users\alice\private\wmux.exe"
            } else {
                "/home/alice/private/wmux"
            }
            .into(),
            lead: vec![],
            electron: false,
        };
        let err = run_json(&missing, &["sidebar-state"], &mut fresh()).unwrap_err();
        assert!(!err.contains("alice"), "{err}");
        assert!(err.contains("wmux"), "{err}");
        let failing = script(
            r"[Console]::Error.WriteLine('failed at C:\Users\alice\x token=abc'); exit 1",
            "echo 'failed at /home/alice/x token=abc' >&2; exit 1",
        );
        let err = run_json(&failing, &["sidebar-state"], &mut fresh()).unwrap_err();
        for leak in ["alice", "abc"] {
            assert!(!err.contains(leak), "{leak} in {err}");
        }
        assert!(err.contains("failed at <path>"), "{err}");
    }

    #[test]
    fn one_deadline_bounds_the_whole_call() {
        let mut budget = Budget::new(Duration::from_secs(1), MAX_OUTPUT_BYTES);
        let started = Instant::now();
        let slow = script("Start-Sleep 30", "sleep 30");
        assert!(run_json(&slow, &["sidebar-state"], &mut budget).is_err());
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "{:?}",
            started.elapsed()
        );
        let fast = script("'{}'", "echo '{}'");
        let err = run_json(&fast, &["agent-state"], &mut budget).unwrap_err();
        assert!(err.contains("ran out"), "{err}");
    }

    /// A wrapper that exits after starting a process that keeps its output open: the call
    /// still ends at the deadline instead of waiting for that process.
    #[test]
    fn a_descendant_that_keeps_the_output_open_does_not_hang_the_call() {
        let mut budget = Budget::new(Duration::from_secs(3), MAX_OUTPUT_BYTES);
        let started = Instant::now();
        let wrapper = script(
            "Start-Process -NoNewWindow pwsh -ArgumentList '-NoProfile','-Command','Start-Sleep 20'; '{}'",
            "sleep 20 & echo '{}'",
        );
        let _ = run_json(&wrapper, &["sidebar-state"], &mut budget);
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn the_output_budget_is_shared_by_the_whole_call() {
        let mut budget = Budget::new(TIMEOUT, 10);
        let chatty = script("'x' * 100", "printf '%0100d' 0");
        let err = run_json(&chatty, &["sidebar-state"], &mut budget).unwrap_err();
        assert!(err.contains("more than"), "{err}");
    }

    /// Outside wmux the `wmux` on PATH is a `.cmd` or `.ps1` shim; both are found and run.
    #[cfg(windows)]
    #[test]
    fn a_cmd_or_ps1_shim_on_path_is_found_and_run() {
        let root = std::env::temp_dir().join(format!("ix-wmux-shim-{}", std::process::id()));
        let cmd_dir = root.join("cmd");
        let ps1_dir = root.join("ps1");
        std::fs::create_dir_all(&cmd_dir).unwrap();
        std::fs::create_dir_all(&ps1_dir).unwrap();
        std::fs::write(cmd_dir.join("wmux.cmd"), "@echo {\"from\":\"cmd\"}\r\n").unwrap();
        std::fs::write(ps1_dir.join("wmux.ps1"), "'{\"from\":\"ps1\"}'\r\n").unwrap();
        for (dir, from) in [(&cmd_dir, "cmd"), (&ps1_dir, "ps1")] {
            let path = format!("{};{}", root.join("empty").display(), dir.display());
            let launch = launcher(&|k: &str| (k == "PATH").then(|| path.clone()));
            let out = run_json(&launch, &["sidebar-state"], &mut fresh())
                .unwrap_or_else(|e| panic!("{from}: {e} ({launch:?})"));
            assert_eq!(out["from"], from);
        }
        let bin = ps1_dir.join("wmux.ps1").display().to_string();
        let launch = launcher(&|k: &str| (k == "IX_WMUX_BIN").then(|| bin.clone()));
        let out = run_json(&launch, &["sidebar-state"], &mut fresh()).unwrap();
        assert_eq!(out["from"], "ps1");
        let _ = std::fs::remove_dir_all(&root);
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

    /// Inside wmux the tool runs the Node client the shims run, with no shell.
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
