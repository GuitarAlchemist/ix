//! `ix_wmux_snapshot` — what the wmux terminal multiplexer is showing, as rows.
//!
//! It runs the wmux CLI (`IX_WMUX_BIN`; else, inside wmux, the Node client wmux injected;
//! else `wmux` on `PATH`), read-only: `sidebar-state`, one `tree --workspace <id>` per workspace,
//! `agent-state` and `agent list`. It sends no key, answers no prompt and opens
//! nothing, so it is classified as a read (Tier 1).
//!
//! Only metadata leaves it, chosen field by field and checked value by value:
//! workspaces, panes and surfaces with their titles and the folder name of their
//! working directory, and each agent's state. Ids, states and numbers leave only in
//! their own shape, else as `null`. Screen contents, prompts, markdown bodies, file
//! paths and command lines stay out. Titles, labels, folder names and a blocked
//! agent's reason are free text the operator wrote or reads, so they keep their words,
//! masked as well as a filter can: URLs and host:port pairs become `<url>`, paths
//! `<path>`, credential-shaped words and the word after a secret-naming one `***`, and
//! each is clipped. A secret typed as an ordinary word can still leave; the operator
//! accepted that residual (2026-10-29). One budget of time and bytes bounds the whole
//! call, finding and starting the CLI included. A descendant of the CLI that keeps its
//! pipes open is not killed: its reader threads stay behind, at most two 64 KiB buffers
//! each, until the pipes close or their next send finds the call gone. A CLI the system refuses to kill is left to it, not waited for.

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
    let mut budget = Budget::new(TIMEOUT, MAX_OUTPUT_BYTES);
    let launch = within(&budget, || {
        launcher(&|k: &str| std::env::var(k).ok().filter(|v| !v.is_empty()))
    })?;
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
        .filter_map(|w| {
            Some((
                w.get("id")?.as_str()?,
                text(w.get("title").and_then(Value::as_str)),
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
                "cwd_name": text(cwd),
                "pane_id": Some(*pane).filter(|p| is_ident(p)),
                "surface_id": id,
                "title": text(
                    s.get("customTitle")
                        .or_else(|| s.get("title"))
                        .and_then(Value::as_str)
                ),
                "type": kind,
                "updated_at_ms": number(state.and_then(|st| st.get("updatedAt"))),
                "workspace_id": Some(ws).filter(|w| is_ident(w)),
                "workspace_title": titles.get(ws).cloned().unwrap_or(Value::Null),
            }));
        }
        workspaces.push(json!({
            "active": w.get("isActive").and_then(Value::as_bool).unwrap_or(false),
            "cwd_name": text(w.get("cwd").and_then(Value::as_str).map(folder_name)),
            "panes": ws_panes.len(),
            "shell": w
                .get("shell")
                .and_then(Value::as_str)
                .map(file_name)
                .filter(|name| is_ident(name)),
            "surfaces": leaves.len(),
            "title": text(w.get("title").and_then(Value::as_str)),
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
                "label": text(a.get("label").and_then(Value::as_str)),
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
/// (and the user name) above it. A drive-relative `C:name` gives `name`.
fn folder_name(path: &str) -> &str {
    let name = file_name(path.trim_end_matches(['\\', '/']));
    match name.split_once(':') {
        Some((_, rest)) if !rest.is_empty() => rest,
        _ => name,
    }
}

const KINDS: [&str; 4] = ["terminal", "browser", "markdown", "diff"];
const STATES: [&str; 3] = ["idle", "working", "blocked"];
const MAX_TEXT_CHARS: usize = 80;

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

/// A title, a label or a folder name is text the operator wrote. It leaves scrubbed like a
/// blocked reason and clipped: a filter masks what it can recognize, and cannot promise
/// that no short secret is left.
fn text(v: Option<&str>) -> Value {
    match v.map(|s| clip(&scrub(s), MAX_TEXT_CHARS)) {
        Some(s) if !s.is_empty() => Value::from(s),
        _ => Value::Null,
    }
}

/// A word shaped like a credential: a known token prefix, or a long run of letters and
/// digits in both cases.
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
            && w.bytes().any(|b| b.is_ascii_uppercase())
            && w.bytes().any(|b| b.is_ascii_lowercase())
            && w.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)))
}

/// `host:port`, such as `localhost:8`, `127.0.0.1:8080`, `[::1]:8080` or `devbox:3000`. A
/// time such as `10:30` has no letter or dot in its host, and a one-label host needs at least
/// two port digits, so `round:4` stays a word.
fn is_host_port(word: &str) -> bool {
    let w = word.trim_end_matches([',', '.', ')', ';']);
    let digits = |port: &str, fewest: usize| {
        (fewest..=5).contains(&port.len()) && port.bytes().all(|b| b.is_ascii_digit())
    };
    if let Some(rest) = w.strip_prefix('[') {
        return match rest.split_once("]:") {
            Some((addr, port)) => addr.contains(':') && digits(port, 1),
            None => false,
        };
    }
    match w.rsplit_once(':') {
        Some((host, port)) => {
            let named = host.eq_ignore_ascii_case("localhost") || host.contains('.');
            host.bytes().any(|b| b.is_ascii_alphabetic() || b == b'.')
                && host
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b".-".contains(&b))
                && digits(port, if named { 1 } else { 2 })
        }
        None => false,
    }
}

/// Free text the operator reads: URLs, addresses and paths become `<url>` and `<path>`,
/// credential-shaped words `***`, then [`mask_secrets`] masks named secrets. Words are split
/// on any whitespace and joined by one space.
fn scrub(text: &str) -> String {
    let words: Vec<&str> = text
        .split_whitespace()
        .map(|w| {
            if w.contains("://") || is_host_port(w) {
                "<url>"
            } else if w.contains(['\\', '/']) || is_drive_relative(w) {
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

/// `C:notes.txt`, a file relative to the current folder of drive C. Without a dot in the
/// name (`A:ready`) it reads as a label, not a path.
fn is_drive_relative(word: &str) -> bool {
    let w = word.trim_end_matches([',', '.', ')', ';']);
    let b = w.as_bytes();
    b.len() > 2
        && b[0].is_ascii_alphabetic()
        && b[1] == b':'
        && b[2] != b':'
        && w[2..].contains('.')
}

/// A key that names a secret: `token`, `secret`, `password`, `auth`… anywhere in it, or a
/// part `key`, `pass` or `pwd` (so `api_key` does, `keyboard` does not).
fn is_secret_key(key: &str) -> bool {
    const ANYWHERE: [&str; 7] = [
        "token",
        "secret",
        "passw",
        "passphrase",
        "apikey",
        "auth",
        "credential",
    ];
    let k = key.to_ascii_lowercase();
    ANYWHERE.iter().any(|m| k.contains(m))
        || k.split(|c: char| !c.is_ascii_alphanumeric())
            .any(|part| matches!(part, "key" | "pass" | "pwd"))
}

/// Masks the value of every `key=value`, `key: value` or `key value` whose key names a
/// secret, and the word after `Bearer`. A masked word that names a secret itself masks the
/// next one too, so `Authorization: Bearer x` and `Authorization:Bearer x` lose both.
fn mask_secrets(text: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut mask_next = false;
    let names_secret = |w: &str| {
        let bare = w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-');
        !bare.is_empty() && (is_secret_key(bare) || bare.eq_ignore_ascii_case("bearer"))
    };
    for word in text.split_whitespace() {
        if mask_next {
            out.push("***".into());
            mask_next = names_secret(word);
            continue;
        }
        match word.find(['=', ':']) {
            Some(at) if at > 0 && is_secret_key(&word[..at]) => {
                if at + 1 < word.len() {
                    out.push(format!("{}***", &word[..=at]));
                    mask_next = names_secret(&word[at + 1..]);
                } else {
                    out.push(word.into());
                    mask_next = true;
                }
            }
            _ => {
                out.push(word.into());
                mask_next = names_secret(word);
            }
        }
    }
    out.join(" ")
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

/// Runs `f` on its own thread and waits for it until the budget's deadline: looking wmux up
/// on PATH can block on an unreachable share.
fn within<T: Send + 'static>(
    budget: &Budget,
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T, String> {
    within_or(budget, "finding the wmux CLI", f, drop)
}

/// [`within`], and a value that comes after the deadline goes to `late`, so a CLI that
/// started too late is stopped. The slot's lock decides which side has it: the waiter
/// empties the slot before it gives up, and the thread sends only while it is full.
fn within_or<T: Send + 'static>(
    budget: &Budget,
    what: &str,
    f: impl FnOnce() -> T + Send + 'static,
    late: impl FnOnce(T) + Send + 'static,
) -> Result<T, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    let slot = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));
    let theirs = slot.clone();
    std::thread::spawn(move || {
        let value = f();
        let guard = theirs.lock().unwrap_or_else(|e| e.into_inner());
        let unsent = match guard.as_ref() {
            Some(tx) => tx.send(value).err().map(|e| e.0),
            None => Some(value),
        };
        drop(guard);
        if let Some(value) = unsent {
            late(value);
        }
    });
    match rx.recv_timeout(budget.deadline.saturating_duration_since(Instant::now())) {
        Ok(value) => Ok(value),
        Err(_) => {
            let mut guard = slot.lock().unwrap_or_else(|e| e.into_inner());
            guard.take();
            rx.try_recv()
                .map_err(|_| format!("{what}: the snapshot's time budget ran out"))
        }
    }
}

type Chunk = (bool, Option<Vec<u8>>);

/// A channel that holds a few chunks: a reader that gets ahead of the call waits for it.
fn chunk_channel() -> (
    std::sync::mpsc::SyncSender<Chunk>,
    std::sync::mpsc::Receiver<Chunk>,
) {
    std::sync::mpsc::sync_channel(4)
}

/// Kills the CLI and reaps it. A kill that fails leaves the process to the system rather
/// than wait for it past the deadline.
fn stop(child: &mut std::process::Child) {
    if child.kill().is_ok() {
        let _ = child.wait();
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
    let lower = launch.program.to_ascii_lowercase();
    if (lower.ends_with(".cmd") || lower.ends_with(".bat")) && launch.program.contains('%') {
        return Err(format!(
            "wmux {verb}: the path of the {name} script holds a %, which cmd.exe would expand; set IX_WMUX_BIN to a path without one"
        ));
    }
    let mut command = Command::new(&launch.program);
    if launch.electron {
        command.env("ELECTRON_RUN_AS_NODE", "1");
    }
    command
        .args(&launch.lead)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = within_or(
        budget,
        &format!("starting wmux {verb}"),
        move || command.spawn(),
        |started| {
            if let Ok(mut child) = started {
                stop(&mut child);
            }
        },
    )?
    .map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => {
            format!("wmux CLI not found ({name}); run inside wmux or set IX_WMUX_BIN")
        }
        _ => format!("could not start wmux ({name}): {}", e.kind()),
    })?;
    let out_pipe = child.stdout.take().ok_or("no stdout for wmux")?;
    let err_pipe = child.stderr.take().ok_or("no stderr for wmux")?;
    // The readers send chunks as they read, and the caps are enforced here as the chunks
    // arrive. When the call returns, the receiver is gone and each reader stops at its next
    // chunk. A process the CLI started can still hold a pipe open: its reader then waits for
    // it with one 64 KiB buffer, and the call does not.
    let (tx, rx) = chunk_channel();
    pump(out_pipe, true, tx.clone());
    pump(err_pipe, false, tx);
    let stopped = |what: &str| {
        format!(
            "wmux {verb}: {what} within the snapshot's {} s budget, and was stopped",
            TIMEOUT.as_secs()
        )
    };
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let (mut out_done, mut err_done) = (false, false);
    let mut status = None;
    loop {
        if status.is_none() {
            status = child.try_wait().map_err(|e| e.kind().to_string())?;
        }
        if status.is_some() && out_done && err_done {
            break;
        }
        let now = Instant::now();
        if now >= budget.deadline {
            if status.is_none() {
                stop(&mut child);
                return Err(stopped("did not finish"));
            }
            return Err(stopped("exited, but its output stayed open"));
        }
        let wait = (budget.deadline - now).min(Duration::from_millis(10));
        match rx.recv_timeout(wait) {
            Ok((true, Some(chunk))) => {
                out.extend_from_slice(&chunk);
                if out.len() as u64 > budget.bytes_left {
                    stop(&mut child);
                    return Err(format!(
                        "wmux {verb}: the snapshot printed more than {MAX_OUTPUT_BYTES} bytes in all"
                    ));
                }
            }
            Ok((false, Some(chunk))) => {
                let room = MAX_STDERR_BYTES.saturating_sub(err.len());
                err.extend_from_slice(&chunk[..chunk.len().min(room)]);
            }
            Ok((true, None)) => out_done = true,
            Ok((false, None)) => err_done = true,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                (out_done, err_done) = (true, true);
            }
        }
    }
    let status = status.expect("the loop ends only after the CLI exited");
    if !status.success() {
        let msg = String::from_utf8_lossy(&err);
        return Err(format!("wmux {verb}: {}", clip(&scrub(msg.trim()), 500)));
    }
    budget.bytes_left -= out.len() as u64;
    serde_json::from_slice(&out).map_err(|e| format!("wmux {verb}: not JSON: {e}"))
}

const MAX_STDERR_BYTES: usize = 64 * 1024;

/// Reads `pipe` on its own thread and sends `(is_stdout, Some(chunk))` for each chunk, then
/// `(is_stdout, None)` at the end. It stops as soon as nobody receives.
fn pump(
    mut pipe: impl Read + Send + 'static,
    is_stdout: bool,
    tx: std::sync::mpsc::SyncSender<Chunk>,
) {
    std::thread::spawn(move || {
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match pipe.read(&mut buf) {
                Ok(0) | Err(_) => {
                    let _ = tx.send((is_stdout, None));
                    return;
                }
                Ok(n) => {
                    if tx.send((is_stdout, Some(buf[..n].to_vec()))).is_err() {
                        return;
                    }
                }
            }
        }
    });
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

    /// A field name on the allowlist is not enough. Ids, states and numbers leave only in
    /// their own shape; titles, labels and folder names are text the operator wrote, which
    /// leaves with its URLs, paths and secrets masked, as well as a filter can tell.
    #[test]
    fn free_text_leaves_with_its_urls_paths_and_secrets_masked() {
        let sidebar = json!({ "workspaces": [{
            "id": "w1", "title": "curl https://internal/?auth=ghp_x", "isActive": true,
            "cwd": "C:\\work\\ghp_0123456789abcdefABCDEF0123", "shell": "pwsh.exe -Command x"
        }]});
        let tree = json!({ "tree": { "type": "leaf", "paneId": "p 1", "surfaces": [
            { "id": "s1", "type": "terminal", "customTitle": "C:\\Users\\alice\\notes",
              "cwd": "C:\\work\\proj", "claudeSessionId": "id; rm -rf /" },
            { "id": "s2", "type": "<script>", "title": "Claude Code" },
            { "id": "s3", "type": "browser", "title": "[Gray] Travail | SONDES" },
            { "id": "s4", "type": "markdown", "title": "localhost:3000 keyboard layout at 10:30" },
            { "id": "s5", "type": "terminal", "cwd": "C:private",
              "title": "gaia-site-publish-tts-kernel-20261010" }
        ]}});
        let states = json!({ "states": [
            { "surfaceId": "s1", "state": "rm -rf", "updatedAt": "soon" }
        ]});
        let agents = json!({ "agents": [{
            "agentId": "a1", "label": "sk-live-0123456789abcdef", "status": "running now",
            "pid": "12", "surfaceId": "s1", "workspaceId": "w1"
        }]});
        let snap = normalize(&sidebar, &[("w1".into(), tree)], &states, &agents);
        let text = snap.to_string();
        for leak in [
            "alice",
            "ghp_",
            "sk-live",
            "internal",
            "localhost",
            "-Command",
        ] {
            assert!(!text.contains(leak), "{leak} leaked");
        }
        let ws = &snap["workspaces"][0];
        assert_eq!(ws["title"], "curl <url>");
        assert_eq!(ws["cwd_name"], "***");
        assert_eq!(ws["shell"], Value::Null);
        let s = |i: usize| &snap["surfaces"][i];
        assert_eq!(s(0)["title"], "<path>");
        assert_eq!(s(0)["workspace_title"], "curl <url>");
        assert_eq!(s(0)["pane_id"], Value::Null);
        assert_eq!(s(0)["claude_session_id"], Value::Null);
        assert_eq!(s(0)["agent_state"], Value::Null);
        assert_eq!(s(0)["updated_at_ms"], Value::Null);
        assert_eq!(s(0)["cwd_name"], "proj");
        assert_eq!(s(1)["type"], "other");
        assert_eq!(s(1)["title"], "Claude Code");
        assert_eq!(s(2)["title"], "[Gray] Travail | SONDES");
        assert_eq!(s(3)["title"], "<url> keyboard layout at 10:30");
        assert_eq!(s(4)["title"], "gaia-site-publish-tts-kernel-20261010");
        assert_eq!(s(4)["cwd_name"], "private");
        assert_eq!(
            snap["counts"]["by_type"],
            json!({ "browser": 1, "markdown": 1, "other": 1, "terminal": 2 })
        );
        assert_eq!(snap["counts"]["by_state"], json!({}));
        let a = &snap["agents"][0];
        assert_eq!(a["label"], "***");
        assert_eq!(a["status"], Value::Null);
        assert_eq!(a["pid"], Value::Null);
        assert_eq!(a["agent_id"], "a1");
    }

    #[test]
    fn a_blocked_reason_loses_its_urls_paths_and_bearer_secrets() {
        let reason = "Fetch https://host/x?token=abc with Authorization: Bearer sk-live-1 \
                      from C:\\Users\\alice\\.ssh\\id_rsa using password hunter2 and \
                      ghp_0123456789abcdefABCDEF0123 then Authorization: Bearer hunter3 \
                      token\tabc4 password\nhunter5 on 127.0.0.1:8080";
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
            "hunter3",
            "abc4",
            "hunter5",
            "127.0.0.1",
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
        let err = run_json(&wrapper, &["sidebar-state"], &mut budget).unwrap_err();
        assert!(err.contains("stayed open"), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn the_output_budget_is_shared_by_the_whole_call() {
        let mut budget = Budget::new(TIMEOUT, 10);
        let small = script("'[1,2]'", "echo '[1,2]'");
        assert_eq!(
            run_json(&small, &["sidebar-state"], &mut budget).unwrap(),
            json!([1, 2])
        );
        let err = run_json(&small, &["agent-state"], &mut budget).unwrap_err();
        assert!(err.contains("more than"), "{err}");
        let chatty = script("'x' * 100000", "printf '%0100000d' 0");
        let err = run_json(&chatty, &["sidebar-state"], &mut fresh_with(1000)).unwrap_err();
        assert!(err.contains("more than"), "{err}");
    }

    fn fresh_with(bytes: u64) -> Budget {
        Budget::new(TIMEOUT, bytes)
    }

    /// `cmd.exe` expands `%NAME%` in the path of the script it runs, so such a path is
    /// refused before anything starts.
    #[test]
    fn a_batch_script_path_with_a_percent_is_refused() {
        for path in [r"C:\tools\%X%\wmux.cmd", r"C:\tools\%X%\wmux.BAT"] {
            let launch = for_program(path.to_string());
            let err = run_json(&launch, &["sidebar-state"], &mut fresh()).unwrap_err();
            assert!(err.contains("%"), "{err}");
            assert!(!err.contains("tools"), "{err}");
        }
    }

    /// A secret-naming value glued to its key keeps the chain going; a drive-relative file,
    /// a one-digit port and a bracketed IPv6 address are masked like the others.
    #[test]
    fn attached_secrets_drive_paths_and_short_or_ipv6_ports_are_masked() {
        let s = scrub(
            "Authorization:Bearer hunter3 token=password hunter2 C:private.txt localhost:8 \
             [::1]:8080 at 10:30",
        );
        for leak in ["hunter3", "hunter2", "private", "localhost", "::1"] {
            assert!(!s.contains(leak), "{leak} in {s}");
        }
        assert!(s.ends_with("at 10:30"), "{s}");
    }

    /// Colon-joined words that are not addresses or files keep their text, or a title made of
    /// them would leave as nothing but masks.
    #[test]
    fn ordinary_colon_words_are_left_alone() {
        let text = "round:4 A:ready [status]:blocked step:x note:done";
        assert_eq!(scrub(text), text);
    }

    /// A value that arrives after the deadline goes to `late`, so a CLI that started too
    /// late is stopped rather than left running.
    #[test]
    fn a_late_start_is_cleaned_up() {
        let budget = Budget::new(Duration::from_millis(200), MAX_OUTPUT_BYTES);
        let (tx, rx) = std::sync::mpsc::channel();
        let err = within_or(
            &budget,
            "starting wmux",
            || {
                std::thread::sleep(Duration::from_millis(600));
                7
            },
            move |v| {
                let _ = tx.send(v);
            },
        )
        .unwrap_err();
        assert!(err.contains("starting wmux"), "{err}");
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)), Ok(7));
    }

    /// The readers hold at most a few chunks nobody received, so a CLI that prints faster
    /// than the call reads cannot fill memory past the caps.
    #[test]
    fn a_reader_nobody_drains_stops_after_a_few_chunks() {
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;
        struct Endless(Arc<AtomicU64>);
        impl Read for Endless {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                let before = self.0.fetch_add(buf.len() as u64, Ordering::SeqCst);
                Ok(if before > 64 << 20 { 0 } else { buf.len() })
            }
        }
        let read = Arc::new(AtomicU64::new(0));
        let (tx, rx) = chunk_channel();
        pump(Endless(read.clone()), true, tx);
        std::thread::sleep(Duration::from_millis(300));
        let n = read.load(Ordering::SeqCst);
        drop(rx);
        assert!(n <= 1 << 20, "{n} bytes read ahead");
    }

    /// Looking wmux up on PATH can block on an unreachable share; it runs inside the budget.
    #[test]
    fn a_slow_lookup_is_bounded_by_the_budget() {
        let budget = Budget::new(Duration::from_secs(1), MAX_OUTPUT_BYTES);
        let started = Instant::now();
        let err = within(&budget, || std::thread::sleep(Duration::from_secs(5))).unwrap_err();
        assert!(err.contains("ran out"), "{err}");
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{:?}",
            started.elapsed()
        );
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
