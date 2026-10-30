//! The hooks check: the harness's command hooks are present, parse, and their fixtures pass.
//!
//! Claude Code runs the hooks in `.claude/settings.json` through bash on every platform, msys
//! bash on Windows. A hook that is missing, that does not parse, or that crashes is a silent
//! no-op: the harness reports nothing and the hook simply never fires. The only trace of such a
//! crash was a `bash.exe.stackdump` in the working tree, and a correction detector that never
//! nudged went unnoticed for months (it matched English openers only; see the fixture
//! `.claude/hooks/test-correction-detector.sh`).
//!
//! Three things are checked, in this order, each cheap and each derived from the tree:
//!
//! 1. every hook file a `command` entry names under `.claude/hooks/` exists;
//! 2. every `.claude/hooks/*.sh` parses (`bash -n`);
//! 3. every behavioural fixture `.claude/hooks/test-*.sh` exits 0, with its last lines kept.
//!
//! Hooks are not run with a synthetic payload: several of them write under `state/digests/` or
//! spawn cargo, and a gate must not leave footprints. The fixture is where a hook's behaviour is
//! exercised; a hook without one is listed (`unfixtured`) and never fails the check.
//!
//! Without bash the check warns rather than passes: an environment that cannot run the hooks
//! cannot vouch for them either.

use super::{CheckResult, Status};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Where the harness's hook configuration lives, relative to the repo root.
pub const SETTINGS_RELPATH: &str = ".claude/settings.json";
/// Where the hook scripts and their fixtures live, relative to the repo root.
pub const HOOKS_RELDIR: &str = ".claude/hooks";
/// A fixture that runs longer than this is killed and reported as timed out.
const FIXTURE_TIMEOUT: Duration = Duration::from_secs(120);
/// How many trailing lines of a fixture's output ride in the report.
const TAIL_LINES: usize = 12;

/// One fixture's run.
#[derive(Debug, Clone, Serialize)]
pub struct FixtureRun {
    /// File name under `.claude/hooks/`.
    pub name: String,
    /// Exit code, `None` when killed by a signal or by the timeout.
    pub exit: Option<i32>,
    /// Killed after [`FIXTURE_TIMEOUT`].
    pub timed_out: bool,
    /// The last lines of stdout and stderr, for the report.
    pub tail: String,
}

impl FixtureRun {
    fn passed(&self) -> bool {
        self.exit == Some(0) && !self.timed_out
    }
}

/// What the scan found.
#[derive(Debug, Default, Serialize)]
pub struct Census {
    /// `.claude/settings.json` exists.
    pub settings_present: bool,
    /// Hook files named by `command` entries, repo-relative, deduplicated, sorted.
    pub wired: Vec<String>,
    /// Wired hook files that do not exist.
    pub missing: Vec<String>,
    /// Every `.claude/hooks/*.sh`, file names, sorted.
    pub scripts: Vec<String>,
    /// Scripts `bash -n` rejects, with bash's message.
    pub syntax_errors: Vec<(String, String)>,
    /// Every `.claude/hooks/test-*.sh`, run.
    pub fixtures: Vec<FixtureRun>,
    /// Wired hooks with no `test-<name>.sh` beside them. Informational.
    pub unfixtured: Vec<String>,
    /// The bash used, when one was found.
    pub bash: Option<String>,
}

/// The bash the hooks run under.
///
/// On Windows that is Git for Windows' bash, the one Claude Code's `bash` hooks resolve to;
/// `System32\bash.exe` is the WSL launcher, which does not take Windows paths, so it is never
/// used here. Elsewhere, `bash` on PATH.
pub fn find_bash() -> Option<PathBuf> {
    if cfg!(windows) {
        let mut candidates: Vec<PathBuf> = Vec::new();
        for var in ["ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"] {
            if let Ok(pf) = std::env::var(var) {
                candidates.push(Path::new(&pf).join("Git/bin/bash.exe"));
                candidates.push(Path::new(&pf).join("Git/usr/bin/bash.exe"));
            }
        }
        candidates.push(PathBuf::from(r"C:\Program Files\Git\bin\bash.exe"));
        candidates.into_iter().find(|c| c.is_file())
    } else {
        Command::new("bash")
            .args(["-c", "exit 0"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .ok()
            .filter(|st| st.success())
            .map(|_| PathBuf::from("bash"))
    }
}

/// Scan `root`. `bash` is the interpreter to parse and run with; `None` skips steps 2 and 3.
pub fn scan(root: &Path, bash: Option<&Path>) -> Result<Census, String> {
    let mut census = Census {
        bash: bash.map(|b| b.display().to_string()),
        ..Census::default()
    };

    let settings_path = root.join(SETTINGS_RELPATH);
    if !settings_path.is_file() {
        return Ok(census);
    }
    census.settings_present = true;
    let text = std::fs::read_to_string(&settings_path)
        .map_err(|e| format!("reading {SETTINGS_RELPATH}: {e}"))?;
    let settings: Value =
        serde_json::from_str(&text).map_err(|e| format!("parsing {SETTINGS_RELPATH}: {e}"))?;

    census.wired = wired_hook_files(&settings);
    census.missing = census
        .wired
        .iter()
        .filter(|rel| !root.join(rel).is_file())
        .cloned()
        .collect();

    let hooks_dir = root.join(HOOKS_RELDIR);
    if hooks_dir.is_dir() {
        let mut names: Vec<String> = std::fs::read_dir(&hooks_dir)
            .map_err(|e| format!("reading {HOOKS_RELDIR}: {e}"))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_file())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".sh"))
            .collect();
        names.sort();
        census.scripts = names;
    }

    census.unfixtured = census
        .wired
        .iter()
        .filter_map(|rel| rel.rsplit('/').next())
        .filter(|name| !census.scripts.iter().any(|s| s == &format!("test-{name}")))
        .map(str::to_string)
        .collect();

    let Some(bash) = bash else {
        return Ok(census);
    };

    for name in &census.scripts {
        let path = hooks_dir.join(name);
        let output = Command::new(bash)
            .arg("-n")
            .arg(posix(&path))
            .current_dir(root)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("running `bash -n` on {name}: {e}"))?;
        if !output.status.success() {
            let msg = String::from_utf8_lossy(&output.stderr).trim().to_string();
            census.syntax_errors.push((name.clone(), msg));
        }
    }

    for name in census
        .scripts
        .iter()
        .filter(|n| n.starts_with("test-"))
        .cloned()
        .collect::<Vec<_>>()
    {
        let run = run_fixture(bash, root, &hooks_dir.join(&name), &name)?;
        census.fixtures.push(run);
    }

    Ok(census)
}

/// Hook files named by `command` entries, repo-relative (`.claude/hooks/x.sh`).
///
/// A command reads like `bash $CLAUDE_PROJECT_DIR/.claude/hooks/x.sh`; whatever the prefix, the
/// token that contains `.claude/hooks/` names the file. Entries of another type (`prompt`,
/// `agent`) have no file and are skipped.
fn wired_hook_files(settings: &Value) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    let Some(events) = settings.get("hooks").and_then(Value::as_object) else {
        return files;
    };
    for groups in events.values() {
        let Some(groups) = groups.as_array() else {
            continue;
        };
        for group in groups {
            let Some(entries) = group.get("hooks").and_then(Value::as_array) else {
                continue;
            };
            for entry in entries {
                let is_command = matches!(
                    entry.get("type").and_then(Value::as_str),
                    None | Some("command")
                );
                let Some(command) = entry.get("command").and_then(Value::as_str) else {
                    continue;
                };
                if !is_command {
                    continue;
                }
                for token in command.split_whitespace() {
                    let token = token.trim_matches(|c| c == '"' || c == '\'');
                    if let Some(idx) = token.find(".claude/hooks/") {
                        files.push(token[idx..].to_string());
                    }
                }
            }
        }
    }
    files.sort();
    files.dedup();
    files
}

/// Run one fixture under `bash`, from `root`, with `CLAUDE_PROJECT_DIR` set as the harness
/// sets it, killed after [`FIXTURE_TIMEOUT`].
fn run_fixture(bash: &Path, root: &Path, path: &Path, name: &str) -> Result<FixtureRun, String> {
    let mut child = Command::new(bash)
        .arg(posix(path))
        .current_dir(root)
        .env("CLAUDE_PROJECT_DIR", root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("running fixture {name}: {e}"))?;

    let stdout = child.stdout.take().map(drain);
    let stderr = child.stderr.take().map(drain);

    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(st) = child
            .try_wait()
            .map_err(|e| format!("waiting for fixture {name}: {e}"))?
        {
            break Some(st);
        }
        if started.elapsed() > FIXTURE_TIMEOUT {
            timed_out = true;
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(25));
    };

    let mut text = String::new();
    for reader in [stdout, stderr].into_iter().flatten() {
        text.push_str(&reader.join().unwrap_or_default());
    }
    let lines: Vec<&str> = text.lines().collect();
    let tail = lines[lines.len().saturating_sub(TAIL_LINES)..].join("\n");

    Ok(FixtureRun {
        name: name.to_string(),
        exit: status.and_then(|st| st.code()),
        timed_out,
        tail,
    })
}

/// Read a child's pipe to the end on its own thread, so a chatty fixture cannot block on a
/// full pipe while the parent waits for it.
fn drain<R: Read + Send + 'static>(mut pipe: R) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = pipe.read_to_string(&mut s);
        s
    })
}

/// A path as msys bash takes it: forward slashes, drive letter kept, and no `\\?\` verbatim
/// prefix. A canonicalized Windows path carries that prefix; forward-slashed it reads as the
/// UNC share `//?/`, which bash then tries to resolve as a host and hangs on (the first run of
/// the fixtures sat 120 s per fixture against the timeout, and a `dirname "$0"` inside one
/// fixture never came back).
fn posix(path: &Path) -> String {
    let text = path.display().to_string();
    let text = text
        .strip_prefix(r"\\?\UNC\")
        .map(|rest| format!(r"\\{rest}"))
        .or_else(|| text.strip_prefix(r"\\?\").map(str::to_string))
        .unwrap_or(text);
    text.replace('\\', "/")
}

/// The check: scan, then say what to do.
pub fn check(root: &Path, bash: Option<&Path>) -> CheckResult {
    const NAME: &str = "hooks";
    let census = match scan(root, bash) {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::new(NAME, Status::Fail, format!("could not scan hooks: {e}"))
                .with_remedy(
                    "this check reads .claude/settings.json and .claude/hooks/ — a read failure \
                     means the scan did not happen, so it is a failure rather than a pass",
                )
        }
    };

    let passed = census.fixtures.iter().filter(|f| f.passed()).count();
    let details = json!({
        "settings_present": census.settings_present,
        "hooks_wired": census.wired.len(),
        "wired": census.wired,
        "missing": census.missing,
        "scripts": census.scripts,
        "syntax_errors": census.syntax_errors,
        "fixtures": census.fixtures,
        "fixtures_passed": passed,
        "unfixtured": census.unfixtured,
        "bash": census.bash,
    });

    if !census.settings_present {
        return CheckResult::new(
            NAME,
            Status::Ok,
            format!("no {SETTINGS_RELPATH}: no hooks to check"),
        )
        .with_details(details);
    }

    if !census.missing.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!(
                "hook(s) wired in {SETTINGS_RELPATH} but absent: {}",
                census.missing.join(", ")
            ),
        )
        .with_remedy(
            "restore the file or drop the entry — a hook the harness cannot find fires nothing \
             and reports nothing",
        )
        .with_details(details);
    }

    if census.bash.is_none() {
        return CheckResult::new(
            NAME,
            Status::Warn,
            format!(
                "{} hook(s) wired and present; bash not found, so syntax and the {} fixture(s) \
                 were not exercised",
                census.wired.len(),
                census
                    .scripts
                    .iter()
                    .filter(|n| n.starts_with("test-"))
                    .count()
            ),
        )
        .with_remedy(
            "install Git for Windows (its bash under Program Files\\Git\\bin is the one the \
             hooks run under; the WSL launcher in System32 is not) or put bash on PATH, then \
             re-run — a green doctor without this step has not looked at the hooks",
        )
        .with_details(details);
    }

    if !census.syntax_errors.is_empty() {
        let listed = census
            .syntax_errors
            .iter()
            .map(|(name, msg)| format!("{name} ({})", msg.lines().last().unwrap_or("")))
            .collect::<Vec<_>>()
            .join("; ");
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!("hook script(s) that do not parse as bash: {listed}"),
        )
        .with_remedy(
            "`bash -n .claude/hooks/<file>` shows the error — a hook that does not parse never \
             fires, and the harness does not say so",
        )
        .with_details(details);
    }

    let failed: Vec<String> = census
        .fixtures
        .iter()
        .filter(|f| !f.passed())
        .map(|f| match (f.timed_out, f.exit) {
            (true, _) => format!("{} (timed out)", f.name),
            (false, Some(code)) => format!("{} (exit {code})", f.name),
            (false, None) => format!("{} (killed)", f.name),
        })
        .collect();
    if !failed.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!("hook fixture(s) failing: {}", failed.join(", ")),
        )
        .with_remedy(
            "run `bash .claude/hooks/<fixture>` for the full output (the last lines ride in \
             details.fixtures[].tail) — the fixture is the hook's behavioural binding, so a red \
             one means the hook does not do what its comment says",
        )
        .with_details(details);
    }

    CheckResult::new(
        NAME,
        Status::Ok,
        format!(
            "{} hook(s) wired, all present and parsing; {} fixture(s) passed ({} hook(s) without \
             a fixture)",
            census.wired.len(),
            passed,
            census.unfixtured.len()
        ),
    )
    .with_details(details)
}
