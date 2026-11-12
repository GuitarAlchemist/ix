//! The hooks check: the harness's command hooks are present, run no shell, and their Go tests pass.
//!
//! Until 2026-11 the hooks were bash scripts under `.claude/hooks/`, run through msys bash on
//! Windows. A hook that crashed or failed to parse was a silent no-op (the only trace was a
//! `bash.exe.stackdump` in the working tree), each one cost seconds per tool call, and a JSON
//! parser that resolved to the WindowsApps `python3` alias quietly stopped the rules. The user's
//! rule since 2026-11-05 is "jamais de shell dans les hooks": every hook is a subcommand of the
//! Go program `go/claudehooks` (`ix-hooks`), installed into the gitignored `.claude/hooks/bin/` by
//! the `session-start` hook, which itself runs through `go run`.
//!
//! Four things are checked, in this order, each derived from the tree:
//!
//! 1. every file a `command` entry names under `.claude/hooks/` exists (the built
//!    `.claude/hooks/bin/` is skipped: it is not in the repository);
//! 2. no `command` runs a shell or a script interpreter (bash, sh, PowerShell, cmd, Python), and
//!    no script (`.sh`, `.ps1`, `.py`, `.cmd`, `.bat`) lies in `.claude/hooks/`;
//! 3. when a hook runs `ix-hooks`, `go/claudehooks` exists;
//! 4. `go test ./...` passes in `go/claudehooks`: those tests are the hooks' behavioural binding.
//!
//! Hooks are not run with a synthetic payload: several of them write under `state/digests/` or
//! spawn cargo, and a gate must not leave footprints. Without Go the check warns rather than
//! passes: an environment that cannot build the hooks cannot vouch for them either.

use super::{CheckResult, Status};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Where the harness's hook configuration lives, relative to the repo root.
pub const SETTINGS_RELPATH: &str = ".claude/settings.json";
/// Where hook files live, relative to the repo root.
pub const HOOKS_RELDIR: &str = ".claude/hooks";
/// Where `session-start` installs the built hooks; gitignored, so never "missing".
pub const HOOKS_BIN_RELDIR: &str = ".claude/hooks/bin/";
/// The Go module every hook is a subcommand of.
pub const GO_MODULE_RELDIR: &str = "go/claudehooks";
/// `go test` running longer than this is killed and reported as timed out.
const TEST_TIMEOUT: Duration = Duration::from_secs(300);
/// How many trailing lines of the test output ride in the report.
const TAIL_LINES: usize = 12;
/// Programs a hook command must not start with.
const INTERPRETERS: &[&str] = &[
    "bash",
    "sh",
    "zsh",
    "pwsh",
    "powershell",
    "cmd",
    "python",
    "python3",
    "py",
];
/// Script extensions that make a hook a shell or interpreter hook.
const SCRIPT_EXTENSIONS: &[&str] = &[".sh", ".ps1", ".py", ".cmd", ".bat"];

/// The run of the hooks' Go tests.
#[derive(Debug, Clone, Serialize)]
pub struct TestRun {
    /// Exit code, `None` when killed by a signal or by the timeout.
    pub exit: Option<i32>,
    /// Killed after [`TEST_TIMEOUT`].
    pub timed_out: bool,
    /// The last lines of stdout and stderr, for the report.
    pub tail: String,
}

impl TestRun {
    fn passed(&self) -> bool {
        self.exit == Some(0) && !self.timed_out
    }
}

/// What the scan found.
#[derive(Debug, Default, Serialize)]
pub struct Census {
    /// `.claude/settings.json` exists.
    pub settings_present: bool,
    /// Every `command` entry, as written.
    pub commands: Vec<String>,
    /// Hook files named by `command` entries outside the built bin directory, repo-relative,
    /// deduplicated, sorted.
    pub wired: Vec<String>,
    /// Wired hook files that do not exist.
    pub missing: Vec<String>,
    /// Commands that run a shell or a script interpreter.
    pub shell_wired: Vec<String>,
    /// Scripts lying in `.claude/hooks/`, file names, sorted.
    pub scripts: Vec<String>,
    /// Some command runs `ix-hooks`.
    pub uses_ix_hooks: bool,
    /// `go/claudehooks/go.mod` exists.
    pub module_present: bool,
    /// `go test ./...` in the module, when Go was found.
    pub go_tests: Option<TestRun>,
    /// The Go used, when one was found.
    pub go: Option<String>,
}

/// The Go toolchain on PATH, if `go version` runs.
pub fn find_go() -> Option<PathBuf> {
    Command::new("go")
        .arg("version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()
        .filter(|st| st.success())
        .map(|_| PathBuf::from("go"))
}

/// Whether a hook command runs a shell or a script interpreter: its program is one of
/// [`INTERPRETERS`], directly or through `env`, or a word names a script by its extension.
/// Only the program counts: a shell's name as an ordinary argument (`node hook.js bash`) does not.
pub fn runs_shell(command: &str) -> bool {
    let words = split_words(command);
    program_is_interpreter(&words)
        || words.iter().any(|w| {
            let w = w.to_ascii_lowercase();
            SCRIPT_EXTENSIONS.iter().any(|ext| w.ends_with(ext))
        })
}

/// The words of a command line, with `'…'` and `"…"` quoting (no escapes), quotes removed: a
/// quoted program may hold spaces (`'C:/Program Files/Git/bin/bash.exe'`).
fn split_words(command: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut word = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    for c in command.chars() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => word.push(c),
            None if c == '"' || c == '\'' => {
                quote = Some(c);
                in_word = true;
            }
            None if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut word));
                    in_word = false;
                }
            }
            None => {
                word.push(c);
                in_word = true;
            }
        }
    }
    if in_word {
        words.push(word);
    }
    words
}

/// Whether `words[0]` is an interpreter, or `env` whose program is one. `env`'s options and
/// `NAME=value` assignments are skipped (`-u NAME` and `-C DIR` take an argument), and
/// `-S` / `--split-string` hands a whole command line, checked as one.
fn program_is_interpreter(words: &[String]) -> bool {
    let Some(program) = words.first() else {
        return false;
    };
    let base = program
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(program)
        .to_ascii_lowercase();
    let base = base.strip_suffix(".exe").unwrap_or(&base);
    if INTERPRETERS.contains(&base) {
        return true;
    }
    if base != "env" {
        return false;
    }
    let mut rest = &words[1..];
    while let Some(word) = rest.first() {
        if word == "-S" || word == "--split-string" {
            return rest.get(1).is_some_and(|line| runs_shell(line));
        }
        if let Some(line) = word.strip_prefix("--split-string=") {
            return runs_shell(line);
        }
        if let Some(line) = word.strip_prefix("-S") {
            if !line.is_empty() && !word.starts_with("--") {
                return runs_shell(line);
            }
        }
        rest = match word.as_str() {
            "-u" | "--unset" | "-C" | "--chdir" => rest.get(2..).unwrap_or(&[]),
            w if w.starts_with('-') || w.contains('=') => &rest[1..],
            _ => break,
        };
    }
    program_is_interpreter(rest)
}

/// Scan `root`. `go` runs the hooks' tests; `None` skips step 4.
pub fn scan(root: &Path, go: Option<&Path>) -> Result<Census, String> {
    let mut census = Census {
        go: go.map(|g| g.display().to_string()),
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

    census.commands = hook_commands(&settings);
    census.wired = wired_hook_files(&census.commands);
    census.missing = census
        .wired
        .iter()
        .filter(|rel| !root.join(rel).is_file())
        .cloned()
        .collect();
    census.shell_wired = census
        .commands
        .iter()
        .filter(|c| runs_shell(c))
        .cloned()
        .collect();
    census.uses_ix_hooks = census.commands.iter().any(|c| c.contains("ix-hooks"));
    census.module_present = root.join(GO_MODULE_RELDIR).join("go.mod").is_file();

    let hooks_dir = root.join(HOOKS_RELDIR);
    if hooks_dir.is_dir() {
        let mut names: Vec<String> = std::fs::read_dir(&hooks_dir)
            .map_err(|e| format!("reading {HOOKS_RELDIR}: {e}"))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_file())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| {
                let lower = name.to_ascii_lowercase();
                SCRIPT_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
            })
            .collect();
        names.sort();
        census.scripts = names;
    }

    if let (Some(go), true) = (go, census.module_present) {
        let mut cmd = Command::new(go);
        cmd.args(["test", "-count=1", "./..."])
            .current_dir(root.join(GO_MODULE_RELDIR));
        census.go_tests = Some(run_bounded(cmd, "go test")?);
    }

    Ok(census)
}

/// Every `command` entry of type `command` (entries of another type, `prompt` or `agent`, run
/// nothing).
fn hook_commands(settings: &Value) -> Vec<String> {
    let mut commands = Vec::new();
    let Some(events) = settings.get("hooks").and_then(Value::as_object) else {
        return commands;
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
                if let (true, Some(command)) =
                    (is_command, entry.get("command").and_then(Value::as_str))
                {
                    commands.push(command.to_string());
                }
            }
        }
    }
    commands
}

/// Hook files named by commands, repo-relative (`.claude/hooks/x`): whatever the prefix, the
/// word that contains `.claude/hooks/` names the file. The built bin directory is skipped.
fn wired_hook_files(commands: &[String]) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for command in commands {
        for token in command.split_whitespace() {
            let token = token.trim_matches(|c| c == '"' || c == '\'');
            if let Some(idx) = token.find(".claude/hooks/") {
                let rel = &token[idx..];
                if !rel.starts_with(HOOKS_BIN_RELDIR) {
                    files.push(rel.to_string());
                }
            }
        }
    }
    files.sort();
    files.dedup();
    files
}

/// Run `cmd` with its output captured, killed after [`TEST_TIMEOUT`].
fn run_bounded(mut cmd: Command, what: &str) -> Result<TestRun, String> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("running {what}: {e}"))?;

    let stdout = child.stdout.take().map(drain);
    let stderr = child.stderr.take().map(drain);

    let started = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(st) = child
            .try_wait()
            .map_err(|e| format!("waiting for {what}: {e}"))?
        {
            break Some(st);
        }
        if started.elapsed() > TEST_TIMEOUT {
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

    Ok(TestRun {
        exit: status.and_then(|st| st.code()),
        timed_out,
        tail,
    })
}

/// Read a child's pipe to the end on its own thread, so a chatty child cannot block on a full
/// pipe while the parent waits for it.
fn drain<R: Read + Send + 'static>(mut pipe: R) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut s = String::new();
        let _ = pipe.read_to_string(&mut s);
        s
    })
}

/// The check: scan, then say what to do.
pub fn check(root: &Path, go: Option<&Path>) -> CheckResult {
    const NAME: &str = "hooks";
    let census = match scan(root, go) {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::new(NAME, Status::Fail, format!("could not scan hooks: {e}"))
                .with_remedy(
                    "this check reads .claude/settings.json and .claude/hooks/ — a read failure \
                     means the scan did not happen, so it is a failure rather than a pass",
                )
        }
    };

    let details = json!({
        "settings_present": census.settings_present,
        "hooks_wired": census.commands.len(),
        "wired": census.wired,
        "missing": census.missing,
        "shell_wired": census.shell_wired,
        "scripts": census.scripts,
        "uses_ix_hooks": census.uses_ix_hooks,
        "module_present": census.module_present,
        "go_tests": census.go_tests,
        "go": census.go,
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

    if !census.shell_wired.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!(
                "hook(s) run through a shell or a script interpreter: {}",
                census.shell_wired.join(" | ")
            ),
        )
        .with_remedy(
            "port the hook to a subcommand of go/claudehooks (ix-hooks <name>) with its test — \
             no hook runs a shell, PowerShell or Python (the user's rule, 2026-11-05)",
        )
        .with_details(details);
    }

    if !census.scripts.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!("script(s) in {HOOKS_RELDIR}: {}", census.scripts.join(", ")),
        )
        .with_remedy(
            "port it to go/claudehooks or delete it — a script in the hooks directory is a \
             shell hook waiting to be wired",
        )
        .with_details(details);
    }

    if census.uses_ix_hooks && !census.module_present {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!("hooks run ix-hooks but {GO_MODULE_RELDIR}/go.mod is absent"),
        )
        .with_remedy("restore go/claudehooks: session-start builds the hooks from it")
        .with_details(details);
    }

    if census.module_present && census.go.is_none() {
        return CheckResult::new(
            NAME,
            Status::Warn,
            format!(
                "{} hook command(s) wired, none through a shell; Go not found, so the hooks' \
                 tests were not run",
                census.commands.len()
            ),
        )
        .with_remedy(
            "install Go (the hooks build with it: session-start runs `go run`) and re-run — a \
             green doctor without this step has not exercised the hooks",
        )
        .with_details(details);
    }

    if let Some(run) = census.go_tests.as_ref().filter(|r| !r.passed()) {
        let why = match (run.timed_out, run.exit) {
            (true, _) => "timed out".to_string(),
            (false, Some(code)) => format!("exit {code}"),
            (false, None) => "killed".to_string(),
        };
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!("go test ./... in {GO_MODULE_RELDIR} failed ({why})"),
        )
        .with_remedy(
            "run `go test ./...` in go/claudehooks for the full output (the last lines ride in \
             details.go_tests.tail) — the tests are the hooks' behavioural binding, so a red one \
             means a hook does not do what its comment says",
        )
        .with_details(details);
    }

    let tested = if census.go_tests.is_some() {
        format!("go test ./... in {GO_MODULE_RELDIR} passed")
    } else {
        "no hooks module to test".to_string()
    };
    CheckResult::new(
        NAME,
        Status::Ok,
        format!(
            "{} hook command(s) wired, none through a shell; {tested}",
            census.commands.len()
        ),
    )
    .with_details(details)
}
