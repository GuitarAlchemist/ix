//! The semantic node: where a loop asks a model.
//!
//! [`ClaudeCode`] runs Claude Code headless on the person's own plan: `claude -p` with no tools,
//! no saved session and JSON output. It never passes `--bare`, which reads only an API key and
//! so would leave the plan. [`Scripted`] answers from a closure, so tests never need `claude`.

use serde_json::Value as Json;
use std::cell::{Cell, RefCell};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// One question to the semantic node.
#[derive(Debug, Clone, PartialEq)]
pub struct Ask {
    pub prompt: String,
    /// `haiku`, `sonnet`, or a full model name.
    pub model: String,
    /// When given, the answer is a JSON value that follows this JSON Schema.
    pub schema: Option<Json>,
    pub system: Option<String>,
}

/// The node's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The text of the answer, or the structured value when the ask had a schema.
    pub value: Json,
    /// What the call would cost at list price. On a plan nothing is billed per call.
    pub notional_usd: f64,
}

/// Something a loop can ask.
pub trait SemanticNode {
    /// Answers `ask`, giving up at `deadline`.
    fn ask(&self, ask: &Ask, deadline: Instant) -> Result<Answer, String>;
}

/// Claude Code, headless, on the person's plan.
#[derive(Debug, Clone)]
pub struct ClaudeCode {
    pub bin: PathBuf,
    /// The folder `claude` runs in. It should be a folder the person trusts and that holds
    /// nothing a model should see.
    pub work_dir: PathBuf,
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        s.chars().take(n).collect::<String>() + "…"
    }
}

fn valid_model(m: &str) -> bool {
    !m.is_empty()
        && m.len() <= 64
        && !m.starts_with('-')
        && m.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | '[' | ']'))
}

impl ClaudeCode {
    pub fn new(work_dir: impl Into<PathBuf>) -> Self {
        Self {
            bin: Self::find_bin(),
            work_dir: work_dir.into(),
        }
    }

    /// `CLAUDE_BIN`, else the native installer's `~/.local/bin/claude`, else `claude` on the path.
    pub fn find_bin() -> PathBuf {
        if let Some(p) = std::env::var_os("CLAUDE_BIN") {
            return PathBuf::from(p);
        }
        let exe = if cfg!(windows) {
            "claude.exe"
        } else {
            "claude"
        };
        if let Some(home) = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")) {
            let local = PathBuf::from(home).join(".local").join("bin").join(exe);
            if local.exists() {
                return local;
            }
        }
        PathBuf::from("claude")
    }

    /// The command line for `ask`, without the program. The prompt goes on stdin.
    pub fn args(&self, ask: &Ask) -> Vec<String> {
        let mut args: Vec<String> = [
            "-p",
            "--safe-mode",
            "--tools",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--model",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        args.push(ask.model.clone());
        if let Some(schema) = &ask.schema {
            args.push("--json-schema".into());
            args.push(schema.to_string());
        }
        if let Some(system) = &ask.system {
            args.push("--system-prompt".into());
            args.push(system.clone());
        }
        args
    }
}

impl SemanticNode for ClaudeCode {
    fn ask(&self, ask: &Ask, deadline: Instant) -> Result<Answer, String> {
        if !valid_model(&ask.model) {
            return Err(format!(
                "claude: not a model name: {}",
                clip(&ask.model, 40)
            ));
        }
        if ask.system.as_deref().is_some_and(|s| s.starts_with('-')) {
            return Err("claude: a system prompt cannot start with '-'".into());
        }
        std::fs::create_dir_all(&self.work_dir)
            .map_err(|e| format!("claude: no working folder: {e}"))?;
        let mut child = Command::new(&self.bin)
            .args(self.args(ask))
            .current_dir(&self.work_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("claude: cannot start {}: {e}", self.bin.display()))?;
        let mut out = child.stdout.take().expect("piped stdout");
        let mut err = child.stderr.take().expect("piped stderr");
        let out_t = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = out.read_to_string(&mut s);
            s
        });
        let err_t = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = err.read_to_string(&mut s);
            s
        });
        // Dropping stdin after the write closes it, which ends the prompt.
        let wrote = child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(ask.prompt.as_bytes());
        loop {
            if child.try_wait().map_err(|e| e.to_string())?.is_some() {
                break;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err("budget: out of time during a call".into());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let text = out_t.join().unwrap_or_default();
        let stderr = err_t.join().unwrap_or_default();
        wrote.map_err(|e| format!("claude: cannot write the prompt: {e}"))?;
        let v: Json = serde_json::from_str(text.trim()).map_err(|e| {
            format!(
                "claude: unreadable output ({e}): {}",
                clip(&(text + &stderr), 300)
            )
        })?;
        let notional_usd = v["total_cost_usd"].as_f64().unwrap_or(0.0);
        if v["is_error"].as_bool().unwrap_or(false) {
            return Err(format!("claude: {}", clip(&v["result"].to_string(), 300)));
        }
        let value = if ask.schema.is_some() {
            match v.get("structured_output").filter(|s| !s.is_null()) {
                Some(s) => s.clone(),
                None => {
                    let raw = v["result"].as_str().unwrap_or("");
                    serde_json::from_str(raw)
                        .map_err(|_| format!("claude: no structured output: {}", clip(raw, 300)))?
                }
            }
        } else {
            v["result"].clone()
        };
        Ok(Answer {
            value,
            notional_usd,
        })
    }
}

type Reply = dyn Fn(&Ask) -> Result<Json, String>;

/// A semantic node that answers from a closure and remembers what it was asked.
pub struct Scripted {
    reply: Box<Reply>,
    asks: RefCell<Vec<Ask>>,
    calls: Cell<usize>,
}

impl Scripted {
    pub fn new(reply: impl Fn(&Ask) -> Result<Json, String> + 'static) -> Self {
        Self {
            reply: Box::new(reply),
            asks: RefCell::new(Vec::new()),
            calls: Cell::new(0),
        }
    }

    /// The same text for every ask.
    pub fn always(text: &str) -> Self {
        let text = text.to_string();
        Self::new(move |_| Ok(Json::String(text.clone())))
    }

    /// How many times it was asked.
    pub fn calls(&self) -> usize {
        self.calls.get()
    }

    pub fn asks(&self) -> Vec<Ask> {
        self.asks.borrow().clone()
    }
}

impl SemanticNode for Scripted {
    fn ask(&self, ask: &Ask, _deadline: Instant) -> Result<Answer, String> {
        self.calls.set(self.calls.get() + 1);
        self.asks.borrow_mut().push(ask.clone());
        Ok(Answer {
            value: (self.reply)(ask)?,
            notional_usd: 0.0,
        })
    }
}
