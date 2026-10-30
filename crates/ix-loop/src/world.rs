//! Nodes that reach past the model: the voice, and GitHub, read only.
//!
//! Both are off unless the person turns them on (`ix-loop run --voice`, `--github`). Each is
//! counted per root run rather than charged as a call: speaking and reading CI cost no model call,
//! but a loop must not flood the speakers or the API.
//!
//! - [`Voice`] speaks through the local voice service ([`VoiceService`], via `ix-voice`).
//! - [`GitHub`] reads the REST API through the person's `gh` ([`GhCli`]). A loop never writes a
//!   path: it calls `gh.prs`, `gh.checks`, `gh.runs` or `gh.log`, and the host builds a GET from
//!   checked parts, for an owner the person allowed.

use serde_json::{json, Value as Json};
use std::cell::RefCell;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::rc::Rc;
use std::time::{Duration, Instant};

/// Utterances one root run may speak.
pub const MAX_SAYS_PER_RUN: u32 = 20;
/// The longest text one `say` may speak, in characters.
pub const MAX_SAY_CHARS: usize = 2000;
/// GitHub reads one root run may make.
pub const MAX_GH_PER_RUN: u32 = 200;
/// The most log lines `gh.log` returns.
pub const MAX_LOG_LINES: usize = 200;

/// Something that speaks aloud.
pub trait Voice {
    /// Queues `text`; with `wait`, returns once it has been spoken.
    fn say(&self, text: &str, wait: bool) -> Result<Json, String>;
    /// Cuts what is being spoken and empties the queue.
    fn stop(&self) -> Result<Json, String>;
}

/// The local voice service (piper-voice), through its client.
pub struct VoiceService(pub ix_voice::Client);

impl Voice for VoiceService {
    fn say(&self, text: &str, wait: bool) -> Result<Json, String> {
        let options = ix_voice::SayOptions {
            wait,
            ..Default::default()
        };
        let said = self.0.say(text, &options).map_err(|e| e.to_string())?;
        serde_json::to_value(said).map_err(|e| e.to_string())
    }

    fn stop(&self) -> Result<Json, String> {
        let stopped = self.0.stop().map_err(|e| e.to_string())?;
        serde_json::to_value(stopped).map_err(|e| e.to_string())
    }
}

/// Something that answers GETs on the GitHub REST API.
pub trait GitHub {
    /// GETs `path`, relative to the API root, giving up at `deadline`. JSON, or the text itself
    /// when the answer is not JSON (a job's log).
    fn get(&self, path: &str, deadline: Instant) -> Result<Json, String>;
}

/// A node that is turned off: every use says how to turn it on.
pub struct Off(pub &'static str);

impl Voice for Off {
    fn say(&self, _: &str, _: bool) -> Result<Json, String> {
        Err(self.0.into())
    }
    fn stop(&self) -> Result<Json, String> {
        Err(self.0.into())
    }
}

impl GitHub for Off {
    fn get(&self, _: &str, _: Instant) -> Result<Json, String> {
        Err(self.0.into())
    }
}

pub const VOICE_OFF: Off = Off("say: the voice is off (run ix-loop with --voice)");
pub const GITHUB_OFF: Off = Off("gh: GitHub is off (run ix-loop with --github)");

/// The person's `gh`, read only: every request is `gh api -X GET <path>`.
pub struct GhCli {
    pub bin: PathBuf,
}

impl GitHub for GhCli {
    fn get(&self, path: &str, deadline: Instant) -> Result<Json, String> {
        let mut child = Command::new(&self.bin)
            .args(["api", "-X", "GET", path])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("gh: cannot start {}: {e}", self.bin.display()))?;
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
        let deadline = deadline.min(Instant::now() + Duration::from_secs(60));
        let status = loop {
            if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
                break s;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Err("gh: out of time".into());
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        let text = out_t.join().unwrap_or_default();
        let stderr = err_t.join().unwrap_or_default();
        if !status.success() {
            return Err(format!("gh: {}", clip(stderr.trim(), 300)));
        }
        Ok(serde_json::from_str(&text).unwrap_or(Json::String(text)))
    }
}

type Reply = dyn Fn(&str) -> Result<Json, String>;

/// Answers GETs from a closure and remembers the paths, for tests. A clone shares the paths.
#[derive(Clone)]
pub struct ScriptedGitHub {
    reply: Rc<Reply>,
    pub paths: Rc<RefCell<Vec<String>>>,
}

impl ScriptedGitHub {
    pub fn new(reply: impl Fn(&str) -> Result<Json, String> + 'static) -> Self {
        Self {
            reply: Rc::new(reply),
            paths: Rc::default(),
        }
    }
}

impl GitHub for ScriptedGitHub {
    fn get(&self, path: &str, _: Instant) -> Result<Json, String> {
        self.paths.borrow_mut().push(path.to_string());
        (self.reply)(path)
    }
}

/// Remembers what it was asked to say, for tests. A clone shares what was said.
#[derive(Default, Clone)]
pub struct ScriptedVoice {
    pub said: Rc<RefCell<Vec<String>>>,
}

impl Voice for ScriptedVoice {
    fn say(&self, text: &str, _: bool) -> Result<Json, String> {
        self.said.borrow_mut().push(text.to_string());
        Ok(json!({ "id": "u1", "sentences": 1 }))
    }
    fn stop(&self) -> Result<Json, String> {
        Ok(json!({ "stopped": 0 }))
    }
}

fn clip(s: &str, n: usize) -> String {
    crate::clip(s, n)
}

fn name_part(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 100
        && s != "."
        && s != ".."
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Checks `owner/name` against the owners the person allowed.
pub fn checked_repo(repo: &str, owners: &[String]) -> Result<String, String> {
    let (owner, name) = repo
        .split_once('/')
        .filter(|(o, n)| name_part(o) && name_part(n))
        .ok_or_else(|| format!("gh: not a repository: {}", clip(repo, 80)))?;
    if !owners.iter().any(|o| o.eq_ignore_ascii_case(owner)) {
        return Err(format!(
            "gh: {owner} is not an allowed owner (allowed: {})",
            owners.join(", ")
        ));
    }
    Ok(format!("{owner}/{name}"))
}

fn branch_ok(b: &str) -> bool {
    !b.is_empty()
        && b.len() <= 200
        && !b.contains("..")
        && b.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/'))
}

/// The open pull requests of `repo`.
pub fn prs(get: &dyn Fn(&str) -> Result<Json, String>, repo: &str) -> Result<Json, String> {
    let v = get(&format!("repos/{repo}/pulls?state=open&per_page=50"))?;
    let list = v.as_array().ok_or("gh: pulls is not a list")?;
    Ok(Json::Array(
        list.iter()
            .map(|p| {
                json!({
                    "number": p["number"].clone(), "title": p["title"].clone(), "draft": p["draft"].clone(),
                    "author": p["user"]["login"].clone(), "branch": p["head"]["ref"].clone(),
                    "sha": p["head"]["sha"].clone(), "updated_at": p["updated_at"].clone(),
                    "url": p["html_url"].clone(),
                })
            })
            .collect(),
    ))
}

/// The checks on the head of pull request `number`: two reads, the pull request, then the check
/// runs on its head commit. A check's `job_id` is what `gh.log` takes.
pub fn checks(
    get: &dyn Fn(&str) -> Result<Json, String>,
    repo: &str,
    number: u64,
) -> Result<Json, String> {
    let pr = get(&format!("repos/{repo}/pulls/{number}"))?;
    let sha = pr["head"]["sha"]
        .as_str()
        .filter(|s| s.len() >= 7 && s.chars().all(|c| c.is_ascii_hexdigit()))
        .ok_or("gh: the pull request has no head commit")?
        .to_string();
    let runs = get(&format!(
        "repos/{repo}/commits/{sha}/check-runs?per_page=100"
    ))?;
    let list = runs["check_runs"]
        .as_array()
        .ok_or("gh: check-runs has no check_runs")?;
    Ok(json!({
        "number": number, "sha": sha,
        "checks": list.iter().map(|c| json!({
            "name": c["name"].clone(), "status": c["status"].clone(), "conclusion": c["conclusion"].clone(),
            "job_id": c["id"].clone(), "url": c["html_url"].clone(),
        })).collect::<Vec<_>>(),
    }))
}

/// The latest workflow runs of `repo`, on `branch` when given.
pub fn runs(
    get: &dyn Fn(&str) -> Result<Json, String>,
    repo: &str,
    branch: Option<&str>,
) -> Result<Json, String> {
    let mut path = format!("repos/{repo}/actions/runs?per_page=20");
    if let Some(b) = branch {
        if !branch_ok(b) {
            return Err(format!("gh: not a branch name: {}", clip(b, 80)));
        }
        path.push_str(&format!("&branch={b}"));
    }
    let v = get(&path)?;
    let list = v["workflow_runs"]
        .as_array()
        .ok_or("gh: runs has no workflow_runs")?;
    Ok(Json::Array(
        list.iter()
            .map(|r| {
                json!({
                    "id": r["id"].clone(), "name": r["name"].clone(), "status": r["status"].clone(),
                    "conclusion": r["conclusion"].clone(), "branch": r["head_branch"].clone(),
                    "sha": r["head_sha"].clone(), "event": r["event"].clone(),
                    "created_at": r["created_at"].clone(), "url": r["html_url"].clone(),
                })
            })
            .collect(),
    ))
}

/// The last `lines` lines of job `job_id`'s log, each clipped to 300 characters.
pub fn log(
    get: &dyn Fn(&str) -> Result<Json, String>,
    repo: &str,
    job_id: u64,
    lines: usize,
) -> Result<Json, String> {
    let v = get(&format!("repos/{repo}/actions/jobs/{job_id}/logs"))?;
    let text = match &v {
        Json::String(t) => t.clone(),
        other => other.to_string(),
    };
    let all: Vec<&str> = text.lines().collect();
    let n = lines.clamp(1, MAX_LOG_LINES);
    let tail = all[all.len().saturating_sub(n)..]
        .iter()
        .map(|l| Json::String(clip(l, 300)))
        .collect();
    Ok(Json::Array(tail))
}
