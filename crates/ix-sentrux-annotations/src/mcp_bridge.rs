//! Stdio JSON-RPC bridge to `sentrux.exe mcp`.
//!
//! The wire protocol mirrors `ga/ReactComponents/ga-react-components/vite.config.ts`
//! (search for `callSentruxTool`). We send four frames newline-separated:
//!
//! 1. `initialize` (id 1)
//! 2. `notifications/initialized` (no id)
//! 3. `tools/call` for `scan` (id 2) — sentrux requires scan before any other tool
//! 4. `tools/call` for `check_rules` (id 99) — the one whose response we care about
//!
//! Sentrux emits one JSON object per line on stdout. We collect until we see
//! the id=99 frame, then kill the child. Timeout default: 60 s.

use crate::rules_response::{parse_check_rules_response, RulesReport};
use crate::test_gaps::{parse_test_gaps_response, TestGapsReport};
use crate::Error;
use serde_json::json;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// Configuration knobs for the sentrux MCP bridge call.
#[derive(Debug, Clone)]
pub struct SentruxConfig {
    /// Path to the sentrux executable. Defaults to
    /// [`crate::DEFAULT_SENTRUX_EXE`].
    pub sentrux_exe: PathBuf,
    /// Workspace to scan. Passed to sentrux's `scan` tool.
    pub workspace: PathBuf,
    /// Overall timeout for the whole initialize → check_rules dance.
    pub timeout: Duration,
}

impl Default for SentruxConfig {
    fn default() -> Self {
        Self {
            sentrux_exe: PathBuf::from(crate::DEFAULT_SENTRUX_EXE),
            workspace: PathBuf::from("."),
            timeout: Duration::from_secs(60),
        }
    }
}

/// Run the full handshake and return the parsed `check_rules` report.
///
/// Errors:
/// - [`Error::SentruxMissing`] if `sentrux_exe` does not exist on disk.
/// - [`Error::Timeout`] if no id=99 response arrives within `timeout`.
/// - [`Error::SentruxExitedEarly`] if sentrux exits without replying.
/// - [`Error::RpcError`] / [`Error::BadResponse`] on protocol errors.
pub fn run_sentrux_check(cfg: &SentruxConfig) -> Result<RulesReport, Error> {
    let envelope = run_sentrux_tool_call(
        cfg,
        json!({"name": "check_rules", "arguments": {}}),
    )?;
    parse_check_rules_response(&envelope).map_err(Error::BadResponse)
}

/// Run the full handshake and return the parsed `test_gaps` report.
///
/// `limit` is the `top-N riskiest untested files` cap passed through to
/// sentrux. The free tier ignores this argument and returns aggregate
/// counts only; the Pro tier honors it and returns up to `limit`
/// per-file paths in [`TestGapsReport::untested_files`].
///
/// Errors mirror [`run_sentrux_check`].
pub fn run_sentrux_test_gaps(cfg: &SentruxConfig, limit: u32) -> Result<TestGapsReport, Error> {
    let envelope = run_sentrux_tool_call(
        cfg,
        json!({"name": "test_gaps", "arguments": {"limit": limit}}),
    )?;
    parse_test_gaps_response(&envelope).map_err(Error::BadResponse)
}

/// Shared transport helper: spawn `sentrux.exe mcp`, run the
/// initialize→scan handshake, then issue ONE `tools/call` (id=99) with the
/// caller-supplied `params` body, and return the id=99 envelope.
fn run_sentrux_tool_call(
    cfg: &SentruxConfig,
    tools_call_params: serde_json::Value,
) -> Result<serde_json::Value, Error> {
    if !cfg.sentrux_exe.exists() {
        return Err(Error::SentruxMissing(cfg.sentrux_exe.display().to_string()));
    }

    let mut child = Command::new(&cfg.sentrux_exe)
        .arg("mcp")
        .current_dir(&cfg.workspace)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    write_frames(&mut child, &cfg.workspace, &tools_call_params)?;
    let envelope = collect_id99(&mut child, cfg.timeout)?;

    // Best-effort kill so we don't leak the child.
    let _ = child.kill();
    let _ = child.wait();

    Ok(envelope)
}

fn write_frames(
    child: &mut Child,
    workspace: &Path,
    tools_call_params: &serde_json::Value,
) -> Result<(), Error> {
    let stdin = child
        .stdin
        .as_mut()
        .ok_or_else(|| Error::BadResponse("child has no stdin".into()))?;

    let abs_workspace = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf())
        .to_string_lossy()
        .replace('\\', "/");

    let frames = [
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "ix-sentrux-annotations", "version": env!("CARGO_PKG_VERSION") }
            }
        }),
        json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }),
        json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": { "name": "scan", "arguments": { "path": abs_workspace } }
        }),
        json!({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": tools_call_params
        }),
    ];

    let mut buf = String::new();
    for f in &frames {
        buf.push_str(&f.to_string());
        buf.push('\n');
    }
    stdin.write_all(buf.as_bytes())?;
    stdin.flush()?;
    // Drop stdin so sentrux sees EOF and finishes after replying. We can't
    // close just the writer half via .as_mut(), so we take stdin out of the
    // child here.
    drop(child.stdin.take());
    Ok(())
}

fn collect_id99(child: &mut Child, timeout: Duration) -> Result<serde_json::Value, Error> {
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| Error::BadResponse("child has no stdout".into()))?;
    let stderr = child.stderr.take();

    let (tx, rx) = mpsc::channel::<Result<serde_json::Value, String>>();

    // stdout reader: emit one parsed object per line, until id=99 or EOF.
    {
        let tx = tx.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => return,
                };
                let trimmed = line.trim();
                if !trimmed.starts_with('{') {
                    continue;
                }
                let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                if parsed.get("id").and_then(|v| v.as_u64()) == Some(99) {
                    let _ = tx.send(Ok(parsed));
                    return;
                }
            }
            let _ = tx.send(Err("stdout closed before id=99 response".into()));
        });
    }

    let deadline = Instant::now() + timeout;
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining.is_zero() {
        return Err(Error::Timeout(timeout.as_millis() as u64));
    }
    match rx.recv_timeout(remaining) {
        Ok(Ok(v)) => {
            if let Some(err) = v.get("error") {
                let msg = err
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("(unknown)")
                    .to_string();
                return Err(Error::RpcError(msg));
            }
            Ok(v)
        }
        Ok(Err(_msg)) => {
            // stdout closed early; try to capture stderr for diagnostics
            let stderr_tail = stderr
                .map(|mut s| {
                    let mut buf = String::new();
                    let _ = s.read_to_string(&mut buf);
                    let len = buf.len();
                    if len > 500 {
                        buf.split_off(len - 500)
                    } else {
                        buf
                    }
                })
                .unwrap_or_default();
            let code = child.try_wait().ok().flatten().and_then(|s| s.code());
            Err(Error::SentruxExitedEarly {
                code,
                stderr: stderr_tail,
            })
        }
        Err(mpsc::RecvTimeoutError::Timeout) => Err(Error::Timeout(timeout.as_millis() as u64)),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(Error::BadResponse(
            "stdout reader channel disconnected".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_when_sentrux_exe_missing() {
        let cfg = SentruxConfig {
            sentrux_exe: PathBuf::from("C:/nonexistent/sentrux.exe"),
            workspace: PathBuf::from("."),
            timeout: Duration::from_secs(5),
        };
        let err = run_sentrux_check(&cfg).expect_err("must fail");
        assert!(matches!(err, Error::SentruxMissing(_)));
    }

    #[test]
    fn test_gaps_errors_when_sentrux_exe_missing() {
        // The test_gaps entrypoint must share the same SentruxMissing guard
        // as `run_sentrux_check` — both go through `run_sentrux_tool_call`.
        let cfg = SentruxConfig {
            sentrux_exe: PathBuf::from("C:/nonexistent/sentrux.exe"),
            workspace: PathBuf::from("."),
            timeout: Duration::from_secs(5),
        };
        let err = run_sentrux_test_gaps(&cfg, 50).expect_err("must fail");
        assert!(matches!(err, Error::SentruxMissing(_)));
    }
}
