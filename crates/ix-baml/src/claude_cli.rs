//! A [`BamlOperation`] backed by the Claude Code CLI.
//!
//! IX has no local model and MCP's `sampling/createMessage` is deprecated
//! (SEP-2577) — Claude Code does not implement it — so a subprocess is
//! currently the only way for an IXQL pipeline to reach a frontier model
//! without provisioning a separate API key. `claude --print` gives exactly the
//! contract this seam wants: JSON in, schema-constrained JSON out.
//!
//! # What was measured, not assumed
//!
//! Against the CLI on 2026-07-29:
//!
//! - `--json-schema` takes the schema **inline**, not a path.
//! - The structured value comes back on the envelope's `structured_output`
//!   field. `result` holds the same thing as a *string*, so parsing `result`
//!   works but is the fallback, not the primary path.
//! - `--output-format json` emits a single object (not the array the GitHub
//!   Action's execution log uses).
//! - The prompt is accepted on **stdin**. That is what this module uses:
//!   Windows caps a command line at ~32 KB and an IXQL payload can exceed it.
//! - A trivial call ("is 7 odd") cost **$0.28** — 45,840 cache-creation input
//!   tokens for the default system prompt, tool definitions and `CLAUDE.md`,
//!   none of which is reused across invocations. See [`ClaudeCliConfig`] for
//!   what can be trimmed and what that costs you.
//!
//! # Authorization boundary
//!
//! `claude` is an *agentic* CLI: by default it can read the repo, run commands
//! and write files. An IXQL step that can do those things no longer produces
//! *understanding* — it pursues goals, which is the Asimov Article 4 line. So
//! two flags are fixed here and are not configurable:
//!
//! - `--permission-mode manual` — non-interactive tool requests fail closed
//!   rather than being auto-approved.
//! - `--strict-mcp-config` — the project's MCP servers are not loaded.
//!
//! `--permission-mode manual` is verified against CLI **2.1.220**, whose
//! `--help` lists it among the accepted modes and which accepted it in a live
//! run. It is not in the published flag reference, so treat it as a version
//! floor: an older CLI that rejects the mode fails at argument parsing, loudly,
//! which is the right direction to fail in.
//!
//! A caller who genuinely wants tool access must say so explicitly through
//! [`ClaudeCliConfig::extra_args`], which makes it a visible decision in the
//! calling code rather than a default nobody chose.
//!
//! # Known limit: the timeout kills one process, not a tree
//!
//! On timeout the child is killed directly. With tools disabled — the default —
//! there is nothing else to kill. If a caller enables tools through
//! `extra_args` and the model has launched a long-running command, that
//! grandchild survives the kill and can keep running or keep writing. Confining
//! it properly needs a process group on Unix and a job object on Windows;
//! until that exists, enabling tools here means accepting that a timeout is not
//! a guaranteed stop.

use std::ffi::OsString;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::{BamlError, BamlOperation};

/// Ambient environment variables that route the CLI to a metered backend,
/// cleared from the child unless [`ClaudeCliConfig::inherit_metered_auth`].
///
/// `ANTHROPIC_API_KEY` is the common route, not the only one — each of these
/// sends the bill somewhere other than the subscription seat this provider
/// exists to use, and does so *silently*, because the run then succeeds.
///
/// Deliberately **not** listed: `AWS_*` / `GOOGLE_*` credentials. Without the
/// routing switches above they select no backend, and clearing them could
/// break unrelated tooling sharing the process tree. Suppress the switch, not
/// the whole cloud.
///
/// The list will grow; [`tests::no_metered_route_survives_into_the_child`]
/// pins the property rather than the contents.
pub const SUPPRESSED_AUTH_VARS: &[&str] = &[
    // Metered Anthropic API — same destination as the key, different name.
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    // Cloud routing switches: bill the AWS account / GCP project instead.
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    // A gateway/proxy that may meter.
    "ANTHROPIC_BASE_URL",
];

/// How the subprocess is launched.
#[derive(Debug, Clone)]
pub struct ClaudeCliConfig {
    /// The executable. Default `claude`, resolved on `PATH`.
    pub binary: OsString,

    /// Model alias or full name (`sonnet`, `opus`, `claude-sonnet-5`). `None`
    /// leaves the CLI's own default in place.
    pub model: Option<String>,

    /// Directory the CLI runs in. This is also the directory whose `CLAUDE.md`
    /// and settings it picks up, so it affects both behaviour and cost.
    pub working_dir: Option<PathBuf>,

    /// How long to wait before killing the child. Default 120s.
    pub timeout: Duration,

    /// Whether to let ambient metered-backend routing reach the subprocess.
    ///
    /// Default `false`, and that default is the point of using the CLI at all:
    /// when such a variable is present the CLI prefers it over the claude.ai
    /// login, so a machine with a spend-limited key set in its environment
    /// gets `400 You have reached your specified API usage limits` even though
    /// the interactive session works fine. Clearing them for the child forces
    /// subscription auth.
    ///
    /// Setting this to `true` moves the call between two budget tiers that
    /// Demerzel's `state/driver/aiw-budget-policy.json` treats differently —
    /// `claude-code-cli` is `local-seat` / no approval, `anthropic-api` is
    /// `metered-cloud` / `requires_manual_approval: true`. Nothing in this
    /// crate enforces that gate, so a caller flipping this is asserting the
    /// approval was obtained elsewhere.
    ///
    /// See [`SUPPRESSED_AUTH_VARS`] for exactly what is cleared.
    pub inherit_metered_auth: bool,

    /// Extra arguments appended verbatim, after the fixed ones.
    ///
    /// This is the only way to re-enable tool access, and doing so has a
    /// consequence beyond the authorization one: the timeout kills the CLI
    /// process but not any command it spawned. See the module docs.
    pub extra_args: Vec<String>,
}

impl Default for ClaudeCliConfig {
    fn default() -> Self {
        Self {
            binary: OsString::from("claude"),
            model: None,
            working_dir: None,
            timeout: Duration::from_secs(120),
            inherit_metered_auth: false,
            extra_args: Vec::new(),
        }
    }
}

/// One BAML-shaped function whose implementation is a `claude --print` call.
pub struct ClaudeCliOperation {
    name: String,
    instruction: String,
    schema: Value,
    config: ClaudeCliConfig,
}

impl ClaudeCliOperation {
    /// `instruction` is the task text; the input value is appended to it as
    /// pretty JSON. `schema` constrains the reply and is passed to
    /// `--json-schema`.
    pub fn new(name: impl Into<String>, instruction: impl Into<String>, schema: Value) -> Self {
        Self {
            name: name.into(),
            instruction: instruction.into(),
            schema,
            config: ClaudeCliConfig::default(),
        }
    }

    pub fn with_config(mut self, config: ClaudeCliConfig) -> Self {
        self.config = config;
        self
    }

    /// The prompt as the CLI will receive it on stdin.
    pub fn render_prompt(&self, input: &Value) -> String {
        format!(
            "{}\n\nInput (JSON):\n{}",
            self.instruction,
            serde_json::to_string_pretty(input).unwrap_or_else(|_| input.to_string())
        )
    }

    /// The argument vector, exposed so the guardrails are testable without
    /// spending a model call.
    pub fn build_args(&self) -> Vec<String> {
        let mut args = vec![
            "--print".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
            "--json-schema".to_string(),
            self.schema.to_string(),
            // Fixed guardrails — see the module docs.
            "--permission-mode".to_string(),
            "manual".to_string(),
            "--strict-mcp-config".to_string(),
        ];
        if let Some(model) = &self.config.model {
            args.push("--model".to_string());
            args.push(model.clone());
        }
        args.extend(self.config.extra_args.iter().cloned());
        args
    }

    fn fail(&self, message: impl Into<String>) -> BamlError {
        BamlError::Invocation {
            function: self.name.clone(),
            message: message.into(),
        }
    }

    /// Clear every ambient route to a metered backend from the child.
    ///
    /// Split out from [`Self::run`] so the property — *no* suppressed variable
    /// reaches the child — can be asserted without spawning anything.
    fn apply_auth_suppression(&self, command: &mut Command) {
        if !self.config.inherit_metered_auth {
            for var in SUPPRESSED_AUTH_VARS {
                command.env_remove(var);
            }
        }
    }

    fn run(&self, prompt: &str) -> Result<String, String> {
        let mut command = Command::new(&self.config.binary);
        command
            .args(self.build_args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(dir) = &self.config.working_dir {
            command.current_dir(dir);
        }
        self.apply_auth_suppression(&mut command);

        let mut child = command.spawn().map_err(|e| {
            format!(
                "could not start `{}`: {e}",
                self.config.binary.to_string_lossy()
            )
        })?;

        // stdin, stdout and stderr are drained on their own threads. Writing a
        // large prompt while nothing reads stdout deadlocks once the pipe
        // buffer fills, and IXQL payloads are exactly the size where that
        // starts happening.
        let mut stdin = child.stdin.take().ok_or("child stdin was not piped")?;
        let prompt = prompt.to_string();
        let writer = std::thread::spawn(move || stdin.write_all(prompt.as_bytes()));

        let mut stdout = child.stdout.take().ok_or("child stdout was not piped")?;
        let reader = std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = stdout.read_to_string(&mut buf);
            buf
        });

        let mut stderr = child.stderr.take().ok_or("child stderr was not piped")?;
        let err_reader = std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = stderr.read_to_string(&mut buf);
            buf
        });

        let deadline = Instant::now() + self.config.timeout;
        let status = loop {
            match child
                .try_wait()
                .map_err(|e| format!("waiting on child: {e}"))?
            {
                Some(status) => break status,
                None if Instant::now() >= deadline => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "timed out after {}s",
                        self.config.timeout.as_secs()
                    ));
                }
                None => std::thread::sleep(Duration::from_millis(50)),
            }
        };

        let _ = writer.join();
        let stdout = reader.join().unwrap_or_default();
        let stderr = err_reader.join().unwrap_or_default();

        if !status.success() && stdout.trim().is_empty() {
            return Err(format!(
                "exited with {status}: {}",
                stderr.trim().chars().take(400).collect::<String>()
            ));
        }
        Ok(stdout)
    }
}

impl BamlOperation for ClaudeCliOperation {
    fn name(&self) -> &str {
        &self.name
    }

    fn invoke(&self, input: &Value) -> Result<Value, BamlError> {
        let stdout = self
            .run(&self.render_prompt(input))
            .map_err(|e| self.fail(e))?;
        parse_envelope(&stdout, &self.schema).map_err(|e| self.fail(e))
    }
}

/// Pull the structured value out of a `--output-format json` envelope.
///
/// Separate from the subprocess so every branch is testable against captured
/// output instead of a live call.
pub fn parse_envelope(stdout: &str, schema: &Value) -> Result<Value, String> {
    // The CLI may print warnings (e.g. about connector auth) before the
    // envelope, so take the last line that parses as a JSON object rather than
    // assuming the whole of stdout is JSON.
    let envelope = stdout
        .lines()
        .rev()
        .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
        .find(Value::is_object)
        .ok_or_else(|| {
            format!(
                "no JSON envelope in output: {}",
                stdout.trim().chars().take(300).collect::<String>()
            )
        })?;

    if envelope
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        let message = envelope
            .get("result")
            .and_then(Value::as_str)
            .unwrap_or("the CLI reported an error with no message");
        return Err(message.to_string());
    }

    // `structured_output` is the parsed value; `result` carries the same thing
    // as a string. Prefer the former and fall back, so a CLI version that
    // stops emitting `structured_output` degrades instead of breaking.
    if let Some(value) = envelope.get("structured_output") {
        if !value.is_null() {
            return Ok(value.clone());
        }
    }

    let text = envelope
        .get("result")
        .and_then(Value::as_str)
        .ok_or("envelope had neither `structured_output` nor a string `result`")?;

    let value: Value = serde_json::from_str(text).map_err(|e| {
        format!(
            "`result` was not JSON ({e}): {}",
            text.chars().take(200).collect::<String>()
        )
    })?;

    // The fallback exists for the case where the CLI stopped honouring
    // `--json-schema` the way we expect — which is exactly the case where its
    // output is least trustworthy. Accepting any syntactically valid JSON here
    // would let a reply missing required fields into the pipeline under a
    // contract that says it was schema-constrained, so it is checked against
    // the same schema the flag carried.
    validate(&value, schema)?;
    Ok(value)
}

/// Check a value against the schema the request declared.
fn validate(value: &Value, schema: &Value) -> Result<(), String> {
    let validator = jsonschema::draft202012::new(schema)
        .map_err(|e| format!("the declared schema does not compile: {e}"))?;
    let errors: Vec<String> = validator
        .iter_errors(value)
        .map(|e| e.to_string())
        .collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "the CLI's `result` fell back to unstructured output that violates the declared schema: {}",
            errors.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A schema that accepts anything — for the envelope tests that are about
    /// parsing, not validation.
    fn permissive() -> Value {
        json!({})
    }

    fn operation() -> ClaudeCliOperation {
        ClaudeCliOperation::new(
            "EvaluateSignalSwarm",
            "Judge whether this signal is within constitutional bounds.",
            json!({"type": "object", "required": ["verdict"]}),
        )
    }

    /// The property, not the list: whatever `SUPPRESSED_AUTH_VARS` grows to
    /// contain, none of it may reach the child. Asserted by iterating the
    /// constant rather than restating it, so adding a variable without wiring
    /// it up fails here.
    #[test]
    fn no_metered_route_survives_into_the_child() {
        let op = operation();
        let mut command = Command::new("claude");
        // Pretend every route is set in the ambient environment.
        for var in SUPPRESSED_AUTH_VARS {
            command.env(var, "set-in-parent");
        }
        op.apply_auth_suppression(&mut command);

        let envs: Vec<_> = command.get_envs().collect();
        for var in SUPPRESSED_AUTH_VARS {
            let entry = envs
                .iter()
                .find(|(k, _)| k.to_str() == Some(var))
                .unwrap_or_else(|| panic!("{var} missing from the child's env changes entirely"));
            assert!(
                entry.1.is_none(),
                "{var} still reaches the child — it routes billing off the subscription seat"
            );
        }
    }

    /// The escape hatch still works, and is the only way any of them survive.
    #[test]
    fn inherit_metered_auth_lets_the_ambient_routes_through() {
        let mut op = operation();
        op.config.inherit_metered_auth = true;
        let mut command = Command::new("claude");
        for var in SUPPRESSED_AUTH_VARS {
            command.env(var, "set-in-parent");
        }
        op.apply_auth_suppression(&mut command);

        assert!(
            command.get_envs().all(|(_, v)| v.is_some()),
            "inherit_metered_auth = true must not clear anything"
        );
    }

    /// `ANTHROPIC_API_KEY` was the only variable suppressed before #276; the
    /// four cloud/gateway routes are the regression this pins.
    #[test]
    fn suppression_list_covers_every_known_metered_route() {
        for expected in [
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_USE_BEDROCK",
            "CLAUDE_CODE_USE_VERTEX",
            "ANTHROPIC_BASE_URL",
        ] {
            assert!(
                SUPPRESSED_AUTH_VARS.contains(&expected),
                "{expected} dropped out of the suppression list"
            );
        }
    }

    /// Captured verbatim from a real call, minus the fields this module never
    /// reads. Keeping it real is the point: a hand-written envelope would only
    /// prove the parser matches my guess about the shape.
    const REAL_SUCCESS: &str = r#"{"is_error":false,"num_turns":2,"stop_reason":"tool_use","subtype":"success","api_error_status":null,"result":"{\"parity\":\"odd\",\"n\":7}","structured_output":{"parity":"odd","n":7},"type":"result","total_cost_usd":0.276261}"#;

    const REAL_API_ERROR: &str = r#"{"is_error":true,"num_turns":1,"subtype":"success","api_error_status":400,"result":"API Error: 400 You have reached your specified API usage limits. You will regain access on 2026-08-01 at 00:00 UTC.","type":"result"}"#;

    #[test]
    fn structured_output_is_the_value_returned() {
        assert_eq!(
            json!({"parity": "odd", "n": 7}),
            parse_envelope(REAL_SUCCESS, &permissive()).unwrap()
        );
    }

    #[test]
    fn a_leading_warning_line_does_not_defeat_the_parser() {
        // The CLI prints this when ANTHROPIC_API_KEY shadows the claude.ai
        // login — verbatim from a real run.
        let noisy = format!(
            "⚠ claude.ai connectors are disabled because ANTHROPIC_API_KEY or another auth source is set\n{REAL_SUCCESS}"
        );
        assert_eq!(
            json!({"parity": "odd", "n": 7}),
            parse_envelope(&noisy, &permissive()).unwrap()
        );
    }

    #[test]
    fn an_api_error_surfaces_the_providers_message() {
        let err = parse_envelope(REAL_API_ERROR, &permissive()).unwrap_err();
        assert!(err.contains("400"), "{err}");
        assert!(err.contains("usage limits"), "{err}");
    }

    #[test]
    fn result_string_is_the_fallback_when_structured_output_is_absent() {
        let envelope = r#"{"is_error":false,"result":"{\"verdict\":\"pass\"}","type":"result"}"#;
        assert_eq!(
            json!({"verdict": "pass"}),
            parse_envelope(envelope, &permissive()).unwrap()
        );
    }

    #[test]
    fn the_fallback_is_checked_against_the_declared_schema() {
        // The fallback exists for a CLI that stopped honouring --json-schema,
        // which is exactly when its output is least trustworthy. Valid JSON
        // that violates the contract must not reach the pipeline.
        let schema = json!({
            "type": "object",
            "required": ["verdict"],
            "properties": {"verdict": {"enum": ["pass", "fail"]}}
        });
        let envelope =
            r#"{"is_error":false,"result":"{\"verdict\":\"probably fine\"}","type":"result"}"#;
        let err = parse_envelope(envelope, &schema).unwrap_err();
        assert!(err.contains("violates the declared schema"), "{err}");

        let good = r#"{"is_error":false,"result":"{\"verdict\":\"pass\"}","type":"result"}"#;
        assert_eq!(
            json!({"verdict": "pass"}),
            parse_envelope(good, &schema).unwrap()
        );
    }

    #[test]
    fn structured_output_is_trusted_without_revalidation() {
        // The CLI already enforced the schema on this path; re-running the
        // validator would only add a second place for the two to disagree.
        let schema = json!({"type": "object", "required": ["nope"]});
        assert_eq!(
            json!({"parity": "odd", "n": 7}),
            parse_envelope(REAL_SUCCESS, &schema).unwrap()
        );
    }

    #[test]
    fn prose_where_json_was_promised_is_an_error_not_a_string_value() {
        let envelope =
            r#"{"is_error":false,"result":"Sure! The signal looks fine.","type":"result"}"#;
        let err = parse_envelope(envelope, &permissive()).unwrap_err();
        assert!(err.contains("was not JSON"), "{err}");
    }

    #[test]
    fn empty_or_garbled_output_is_reported_rather_than_panicking() {
        assert!(parse_envelope("", &permissive()).is_err());
        assert!(parse_envelope("Killed", &permissive()).is_err());
    }

    #[test]
    fn the_authorization_guardrails_are_always_on_the_command_line() {
        let args = operation().build_args();
        for expected in [
            "--print",
            "--permission-mode",
            "manual",
            "--strict-mcp-config",
        ] {
            assert!(
                args.iter().any(|a| a == expected),
                "missing {expected}: {args:?}"
            );
        }
    }

    #[test]
    fn the_schema_is_passed_inline_not_as_a_path() {
        // Measured: `--json-schema` takes the schema text itself.
        let args = operation().build_args();
        let index = args.iter().position(|a| a == "--json-schema").unwrap();
        let schema: Value = serde_json::from_str(&args[index + 1]).expect("inline JSON");
        assert_eq!(json!(["verdict"]), schema["required"]);
    }

    #[test]
    fn extra_args_come_after_the_fixed_ones_so_they_cannot_be_overridden_silently() {
        let op = operation().with_config(ClaudeCliConfig {
            model: Some("sonnet".into()),
            extra_args: vec!["--allowedTools".into(), "Read".into()],
            ..Default::default()
        });
        let args = op.build_args();
        let strict = args
            .iter()
            .position(|a| a == "--strict-mcp-config")
            .unwrap();
        let extra = args.iter().position(|a| a == "--allowedTools").unwrap();
        assert!(strict < extra);
        assert!(args
            .windows(2)
            .any(|w| w[0] == "--model" && w[1] == "sonnet"));
    }

    #[test]
    fn the_prompt_carries_the_input_as_json() {
        let prompt = operation().render_prompt(&json!({"thd": 0.02}));
        assert!(prompt.contains("constitutional bounds"));
        assert!(prompt.contains("\"thd\": 0.02"));
    }

    #[test]
    fn a_missing_binary_fails_the_operation_rather_than_hanging() {
        let op = operation().with_config(ClaudeCliConfig {
            binary: OsString::from("claude-which-does-not-exist"),
            ..Default::default()
        });
        let err = op.invoke(&json!({})).unwrap_err();
        assert!(err.to_string().contains("could not start"), "{err}");
    }

    #[test]
    #[ignore = "spends real tokens; run with --ignored when verifying against a live CLI"]
    fn live_call_returns_a_schema_shaped_value() {
        let op = ClaudeCliOperation::new(
            "Parity",
            "Return the parity of the given number.",
            json!({
                "type": "object",
                "additionalProperties": false,
                "required": ["parity", "n"],
                "properties": {
                    "parity": {"enum": ["even", "odd"]},
                    "n": {"type": "integer"}
                }
            }),
        )
        .with_config(ClaudeCliConfig {
            model: Some("sonnet".into()),
            ..Default::default()
        });

        let out = op.invoke(&json!({"n": 7})).expect("live CLI call");
        assert_eq!(json!("odd"), out["parity"]);
        assert_eq!(json!(7), out["n"]);
    }
}
