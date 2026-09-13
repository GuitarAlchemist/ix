//! An IXQL step implemented in V, reached through an out-of-process adapter.
//!
//! `fixtures/v/explanation_requirement.v` is compiled once per test binary and
//! registered under `explanation_requirement`. The evaluator cannot tell it
//! from a Rust closure: that is the property the capability port exists for.
//!
//! Wire contract (the adapter below is the whole of it):
//!
//! - stdin: `{"piped": …, "positional": […], "named": {…}}`
//! - stdout: `{"truth": "T|P|U|D|F|C", "confidence": 0..1}`
//! - non-zero exit: refusal, with the reason on stderr
//!
//! A check passes its piped value through; only the verdict comes from V.
//!
//! These tests are `#[ignore]`d because CI has no V compiler. `cargo test`
//! reports them as *ignored* rather than passed, and once asked for with
//! `--ignored` they fail if V cannot be found — they never pass by skipping.
//! Run them with:
//!
//! ```text
//! IX_V_EXE=/path/to/v cargo test -p ix-ixql --test v_step_tests -- --ignored
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, OnceLock};

use ix_ixql::{CallArgs, EvalError, Executor, MemoryHost, Produced, RunError, Verdict};
use ix_types::Hexavalent;
use serde_json::{json, Value};

const V_SOURCE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/v/explanation_requirement.v"
);
const PROCESSED_PATH: &str = "state/conscience/processed.json";
const RESOLVED_PATH: &str = "state/conscience/resolved.json";

/// Compile the V step once; every test shares the binary.
fn v_step_binary() -> &'static Path {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY.get_or_init(|| {
        let v = std::env::var_os("IX_V_EXE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("v"));
        let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
            "explanation_requirement{}",
            std::env::consts::EXE_SUFFIX
        ));
        let status = Command::new(&v)
            .arg("-o")
            .arg(&out)
            .arg(V_SOURCE)
            .status()
            .unwrap_or_else(|e| {
                panic!(
                    "could not run the V compiler `{}` ({e}). These tests were asked for \
                     explicitly, so a missing compiler is a failure, not a skip.",
                    v.display()
                )
            });
        assert!(status.success(), "`v -o` failed for {V_SOURCE}");
        out
    })
}

/// The out-of-process adapter: JSON in on stdin, a verdict out on stdout.
fn process_check(binary: &'static Path) -> impl Fn(CallArgs) -> Result<Produced, String> {
    move |args: CallArgs| {
        let envelope = json!({
            "piped": args.piped,
            "positional": args.positional,
            "named": args.named,
        });
        let mut child = Command::new(binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawning {}: {e}", binary.display()))?;
        child
            .stdin
            .take()
            .expect("stdin is piped")
            .write_all(envelope.to_string().as_bytes())
            .map_err(|e| format!("writing the call envelope: {e}"))?;
        let output = child
            .wait_with_output()
            .map_err(|e| format!("waiting for the step: {e}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).trim().to_string());
        }

        let reply: Value = serde_json::from_slice(&output.stdout)
            .map_err(|e| format!("stdout is not JSON ({e})"))?;
        let truth = reply["truth"]
            .as_str()
            .and_then(|s| s.chars().next().filter(|_| s.len() == 1))
            .and_then(Hexavalent::from_char)
            .ok_or_else(|| format!("no hexavalent `truth` in {reply}"))?;
        let confidence = reply["confidence"]
            .as_f64()
            .ok_or_else(|| format!("no numeric `confidence` in {reply}"))?;
        Ok(Produced::judged(
            args.piped.unwrap_or(Value::Null),
            Verdict { truth, confidence },
        ))
    }
}

fn run(processed: Value, source: &str) -> (Result<ix_ixql::RunOutcome, RunError>, Arc<MemoryHost>) {
    let host = Arc::new(MemoryHost::frozen());
    host.seed(PROCESSED_PATH, processed);
    let mut executor = Executor::new(host.clone());
    executor
        .capabilities()
        .register("explanation_requirement", process_check(v_step_binary()))
        .unwrap();
    // The other check in the verbatim fixture stays in Rust: one pipeline,
    // two languages, one port.
    executor
        .capabilities()
        .register("reversibility_check", |args: CallArgs| {
            Ok(Produced::plain(args.piped.unwrap_or(Value::Null)))
        })
        .unwrap();
    (executor.run_source(source), host)
}

const GATED: &str = r#"processed <- ix.io.read("state/conscience/processed.json")
checked <- processed
  → explanation_requirement
  → when T >= 0.7: ix.io.write("state/conscience/resolved.json", processed)"#;

#[test]
#[ignore = "needs the V compiler: set IX_V_EXE or put `v` on PATH, then run with --ignored"]
fn an_unexplained_resolution_gets_f_from_v_and_the_gate_stays_shut() {
    let (outcome, host) = run(
        json!({ "resolutions": [
            { "id": "r-1", "explanation": "duplicate signal" },
            { "id": "r-2" }
        ]}),
        GATED,
    );
    let outcome = outcome.expect("an F verdict is an answer, not a failure");
    assert_eq!(outcome.gates.len(), 1);
    assert_eq!(outcome.gates[0].verdict.truth, Hexavalent::False);
    assert_eq!(outcome.gates[0].matched, None);
    assert!(!host.files().contains_key(RESOLVED_PATH));
}

#[test]
#[ignore = "needs the V compiler: set IX_V_EXE or put `v` on PATH, then run with --ignored"]
fn fully_explained_resolutions_get_t_from_v_and_are_written() {
    let processed = json!({ "resolutions": [
        { "id": "r-1", "explanation": "duplicate signal" }
    ]});
    let (outcome, host) = run(processed.clone(), GATED);
    let outcome = outcome.expect("runs");
    assert_eq!(outcome.gates[0].verdict.truth, Hexavalent::True);
    assert_eq!(outcome.gates[0].matched, Some(0));
    assert_eq!(host.files().get(RESOLVED_PATH), Some(&processed));
}

#[test]
#[ignore = "needs the V compiler: set IX_V_EXE or put `v` on PATH, then run with --ignored"]
fn nothing_to_explain_is_u_not_t() {
    let (outcome, host) = run(json!({ "resolutions": [] }), GATED);
    let outcome = outcome.expect("runs");
    assert_eq!(outcome.gates[0].verdict.truth, Hexavalent::Unknown);
    assert!(!host.files().contains_key(RESOLVED_PATH));
}

#[test]
#[ignore = "needs the V compiler: set IX_V_EXE or put `v` on PATH, then run with --ignored"]
fn a_refusal_from_the_v_process_fails_the_run_with_its_reason() {
    let (outcome, host) = run(json!("not a record"), GATED);
    let err = outcome.expect_err("V exits non-zero on a malformed envelope");
    let RunError::Eval(EvalError::Capability { function, message }) = &err else {
        panic!("expected a capability failure, got {err}");
    };
    assert_eq!(function, "explanation_requirement");
    assert!(message.contains("not a call envelope"), "{message}");
    assert!(!host.files().contains_key(RESOLVED_PATH));
}

#[test]
#[ignore = "needs the V compiler: set IX_V_EXE or put `v` on PATH, then run with --ignored"]
fn the_verbatim_corpus_lines_run_with_one_check_in_rust_and_one_in_v() {
    let (outcome, _host) = run(
        json!({ "resolutions": [{ "id": "r-1", "explanation": "duplicate signal" }] }),
        include_str!("fixtures/named-checks.ixql"),
    );
    let outcome = outcome.expect("conscience-cycle.ixql:88-89 runs across both languages");
    assert!(outcome.binding("processed").is_some());
}
