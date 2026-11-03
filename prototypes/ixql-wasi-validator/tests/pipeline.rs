//! End to end: IXQL source → `go.validate` capability → Go validator built with `GOOS=wasip1`
//! → ix-wasm (wasmi, WASI preview 1) → report → verdict → gate → write or alert.
//!
//! Needs Go 1.21+ on `PATH` (or `IX_GO`). The Go build is done once per test binary, through
//! ix-wasm's existing content-addressed `Toolchain`, into a temporary cache.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use ix_ixql::{CallArgs, Executor, MemoryHost, Produced, RunError};
use ix_wasm::build::Toolchain;
use ix_wasm::{Limits, Program};
use ixql_wasi_validator_prototype::WasiValidator;
use serde_json::{json, Value};

const PIPELINE: &str = include_str!("../fixtures/pipeline.ixql");
const VALIDATOR_GO: &str = include_str!("../validator/main.go");
const VALID: &str = include_str!("../fixtures/candidate-valid.json");
const INVALID: &str = include_str!("../fixtures/candidate-invalid.json");

fn validator() -> Arc<Program> {
    static P: OnceLock<(tempfile::TempDir, Arc<Program>)> = OnceLock::new();
    P.get_or_init(|| {
        let cache = tempfile::tempdir().unwrap();
        let go = std::env::var_os("IX_GO").unwrap_or_else(|| "go".into());
        let built = Toolchain::go(go, cache.path())
            .build(VALIDATOR_GO, Instant::now() + Duration::from_secs(300))
            .expect("the fixture validator builds with GOOS=wasip1");
        eprintln!("validator stage {} at {}", built.id, built.path.display());
        let program = Program::new(&std::fs::read(&built.path).unwrap()).unwrap();
        (cache, Arc::new(program))
    })
    .1
    .clone()
}

struct Run {
    host: Arc<MemoryHost>,
    alerts: Arc<Mutex<Vec<CallArgs>>>,
    result: Result<ix_ixql::RunOutcome, RunError>,
}

fn run(candidate: Value) -> Run {
    let host = Arc::new(MemoryHost::frozen());
    host.seed("sandbox/candidate.json", candidate);
    let mut executor = Executor::new(host.clone());
    executor
        .capabilities()
        .register(
            "go.validate",
            WasiValidator::new(validator(), Limits::default()),
        )
        .unwrap();
    let alerts = Arc::new(Mutex::new(Vec::new()));
    let sink = alerts.clone();
    executor
        .capabilities()
        .register("alert", move |args: CallArgs| {
            sink.lock().unwrap().push(args);
            Ok(Produced::plain(Value::Null))
        })
        .unwrap();
    let result = executor.run_source(PIPELINE);
    Run {
        host,
        alerts,
        result,
    }
}

fn parse(s: &str) -> Value {
    serde_json::from_str(s).unwrap()
}

#[test]
fn a_valid_candidate_is_judged_true_and_written() {
    let r = run(parse(VALID));
    let outcome = r.result.expect("runs");
    assert_eq!(outcome.gates.len(), 1);
    assert_eq!(outcome.gates[0].matched, Some(0), "the T arm opened");
    assert_eq!(
        r.host.files().get("sandbox/accepted.json"),
        Some(&parse(VALID))
    );
    assert!(r.alerts.lock().unwrap().is_empty());
}

#[test]
fn an_invalid_candidate_is_judged_false_and_only_alerts() {
    let r = run(parse(INVALID));
    let outcome = r
        .result
        .expect("an F verdict with an F arm is not an error");
    assert_eq!(outcome.gates[0].matched, Some(1), "the F arm opened");
    assert!(!r.host.files().contains_key("sandbox/accepted.json"));
    let alerts = r.alerts.lock().unwrap();
    assert_eq!(alerts.len(), 1);
    // The arm receives the validator's report: the Go code's errors, sorted, verbatim.
    let piped = alerts[0].piped.as_ref().unwrap();
    assert_eq!(
        piped["report"],
        json!({
            "validator": "ix-prototype-fixture-validator/0.1.0",
            "valid": false,
            "errors": [
                "schema_version: must be 1, got 2",
                "target.repo: required non-empty string",
                "unknown field: extra",
                "verdict: must be one of informational|pass|fail, got maybe"
            ]
        })
    );
    assert_eq!(
        piped["refused"],
        json!([]),
        "the validator asked for no file or socket"
    );
}

#[test]
fn a_validator_failure_fails_the_run_instead_of_becoming_a_verdict() {
    // A fuel budget too small for the Go runtime to start: the sandbox stops the module.
    let host = Arc::new(MemoryHost::frozen());
    host.seed("sandbox/candidate.json", parse(VALID));
    let mut executor = Executor::new(host.clone());
    let starved = Limits {
        fuel: 10_000,
        ..Limits::default()
    };
    executor
        .capabilities()
        .register("go.validate", WasiValidator::new(validator(), starved))
        .unwrap();
    executor
        .capabilities()
        .register("alert", |_: CallArgs| Ok(Produced::plain(Value::Null)))
        .unwrap();
    let err = executor.run_source(PIPELINE).unwrap_err();
    assert!(err.to_string().contains("out of fuel"), "{err}");
    assert!(!host.files().contains_key("sandbox/accepted.json"));
}
