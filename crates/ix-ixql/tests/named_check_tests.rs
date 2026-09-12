//! `→ explanation_requirement` — named governance checks as capability steps.
//!
//! The executor does not know what Article 2 or Article 3 require; that is
//! Demerzel's to define. What it guarantees is the dispatch: a bare name in
//! step position reaches the adapter registered under it, with the piped
//! value, in order — and a check nobody registered fails the run instead of
//! passing silently.

use std::sync::{Arc, Mutex};

use ix_ixql::{
    compile_program, parse_program, CallArgs, EvalError, Executor, MemoryHost, Produced, RunError,
    StageKind, Verdict,
};
use ix_types::Hexavalent;
use serde_json::{json, Value};

const PIPELINE: &str = include_str!("fixtures/named-checks.ixql");
const PROCESSED_PATH: &str = "state/conscience/processed.json";

fn processed() -> Value {
    json!({
        "resolutions": [
            { "id": "r-1", "reversible": true, "explanation": "duplicate signal" },
            { "id": "r-2", "reversible": true }
        ]
    })
}

type CallLog = Arc<Mutex<Vec<(String, Option<Value>)>>>;

fn executor_with_host() -> (Executor, Arc<MemoryHost>) {
    let host = Arc::new(MemoryHost::frozen());
    host.seed(PROCESSED_PATH, processed());
    (Executor::new(host.clone()), host)
}

/// A check that records its call and passes the value through unchanged.
fn register_passing(executor: &mut Executor, name: &'static str, log: &CallLog) {
    let log = log.clone();
    executor
        .capabilities()
        .register(name, move |args: CallArgs| {
            log.lock()
                .unwrap()
                .push((name.to_string(), args.piped.clone()));
            Ok(Produced::plain(args.piped.unwrap_or(Value::Null)))
        })
        .unwrap();
}

#[test]
fn checks_run_in_order_on_the_piped_value() {
    let (mut executor, _host) = executor_with_host();
    let log: CallLog = Arc::default();
    register_passing(&mut executor, "reversibility_check", &log);
    register_passing(&mut executor, "explanation_requirement", &log);

    let outcome = executor.run_source(PIPELINE).expect("runs");

    let log = log.lock().unwrap();
    let order: Vec<&str> = log.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(order, ["reversibility_check", "explanation_requirement"]);
    assert!(log.iter().all(|(_, piped)| piped == &Some(processed())));
    assert_eq!(outcome.binding("processed"), Some(&processed()));
}

#[test]
fn a_refusing_check_fails_the_run_and_names_itself() {
    let (mut executor, _host) = executor_with_host();
    let log: CallLog = Arc::default();
    register_passing(&mut executor, "reversibility_check", &log);
    executor
        .capabilities()
        .register("explanation_requirement", |args: CallArgs| {
            let value = args.piped.unwrap_or(Value::Null);
            let unexplained: Vec<&str> = value["resolutions"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|r| r.get("explanation").is_none())
                .filter_map(|r| r["id"].as_str())
                .collect();
            if unexplained.is_empty() {
                Ok(Produced::plain(value))
            } else {
                Err(format!(
                    "Article 2 (Transparency): no explanation for {}",
                    unexplained.join(", ")
                ))
            }
        })
        .unwrap();

    let err = executor
        .run_source(PIPELINE)
        .expect_err("r-2 has no explanation");
    let RunError::Eval(EvalError::Capability { function, message }) = &err else {
        panic!("expected a capability failure, got {err}");
    };
    assert_eq!(function, "explanation_requirement");
    assert!(message.contains("r-2"), "{message}");
    assert_eq!(log.lock().unwrap().len(), 1, "the earlier check still ran");
}

#[test]
fn an_unregistered_check_fails_closed() {
    let (mut executor, _host) = executor_with_host();
    let log: CallLog = Arc::default();
    register_passing(&mut executor, "reversibility_check", &log);

    let err = executor
        .run_source(PIPELINE)
        .expect_err("a governance check nobody implements must not pass");
    let RunError::Eval(EvalError::UnknownFunction(name)) = &err else {
        panic!("expected an unknown function, got {err}");
    };
    assert_eq!(name, "explanation_requirement");
    assert!(
        err.to_string().contains("Executor::capabilities()"),
        "{err}"
    );
}

#[test]
fn a_check_verdict_can_gate_the_next_step() {
    for (truth, confidence, expect_write) in [
        (Hexavalent::True, 0.9, true),
        (Hexavalent::False, 1.0, false),
    ] {
        let (mut executor, host) = executor_with_host();
        executor
            .capabilities()
            .register("reversibility_check", move |args: CallArgs| {
                Ok(Produced::judged(
                    args.piped.unwrap_or(Value::Null),
                    Verdict { truth, confidence },
                ))
            })
            .unwrap();

        let outcome = executor
            .run_source(
                r#"processed <- ix.io.read("state/conscience/processed.json")
                   checked <- processed
                     → reversibility_check
                     → when T >= 0.7: ix.io.write("state/conscience/resolved.json", processed)"#,
            )
            .expect("runs");

        assert_eq!(
            host.files().contains_key("state/conscience/resolved.json"),
            expect_write,
            "{truth:?} at {confidence}"
        );
        assert_eq!(outcome.gates.len(), 1);
    }
}

#[test]
fn a_dotted_bare_name_resolves_as_one_capability() {
    let (mut executor, _host) = executor_with_host();
    let log: CallLog = Arc::default();
    register_passing(&mut executor, "tars.bias_assessment", &log);
    executor
        .run_source(r#"x <- ix.io.read("state/conscience/processed.json") → tars.bias_assessment"#)
        .expect("runs");
    assert_eq!(log.lock().unwrap().len(), 1);
}

#[test]
fn named_checks_compile_to_ordered_pipe_stages() {
    let plan = compile_program(&parse_program(PIPELINE).unwrap()).expect("compiles");
    let ops: Vec<&str> = plan
        .stages()
        .iter()
        .filter(|stage| stage.kind == StageKind::Pipe)
        .map(|stage| stage.op.as_str())
        .collect();
    assert_eq!(ops, ["reversibility_check", "explanation_requirement"]);
}
