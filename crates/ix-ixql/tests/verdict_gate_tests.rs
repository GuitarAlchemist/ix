//! `→ when T >= 0.8:` gates, end to end: a peer capability attaches a verdict,
//! the match picks an arm, and the arm's effect lands — or does not.
//!
//! The pipeline under test is `fixtures/verdict-gates.ixql`, the shape
//! `Demerzel/pipelines/metasync.ixql` gates a contradiction resolution with.
//! `tars.research` and `alert` are registered test adapters; `ix.io.write`
//! is the real built-in, so an admitted value still crosses the host.

use std::sync::{Arc, Mutex};

use ix_ixql::{
    compile_program, parse_program, CallArgs, EvalError, Executor, MemoryHost, PipeStep, Produced,
    RunError, StageKind, Statement, Verdict,
};
use ix_types::Hexavalent;
use serde_json::{json, Value};

const PIPELINE: &str = include_str!("fixtures/verdict-gates.ixql");
const BELIEF_PATH: &str = "state/beliefs/tempo-claim.json";

/// What `tars.research` returns in every run; only its verdict varies.
fn research_output() -> Value {
    json!({ "claim": "tempo drifts under load", "sources": 3 })
}

struct Harness {
    host: Arc<MemoryHost>,
    executor: Executor,
    alerts: Arc<Mutex<Vec<CallArgs>>>,
}

fn harness(truth: Hexavalent, confidence: f64) -> Harness {
    let host = Arc::new(MemoryHost::frozen());
    host.seed(
        BELIEF_PATH,
        json!({
            "name": "tempo-drift",
            "claim": "tempo drifts under load",
            "path": "state/beliefs/tempo-drift.belief.json"
        }),
    );
    let mut executor = Executor::new(host.clone());
    executor
        .capabilities()
        .register("tars.research", move |_args: CallArgs| {
            Ok(Produced::judged(
                research_output(),
                Verdict { truth, confidence },
            ))
        })
        .expect("tars.research registers");

    let alerts = Arc::new(Mutex::new(Vec::new()));
    let sink = alerts.clone();
    executor
        .capabilities()
        .register("alert", move |args: CallArgs| {
            sink.lock().expect("alert log").push(args);
            Ok(Produced::plain(Value::Null))
        })
        .expect("alert registers");

    Harness {
        host,
        executor,
        alerts,
    }
}

fn written(host: &MemoryHost) -> Option<Value> {
    host.files()
        .get("state/beliefs/tempo-drift.belief.json")
        .cloned()
}

#[test]
fn consecutive_arms_parse_as_one_match_not_two_filters() {
    let program = parse_program(PIPELINE).expect("fixture parses");
    let Statement::Assign(name, expr) = &program[1] else {
        panic!("second statement is the gated binding: {program:#?}");
    };
    assert_eq!(name, "resolution");
    let ix_ixql::Expr::Pipeline(_, steps) = expr.as_ref() else {
        panic!("the binding is a pipeline: {expr:#?}");
    };
    assert_eq!(steps.len(), 1, "two arms, one step: {steps:#?}");
    let PipeStep::VerdictMatch(arms) = &steps[0] else {
        panic!("the step is a verdict match: {steps:#?}");
    };
    let guards: Vec<String> = arms.iter().map(|a| a.guard.render()).collect();
    assert_eq!(guards, ["T>=0.8", "C"]);
}

#[test]
fn a_confident_true_runs_the_first_arm_and_writes() {
    let h = harness(Hexavalent::True, 0.9);
    let outcome = h.executor.run_source(PIPELINE).expect("runs");

    assert_eq!(
        written(&h.host),
        Some(json!({ "value": "tempo drifts under load" }))
    );
    assert!(h.alerts.lock().unwrap().is_empty(), "no alert on T");
    assert_eq!(outcome.gates.len(), 1);
    assert_eq!(outcome.gates[0].matched, Some(0));
    // What the arm returned is what flows on and gets bound.
    assert_eq!(
        outcome.binding("resolution"),
        Some(&json!({ "value": "tempo drifts under load" }))
    );
}

#[test]
fn a_true_below_the_bound_opens_no_arm_and_says_so() {
    let h = harness(Hexavalent::True, 0.5);
    let outcome = h
        .executor
        .run_source(PIPELINE)
        .expect("a closed gate is not an error");

    assert_eq!(written(&h.host), None, "0.5 < 0.8 must not write");
    assert!(h.alerts.lock().unwrap().is_empty());
    assert!(outcome.writes.is_empty());
    // Skipped, but visibly: the record is how a closed gate differs from a
    // pipeline that never reached one.
    assert_eq!(outcome.gates.len(), 1);
    assert_eq!(outcome.gates[0].matched, None);
    assert_eq!(outcome.gates[0].verdict.confidence, 0.5);
    // The binding holds the value as it reached the gate.
    assert_eq!(outcome.binding("resolution"), Some(&research_output()));
}

#[test]
fn a_handled_contradiction_runs_its_own_arm() {
    let h = harness(Hexavalent::Contradictory, 0.95);
    let outcome = h.executor.run_source(PIPELINE).expect("runs");

    assert_eq!(written(&h.host), None, "C must not write the resolution");
    let alerts = h.alerts.lock().unwrap();
    assert_eq!(alerts.len(), 1);
    assert_eq!(
        alerts[0].positional,
        vec![
            json!("discord"),
            json!("Contradiction persists: tempo-drift — escalate to human")
        ]
    );
    assert_eq!(
        alerts[0].piped,
        Some(research_output()),
        "an arm receives the value the gate admitted"
    );
    assert_eq!(outcome.gates[0].matched, Some(1));
}

#[test]
fn an_unmatched_false_skips_rather_than_failing() {
    let h = harness(Hexavalent::False, 1.0);
    let outcome = h
        .executor
        .run_source(PIPELINE)
        .expect("F with no F arm skips");
    assert_eq!(written(&h.host), None);
    assert_eq!(outcome.gates[0].matched, None);
}

#[test]
fn an_unhandled_contradiction_fails_the_run() {
    let mut h = harness(Hexavalent::Contradictory, 0.99);
    h.executor
        .capabilities()
        .register("tars.assess", |_args: CallArgs| {
            Ok(Produced::judged(
                json!({}),
                Verdict {
                    truth: Hexavalent::Contradictory,
                    confidence: 0.99,
                },
            ))
        })
        .unwrap();
    let err = h
        .executor
        .run_source(r#"x <- tars.assess() → when T >= 0.8: ix.io.write("state/x.json", {})"#)
        .expect_err("C with no C arm is escalated, not skipped");
    assert!(
        matches!(
            err,
            RunError::Eval(EvalError::UnhandledContradiction { .. })
        ),
        "{err}"
    );
    assert!(!h.host.files().contains_key("state/x.json"));
}

#[test]
fn a_gate_after_a_value_with_no_verdict_is_an_error() {
    let h = harness(Hexavalent::True, 1.0);
    let err = h
        .executor
        .run_source(
            r#"x <- ix.io.read("state/beliefs/tempo-claim.json") → when T: alert("a", "b")"#,
        )
        .expect_err("a built-in attaches no verdict");
    assert!(
        matches!(err, RunError::Eval(EvalError::NoVerdict { .. })),
        "{err}"
    );
    assert!(h.alerts.lock().unwrap().is_empty());
}

#[test]
fn a_verdict_outside_zero_to_one_is_refused_at_the_port() {
    let h = harness(Hexavalent::True, 1.5);
    let err = h
        .executor
        .run_source(PIPELINE)
        .expect_err("confidence 1.5 is not a confidence");
    assert!(
        matches!(err, RunError::Eval(EvalError::InvalidVerdict { .. })),
        "{err}"
    );
    assert_eq!(written(&h.host), None);
}

#[test]
fn the_first_matching_arm_wins() {
    let h = harness(Hexavalent::True, 0.95);
    let outcome = h
        .executor
        .run_source(
            r#"x <- tars.research() → when T: alert("broad", "") → when T >= 0.9: alert("narrow", "")"#,
        )
        .expect("runs");
    let alerts = h.alerts.lock().unwrap();
    assert_eq!(alerts.len(), 1);
    assert_eq!(alerts[0].positional[0], json!("broad"));
    assert_eq!(outcome.gates[0].matched, Some(0));
}

#[test]
fn steps_after_a_closed_gate_do_not_run() {
    for (confidence, expect_alert) in [(0.5, false), (0.9, true)] {
        let h = harness(Hexavalent::True, confidence);
        h.executor
            .run_source(
                r#"x <- tars.research() → when T >= 0.8: ix.io.write("state/y.json", {}) → alert("after", "")"#,
            )
            .expect("runs");
        assert_eq!(
            !h.alerts.lock().unwrap().is_empty(),
            expect_alert,
            "confidence {confidence}: the step after the gate runs only if the gate opened"
        );
    }
}

#[test]
fn a_guard_bound_outside_zero_to_one_does_not_parse() {
    let err =
        parse_program(r#"x <- a() → when T >= 1.5: b()"#).expect_err("1.5 is not a confidence");
    assert!(err.to_string().contains("[0, 1]"), "{err}");
}

#[test]
fn an_arm_must_be_a_call() {
    // `→ when T >= 0.8: { action: "archive" }` is real corpus
    // (conscience-cycle.ixql:101); a record arm is not decided yet.
    let err = parse_program(r#"x <- a() → when T >= 0.8: { action: "archive" }"#)
        .expect_err("a record arm is not supported");
    assert!(err.to_string().contains("must be a call"), "{err}");
}

#[test]
fn the_match_compiles_to_one_when_stage_spelling_every_arm() {
    let plan = compile_program(&parse_program(PIPELINE).unwrap()).expect("compiles");
    let gate = plan
        .stages()
        .iter()
        .find(|stage| stage.kind == StageKind::When)
        .expect("a when stage");
    assert_eq!(gate.op, "T>=0.8:ix.io.write|C:alert");
    assert!(
        gate.deps.iter().any(|d| d.starts_with("belief")),
        "the arms read `belief`, so the gate depends on it: {:?}",
        gate.deps
    );
}
