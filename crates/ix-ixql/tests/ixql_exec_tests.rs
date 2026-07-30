//! End-to-end: run Demerzel's real `qa-architect-cycle.ixql` offline.
//!
//! The fixture at `tests/fixtures/qa-architect-cycle.ixql` is a byte-for-byte
//! copy of `Demerzel/pipelines/qa-architect-cycle.ixql`. It is vendored rather
//! than read from a sibling clone so this test passes on CI, where Demerzel is
//! not checked out. If the pipeline changes upstream and this copy does not,
//! the test keeps proving the executor runs *the version it was written
//! against* — which is the honest guarantee for a cross-repo fixture.
//!
//! Nothing here touches the network, a provider, or the filesystem: the host
//! is in-memory with a frozen clock, so the whole run is deterministic down to
//! the generated verdict id.

use std::sync::Arc;

use ix_baml::{FnOperation, StaticResponse};
use ix_ixql::{CompoundRecord, Executor, MemoryHost};
use serde_json::{json, Value};

const PIPELINE: &str = include_str!("fixtures/qa-architect-cycle.ixql");
const VERDICT_SCHEMA: &str = include_str!("fixtures/qa-verdict.subset.schema.json");

/// The frozen clock is `2026-01-02T03:04:05Z` (see `MemoryHost::frozen`), and
/// the pipeline builds its id from `now_utc("yyyy-MM-ddTHH-mm-ssZ")`.
const EXPECTED_VERDICT_ID: &str =
    "2026-01-02T03-04-05Z-scheduled_sweep-skeleton-qa-architect-cycle";
const EXPECTED_PATH: &str =
    "state/quality/verdicts/guitar-alchemist/ga/skeleton/2026-01-02T03-04-05Z-scheduled_sweep-skeleton-qa-architect-cycle.json";

fn gated_executor(host: Arc<MemoryHost>) -> Executor {
    let mut executor = Executor::new(host);
    let schema: Value = serde_json::from_str(VERDICT_SCHEMA).expect("fixture schema parses");
    executor
        .schema_gate()
        .register(
            "state/quality/verdicts/",
            "qa-verdict.subset.schema.json",
            &schema,
        )
        .expect("fixture schema compiles");
    executor
}

#[test]
fn the_real_pipeline_parses() {
    let program = ix_ixql::parse_program(PIPELINE).expect("qa-architect-cycle.ixql parses");
    // Eight bindings — trigger_context, blast_radius, reviewer_chain,
    // produced_at_iso, produced_at_safe, verdict_id, verdict, verdict_path —
    // plus the trailing `ix.io.write(…)` pipeline statement.
    assert_eq!(9, program.len(), "statement count changed: {program:#?}");
}

#[test]
fn the_real_pipeline_runs_offline_and_emits_a_contract_shaped_verdict() {
    let host = Arc::new(MemoryHost::frozen());
    let outcome = gated_executor(host.clone())
        .run_source(PIPELINE)
        .expect("pipeline runs");

    // Step 1: no trigger file exists, so `→ default(…)` supplies the sweep.
    assert_eq!(
        &json!({
            "kind": "scheduled_sweep",
            "repo": "guitar-alchemist/ga",
            "ref": "skeleton",
            "sha": null,
            "base_sha": null
        }),
        outcome.binding("trigger_context").unwrap()
    );

    // Step 4: the id is interpolated from the frozen clock and the trigger.
    assert_eq!(
        &json!(EXPECTED_VERDICT_ID),
        outcome.binding("verdict_id").unwrap()
    );

    let verdict = outcome.binding("verdict").unwrap();
    assert_eq!(json!(1), verdict["schema_version"]);
    assert_eq!(json!("informational"), verdict["verdict"]);
    assert_eq!(json!("2026-01-02T03:04:05Z"), verdict["produced_at"]);
    assert_eq!(json!("guitar-alchemist/ga"), verdict["target"]["repo"]);

    // Step 5: one write, at the path the contract's layout prescribes, and it
    // actually reached the host.
    assert_eq!(1, outcome.writes.len());
    assert_eq!(EXPECTED_PATH, outcome.writes[0].path);
    assert_eq!(Some(verdict), host.files().get(EXPECTED_PATH));

    // Step 6: the compound tail harvested the (empty) followups and logged.
    assert_eq!(
        vec![
            CompoundRecord::Harvested { value: json!([]) },
            CompoundRecord::Logged {
                id: "qa_architect_cycle".into(),
                destination: "state/evolution/".into(),
                value: verdict.clone(),
            },
        ],
        outcome.compound
    );
}

#[test]
fn a_seeded_trigger_file_overrides_the_default_and_moves_the_write() {
    let host = Arc::new(MemoryHost::frozen());
    host.seed(
        "state/qa-architect/trigger.json",
        json!({
            "kind": "pull_request",
            "repo": "GuitarAlchemist/ix",
            "ref": "feat/ixql-executor",
            "sha": "deadbeef",
            "base_sha": "cafebabe"
        }),
    );

    let outcome = gated_executor(host)
        .run_source(PIPELINE)
        .expect("pipeline runs");

    assert_eq!(
        "state/quality/verdicts/GuitarAlchemist/ix/feat/ixql-executor/\
         2026-01-02T03-04-05Z-pull_request-skeleton-qa-architect-cycle.json",
        outcome.writes[0].path
    );
    assert_eq!(
        json!("deadbeef"),
        outcome.binding("verdict").unwrap()["target"]["sha"]
    );
}

#[test]
fn the_gate_refuses_a_verdict_that_breaks_the_contract() {
    // `risk_tier: "P9"` is not in the enum. The write must fail, and — the
    // point of gating before the host rather than after — nothing may land.
    let tampered = PIPELINE.replace(r#"risk_tier: "P3""#, r#"risk_tier: "P9""#);
    assert_ne!(PIPELINE, tampered, "the tamper target moved; fix the test");

    let host = Arc::new(MemoryHost::frozen());
    let err = gated_executor(host.clone())
        .run_source(&tampered)
        .unwrap_err();

    assert!(err.to_string().contains("violates schema"), "{err}");
    assert!(host.files().is_empty(), "a rejected verdict was persisted");
}

#[test]
fn tars_validate_stops_the_pipeline_with_the_authors_message() {
    // Step 5's guard is `verdict.schema_version == 1`. Break the version and
    // the run must stop with the pipeline's own reject_message. Note the write
    // itself already happened — the guard is a post-write assertion in this
    // pipeline, which is exactly why the schema gate exists separately.
    let tampered = PIPELINE.replace("schema_version: 1,", "schema_version: 2,");
    assert_ne!(PIPELINE, tampered, "the tamper target moved; fix the test");

    let host = Arc::new(MemoryHost::frozen());
    let mut executor = Executor::new(host);
    // No schema registered here: this test is about the in-pipeline guard, not
    // the gate.
    let err = executor
        .run_source(&tampered)
        .expect_err("the guard must reject a v2 verdict");

    assert!(
        err.to_string()
            .contains("Schema mismatch — refusing to persist verdict outside v1."),
        "{err}"
    );
    assert!(executor.schema_gate().is_empty());
}

#[test]
fn a_pipeline_with_an_llm_step_runs_against_the_baml_seam() {
    // The tracer bullet for the BAML half: the same evaluator, a step that
    // crosses into ix-baml, and a deterministic stand-in for the provider.
    let source = r#"
--- Read telemetry, ask the swarm, persist the vote.
telemetry <- ix.io.read("state/dsp/telemetry.json")
  → default({ thd: 0.02, std_dev: 0.7 })

vote <- telemetry
  → baml.EvaluateSignalSwarm()

ix.io.write("state/dsp/vote.json", vote)
  → tars.validate(
      check: "vote.verdict == \"TrueVal\"",
      reject_message: "Swarm did not vote true."
    )
  → compound:
      harvest vote
      promote dsp_bounds
"#;

    let host = Arc::new(MemoryHost::frozen());
    let mut executor = Executor::new(host.clone());
    executor
        .baml_registry()
        .register(Arc::new(FnOperation::new(
            "EvaluateSignalSwarm",
            |input: &Value| {
                // Prove the evaluator handed the LLM step the piped telemetry.
                assert_eq!(json!(0.02), input["thd"]);
                Ok(json!({"verdict": "TrueVal", "confidence": 0.91}))
            },
        )))
        .register(Arc::new(StaticResponse::new("Unused", json!(null))));

    let outcome = executor.run_source(source).expect("pipeline runs");

    assert_eq!(json!(0.91), outcome.binding("vote").unwrap()["confidence"]);
    assert_eq!(1, outcome.writes.len());
    assert_eq!("state/dsp/vote.json", outcome.writes[0].path);
    assert_eq!(
        vec![
            CompoundRecord::Harvested {
                value: json!({"verdict": "TrueVal", "confidence": 0.91}),
            },
            CompoundRecord::Promoted {
                id: "dsp_bounds".into(),
            },
        ],
        outcome.compound
    );
}

#[test]
fn an_llm_step_with_no_registered_function_fails_the_run() {
    // The failure mode this guards against: a pipeline whose LLM step quietly
    // evaluates to null, after which the schema gate happily validates a
    // hollow verdict.
    let host = Arc::new(MemoryHost::frozen());
    let err = Executor::new(host)
        .run_source("t <- { thd: 0.02 }\nv <- t\n  → baml.EvaluateSignalSwarm()")
        .unwrap_err();

    assert!(
        err.to_string().contains("no BAML function registered"),
        "{err}"
    );
}
