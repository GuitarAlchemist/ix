use std::collections::BTreeSet;

use std::sync::Arc;

use ix_ixql::{
    Capability, Compiler, EffectAdapter, Executor, MemoryHost, ValueType, VerificationPolicy,
    Verifier,
};
use sha2::{Digest, Sha256};

fn value_digest(value: &serde_json::Value) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("JSON serializes"))
    )
}

fn strict_executor(host: Arc<MemoryHost>) -> (Executor, String) {
    let mut executor = Executor::new(host);
    executor
        .schema_gate()
        .register(
            "state/",
            "strict-test.schema.json",
            &serde_json::json!({"type": "object"}),
        )
        .expect("test schema compiles");
    let digest = executor.schema_gate().identity_digest();
    (executor, digest)
}

#[test]
fn compiler_is_pure_and_emits_typed_capability_metadata() {
    let source = r#"
stamp <- now_utc("yyyy-MM-dd")
config <- ix.io.read("state/config.json")
count <- 3
ix.io.write("state/result.json", { count: count })
"#;

    let first = Compiler::compile(source).expect("source compiles");
    let second = Compiler::compile(source).expect("same source compiles again");

    assert_eq!(first.source_digest(), second.source_digest());
    assert_eq!(64, first.source_digest().len());
    assert_eq!(Some(ValueType::String), first.binding_type("stamp"));
    assert_eq!(Some(ValueType::Number), first.binding_type("count"));
    assert_eq!(4, first.typed_ir().statements().len());
    assert_eq!(2, first.effect_count());
    assert_eq!(
        [
            Capability::Clock,
            Capability::ReadArtifact,
            Capability::WriteArtifact,
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        first.required_capabilities().clone()
    );
}

#[test]
fn verifier_rejects_forbidden_capabilities_and_budget_overruns_before_execution() {
    let program = Compiler::compile(
        r#"
input <- ix.io.read("state/input.json")
ix.io.write("state/output.json", input)
"#,
    )
    .expect("source compiles");
    let policy = VerificationPolicy::new([Capability::ReadArtifact])
        .with_max_statements(1)
        .with_max_effects(1);

    let diagnostics = Verifier::new(policy)
        .verify(program)
        .expect_err("write and budgets must be rejected");
    let codes = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code)
        .collect::<BTreeSet<_>>();

    assert_eq!(
        [
            "IXQL_FORBIDDEN_CAPABILITY",
            "IXQL_STATEMENT_BUDGET",
            "IXQL_EFFECT_BUDGET",
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        codes
    );
}

#[test]
fn verifier_requires_digest_bound_freshness_evidence_for_every_artifact_read() {
    let input = serde_json::json!({"revision": 7});
    let digest = value_digest(&input);
    let program =
        Compiler::compile(r#"input <- ix.io.read("state/./input.json")"#).expect("source compiles");
    let missing = VerificationPolicy::new([Capability::ReadArtifact]).require_fresh_reads();

    let diagnostics = Verifier::new(missing)
        .verify(program.clone())
        .expect_err("unproven freshness must fail closed");
    assert_eq!("IXQL_STALE_OR_UNKNOWN_INPUT", diagnostics.0[0].code);

    let proven = VerificationPolicy::new([Capability::ReadArtifact])
        .require_fresh_reads()
        .with_fresh_artifact("state/input.json", digest);
    let verified = Verifier::new(proven)
        .verify(program)
        .expect("digest-bound fresh input is accepted");

    assert_eq!(1, verified.program().artifact_reads().len());

    let host = Arc::new(MemoryHost::frozen());
    host.seed("state/input.json", input);
    Executor::new(host)
        .plan_verified(&verified)
        .expect("the evidence digest is checked against the value actually read");
}

#[test]
fn freshness_evidence_is_rejected_when_the_read_value_has_changed() {
    let expected = serde_json::json!({"revision": 7});
    let program =
        Compiler::compile(r#"input <- ix.io.read("state/input.json")"#).expect("source compiles");
    let verified = Verifier::new(
        VerificationPolicy::new([Capability::ReadArtifact])
            .require_fresh_reads()
            .with_fresh_artifact("state/input.json", value_digest(&expected)),
    )
    .verify(program)
    .expect("static evidence is well formed");
    let host = Arc::new(MemoryHost::frozen());
    host.seed("state/input.json", serde_json::json!({"revision": 8}));

    let error = Executor::new(host)
        .plan_verified(&verified)
        .expect_err("stale content must fail before a plan exists");
    assert!(error.to_string().contains("freshness"));
}

#[test]
fn validate_predicates_cannot_hide_effects_from_static_verification() {
    let program = Compiler::compile(
        r#"
seed <- { ok: true }
seed -> tars.validate(
  check: "ix.io.write(\"state/pwned.json\", seed, idempotency_key: \"k\", expected_state: \"any\", compensation: \"none\", authority: \"gaia:test\") != null",
  reject_message: "n/a"
)
"#,
    )
    .expect("literal predicate compiles recursively");

    assert!(program
        .required_capabilities()
        .contains(&Capability::WriteArtifact));
    assert_eq!(1, program.effect_count());
    Verifier::new(
        VerificationPolicy::new([Capability::Validate])
            .with_max_effects(0)
            .require_declared_effects(),
    )
    .verify(program)
    .expect_err("hidden writes must be visible to capability and budget checks");
}

#[test]
fn strict_planning_rejects_inline_model_invocation() {
    let program =
        Compiler::compile("answer <- baml.Answer({ prompt: \"hello\" })").expect("source compiles");
    let verified = Verifier::new(VerificationPolicy::new([Capability::InvokeModel]))
        .verify(program)
        .expect("capability is explicitly granted");

    Executor::new(Arc::new(MemoryHost::frozen()))
        .plan_verified(&verified)
        .expect_err("strict planning cannot execute an unreceipted model effect");
}

#[test]
fn compiler_rejects_statically_provable_type_errors() {
    let diagnostics = Compiler::compile("flag <- 1 && 2")
        .expect_err("numeric operands cannot satisfy a boolean operator");

    assert_eq!("IXQL_TYPE_MISMATCH", diagnostics.0[0].code);
}

#[test]
fn strict_effect_policy_rejects_an_undeclared_mutation_contract() {
    let program = Compiler::compile(r#"ix.io.write("state/output.json", { ok: true })"#)
        .expect("source compiles");
    let policy = VerificationPolicy::new([Capability::WriteArtifact])
        .require_declared_effects()
        .allow_authority("gaia:test");

    let diagnostics = Verifier::new(policy)
        .verify(program)
        .expect_err("undeclared mutation metadata must fail closed");
    let codes = diagnostics
        .iter()
        .map(|diagnostic| diagnostic.code)
        .collect::<BTreeSet<_>>();

    assert_eq!(
        [
            "IXQL_EFFECT_AUTHORITY_REQUIRED",
            "IXQL_EFFECT_COMPENSATION_REQUIRED",
            "IXQL_EFFECT_EXPECTED_STATE_REQUIRED",
            "IXQL_EFFECT_IDEMPOTENCY_REQUIRED",
            "IXQL_SCHEMA_GATE_REQUIRED",
        ]
        .into_iter()
        .collect::<BTreeSet<_>>(),
        codes
    );
}

#[test]
fn mutation_authority_must_be_granted_by_policy() {
    let program = Compiler::compile(
        r#"ix.io.write("state/output.json", { ok: true }, idempotency_key: "k", expected_state: "absent", compensation: "delete", authority: "self-granted")"#,
    )
    .expect("source compiles");
    let (_, schema_gate_digest) = strict_executor(Arc::new(MemoryHost::frozen()));
    let diagnostics = Verifier::new(
        VerificationPolicy::new([Capability::WriteArtifact])
            .require_declared_effects()
            .with_schema_gate_digest(schema_gate_digest),
    )
    .verify(program)
    .expect_err("a source claim cannot grant its own authority");

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "IXQL_FORBIDDEN_AUTHORITY"));
}

#[test]
fn authority_allowlist_rejects_a_dynamic_authority_even_without_strict_metadata_mode() {
    let program = Compiler::compile(
        r#"
who <- "self-granted"
ix.io.write(
  "state/output.json", { ok: true },
  idempotency_key: "k", expected_state: "absent",
  compensation: "delete", authority: "{{who}}"
)
"#,
    )
    .expect("source compiles");

    let diagnostics = Verifier::new(
        VerificationPolicy::new([Capability::WriteArtifact]).allow_authority("demerzel-governance"),
    )
    .verify(program)
    .expect_err("an allowlist must reject an authority it cannot prove statically");

    assert!(diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == "IXQL_EFFECT_AUTHORITY_REQUIRED"));
}

#[test]
fn strict_planning_rejects_unreceipted_compound_operations() {
    let program = Compiler::compile(
        r#"
value <- { followups: ["a"] }
value
  → compound:
      harvest value.followups
"#,
    )
    .expect("source compiles");
    let verified = Verifier::new(VerificationPolicy::new([Capability::Compound]))
        .verify(program)
        .expect("capability is explicitly granted");

    Executor::new(Arc::new(MemoryHost::frozen()))
        .plan_verified(&verified)
        .expect_err("strict planning cannot drop an unreceipted compound effect");
}

#[test]
fn planning_requires_the_schema_gate_that_policy_verified() {
    let source = r#"ix.io.write("state/output.json", { ok: true }, idempotency_key: "k", expected_state: "absent", compensation: "delete", authority: "gaia:test")"#;
    let program = Compiler::compile(source).expect("source compiles");
    let (_, schema_gate_digest) = strict_executor(Arc::new(MemoryHost::frozen()));
    let verified = Verifier::new(
        VerificationPolicy::new([Capability::WriteArtifact])
            .require_declared_effects()
            .allow_authority("gaia:test")
            .with_schema_gate_digest(schema_gate_digest),
    )
    .verify(program)
    .expect("policy passes");

    Executor::new(Arc::new(MemoryHost::frozen()))
        .plan_verified(&verified)
        .expect_err("evaluation cannot silently use a different schema gate");
}

#[test]
fn verified_effects_are_staged_then_committed_atomically_and_idempotently() {
    let source = r#"
value <- { ok: true }
ix.io.write(
  "state/output.json",
  value,
  idempotency_key: "run-42/output",
  expected_state: "absent",
  compensation: "delete",
  authority: "gaia:test"
)
"#;
    let program = Compiler::compile(source).expect("source compiles");
    let host = Arc::new(MemoryHost::frozen());
    let (executor, schema_gate_digest) = strict_executor(host.clone());
    let policy = VerificationPolicy::new([Capability::WriteArtifact])
        .require_declared_effects()
        .allow_authority("gaia:test")
        .with_schema_gate_digest(schema_gate_digest);
    let verified = Verifier::new(policy)
        .verify(program)
        .expect("policy passes");
    let plan = executor
        .plan_verified(&verified)
        .expect("evaluation produces an effect plan");

    assert!(host.files().is_empty(), "planning must not persist effects");
    assert_eq!(1, plan.effects().len());

    let first = host.commit(&plan).expect("first commit succeeds");
    let second = host.commit(&plan).expect("idempotent replay succeeds");
    assert_eq!(first, second);
    assert_eq!(
        plan.verification_policy_digest(),
        first.verification_policy_digest
    );
    assert_eq!(
        plan.effects()[0].expected_state,
        first.effects[0].expected_state
    );
    assert_eq!(
        plan.effects()[0].compensation,
        first.effects[0].compensation
    );
    assert_eq!(
        serde_json::json!({"ok": true}),
        host.files()["state/output.json"]
    );
}

#[test]
fn a_failure_after_a_planned_write_persists_nothing() {
    let source = r#"
value <- { schema_version: 2 }
ix.io.write(
  "state/output.json",
  value,
  idempotency_key: "run-43/output",
  expected_state: "absent",
  compensation: "delete",
  authority: "gaia:test"
)
  → tars.validate(
      check: "value.schema_version == 1",
      reject_message: "wrong version"
    )
"#;
    let program = Compiler::compile(source).expect("source compiles");
    let host = Arc::new(MemoryHost::frozen());
    let (executor, schema_gate_digest) = strict_executor(host.clone());
    let policy = VerificationPolicy::new([Capability::WriteArtifact, Capability::Validate])
        .require_declared_effects()
        .allow_authority("gaia:test")
        .with_schema_gate_digest(schema_gate_digest);
    let verified = Verifier::new(policy)
        .verify(program)
        .expect("policy passes");
    executor
        .plan_verified(&verified)
        .expect_err("late validation must reject the whole plan");
    assert!(
        host.files().is_empty(),
        "failed planning must persist nothing"
    );
}

#[test]
fn a_failed_expected_state_rolls_back_the_entire_effect_batch() {
    let source = r#"
ix.io.write(
  "state/first.json", { value: 1 },
  idempotency_key: "batch/first", expected_state: "absent",
  compensation: "delete", authority: "gaia:test"
)
ix.io.write(
  "state/existing.json", { value: 2 },
  idempotency_key: "batch/existing", expected_state: "absent",
  compensation: "restore_previous", authority: "gaia:test"
)
"#;
    let program = Compiler::compile(source).expect("source compiles");
    let host = Arc::new(MemoryHost::frozen());
    let (executor, schema_gate_digest) = strict_executor(host.clone());
    let verified = Verifier::new(
        VerificationPolicy::new([Capability::WriteArtifact])
            .require_declared_effects()
            .allow_authority("gaia:test")
            .with_schema_gate_digest(schema_gate_digest),
    )
    .verify(program)
    .expect("policy passes");
    host.seed("state/existing.json", serde_json::json!({"original": true}));
    let plan = executor
        .plan_verified(&verified)
        .expect("planning succeeds");

    host.commit(&plan)
        .expect_err("second precondition rejects the batch");

    assert!(!host.files().contains_key("state/first.json"));
    assert_eq!(
        serde_json::json!({"original": true}),
        host.files()["state/existing.json"]
    );
}
