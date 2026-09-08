//! Compiling IXQL to a stage DAG.
//!
//! The load-bearing fixture is Demerzel's real `qa-architect-cycle.ixql` — the
//! same vendored copy `ixql_exec_tests.rs` runs. Compiling a synthetic program
//! would prove the compiler handles programs written to suit it; compiling the
//! production pipeline proves it handles the one that exists.
//!
//! The refusal tests are the other half. Issue #281 reserves the element-scope
//! decision for a maintainer, so every spelling the corpus uses for a
//! higher-order call must be *refused*, not guessed at. A test that only
//! checked the happy path would let a future change quietly start compiling
//! them.

use ix_ixql::compile::{CompileError, ELEMENT_SCOPED_FUNCTIONS};
use ix_ixql::{compile, StageKind};

const PIPELINE: &str = include_str!("fixtures/qa-architect-cycle.ixql");

fn plan() -> ix_ixql::CompiledPlan {
    compile(PIPELINE).expect("the real qa-architect-cycle.ixql compiles")
}

fn deps_of(plan: &ix_ixql::CompiledPlan, id: &str) -> Vec<String> {
    plan.stages()
        .iter()
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("no stage {id}; have {:?}", ids(plan)))
        .deps
        .clone()
}

fn ids(plan: &ix_ixql::CompiledPlan) -> Vec<String> {
    plan.stages().iter().map(|s| s.id.clone()).collect()
}

// ------------------------------------------------------- the real pipeline

#[test]
fn the_real_pipeline_compiles_to_twelve_stages() {
    // Nine statements become twelve stages: the two `→` chains expand to one
    // stage per link. If this count moves, the pipeline or the expansion
    // changed and the DuckDB fixture needs regenerating.
    let plan = plan();
    assert_eq!(
        ids(&plan),
        vec![
            "trigger_context~0",
            "trigger_context",
            "blast_radius",
            "reviewer_chain",
            "produced_at_iso",
            "produced_at_safe",
            "verdict_id",
            "verdict",
            "verdict_path",
            "s08~0",
            "s08~1",
            "s08",
        ]
    );
}

#[test]
fn a_pipe_chain_expands_with_the_binding_name_on_the_last_stage() {
    // `trigger_context <- ix.io.read(…) → default(…)`. A later reference to
    // `trigger_context` must mean the value *after* the default was applied,
    // so the binding name belongs to the last link, not the first.
    let plan = plan();
    let head = plan.stages().iter().find(|s| s.id == "trigger_context~0").unwrap();
    assert_eq!(head.kind, StageKind::Source);
    assert_eq!(head.op, "ix.io.read");
    assert!(head.deps.is_empty());

    let tail = plan.stages().iter().find(|s| s.id == "trigger_context").unwrap();
    assert_eq!(tail.kind, StageKind::Pipe);
    assert_eq!(tail.op, "default");
    assert_eq!(tail.deps, vec!["trigger_context~0"]);
}

#[test]
fn record_fields_become_dependency_edges() {
    // `verdict` is built from a record literal mentioning five earlier
    // bindings. Reading those out of the expression is the whole point of
    // compiling: the interpreter never materialises them.
    assert_eq!(
        deps_of(&plan(), "verdict"),
        vec![
            "blast_radius",
            "produced_at_iso",
            "reviewer_chain",
            "trigger_context",
            "verdict_id",
        ]
    );
}

#[test]
fn string_interpolation_becomes_dependency_edges() {
    // verdict_path interpolates `{{verdict.target.repo}}` and `{{verdict_id}}`.
    assert_eq!(deps_of(&plan(), "verdict_path"), vec!["verdict", "verdict_id"]);
    // verdict_id interpolates `{{produced_at_safe}}` and `{{trigger_context.kind}}`.
    assert_eq!(
        deps_of(&plan(), "verdict_id"),
        vec!["produced_at_safe", "trigger_context"]
    );
}

#[test]
fn a_namespaced_callee_is_not_mistaken_for_a_dependency() {
    // `ix.io.read` is Member(Member(Var("ix"), "io"), "read"). The root `ix` is
    // not a binding, so it contributes no edge — namespaces fall out of the
    // binding lookup rather than needing a keyword list. Without this the head
    // stage would depend on a phantom stage called `ix`.
    let plan = plan();
    assert!(
        !ids(&plan).contains(&"ix".to_string()),
        "a namespace segment leaked in as a stage"
    );
    assert!(deps_of(&plan, "trigger_context~0").is_empty());
}

#[test]
fn a_validate_step_whose_check_is_a_string_gains_no_edge_from_it() {
    // `tars.validate(check: "verdict.schema_version == 1")` mentions `verdict`
    // inside a *string literal*. That is not an expression reference, and
    // treating it as one would invent an edge the source does not have.
    let deps = deps_of(&plan(), "s08~1");
    assert_eq!(deps, vec!["s08~0"], "only the previous link in the chain");
}

#[test]
fn the_compound_block_depends_on_what_it_harvests() {
    // `→ compound: harvest verdict.followups` — the harvested expression is a
    // real reference and must produce an edge.
    let deps = deps_of(&plan(), "s08");
    assert!(deps.contains(&"verdict".to_string()), "got {deps:?}");
    assert!(deps.contains(&"s08~1".to_string()), "got {deps:?}");
}

#[test]
fn independent_bindings_land_on_the_same_parallel_level() {
    // The five stages with no upstream can run together; everything after the
    // fan-in is sequential. This is the answer the interpreter cannot give.
    let plan = plan();
    let schedule = plan.schedule();
    assert_eq!(
        schedule[0],
        vec![
            "blast_radius",
            "produced_at_iso",
            "produced_at_safe",
            "reviewer_chain",
            "trigger_context~0",
        ]
    );
    assert_eq!(plan.level_of("verdict_id"), Some(2));
    assert_eq!(plan.level_of("verdict"), Some(3));
    assert_eq!(schedule.len(), 8, "eight levels: {schedule:?}");
}

#[test]
fn every_dependency_points_backwards_in_compiled_order() {
    // The acyclicity invariant, checked directly rather than assumed. It is
    // also what lets the DuckDB recursive CTE terminate, so the SQL side
    // validates the same rule.
    let plan = plan();
    let ordinal = |id: &str| {
        plan.stages()
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.ordinal)
            .unwrap_or_else(|| panic!("dangling dependency: {id}"))
    };
    for stage in plan.stages() {
        for dep in &stage.deps {
            assert!(
                ordinal(dep) < stage.ordinal,
                "{} depends on {dep}, which is not earlier",
                stage.id
            );
        }
    }
}

#[test]
fn the_topological_order_covers_every_stage_exactly_once() {
    let plan = plan();
    let mut topo = plan.topological_order().to_vec();
    let mut all = ids(&plan);
    topo.sort();
    all.sort();
    assert_eq!(topo, all);
}

// ------------------------------------------------- issue #281 is not decided

#[test]
fn a_lambda_argument_is_refused_rather_than_compiled() {
    let error = compile("xs <- items\n  → map(x => x + 1)")
        .expect_err("map takes an element-scoped argument");
    // The callee is named in preference to the binder: `map` tells the reader
    // which semantics is missing, `x` does not.
    match error {
        CompileError::RequiresElementScope { function, .. } => assert_eq!(function, "map"),
        other => panic!("expected RequiresElementScope, got {other:?}"),
    }
    assert!(
        error_text("xs <- items\n  → map(x => x + 1)").contains("#281"),
        "the refusal must point at the issue that owns the decision"
    );
}

#[test]
fn a_binder_less_element_predicate_is_refused_too() {
    // `group.filter(signal_type == "pain")` has no lambda at all — the
    // predicate reads fields off the element with no binder. Checking only for
    // lambdas would let this one through, and it is 9 of the 88 corpus sites.
    let error =
        compile("ys <- group.filter(signal_type == \"pain\")").expect_err("element scope");
    match error {
        CompileError::RequiresElementScope { function, .. } => assert_eq!(function, "filter"),
        other => panic!("expected RequiresElementScope, got {other:?}"),
    }
}

#[test]
fn method_form_higher_order_calls_are_refused() {
    // 66 of the 88 corpus sites are method-form (`nodes.map(node => …)`).
    let error = compile("zs <- nodes.map(node => node.id)").expect_err("element scope");
    match error {
        CompileError::RequiresElementScope { function, .. } => assert_eq!(function, "map"),
        other => panic!("expected RequiresElementScope, got {other:?}"),
    }
}

#[test]
fn every_element_scoped_function_is_refused_in_both_spellings() {
    for function in ELEMENT_SCOPED_FUNCTIONS {
        for source in [
            format!("out <- xs.{function}(p => p)"),
            format!("out <- xs\n  → {function}(p => p)"),
            format!("out <- {function}(field == 1)"),
        ] {
            let error = compile(&source)
                .err()
                .unwrap_or_else(|| panic!("{source:?} compiled but should be refused"));
            assert!(
                matches!(
                    error,
                    CompileError::RequiresElementScope { .. }
                        | CompileError::LambdaRequiresElementScope { .. }
                ),
                "{source:?} produced the wrong error: {error}"
            );
        }
    }
}

#[test]
fn a_lambda_under_an_unrecognised_callee_is_still_refused() {
    // The callee list cannot be exhaustive — a host could add another
    // higher-order function tomorrow. The lambda check is the backstop.
    let error = compile("out <- custom_fold(acc => acc)").expect_err("lambda present");
    match error {
        CompileError::LambdaRequiresElementScope { params, .. } => assert_eq!(params, "acc"),
        other => panic!("expected LambdaRequiresElementScope, got {other:?}"),
    }
}

#[test]
fn a_lambda_nested_deep_inside_a_record_is_still_refused() {
    // A refusal that only looked at the top of the expression would miss this.
    let error = compile("out <- { a: [ { b: xs.map(x => x) } ] }").expect_err("nested");
    assert!(matches!(
        error,
        CompileError::RequiresElementScope { .. } | CompileError::LambdaRequiresElementScope { .. }
    ));
}

#[test]
fn an_ordinary_method_call_still_compiles() {
    // The refusal must be scoped to the six element-scoped names, not to
    // method syntax in general — otherwise it would be a much larger claim
    // about the language than #281 makes.
    let plan = compile("out <- a.b()").expect("a.b() is not element-scoped");
    assert_eq!(plan.stages()[0].op, "a.b");
}

// ------------------------------------------------------------- emission

#[test]
fn the_plan_relation_needs_no_csv_quoting() {
    // The emitter is hand-rolled, like the Pareto fixture reader. That is only
    // safe while no field can contain a comma, a quote or a newline.
    let plan = plan();
    for stage in plan.stages() {
        for field in [&stage.id, &stage.op, &stage.deps.join(";")] {
            assert!(
                !field.contains(',') && !field.contains('"') && !field.contains('\n'),
                "field {field:?} would need CSV quoting"
            );
        }
    }
}

#[test]
fn the_plan_relation_carries_no_scheduling_column() {
    // The DuckDB side has to *derive* the schedule from the structure. If a
    // level column leaked into the plan the SQL could echo it, and the
    // cross-surface golden would stop proving anything.
    let header = plan().to_plan_csv().lines().next().unwrap().to_string();
    assert_eq!(header, "ordinal,stage_id,kind,op,deps");
}

#[test]
fn compilation_is_deterministic() {
    assert_eq!(plan().to_plan_csv(), plan().to_plan_csv());
    assert_eq!(plan().to_schedule_csv(), plan().to_schedule_csv());
}

#[test]
fn the_plan_renders_as_a_shape_valid_pipeline_spec() {
    // It is the workspace's canonical pipeline IR and round-trips through
    // YAML. It is NOT executable — see the module docs: the `skill` fields are
    // IXQL host functions, which `ix_pipeline::lower` would reject.
    let spec = plan().to_pipeline_spec();
    spec.validate_shape().expect("shape is valid");
    assert_eq!(spec.stages.len(), 12);
    let yaml = spec.to_yaml_string().expect("serialises");
    let round_tripped =
        ix_pipeline::spec::PipelineSpec::from_yaml_str(&yaml).expect("round-trips");
    assert_eq!(round_tripped.stages.len(), 12);
}

#[test]
fn a_rebound_name_gets_its_own_stage() {
    // IXQL does not forbid rebinding. Reusing the id would silently drop the
    // first stage from the DAG.
    let plan = compile("a <- 1\na <- 2").expect("rebinding compiles");
    assert_eq!(ids(&plan), vec!["a", "a#2"]);
}

#[test]
fn an_empty_program_compiles_to_an_empty_plan() {
    let plan = compile("").expect("empty program");
    assert!(plan.stages().is_empty());
    assert_eq!(plan.to_plan_csv(), "ordinal,stage_id,kind,op,deps\n");
}

fn error_text(source: &str) -> String {
    compile(source).expect_err("expected refusal").to_string()
}
