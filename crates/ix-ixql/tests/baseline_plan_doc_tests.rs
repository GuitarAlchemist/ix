//! Structural regression for the data-only intervention example; no provider or effects.
use ix_ixql::{compile, StageKind};

const SOURCE: &str = include_str!(
    "../../../docs/methodology/baseline/claude-bin-unblock.v0.1.0.ixql"
);

#[test]
fn the_documented_chain_is_data_only_and_deterministic() {
    let first = compile(SOURCE).expect("documented IXQL syntax");
    let second = compile(SOURCE).expect("same source");
    assert_eq!(first.to_plan_csv(), second.to_plan_csv());
    assert_eq!(first.to_schedule_csv(), second.to_schedule_csv());
    assert!(first.stages().iter().all(|s| s.kind == StageKind::Bind && s.op.is_empty()));
    assert_eq!(first.to_plan_csv(), concat!(
        "ordinal,stage_id,kind,op,deps\n",
        "0,intervention,bind,,\n",
        "1,independent_verification,bind,,intervention\n",
        "2,parent_resume,bind,,independent_verification\n",
    ));
    assert_eq!(first.to_schedule_csv(), concat!(
        "level,stage_count,stages\n",
        "0,1,intervention\n",
        "1,1,independent_verification\n",
        "2,1,parent_resume\n",
    ));
}

#[test]
fn a_quoted_requirement_is_not_a_dependency_or_enforced_gate() {
    let changed = SOURCE.replace(
        "verification: independent_verification,",
        "verification: \"independent_verification\",",
    );
    assert_ne!(changed, SOURCE, "mutation must match the actual example");
    let plan = compile(&changed).expect("quoted field remains valid syntax");
    let parent = plan.stages().iter().find(|s| s.id == "parent_resume").unwrap();
    assert!(parent.deps.is_empty(), "text does not establish the dependency");
    assert_ne!(plan.to_schedule_csv(), compile(SOURCE).unwrap().to_schedule_csv());
}
