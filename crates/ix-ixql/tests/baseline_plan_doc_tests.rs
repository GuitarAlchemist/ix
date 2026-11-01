//! Structural and real-CLI regression for the data-only example; no providers.
use ix_ixql::{compile, StageKind};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const SOURCE: &str =
    include_str!("../../../docs/methodology/baseline/claude-bin-unblock.v0.1.0.ixql");
const PLAN: &str = concat!(
    "ordinal,stage_id,kind,op,deps\n",
    "0,intervention,bind,,\n",
    "1,independent_verification,bind,,intervention\n",
    "2,parent_resume,bind,,independent_verification\n",
);
const SCHEDULE: &str = concat!(
    "level,stage_count,stages\n",
    "0,1,intervention\n",
    "1,1,independent_verification\n",
    "2,1,parent_resume\n",
);

#[test]
fn the_documented_chain_is_data_only_and_deterministic() {
    let first = compile(SOURCE).expect("documented IXQL syntax");
    let second = compile(SOURCE).expect("same source");
    assert_eq!(first.to_plan_csv(), second.to_plan_csv());
    assert_eq!(first.to_schedule_csv(), second.to_schedule_csv());
    assert!(first
        .stages()
        .iter()
        .all(|s| s.kind == StageKind::Bind && s.op.is_empty()));
    assert_eq!(first.to_plan_csv(), PLAN);
    assert_eq!(first.to_schedule_csv(), SCHEDULE);
}

#[test]
fn a_quoted_requirement_is_not_a_dependency_or_enforced_gate() {
    let changed = SOURCE.replace(
        "verification: independent_verification,",
        "verification: \"independent_verification\",",
    );
    assert_ne!(changed, SOURCE, "mutation must match the actual example");
    let plan = compile(&changed).expect("quoted field remains valid syntax");
    let parent = plan
        .stages()
        .iter()
        .find(|s| s.id == "parent_resume")
        .unwrap();
    assert!(parent.deps.is_empty(), "text does not establish the dependency");
    assert_ne!(
        plan.to_schedule_csv(),
        compile(SOURCE).unwrap().to_schedule_csv()
    );
}

#[test]
fn the_real_cli_prints_the_documented_plan_and_schedule() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/methodology/baseline/claude-bin-unblock.v0.1.0.ixql");
    for (schedule, expected) in [(false, PLAN), (true, SCHEDULE)] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ixql-plan"));
        if schedule {
            command.arg("--schedule");
        }
        let output = command.arg(&fixture).output().expect("run ixql-plan");
        assert!(
            output.status.success(),
            "CLI failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected.as_bytes());
        assert!(output.stderr.is_empty());
    }
}

// An atomically created private directory avoids shell/path quoting and works on
// Windows too. No file handles remain open when the CLI reads or cleanup runs.
struct TemporarySource(PathBuf);

impl TemporarySource {
    fn invalid() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "ixql-baseline-invalid-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).expect("create isolated fixture directory");
        let fixture = Self(directory);
        std::fs::write(fixture.path(), "intervention <- {").expect("write invalid IXQL");
        fixture
    }

    fn path(&self) -> PathBuf {
        self.0.join("invalid.ixql")
    }
}

impl Drop for TemporarySource {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn the_real_cli_rejects_invalid_source_without_a_plan() {
    let fixture = TemporarySource::invalid();
    let output = Command::new(env!("CARGO_BIN_EXE_ixql-plan"))
        .arg(fixture.path())
        .output()
        .expect("run ixql-plan on invalid syntax");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty(), "invalid input must emit no plan");
    let error = String::from_utf8(output.stderr).expect("UTF-8 diagnostic");
    assert!(error.contains("invalid.ixql"), "diagnostic identifies input");
    assert!(error.contains("parse error"), "diagnostic identifies rejection");
}
