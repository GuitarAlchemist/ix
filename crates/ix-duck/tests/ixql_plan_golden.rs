//! The frozen-golden cross-check for compiled IXQL pipelines (refs #281).
//!
//! Two independent surfaces derive the same execution schedule from the same
//! compiled plan: `ix_pipeline::dag::Dag::parallel_levels` (reached through
//! `ix_ixql::compile`) and the recursive CTE in
//! `crates/ix-duck/sql/ixql_plan.sql`. Neither is the source of truth for the
//! other — `tests/fixtures/ixql/golden-schedule.csv` is — so drift in either
//! surface shows up here as a byte difference.
//!
//! The plan the two read carries no scheduling column, so agreeing on the
//! golden means each side computed the levels rather than echoing them.
//!
//! These tests do NOT need the `duck` feature: the Rust half compiles the same
//! IXQL source the SQL's input was generated from, so it executes on the
//! default `cargo test --workspace` path that CI actually runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

mod common;

fn repo_root() -> PathBuf {
    // crates/ix-duck -> crates -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("ix-duck lives two levels below the repo root")
        .to_path_buf()
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/ixql")
}

/// The IXQL source, read from `ix-ixql`'s fixture rather than copied here.
///
/// One copy on purpose: if the vendored Demerzel pipeline is updated and the
/// plan fixture below is not regenerated, that has to surface as a failure, not
/// as two fixtures quietly disagreeing.
fn ixql_source() -> String {
    let path = repo_root().join("crates/ix-ixql/tests/fixtures/qa-architect-cycle.ixql");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

fn read_fixture(name: &str) -> String {
    let path = fixture_dir().join(name);
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let text = String::from_utf8(bytes).expect("fixture is UTF-8");
    assert!(
        !text.contains('\r'),
        "{name} picked up CR bytes — check the `-text` entry in .gitattributes"
    );
    text
}

fn compiled() -> ix_ixql::CompiledPlan {
    ix_ixql::compile(&ixql_source()).expect("the vendored pipeline compiles")
}

#[test]
fn the_compiler_reproduces_the_frozen_plan() {
    // Compilation regression guard. The plan fixture is the DuckDB side's
    // input, so a change in how IXQL compiles has to be re-frozen here before
    // the SQL half means anything.
    assert_eq!(
        compiled().to_plan_csv(),
        read_fixture("qa-architect-cycle-plan.csv"),
        "the compiler no longer reproduces the frozen plan — regenerate with \
         `cargo run -q -p ix-ixql --bin ixql-plan -- \
         crates/ix-ixql/tests/fixtures/qa-architect-cycle.ixql`"
    );
}

#[test]
fn the_rust_scheduler_reproduces_the_frozen_golden() {
    assert_eq!(compiled().to_schedule_csv(), read_fixture("golden-schedule.csv"));
}

#[test]
fn the_frozen_plan_is_not_already_in_schedule_order() {
    // Guards the golden tests. If the plan happened to arrive grouped by level,
    // reproducing a level-ordered golden would say nothing about whether either
    // surface derives the schedule or merely preserves an order it was handed.
    let plan = compiled();
    let compiled_order: Vec<&str> = plan.stages().iter().map(|s| s.id.as_str()).collect();
    let scheduled_order: Vec<String> = plan.schedule().into_iter().flatten().collect();
    assert_ne!(
        compiled_order, scheduled_order,
        "the fixture must not be pre-sorted by level for the golden to prove anything"
    );
}

#[test]
fn the_plan_fixture_exposes_real_parallelism() {
    // A fully sequential plan would make the schedule a trivial re-listing and
    // the DuckDB recursion untested in the branch that matters.
    let plan = compiled();
    let widest = plan.schedule().iter().map(Vec::len).max().unwrap_or(0);
    assert!(
        widest > 1,
        "the fixture has no independent stages, so the level derivation is untested"
    );
}

/// Runs the one documented DuckDB CLI command and compares its stdout to the
/// same frozen golden.
///
/// The binding is **conditional, not optional**: it runs the SQL surface
/// wherever a `duckdb` binary is reachable, and the `duckdb-sql` job in
/// `.github/workflows/ci.yml` installs a pinned one and sets
/// `IX_REQUIRE_DUCKDB=1`, under which failing to reach the CLI is a failure
/// rather than a skip (ix#294). On a machine without duckdb it still names what
/// went unchecked and returns, so `cargo test --workspace` stays runnable for
/// contributors who have not installed it.
#[test]
fn duckdb_cli_reproduces_the_frozen_golden_when_duckdb_is_installed() {
    let script = "crates/ix-duck/sql/ixql_plan_golden.sql";
    let output = match Command::new("duckdb")
        .current_dir(repo_root())
        .args(["-csv", "-c", &format!(".read {script}")])
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            common::sql_surface_unchecked(
                &error,
                "the SQL half of the IXQL plan schedule",
                &format!("duckdb -csv -c \".read {script}\""),
            );
            return;
        }
    };

    assert!(
        output.status.success(),
        "duckdb exited with {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let produced = String::from_utf8(output.stdout).expect("duckdb output is UTF-8");
    assert_eq!(
        produced.replace("\r\n", "\n"),
        read_fixture("golden-schedule.csv"),
        "the DuckDB SQL surface and the frozen golden disagree"
    );
}

/// The fail-closed guard, driven with a deliberately corrupted plan.
///
/// A validation macro nobody has watched reject anything is not a validation
/// macro. Each seed below is a distinct rule, and every one must stop the
/// schedule from being emitted at all rather than produce a partial answer.
#[test]
fn duckdb_refuses_a_corrupted_plan_when_duckdb_is_installed() {
    let seeds = [
        // (rule, SQL that corrupts the loaded plan, expected violation code)
        (
            "dangling dependency",
            "UPDATE ix_ixql_plan_input SET deps = 'ghost' WHERE stage_id = 'verdict_path';",
            "DanglingDependency",
        ),
        (
            "forward dependency",
            "UPDATE ix_ixql_plan_input SET deps = 's08' WHERE stage_id = 'blast_radius';",
            "ForwardDependency",
        ),
        (
            "unknown kind",
            "UPDATE ix_ixql_plan_input SET kind = 'lambda' WHERE stage_id = 'verdict';",
            "UnknownKind",
        ),
        (
            "duplicate stage id",
            "INSERT INTO ix_ixql_plan_input VALUES (99, 'verdict', 'bind', '', '');",
            "DuplicateStageId",
        ),
    ];

    // `expect`, not a silent `return`: an unavailable temp dir means none of
    // the seeds below ran, and this test reporting success on that basis is
    // the same skip-shaped hole ix#294 closed for the duckdb binding.
    let temp = tempfile::Builder::new()
        .prefix("ixql-plan-seed")
        .tempdir()
        .expect("a temp dir for the corrupted-plan seeds");

    for (rule, corruption, expected_code) in seeds {
        let script = temp.path().join("seed.sql");
        fs::write(
            &script,
            format!(
                "CREATE OR REPLACE TABLE ix_ixql_plan_input AS SELECT * FROM read_csv(\n  \
                 'crates/ix-duck/tests/fixtures/ixql/qa-architect-cycle-plan.csv',\n  \
                 header = true, columns = {{'ordinal': 'BIGINT', 'stage_id': 'VARCHAR', \
                 'kind': 'VARCHAR', 'op': 'VARCHAR', 'deps': 'VARCHAR'}});\n\
                 UPDATE ix_ixql_plan_input SET deps = '' WHERE deps IS NULL;\n\
                 UPDATE ix_ixql_plan_input SET op = '' WHERE op IS NULL;\n\
                 {corruption}\n\
                 .read crates/ix-duck/sql/ixql_plan.sql\n\
                 SELECT * FROM ix_ixql_schedule();\n"
            ),
        )
        .expect("seed script writable");

        let output = match Command::new("duckdb")
            .current_dir(repo_root())
            // DuckDB's `.read` dot-command does not unescape backslashes, so a
            // native Windows path arrives mangled. Forward slashes are accepted
            // on every platform.
            .args([
                "-csv",
                "-c",
                &format!(".read {}", script.display().to_string().replace('\\', "/")),
            ])
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                common::sql_surface_unchecked(
                    &error,
                    "the IXQL plan fail-closed seeds",
                    "cargo test -p ix-duck --test ixql_plan_golden",
                );
                return;
            }
        };

        let combined = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            combined.contains(expected_code),
            "seeding a {rule} did not raise {expected_code}; duckdb said:\n{combined}"
        );
        assert!(
            !combined.contains("level,stage_count,stages"),
            "seeding a {rule} still emitted a schedule — the guard is not fail-closed:\n{combined}"
        );
    }
}
