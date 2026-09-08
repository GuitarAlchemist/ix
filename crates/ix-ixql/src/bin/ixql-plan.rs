//! Compile an `.ixql` file and print the compiled plan, or its schedule.
//!
//! This is how the DuckDB-side fixture is regenerated. It writes to stdout and
//! touches nothing else, so the artifact it produces is reproducible with one
//! command from the repository root:
//!
//! ```text
//! cargo run -q -p ix-ixql --bin ixql-plan -- \
//!   crates/ix-ixql/tests/fixtures/qa-architect-cycle.ixql \
//!   > crates/ix-duck/tests/fixtures/ixql/qa-architect-cycle-plan.csv
//! ```
//!
//! `--schedule` prints the derived parallel levels instead. That is the output
//! the DuckDB macros have to reproduce from the plan alone.

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    let mut schedule = false;
    let mut spec = false;

    for argument in args.by_ref() {
        match argument.as_str() {
            "--schedule" => schedule = true,
            "--spec" => spec = true,
            "-h" | "--help" => {
                eprintln!("usage: ixql-plan [--schedule|--spec] <file.ixql>");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown flag: {other}");
                return ExitCode::from(64);
            }
            other => path = Some(other.to_string()),
        }
    }

    let Some(path) = path else {
        eprintln!("usage: ixql-plan [--schedule|--spec] <file.ixql>");
        return ExitCode::from(64);
    };

    let source = match std::fs::read_to_string(&path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("reading {path}: {error}");
            return ExitCode::from(66);
        }
    };

    let plan = match ix_ixql::compile(&source) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("{path}: {error}");
            return ExitCode::FAILURE;
        }
    };

    if spec {
        match plan.to_pipeline_spec().to_yaml_string() {
            Ok(yaml) => print!("{yaml}"),
            Err(error) => {
                eprintln!("rendering spec: {error}");
                return ExitCode::FAILURE;
            }
        }
    } else if schedule {
        print!("{}", plan.to_schedule_csv());
    } else {
        print!("{}", plan.to_plan_csv());
    }
    ExitCode::SUCCESS
}
