//! The frozen-golden cross-check for the deterministic Pareto frontier
//! pipeline (issue #294).
//!
//! Two independent surfaces compute the same frontier: the Rust primitive
//! `ix_evolution::frontier` and the DuckDB SQL in
//! `crates/ix-duck/sql/pareto_frontier.sql`. Neither is the source of truth for
//! the other — `tests/fixtures/pareto/golden-frontier.csv` is — so drift in
//! either surface shows up here as a byte difference.
//!
//! These tests do NOT need the `duck` feature: they read the same CSV fixture
//! the SQL reads and run the Rust half over it, so they execute on the default
//! `cargo test --workspace` path that CI actually runs.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ix_evolution::frontier::{frontier, to_csv, ObjectiveRow};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pareto")
}

fn repo_root() -> PathBuf {
    // crates/ix-duck -> crates -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("ix-duck lives two levels below the repo root")
        .to_path_buf()
}

/// Read the long-form fixture. Deliberately hand-rolled rather than pulled from
/// a CSV crate: the fixture has no quoting, no embedded separators and no
/// escapes, and a parser dependency would be a second thing that could drift.
fn read_objective_rows() -> Vec<ObjectiveRow> {
    let text = fs::read_to_string(fixture_dir().join("objectives.csv")).expect("fixture readable");
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("subject_revision,task_class,candidate_id,metric,direction,value"),
        "fixture header changed"
    );
    lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let field: Vec<&str> = line.split(',').collect();
            assert_eq!(field.len(), 6, "malformed fixture line: {line}");
            ObjectiveRow::new(
                field[0],
                field[1],
                field[2],
                field[3],
                field[4],
                field[5].parse::<f64>().expect("numeric value"),
            )
        })
        .collect()
}

fn golden() -> String {
    let bytes = fs::read(fixture_dir().join("golden-frontier.csv")).expect("golden readable");
    let text = String::from_utf8(bytes).expect("golden is UTF-8");
    assert!(
        !text.contains('\r'),
        "the golden picked up CR bytes — check the `-text` entry in .gitattributes"
    );
    text
}

#[test]
fn the_rust_primitive_reproduces_the_frozen_golden() {
    let rows = read_objective_rows();
    assert!(rows.len() > 20, "fixture shrank unexpectedly");

    let produced = to_csv(&frontier(&rows).expect("fixture is a valid objective table"));
    assert_eq!(produced, golden());
}

#[test]
fn the_fixture_rows_are_not_already_in_output_order() {
    // Guards the golden test above. If the fixture happened to arrive sorted
    // and grouped, reproducing a sorted golden would say nothing about whether
    // the pipeline imposes an order or merely preserves one.
    let rows = read_objective_rows();
    let mut sorted = rows.clone();
    sorted.sort_by(|left, right| {
        (
            &left.subject_revision,
            &left.task_class,
            &left.candidate_id,
            &left.metric,
        )
            .cmp(&(
                &right.subject_revision,
                &right.task_class,
                &right.candidate_id,
                &right.metric,
            ))
    });
    assert_ne!(
        rows, sorted,
        "the fixture must be scrambled for the golden to prove anything"
    );
}

#[test]
fn the_golden_is_reproduced_after_reversing_the_fixture() {
    let mut rows = read_objective_rows();
    rows.reverse();
    let produced = to_csv(&frontier(&rows).expect("fixture is a valid objective table"));
    assert_eq!(produced, golden());
}

/// Runs the one documented DuckDB CLI command and compares its stdout to the
/// same frozen golden.
///
/// This binding is **opportunistic**: it exercises the SQL surface only where a
/// `duckdb` binary is on PATH. On a machine without one it returns without
/// asserting, so it is NOT evidence that the SQL half is covered by CI — the
/// Rust half above is what CI actually checks. Where duckdb *is* present, a
/// mismatch is a hard failure, never a skip.
#[test]
fn duckdb_cli_reproduces_the_frozen_golden_when_duckdb_is_installed() {
    let script = "crates/ix-duck/sql/pareto_frontier_golden.sql";
    let output = match Command::new("duckdb")
        .current_dir(repo_root())
        .args(["-csv", "-c", &format!(".read {script}")])
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            eprintln!(
                "duckdb CLI not runnable ({error}); the SQL half of #294 was NOT checked by this \
                 run. Reproduce manually from the repo root:\n  \
                 duckdb -csv -c \".read {script}\""
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
        golden(),
        "the DuckDB SQL surface and the frozen golden disagree"
    );
}
