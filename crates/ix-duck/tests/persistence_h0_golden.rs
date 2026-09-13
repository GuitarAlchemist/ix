//! The frozen-golden cross-check for H0 persistent homology over a DuckDB
//! column (gap-matrix row B1b).
//!
//! Two independent surfaces compute the same diagram:
//!
//! * the **general engine** — `ix_topo::simplex::rips_complex` +
//!   `ix_topo::persistence::compute_persistence`, i.e. an O(m^2)-ish
//!   boundary-matrix reduction over Z/2 that knows nothing about the input
//!   being one-dimensional; and
//! * the **SQL closed form** in `crates/ix-duck/sql/persistence_h0.sql`, which
//!   exploits the fact that the MST of a 1-D point set is the path through its
//!   sorted values and reads the H0 deaths straight off a `lag()` window.
//!
//! Neither is the source of truth for the other —
//! `tests/fixtures/persistence/golden-h0.csv` is — so drift in either surface
//! shows up here as a byte difference. The two derivations share no code and no
//! algorithm, which is what makes the agreement worth anything.
//!
//! These tests do NOT need the `duck` feature: they read the same CSV fixture
//! the SQL reads and run the engine over it, so they execute on the default
//! `cargo test --workspace` path that CI actually runs.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ix_topo::persistence::compute_persistence;
use ix_topo::simplex::rips_complex;

mod common;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/persistence")
}

fn repo_root() -> PathBuf {
    // crates/ix-duck -> crates -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("ix-duck lives two levels below the repo root")
        .to_path_buf()
}

/// Read the fixture into `series -> values`, preserving duplicates. Hand-rolled
/// for the same reason as the Pareto fixture reader: two fields, no quoting, no
/// escapes, and a CSV crate would be a second thing that could drift.
///
/// `BTreeMap` gives the series the same lexicographic grouping DuckDB's
/// `ORDER BY series` produces.
fn read_series() -> BTreeMap<String, Vec<f64>> {
    let text = fs::read_to_string(fixture_dir().join("series.csv")).expect("fixture readable");
    let mut lines = text.lines();
    assert_eq!(
        lines.next().map(str::trim),
        Some("series,value"),
        "fixture header changed"
    );
    let mut out: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let field: Vec<&str> = line.trim().split(',').collect();
        assert_eq!(field.len(), 2, "malformed fixture line: {line}");
        out.entry(field[0].to_string())
            .or_default()
            .push(field[1].parse::<f64>().expect("numeric value"));
    }
    out
}

fn golden() -> String {
    let bytes = fs::read(fixture_dir().join("golden-h0.csv")).expect("golden readable");
    let text = String::from_utf8(bytes).expect("golden is UTF-8");
    assert!(
        !text.contains('\r'),
        "the golden picked up CR bytes — check the `-text` entry in .gitattributes"
    );
    text
}

/// H0 (birth, death) pairs for a 1-D point cloud, straight from the general
/// engine. Points are lifted to `Vec<Vec<f64>>` because `rips_complex` is
/// dimension-agnostic; over R^1 its Euclidean distance is `|a - b|`.
///
/// `max_radius` is the full range, which is exactly the largest pairwise
/// distance, so every edge enters the filtration and no H0 class is left
/// spuriously essential by truncation.
fn engine_h0(values: &[f64]) -> Vec<(f64, f64)> {
    let points: Vec<Vec<f64>> = values.iter().map(|v| vec![*v]).collect();
    let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let stream = rips_complex(&points, 1, hi - lo);

    let mut pairs: Vec<(f64, f64)> = compute_persistence(&stream)
        .into_iter()
        .filter(|diagram| diagram.dimension == 0)
        .flat_map(|diagram| diagram.pairs)
        .collect();
    // Death ascending, infinity last. The multiset is what the diagram is; the
    // order is imposed here exactly as the SQL imposes it.
    pairs.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .expect("no NaN in a persistence diagram")
    });
    pairs
}

/// Render the engine's diagrams in the CSV shape `ix_persistence_h0()` emits.
fn engine_csv() -> String {
    let mut out = String::from("series,ordinal,dim,birth,death\n");
    for (series, values) in read_series() {
        for (i, (birth, death)) in engine_h0(&values).into_iter().enumerate() {
            let death = if death.is_finite() {
                format!("{death:.6}")
            } else {
                "inf".to_string()
            };
            out.push_str(&format!("{series},{},0,{birth:.6},{death}\n", i + 1));
        }
    }
    out
}

#[test]
fn the_general_engine_reproduces_the_frozen_golden() {
    let series = read_series();
    assert!(series.len() >= 5, "fixture shrank unexpectedly");
    assert_eq!(engine_csv(), golden());
}

#[test]
fn the_fixture_rows_are_not_already_in_output_order() {
    // Guards the golden test above. If the fixture happened to arrive grouped by
    // series and sorted by value, reproducing a grouped-and-sorted golden would
    // say nothing about whether the pipeline imposes an order or merely
    // preserves one.
    let text = fs::read_to_string(fixture_dir().join("series.csv")).expect("fixture readable");
    let rows: Vec<&str> = text
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .collect();
    let mut sorted = rows.clone();
    sorted.sort_by_key(|line| {
        let (series, value) = line.trim().split_once(',').expect("two fields");
        (
            series.to_string(),
            value.parse::<f64>().expect("numeric").to_bits(),
        )
    });
    assert_ne!(
        rows, sorted,
        "the fixture must be scrambled for the golden to prove anything"
    );
}

#[test]
fn shuffling_the_fixture_does_not_move_the_diagram() {
    // A persistence diagram is a multiset: input order is not part of it. Feed
    // the engine each series reversed and the rendered output must not budge.
    let mut out = String::from("series,ordinal,dim,birth,death\n");
    for (series, mut values) in read_series() {
        values.reverse();
        for (i, (birth, death)) in engine_h0(&values).into_iter().enumerate() {
            let death = if death.is_finite() {
                format!("{death:.6}")
            } else {
                "inf".to_string()
            };
            out.push_str(&format!("{series},{},0,{birth:.6},{death}\n", i + 1));
        }
    }
    assert_eq!(out, golden());
}

/// The closed form the SQL relies on, asserted directly against the engine.
///
/// This is the load-bearing claim of `persistence_h0.sql`: over R^1 the finite
/// H0 deaths are the consecutive gaps of the sorted *distinct* values. If the
/// engine ever stops agreeing, the SQL is wrong even if the golden was
/// regenerated from it, so this test is checked against freshly computed gaps
/// rather than against the frozen file.
#[test]
fn engine_h0_deaths_are_the_consecutive_gaps_of_the_sorted_values() {
    for (series, values) in read_series() {
        let mut distinct: Vec<f64> = values.clone();
        distinct.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
        distinct.dedup();
        // A diagram is a multiset, so compare the gaps in the same ascending
        // order `engine_h0` imposes rather than in sorted-value order.
        let mut expected_finite: Vec<f64> = distinct.windows(2).map(|w| w[1] - w[0]).collect();
        expected_finite.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));

        let produced = engine_h0(&values);
        let (finite, essential): (Vec<_>, Vec<_>) =
            produced.iter().partition(|(_, d)| d.is_finite());

        assert_eq!(
            essential.len(),
            1,
            "{series}: a point cloud has exactly one essential H0 class"
        );
        assert_eq!(
            finite.len(),
            expected_finite.len(),
            "{series}: one finite H0 class per gap between distinct values"
        );
        for ((birth, death), gap) in finite.iter().zip(expected_finite.iter()) {
            assert_eq!(*birth, 0.0, "{series}: H0 classes are all born at radius 0");
            // Six decimals, not bit equality: the engine reaches the gap through
            // `sqrt((a-b)^2)`, the SQL through `a - b`, and those can differ in
            // the last ulp. Six decimals is the contract the golden renders at.
            assert_eq!(
                format!("{death:.6}"),
                format!("{gap:.6}"),
                "{series}: H0 death does not match the sorted gap"
            );
        }
    }
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
    let script = "crates/ix-duck/sql/persistence_h0_golden.sql";
    let output = match Command::new("duckdb")
        .current_dir(repo_root())
        .args(["-csv", "-c", &format!(".read {script}")])
        .output()
    {
        Ok(output) => output,
        Err(error) => {
            common::sql_surface_unchecked(
                &error,
                "the SQL half of B1b",
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
        golden(),
        "the DuckDB SQL surface and the frozen golden disagree"
    );
}

/// The validation macro must fail closed. Same conditional binding as above.
///
/// Each case feeds `ix_topo_input` something the two surfaces would disagree
/// about and asserts that `ix_persistence_h0()` raises instead of emitting a
/// diagram — an empty result would be the dangerous outcome, not an error.
#[test]
fn the_sql_guard_rejects_bad_input_when_duckdb_is_installed() {
    let macros = "crates/ix-duck/sql/persistence_h0.sql";
    let cases: [(&str, &str, &str); 4] = [
        (
            "EmptyInput",
            "CREATE TABLE ix_topo_input(series VARCHAR, value DOUBLE);",
            "EmptyInput",
        ),
        (
            "EmptyField",
            "CREATE TABLE ix_topo_input AS SELECT '' AS series, 1.0::DOUBLE AS value;",
            "EmptyField",
        ),
        (
            "NonFiniteValue",
            "CREATE TABLE ix_topo_input AS SELECT 's' AS series, 'NaN'::DOUBLE AS value;",
            "NonFiniteValue",
        ),
        (
            "NearDuplicateValue",
            "CREATE TABLE ix_topo_input AS SELECT 's' AS series, 1.0::DOUBLE AS value \
             UNION ALL SELECT 's', 1.0000000001::DOUBLE;",
            "NearDuplicateValue",
        ),
    ];

    for (name, setup, expected_code) in cases {
        // Three separate `-c` arguments, not one script: a DuckDB CLI dot
        // command has to be the whole of its own `-c`, and the input table must
        // exist before the macro file is read (table macros bind at CREATE).
        let output = match Command::new("duckdb")
            .current_dir(repo_root())
            .args([
                "-csv",
                "-c",
                setup,
                "-c",
                &format!(".read {macros}"),
                "-c",
                "SELECT * FROM ix_persistence_h0();",
            ])
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                common::sql_surface_unchecked(
                    &error,
                    &format!("the SQL guard case {name}"),
                    &format!("duckdb -csv -c \".read {macros}\""),
                );
                return;
            }
        };
        assert!(
            !output.status.success(),
            "{name}: the guard let bad input through (stdout: {})",
            String::from_utf8_lossy(&output.stdout)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected_code),
            "{name}: expected violation code {expected_code} in stderr, got: {stderr}"
        );
    }
}
