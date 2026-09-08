//! Runs the crate's Python contract tests as part of `cargo test`.
//!
//! Why this file exists: `python/test_partition_contract.py` and
//! `python/test_activations_coverage.py` are the only mechanical guards on the
//! trainer's contract behaviour — the split-additivity invariant, the coverage
//! declaration, the guard-ordering AST check, and the declared-vs-observed
//! reconciliation. Nothing in `.github/workflows/` invoked them, and no Rust
//! test reached them, so they were enforced by nobody: a green CI run said
//! nothing about whether the ix #248 guards still worked. That is the
//! green-but-dead failure mode this repo has hit before.
//!
//! CI runs `cargo test --workspace`. Routing the Python suites through a Rust
//! integration test puts them on that path without touching any workflow file.
//!
//! The Python suites are stdlib-only by deliberate design (no torch, numpy,
//! pandas or pyarrow), so any interpreter a runner already has can execute
//! them. The parquet-reading half of `optick_coverage` is exercised at produce
//! time and by `ix-optick-sae verify`, not here.

use std::path::{Path, PathBuf};
use std::process::Command;

const PYTHON_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/python");

/// Suites to run. Both are stdlib-only.
const SUITES: &[&str] = &["test_partition_contract", "test_activations_coverage"];

/// Escape hatch for a machine with no Python at all. Set it and you lose the
/// contract guards for that run — which is why absence alone does not skip.
const SKIP_ENV: &str = "IX_SKIP_PYTHON_CONTRACT_TESTS";

#[test]
fn python_contract_suites_pass() {
    if std::env::var_os(SKIP_ENV).is_some() {
        eprintln!("{SKIP_ENV} set — skipping the Python contract suites.");
        return;
    }

    let python_dir = PathBuf::from(PYTHON_DIR);
    let Some(python) = find_interpreter(&python_dir) else {
        panic!(
            "no working Python interpreter found (tried {:?}).\n\
             \n\
             The trainer's contract guards live in {} and are stdlib-only, so any\n\
             Python 3 will do. Install one, or set {SKIP_ENV}=1 to run the Rust\n\
             tests without them — knowing that skipping leaves the ix #248\n\
             coverage guards unverified for this run.",
            candidates(),
            python_dir.display(),
        );
    };

    for suite in SUITES {
        let output = Command::new(&python)
            .current_dir(&python_dir)
            .args(["-m", "unittest", suite])
            .output()
            .unwrap_or_else(|e| panic!("failed to run {python} -m unittest {suite}: {e}"));

        assert!(
            output.status.success(),
            "Python contract suite `{suite}` failed.\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

/// Interpreter names to try, platform-preferred first.
///
/// Windows ships a `python3.exe` Store stub that fails to execute, so `python`
/// leads there; POSIX follows PEP 394. Both are tried regardless, because a
/// runner may only have the other one.
fn candidates() -> [&'static str; 2] {
    if cfg!(target_os = "windows") {
        ["python", "python3"]
    } else {
        ["python3", "python"]
    }
}

/// The first candidate that can actually import the module under test.
///
/// `--version` is not enough: the Windows Store stub answers it and then fails
/// on real work. Importing `optick_coverage` proves the interpreter can run the
/// suites, so a false positive here cannot turn into a silent skip.
fn find_interpreter(python_dir: &Path) -> Option<String> {
    candidates().into_iter().find_map(|bin| {
        let ok = Command::new(bin)
            .current_dir(python_dir)
            .args(["-c", "import optick_coverage"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        ok.then(|| bin.to_string())
    })
}
