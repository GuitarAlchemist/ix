//! The shared DuckDB CLI binding for `ix-duck`'s frozen-golden cross-checks.
//!
//! Every golden test in this directory covers two independent surfaces: a Rust
//! one that CI always runs, and a SQL one that needs the `duckdb` CLI on PATH.
//! Where the binary is absent the SQL half cannot run at all, and the whole
//! question is what that absence should mean. Both simple answers are wrong:
//!
//! * **Always skip** — what these tests did before ix#294 — reports the SQL
//!   surface as a passing test in `cargo test --workspace` while it executes
//!   nothing. That is the same class of invisibility as ix#315: not a test that
//!   fails, a test that *cannot* fail.
//! * **Always fail** would break `cargo test --workspace` for every contributor
//!   without duckdb installed, and for all four legs of the CI build matrix,
//!   none of which has a reason to carry a DuckDB install.
//!
//! So the requirement is made explicit instead of guessed. The `duckdb-sql` job
//! in `.github/workflows/ci.yml` installs a pinned CLI and sets
//! `IX_REQUIRE_DUCKDB=1`; under that variable an unreachable binary is a hard
//! failure. Everywhere else it stays a loud, self-describing skip.

use std::io;

/// The variable the `duckdb-sql` CI job sets to make the SQL surfaces mandatory.
pub const REQUIRE_VAR: &str = "IX_REQUIRE_DUCKDB";

/// True when the caller is running somewhere that must reach the duckdb CLI.
///
/// `0` and the empty string read as "not required", so the variable can be
/// switched off in a shell without unsetting it.
fn duckdb_is_required() -> bool {
    std::env::var_os(REQUIRE_VAR).is_some_and(|value| !value.is_empty() && value != "0")
}

/// Decides what an unrunnable `duckdb` means for the calling test.
///
/// Panics where the CLI is required, because there, failing to reach duckdb is
/// the job's entire purpose going unmet — the job must not be able to go green
/// by doing nothing. Everywhere else it names the surface that went unchecked
/// and the command that checks it by hand, and the caller returns.
pub fn sql_surface_unchecked(error: &io::Error, surface: &str, reproduce: &str) {
    let report = format!(
        "duckdb CLI not runnable ({error}); {surface} was NOT checked by this run.\n\
         Reproduce from the repo root:\n  {reproduce}"
    );
    assert!(
        !duckdb_is_required(),
        "{REQUIRE_VAR} is set, so an unreachable duckdb CLI is a failure, not a skip.\n{report}"
    );
    eprintln!("{report}");
}
