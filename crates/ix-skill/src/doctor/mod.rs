//! `ix doctor` — the single pre-PR gate for IX's repo-specific surfaces.
//!
//! IX accumulated a lot of guardrails: a link-time capability registry, an MCP
//! tool surface, stable-surface hashes, `@ai:` annotation drift, governance
//! artifacts. Each is cheap on its own and expensive to *remember*. ix#185
//! asked for one command that runs them and says what to do about a failure.
//!
//! This is a convenience wrapper, not a new authority layer. Every check here
//! either reads an existing artifact or shells out to the existing gate; none
//! of them invent a new source of truth.
//!
//! ```text
//! ix doctor            # fast, in-process surface checks
//! ix doctor --write    # regenerate the registry snapshot, then re-check
//! ix doctor --full     # the above plus the CI clippy + test invocation
//! ```
//!
//! Exit codes follow the repo's hexavalent convention (see [`crate::exit`]):
//! `0` all green, `1` warnings only, `4` at least one failure.

pub mod dark_features;
pub mod orphan_traits;
pub mod registry_snapshot;

use crate::exit;
use crate::output::{self, Format};
use serde::Serialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// MCP tools that exist only behind a non-default cargo feature.
///
/// `maintain-gate` pulls bundled DuckDB and is off in the default/CI build, so
/// its tool must not count as drift either way.
const FEATURE_GATED_TOOLS: &[&str] = &["ix_maintain_gate"];

/// Outcome of a single check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Nothing to report.
    Ok,
    /// Non-blocking: worth a look, does not fail the gate.
    Warn,
    /// Blocking.
    Fail,
    /// Not run in this mode.
    Skip,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Warn => "warn",
            Status::Fail => "FAIL",
            Status::Skip => "skip",
        }
    }
}

/// One check's result, including what to do about it.
#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    /// Stable identifier, e.g. `registry-snapshot`.
    pub name: String,
    /// Pass/fail.
    pub status: Status,
    /// One line describing what was found.
    pub summary: String,
    /// What the contributor should do next. Present whenever `status` is not
    /// `Ok` — the issue asked for actionable messages, not raw test noise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remedy: Option<String>,
    /// Machine-readable detail for whoever is scripting against this.
    #[serde(skip_serializing_if = "Value::is_null")]
    pub details: Value,
}

impl CheckResult {
    fn new(name: &str, status: Status, summary: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            status,
            summary: summary.into(),
            remedy: None,
            details: Value::Null,
        }
    }

    fn with_remedy(mut self, remedy: impl Into<String>) -> Self {
        self.remedy = Some(remedy.into());
        self
    }

    fn with_details(mut self, details: Value) -> Self {
        self.details = details;
        self
    }
}

/// The full run.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// Repo root the checks ran against.
    pub root: String,
    /// Every check, in run order.
    pub checks: Vec<CheckResult>,
}

impl Report {
    /// Highest severity across all checks.
    pub fn worst(&self) -> Status {
        if self.checks.iter().any(|c| c.status == Status::Fail) {
            Status::Fail
        } else if self.checks.iter().any(|c| c.status == Status::Warn) {
            Status::Warn
        } else {
            Status::Ok
        }
    }

    /// Hexavalent verdict letter: `T` green, `P` warnings, `F` failures.
    pub fn verdict(&self) -> &'static str {
        match self.worst() {
            Status::Fail => "F",
            Status::Warn => "P",
            _ => "T",
        }
    }

    /// Process exit code for this report.
    pub fn exit_code(&self) -> i32 {
        match self.worst() {
            Status::Fail => exit::FALSE,
            Status::Warn => exit::PROBABLE,
            _ => exit::OK_TRUE,
        }
    }
}

/// What to run.
#[derive(Debug, Clone, Copy, Default)]
pub struct Options {
    /// Regenerate the registry snapshot before checking it.
    pub write: bool,
    /// Also run the CI clippy + test invocation.
    pub full: bool,
}

/// Locate the workspace root by walking up from `start` looking for the
/// virtual manifest that owns `crates/`.
pub fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        if d.join("Cargo.toml").is_file() && d.join("crates").is_dir() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

/// Run the in-process checks against `root`.
///
/// `--full` shell-outs are handled by [`run_full_checks`] so this stays fast
/// and side-effect-free enough to call from tests.
pub fn run(root: &Path, opts: Options) -> Report {
    let mut checks = Vec::new();
    checks.push(check_registry_snapshot(root, opts.write));
    checks.push(check_orphan_traits(root));
    checks.push(check_dark_features(root));
    checks.extend(check_environment(root));
    Report {
        root: root.display().to_string(),
        checks,
    }
}

/// The registry-snapshot check: live inventory vs `skills.snapshot.json`.
fn check_registry_snapshot(root: &Path, write: bool) -> CheckResult {
    const NAME: &str = "registry-snapshot";
    let live = registry_snapshot::live();
    let gated: Vec<String> = FEATURE_GATED_TOOLS.iter().map(|s| s.to_string()).collect();

    if write {
        let snap = registry_snapshot::snapshot_of(&live, gated);
        return match snap.write(root) {
            Ok(()) => CheckResult::new(
                NAME,
                Status::Ok,
                format!(
                    "wrote {} — {} skills, {} MCP tools",
                    registry_snapshot::SNAPSHOT_PATH,
                    snap.skills.count,
                    snap.mcp_tools.count
                ),
            ),
            Err(e) => CheckResult::new(NAME, Status::Fail, e)
                .with_remedy("check filesystem permissions on state/registry/"),
        };
    }

    let snapshot = match registry_snapshot::Snapshot::load(root) {
        Ok(s) => s,
        Err(e) => {
            return CheckResult::new(NAME, Status::Fail, e).with_remedy(
                "run `cargo run -p ix-skill --bin ix -- doctor --write` and commit the snapshot",
            )
        }
    };

    let drift = registry_snapshot::diff(&snapshot, &live);
    if drift.is_clean() {
        return CheckResult::new(
            NAME,
            Status::Ok,
            format!(
                "{} skills and {} MCP tools match the snapshot",
                live.skills.count, live.mcp_tools.count
            ),
        );
    }

    let mut parts = Vec::new();
    for (label, names) in [
        ("skills added", &drift.skills_added),
        ("skills removed", &drift.skills_removed),
        ("MCP tools added", &drift.tools_added),
        ("MCP tools removed", &drift.tools_removed),
    ] {
        if !names.is_empty() {
            parts.push(format!("{label}: {}", names.join(", ")));
        }
    }
    CheckResult::new(
        NAME,
        Status::Fail,
        format!("registry drifted from the snapshot — {}", parts.join("; ")),
    )
    .with_remedy(
        "if the change is intended, run `cargo run -p ix-skill --bin ix -- doctor --write` \
         and commit the snapshot diff (also update EXPECTED in \
         crates/ix-agent/tests/parity.rs for MCP tools); if it is not, you have \
         accidentally added or dropped a capability",
    )
    .with_details(serde_json::to_value(&drift).unwrap_or(Value::Null))
}

/// The orphan-trait check: declared public surface nothing satisfies.
fn check_orphan_traits(root: &Path) -> CheckResult {
    const NAME: &str = "orphan-traits";
    let census = match orphan_traits::scan(root) {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::new(NAME, Status::Fail, e)
                .with_remedy("fix or delete state/registry/orphan-traits.allow.json")
        }
    };

    // Informational, never load-bearing: traits with no in-tree implementor
    // but a live generic bound. Those are open extension points, not defects —
    // but they are the population the orphan rule is carved out of, so the
    // number is worth having in the report.
    let no_implementor: Vec<&str> = census
        .traits
        .iter()
        .filter(|t| t.impls == 0)
        .map(|t| t.name.as_str())
        .collect();

    let details = json!({
        "scanned_traits": census.traits.len(),
        "no_implementor": no_implementor,
        "unlisted_orphans": census.unlisted_orphans,
        "allowed_orphans": census.allowed_orphans,
        "stale_allowlist": census.stale_allowlist,
        "reasonless_allowlist": census.reasonless_allowlist,
    });

    if !census.reasonless_allowlist.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!(
                "allowlist entries without a reason: {}",
                census.reasonless_allowlist.join(", ")
            ),
        )
        .with_remedy(
            "every entry in state/registry/orphan-traits.allow.json needs a non-empty \
             `reason` — an exemption nobody justified is a silencer, not a decision",
        )
        .with_details(details);
    }

    if !census.unlisted_orphans.is_empty() {
        let listed = census
            .unlisted_orphans
            .iter()
            .map(|t| format!("{} ({}:{})", t.name, t.file, t.line))
            .collect::<Vec<_>>()
            .join(", ");
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!("public traits with no implementor and no generic bound: {listed}"),
        )
        .with_remedy(
            "each of these is either dead surface or an unsatisfied contract — delete the \
             trait, implement it, or add it to state/registry/orphan-traits.allow.json with \
             a `reason` saying which",
        )
        .with_details(details);
    }

    if !census.stale_allowlist.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Warn,
            format!(
                "allowlist entries no longer orphaned: {}",
                census.stale_allowlist.join(", ")
            ),
        )
        .with_remedy(
            "these traits now have implementors or were deleted — drop their entries from \
             state/registry/orphan-traits.allow.json so the allowlist stays honest",
        )
        .with_details(details);
    }

    CheckResult::new(
        NAME,
        Status::Ok,
        format!(
            "{} public traits scanned, {} allowlisted orphan(s), 0 unlisted",
            census.traits.len(),
            census.allowed_orphans.len()
        ),
    )
    .with_details(details)
}

/// The dark-feature check: source that no compiled configuration reaches.
///
/// Unlike its two neighbours this one shells out — `cargo metadata` is the only
/// authority on feature resolution, and re-deriving it here would create a
/// second one. When cargo is unavailable the check warns rather than fails: an
/// environment without cargo cannot have introduced a dark feature either.
fn check_dark_features(root: &Path) -> CheckResult {
    const NAME: &str = "dark-features";
    let census = match dark_features::scan(root) {
        Ok(c) => c,
        Err(e) => {
            return CheckResult::new(
                NAME,
                Status::Warn,
                format!("could not resolve features: {e}"),
            )
            .with_remedy(
                "this check needs `cargo metadata` to run in the repo root — it is skipped, \
                     not passed, so re-run it somewhere cargo is available before relying on a \
                     green doctor",
            )
        }
    };

    let details = json!({
        "dark": census.dark,
        "unlisted": census.unlisted,
        "allowed": census.allowed,
        "stale_allowlist": census.stale_allowlist,
        "reasonless_allowlist": census.reasonless_allowlist,
        "unattributed_allowlist": census.unattributed_allowlist,
        "dark_loc": census.dark_loc,
        "dark_tests": census.dark_tests,
        "tracked_debt": census.tracked_debt,
    });

    if !census.reasonless_allowlist.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!(
                "allowlist entries without a reason: {}",
                census.reasonless_allowlist.join(", ")
            ),
        )
        .with_remedy(
            "every entry in state/registry/dark-features.allow.json needs a non-empty \
             `reason` — an exemption nobody justified is a silencer, not a decision",
        )
        .with_details(details);
    }

    if !census.unattributed_allowlist.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!(
                "`tracked` allowlist entries with no issue: {}",
                census.unattributed_allowlist.join(", ")
            ),
        )
        .with_remedy(
            "a `tracked` exemption says the feature could be compiled but is not wired up yet, \
             so it needs an `issue` naming who is doing that — use kind `environment` instead if \
             it genuinely cannot be built here",
        )
        .with_details(details);
    }

    if !census.unlisted.is_empty() {
        let listed = census
            .unlisted
            .iter()
            .map(|d| {
                format!(
                    "{} ({} lines, {} test(s) across {})",
                    d.key(),
                    d.loc,
                    d.tests,
                    if d.modules.is_empty() {
                        format!("{} gated item(s)", d.items)
                    } else {
                        d.modules.join(", ")
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        return CheckResult::new(
            NAME,
            Status::Fail,
            format!("no compiled configuration reaches: {listed}"),
        )
        .with_remedy(
            "this code is never type-checked and its tests can neither pass nor fail — enable \
             the feature from a workspace member so the default build compiles it, delete it, \
             or add it to state/registry/dark-features.allow.json with a `reason` (kind \
             `environment` if it cannot be built here, `tracked` plus an `issue` if it just \
             is not wired up yet)",
        )
        .with_details(details);
    }

    if !census.stale_allowlist.is_empty() {
        return CheckResult::new(
            NAME,
            Status::Warn,
            format!(
                "allowlist entries no longer dark: {}",
                census.stale_allowlist.join(", ")
            ),
        )
        .with_remedy(
            "these features are now enabled by some workspace member, or no longer declared — \
             drop their entries from state/registry/dark-features.allow.json so the allowlist \
             stays honest",
        )
        .with_details(details);
    }

    CheckResult::new(
        NAME,
        Status::Ok,
        format!(
            "{} dark feature(s), {} significant and all allowlisted ({} tracked); \
             {} lines and {} test(s) never compiled",
            census.dark.len(),
            census.allowed.len(),
            census.tracked_debt,
            census.dark_loc,
            census.dark_tests,
        ),
    )
    .with_details(details)
}

/// Environment checks inherited from the original `ix check doctor`: the
/// governance submodule and the state directory.
fn check_environment(root: &Path) -> Vec<CheckResult> {
    let gov_dir =
        std::env::var("IX_GOVERNANCE_DIR").unwrap_or_else(|_| "governance/demerzel".to_string());
    let gov_path = root.join(&gov_dir);
    let gov_ok = gov_path.is_dir();

    let mut out = vec![if gov_ok {
        CheckResult::new("demerzel-governance", Status::Ok, format!("{gov_dir} present"))
    } else {
        CheckResult::new(
            "demerzel-governance",
            Status::Warn,
            format!("{gov_dir} missing"),
        )
        .with_remedy("run `git submodule update --init` to fetch the governance submodule")
    }];

    let constitution = gov_path.join("constitutions/default.constitution.md");
    out.push(if constitution.is_file() {
        CheckResult::new("default-constitution", Status::Ok, "constitution present")
    } else if gov_ok {
        CheckResult::new(
            "default-constitution",
            Status::Warn,
            "default.constitution.md missing",
        )
        .with_remedy("the governance submodule is checked out but incomplete — re-sync it")
    } else {
        CheckResult::new(
            "default-constitution",
            Status::Skip,
            "skipped (governance submodule absent)",
        )
    });

    out.push(if root.join("state").is_dir() {
        CheckResult::new("state-directory", Status::Ok, "state/ present")
    } else {
        CheckResult::new("state-directory", Status::Warn, "state/ missing")
            .with_remedy("belief and snapshot artifacts live in state/ — expected at the repo root")
    });

    out
}

/// The `--full` shell-outs: the same clippy and test invocation CI runs.
///
/// Kept separate from [`run`] so the fast path stays callable from tests.
/// CI additionally runs the clippy leg under the pinned `nightly-2026-08-23`
/// toolchain; this runs whatever toolchain is default locally, matching
/// `scripts/verify.ps1`.
pub fn run_full_checks() -> Vec<CheckResult> {
    vec![
        shell_check(
            "clippy",
            &["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
            "fix the lints, or justify an `#[allow]` in the diff — CI runs this same \
             invocation on stable and on the pinned nightly-2026-08-23",
        ),
        shell_check(
            "cargo-test",
            &["test", "--workspace"],
            "re-run the failing test alone for readable output: `cargo test -p <crate> <name>`",
        ),
    ]
}

/// Run `cargo <args>` and turn its exit status into a [`CheckResult`].
fn shell_check(name: &str, args: &[&str], remedy: &str) -> CheckResult {
    match std::process::Command::new("cargo").args(args).status() {
        Ok(st) if st.success() => {
            CheckResult::new(name, Status::Ok, format!("`cargo {}` passed", args.join(" ")))
        }
        Ok(st) => CheckResult::new(
            name,
            Status::Fail,
            format!("`cargo {}` failed ({st})", args.join(" ")),
        )
        .with_remedy(remedy),
        Err(e) => CheckResult::new(name, Status::Fail, format!("could not run cargo: {e}"))
            .with_remedy("is cargo on PATH?"),
    }
}

/// CLI entry point for `ix doctor`.
pub fn main(format: Format, opts: Options) -> Result<i32, String> {
    let cwd = std::env::current_dir().map_err(|e| format!("resolving cwd: {e}"))?;
    let root = find_repo_root(&cwd).ok_or_else(|| {
        format!(
            "no IX workspace root above {} — run `ix doctor` from inside the repo",
            cwd.display()
        )
    })?;

    let mut report = run(&root, opts);
    if opts.full {
        report.checks.extend(run_full_checks());
    }

    let payload = json!({
        "verdict": report.verdict(),
        "exit_code": report.exit_code(),
        "root": report.root,
        "checks": report.checks,
    });

    match format.resolve() {
        Format::Table => render_human(&report),
        other => output::emit(&payload, other).map_err(|e| format!("writing output: {e}"))?,
    }
    Ok(report.exit_code())
}

/// Human-readable rendering: one line per check, remedies underneath.
///
/// The default table renderer flattens nested detail into noise, so the
/// terminal path is hand-rolled to keep the remedy readable.
fn render_human(report: &Report) {
    println!("ix doctor — {}", report.root);
    println!();
    for c in &report.checks {
        println!("  [{:>4}] {:<22} {}", c.status.label(), c.name, c.summary);
        if let Some(r) = &c.remedy {
            for line in wrap(r, 72) {
                println!("         -> {line}");
            }
        }
    }
    println!();
    let fails = report
        .checks
        .iter()
        .filter(|c| c.status == Status::Fail)
        .count();
    let warns = report
        .checks
        .iter()
        .filter(|c| c.status == Status::Warn)
        .count();
    println!(
        "verdict {} ({} check(s), {fails} failing, {warns} warning(s)) — exit {}",
        report.verdict(),
        report.checks.len(),
        report.exit_code()
    );
}

/// Greedy word wrap, so a long remedy stays readable in a terminal.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}
