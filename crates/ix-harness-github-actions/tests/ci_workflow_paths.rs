//! `ci.yml` follows the inputs its jobs read, and nothing else.
//!
//! A pull request that changes only prose, hooks or skills used to run the whole Rust matrix
//! (four build-and-test rows, six gated-features rows, the DuckDB job): forty runner-minutes
//! for files no job opens. The filter below says which files the workflow reads. Some text files
//! stay in because a job does read them: `go/ixctl/README.md` and `go/ixctl/README.fr.md` carry
//! the check count the gate's E24 holds to what ran, `docs/contracts/` holds schemas that
//! crates `include_str!`, and `ix doctor` reads the Markdown under `crates/` (hardcoded-paths)
//! and `.claude/settings.json` with the hooks it wires (hooks). Skills stay out.
use serde_yaml::Value;

const WORKFLOW: &str = include_str!("../../../.github/workflows/ci.yml");

/// GitHub's `paths` semantics for the pattern shapes used here: patterns apply in order, the
/// last one that matches a file decides, and a leading `!` excludes.
fn triggers(patterns: &[&str], file: &str) -> bool {
    let mut included = false;
    for pattern in patterns {
        let (negated, pat) = match pattern.strip_prefix('!') {
            Some(p) => (true, p),
            None => (false, *pattern),
        };
        let hit = if pat == "**" {
            true
        } else if let Some(suffix) = pat.strip_prefix("**") {
            file.ends_with(suffix)
        } else if let Some(prefix) = pat.strip_suffix("**") {
            file.starts_with(prefix)
        } else if let Some((prefix, suffix)) = pat.split_once("**") {
            file.starts_with(prefix) && file.ends_with(suffix)
        } else {
            file == pat
        };
        if hit {
            included = !negated;
        }
    }
    included
}

/// `branch` is the one branch the event is limited to, or `None` for every base.
fn check_event(event: &str, branch: Option<&str>) {
    let workflow: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    let trigger = &workflow["on"][event];
    match branch {
        Some(b) => assert_eq!(trigger["branches"][0].as_str(), Some(b)),
        None => assert!(
            trigger["branches"].is_null(),
            "{event}: a PR stacked on an exp/* branch must get CI too (ix#423)"
        ),
    }
    assert!(
        trigger["paths-ignore"].is_null(),
        "{event}: use `paths` with negations, so a text file a job reads can be re-included"
    );
    let paths: Vec<&str> = trigger["paths"]
        .as_sequence()
        .expect("event must have a paths filter")
        .iter()
        .map(|path| path.as_str().expect("path must be a string"))
        .collect();
    // Order matters to GitHub, so the list is pinned as written, not as a set.
    assert_eq!(
        paths,
        [
            "**",
            "!**.md",
            "crates/**.md",
            "go/ixctl/README.md",
            "go/ixctl/README.fr.md",
            "!docs/**",
            "docs/contracts/**",
            "!.claude/**",
            ".claude/settings.json",
            ".claude/hooks/**",
        ],
        "{event}: the filter drifted; update the list here and the reasons above together"
    );

    for file in [
        "Cargo.toml",
        "Cargo.lock",
        "crates/ix-cli/src/doctor/mod.rs",
        "crates/ix-cli/tests/doctor.rs",
        "go/ixctl/ws.go",
        "go/ixctl/e2e/e2e.py",
        "go/ixctl/README.md",
        "go/ixctl/README.fr.md",
        "docs/contracts/quality-snapshot.schema.json",
        "docs/contracts/2026-10-21-ixctl-probe-observation.contract.md",
        "state/registry/tools.snapshot.json",
        "state/assumptions/annotations.snapshot.json",
        "examples/showcase/advanced/some-pipeline.yaml",
        ".github/workflows/ci.yml",
        ".gitattributes",
        "scripts/verify.ps1",
        // ix doctor's hardcoded-paths check scans the Markdown under crates/.
        "crates/ix-code-tools/README.md",
        // ix doctor's hooks check reads the wiring, parses each hook, runs each fixture.
        ".claude/settings.json",
        ".claude/hooks/correction-detector.sh",
        ".claude/hooks/test-correction-detector.sh",
    ] {
        assert!(triggers(&paths, file), "{event}: must trigger for {file}");
    }
    for file in [
        "README.md",
        "README.fr.md",
        "CLAUDE.md",
        "CONTEXT.md",
        "docs/CHECKLISTS.md",
        "docs/fr/CHECKLISTS.md",
        "docs/session-learned-rules.md",
        "docs/solutions/test-failures/2026-10-22-some-note.md",
        "docs/plans/2026-10-22-some-plan.md",
        "docs/adr/0011-something.md",
        ".claude/skills/learnings/SKILL.md",
        ".claude/settings.local.json",
    ] {
        assert!(
            !triggers(&paths, file),
            "{event}: no job reads {file}; it must not run the matrix"
        );
    }
}

#[test]
fn push_runs_for_the_inputs_the_jobs_read_and_not_for_prose() {
    check_event("push", Some("main"));
}

#[test]
fn pull_request_runs_for_the_inputs_the_jobs_read_and_not_for_prose() {
    check_event("pull_request", None);
}

#[test]
fn the_matcher_applies_patterns_in_order_with_the_last_hit_deciding() {
    let p = [
        "**",
        "!**.md",
        "go/ixctl/README.md",
        "!docs/**",
        "docs/contracts/**",
    ];
    assert!(triggers(&p, "src/lib.rs"));
    assert!(!triggers(&p, "notes.md"));
    assert!(triggers(&p, "go/ixctl/README.md"));
    assert!(!triggers(&p, "docs/guide.md"));
    assert!(!triggers(&p, "docs/data.json"));
    assert!(triggers(&p, "docs/contracts/x.schema.json"));
    assert!(triggers(&p, "docs/contracts/x.contract.md"));
    assert!(!triggers(&["!**"], "anything"));
    let crate_md = ["**", "!**.md", "crates/**.md"];
    assert!(triggers(&crate_md, "crates/seed/README.md"));
    assert!(!triggers(&crate_md, "docs/seed/README.md"));
}

/// The matrix runs pull requests' code; nothing in it writes to the repository.
#[test]
fn ci_holds_a_read_only_token() {
    let workflow: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    assert_eq!(
        workflow["permissions"]["contents"].as_str(),
        Some("read"),
        "the repository's default token is write; ci.yml must ask for read"
    );
}
