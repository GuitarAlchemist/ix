//! `ci.yml` follows the inputs its jobs read, and nothing else.
//!
//! A pull request that changes only prose, hooks or skills used to run the whole Rust matrix
//! (four build-and-test rows, six gated-features rows, the DuckDB job): forty runner-minutes
//! for files no job opens. The filter below says which files the workflow reads. Two text files
//! stay in because a job does read them: `go/ixctl/README.md` and `go/ixctl/README.fr.md` carry
//! the check count the gate's E24 holds to what ran, and `docs/contracts/` holds schemas that
//! crates `include_str!`.
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
        } else {
            file == pat
        };
        if hit {
            included = !negated;
        }
    }
    included
}

fn check_event(event: &str) {
    let workflow: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    let trigger = &workflow["on"][event];
    assert_eq!(trigger["branches"][0].as_str(), Some("main"));
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
            "go/ixctl/README.md",
            "go/ixctl/README.fr.md",
            "!docs/**",
            "docs/contracts/**",
            "!.claude/**",
        ],
        "{event}: the filter drifted; update the list here and the reasons above together"
    );

    for file in [
        "Cargo.toml",
        "Cargo.lock",
        "crates/ix-skill/src/doctor/mod.rs",
        "crates/ix-skill/tests/doctor.rs",
        "go/ixctl/ws.go",
        "go/ixctl/e2e/e2e.py",
        "go/ixctl/README.md",
        "go/ixctl/README.fr.md",
        "docs/contracts/quality-snapshot.schema.json",
        "docs/contracts/2026-10-21-ixctl-probe-observation.contract.md",
        "state/registry/skills.snapshot.json",
        "state/assumptions/annotations.snapshot.json",
        "examples/showcase/advanced/some-pipeline.yaml",
        ".github/workflows/ci.yml",
        ".gitattributes",
        "scripts/verify.ps1",
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
        "crates/ix-code-tools/README.md",
        ".claude/settings.json",
        ".claude/hooks/correction-detector.sh",
        ".claude/hooks/test-correction-detector.sh",
        ".claude/skills/learnings/SKILL.md",
    ] {
        assert!(
            !triggers(&paths, file),
            "{event}: no job reads {file}; it must not run the matrix"
        );
    }
}

#[test]
fn push_runs_for_the_inputs_the_jobs_read_and_not_for_prose() {
    check_event("push");
}

#[test]
fn pull_request_runs_for_the_inputs_the_jobs_read_and_not_for_prose() {
    check_event("pull_request");
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
}
