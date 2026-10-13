//! PR #346: the semantic integration gate must follow its direct inputs.
use serde_yaml::Value;

const WORKFLOW: &str = include_str!("../../../.github/workflows/ix-duck-chatbot.yml");

fn check_event(event: &str) {
    let workflow: Value = serde_yaml::from_str(WORKFLOW).expect("valid workflow YAML");
    let trigger = &workflow["on"][event];
    assert_eq!(trigger["branches"][0].as_str(), Some("main"));
    assert!(trigger["paths-ignore"].is_null(), "{event}: unexpected exclusion");
    let paths: Vec<&str> = trigger["paths"]
        .as_sequence()
        .expect("event must have a paths filter")
        .iter()
        .map(|path| path.as_str().expect("path must be a string"))
        .collect();
    let mut actual = paths.clone();
    actual.sort_unstable();
    let mut expected = vec![
        "crates/ix-duck/**",
        "crates/ix-code/**",
        "Cargo.toml",
        "Cargo.lock",
        ".github/workflows/ix-duck-chatbot.yml",
        "crates/ix-harness-github-actions/tests/ix_duck_workflow_paths.rs",
    ];
    expected.sort_unstable();
    // Keep the allowlist narrow. This also rejects negations and broader globs.
    assert_eq!(actual, expected, "{event}: direct-input allowlist drifted");

    // The allowlist above uses only exact paths and recursive directory globs.
    let matches = |file: &str| {
        paths.iter().any(|pattern| {
            pattern.strip_suffix("**").map_or(file == *pattern, |prefix| {
                file.starts_with(prefix)
            })
        })
    };
    for file in [
        "crates/ix-code/src/semantic.rs",
        "crates/ix-code/src/nested/mod.rs",
        "crates/ix-code/Cargo.toml",
        "crates/ix-duck/src/code.rs",
        "crates/ix-duck/tests/fixtures/chatbot-qa/trace.json",
        "crates/ix-duck/Cargo.toml",
        "Cargo.toml",
        "Cargo.lock",
        ".github/workflows/ix-duck-chatbot.yml",
        "crates/ix-harness-github-actions/tests/ix_duck_workflow_paths.rs",
    ] {
        assert!(matches(file), "{event}: must trigger for {file}");
    }
    for file in [
        "README.md",
        "docs/guides/graph-theory-in-ix.md",
        "docs/fr/CHECKLISTS.md",
        "crates/ix-code-tools/README.md",
        ".github/workflows/wiki-sync.yml",
    ] {
        assert!(!matches(file), "{event}: unrelated file must not trigger: {file}");
    }
}

#[test]
fn push_covers_semantic_inputs_without_unrelated_docs() {
    check_event("push");
}

#[test]
fn pull_request_covers_semantic_inputs_without_unrelated_docs() {
    check_event("pull_request");
}
