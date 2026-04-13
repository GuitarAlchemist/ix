//! Integration tests for the R3 registry compatibility checker.
//!
//! Each test builds two in-memory `Registry` values (not disk files)
//! and runs `compare` on them. This keeps the fixtures self-contained
//! and avoids coupling to the real governance submodule layout.

use ix_registry_check::{compare, has_breaking, Registry, RegistryRepo, Severity};
use std::collections::BTreeMap;

fn repo(server: &str, tools: &[(&str, &[&str])]) -> RegistryRepo {
    let mut map: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (cat, tool_list) in tools {
        map.insert(
            (*cat).to_string(),
            tool_list.iter().map(|s| s.to_string()).collect(),
        );
    }
    RegistryRepo {
        server: Some(server.into()),
        description: None,
        domains: vec![],
        tools: map,
    }
}

fn reg(version: &str, repos: &[(&str, RegistryRepo)]) -> Registry {
    let mut map = BTreeMap::new();
    for (k, v) in repos {
        map.insert((*k).to_string(), v.clone());
    }
    Registry {
        version: Some(version.into()),
        repos: map,
    }
}

#[test]
fn identical_registries_have_no_findings() {
    let ix = repo("ix", &[("core-math", &["ix_stats", "ix_distance"])]);
    let r = reg("2.0.0", &[("ix", ix)]);
    let findings = compare(&r, &r);
    assert!(findings.is_empty(), "identical registries: {findings:?}");
    assert!(!has_breaking(&findings));
}

#[test]
fn adding_a_tool_is_compatible() {
    let old_ix = repo("ix", &[("core-math", &["ix_stats"])]);
    let new_ix = repo("ix", &[("core-math", &["ix_stats", "ix_distance"])]);
    let old = reg("2.0.0", &[("ix", old_ix)]);
    let new_reg = reg("2.0.0", &[("ix", new_ix)]);
    let findings = compare(&old, &new_reg);
    assert!(!has_breaking(&findings));
    assert!(findings
        .iter()
        .any(|f| f.severity == Severity::Compatible && f.path.ends_with("ix_distance")));
}

#[test]
fn removing_a_tool_is_breaking() {
    let old_ix = repo("ix", &[("core-math", &["ix_stats", "ix_distance"])]);
    let new_ix = repo("ix", &[("core-math", &["ix_stats"])]);
    let old = reg("2.0.0", &[("ix", old_ix)]);
    let new_reg = reg("2.0.0", &[("ix", new_ix)]);
    let findings = compare(&old, &new_reg);
    assert!(has_breaking(&findings));
    let breaking: Vec<_> = findings
        .iter()
        .filter(|f| f.severity == Severity::Breaking)
        .collect();
    assert_eq!(breaking.len(), 1);
    assert!(breaking[0].path.ends_with("ix_distance"));
}

#[test]
fn removing_a_category_is_breaking() {
    let old_ix = repo(
        "ix",
        &[
            ("core-math", &["ix_stats"]),
            ("signal", &["ix_fft"]),
        ],
    );
    let new_ix = repo("ix", &[("core-math", &["ix_stats"])]);
    let old = reg("2.0.0", &[("ix", old_ix)]);
    let new_reg = reg("2.0.0", &[("ix", new_ix)]);
    let findings = compare(&old, &new_reg);
    assert!(has_breaking(&findings));
    assert!(findings
        .iter()
        .any(|f| f.severity == Severity::Breaking && f.path.ends_with(".signal")));
}

#[test]
fn removing_a_repo_is_breaking() {
    let ix = repo("ix", &[("core-math", &["ix_stats"])]);
    let tars = repo("tars", &[("grammar", &["weighted_cfg"])]);
    let old = reg("2.0.0", &[("ix", ix.clone()), ("tars", tars)]);
    let new_reg = reg("2.0.0", &[("ix", ix)]);
    let findings = compare(&old, &new_reg);
    assert!(has_breaking(&findings));
    assert!(findings
        .iter()
        .any(|f| f.severity == Severity::Breaking && f.path.ends_with(".tars")));
}

#[test]
fn renaming_a_tool_is_detected_as_breaking_plus_compatible() {
    // Rename ix_stats → ix_stats_v2: the old name is removed (breaking)
    // and the new name is added (compatible). Two findings total.
    let old_ix = repo("ix", &[("core-math", &["ix_stats"])]);
    let new_ix = repo("ix", &[("core-math", &["ix_stats_v2"])]);
    let old = reg("2.0.0", &[("ix", old_ix)]);
    let new_reg = reg("2.0.0", &[("ix", new_ix)]);
    let findings = compare(&old, &new_reg);
    assert_eq!(findings.len(), 2);
    assert!(findings
        .iter()
        .any(|f| f.severity == Severity::Breaking && f.path.ends_with("ix_stats")));
    assert!(findings
        .iter()
        .any(|f| f.severity == Severity::Compatible && f.path.ends_with("ix_stats_v2")));
}

#[test]
fn version_bump_is_informational_only() {
    let ix = repo("ix", &[("core-math", &["ix_stats"])]);
    let old = reg("2.0.0", &[("ix", ix.clone())]);
    let new_reg = reg("2.1.0", &[("ix", ix)]);
    let findings = compare(&old, &new_reg);
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].severity, Severity::Informational);
    assert_eq!(findings[0].path, "$.version");
    assert!(!has_breaking(&findings));
}

#[test]
fn compare_handles_the_real_capability_registry() {
    // Smoke test: load the actual governance file if it's present (ie
    // the submodule is checked out). Compare it against itself — no
    // findings. If the submodule is absent, skip the test quietly.
    let path = std::path::Path::new("..")
        .join("..")
        .join("governance")
        .join("demerzel")
        .join("schemas")
        .join("capability-registry.json");
    if !path.exists() {
        eprintln!("skipping: governance submodule not checked out at {}", path.display());
        return;
    }
    let reg = ix_registry_check::load(&path)
        .unwrap_or_else(|e| panic!("loading real registry: {e}"));
    let findings = compare(&reg, &reg);
    assert!(
        findings.is_empty(),
        "comparing the real registry against itself should yield no findings, got: {findings:?}"
    );
    assert!(!reg.repos.is_empty(), "registry should not be empty");
}
