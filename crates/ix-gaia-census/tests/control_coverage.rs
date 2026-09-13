//! The control-coverage map is machine-readable and complete.
//!
//! The M2 test obligation is the union of the twenty-eight controls enumerated
//! at specification §11 and the fourteen at engineering-doctrine §4. The
//! enumerated lists are authoritative here: the specification also states the
//! §11 cardinality as eighteen and as twenty-six elsewhere, and repairing that
//! is not this crate's business. Every control appears in the artifact exactly
//! once, either covered by a named test or explicitly not applicable with the
//! subject it is missing.

use std::collections::BTreeSet;
use std::path::PathBuf;

use ix_gaia_census::{census, CensusRequest, ControlStatus};

fn fixture_request() -> CensusRequest {
    CensusRequest {
        evidence_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gaia-s1-r18"),
        manifest_file_name: "gaia-s1-r2-bundle-manifest.md".to_string(),
    }
}

fn expected_ids() -> Vec<String> {
    (1..=28)
        .map(|n| format!("spec-11-{n:02}"))
        .chain((1..=14).map(|n| format!("doctrine-4-{n:02}")))
        .collect()
}

#[test]
fn every_enumerated_control_appears_exactly_once_and_in_order() {
    let artifact = census(&fixture_request()).expect("pristine fixture");

    let ids: Vec<String> = artifact
        .controls
        .iter()
        .map(|control| control.control_id.clone())
        .collect();

    assert_eq!(ids, expected_ids(), "42 enumerated controls, each once");
}

#[test]
fn no_control_is_silently_omitted_or_waved_through() {
    let artifact = census(&fixture_request()).expect("pristine fixture");

    for control in &artifact.controls {
        assert!(
            !control.detail.trim().is_empty(),
            "{} states nothing",
            control.control_id
        );
        if control.status == ControlStatus::NotApplicable {
            let detail = control.detail.to_lowercase();
            assert!(
                !detail.contains("out of scope"),
                "{} must name the subject it is missing, not declare itself out of scope",
                control.control_id
            );
        }
    }
}

#[test]
fn every_covered_control_names_a_test_that_exists() {
    let artifact = census(&fixture_request()).expect("pristine fixture");
    let declared = declared_test_names();

    let mut covered = 0;
    for control in &artifact.controls {
        if control.status != ControlStatus::Covered {
            continue;
        }
        covered += 1;
        assert!(
            names_a_declared_test(&control.detail, &declared),
            "{} claims coverage but names no test that exists in this crate: {}",
            control.control_id,
            control.detail
        );
    }
    assert!(covered > 0, "a coverage map covering nothing is a placeholder");
}

/// The coverage gate must not be satisfiable by anything except a real test.
///
/// This is the mutation control for the gate itself. A helper function is not a
/// test however test-shaped its name; an ordinary English word that happens to
/// be a helper is not a test; and a real test name buried inside a longer
/// identifier is not a reference to that test. Each of these once satisfied the
/// gate, which is why each is pinned here.
#[test]
fn a_helper_name_or_a_substring_cannot_satisfy_the_coverage_gate() {
    let declared = declared_test_names();

    assert!(
        declared.contains("every_reported_state_is_attributable"),
        "the scraper must still find a real #[test] function"
    );
    for helper in [
        "declared_test_names",
        "names_a_declared_test",
        "staged_copy",
        "inventory_row",
        "expected_ids",
        "listing",
        "walk",
        "object_keys",
    ] {
        assert!(
            !declared.contains(helper),
            "{helper} is a helper, not a #[test]; it must never count as a declared test"
        );
    }

    for fabricated in [
        "a listing of the artifact keys shows it carries no verdict or acceptance value.",
        "staged_copy stages the evidence, so this control is met.",
        "not_a_test_every_reported_state_is_attributable observes the reported state.",
        "every_reported_state_is_attributable_extended observes the reported state.",
    ] {
        assert!(
            !names_a_declared_test(fabricated, &declared),
            "{fabricated:?} names no #[test] and must not satisfy the coverage gate"
        );
    }
    assert!(
        names_a_declared_test(
            "every_reported_state_is_attributable binds every value to a named file.",
            &declared
        ),
        "a detail naming a real #[test] as a whole identifier must satisfy the gate"
    );
}

/// Every `#[test]` function name declared in this crate's test files.
///
/// A function counts only when a `#[test]` attribute stands directly above it,
/// separated by nothing but blank lines, comments, and further attributes. A
/// helper function is not a test however test-shaped its name, and is not
/// collected here.
fn declared_test_names() -> BTreeSet<String> {
    let tests_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut names = BTreeSet::new();
    for entry in std::fs::read_dir(&tests_dir).expect("tests directory") {
        let path = entry.expect("test entry").path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read test source");
        let lines: Vec<&str> = source.lines().map(str::trim).collect();
        for (index, line) in lines.iter().enumerate() {
            let Some(rest) = line.strip_prefix("fn ") else {
                continue;
            };
            let Some(name) = rest.split('(').next() else {
                continue;
            };
            if carries_the_test_attribute(&lines, index) {
                names.insert(name.to_string());
            }
        }
    }
    names
}

/// Whether `#[test]` stands above the function declared at `index`.
fn carries_the_test_attribute(lines: &[&str], index: usize) -> bool {
    for line in lines[..index].iter().rev() {
        if *line == "#[test]" {
            return true;
        }
        if line.is_empty() || line.starts_with("//") || line.starts_with("#[") {
            continue;
        }
        return false;
    }
    false
}

/// Whether a `Covered` detail names a test that exists.
///
/// The convention a detail must follow is small and deterministic: it names its
/// discriminating tests as bare identifiers, and every identifier-shaped word it
/// contains — every word carrying an underscore — is the exact name of a
/// `#[test]` in this crate. At least one such name must be present. An exact
/// whole-identifier match is required in both directions, so neither a helper
/// name, nor a substring of a test name, nor a test name embedded in a longer
/// identifier can satisfy the gate.
fn names_a_declared_test(detail: &str, declared: &BTreeSet<String>) -> bool {
    let words: Vec<&str> = detail
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|word| !word.is_empty())
        .collect();
    let names_one = words.iter().any(|word| declared.contains(*word));
    let names_a_stranger = words
        .iter()
        .any(|word| word.contains('_') && !declared.contains(*word));
    names_one && !names_a_stranger
}
