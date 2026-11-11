//! Runs the PR coordination room fixtures against the schema.
//!
//! `fixtures.json` is the index: each fixture says whether it is valid and why.
//! Each invalid fixture must also fail for the reason it names, so a fixture
//! cannot pass by breaking some other rule.

use std::path::PathBuf;

use ix_quality_validate::coord::{build_coord_validator, parse_block, validate_comment};
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/contracts/fixtures/pr-coordination-room")
}

fn index() -> Vec<Value> {
    let text = std::fs::read_to_string(fixtures_dir().join("fixtures.json")).expect("index reads");
    let v: Value = serde_json::from_str(&text).expect("index parses");
    v["fixtures"].as_array().expect("fixtures array").clone()
}

fn read(file: &str) -> String {
    std::fs::read_to_string(fixtures_dir().join(file)).expect("fixture reads")
}

/// A substring the error list of the named invalid fixture must contain.
fn expected_error(file: &str) -> &'static str {
    match file {
        "invalid-missing-id.md" => "\"id\" is a required property",
        "invalid-result-no-reply-to.md" => "\"reply_to\" is a required property",
        "invalid-unknown-type.md" => "\"notify\" is not one of",
        "invalid-ack-state-done.md" => "\"received\"",
        other => panic!("no expected error declared for {other}"),
    }
}

#[test]
fn index_lists_three_valid_and_four_invalid() {
    let fixtures = index();
    let valid = fixtures.iter().filter(|f| f["valid"] == true).count();
    assert_eq!((valid, fixtures.len() - valid), (3, 4));
}

#[test]
fn index_covers_every_fixture_file() {
    let mut on_disk: Vec<String> = std::fs::read_dir(fixtures_dir())
        .expect("dir reads")
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".md"))
        .collect();
    let mut indexed: Vec<String> = index()
        .iter()
        .map(|f| f["file"].as_str().unwrap().to_string())
        .collect();
    on_disk.sort();
    indexed.sort();
    assert_eq!(on_disk, indexed);
}

#[test]
fn valid_fixtures_pass() {
    let validator = build_coord_validator().expect("schema compiles");
    for f in index().iter().filter(|f| f["valid"] == true) {
        let file = f["file"].as_str().unwrap();
        let res = validate_comment(&validator, &read(file));
        assert!(res.is_ok(), "{file} should be valid, got {res:?}");
    }
}

#[test]
fn invalid_fixtures_fail_for_the_named_reason() {
    let validator = build_coord_validator().expect("schema compiles");
    for f in index().iter().filter(|f| f["valid"] == false) {
        let file = f["file"].as_str().unwrap();
        let errors = validate_comment(&validator, &read(file))
            .expect_err(&format!("{file} should be invalid"));
        let want = expected_error(file);
        assert!(
            errors.iter().any(|e| e.contains(want)),
            "{file}: no error contains {want:?}, got {errors:?}"
        );
    }
}

#[test]
fn free_text_after_the_block_is_ignored() {
    let pairs = parse_block(&read("valid-request.md")).expect("parses");
    assert_eq!(pairs.len(), 10);
}

#[test]
fn value_is_split_at_the_first_separator_only() {
    let pairs = parse_block("```coord\nnext: a: b\n```\n").expect("parses");
    assert_eq!(pairs["next"], "a: b");
}

#[test]
fn unparseable_blocks_are_rejected() {
    for (name, text) in [
        ("no block", "just a comment"),
        ("not first", "hello\n```coord\nv: 1\n```\n"),
        ("no separator", "```coord\nv 1\n```\n"),
        ("repeated key", "```coord\nv: 1\nv: 1\n```\n"),
        ("unclosed", "```coord\nv: 1\n"),
    ] {
        assert!(parse_block(text).is_err(), "{name} should not parse");
    }
}

#[test]
fn extra_field_and_mention_are_rejected() {
    let validator = build_coord_validator().expect("schema compiles");
    let base = read("valid-request.md");
    let extra = base.replacen("eta: unknown\n", "eta: unknown\nnote: x\n", 1);
    assert!(validate_comment(&validator, &extra).is_err(), "extra field");
    let mention = base.replacen("to: gaia", "to: @gaia", 1);
    assert!(validate_comment(&validator, &mention).is_err(), "at-sign in to");
}
