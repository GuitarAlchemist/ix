//! `ix doctor` hardcoded-path check (ix#327, ix#328).
//!
//! The seeded cases pin the scanner's behaviour on throwaway workspaces; the
//! last test scans this workspace and was red at eleven lines before those
//! lines were fixed. It is the one that keeps them fixed.

use ix_skill::doctor::hardcoded_paths::{self, AllowEntry, Allowlist};
use ix_skill::doctor::Status;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

/// A throwaway workspace with the given `crates/...` files.
fn repo(files: &[(&str, &str)]) -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    for (rel, body) in files {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).expect("mkdir");
        std::fs::write(&path, body).expect("write");
    }
    dir
}

fn allow(file: &str, contains: &str, reason: &str) -> Allowlist {
    Allowlist {
        note: String::new(),
        allow: vec![AllowEntry {
            file: file.to_string(),
            contains: contains.to_string(),
            reason: reason.to_string(),
        }],
    }
}

const SEED: &str = "crates/seed/src/lib.rs";

// ------------------------------------------------------------------ matching

#[test]
fn a_windows_home_path_is_reported_with_its_line_and_account() {
    let src =
        "pub fn tool() -> &'static str {\n    \"C:\\\\Users\\\\jane\\\\bin\\\\tool.exe\"\n}\n";
    let found = hardcoded_paths::findings_in(SEED, src);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(found[0].line, 2);
    assert_eq!(found[0].user, "jane");
}

#[test]
fn forward_slash_and_lower_case_drive_forms_are_reported() {
    for line in [
        r#"const P: &str = "C:/Users/jane/bin/tool.exe";"#,
        r#"const P: &str = "c:/users/jane/bin/tool.exe";"#,
        r#"const P: &str = r"c:\users\jane\bin\tool.exe";"#,
        r#"//!     --index "D:/Users/jane/index.bin""#,
    ] {
        assert_eq!(
            hardcoded_paths::home_user_in(line).as_deref(),
            Some("jane"),
            "{line}"
        );
    }
}

#[test]
fn mixed_repeated_and_doubly_escaped_separators_are_reported() {
    // Codex round 53: these spellings evaded the first matcher.
    for line in [
        r#"let p = "C:\\Users/jane/bin";"#,
        r#"let p = "C:/Users\\jane\\bin";"#,
        r#"let p = "C:/Users//jane/bin";"#,
        r#"let json = "{\"path\":\"C:\\\\Users\\\\jane\\\\corpus.json\"}";"#,
        r#"let url = "file:///C:/Users/jane/index.bin";"#,
        r#"let verbatim = r"\\?\C:\Users\jane\index.bin";"#,
    ] {
        assert_eq!(
            hardcoded_paths::home_user_in(line).as_deref(),
            Some("jane"),
            "{line}"
        );
    }
}

#[test]
fn urls_two_word_shared_accounts_and_root_relative_paths() {
    // Codex round 54.
    for line in [
        r#"let u = "https://users/jane/profile";"#,
        r#"let p = "C:\\Users\\All Users\\Desktop";"#,
        r#"let p = r"C:\Users\Default User\ntuser.dat";"#,
    ] {
        assert_eq!(hardcoded_paths::home_user_in(line), None, "{line}");
    }
    for line in [
        r#"let p = "\\Users\\jane\\file";"#,
        r#"let p = r"\Users\jane\file";"#,
    ] {
        assert_eq!(
            hardcoded_paths::home_user_in(line).as_deref(),
            Some("jane"),
            "{line}"
        );
    }
}

#[test]
fn posix_home_paths_are_reported() {
    for line in [
        r#"let bad = "/home/jane/corpus.json";"#,
        r#"let bad = "/Users/jane/corpus.json";"#,
    ] {
        assert_eq!(
            hardcoded_paths::home_user_in(line).as_deref(),
            Some("jane"),
            "{line}"
        );
    }
}

#[test]
fn placeholders_shared_accounts_and_routes_are_not_findings() {
    for line in [
        r#"//! cargo run -- --index "C:/Users/<you>/index.bin""#,
        r#"let p = "/home/$USER/.cache";"#,
        r#"let p = "/home/${USER}/.cache";"#,
        r#"let p = "C:\\Users\\%USERNAME%\\bin";"#,
        r#"let p = "C:\\Users\\Public\\Documents";"#,
        r#"let p = "c:/users/default/ntuser.dat";"#,
        r#"let p = "~/.cache/ix";"#,
        r#"router.get("/users/{id}")"#,
        r#"router.get("/users/42")"#,
        r#"let p = "../ga/state/voicings/optick.index";"#,
        r#"const MARKER: &str = "/home/";"#,
    ] {
        assert_eq!(hardcoded_paths::home_user_in(line), None, "{line}");
    }
}

// ----------------------------------------------------------------- allowlist

#[test]
fn an_unlisted_finding_fails_the_check_and_names_the_site() {
    let dir = repo(&[(SEED, "pub const P: &str = \"/home/jane/corpus.json\";\n")]);
    let result = hardcoded_paths::check(dir.path());
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(
        result.summary.contains("crates/seed/src/lib.rs:1"),
        "{}",
        result.summary
    );
    assert!(result.summary.contains("`jane`"), "{}", result.summary);
    assert!(result.remedy.is_some());
}

#[test]
fn an_allowlist_entry_with_a_reason_exempts_the_matching_line_only() {
    let dir = repo(&[(
        SEED,
        "pub const BAD: &str = \"/home/jane/corpus.json\";\npub const WORSE: &str = \"/home/jane/other.json\";\n",
    )]);
    let list = allow(SEED, "corpus.json", "the sanitizer's negative input");
    let census = hardcoded_paths::scan_with_allowlist(dir.path(), &list).expect("scan");
    assert_eq!(census.findings.len(), 2);
    assert_eq!(census.allowed, vec!["crates/seed/src/lib.rs:1".to_string()]);
    assert_eq!(census.unlisted.len(), 1);
    assert_eq!(census.unlisted[0].line, 2);
    assert!(census.stale_allowlist.is_empty());
}

#[test]
fn a_covered_workspace_is_ok() {
    let dir = repo(&[(SEED, "pub const BAD: &str = \"/home/jane/corpus.json\";\n")]);
    std::fs::create_dir_all(dir.path().join("state/registry")).unwrap();
    std::fs::write(
        dir.path().join(hardcoded_paths::ALLOWLIST_PATH),
        serde_json::to_string_pretty(&allow(SEED, "corpus.json", "negative input")).unwrap(),
    )
    .unwrap();
    let result = hardcoded_paths::check(dir.path());
    assert_eq!(result.status, Status::Ok, "{result:?}");
    assert!(
        result.summary.contains("1 allowlisted"),
        "{}",
        result.summary
    );
}

#[test]
fn an_allowlist_entry_without_a_reason_is_rejected() {
    let dir = repo(&[(SEED, "pub const BAD: &str = \"/home/jane/corpus.json\";\n")]);
    std::fs::create_dir_all(dir.path().join("state/registry")).unwrap();
    std::fs::write(
        dir.path().join(hardcoded_paths::ALLOWLIST_PATH),
        serde_json::to_string_pretty(&allow(SEED, "corpus.json", "  ")).unwrap(),
    )
    .unwrap();
    let result = hardcoded_paths::check(dir.path());
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(
        result.summary.contains("without a reason"),
        "{}",
        result.summary
    );
}

#[test]
fn an_allowlist_entry_matching_no_line_is_stale() {
    let dir = repo(&[(SEED, "pub const FINE: &str = \"../ga/corpus.json\";\n")]);
    std::fs::create_dir_all(dir.path().join("state/registry")).unwrap();
    std::fs::write(
        dir.path().join(hardcoded_paths::ALLOWLIST_PATH),
        serde_json::to_string_pretty(&allow(SEED, "corpus.json", "was fixed")).unwrap(),
    )
    .unwrap();
    let result = hardcoded_paths::check(dir.path());
    assert_eq!(result.status, Status::Warn, "{result:?}");
    assert!(
        result.summary.contains("match no line"),
        "{}",
        result.summary
    );
}

#[test]
fn contains_is_matched_on_the_whole_line_not_the_truncated_excerpt() {
    // Codex round 53: a discriminator past column 120 must still exempt the line.
    let padding = "x".repeat(130);
    let src = format!("let p = \"/home/jane/{padding}/corpus.json\";\n");
    let dir = repo(&[(SEED, src.as_str())]);
    let list = allow(SEED, "corpus.json", "negative input");
    let census = hardcoded_paths::scan_with_allowlist(dir.path(), &list).expect("scan");
    assert_eq!(census.allowed.len(), 1, "{census:?}");
    assert!(census.unlisted.is_empty());
    assert!(census.stale_allowlist.is_empty());
    assert!(census.findings[0].excerpt.ends_with('…'));
}

#[test]
fn scripts_sql_and_docs_beside_the_crates_are_scanned_too() {
    let dir = repo(&[
        (
            "crates/seed/examples.sql",
            "LOAD '/home/jane/ix.duckdb_extension';\n",
        ),
        (
            "crates/seed/run.ps1",
            "Set-Alias x 'C:\\Users\\jane\\run.ps1'\n",
        ),
        (
            "crates/seed/README.md",
            "Example: `/home/jane/corpus.json`\n",
        ),
        (
            "crates/seed/fixture.json",
            "{\"path\": \"/home/jane/not-scanned\"}\n",
        ),
    ]);
    let census = hardcoded_paths::scan(dir.path()).expect("scan");
    assert_eq!(census.files_scanned, 3, "{census:?}");
    let files: Vec<&str> = census.findings.iter().map(|f| f.file.as_str()).collect();
    assert_eq!(
        files,
        [
            "crates/seed/README.md",
            "crates/seed/examples.sql",
            "crates/seed/run.ps1"
        ]
    );
}

#[test]
fn a_root_without_crates_is_a_failed_scan_not_a_clean_one() {
    // Codex round 53: the first version swallowed read errors and reported ok.
    let dir = tempfile::tempdir().expect("tempdir");
    let result = hardcoded_paths::check(dir.path());
    assert_eq!(result.status, Status::Fail, "{result:?}");
    assert!(
        result.summary.contains("could not scan"),
        "{}",
        result.summary
    );
}

#[test]
fn two_entries_covering_one_line_are_both_live() {
    // Codex round 54: the first match used to be the only one marked used.
    let dir = repo(&[(SEED, "let bad = \"/home/jane/corpus.json\";\n")]);
    let list = Allowlist {
        note: String::new(),
        allow: vec![
            AllowEntry {
                file: SEED.to_string(),
                contains: "jane".to_string(),
                reason: "fixture".to_string(),
            },
            AllowEntry {
                file: SEED.to_string(),
                contains: "corpus.json".to_string(),
                reason: "negative input".to_string(),
            },
        ],
    };
    let census = hardcoded_paths::scan_with_allowlist(dir.path(), &list).expect("scan");
    assert_eq!(census.allowed.len(), 1);
    assert!(
        census.stale_allowlist.is_empty(),
        "{:?}",
        census.stale_allowlist
    );
}

#[test]
fn extensions_are_matched_case_insensitively() {
    let dir = repo(&[
        ("crates/seed/README.MD", "see /home/jane/corpus.json\n"),
        ("crates/seed/query.SQL", "LOAD '/home/jane/ext';\n"),
    ]);
    let census = hardcoded_paths::scan(dir.path()).expect("scan");
    assert_eq!(census.files_scanned, 2, "{census:?}");
    assert_eq!(census.findings.len(), 2);
}

#[test]
fn an_unparseable_allowlist_is_a_failure_not_an_empty_list() {
    let dir = repo(&[(SEED, "pub const FINE: &str = \"ok\";\n")]);
    std::fs::create_dir_all(dir.path().join("state/registry")).unwrap();
    std::fs::write(
        dir.path().join(hardcoded_paths::ALLOWLIST_PATH),
        "{ not json",
    )
    .unwrap();
    let result = hardcoded_paths::check(dir.path());
    assert_eq!(result.status, Status::Fail, "{result:?}");
}

// ------------------------------------------------------------ the live repo

#[test]
fn this_workspace_names_nobodys_home_directory() {
    let census = hardcoded_paths::scan(&repo_root()).expect("scan");
    assert!(
        census.files_scanned > 100,
        "scanned {}",
        census.files_scanned
    );
    assert!(
        census.unlisted.is_empty(),
        "lines naming a home directory:\n{}",
        census
            .unlisted
            .iter()
            .map(|f| format!("  {}:{}  {}", f.file, f.line, f.excerpt))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        census.stale_allowlist.is_empty(),
        "stale allowlist entries: {:?}",
        census.stale_allowlist
    );
}
