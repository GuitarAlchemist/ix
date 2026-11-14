//! The `rustfmt-touched` check of `ix doctor`: every Rust file a change adds or modifies is
//! rustfmt-clean.
//!
//! ix CI ran no format check, and main carries rustfmt drift in 77 files, so a whole-tree
//! `cargo fmt --check` fails every PR and `cargo fmt --all` rewrites files other work is editing.
//! The check holds a change to adding no drift: a touched file that rustfmt would change fails,
//! unless it was already unformatted at the base, which warns. These tests build a throwaway git
//! repo for each case and run the `rustfmt` on `PATH`.

use ix_cli::doctor::rustfmt_touched;
use ix_cli::doctor::Status;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

const CLEAN: &str = "fn main() {}\n";
const UNFORMATTED: &str = "fn main( ) {}\n";

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .expect("git, which this check runs on");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A repo whose first commit holds `files`, with `origin/main` pointing at it. Returns the repo
/// and that commit.
fn repo(files: &[(&str, &str)]) -> (TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let p = dir.path();
    git(p, &["init", "-q"]);
    for (k, v) in [
        ("user.email", "doctor@example.invalid"),
        ("user.name", "doctor"),
        ("core.autocrlf", "false"),
        ("commit.gpgsign", "false"),
    ] {
        git(p, &["config", k, v]);
    }
    write(
        p,
        "Cargo.toml",
        "[workspace]\nmembers = []\n[workspace.package]\nedition = \"2021\"\n",
    );
    for (path, body) in files {
        write(p, path, body);
    }
    git(p, &["add", "-A"]);
    git(p, &["commit", "-q", "-m", "base"]);
    let base = git(p, &["rev-parse", "HEAD"]);
    git(p, &["update-ref", "refs/remotes/origin/main", &base]);
    (dir, base)
}

fn write(root: &Path, path: &str, body: &str) {
    let f = root.join(path);
    std::fs::create_dir_all(f.parent().unwrap()).unwrap();
    std::fs::write(f, body).unwrap();
}

fn commit(root: &Path, msg: &str) {
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", msg]);
}

fn rustfmt() -> PathBuf {
    PathBuf::from("rustfmt")
}

#[test]
fn drift_added_to_a_touched_file_fails_and_names_it() {
    let (r, _) = repo(&[("src/a.rs", CLEAN)]);
    write(r.path(), "src/a.rs", UNFORMATTED);
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Fail, "{c:?}");
    assert!(c.summary.contains("src/a.rs"), "{}", c.summary);
}

#[test]
fn an_untouched_unformatted_file_is_not_reported() {
    let (r, _) = repo(&[("src/a.rs", CLEAN), ("src/old.rs", UNFORMATTED)]);
    write(r.path(), "src/a.rs", "fn main() {}\n\nfn other() {}\n");
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Ok, "{c:?}");
    assert!(!c.summary.contains("old.rs"), "{}", c.summary);
}

#[test]
fn a_file_already_unformatted_at_the_base_warns_rather_than_fails() {
    let (r, _) = repo(&[("src/old.rs", UNFORMATTED)]);
    write(r.path(), "src/old.rs", "fn main( ) {}\n\nfn more() {}\n");
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Warn, "{c:?}");
    assert!(c.summary.contains("src/old.rs"), "{}", c.summary);
}

#[test]
fn files_committed_staged_or_untracked_since_the_base_are_all_checked() {
    let (r, _) = repo(&[("src/a.rs", CLEAN)]);
    let p = r.path();
    write(p, "src/committed.rs", UNFORMATTED);
    commit(p, "on the branch");
    write(p, "src/staged.rs", UNFORMATTED);
    git(p, &["add", "src/staged.rs"]);
    write(p, "src/untracked.rs", UNFORMATTED);
    let c = rustfmt_touched::check(p, None, &rustfmt());
    assert_eq!(c.status, Status::Fail, "{c:?}");
    for f in ["src/committed.rs", "src/staged.rs", "src/untracked.rs"] {
        assert!(c.summary.contains(f), "{f} missing: {}", c.summary);
    }
}

#[test]
fn a_deleted_file_and_a_formatted_one_pass() {
    let (r, _) = repo(&[("src/a.rs", CLEAN), ("src/gone.rs", UNFORMATTED)]);
    std::fs::remove_file(r.path().join("src/gone.rs")).unwrap();
    write(r.path(), "src/b.rs", CLEAN);
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Ok, "{c:?}");
}

#[test]
fn files_are_formatted_in_the_workspace_edition() {
    // `async fn` does not parse as edition 2015, rustfmt's default.
    let (r, _) = repo(&[("src/a.rs", CLEAN)]);
    write(r.path(), "src/a.rs", "async fn f() {}\n");
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Ok, "{c:?}");
}

#[test]
fn crlf_line_endings_are_not_drift() {
    // A Windows checkout writes CRLF; rustfmt reading stdin answers in LF.
    let (r, _) = repo(&[("src/a.rs", CLEAN)]);
    write(
        r.path(),
        "src/a.rs",
        "fn main() {}\r\n\r\nfn other() {}\r\n",
    );
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Ok, "{c:?}");
}

#[test]
fn a_renamed_file_is_compared_with_its_old_path_at_the_base() {
    let (r, _) = repo(&[("src/old.rs", UNFORMATTED)]);
    let p = r.path();
    git(p, &["mv", "src/old.rs", "src/new.rs"]);
    commit(p, "rename");
    let c = rustfmt_touched::check(p, None, &rustfmt());
    assert_eq!(c.status, Status::Warn, "{c:?}");
    assert!(c.summary.contains("src/new.rs"), "{}", c.summary);
}

#[test]
fn a_touched_file_rustfmt_cannot_parse_warns() {
    let (r, _) = repo(&[("src/a.rs", CLEAN)]);
    write(r.path(), "src/a.rs", "fn main( {}\n");
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Warn, "{c:?}");
    assert!(c.summary.contains("src/a.rs"), "{}", c.summary);
}

#[test]
fn a_named_base_is_the_one_used() {
    let (r, base) = repo(&[("src/a.rs", CLEAN)]);
    let p = r.path();
    write(p, "src/b.rs", UNFORMATTED);
    commit(p, "drift");
    // origin/main moves past the drift; the named base still sees it as this change's.
    let head = git(p, &["rev-parse", "HEAD"]);
    git(p, &["update-ref", "refs/remotes/origin/main", &head]);
    assert_eq!(
        rustfmt_touched::check(p, None, &rustfmt()).status,
        Status::Ok
    );
    let c = rustfmt_touched::check(p, Some(&base), &rustfmt());
    assert_eq!(c.status, Status::Fail, "{c:?}");
    assert!(c.summary.contains("src/b.rs"), "{}", c.summary);
}

#[test]
fn without_origin_main_the_check_warns_that_it_did_not_run() {
    let (r, _) = repo(&[("src/a.rs", CLEAN)]);
    git(r.path(), &["update-ref", "-d", "refs/remotes/origin/main"]);
    write(r.path(), "src/a.rs", UNFORMATTED);
    let c = rustfmt_touched::check(r.path(), None, &rustfmt());
    assert_eq!(c.status, Status::Warn, "{c:?}");
    assert!(c.remedy.is_some(), "{c:?}");
}

#[test]
fn a_named_base_that_does_not_resolve_fails() {
    let (r, _) = repo(&[("src/a.rs", CLEAN)]);
    let c = rustfmt_touched::check(r.path(), Some("no-such-ref"), &rustfmt());
    assert_eq!(c.status, Status::Fail, "{c:?}");
}

#[test]
fn no_rustfmt_warns_and_fails_once_a_base_is_named() {
    let (r, base) = repo(&[("src/a.rs", CLEAN)]);
    write(r.path(), "src/a.rs", UNFORMATTED);
    let missing = Path::new("rustfmt-that-is-not-installed");
    assert_eq!(
        rustfmt_touched::check(r.path(), None, missing).status,
        Status::Warn
    );
    assert_eq!(
        rustfmt_touched::check(r.path(), Some(&base), missing).status,
        Status::Fail
    );
}
