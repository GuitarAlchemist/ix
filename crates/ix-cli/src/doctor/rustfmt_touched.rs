//! Rustfmt check scoped to the change — every Rust file it adds or modifies is formatted.
//!
//! main carries rustfmt drift (77 files on 2026-10-22), so a whole-tree `cargo fmt --check` fails
//! every pull request, and `cargo fmt --all` rewrites files that other work is editing. Until this
//! check, CI ran neither, and each change could add drift unseen.
//!
//! The change is the diff from the merge-base of `HEAD` and a base ref to the working tree, plus
//! untracked files: what is committed, staged or only written since the branch left the base.
//! Each touched `.rs` file is piped through `rustfmt` in the workspace edition and compared with
//! what rustfmt gives back, line endings aside (a Windows checkout writes CRLF, and rustfmt reading
//! stdin answers in LF). A file rustfmt would change fails the check, unless the same file
//! (under its old name, for a rename) was already unformatted at the base: that drift is not this
//! change's, so it warns. A file rustfmt cannot parse warns too; the compiler is the gate for
//! that.
//!
//! # Base and strictness
//!
//! CI names the base in [`BASE_ENV`]. A named base is strict: if it does not resolve, or rustfmt
//! cannot be run, the check fails, since CI asked for it and it did not happen. With no base named
//! the check uses [`DEFAULT_BASE`] and warns, rather than fails, when that ref or rustfmt is
//! missing: a fresh clone or a shallow checkout has no `origin/main`.

use super::{CheckResult, Status};
use serde_json::json;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

const NAME: &str = "rustfmt-touched";

/// The base when none is named: the branch every pull request targets.
pub const DEFAULT_BASE: &str = "origin/main";

/// Names the base ref and makes the check strict. CI sets it.
pub const BASE_ENV: &str = "IX_DOCTOR_BASE";

/// A touched file, and the path it had at the base when it existed there.
struct Touched {
    path: String,
    base_path: Option<String>,
}

/// What rustfmt made of one file's text.
enum Verdict {
    Clean,
    Drift,
    /// rustfmt did not give the file back: it did not parse, or rustfmt failed.
    Unparsed,
}

/// The check: scope the change, then run rustfmt on what it touches.
pub fn check(root: &Path, base: Option<&str>, rustfmt: &Path) -> CheckResult {
    let strict = base.is_some();
    let base_ref = base.unwrap_or(DEFAULT_BASE);
    let not_run = |summary: String, remedy: String| {
        let status = if strict { Status::Fail } else { Status::Warn };
        CheckResult::new(NAME, status, summary).with_remedy(remedy)
    };

    let runs = Command::new(rustfmt)
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success());
    if !runs {
        return not_run(
            format!(
                "{} did not run: the touched Rust files were not checked",
                rustfmt.display()
            ),
            "`rustup component add rustfmt` for the toolchain in use — a green doctor without \
             this step has not looked at the format"
                .to_string(),
        );
    }

    let edition = match workspace_edition(root) {
        Ok(e) => e,
        Err(e) => {
            return CheckResult::new(NAME, Status::Fail, e).with_remedy(
                "the root Cargo.toml names the edition rustfmt formats in, under \
                 [workspace.package]",
            )
        }
    };

    let merge_base = match git(root, &["merge-base", "HEAD", base_ref]) {
        Ok(out) => String::from_utf8_lossy(&out).trim().to_string(),
        Err(e) => {
            return not_run(
                format!("no merge-base with {base_ref} ({e}): the change was not scoped"),
                format!(
                    "`git fetch origin main`, or name the base with {BASE_ENV}=<ref>; CI names \
                     it, and there an unresolvable base fails"
                ),
            )
        }
    };

    let touched = match touched_files(root, &merge_base) {
        Ok(t) => t,
        Err(e) => {
            return CheckResult::new(
                NAME,
                Status::Fail,
                format!("could not list the touched files: {e}"),
            )
            .with_remedy("a listing that failed is a scan that did not happen, not an empty one")
        }
    };

    let mut new_drift = Vec::new();
    let mut old_drift = Vec::new();
    let mut unparsed = Vec::new();
    for t in &touched {
        let now = match std::fs::read(root.join(&t.path)) {
            Ok(b) => b,
            Err(e) => {
                return CheckResult::new(
                    NAME,
                    Status::Fail,
                    format!("could not read {}: {e}", t.path),
                )
                .with_remedy("a file that cannot be read was not checked")
            }
        };
        match verdict(root, rustfmt, &edition, &now) {
            Verdict::Clean => continue,
            Verdict::Unparsed => {
                unparsed.push(t.path.clone());
                continue;
            }
            Verdict::Drift => {}
        }
        let was_drifted = t
            .base_path
            .as_ref()
            .and_then(|p| git(root, &["show", &format!("{merge_base}:{p}")]).ok())
            .is_some_and(|was| matches!(verdict(root, rustfmt, &edition, &was), Verdict::Drift));
        if was_drifted {
            old_drift.push(t.path.clone());
        } else {
            new_drift.push(t.path.clone());
        }
    }

    let short = &merge_base[..merge_base.len().min(10)];
    let details = json!({
        "base": base_ref,
        "merge_base": merge_base,
        "edition": edition,
        "touched": touched.iter().map(|t| &t.path).collect::<Vec<_>>(),
        "new_drift": new_drift,
        "old_drift": old_drift,
        "unparsed": unparsed,
    });

    let mut notes = Vec::new();
    if !old_drift.is_empty() {
        notes.push(format!(
            "already unformatted at the base: {}",
            old_drift.join(", ")
        ));
    }
    if !unparsed.is_empty() {
        notes.push(format!("rustfmt could not parse: {}", unparsed.join(", ")));
    }

    if !new_drift.is_empty() {
        let mut summary = format!(
            "rustfmt would change {} file(s) this change touches: {}",
            new_drift.len(),
            new_drift.join(", ")
        );
        for n in &notes {
            summary.push_str("; ");
            summary.push_str(n);
        }
        return CheckResult::new(NAME, Status::Fail, summary)
            .with_remedy(format!(
                "`rustfmt --edition {edition} <file>` on each one, never `cargo fmt --all`, which \
                 rewrites the drift main carries in files other work is editing; a fixture that \
                 must stay unformatted takes `#![rustfmt::skip]`"
            ))
            .with_details(details);
    }

    if !notes.is_empty() {
        return CheckResult::new(NAME, Status::Warn, notes.join("; "))
            .with_remedy(format!(
                "formatting a file you touch is welcome and not required when the drift predates \
                 the change (`rustfmt --edition {edition} <file>`); a file rustfmt cannot parse \
                 is the compiler's to report"
            ))
            .with_details(details);
    }

    CheckResult::new(
        NAME,
        Status::Ok,
        format!(
            "{} touched Rust file(s) since {base_ref} ({short}), all rustfmt-clean",
            touched.len()
        ),
    )
    .with_details(details)
}

/// `[workspace.package] edition` of the root manifest.
fn workspace_edition(root: &Path) -> Result<String, String> {
    let path = root.join("Cargo.toml");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    let doc: toml::Table =
        toml::from_str(&text).map_err(|e| format!("could not parse {}: {e}", path.display()))?;
    doc.get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("edition"))
        .and_then(|e| e.as_str())
        .map(str::to_string)
        .ok_or_else(|| format!("{} sets no [workspace.package] edition", path.display()))
}

/// The `.rs` files changed from `merge_base` to the working tree, then the untracked ones.
fn touched_files(root: &Path, merge_base: &str) -> Result<Vec<Touched>, String> {
    let diff = git(
        root,
        &[
            "diff",
            "-z",
            "--name-status",
            "--diff-filter=ACMR",
            "-M",
            merge_base,
            "--",
            "*.rs",
        ],
    )?;
    let mut fields = diff
        .split(|&b| b == 0)
        .filter(|f| !f.is_empty())
        .map(|f| String::from_utf8_lossy(f).into_owned());
    let mut out = Vec::new();
    while let Some(status) = fields.next() {
        let first = fields
            .next()
            .ok_or_else(|| format!("git diff: no path after status {status}"))?;
        let touched = match status.as_bytes().first() {
            Some(b'A') => Touched {
                path: first,
                base_path: None,
            },
            Some(b'R' | b'C') => {
                let path = fields
                    .next()
                    .ok_or_else(|| format!("git diff: no new path after {first}"))?;
                Touched {
                    path,
                    base_path: Some(first),
                }
            }
            _ => Touched {
                base_path: Some(first.clone()),
                path: first,
            },
        };
        out.push(touched);
    }

    let others = git(
        root,
        &[
            "ls-files",
            "-z",
            "--others",
            "--exclude-standard",
            "--",
            "*.rs",
        ],
    )?;
    out.extend(
        others
            .split(|&b| b == 0)
            .filter(|f| !f.is_empty())
            .map(|f| Touched {
                path: String::from_utf8_lossy(f).into_owned(),
                base_path: None,
            }),
    );
    Ok(out)
}

/// Pipe `text` through rustfmt and compare what comes back, both sides in LF.
///
/// Not `--check`: reading stdin, rustfmt 1.8 prints the diff and still exits 0. Stdin rather
/// than the path because a path makes rustfmt format the file's `mod` children too.
fn verdict(root: &Path, rustfmt: &Path, edition: &str, text: &[u8]) -> Verdict {
    let input = lf(text);
    let child = Command::new(rustfmt)
        .args(["--edition", edition])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let Ok(mut child) = child else {
        return Verdict::Unparsed;
    };
    let mut stdin = child.stdin.take().expect("stdin is piped");
    let fed = input.clone();
    // A writer of its own, so a file larger than the pipe cannot wedge against unread stdout.
    let writer = std::thread::spawn(move || stdin.write_all(&fed));
    let Ok(out) = child.wait_with_output() else {
        return Verdict::Unparsed;
    };
    let _ = writer.join();
    if !out.status.success() {
        return Verdict::Unparsed;
    }
    if lf(&out.stdout) == input {
        Verdict::Clean
    } else {
        Verdict::Drift
    }
}

fn lf(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut bytes = text.iter().peekable();
    while let Some(&b) = bytes.next() {
        if b == b'\r' && bytes.peek() == Some(&&b'\n') {
            continue;
        }
        out.push(b);
    }
    out
}

fn git(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|e| format!("git did not run: {e}"))?;
    if out.status.success() {
        Ok(out.stdout)
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}
