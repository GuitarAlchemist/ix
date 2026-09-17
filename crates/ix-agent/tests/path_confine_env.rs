//! Path confinement that depends on the operator's environment: `IX_ROOT`,
//! `IX_EXTRA_ROOTS` and the home directory behind `~/.ga/traces`, exercised
//! through the MCP entry point.
//!
//! Own test binary with a single test, so changing the environment races no
//! other test.

use ix_agent::registry_bridge::shared_loop_detector;
use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::json;
use std::path::{Path, PathBuf};

#[cfg(unix)]
fn link_dir(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

/// Symlinks need Developer Mode or elevation on Windows; a directory junction
/// does not.
#[cfg(windows)]
fn link_dir(target: &Path, link: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(target, link).or_else(|_| {
        std::process::Command::new("cmd")
            .arg("/C")
            .arg("mklink")
            .arg("/J")
            .arg(link)
            .arg(target)
            .output()
            .and_then(|o| {
                if o.status.success() {
                    Ok(())
                } else {
                    Err(std::io::Error::other("mklink /J failed"))
                }
            })
    })
}

fn same_file(a: &Path, b: &Path) -> bool {
    a.canonicalize().unwrap() == b.canonicalize().unwrap()
}

#[test]
fn confinement_follows_the_operator_environment() {
    let saved: Vec<(&str, Option<std::ffi::OsString>)> =
        ["IX_ROOT", "IX_EXTRA_ROOTS", "HOME", "USERPROFILE"]
            .into_iter()
            .map(|k| (k, std::env::var_os(k)))
            .collect();

    let (ctx, _rx) = ServerContext::new();
    let registry = ToolRegistry::new();
    let call = |tool: &str, args: serde_json::Value| {
        shared_loop_detector().clear_key(tool);
        registry.call_with_ctx(tool, args, &ctx)
    };

    let base = tempfile::tempdir().unwrap();
    let root = base.path().join("ws");
    let sibling = base.path().join("sibling");
    let home = base.path().join("home");
    for dir in [&root, &sibling, &home] {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(root.join("in.rs"), "fn a() {}").unwrap();
    std::fs::write(sibling.join("sib.rs"), "fn b() {}").unwrap();
    let outside_file = base.path().join("outside.rs");
    std::fs::write(&outside_file, "fn c() {}").unwrap();
    std::env::set_var("HOME", &home);
    std::env::set_var("USERPROFILE", &home);
    std::env::remove_var("IX_EXTRA_ROOTS");

    // A root that is the home directory or a volume root fails closed.
    let volume = base
        .path()
        .canonicalize()
        .unwrap()
        .ancestors()
        .last()
        .unwrap()
        .to_path_buf();
    for bad_root in [home.clone(), volume] {
        std::env::set_var("IX_ROOT", &bad_root);
        let err = call("ix_code_analyze", json!({ "path": "in.rs" }))
            .expect_err("a home or volume root must refuse every path");
        assert!(
            err.contains("volume root or the home directory"),
            "{}: {err}",
            bad_root.display()
        );
    }

    // Relative IX_EXTRA_ROOTS entries resolve against IX_ROOT, not the cwd.
    std::env::set_var("IX_ROOT", &root);
    std::env::set_var("IX_EXTRA_ROOTS", "../sibling");
    call("ix_code_analyze", json!({ "path": "in.rs" })).expect("in-root file");
    let sib = sibling.join("sib.rs");
    call("ix_code_analyze", json!({ "path": sib.to_str().unwrap() }))
        .expect("a file in a relative extra root must be admitted");
    let err = call(
        "ix_code_analyze",
        json!({ "path": outside_file.to_str().unwrap() }),
    )
    .expect_err("a file outside every root must be refused");
    assert!(err.contains("inside an allowed root"), "{err}");
    std::env::remove_var("IX_EXTRA_ROOTS");
    call("ix_code_analyze", json!({ "path": sib.to_str().unwrap() }))
        .expect_err("without IX_EXTRA_ROOTS the sibling is outside");

    // Export destinations: ~/.ga/traces (relative paths resolve against it), but
    // not the workspace.
    let traces = home.join(".ga").join("traces");
    drop(ix_session::SessionLog::open(root.join("session.jsonl")).unwrap());
    let out = call(
        "ix_session_flywheel_export",
        json!({ "session_log": "session.jsonl" }),
    )
    .expect("the default destination must export");
    let written = PathBuf::from(out["written"].as_str().unwrap());
    assert!(same_file(&written, &traces.join("session.json")), "{out}");

    let out = call(
        "ix_session_flywheel_export",
        json!({ "session_log": "session.jsonl", "trace_dir": "sub" }),
    )
    .expect("a relative destination under ~/.ga/traces must export");
    let written = PathBuf::from(out["written"].as_str().unwrap());
    assert!(
        same_file(&written, &traces.join("sub/session.json")),
        "{out}"
    );

    let err = call(
        "ix_session_flywheel_export",
        json!({ "session_log": "session.jsonl", "trace_dir": root.to_str().unwrap() }),
    )
    .expect_err("the workspace is not an export destination");
    assert!(err.contains("not inside an allowed destination root"), "{err}");

    // An in-root directory is not a session log.
    std::fs::create_dir_all(root.join("logs")).unwrap();
    let err = call(
        "ix_session_flywheel_export",
        json!({ "session_log": "logs" }),
    )
    .expect_err("a directory is not a session log");
    assert!(err.contains("is not a file"), "{err}");

    // A dangling link inside the trace root is refused, not skipped past.
    let gone = base.path().join("gone");
    std::fs::create_dir_all(&gone).unwrap();
    link_dir(&gone, &traces.join("dang")).expect("create a symlink or junction");
    std::fs::remove_dir(&gone).unwrap();
    let err = call(
        "ix_session_flywheel_export",
        json!({ "session_log": "session.jsonl", "trace_dir": "dang/x" }),
    )
    .expect_err("a destination through a dangling link must be refused");
    assert!(err.contains("not inside an allowed destination root"), "{err}");
    assert!(!gone.exists(), "nothing is created at the link target");

    // A link planted at the output file is replaced, not written through.
    let victim = base.path().join("victim.json");
    std::fs::write(&victim, "KEEP").unwrap();
    std::fs::remove_file(traces.join("session.json")).unwrap();
    std::fs::hard_link(&victim, traces.join("session.json")).unwrap();
    call(
        "ix_session_flywheel_export",
        json!({ "session_log": "session.jsonl" }),
    )
    .expect("export over a planted link");
    assert_eq!(std::fs::read_to_string(&victim).unwrap(), "KEEP");

    for (key, value) in saved {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
}
