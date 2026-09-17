//! `ix_session_flywheel_export` runs without a prompt, so its destination is
//! confined through the MCP entry point: `trace_dir` must lie inside
//! `~/.ga/traces` or the `traces/` directory beside the installed session log,
//! `session_log` must be an existing file, and the export never writes through
//! or clears the protection of whatever sits at `<trace_dir>/<trace_id>.json`.
//!
//! Own test binary with a single test: it installs the process-wide session
//! log and points `HOME`/`USERPROFILE` at a temporary directory, so
//! `~/.ga/traces` never touches the real home directory.

use ix_agent::registry_bridge::{clear_session_log, install_session_log, shared_loop_detector};
use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;
use serde_json::{json, Value};
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

#[test]
#[allow(clippy::permissions_set_readonly_false)]
fn flywheel_export_confines_its_destination() {
    let home = tempfile::tempdir().unwrap();
    // This binary runs a single test, so no other thread reads the environment.
    std::env::set_var("HOME", home.path());
    std::env::set_var("USERPROFILE", home.path());
    let ga_traces = home.path().join(".ga").join("traces");

    let (ctx, _rx) = ServerContext::new();
    let registry = ToolRegistry::new();
    let call = |args: Value| {
        shared_loop_detector().clear_key("ix_session_flywheel_export");
        registry.call_with_ctx("ix_session_flywheel_export", args, &ctx)
    };

    let work = tempfile::tempdir().unwrap();
    let log_path = work.path().join("session.jsonl");
    install_session_log(ix_session::SessionLog::open(&log_path).unwrap());
    let log = log_path.to_str().unwrap();
    let beside_log = work.path().join("traces");
    let beside_log_arg = beside_log.to_str().unwrap();
    let export_beside_log = |id: &str| {
        call(json!({ "session_log": log, "trace_dir": beside_log_arg, "trace_id": id }))
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // ── trace_dir outside the trace roots: refused, nothing created ──
        let outside = tempfile::tempdir().unwrap();
        let victim = outside.path().join("session.json");
        std::fs::write(&victim, "KEEP").unwrap();
        let new_outside = outside.path().join("new");
        for dir in [outside.path().to_path_buf(), new_outside.clone()] {
            let err = call(json!({ "session_log": log, "trace_dir": dir.to_str().unwrap() }))
                .expect_err("a trace_dir outside the trace roots must be refused");
            assert!(err.contains("not inside an allowed destination root"), "{err}");
        }
        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "KEEP");
        assert!(!new_outside.exists());
        // The workspace-like parent of the log is not a trace root either.
        let err = call(json!({ "session_log": log, "trace_dir": work.path().to_str().unwrap() }))
            .expect_err("the log's own directory is not a trace root");
        assert!(err.contains("not inside an allowed destination root"), "{err}");

        // ── path shapes refused before anything is resolved ──
        let dotdot = format!("{}/../escape", beside_log.display());
        let err = call(json!({ "session_log": log, "trace_dir": dotdot }))
            .expect_err("`..` must be refused");
        assert!(err.contains("`..` is not allowed"), "{err}");
        assert!(!work.path().join("escape").exists());
        #[cfg(windows)]
        for raw in [
            r"\\attacker.example\share\traces",
            r"\\.\pipe\ix-flywheel-test",
            r"\\?\C:\traces",
            "C:traces",
        ] {
            let err = call(json!({ "session_log": log, "trace_dir": raw }))
                .expect_err("a non-local path shape must be refused");
            assert!(err.contains("absolute path on a local drive"), "{raw}: {err}");
        }

        // ── session_log must be an existing file and is never created ──
        let missing_log = work.path().join("sub").join("missing.jsonl");
        let err = call(json!({ "session_log": missing_log.to_str().unwrap() }))
            .expect_err("a missing session_log must be refused");
        assert!(err.contains("is not an existing file"), "{err}");
        assert!(!missing_log.exists() && !work.path().join("sub").exists());
        let err = call(json!({ "session_log": work.path().to_str().unwrap() }))
            .expect_err("a directory is not a session log");
        assert!(err.contains("is not an existing file"), "{err}");

        // ── in-root exports work ──
        let out = call(json!({ "session_log": log, "trace_dir": beside_log.to_str().unwrap() }))
            .expect("the traces/ directory beside the session log is a trace root");
        let written = PathBuf::from(out["written"].as_str().unwrap());
        assert_eq!(
            written.canonicalize().unwrap(),
            beside_log.join("session.json").canonicalize().unwrap()
        );
        let out = call(json!({ "session_log": log, "trace_dir": "nested", "trace_id": "rel" }))
            .expect("a relative trace_dir resolves against ~/.ga/traces");
        let written = PathBuf::from(out["written"].as_str().unwrap());
        assert!(written.exists(), "{out}");
        assert!(ga_traces.join("nested").join("rel.json").is_file(), "{out}");
        let out = call(json!({ "session_log": log, "trace_id": "default" }))
            .expect("the default trace_dir is ~/.ga/traces");
        assert!(ga_traces.join("default.json").is_file(), "{out}");

        // ── a read-only destination is not replaced ──
        let read_only = beside_log.join("ro.json");
        std::fs::write(&read_only, "KEEP").unwrap();
        let mut perms = std::fs::metadata(&read_only).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&read_only, perms.clone()).unwrap();
        let err = export_beside_log("ro")
            .expect_err("a read-only destination must not be replaced");
        assert!(err.contains("read-only"), "{err}");
        assert_eq!(std::fs::read_to_string(&read_only).unwrap(), "KEEP");
        perms.set_readonly(false);
        std::fs::set_permissions(&read_only, perms).unwrap();

        // ── a directory or a link at the destination is refused ──
        let as_dir = beside_log.join("dir.json");
        std::fs::create_dir(&as_dir).unwrap();
        std::fs::write(as_dir.join("inner"), "KEEP").unwrap();
        let err = export_beside_log("dir")
            .expect_err("a directory at the destination must be refused");
        assert!(err.contains("not a regular file"), "{err}");
        assert_eq!(std::fs::read_to_string(as_dir.join("inner")).unwrap(), "KEEP");

        let link_target = tempfile::tempdir().unwrap();
        std::fs::write(link_target.path().join("inner"), "KEEP").unwrap();
        link_dir(link_target.path(), &beside_log.join("link.json"))
            .expect("create a symlink or junction");
        let err = export_beside_log("link")
            .expect_err("a symlink or junction at the destination must be refused");
        assert!(err.contains("not a regular file"), "{err}");
        assert!(beside_log.join("link.json").symlink_metadata().is_ok());
        assert_eq!(
            std::fs::read_to_string(link_target.path().join("inner")).unwrap(),
            "KEEP"
        );
        assert_eq!(std::fs::read_dir(link_target.path()).unwrap().count(), 1);

        // ── a failed replace keeps the previous trace ──
        // An open handle that does not share delete access makes the rename
        // over the destination fail on Windows.
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            let previous = beside_log.join("locked.json");
            std::fs::write(&previous, "PREVIOUS").unwrap();
            let before: Vec<_> = std::fs::read_dir(&beside_log).unwrap().collect();
            let lock = std::fs::OpenOptions::new()
                .read(true)
                .share_mode(1) // FILE_SHARE_READ only
                .open(&previous)
                .unwrap();
            let err = export_beside_log("locked")
                .expect_err("the rename over a locked destination must fail");
            drop(lock);
            assert!(err.contains("locked.json"), "{err}");
            assert_eq!(std::fs::read_to_string(&previous).unwrap(), "PREVIOUS");
            let after: Vec<_> = std::fs::read_dir(&beside_log).unwrap().collect();
            assert_eq!(before.len(), after.len(), "no temporary file is left behind");
        }

        // ── without HOME and USERPROFILE only the session-log root remains ──
        std::env::remove_var("HOME");
        std::env::remove_var("USERPROFILE");
        let err = call(json!({ "session_log": log }))
            .expect_err("no default trace_dir without a home directory");
        assert!(err.contains("neither HOME nor USERPROFILE"), "{err}");
        let err = call(json!({ "session_log": log, "trace_dir": ga_traces.to_str().unwrap() }))
            .expect_err("~/.ga/traces is no longer a root");
        assert!(err.contains("not inside an allowed destination root"), "{err}");
        assert!(export_beside_log("nohome").is_ok());
        clear_session_log();
        let err = call(json!({ "session_log": log, "trace_dir": beside_log.to_str().unwrap() }))
            .expect_err("no root at all without a home directory or a session log");
        assert!(err.contains("no allowed root"), "{err}");
    }));
    clear_session_log();
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
}
