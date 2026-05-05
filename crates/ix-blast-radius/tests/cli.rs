//! Integration test: drive the `ix-blast-radius` binary via stdin and
//! check the JSON output. Heavy logic tests live in `src/lib.rs`.

use std::io::Write;

#[test]
fn binary_reads_stdin_and_emits_json() {
    let exe = env!("CARGO_BIN_EXE_ix-blast-radius");
    let mut child = std::process::Command::new(exe)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn ix-blast-radius");
    let stdin = child.stdin.as_mut().expect("stdin handle");
    stdin
        .write_all(b"crates/ga-chatbot/src/coverage.rs\n")
        .expect("write stdin");
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait child");
    assert!(out.status.success(), "non-zero exit: {:?}", out.status);
    let stdout = String::from_utf8(out.stdout).expect("utf-8 stdout");
    assert!(stdout.contains("\"layers_touched\""));
    assert!(stdout.contains("\"domain\""));
    assert!(stdout.contains("\"ga-chatbot\""));
}

#[test]
fn binary_handles_empty_input_gracefully() {
    let exe = env!("CARGO_BIN_EXE_ix-blast-radius");
    let mut child = std::process::Command::new(exe)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn");
    drop(child.stdin.take());
    let out = child.wait_with_output().expect("wait child");
    assert!(out.status.success());
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert!(stdout.contains("\"estimated_blast_score\":0"));
}
