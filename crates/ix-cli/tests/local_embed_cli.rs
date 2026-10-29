#![cfg(feature = "embeddings")]

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn local_embed_refuses_an_incomplete_cache_without_downloading() {
    let dir = tempfile::tempdir().expect("temp dir");
    let request = dir.path().join("request.json");
    std::fs::write(
        &request,
        r#"{"schema":"ix-local-embedding-request/1","mode":"passage","items":[{"id":"doc-a","text":"fencing tokens"}]}"#,
    )
    .expect("request fixture");

    Command::cargo_bin("ix-embed")
        .expect("ix-embed binary built")
        .args([
            "--model-cache",
            dir.path().to_str().expect("UTF-8 cache path"),
            "--input",
            request.to_str().expect("UTF-8 request path"),
        ])
        .env("HF_ENDPOINT", "https://must-not-be-used.invalid")
        .assert()
        .code(3)
        .stdout(predicate::str::is_empty())
        .stderr(
            predicate::str::contains("\"schema\":\"ix-local-embedding-error/1\"").and(
                predicate::str::contains("\"code\":\"MODEL_CACHE_INCOMPLETE\""),
            ),
        );
}

#[test]
fn local_embed_emits_vectors_from_a_preseeded_cache() {
    let Ok(cache) = std::env::var("IX_TEST_FASTEMBED_CACHE") else {
        eprintln!("SKIP: IX_TEST_FASTEMBED_CACHE is not set");
        return;
    };
    let output = Command::cargo_bin("ix-embed")
        .expect("ix-embed binary built")
        .args(["--model-cache", &cache, "--input", "-"])
        .write_stdin(
            r#"{"schema":"ix-local-embedding-request/1","mode":"passage","items":[{"id":"doc-a","text":"Leases and fencing tokens prevent stale writers."}]}"#,
        )
        .env("HF_ENDPOINT", "http://127.0.0.1:9")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let response: serde_json::Value = serde_json::from_slice(&output).expect("response JSON");
    assert_eq!(response["schema"], "ix-local-embedding-response/1");
    assert_eq!(response["model"]["id"], "Xenova/bge-base-en-v1.5");
    assert_eq!(response["model"]["dimensions"], 768);
    assert_eq!(response["model"]["localOnly"], true);
    assert_eq!(response["items"][0]["id"], "doc-a");
    let vector = response["items"][0]["embedding"]
        .as_array()
        .expect("embedding array");
    assert_eq!(vector.len(), 768);
    assert!(vector
        .iter()
        .any(|value| value.as_f64().unwrap_or(0.0) != 0.0));
}
