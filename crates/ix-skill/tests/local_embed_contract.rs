#![cfg(feature = "embeddings")]

use ix_skill::local_embed::embed_request_with;

#[test]
fn query_embedding_response_binds_model_revision_text_and_dimensions() {
    let request = r#"{"schema":"ix-local-embedding-request/1","mode":"query","items":[{"id":"query-a","text":"fencing tokens"}]}"#;
    let mut observed = Vec::new();

    let response = embed_request_with(
        request,
        "4d6cd88e18e51a5e020c2c305726d76ada9c03cf",
        |texts| {
            observed = texts;
            Ok(vec![vec![0.6, 0.8]])
        },
    )
    .expect("valid response");

    assert_eq!(
        observed,
        ["Represent this sentence for searching relevant passages: fencing tokens"]
    );
    assert_eq!(response.schema, "ix-local-embedding-response/1");
    assert_eq!(response.mode, "query");
    assert_eq!(response.model.id, "Xenova/bge-base-en-v1.5");
    assert_eq!(
        response.model.revision,
        "4d6cd88e18e51a5e020c2c305726d76ada9c03cf"
    );
    assert_eq!(response.model.dimensions, 2);
    assert!(response.model.local_only);
    assert_eq!(response.items[0].id, "query-a");
    assert_eq!(
        response.items[0].text_sha256,
        "0545c54e24c94e1e96754e48d9087e14007f9f5aa5075e5d2c8408e98f28aa6c"
    );
    assert_eq!(response.items[0].embedding, [0.6, 0.8]);
}

#[test]
fn embedding_request_refuses_duplicate_ids_before_inference() {
    let request = r#"{"schema":"ix-local-embedding-request/1","mode":"passage","items":[{"id":"same","text":"one"},{"id":"same","text":"two"}]}"#;
    let mut called = false;

    let error = embed_request_with(request, "4d6cd88e18e51a5e020c2c305726d76ada9c03cf", |_| {
        called = true;
        Ok(vec![vec![1.0], vec![1.0]])
    })
    .expect_err("duplicate ids must be refused");

    assert_eq!(error.code, "REQUEST_INVALID");
    assert!(!called, "invalid requests must not reach model inference");
}

#[test]
fn embedding_response_refuses_non_finite_vectors() {
    let request = r#"{"schema":"ix-local-embedding-request/1","mode":"passage","items":[{"id":"passage-a","text":"fencing tokens"}]}"#;

    let error = embed_request_with(request, "4d6cd88e18e51a5e020c2c305726d76ada9c03cf", |_| {
        Ok(vec![vec![f32::NAN, 1.0]])
    })
    .expect_err("non-finite vectors must be refused");

    assert_eq!(error.code, "EMBEDDING_RUNTIME_INVALID");
}
