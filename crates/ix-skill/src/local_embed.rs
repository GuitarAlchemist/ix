use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

pub const MODEL_ID: &str = "Xenova/bge-base-en-v1.5";
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

#[derive(Debug)]
pub struct LocalEmbedError {
    pub code: &'static str,
    pub message: String,
    pub exit_code: i32,
}

#[derive(Deserialize)]
struct LocalEmbeddingRequest {
    schema: String,
    mode: String,
    items: Vec<LocalEmbeddingRequestItem>,
}

#[derive(Deserialize)]
struct LocalEmbeddingRequestItem {
    id: String,
    text: String,
}

#[derive(Debug, Serialize)]
pub struct LocalEmbeddingResponse {
    pub schema: &'static str,
    pub mode: String,
    pub model: LocalEmbeddingModel,
    pub items: Vec<LocalEmbeddingItem>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalEmbeddingModel {
    pub id: &'static str,
    pub revision: String,
    pub dimensions: usize,
    pub runtime: &'static str,
    pub local_only: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalEmbeddingItem {
    pub id: String,
    pub text_sha256: String,
    pub embedding: Vec<f32>,
}

pub fn embed_request_with<F>(
    request_json: &str,
    model_revision: &str,
    mut embed: F,
) -> Result<LocalEmbeddingResponse, LocalEmbedError>
where
    F: FnMut(Vec<String>) -> Result<Vec<Vec<f32>>, LocalEmbedError>,
{
    let request: LocalEmbeddingRequest = serde_json::from_str(request_json)
        .map_err(|_| invalid_request("request is not valid JSON"))?;
    if request.schema != "ix-local-embedding-request/1"
        || !matches!(request.mode.as_str(), "query" | "passage")
        || request.items.is_empty()
    {
        return Err(invalid_request(
            "request schema, mode, or items are invalid",
        ));
    }
    let mut ids = HashSet::with_capacity(request.items.len());
    if request.items.iter().any(|item| {
        item.id.trim().is_empty() || item.text.trim().is_empty() || !ids.insert(item.id.as_str())
    }) {
        return Err(invalid_request(
            "item ids must be unique and ids and text must be non-empty",
        ));
    }
    let texts = request
        .items
        .iter()
        .map(|item| {
            if request.mode == "query" {
                format!("{QUERY_PREFIX}{}", item.text)
            } else {
                item.text.clone()
            }
        })
        .collect();
    let vectors = embed(texts)?;
    let dimensions = vectors.first().map(Vec::len).unwrap_or(0);
    if vectors.len() != request.items.len()
        || dimensions == 0
        || vectors.iter().any(|vector| {
            vector.len() != dimensions || vector.iter().any(|value| !value.is_finite())
        })
    {
        return Err(LocalEmbedError {
            code: "EMBEDDING_RUNTIME_INVALID",
            message: "embedding runtime returned an invalid vector set".to_owned(),
            exit_code: 10,
        });
    }
    let items = request
        .items
        .into_iter()
        .zip(vectors)
        .map(|(item, embedding)| LocalEmbeddingItem {
            id: item.id,
            text_sha256: format!("{:x}", Sha256::digest(item.text.as_bytes())),
            embedding,
        })
        .collect();
    Ok(LocalEmbeddingResponse {
        schema: "ix-local-embedding-response/1",
        mode: request.mode,
        model: LocalEmbeddingModel {
            id: MODEL_ID,
            revision: model_revision.to_owned(),
            dimensions,
            runtime: "fastembed",
            local_only: true,
        },
        items,
    })
}

fn invalid_request(message: &str) -> LocalEmbedError {
    LocalEmbedError {
        code: "REQUEST_INVALID",
        message: message.to_owned(),
        exit_code: 64,
    }
}

pub fn require_cached_model(cache: &Path) -> Result<(PathBuf, String), LocalEmbedError> {
    let physical_cache = cache.canonicalize().map_err(|_| cache_incomplete())?;
    let repository = cache.join("models--Xenova--bge-base-en-v1.5");
    let revision = std::fs::read_to_string(repository.join("refs/main"))
        .map(|value| value.trim().to_owned())
        .map_err(|_| cache_incomplete())?;
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(cache_incomplete());
    }
    let snapshot = repository.join("snapshots").join(&revision);
    for relative in [
        "config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "onnx/model.onnx",
    ] {
        let physical_file = snapshot
            .join(relative)
            .canonicalize()
            .map_err(|_| cache_incomplete())?;
        let metadata = std::fs::metadata(&physical_file).map_err(|_| cache_incomplete())?;
        if !physical_file.starts_with(&physical_cache) || !metadata.is_file() || metadata.len() == 0
        {
            return Err(cache_incomplete());
        }
    }
    Ok((repository, revision))
}

fn cache_incomplete() -> LocalEmbedError {
    LocalEmbedError {
        code: "MODEL_CACHE_INCOMPLETE",
        message: format!("{MODEL_ID} is not fully present in the declared local cache"),
        exit_code: 3,
    }
}
