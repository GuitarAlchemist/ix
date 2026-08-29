use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MODEL_ID: &str = "Xenova/bge-base-en-v1.5";
pub const MODEL_REVISION: &str = "4d6cd88e18e51a5e020c2c305726d76ada9c03cf";
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";
const MODEL_ARTIFACTS: [(&str, &str); 5] = [
    (
        "config.json",
        "d83c21fa7366994560727112ef0a31d8a2ec1c280c2a3e66326fdb877f64c91e",
    ),
    (
        "special_tokens_map.json",
        "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    ),
    (
        "tokenizer_config.json",
        "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    ),
    (
        "tokenizer.json",
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    ),
    (
        "onnx/model.onnx",
        "9bc579acdba21c253c62a9bf866891355a63ffa3442b52c8a37d75b2ccb91848",
    ),
];

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
    if revision != MODEL_REVISION {
        return Err(cache_integrity());
    }
    let snapshot = repository.join("snapshots").join(&revision);
    for (relative, trusted_sha256) in MODEL_ARTIFACTS {
        let physical_file = snapshot
            .join(relative)
            .canonicalize()
            .map_err(|_| cache_incomplete())?;
        let metadata = std::fs::metadata(&physical_file).map_err(|_| cache_incomplete())?;
        if !physical_file.starts_with(&physical_cache) || !metadata.is_file() || metadata.len() == 0
        {
            return Err(cache_incomplete());
        }
        if hash_file_sha256(&physical_file)? != trusted_sha256 {
            return Err(cache_integrity());
        }
    }
    Ok((repository, revision))
}

fn hash_file_sha256(path: &Path) -> Result<String, LocalEmbedError> {
    let mut file = std::fs::File::open(path).map_err(|_| cache_incomplete())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(|_| cache_incomplete())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn cache_incomplete() -> LocalEmbedError {
    LocalEmbedError {
        code: "MODEL_CACHE_INCOMPLETE",
        message: format!("{MODEL_ID} is not fully present in the declared local cache"),
        exit_code: 3,
    }
}

fn cache_integrity() -> LocalEmbedError {
    LocalEmbedError {
        code: "MODEL_CACHE_INTEGRITY",
        message: format!("{MODEL_ID} cache bytes do not match the pinned trusted manifest"),
        exit_code: 3,
    }
}
