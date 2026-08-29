use clap::Parser;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use ix_skill::local_embed::{embed_request_with, require_cached_model, LocalEmbedError};
use serde_json::json;
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "ix-embed")]
struct Cli {
    #[arg(long)]
    model_cache: PathBuf,
    #[arg(long)]
    input: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    match run(cli) {
        Ok(response) => println!(
            "{}",
            serde_json::to_string(&response).expect("serializable")
        ),
        Err(error) => {
            eprintln!(
                "{}",
                json!({
                    "schema": "ix-local-embedding-error/1",
                    "code": error.code,
                    "message": error.message,
                })
            );
            std::process::exit(error.exit_code);
        }
    }
}

fn run(cli: Cli) -> Result<ix_skill::local_embed::LocalEmbeddingResponse, LocalEmbedError> {
    let (_, revision) = require_cached_model(&cli.model_cache)?;
    let request = if cli.input.as_os_str() == "-" {
        let mut value = String::new();
        std::io::stdin().read_to_string(&mut value).map(|_| value)
    } else {
        std::fs::read_to_string(&cli.input)
    }
    .map_err(|_| LocalEmbedError {
        code: "REQUEST_INVALID",
        message: "request file is not readable UTF-8".to_owned(),
        exit_code: 64,
    })?;

    // All required cache objects were resolved above. A cache race may still remove one;
    // forcing the hub endpoint to loopback makes that failure local instead of downloading.
    std::env::set_var("HF_HOME", &cli.model_cache);
    std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:9");
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGEBaseENV15)
            .with_cache_dir(cli.model_cache)
            .with_show_download_progress(false),
    )
    .map_err(|_| LocalEmbedError {
        code: "MODEL_RUNTIME_UNAVAILABLE",
        message: "cached local embedding model could not be initialized".to_owned(),
        exit_code: 10,
    })?;
    embed_request_with(&request, &revision, |texts| {
        model.embed(texts, None).map_err(|_| LocalEmbedError {
            code: "EMBEDDING_RUNTIME_FAILED",
            message: "local embedding inference failed".to_owned(),
            exit_code: 10,
        })
    })
}
