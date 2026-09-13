use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("watch error: {0}")]
    Watch(#[from] notify::Error),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("connection error: {0}")]
    Connection(String),

    #[error("pipe error: {0}")]
    Pipe(String),

    #[error("timeout")]
    Timeout,

    /// A caller-supplied or crate-imposed bound was exceeded — an oversized
    /// HTTP body, a refused URL scheme, a zero batch size. Distinct from
    /// [`IoError::Parse`] on purpose: the input was well-formed, we declined
    /// to process that much of it.
    #[error("limit exceeded: {0}")]
    Limit(String),
}
