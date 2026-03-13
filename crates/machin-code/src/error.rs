use thiserror::Error;

#[derive(Debug, Error)]
pub enum CodeError {
    #[error("unsupported language: {0} (compile with the `lang-{0}` feature)")]
    UnsupportedLanguage(String),

    #[error("parse failed: tree-sitter returned None (possible timeout)")]
    ParseFailed,

    #[error("empty input: source code is empty or whitespace-only")]
    EmptyInput,

    #[error("tree too large: {0} named nodes exceeds limit of {1} for dense adjacency (use adjacency_graph instead)")]
    TreeTooLarge(usize, usize),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
