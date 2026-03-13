use std::io::Read;
use tree_sitter::{Parser, Tree, Language};
use crate::error::CodeError;

const DEFAULT_PARSE_TIMEOUT_US: u64 = 30_000_000; // 30 seconds
const MAX_FILE_SIZE: u64 = 10 * 1024 * 1024; // 10 MB

/// A parsed source code tree with metadata.
///
/// Fields are private to enforce the invariant that `tree` was produced
/// by parsing `source` with the grammar for `language`.
pub struct CodeTree {
    tree: Tree,
    source: String,
    language_name: String,
    language: Language,
    has_errors: bool,
}

impl CodeTree {
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn language_name(&self) -> &str {
        &self.language_name
    }

    pub fn language(&self) -> &Language {
        &self.language
    }

    pub fn has_errors(&self) -> bool {
        self.has_errors
    }

    /// Extract the text for a node by slicing the owned source.
    ///
    /// Returns `None` if the byte range falls on a non-UTF-8 boundary
    /// (possible with tree-sitter error recovery on multi-byte input).
    pub fn node_text(&self, node: tree_sitter::Node) -> Option<&str> {
        self.source.get(node.byte_range())
    }
}

/// Resolve a language name to a tree-sitter Language.
///
/// Returns `None` if the language feature is not compiled in.
fn resolve_language(name: &str) -> Option<Language> {
    match name {
        #[cfg(feature = "lang-rust")]
        "rust" | "rs" => Some(tree_sitter_rust::LANGUAGE.into()),

        #[cfg(feature = "lang-python")]
        "python" | "py" => Some(tree_sitter_python::LANGUAGE.into()),

        #[cfg(feature = "lang-javascript")]
        "javascript" | "js" => Some(tree_sitter_javascript::LANGUAGE.into()),

        #[cfg(feature = "lang-java")]
        "java" => Some(tree_sitter_java::LANGUAGE.into()),

        #[cfg(feature = "lang-csharp")]
        "csharp" | "c#" | "cs" => Some(tree_sitter_c_sharp::LANGUAGE.into()),

        #[cfg(feature = "lang-cpp")]
        "cpp" | "c++" | "cxx" => Some(tree_sitter_cpp::LANGUAGE.into()),

        #[cfg(feature = "lang-go")]
        "go" | "golang" => Some(tree_sitter_go::LANGUAGE.into()),

        #[cfg(feature = "lang-typescript")]
        "typescript" | "ts" => Some(tree_sitter_typescript::LANGUAGE.into()),

        _ => None,
    }
}

/// Parse source code into a `CodeTree`.
///
/// # Arguments
/// * `language` - Language name (e.g., "rust", "python", "csharp")
/// * `source` - Source code string
///
/// # Errors
/// * `CodeError::EmptyInput` if source is empty or whitespace-only
/// * `CodeError::UnsupportedLanguage` if the language feature is not compiled in
/// * `CodeError::ParseFailed` if tree-sitter fails to produce a tree (e.g., timeout)
pub fn parse(language: &str, source: &str) -> Result<CodeTree, CodeError> {
    if source.trim().is_empty() {
        return Err(CodeError::EmptyInput);
    }

    let ts_lang = resolve_language(language)
        .ok_or_else(|| CodeError::UnsupportedLanguage(language.to_string()))?;

    let mut parser = Parser::new();
    parser
        .set_language(&ts_lang)
        .expect("tree-sitter language version mismatch — grammar ABI incompatible with runtime");
    // set_timeout_micros is deprecated in 0.25 but the replacement API
    // (parse_with_options) is not yet ergonomic. This still works correctly.
    #[allow(deprecated)]
    parser.set_timeout_micros(DEFAULT_PARSE_TIMEOUT_US);

    let tree = parser.parse(source, None).ok_or(CodeError::ParseFailed)?;
    let has_errors = tree.root_node().has_error();

    Ok(CodeTree {
        tree,
        source: source.to_string(),
        language_name: language.to_string(),
        language: ts_lang,
        has_errors,
    })
}

/// Parse source code from a file path.
///
/// Uses a size-bounded reader to avoid TOCTOU races. Rejects non-regular
/// files (symlinks, device files, FIFOs).
///
/// # Errors
/// * `CodeError::FileTooLarge` if the file exceeds 10 MB
/// * `CodeError::Io` if the file cannot be read or is not a regular file
/// * All errors from `parse()`
pub fn parse_file(language: &str, path: &std::path::Path) -> Result<CodeTree, CodeError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(CodeError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        )));
    }

    // Use bounded reader to prevent TOCTOU — don't trust metadata.len() alone
    let file = std::fs::File::open(path)?;
    let mut reader = file.take(MAX_FILE_SIZE + 1);
    let mut source = String::new();
    reader.read_to_string(&mut source)?;

    if source.len() as u64 > MAX_FILE_SIZE {
        return Err(CodeError::FileTooLarge(source.len() as u64, MAX_FILE_SIZE));
    }

    parse(language, &source)
}

/// List all compiled-in language names.
pub fn supported_languages() -> Vec<&'static str> {
    let mut langs = Vec::new();

    #[cfg(feature = "lang-rust")]
    langs.push("rust");
    #[cfg(feature = "lang-python")]
    langs.push("python");
    #[cfg(feature = "lang-javascript")]
    langs.push("javascript");
    #[cfg(feature = "lang-java")]
    langs.push("java");
    #[cfg(feature = "lang-csharp")]
    langs.push("csharp");
    #[cfg(feature = "lang-cpp")]
    langs.push("cpp");
    #[cfg(feature = "lang-go")]
    langs.push("go");
    #[cfg(feature = "lang-typescript")]
    langs.push("typescript");

    langs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust_snippet() {
        let tree = parse("rust", "fn main() { let x = 42; }").unwrap();
        assert_eq!(tree.tree().root_node().kind(), "source_file");
        assert!(!tree.has_errors());
        assert_eq!(tree.language_name(), "rust");
    }

    #[test]
    fn test_parse_rust_with_alias() {
        let tree = parse("rs", "fn main() {}").unwrap();
        assert_eq!(tree.language_name(), "rs");
        assert!(!tree.has_errors());
    }

    #[test]
    fn test_parse_invalid_rust() {
        let tree = parse("rust", "fn foo( {").unwrap();
        assert!(tree.has_errors());
        // Tree is still produced — partial parse
        assert_eq!(tree.tree().root_node().kind(), "source_file");
    }

    #[test]
    fn test_parse_empty_input() {
        let result = parse("rust", "");
        assert!(matches!(result, Err(CodeError::EmptyInput)));
    }

    #[test]
    fn test_parse_whitespace_only() {
        let result = parse("rust", "   \n\t  ");
        assert!(matches!(result, Err(CodeError::EmptyInput)));
    }

    #[test]
    fn test_parse_unsupported_language() {
        let result = parse("brainfuck", "+++");
        assert!(matches!(result, Err(CodeError::UnsupportedLanguage(_))));
    }

    #[test]
    fn test_node_text() {
        let tree = parse("rust", "fn main() {}").unwrap();
        let root = tree.tree().root_node();
        let func = root.child(0).unwrap();
        let name = func.child_by_field_name("name").unwrap();
        assert_eq!(tree.node_text(name), Some("main"));
    }

    #[test]
    fn test_supported_languages() {
        let langs = supported_languages();
        assert!(langs.contains(&"rust"));
    }
}
