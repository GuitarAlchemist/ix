//! Lightweight source code analysis.
//!
//! Extracts complexity metrics, line counts, and Halstead metrics from
//! source code using language-aware keyword/operator counting.
//! No external parser dependencies — works via regex-free line analysis.

use std::path::Path;

use crate::metrics::{CodeMetrics, FileMetrics};

/// Supported languages for code analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Cpp,
    Java,
    Go,
    CSharp,
    FSharp,
}

impl Language {
    /// Detect language from file extension.
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "rs" => Some(Language::Rust),
            "py" => Some(Language::Python),
            "js" | "jsx" | "mjs" => Some(Language::JavaScript),
            "ts" | "tsx" => Some(Language::TypeScript),
            "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" => Some(Language::Cpp),
            "java" => Some(Language::Java),
            "go" => Some(Language::Go),
            "cs" => Some(Language::CSharp),
            "fs" | "fsx" => Some(Language::FSharp),
            _ => None,
        }
    }

    /// Detect language from a file path.
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|e| e.to_str())
            .and_then(Self::from_extension)
    }

    /// Display name.
    pub fn name(&self) -> &'static str {
        match self {
            Language::Rust => "Rust",
            Language::Python => "Python",
            Language::JavaScript => "JavaScript",
            Language::TypeScript => "TypeScript",
            Language::Cpp => "C/C++",
            Language::Java => "Java",
            Language::Go => "Go",
            Language::CSharp => "C#",
            Language::FSharp => "F#",
        }
    }

    /// Branch keywords that increment cyclomatic complexity.
    fn branch_keywords(&self) -> &[&str] {
        match self {
            Language::Rust => &["if", "else if", "match", "while", "for", "loop", "&&", "||", "?"],
            Language::Python => &["if", "elif", "while", "for", "and", "or", "except", "with"],
            Language::JavaScript | Language::TypeScript =>
                &["if", "else if", "while", "for", "switch", "case", "catch", "&&", "||", "?", "??"],
            Language::Cpp =>
                &["if", "else if", "while", "for", "switch", "case", "catch", "&&", "||", "?"],
            Language::Java =>
                &["if", "else if", "while", "for", "switch", "case", "catch", "&&", "||", "?"],
            Language::Go =>
                &["if", "else if", "for", "switch", "case", "select", "&&", "||"],
            Language::CSharp =>
                &["if", "else if", "while", "for", "foreach", "switch", "case", "catch", "&&", "||", "?", "??"],
            Language::FSharp =>
                &["if", "elif", "match", "while", "for", "&&", "||", "|>"],
        }
    }

    /// Nesting keywords that increment cognitive complexity.
    fn nesting_keywords(&self) -> &[&str] {
        match self {
            Language::Rust => &["if", "match", "while", "for", "loop", "fn", "impl"],
            Language::Python => &["if", "while", "for", "def", "class", "try", "with"],
            Language::JavaScript | Language::TypeScript =>
                &["if", "while", "for", "function", "class", "try", "switch"],
            Language::Cpp | Language::Java | Language::CSharp =>
                &["if", "while", "for", "switch", "try", "class"],
            Language::Go => &["if", "for", "switch", "select", "func"],
            Language::FSharp => &["if", "match", "while", "for", "let", "module"],
        }
    }

    /// Line comment prefix.
    fn line_comment(&self) -> &str {
        match self {
            Language::Python | Language::FSharp => "#",
            _ => "//",
        }
    }

    /// Whether the language uses `/* */` block comments.
    fn has_block_comments(&self) -> bool {
        !matches!(self, Language::Python)
    }

    /// Operators for Halstead analysis.
    fn operators(&self) -> &[&str] {
        // Common operators across most languages
        &[
            "+=", "-=", "*=", "/=", "%=", "==", "!=", "<=", ">=", "&&", "||",
            "<<", ">>", "->", "=>", "::", "..", "?.",
            "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~",
            "(", ")", "[", "]", "{", "}", ",", ";", ":", ".",
        ]
    }
}

/// Analyze source code from a string.
pub fn analyze_source(source: &str, language: Language, path: &Path) -> FileMetrics {
    let file_scope = analyze_code(source, language, "");
    let functions = extract_functions(source, language);

    FileMetrics {
        path: path.display().to_string(),
        language: language.name().to_string(),
        file_scope,
        functions,
    }
}

/// Analyze a source file, auto-detecting language from extension.
pub fn analyze_file(path: &Path) -> Option<FileMetrics> {
    let language = Language::from_path(path)?;
    let source = std::fs::read_to_string(path).ok()?;
    Some(analyze_source(&source, language, path))
}

/// Compute metrics for a code block.
fn analyze_code(source: &str, lang: Language, name: &str) -> CodeMetrics {
    let lines: Vec<&str> = source.lines().collect();
    let total_lines = lines.len();

    // Line classification
    let mut sloc = 0usize;
    let mut blank = 0usize;
    let mut cloc = 0usize;
    let mut in_block_comment = false;

    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank += 1;
            continue;
        }

        if in_block_comment {
            cloc += 1;
            if trimmed.contains("*/") {
                in_block_comment = false;
            }
            continue;
        }

        if trimmed.starts_with(lang.line_comment()) {
            cloc += 1;
            continue;
        }

        if lang.has_block_comments() && trimmed.contains("/*") {
            cloc += 1;
            if !trimmed.contains("*/") {
                in_block_comment = true;
            }
            continue;
        }

        // Python docstrings (simplified)
        if matches!(lang, Language::Python) && (trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''")) {
            cloc += 1;
            continue;
        }

        sloc += 1;
    }

    let ploc = total_lines;

    // Count logical lines (lines ending with ; or { or : for Python)
    let lloc = lines.iter().filter(|l| {
        let t = l.trim();
        match lang {
            Language::Python => t.ends_with(':') && !t.starts_with('#'),
            _ => t.ends_with(';') || t.ends_with('{'),
        }
    }).count();

    // Cyclomatic complexity: 1 + count of branch keywords
    let mut cyclomatic = 1.0;
    for kw in lang.branch_keywords() {
        for line in &lines {
            let trimmed = line.trim();
            // Skip comments
            if trimmed.starts_with(lang.line_comment()) {
                continue;
            }
            // Count keyword occurrences (word-boundary aware for alpha keywords)
            if kw.chars().all(|c| c.is_alphanumeric() || c == ' ') {
                // Alpha keyword: check word boundaries
                cyclomatic += count_keyword_occurrences(trimmed, kw) as f64;
            } else {
                // Operator: simple count
                cyclomatic += trimmed.matches(kw).count() as f64;
            }
        }
    }

    // Cognitive complexity: like cyclomatic but with nesting penalties
    let cognitive = estimate_cognitive(source, lang);

    // Count function exit points (return statements)
    let n_exits = lines.iter().filter(|l| {
        let t = l.trim();
        !t.starts_with(lang.line_comment()) && contains_keyword(t, "return")
    }).count().max(1) as f64;

    // Halstead metrics
    let (h_u_ops, h_u_opnds, h_total_ops, h_total_opnds) = halstead_counts(source, lang);
    let h_vocabulary = (h_u_ops + h_u_opnds) as f64;
    let h_length = (h_total_ops + h_total_opnds) as f64;
    let h_volume = if h_vocabulary > 0.0 {
        h_length * h_vocabulary.log2()
    } else {
        0.0
    };
    let h_difficulty = if h_u_opnds > 0 {
        (h_u_ops as f64 / 2.0) * (h_total_opnds as f64 / h_u_opnds as f64)
    } else {
        0.0
    };
    let h_effort = h_difficulty * h_volume;
    let h_bugs = h_volume / 3000.0;

    // Maintainability Index: 171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(SLOC)
    let mi = if h_volume > 0.0 && sloc > 0 {
        let raw = 171.0 - 5.2 * h_volume.ln() - 0.23 * cyclomatic - 16.2 * (sloc as f64).ln();
        raw.clamp(0.0, 171.0)
    } else {
        171.0
    };

    CodeMetrics {
        name: name.to_string(),
        start_line: 1,
        end_line: total_lines,
        cyclomatic,
        cognitive,
        n_exits,
        n_args: 0.0, // Set by caller for functions
        sloc: sloc as f64,
        ploc: ploc as f64,
        lloc: lloc as f64,
        cloc: cloc as f64,
        blank: blank as f64,
        h_u_ops: h_u_ops as f64,
        h_u_opnds: h_u_opnds as f64,
        h_total_ops: h_total_ops as f64,
        h_total_opnds: h_total_opnds as f64,
        h_vocabulary,
        h_length,
        h_volume,
        h_difficulty,
        h_effort,
        h_bugs,
        maintainability_index: mi,
    }
}

/// Extract function-level metrics from source code.
fn extract_functions(source: &str, lang: Language) -> Vec<CodeMetrics> {
    let lines: Vec<&str> = source.lines().collect();
    let mut functions = Vec::new();

    let fn_keyword = match lang {
        Language::Rust => "fn ",
        Language::Python => "def ",
        Language::JavaScript | Language::TypeScript => "function ",
        Language::Go => "func ",
        Language::Java | Language::CSharp | Language::Cpp => "", // harder to detect, skip
        Language::FSharp => "let ",
    };

    if fn_keyword.is_empty() {
        return functions;
    }

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with(lang.line_comment()) {
            continue;
        }

        if contains_keyword(trimmed, fn_keyword.trim()) {
            // Extract function name
            let name = extract_fn_name(trimmed, lang);
            let n_args = count_fn_args(trimmed);

            // Find function body end (simplified: brace/indent counting)
            let end_line = find_fn_end(&lines, i, lang);
            let fn_body: String = lines[i..=end_line].join("\n");

            let mut m = analyze_code(&fn_body, lang, &name);
            m.start_line = i + 1;
            m.end_line = end_line + 1;
            m.n_args = n_args as f64;
            functions.push(m);
        }
    }

    functions
}

/// Extract function name from a function declaration line.
fn extract_fn_name(line: &str, lang: Language) -> String {
    let prefix = match lang {
        Language::Rust => "fn ",
        Language::Python => "def ",
        Language::JavaScript | Language::TypeScript => "function ",
        Language::Go => "func ",
        Language::FSharp => "let ",
        _ => return String::new(),
    };

    if let Some(start) = line.find(prefix) {
        let rest = &line[start + prefix.len()..];
        // Handle Rust pub fn, async fn, etc.
        let rest = if matches!(lang, Language::Rust) && rest.starts_with("fn ") {
            &rest[3..]
        } else {
            rest
        };
        let end = rest.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(rest.len());
        rest[..end].to_string()
    } else {
        String::new()
    }
}

/// Count function arguments (number of commas + 1 between parens, or 0 if empty parens).
fn count_fn_args(line: &str) -> usize {
    if let Some(start) = line.find('(') {
        if let Some(end) = line[start..].find(')') {
            let params = &line[start + 1..start + end];
            let trimmed = params.trim();
            if trimmed.is_empty() {
                return 0;
            }
            return trimmed.split(',').count();
        }
    }
    0
}

/// Find the end line of a function body.
fn find_fn_end(lines: &[&str], start: usize, lang: Language) -> usize {
    match lang {
        Language::Python | Language::FSharp => {
            // Indent-based: find next line at same or lower indent level
            let base_indent = lines[start].len() - lines[start].trim_start().len();
            for (i, line) in lines.iter().enumerate().skip(start + 1) {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                let indent = line.len() - line.trim_start().len();
                if indent <= base_indent {
                    return (i - 1).max(start);
                }
            }
            lines.len() - 1
        }
        _ => {
            // Brace-based: count { and }
            let mut depth = 0i32;
            let mut found_open = false;
            for (i, line) in lines.iter().enumerate().skip(start) {
                for ch in line.chars() {
                    if ch == '{' { depth += 1; found_open = true; }
                    if ch == '}' { depth -= 1; }
                }
                if found_open && depth <= 0 {
                    return i;
                }
            }
            lines.len() - 1
        }
    }
}

/// Count occurrences of a keyword at word boundaries.
fn count_keyword_occurrences(line: &str, keyword: &str) -> usize {
    let mut count = 0;
    let bytes = line.as_bytes();
    let kw_bytes = keyword.as_bytes();
    let kw_len = kw_bytes.len();

    let mut i = 0;
    while i + kw_len <= bytes.len() {
        if &bytes[i..i + kw_len] == kw_bytes {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_alphanumeric() && bytes[i - 1] != b'_';
            let after_ok = i + kw_len >= bytes.len()
                || !bytes[i + kw_len].is_ascii_alphanumeric() && bytes[i + kw_len] != b'_';
            if before_ok && after_ok {
                count += 1;
                i += kw_len;
                continue;
            }
        }
        i += 1;
    }
    count
}

/// Check if line contains a keyword (word-boundary aware).
fn contains_keyword(line: &str, keyword: &str) -> bool {
    count_keyword_occurrences(line, keyword) > 0
}

/// Estimate cognitive complexity with nesting awareness.
fn estimate_cognitive(source: &str, lang: Language) -> f64 {
    let mut cognitive = 0.0;
    let mut nesting = 0i32;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(lang.line_comment()) || trimmed.is_empty() {
            continue;
        }

        // Check for nesting increasers
        let opens_scope = lang.nesting_keywords().iter()
            .any(|kw| contains_keyword(trimmed, kw));

        // Count branch keywords with nesting penalty
        for kw in lang.branch_keywords() {
            if kw.chars().all(|c| c.is_alphanumeric() || c == ' ') {
                let n = count_keyword_occurrences(trimmed, kw);
                cognitive += n as f64 * (1.0 + nesting as f64);
            }
        }

        // Track nesting depth
        if opens_scope {
            nesting += 1;
        }
        // Decrease on closing braces/dedent
        let closes = trimmed.chars().filter(|&c| c == '}').count() as i32;
        nesting = (nesting - closes).max(0);
    }

    cognitive
}

/// Count distinct and total operators/operands for Halstead metrics.
fn halstead_counts(source: &str, lang: Language) -> (usize, usize, usize, usize) {
    use std::collections::HashSet;

    let ops = lang.operators();
    let mut unique_ops = HashSet::new();
    let mut unique_opnds = HashSet::new();
    let mut total_ops = 0usize;
    let mut total_opnds = 0usize;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with(lang.line_comment()) || trimmed.is_empty() {
            continue;
        }

        // Count operators
        for op in ops {
            let count = trimmed.matches(op).count();
            if count > 0 {
                unique_ops.insert(*op);
                total_ops += count;
            }
        }

        // Count operands: identifiers and literals (simplified)
        for token in tokenize_operands(trimmed) {
            unique_opnds.insert(token.clone());
            total_opnds += 1;
        }
    }

    (unique_ops.len(), unique_opnds.len(), total_ops, total_opnds)
}

/// Simple tokenizer to extract operand-like tokens (identifiers, numbers, strings).
fn tokenize_operands(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();
    let mut current = String::new();

    while let Some(&ch) = chars.peek() {
        if ch.is_alphanumeric() || ch == '_' {
            current.push(ch);
            chars.next();
        } else if ch == '"' || ch == '\'' {
            // Skip string literals as single operand
            chars.next();
            let mut s = String::from(ch);
            while let Some(&c) = chars.peek() {
                s.push(c);
                chars.next();
                if c == ch { break; }
            }
            tokens.push(s);
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            chars.next();
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_language_from_extension() {
        assert_eq!(Language::from_extension("rs"), Some(Language::Rust));
        assert_eq!(Language::from_extension("py"), Some(Language::Python));
        assert_eq!(Language::from_extension("js"), Some(Language::JavaScript));
        assert_eq!(Language::from_extension("ts"), Some(Language::TypeScript));
        assert_eq!(Language::from_extension("cpp"), Some(Language::Cpp));
        assert_eq!(Language::from_extension("java"), Some(Language::Java));
        assert_eq!(Language::from_extension("go"), Some(Language::Go));
        assert_eq!(Language::from_extension("cs"), Some(Language::CSharp));
        assert_eq!(Language::from_extension("fs"), Some(Language::FSharp));
        assert_eq!(Language::from_extension("xyz"), None);
    }

    #[test]
    fn test_language_from_path() {
        assert_eq!(Language::from_path(Path::new("main.rs")), Some(Language::Rust));
        assert_eq!(Language::from_path(Path::new("script.py")), Some(Language::Python));
    }

    #[test]
    fn test_analyze_rust_source() {
        let source = r#"
fn hello(name: &str) -> String {
    if name.is_empty() {
        return "Hello, World!".to_string();
    }
    format!("Hello, {}!", name)
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;
        let path = Path::new("test.rs");
        let result = analyze_source(source, Language::Rust, path);
        assert_eq!(result.language, "Rust");
        assert!(!result.functions.is_empty(), "should find functions");

        // hello function should have cyclomatic >= 2 (if branch)
        let hello_fn = result.functions.iter().find(|f| f.name == "hello");
        assert!(hello_fn.is_some(), "should find hello function");
        let hello = hello_fn.unwrap();
        assert!(hello.cyclomatic >= 2.0, "hello should have CC >= 2, got {}", hello.cyclomatic);

        // add function should be simple
        let add_fn = result.functions.iter().find(|f| f.name == "add");
        assert!(add_fn.is_some(), "should find add function");
        let add = add_fn.unwrap();
        assert!((add.n_args - 2.0).abs() < 0.01, "add should have 2 args, got {}", add.n_args);
    }

    #[test]
    fn test_analyze_python_source() {
        let source = r#"
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
"#;
        let path = Path::new("test.py");
        let result = analyze_source(source, Language::Python, path);
        assert_eq!(result.language, "Python");
        assert!(!result.functions.is_empty(), "should find functions");

        let fib = result.functions.iter().find(|f| f.name == "fibonacci");
        assert!(fib.is_some(), "should find fibonacci function");
        let fib = fib.unwrap();
        assert!(fib.cyclomatic >= 3.0, "fibonacci CC should be >= 3, got {}", fib.cyclomatic);
    }

    #[test]
    fn test_analyze_javascript_source() {
        let source = r#"
function greet(name) {
    if (name) {
        return "Hello, " + name;
    }
    return "Hello, World";
}
"#;
        let path = Path::new("test.js");
        let result = analyze_source(source, Language::JavaScript, path);
        assert_eq!(result.language, "JavaScript");
        assert!(!result.functions.is_empty());
    }

    #[test]
    fn test_line_counts() {
        let source = "// comment\n\nfn main() {\n    println!(\"hello\");\n}\n";
        let path = Path::new("test.rs");
        let result = analyze_source(source, Language::Rust, path);
        assert!(result.file_scope.cloc >= 1.0, "should count comment lines");
        assert!(result.file_scope.blank >= 1.0, "should count blank lines");
        assert!(result.file_scope.sloc >= 2.0, "should count source lines");
    }

    #[test]
    fn test_feature_vector() {
        let source = "fn main() { println!(\"hello\"); }\n";
        let path = Path::new("test.rs");
        let result = analyze_source(source, Language::Rust, path);
        let features = result.file_scope.to_features();
        assert_eq!(features.len(), 20);
    }

    #[test]
    fn test_halstead_metrics() {
        let source = r#"
fn compute(x: f64, y: f64) -> f64 {
    let a = x + y;
    let b = x * y;
    let c = a - b;
    c / (a + 1.0)
}
"#;
        let path = Path::new("test.rs");
        let result = analyze_source(source, Language::Rust, path);
        assert!(result.file_scope.h_length > 0.0, "Halstead length should be > 0");
        assert!(result.file_scope.h_volume > 0.0, "Halstead volume should be > 0");
    }

    #[test]
    fn test_maintainability_index() {
        let source = r#"
fn simple() -> i32 {
    42
}
"#;
        let path = Path::new("test.rs");
        let result = analyze_source(source, Language::Rust, path);
        let mi = result.file_scope.maintainability_index;
        assert!(mi > 0.0 && mi <= 171.0, "MI should be in [0, 171], got {mi}");
    }

    #[test]
    fn test_count_keyword_occurrences() {
        assert_eq!(count_keyword_occurrences("if x > 0 { if y > 0 { } }", "if"), 2);
        assert_eq!(count_keyword_occurrences("iffy", "if"), 0);
        assert_eq!(count_keyword_occurrences("else if x > 0", "else if"), 1);
    }

    #[test]
    fn test_extract_fn_name() {
        assert_eq!(extract_fn_name("fn hello(x: i32) -> i32 {", Language::Rust), "hello");
        assert_eq!(extract_fn_name("pub fn greet() {", Language::Rust), "greet");
        assert_eq!(extract_fn_name("def fibonacci(n):", Language::Python), "fibonacci");
        assert_eq!(extract_fn_name("function greet(name) {", Language::JavaScript), "greet");
        assert_eq!(extract_fn_name("func main() {", Language::Go), "main");
    }

    #[test]
    fn test_count_fn_args() {
        assert_eq!(count_fn_args("fn hello(x: i32, y: i32) -> i32"), 2);
        assert_eq!(count_fn_args("fn main()"), 0);
        assert_eq!(count_fn_args("fn one(x: &str)"), 1);
    }

    #[test]
    fn test_cognitive_complexity() {
        // Nested ifs should have higher cognitive complexity
        let simple = "fn f() { if x > 0 { } }";
        let nested = "fn f() { if x > 0 { if y > 0 { if z > 0 { } } } }";

        let c1 = estimate_cognitive(simple, Language::Rust);
        let c2 = estimate_cognitive(nested, Language::Rust);
        assert!(c2 > c1, "nested code should have higher cognitive complexity: {c1} vs {c2}");
    }

    #[test]
    fn test_go_analysis() {
        let source = r#"
func add(a int, b int) int {
    return a + b
}

func max(a int, b int) int {
    if a > b {
        return a
    }
    return b
}
"#;
        let path = Path::new("test.go");
        let result = analyze_source(source, Language::Go, path);
        assert_eq!(result.language, "Go");
        assert!(result.functions.len() >= 2, "should find 2 functions");
    }
}
