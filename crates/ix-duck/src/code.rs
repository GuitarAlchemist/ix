//! Code-analysis UDFs over `ix-code` — SQL over a whole codebase.
//!
//! Pair these with DuckDB's `read_text('crates/**/*.rs')` (which yields
//! `(filename, content)` rows) to query complexity, smells, and — with the
//! `code-semantic` feature — tree-sitter AST matches across a corpus:
//!
//! ```sql
//! SELECT filename, ix_code_complexity(content, filename) AS cc
//! FROM read_text('crates/**/*.rs') ORDER BY cc DESC LIMIT 20;
//! ```
//!
//! All are **scalar** UDFs `(source VARCHAR, language VARCHAR[, query]) -> …` so
//! they run row-wise over a source column (table functions can't take a column
//! argument). The `language` arg is flexible: an extension (`'rs'`), a path/
//! filename (`'src/x.rs'`, so `read_text`'s `filename` works directly), or a
//! name (`'rust'`). Pure wraps of `ix-code` — no analysis logic here.
//!
//! | tier | UDF | wraps |
//! |---|---|---|
//! | A | `ix_code_complexity(src, lang) -> DOUBLE` | `analyze_source` file-scope cyclomatic |
//! | A | `ix_code_metrics(src, lang) -> VARCHAR` (JSON `FileMetrics`) | `analyze_source` |
//! | A | `ix_code_smells(src, lang) -> VARCHAR` (JSON `[CodeSmell]`) | `detect_smells` |
//! | B | `ix_semantic_metrics(src, lang) -> VARCHAR` (JSON) | `extract_semantic_metrics_for` (tree-sitter) |
//! | B | `ix_ast_query(src, lang, query) -> VARCHAR` (JSON `[AstMatch]`) | `run_ast_query` (tree-sitter) |

use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::ffi::duckdb_string_t;
use duckdb::types::DuckString;
use duckdb::vscalar::{ScalarFunctionSignature, VScalar};
use duckdb::vtab::arrow::WritableVector;
use duckdb::Connection;
use ix_code::analyze::{analyze_source, Language};
use ix_code::smells::detect_smells;
use std::error::Error;
use std::ffi::CString;
use std::path::Path;

type Res = Result<(), Box<dyn Error>>;

fn ty(id: LogicalTypeId) -> LogicalTypeHandle {
    LogicalTypeHandle::from(id)
}

/// Read a `VARCHAR` column into owned `String`s (one per row).
fn read_varchar_col(input: &mut DataChunkHandle, col: usize, n: usize) -> Vec<String> {
    let v = input.flat_vector(col);
    let slice = unsafe { v.as_slice_with_len::<duckdb_string_t>(n) };
    slice
        .iter()
        .map(|ptr| DuckString::new(&mut { *ptr }).as_str().to_string())
        .collect()
}

fn write_varchar(output: &mut dyn WritableVector, vals: &[String]) -> Res {
    let out = output.flat_vector();
    for (i, v) in vals.iter().enumerate() {
        out.insert(i, CString::new(v.as_str())?);
    }
    Ok(())
}

fn write_f64(output: &mut dyn WritableVector, vals: &[f64]) {
    let mut out = output.flat_vector();
    let slice = unsafe { out.as_mut_slice_with_len::<f64>(vals.len()) };
    slice.copy_from_slice(vals);
}

/// Resolve a language token: an extension (`rs`), a path/filename (`a/b.rs`), or
/// a language name (`rust`). `None` if unrecognised.
fn parse_language(tok: &str) -> Option<Language> {
    let t = tok.trim();
    Language::from_extension(t)
        .or_else(|| Language::from_path(Path::new(t)))
        .or_else(|| match t.to_lowercase().as_str() {
            "rust" => Some(Language::Rust),
            "python" => Some(Language::Python),
            "javascript" => Some(Language::JavaScript),
            "typescript" => Some(Language::TypeScript),
            "cpp" | "c" | "c++" => Some(Language::Cpp),
            "java" => Some(Language::Java),
            "go" | "golang" => Some(Language::Go),
            "csharp" | "c#" => Some(Language::CSharp),
            "fsharp" | "f#" => Some(Language::FSharp),
            "php" => Some(Language::Php),
            "ruby" => Some(Language::Ruby),
            _ => None,
        })
}

/// Resolve the language token or produce a SQL error naming the offending value.
fn lang_or_err(fname: &str, tok: &str) -> Result<Language, Box<dyn Error>> {
    parse_language(tok)
        .ok_or_else(|| format!("{fname}: unrecognised language/extension {tok:?}").into())
}

fn src_lang_sig(ret: LogicalTypeId) -> Vec<ScalarFunctionSignature> {
    vec![ScalarFunctionSignature::exact(
        vec![ty(LogicalTypeId::Varchar), ty(LogicalTypeId::Varchar)],
        ty(ret),
    )]
}

// ── Tier A: metrics / smells (no tree-sitter) ────────────────────────────────

struct IxCodeComplexity;
impl VScalar for IxCodeComplexity {
    type State = ();
    // @ai:invariant ix_code_complexity returns the file-scope cyclomatic complexity from ix_code::analyze_source for the source+language; unrecognised language -> SQL error [T:test conf:0.85 src:ix_duck::code::tests::complexity_counts_branches]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let src = read_varchar_col(input, 0, n);
        let lang = read_varchar_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let l = lang_or_err("ix_code_complexity", &lang[i])?;
            out.push(analyze_source(&src[i], l, Path::new(&lang[i])).file_scope.cyclomatic);
        }
        write_f64(output, &out);
        Ok(())
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        src_lang_sig(LogicalTypeId::Double)
    }
}

struct IxCodeMetrics;
impl VScalar for IxCodeMetrics {
    type State = ();
    // @ai:invariant ix_code_metrics returns the full ix_code::FileMetrics (file_scope + per-function complexity/Halstead/SLOC) as JSON; unrecognised language -> SQL error [T:test conf:0.85 src:ix_duck::code::tests::metrics_json_has_fields]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let src = read_varchar_col(input, 0, n);
        let lang = read_varchar_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let l = lang_or_err("ix_code_metrics", &lang[i])?;
            let fm = analyze_source(&src[i], l, Path::new(&lang[i]));
            out.push(serde_json::to_string(&fm)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        src_lang_sig(LogicalTypeId::Varchar)
    }
}

struct IxCodeSmells;
impl VScalar for IxCodeSmells {
    type State = ();
    // @ai:invariant ix_code_smells returns ix_code::detect_smells (lexical; AST-based too under code-semantic) as a JSON array of {name,line,severity,message}; unrecognised language -> SQL error [T:test conf:0.85 src:ix_duck::code::tests::smells_flags_todo]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let src = read_varchar_col(input, 0, n);
        let lang = read_varchar_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let l = lang_or_err("ix_code_smells", &lang[i])?;
            out.push(serde_json::to_string(&detect_smells(&src[i], l))?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        src_lang_sig(LogicalTypeId::Varchar)
    }
}

// ── Tier B: tree-sitter AST queries + semantic metrics ───────────────────────

#[cfg(feature = "code-semantic")]
struct IxSemanticMetrics;
#[cfg(feature = "code-semantic")]
impl VScalar for IxSemanticMetrics {
    type State = ();
    // @ai:invariant ix_semantic_metrics returns ix_code::extract_semantic_metrics_for (tree-sitter) as JSON — parse_quality, ast_node_count, nesting, error-handling density, unsafe_blocks, call graph; languages without a bundled grammar yield parse_quality 0 [T:test conf:0.8 src:ix_duck::code::tests::semantic_metrics_parses_rust]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let src = read_varchar_col(input, 0, n);
        let lang = read_varchar_col(input, 1, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let l = lang_or_err("ix_semantic_metrics", &lang[i])?;
            let m = ix_code::semantic::extract_semantic_metrics_for(&src[i], l);
            out.push(serde_json::to_string(&m)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        src_lang_sig(LogicalTypeId::Varchar)
    }
}

#[cfg(feature = "code-semantic")]
struct IxAstQuery;
#[cfg(feature = "code-semantic")]
impl VScalar for IxAstQuery {
    type State = ();
    // @ai:invariant ix_ast_query runs a tree-sitter S-expression query (arg 3) against the source for the language, returning matches as a JSON array of {capture,text,start_line,end_line,start_col}; invalid query or grammar-less language -> SQL error [T:test conf:0.8 src:ix_duck::code::tests::ast_query_finds_functions]
    unsafe fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> Res {
        let n = input.len();
        let src = read_varchar_col(input, 0, n);
        let lang = read_varchar_col(input, 1, n);
        let query = read_varchar_col(input, 2, n);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let l = lang_or_err("ix_ast_query", &lang[i])?;
            let matches = ix_code::semantic::run_ast_query(&src[i], l, &query[i])
                .map_err(|e| format!("ix_ast_query: {e}"))?;
            out.push(serde_json::to_string(&matches)?);
        }
        write_varchar(output, &out)
    }
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![
                ty(LogicalTypeId::Varchar),
                ty(LogicalTypeId::Varchar),
                ty(LogicalTypeId::Varchar),
            ],
            ty(LogicalTypeId::Varchar),
        )]
    }
}

/// Register the code-analysis UDFs (Tier A always; Tier B under `code-semantic`).
pub(crate) fn register(conn: &Connection) -> duckdb::Result<()> {
    conn.register_scalar_function::<IxCodeComplexity>("ix_code_complexity")?;
    conn.register_scalar_function::<IxCodeMetrics>("ix_code_metrics")?;
    conn.register_scalar_function::<IxCodeSmells>("ix_code_smells")?;
    #[cfg(feature = "code-semantic")]
    {
        conn.register_scalar_function::<IxSemanticMetrics>("ix_semantic_metrics")?;
        conn.register_scalar_function::<IxAstQuery>("ix_ast_query")?;
    }
    Ok(())
}

#[cfg(all(test, feature = "duck"))]
mod tests {
    use crate::open_bench;

    fn s(sql: &str) -> String {
        open_bench()
            .unwrap()
            .query_row(sql, [], |r| r.get::<_, String>(0))
            .unwrap()
    }
    fn f(sql: &str) -> f64 {
        open_bench()
            .unwrap()
            .query_row(sql, [], |r| r.get::<_, f64>(0))
            .unwrap()
    }

    const BRANCHY: &str = "fn f(x:i32)->i32{ if x>0 { 1 } else if x<0 { 2 } else { 3 } }";

    #[test]
    fn complexity_counts_branches() {
        // Multiple branches → cyclomatic well above 1.
        assert!(f(&format!("SELECT ix_code_complexity('{BRANCHY}', 'rust')")) > 1.0);
        // Extension and path tokens resolve the same language.
        assert!(f(&format!("SELECT ix_code_complexity('{BRANCHY}', 'rs')")) > 1.0);
        assert!(f(&format!("SELECT ix_code_complexity('{BRANCHY}', 'a/b.rs')")) > 1.0);
    }

    #[test]
    fn metrics_json_has_fields() {
        let j = s(&format!("SELECT ix_code_metrics('{BRANCHY}', 'rust')"));
        assert!(j.contains("cyclomatic") && j.contains("maintainability_index"));
    }

    #[test]
    fn smells_flags_todo() {
        // A TODO marker is a lexical smell, available without tree-sitter.
        let j = s("SELECT ix_code_smells('// TODO: fix this\nlet y = 1;', 'rust')");
        assert!(j.contains("todo") || j.to_lowercase().contains("todo"));
    }

    #[test]
    fn unrecognised_language_errors() {
        let conn = open_bench().unwrap();
        let r = conn.query_row("SELECT ix_code_complexity('x', 'klingon')", [], |r| {
            r.get::<_, f64>(0)
        });
        assert!(r.is_err(), "unknown language should be a SQL error");
    }

    #[cfg(feature = "code-semantic")]
    #[test]
    fn semantic_metrics_parses_rust() {
        let j = s("SELECT ix_semantic_metrics('fn main(){ let x: i32 = 1; }', 'rust')");
        assert!(j.contains("parse_quality") && j.contains("ast_node_count"));
    }

    #[cfg(feature = "code-semantic")]
    #[test]
    fn ast_query_finds_functions() {
        // Tree-sitter query for Rust function names.
        let j = s(
            "SELECT ix_ast_query('fn add(){} fn main(){}', 'rust', \
             '(function_item name: (identifier) @fn)')",
        );
        assert!(j.contains("add") && j.contains("main"));
    }

    #[cfg(feature = "code-semantic")]
    #[test]
    fn ast_query_invalid_query_errors() {
        let conn = open_bench().unwrap();
        let r = conn.query_row(
            "SELECT ix_ast_query('fn main(){}', 'rust', '(this is not valid')",
            [],
            |r| r.get::<_, String>(0),
        );
        assert!(r.is_err(), "invalid tree-sitter query should be a SQL error");
    }
}
