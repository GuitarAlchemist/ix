//! Layer 2 — Semantic analysis via tree-sitter.
//!
//! This module provides AST-based semantic metrics for Rust source code. It
//! complements the lexical metrics in [`crate::metrics`] by operating on a
//! concrete syntax tree produced by the `tree-sitter` parser.
//!
//! The public surface is intentionally small and pure: feed a source string
//! in, get a [`SemanticMetrics`] value (or a [`CallGraph`]) out. On parse
//! failure the extractor returns a degraded-but-well-formed metrics value
//! with `parse_quality = 0.0` rather than panicking, so downstream ML
//! pipelines can still consume the feature vector.
//!
//! # Example
//!
//! ```no_run
//! use ix_code::semantic::extract_semantic_metrics;
//!
//! let src = r#"
//!     fn add(a: i32, b: i32) -> i32 { a + b }
//!     fn main() { let _ = add(1, 2); }
//! "#;
//! let m = extract_semantic_metrics(src);
//! assert!(m.parse_quality > 0.9);
//! ```

use serde::{Deserialize, Serialize};

#[cfg(feature = "semantic")]
use streaming_iterator::StreamingIterator;
#[cfg(feature = "semantic")]
use tree_sitter::{Node, Parser, Query, QueryCursor, Tree};

/// A single directed edge in a call graph: `caller -> callee` at a specific
/// source line, with an aggregation `weight` (number of call sites fused into
/// this edge).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEdge {
    /// Name of the enclosing function that performs the call. `"<top-level>"`
    /// is used for calls that occur outside of any function body.
    pub caller: String,
    /// Name of the function being invoked (identifier, method name, or last
    /// segment of a scoped path).
    pub callee: String,
    /// 1-based source line of the call site.
    pub call_site_line: usize,
    /// Number of distinct call sites merged into this edge. Populated as 1 by
    /// [`extract_call_graph`]; downstream passes may aggregate.
    pub weight: u32,
}

/// A static call graph extracted from one translation unit. Nodes are
/// function names (both defined functions and externally-called functions)
/// and edges are [`CallEdge`] records.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallGraph {
    /// All function-like names seen in the source (definitions and call
    /// targets). Deduplicated, order of first appearance.
    pub nodes: Vec<String>,
    /// Directed call edges.
    pub edges: Vec<CallEdge>,
}

/// Semantic metrics derived from the AST of a single source file.
///
/// All fields are safe to feed into an ML pipeline: ratios are in `[0, 1]`,
/// counts are non-negative, and a failed parse yields a well-defined default
/// with `parse_quality = 0.0`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticMetrics {
    /// Parse success in `[0, 1]`. `1.0` = clean parse, `0.0` = parser failed
    /// to construct a tree at all, intermediate values reflect
    /// error-node density.
    pub parse_quality: f64,
    /// Total number of AST nodes.
    pub ast_node_count: usize,
    /// Maximum nesting depth of any block construct in the tree.
    pub nesting_depth_max: usize,
    /// Mean nesting depth of block constructs.
    pub nesting_depth_mean: f64,
    /// Density of explicit error-handling constructs (`Result`, `?`, `match`
    /// on `Err`, `panic!`, `.unwrap`, `.expect`) per 100 AST nodes.
    pub error_handling_density: f64,
    /// Number of `unsafe { ... }` blocks and `unsafe fn` items.
    pub unsafe_blocks: usize,
    /// Ratio of typed bindings (`let x: T = ...`, typed parameters, typed
    /// return values) to total bindings, in `[0, 1]`.
    pub type_annotation_ratio: f64,
    /// Extracted call graph. Empty when the parse fails.
    pub call_graph: CallGraph,
}

// ---------------------------------------------------------------------------
// Feature-gated implementation
// ---------------------------------------------------------------------------

/// Extract [`SemanticMetrics`] from a Rust source snippet.
///
/// Returns a default-initialized `SemanticMetrics` with `parse_quality = 0.0`
/// when the parser cannot build a tree (e.g. invalid UTF-8 or parser init
/// failure). Syntax errors in otherwise-parseable source yield a non-zero but
/// degraded `parse_quality`.
#[cfg(feature = "semantic")]
pub fn extract_semantic_metrics(source: &str) -> SemanticMetrics {
    let Some((tree, _)) = parse_rust(source) else {
        return SemanticMetrics {
            parse_quality: 0.0,
            ..Default::default()
        };
    };

    let root = tree.root_node();
    let ast_node_count = count_nodes(root);
    let error_nodes = count_error_nodes(root);
    let parse_quality = if ast_node_count == 0 {
        0.0
    } else {
        // Linear penalty: clean tree -> 1.0, tree where every node is an
        // error -> 0.0.
        1.0 - (error_nodes as f64 / ast_node_count as f64).min(1.0)
    };

    let (nesting_depth_max, nesting_depth_mean) = nesting_stats(root);
    let unsafe_blocks = count_kind(root, &["unsafe_block"]);
    let error_handling_density =
        error_handling_count(source, root) as f64 * 100.0 / ast_node_count.max(1) as f64;
    let type_annotation_ratio = type_annotation_ratio(root);

    let call_graph = extract_call_graph_impl(source, &tree).unwrap_or_default();

    SemanticMetrics {
        parse_quality,
        ast_node_count,
        nesting_depth_max,
        nesting_depth_mean,
        error_handling_density,
        unsafe_blocks,
        type_annotation_ratio,
        call_graph,
    }
}

/// Extract a [`CallGraph`] from a Rust source snippet.
///
/// Returns `None` if the source cannot be parsed at all.
#[cfg(feature = "semantic")]
pub fn extract_call_graph(source: &str) -> Option<CallGraph> {
    let (tree, _) = parse_rust(source)?;
    extract_call_graph_impl(source, &tree)
}

#[cfg(feature = "semantic")]
fn parse_rust(source: &str) -> Option<(Tree, ())> {
    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    parser.set_language(&language).ok()?;
    let tree = parser.parse(source, None)?;
    Some((tree, ()))
}

#[cfg(feature = "semantic")]
fn count_nodes(node: Node) -> usize {
    let mut count = 1;
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        count += count_nodes(child);
    }
    count
}

#[cfg(feature = "semantic")]
fn count_error_nodes(node: Node) -> usize {
    let mut count = 0;
    if node.is_error() || node.is_missing() {
        count += 1;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        count += count_error_nodes(child);
    }
    count
}

/// Walks the tree and returns `(max_depth, mean_depth)` over block-bearing
/// nodes (functions, blocks, loops, conditionals, matches).
#[cfg(feature = "semantic")]
fn nesting_stats(root: Node) -> (usize, f64) {
    const BLOCK_KINDS: &[&str] = &[
        "block",
        "if_expression",
        "match_expression",
        "while_expression",
        "for_expression",
        "loop_expression",
        "function_item",
    ];

    fn is_block(kind: &str) -> bool {
        BLOCK_KINDS.contains(&kind)
    }

    fn walk(node: Node, depth: usize, acc: &mut (usize, usize, usize)) {
        let next_depth = if is_block(node.kind()) {
            acc.0 = acc.0.max(depth + 1);
            acc.1 += depth + 1;
            acc.2 += 1;
            depth + 1
        } else {
            depth
        };
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            walk(child, next_depth, acc);
        }
    }

    let mut acc = (0usize, 0usize, 0usize);
    walk(root, 0, &mut acc);
    let mean = if acc.2 == 0 {
        0.0
    } else {
        acc.1 as f64 / acc.2 as f64
    };
    (acc.0, mean)
}

#[cfg(feature = "semantic")]
fn count_kind(root: Node, kinds: &[&str]) -> usize {
    let mut count = 0;
    if kinds.contains(&root.kind()) {
        count += 1;
    }
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        count += count_kind(child, kinds);
    }
    count
}

/// Count tokens associated with explicit error handling. We combine an
/// AST-node scan (for `try_expression` = `?`, `match_expression`) with a
/// cheap lexical scan for method-call sentinels (`unwrap`, `expect`, `panic!`)
/// that are hard to match purely on kind.
#[cfg(feature = "semantic")]
fn error_handling_count(source: &str, root: Node) -> usize {
    let ast_hits = count_kind(root, &["try_expression"]);
    let text_hits = source.matches(".unwrap(").count()
        + source.matches(".expect(").count()
        + source.matches("panic!(").count()
        + source.matches("Result<").count()
        + source.matches("Err(").count();
    ast_hits + text_hits
}

/// Ratio of explicitly-typed `let` bindings to total `let` bindings.
/// Returns `1.0` when there are no bindings at all (vacuously "fully typed").
#[cfg(feature = "semantic")]
fn type_annotation_ratio(root: Node) -> f64 {
    fn walk(node: Node, total: &mut usize, typed: &mut usize) {
        if node.kind() == "let_declaration" {
            *total += 1;
            if node.child_by_field_name("type").is_some() {
                *typed += 1;
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            walk(child, total, typed);
        }
    }

    let mut total = 0;
    let mut typed = 0;
    walk(root, &mut total, &mut typed);
    if total == 0 {
        1.0
    } else {
        typed as f64 / total as f64
    }
}

#[cfg(feature = "semantic")]
fn extract_call_graph_impl(source: &str, tree: &Tree) -> Option<CallGraph> {
    const QUERY_SRC: &str = r#"
        (function_item name: (identifier) @func.def)
        (call_expression function: (identifier) @func.call)
        (call_expression
          function: (field_expression field: (field_identifier) @func.call))
        (call_expression
          function: (scoped_identifier
                      name: (identifier) @func.call))
    "#;

    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    let query = Query::new(&language, QUERY_SRC).ok()?;

    let bytes = source.as_bytes();
    let mut cursor = QueryCursor::new();

    let def_idx = query.capture_index_for_name("func.def");
    let call_idx = query.capture_index_for_name("func.call");

    let mut graph = CallGraph::default();
    let mut seen_nodes = std::collections::HashSet::new();

    // First pass: collect definitions (so callers can be resolved by enclosing
    // function when we see calls).
    let mut defs: Vec<(usize, usize, String)> = Vec::new(); // (start_byte, end_byte, name)
    {
        let mut matches = cursor.matches(&query, tree.root_node(), bytes);
        while let Some(m) = matches.next() {
            for cap in m.captures {
                if Some(cap.index) == def_idx {
                    let text: &str = match cap.node.utf8_text(bytes) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    // The parent function_item gives us the enclosing span.
                    if let Some(parent) = cap.node.parent() {
                        defs.push((parent.start_byte(), parent.end_byte(), text.to_string()));
                    }
                    let name: String = text.to_string();
                    if seen_nodes.insert(name.clone()) {
                        graph.nodes.push(name);
                    }
                }
            }
        }
    }

    // Second pass: collect calls and resolve the enclosing function by
    // searching `defs` for the innermost containing span.
    let mut cursor2 = QueryCursor::new();
    let mut matches = cursor2.matches(&query, tree.root_node(), bytes);
    while let Some(m) = matches.next() {
        for cap in m.captures {
            if Some(cap.index) == call_idx {
                let text: &str = match cap.node.utf8_text(bytes) {
                    Ok(t) => t,
                    Err(_) => continue,
                };
                let callee: String = text.to_string();
                if seen_nodes.insert(callee.clone()) {
                    graph.nodes.push(callee.clone());
                }

                let call_byte = cap.node.start_byte();
                let caller = defs
                    .iter()
                    .filter(|(s, e, _)| *s <= call_byte && call_byte < *e)
                    .min_by_key(|(s, e, _)| e - s)
                    .map(|(_, _, name)| name.clone())
                    .unwrap_or_else(|| "<top-level>".to_string());

                let line = cap.node.start_position().row + 1;
                graph.edges.push(CallEdge {
                    caller,
                    callee,
                    call_site_line: line,
                    weight: 1,
                });
            }
        }
    }

    Some(graph)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "semantic"))]
mod tests {
    use super::*;

    const SIMPLE: &str = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let x: i32 = add(1, 2);
    let _ = x;
}
"#;

    #[test]
    fn test_rust_function_extraction() {
        let g = extract_call_graph(SIMPLE).expect("parse ok");
        assert!(g.nodes.iter().any(|n| n == "add"));
        assert!(g.nodes.iter().any(|n| n == "main"));
    }

    #[test]
    fn test_rust_call_graph() {
        let g = extract_call_graph(SIMPLE).expect("parse ok");
        let has_main_to_add = g
            .edges
            .iter()
            .any(|e| e.caller == "main" && e.callee == "add");
        assert!(
            has_main_to_add,
            "expected main -> add edge, got: {:?}",
            g.edges
        );
    }

    #[test]
    fn test_parse_quality_on_broken_syntax() {
        // Deliberately malformed: missing closing brace and stray tokens.
        let broken = "fn oops( { let x = ;;; ";
        let m = extract_semantic_metrics(broken);
        assert!(
            m.parse_quality < 1.0,
            "broken source must not report perfect parse_quality, got {}",
            m.parse_quality
        );
    }

    #[test]
    fn test_nesting_depth() {
        let src = r#"
fn deep() {
    if true {
        for _ in 0..10 {
            while false {
                let _ = 1;
            }
        }
    }
}
"#;
        let m = extract_semantic_metrics(src);
        // function + if + for + while + inner block => depth >= 4
        assert!(
            m.nesting_depth_max >= 4,
            "expected nesting depth >= 4, got {}",
            m.nesting_depth_max
        );
        assert!(m.nesting_depth_mean > 0.0);
    }

    #[test]
    fn test_defaults_on_empty_source() {
        let m = extract_semantic_metrics("");
        // Empty source still parses successfully to an empty source_file.
        assert!(m.parse_quality >= 0.0);
    }
}
