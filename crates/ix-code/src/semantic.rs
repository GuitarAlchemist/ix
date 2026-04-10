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

/// A hint describing the syntactic shape of a call site's target, as seen
/// at the AST level — **before** any cross-file symbol resolution. Downstream
/// consumers (e.g. `ix-context`) use this to drive project-wide call-site
/// resolution instead of guessing from a bare identifier.
///
/// This type intentionally preserves *more* information than the call graph
/// itself strictly needs, so that a resolver can disambiguate calls that
/// look identical at the bare-name level (`foo` vs. `bar::foo` vs.
/// `receiver.foo()`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CalleeHint {
    /// A bare identifier call, e.g. `foo()`. The string is the identifier
    /// text verbatim, with no scoping.
    Bare {
        /// The identifier name as written at the call site.
        name: String,
    },
    /// A scoped path call, e.g. `foo::bar::baz()` or `crate::m::f()`. The
    /// segments are split on `::` in source order, so `foo::bar::baz`
    /// becomes `["foo", "bar", "baz"]`. The final segment is the function
    /// name; the prefix is the scope hint.
    Scoped {
        /// Path segments in source order. Always non-empty.
        segments: Vec<String>,
    },
    /// A method-call expression, e.g. `rx.send(msg)`. Captures the receiver
    /// text (as a hint — the receiver may be an arbitrary expression the
    /// resolver cannot interpret) and the method name.
    MethodCall {
        /// Raw source text of the receiver expression, if small enough to
        /// be useful. `None` when the receiver is a large expression the
        /// extractor chose not to capture.
        receiver_hint: Option<String>,
        /// Method identifier as written at the call site.
        method: String,
    },
}

impl CalleeHint {
    /// Returns the "final segment" name — the bare identifier the call
    /// resolves to ignoring all scoping. Useful for cheap name-based
    /// indexing when full resolution is overkill.
    pub fn name(&self) -> &str {
        match self {
            CalleeHint::Bare { name } => name,
            CalleeHint::Scoped { segments } => segments.last().map(String::as_str).unwrap_or(""),
            CalleeHint::MethodCall { method, .. } => method,
        }
    }
}

/// A single directed edge in a call graph: `caller -> callee` at a specific
/// source line, with an aggregation `weight` (number of call sites fused into
/// this edge).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallEdge {
    /// Name of the enclosing function that performs the call. `"<top-level>"`
    /// is used for calls that occur outside of any function body.
    pub caller: String,
    /// Syntactic hint describing the call target. Richer than a bare
    /// string so downstream cross-file resolvers can disambiguate
    /// scoped-path, bare, and method calls without reparsing the source.
    pub callee_hint: CalleeHint,
    /// 1-based source line of the call site.
    pub call_site_line: usize,
    /// Number of distinct call sites merged into this edge. Populated as 1 by
    /// [`extract_call_graph`]; downstream passes may aggregate.
    pub weight: u32,
}

impl CallEdge {
    /// Bare-name convenience: the final segment of whatever the hint
    /// captured. Equivalent to `self.callee_hint.name()`.
    pub fn callee_name(&self) -> &str {
        self.callee_hint.name()
    }
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

/// Iterative pre-order walk over the entire tree rooted at `root`.
///
/// All tree-walkers in this module use a `TreeCursor` instead of native
/// recursion because tree-sitter can produce very deep syntax trees —
/// machine-generated code with 10,000+ nested blocks will blow the Rust
/// stack long before it runs out of memory. The iterative walker uses
/// heap storage proportional to tree *depth* only.
#[cfg(feature = "semantic")]
fn walk_preorder<F>(root: Node, mut visit: F)
where
    F: FnMut(Node),
{
    let mut cursor = root.walk();
    // Standard cursor-based pre-order traversal: visit, descend, then
    // backtrack to the next sibling. Depth tracking is maintained by the
    // cursor itself via goto_parent on backtrack.
    visit(cursor.node());
    loop {
        if cursor.goto_first_child() {
            visit(cursor.node());
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                visit(cursor.node());
                break;
            }
            if !cursor.goto_parent() {
                return;
            }
        }
    }
}

#[cfg(feature = "semantic")]
fn count_nodes(node: Node) -> usize {
    let mut count = 0;
    walk_preorder(node, |_| count += 1);
    count
}

#[cfg(feature = "semantic")]
fn count_error_nodes(node: Node) -> usize {
    let mut count = 0;
    walk_preorder(node, |n| {
        if n.is_error() || n.is_missing() {
            count += 1;
        }
    });
    count
}

/// Walks the tree and returns `(max_depth, mean_depth)` over block-bearing
/// nodes (functions, blocks, loops, conditionals, matches).
///
/// Iterative implementation using a `TreeCursor` with an explicit stack of
/// per-level block depths, safe on arbitrarily deep trees.
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

    let mut cursor = root.walk();
    // Parallel stack of "block depth at this cursor position". Push on
    // descent, pop on parent. The current depth is always stack.last().
    let mut depth_stack: Vec<usize> = Vec::with_capacity(64);
    let root_depth = if is_block(cursor.node().kind()) { 1 } else { 0 };
    depth_stack.push(root_depth);

    let mut max_depth = root_depth;
    let mut depth_sum = root_depth;
    let mut block_count = if root_depth > 0 { 1 } else { 0 };

    loop {
        if cursor.goto_first_child() {
            let parent_depth = *depth_stack.last().unwrap();
            let child_depth = if is_block(cursor.node().kind()) {
                parent_depth + 1
            } else {
                parent_depth
            };
            depth_stack.push(child_depth);
            if is_block(cursor.node().kind()) {
                max_depth = max_depth.max(child_depth);
                depth_sum += child_depth;
                block_count += 1;
            }
            continue;
        }
        loop {
            depth_stack.pop();
            if cursor.goto_next_sibling() {
                let parent_depth = *depth_stack.last().unwrap();
                let sib_depth = if is_block(cursor.node().kind()) {
                    parent_depth + 1
                } else {
                    parent_depth
                };
                depth_stack.push(sib_depth);
                if is_block(cursor.node().kind()) {
                    max_depth = max_depth.max(sib_depth);
                    depth_sum += sib_depth;
                    block_count += 1;
                }
                break;
            }
            if !cursor.goto_parent() {
                let mean = if block_count == 0 {
                    0.0
                } else {
                    depth_sum as f64 / block_count as f64
                };
                return (max_depth, mean);
            }
        }
    }
}

#[cfg(feature = "semantic")]
fn count_kind(root: Node, kinds: &[&str]) -> usize {
    let mut count = 0;
    walk_preorder(root, |n| {
        if kinds.contains(&n.kind()) {
            count += 1;
        }
    });
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
    // Two query families:
    //   1. Function definitions — used to resolve the enclosing caller.
    //   2. Call targets — we capture the *whole* call_expression and then
    //      inspect its `function` child inside the Rust-side loop so we can
    //      distinguish bare / scoped / method calls and build a
    //      `CalleeHint` variant accordingly.
    //
    // Capturing at the call_expression level (rather than narrowly on the
    // leaf identifier) is deliberate: the earlier design captured only the
    // final segment of scoped paths, which silently dropped the scope
    // information the downstream resolver in `ix-context` relies on.
    const QUERY_SRC: &str = r#"
        (function_item name: (identifier) @func.def)
        (call_expression) @call.expr
    "#;

    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    let query = Query::new(&language, QUERY_SRC).ok()?;

    let bytes = source.as_bytes();
    let mut cursor = QueryCursor::new();

    let def_idx = query.capture_index_for_name("func.def");
    let call_expr_idx = query.capture_index_for_name("call.expr");

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

    // Second pass: inspect each call_expression to classify the call target.
    let mut cursor2 = QueryCursor::new();
    let mut matches = cursor2.matches(&query, tree.root_node(), bytes);
    while let Some(m) = matches.next() {
        for cap in m.captures {
            if Some(cap.index) != call_expr_idx {
                continue;
            }
            let call_node = cap.node;
            let Some(func_node) = call_node.child_by_field_name("function") else {
                continue;
            };

            let Some(hint) = classify_call_target(func_node, bytes) else {
                // Unclassifiable call (macro expansion, closure invocation,
                // etc.) — drop rather than guess.
                continue;
            };

            // Bare-name for node-set bookkeeping; the richer hint is stored
            // on the edge.
            let bare_name = hint.name().to_string();
            if bare_name.is_empty() {
                continue;
            }
            if seen_nodes.insert(bare_name.clone()) {
                graph.nodes.push(bare_name);
            }

            let call_byte = func_node.start_byte();
            let caller = defs
                .iter()
                .filter(|(s, e, _)| *s <= call_byte && call_byte < *e)
                .min_by_key(|(s, e, _)| e - s)
                .map(|(_, _, name)| name.clone())
                .unwrap_or_else(|| "<top-level>".to_string());

            let line = func_node.start_position().row + 1;
            graph.edges.push(CallEdge {
                caller,
                callee_hint: hint,
                call_site_line: line,
                weight: 1,
            });
        }
    }

    Some(graph)
}

/// Inspect a tree-sitter `function`-field node attached to a call expression
/// and turn it into a [`CalleeHint`] variant. Returns `None` for shapes the
/// extractor deliberately refuses to classify (closure calls, macro-expanded
/// invocations, etc.) so that the caller can drop them rather than pick a
/// misleading default.
#[cfg(feature = "semantic")]
fn classify_call_target(func_node: Node, bytes: &[u8]) -> Option<CalleeHint> {
    match func_node.kind() {
        // Bare `foo()` call.
        "identifier" => {
            let name = func_node.utf8_text(bytes).ok()?.to_string();
            Some(CalleeHint::Bare { name })
        }
        // Scoped `foo::bar::baz()` call — walk the full scoped_identifier
        // and collect every `identifier` leaf in source order.
        "scoped_identifier" => {
            let segments = collect_scoped_segments(func_node, bytes);
            if segments.is_empty() {
                None
            } else {
                Some(CalleeHint::Scoped { segments })
            }
        }
        // Method call `receiver.method()` — pull the method name from the
        // `field` child and the receiver text from the `value` child.
        "field_expression" => {
            let method = func_node
                .child_by_field_name("field")?
                .utf8_text(bytes)
                .ok()?
                .to_string();
            let receiver_hint = func_node
                .child_by_field_name("value")
                .and_then(|n| n.utf8_text(bytes).ok())
                .map(|s| {
                    // Keep the hint compact: long receivers (indexing
                    // chains, closures, full expressions) stop being useful
                    // after a few dozen chars.
                    if s.len() <= 64 {
                        s.to_string()
                    } else {
                        format!("{}…", &s[..60])
                    }
                });
            Some(CalleeHint::MethodCall {
                receiver_hint,
                method,
            })
        }
        // Generic method call like `foo::<T>()` or `Vec::<i32>::new()` —
        // the `function` child is a `generic_function` wrapping the real
        // target. Recurse on the inner `function` field.
        "generic_function" => {
            let inner = func_node.child_by_field_name("function")?;
            classify_call_target(inner, bytes)
        }
        // Everything else (closure calls, paren-wrapped expressions, etc.)
        // is deliberately unclassified.
        _ => None,
    }
}

/// Collect the `::`-separated segments of a `scoped_identifier` subtree in
/// source order. Walks the `path`/`name` field pair recursively so that
/// nested scoping (`a::b::c::d`) is fully flattened.
#[cfg(feature = "semantic")]
fn collect_scoped_segments(node: Node, bytes: &[u8]) -> Vec<String> {
    let mut segments: Vec<String> = Vec::new();
    walk_scoped(node, bytes, &mut segments);
    segments
}

#[cfg(feature = "semantic")]
fn walk_scoped(node: Node, bytes: &[u8], out: &mut Vec<String>) {
    match node.kind() {
        "scoped_identifier" => {
            if let Some(path) = node.child_by_field_name("path") {
                walk_scoped(path, bytes, out);
            }
            if let Some(name) = node.child_by_field_name("name") {
                if let Ok(text) = name.utf8_text(bytes) {
                    out.push(text.to_string());
                }
            }
        }
        "identifier" | "super" | "self" | "crate" | "metavariable" => {
            if let Ok(text) = node.utf8_text(bytes) {
                out.push(text.to_string());
            }
        }
        _ => {
            // Unknown path shape; try direct text fallback once.
            if let Ok(text) = node.utf8_text(bytes) {
                out.push(text.to_string());
            }
        }
    }
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
            .any(|e| e.caller == "main" && e.callee_name() == "add");
        assert!(
            has_main_to_add,
            "expected main -> add edge, got: {:?}",
            g.edges
        );
    }

    #[test]
    fn test_bare_call_hint_variant() {
        let g = extract_call_graph(SIMPLE).expect("parse ok");
        let edge = g
            .edges
            .iter()
            .find(|e| e.callee_name() == "add")
            .expect("main -> add edge missing");
        assert!(
            matches!(&edge.callee_hint, CalleeHint::Bare { name } if name == "add"),
            "expected Bare hint for `add()`, got {:?}",
            edge.callee_hint
        );
    }

    #[test]
    fn test_scoped_call_preserves_all_segments() {
        // The key regression Prerequisite B fixes: the old extractor
        // captured only the final `::` segment, so a call site to
        // `foo::bar::baz()` was indistinguishable from a bare `baz()`.
        // The new hint must preserve every segment in source order.
        let src = r#"
mod foo {
    pub mod bar {
        pub fn baz() -> i32 { 42 }
    }
}

fn caller() {
    let _ = foo::bar::baz();
}
"#;
        let g = extract_call_graph(src).expect("parse ok");
        let edge = g
            .edges
            .iter()
            .find(|e| e.caller == "caller")
            .expect("caller -> baz edge missing");
        match &edge.callee_hint {
            CalleeHint::Scoped { segments } => {
                assert_eq!(
                    segments,
                    &vec!["foo".to_string(), "bar".to_string(), "baz".to_string()],
                    "expected full scoped path preserved, got {:?}",
                    segments
                );
            }
            other => panic!("expected Scoped hint, got {:?}", other),
        }
        assert_eq!(edge.callee_name(), "baz");
    }

    #[test]
    fn test_method_call_captures_receiver_and_method() {
        let src = r#"
struct Rx;
impl Rx {
    fn send(&self, _msg: i32) {}
}

fn main() {
    let rx = Rx;
    rx.send(1);
}
"#;
        let g = extract_call_graph(src).expect("parse ok");
        let edge = g
            .edges
            .iter()
            .find(|e| e.callee_name() == "send")
            .expect("rx.send edge missing");
        match &edge.callee_hint {
            CalleeHint::MethodCall {
                receiver_hint,
                method,
            } => {
                assert_eq!(method, "send");
                assert_eq!(receiver_hint.as_deref(), Some("rx"));
            }
            other => panic!("expected MethodCall hint, got {:?}", other),
        }
    }

    #[test]
    fn test_callee_hint_serde_roundtrip() {
        // CalleeHint is serialized into ContextBundle JSON by downstream
        // crates — verify round-trip stays stable.
        let hint = CalleeHint::Scoped {
            segments: vec!["ix_math".into(), "eigen".into(), "jacobi".into()],
        };
        let json = serde_json::to_string(&hint).expect("serialize");
        let back: CalleeHint = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(hint, back);
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

    #[test]
    fn test_deep_nesting_does_not_stack_overflow() {
        // Build a file with 200 nested `if true {}` blocks. The previous
        // recursive walker would consume ~200 Rust stack frames per
        // traversal (one per function call per child), which on Windows
        // with the default 1 MB thread stack and debug-build frame sizes
        // approached the crash threshold. The iterative walker uses only
        // heap allocation proportional to tree *depth*, so this should be
        // comfortable.
        //
        // (tree-sitter's own C parser has its own recursion limit that
        // kicks in around 500-1000 levels depending on platform, so we
        // stay well under it to isolate the walker behavior.)
        let mut src = String::from("fn deep() {\n");
        for _ in 0..200 {
            src.push_str("  if true {\n");
        }
        src.push_str("    let _ = 1;\n");
        for _ in 0..200 {
            src.push_str("  }\n");
        }
        src.push_str("}\n");

        // Should not panic or stack-overflow.
        let m = extract_semantic_metrics(&src);
        assert!(m.ast_node_count > 0);
        // Deep nesting should be visible in the max depth.
        assert!(m.nesting_depth_max > 10);
    }
}
