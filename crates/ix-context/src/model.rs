//! Node, edge, and bundle types for the context DAG.
//!
//! # The governance instrument contract
//!
//! Every [`ContextBundle`] produced by a [`crate::walk::Walker`] must satisfy
//! five properties to function as a governance instrument:
//!
//! 1. **Replayable** — `walk_trace` is sufficient to reconstruct the exact
//!    set of visited nodes and edges when fed back through the same
//!    [`crate::index::ProjectIndex`] at the same git SHA.
//! 2. **Hexavalent-labelled** — every node and edge carries a
//!    [`ix_types::Hexavalent`] value. MVP defaults to `Unknown` unless
//!    `ix-code::gates` provides a concrete verdict.
//! 3. **Provenance-tagged** — every node cites its [`NodeProvenance`],
//!    every edge its [`EdgeProvenance`]. No anonymous edges.
//! 4. **Unresolved-preserving** — ambiguous call sites are surfaced as
//!    [`ResolvedOrAmbiguous::Ambiguous`], not silently dropped or
//!    arbitrarily chosen. Ambiguity is signal.
//! 5. **Budget-honouring** — when the walker hits a budget cap it returns a
//!    partial bundle with `truncated: true` rather than panicking or
//!    timing out.

use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Stable IDs
// ---------------------------------------------------------------------------

/// Format a stable function node ID of the form
/// `fn:crate::module::name@path#Lstart-Lend`.
///
/// Stable IDs survive renames and re-indexing because they encode the
/// function's location, not its label. Labels may change (e.g., via rename
/// refactors); IDs may not without an explicit migration pass.
///
/// # Example
///
/// ```
/// use ix_context::model::fn_id;
/// let id = fn_id("ix_math", &["eigen"], "jacobi", "crates/ix-math/src/eigen.rs", 42, 103);
/// assert_eq!(id, "fn:ix_math::eigen::jacobi@crates/ix-math/src/eigen.rs#L42-L103");
/// ```
pub fn fn_id(
    crate_name: &str,
    module_path: &[&str],
    name: &str,
    file: &str,
    line_start: usize,
    line_end: usize,
) -> String {
    let mut path = String::with_capacity(crate_name.len() + 16 + name.len());
    path.push_str(crate_name);
    for segment in module_path {
        path.push_str("::");
        path.push_str(segment);
    }
    path.push_str("::");
    path.push_str(name);
    format!("fn:{path}@{file}#L{line_start}-L{line_end}")
}

/// Format a stable module node ID of the form `mod:crate::module@path`.
pub fn module_id(crate_name: &str, module_path: &[&str], file: &str) -> String {
    let mut path = String::with_capacity(crate_name.len() + 16);
    path.push_str(crate_name);
    for segment in module_path {
        path.push_str("::");
        path.push_str(segment);
    }
    format!("mod:{path}@{file}")
}

/// Format a stable file node ID of the form `file:path`.
pub fn file_id(file: &str) -> String {
    format!("file:{file}")
}

/// Format a stable commit node ID of the form `commit:sha`.
pub fn commit_id(sha: &str) -> String {
    format!("commit:{sha}")
}

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// Shared metadata on every [`ContextNode`] variant — ID, label, belief, and
/// provenance.
///
/// `id` is the stable identifier used in edges; `label` is the
/// human-readable string that may drift across renames.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeMeta {
    /// Stable ID, e.g. `fn:ix_math::eigen::jacobi@crates/ix-math/src/eigen.rs#L42-L103`.
    pub id: String,
    /// Human-readable label. May change across renames; `id` may not.
    pub label: String,
    /// Hexavalent belief about this node's presence/validity. Default
    /// [`Hexavalent::Unknown`] unless `ix-code::gates` overrides.
    pub belief: Hexavalent,
    /// Where this node was discovered.
    pub provenance: NodeProvenance,
}

/// Where a [`ContextNode`] was discovered in the source of truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum NodeProvenance {
    /// Parsed from source via `tree-sitter` at `file` spanning byte range
    /// `span.0..span.1`. Line-accurate spans live inside the node variants
    /// that care about them.
    TreeSitter { file: String, span: (usize, usize) },
    /// Surfaced by a git-history walk at the given commit SHA.
    GitHistory { commit: String },
    /// Produced by persistent-homology clustering of the call graph. Not
    /// used in MVP walks but reserved so v2 stopping-rule integration does
    /// not require a schema change.
    Topology { cluster_id: u32 },
    /// Labelled by `ix-code::gates` based on a hexavalent verdict derivation.
    Gate { verdict_source: String },
}

/// A function definition or call target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionNode {
    pub meta: NodeMeta,
    /// Fully-qualified crate path, e.g. `ix_math::eigen::jacobi`.
    pub qualified: String,
    /// File the function was defined in (may be empty for external calls).
    pub file: String,
    /// 1-based line range `(start, end)` if known.
    pub span: Option<(usize, usize)>,
}

/// A Rust module (file or inline `mod { ... }`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModuleNode {
    pub meta: NodeMeta,
    /// Fully-qualified module path, e.g. `ix_math::eigen`.
    pub qualified: String,
    /// File backing the module, if any.
    pub file: String,
}

/// A source file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileNode {
    pub meta: NodeMeta,
    /// Workspace-relative path.
    pub path: String,
}

/// A test function (annotated with `#[test]` or living under `#[cfg(test)]`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestNode {
    pub meta: NodeMeta,
    pub qualified: String,
    pub file: String,
    pub span: Option<(usize, usize)>,
}

/// A git commit surfaced by the trajectory walker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommitNode {
    pub meta: NodeMeta,
    pub sha: String,
    /// ISO-8601 commit timestamp, if available.
    pub timestamp: Option<String>,
    /// Commit summary (first line of the message).
    pub summary: String,
}

/// A symbol that couldn't be classified into one of the concrete node
/// variants — e.g., an external dependency identifier or a macro-generated
/// name. Surfaced rather than hidden so walks remain legible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolNode {
    pub meta: NodeMeta,
    pub name: String,
}

/// The six kinds of node that can appear in a [`ContextBundle`].
///
/// This is a tagged enum rather than a flat `{ kind, data }` dict because
/// Rust's exhaustive match beats schemaless dicts for consumer correctness.
/// Serialized with `#[serde(tag = "kind")]` for wire-format clarity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContextNode {
    Function(FunctionNode),
    Module(ModuleNode),
    File(FileNode),
    Test(TestNode),
    Commit(CommitNode),
    Symbol(SymbolNode),
}

impl ContextNode {
    /// Borrow the shared [`NodeMeta`] regardless of variant.
    pub fn meta(&self) -> &NodeMeta {
        match self {
            ContextNode::Function(n) => &n.meta,
            ContextNode::Module(n) => &n.meta,
            ContextNode::File(n) => &n.meta,
            ContextNode::Test(n) => &n.meta,
            ContextNode::Commit(n) => &n.meta,
            ContextNode::Symbol(n) => &n.meta,
        }
    }

    /// Stable node ID — shorthand for `self.meta().id.as_str()`.
    pub fn id(&self) -> &str {
        self.meta().id.as_str()
    }
}

// ---------------------------------------------------------------------------
// Edge types
// ---------------------------------------------------------------------------

/// The target end of a [`ContextEdge`]. Preserves resolver ambiguity rather
/// than arbitrarily collapsing it — the multi-AI brainstorm's Codex insight:
/// *"ambiguity is signal, the resolver must not lie about confidence."*
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ResolvedOrAmbiguous {
    /// Single unambiguous target node ID.
    Resolved { id: String },
    /// Multiple candidate target node IDs — e.g., a trait method with
    /// several implementations. Walks surface *all* candidates, never a
    /// picked-at-random one.
    Ambiguous { candidates: Vec<String> },
    /// No candidate found. The original textual hint is preserved so
    /// downstream consumers (or humans) can reason about why.
    Unresolved { hint: String },
}

/// Where a [`ContextEdge`] came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum EdgeProvenance {
    /// A syntactic call in source at the given 1-based line number.
    AstCall { call_site_line: usize },
    /// A resolver hop through a `use X::Y as Z;` alias.
    ImportHint { via_use_alias: String },
    /// An invisible structural edge from git trajectory: the two files are
    /// co-changed in `commits_shared` commits out of the target's history,
    /// yielding a confidence in `[0, 1]`.
    GitCochange {
        commits_shared: u32,
        confidence: f64,
    },
    /// An edge from a function to one of its covering test functions.
    TestReference { test_fn_id: String },
    /// An edge from a node to a sibling in the same parent module.
    Sibling { parent_module: String },
}

/// A directed edge in the context DAG.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextEdge {
    /// Source node ID (stable).
    pub from: String,
    /// Target node — resolved, ambiguous, or unresolved.
    pub to: ResolvedOrAmbiguous,
    /// Where this edge came from.
    pub provenance: EdgeProvenance,
    /// Edge weight. `1.0` for plain AST calls; co-change edges carry their
    /// confidence-weighted share here.
    pub weight: f64,
    /// Hexavalent belief about the edge's validity. MVP default is
    /// [`Hexavalent::Unknown`] unless a gate verdict narrows it.
    pub belief: Hexavalent,
}

// ---------------------------------------------------------------------------
// Walk trace + bundle
// ---------------------------------------------------------------------------

/// One step in a walker's replayable trace.
///
/// The trace is the governance instrument: replaying the same trace through
/// the same [`crate::index::ProjectIndex`] at the same git SHA must
/// reconstruct the exact set of visited nodes and edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WalkStep {
    /// Monotonic step index within a single walk.
    pub step: usize,
    /// Node being acted on at this step (the "current cursor").
    pub node_id: String,
    /// What the walker did at this step.
    pub action: WalkAction,
}

/// The concrete action recorded in a [`WalkStep`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WalkAction {
    /// Walker started from the given root.
    Start,
    /// Walker visited a node and added it to the bundle.
    VisitNode,
    /// Walker traversed an edge from the current node to `target`.
    TraverseEdge { target: ResolvedOrAmbiguous },
    /// Walker skipped a candidate because the budget was exhausted.
    BudgetSkip { reason: String },
    /// Walker reached its configured budget and truncated the bundle.
    Truncated { reason: String },
    /// Walker completed normally, exhausting the frontier.
    Complete,
}

/// The top-level output of a walk — nodes, edges, metadata, and the
/// replayable trace that makes it a governance instrument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextBundle {
    /// Root node ID the walk started from.
    pub root: String,
    /// The strategy that produced this bundle (echoed for replay).
    pub strategy: String,
    /// All nodes collected during the walk, in visit order.
    pub nodes: Vec<ContextNode>,
    /// All edges collected during the walk.
    pub edges: Vec<ContextEdge>,
    /// Count of edges whose `to` is
    /// [`ResolvedOrAmbiguous::Ambiguous`] or
    /// [`ResolvedOrAmbiguous::Unresolved`]. Surfaced so downstream agents
    /// can reason about walk fidelity without having to re-scan the edges.
    pub unresolved_count: usize,
    /// Replayable trace — see struct docs for the invariant.
    pub walk_trace: Vec<WalkStep>,
    /// `true` iff the walker hit its [`crate::walk::WalkBudget`] before
    /// exhausting the frontier. Truncated bundles are still valid results
    /// and must never be thrown away silently.
    pub truncated: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Stable ID formatters ────────────────────────────────────────────

    #[test]
    fn fn_id_with_single_module() {
        let id = fn_id(
            "ix_math",
            &["eigen"],
            "jacobi",
            "crates/ix-math/src/eigen.rs",
            42,
            103,
        );
        assert_eq!(
            id,
            "fn:ix_math::eigen::jacobi@crates/ix-math/src/eigen.rs#L42-L103"
        );
    }

    #[test]
    fn fn_id_with_nested_modules() {
        let id = fn_id(
            "ix_code",
            &["semantic", "extractor"],
            "walk",
            "crates/ix-code/src/semantic.rs",
            160,
            192,
        );
        assert_eq!(
            id,
            "fn:ix_code::semantic::extractor::walk@crates/ix-code/src/semantic.rs#L160-L192"
        );
    }

    #[test]
    fn fn_id_with_no_module_path() {
        let id = fn_id("root", &[], "main", "src/main.rs", 1, 5);
        assert_eq!(id, "fn:root::main@src/main.rs#L1-L5");
    }

    #[test]
    fn module_id_format() {
        let id = module_id("ix_math", &["eigen"], "crates/ix-math/src/eigen.rs");
        assert_eq!(id, "mod:ix_math::eigen@crates/ix-math/src/eigen.rs");
    }

    #[test]
    fn file_id_format() {
        assert_eq!(file_id("crates/ix-math/src/eigen.rs"), "file:crates/ix-math/src/eigen.rs");
    }

    #[test]
    fn commit_id_format() {
        assert_eq!(commit_id("abcdef0"), "commit:abcdef0");
    }

    // ── ContextNode::meta() dispatch ────────────────────────────────────

    fn sample_meta(id: &str) -> NodeMeta {
        NodeMeta {
            id: id.to_string(),
            label: "label".to_string(),
            belief: Hexavalent::Unknown,
            provenance: NodeProvenance::TreeSitter {
                file: "fix.rs".to_string(),
                span: (0, 10),
            },
        }
    }

    #[test]
    fn context_node_function_meta_borrow() {
        let n = ContextNode::Function(FunctionNode {
            meta: sample_meta("fn:foo"),
            qualified: "foo".to_string(),
            file: "foo.rs".to_string(),
            span: Some((1, 2)),
        });
        assert_eq!(n.id(), "fn:foo");
        assert_eq!(n.meta().label, "label");
    }

    #[test]
    fn context_node_file_meta_borrow() {
        let n = ContextNode::File(FileNode {
            meta: sample_meta("file:a.rs"),
            path: "a.rs".to_string(),
        });
        assert_eq!(n.id(), "file:a.rs");
    }

    // ── Serde round-trip ────────────────────────────────────────────────

    #[test]
    fn context_node_serde_roundtrip_function() {
        let original = ContextNode::Function(FunctionNode {
            meta: sample_meta("fn:ix_math::eigen::jacobi@x.rs#L1-L10"),
            qualified: "ix_math::eigen::jacobi".to_string(),
            file: "x.rs".to_string(),
            span: Some((1, 10)),
        });
        let json = serde_json::to_string(&original).expect("serialize");
        assert!(
            json.contains(r#""kind":"function""#),
            "tagged serialization missing: {json}"
        );
        let back: ContextNode = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, original);
    }

    #[test]
    fn resolved_or_ambiguous_serde_variants() {
        let resolved = ResolvedOrAmbiguous::Resolved {
            id: "fn:foo".to_string(),
        };
        let ambiguous = ResolvedOrAmbiguous::Ambiguous {
            candidates: vec!["fn:a".to_string(), "fn:b".to_string()],
        };
        let unresolved = ResolvedOrAmbiguous::Unresolved {
            hint: "unknown_call".to_string(),
        };

        for variant in [resolved, ambiguous, unresolved] {
            let json = serde_json::to_string(&variant).expect("serialize");
            let back: ResolvedOrAmbiguous = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, variant);
        }
    }

    #[test]
    fn edge_provenance_tagged_wire_format() {
        let e = EdgeProvenance::GitCochange {
            commits_shared: 7,
            confidence: 0.42,
        };
        let json = serde_json::to_string(&e).expect("serialize");
        assert!(json.contains(r#""source":"git_cochange""#));
        let back: EdgeProvenance = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, e);
    }

    #[test]
    fn context_edge_full_roundtrip() {
        let edge = ContextEdge {
            from: "fn:caller@x.rs#L1-L10".to_string(),
            to: ResolvedOrAmbiguous::Ambiguous {
                candidates: vec!["fn:a".into(), "fn:b".into()],
            },
            provenance: EdgeProvenance::AstCall { call_site_line: 42 },
            weight: 1.0,
            belief: Hexavalent::Probable,
        };
        let json = serde_json::to_string(&edge).expect("serialize");
        let back: ContextEdge = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, edge);
    }

    #[test]
    fn context_bundle_empty_roundtrip() {
        let bundle = ContextBundle {
            root: "fn:root@x.rs#L1-L5".to_string(),
            strategy: "callers_transitive".to_string(),
            nodes: Vec::new(),
            edges: Vec::new(),
            unresolved_count: 0,
            walk_trace: vec![WalkStep {
                step: 0,
                node_id: "fn:root@x.rs#L1-L5".to_string(),
                action: WalkAction::Start,
            }],
            truncated: false,
        };
        let json = serde_json::to_string(&bundle).expect("serialize");
        let back: ContextBundle = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, bundle);
    }

    #[test]
    fn walk_action_serialization_is_tagged() {
        let action = WalkAction::TraverseEdge {
            target: ResolvedOrAmbiguous::Resolved {
                id: "fn:callee".to_string(),
            },
        };
        let json = serde_json::to_string(&action).expect("serialize");
        assert!(json.contains(r#""kind":"traverse_edge""#));
        let back: WalkAction = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, action);
    }

    #[test]
    fn walk_action_truncated_carries_reason() {
        let action = WalkAction::Truncated {
            reason: "max_nodes=100 reached".to_string(),
        };
        let json = serde_json::to_string(&action).expect("serialize");
        let back: WalkAction = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, action);
    }

    // ── Hexavalent label is preserved across wire format ────────────────

    #[test]
    fn node_meta_hexavalent_serializes_as_symbol() {
        let meta = NodeMeta {
            id: "fn:x".to_string(),
            label: "x".to_string(),
            belief: Hexavalent::Probable,
            provenance: NodeProvenance::GitHistory {
                commit: "abcdef".to_string(),
            },
        };
        let json = serde_json::to_string(&meta).expect("serialize");
        assert!(
            json.contains(r#""belief":"P""#),
            "hexavalent belief should serialize as single-letter symbol: {json}"
        );
        let back: NodeMeta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, meta);
    }
}
