//! Walker with four MVP strategies over the resolved project graph.
//!
//! The [`Walker`] wraps a [`crate::index::ProjectIndex`] and a
//! [`crate::resolve::CallSiteResolver`], lazily computes a global reverse
//! adjacency from the per-file call graphs, and produces replayable
//! [`ContextBundle`]s via one of four strategies.
//!
//! # MVP scope
//!
//! - **Caller/callee** walks operate on **free functions only**. Edges
//!   where the caller is an impl-block method fall back to a best-effort
//!   bare-name match and may under-count or misattribute. Full
//!   method-as-caller support is deferred to v2.
//! - **Git co-change** uses [`ix_code::trajectory::compute_trajectory`]
//!   internally; the walk can only succeed when the workspace is a git
//!   working tree. Outside a git working tree the walk returns an empty
//!   bundle with a single `BudgetSkip` trace step explaining why.
//! - **Module siblings** is a purely structural walk — it does not touch
//!   the call graph, just the symbol table.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use ix_code::semantic::CalleeHint;
use ix_types::Hexavalent;
use serde::{Deserialize, Serialize};

use crate::index::{DefSite, ProjectIndex, SymbolKey};
use crate::model::{
    ContextBundle, ContextEdge, ContextNode, EdgeProvenance, FunctionNode, NodeMeta,
    NodeProvenance, ResolvedOrAmbiguous, WalkAction, WalkStep,
};
use crate::resolve::CallSiteResolver;

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/// Budget constraints enforced on every walk.
///
/// When the walker hits any of the three caps it stops visiting new nodes,
/// records a [`WalkAction::Truncated`] step, and returns a bundle with
/// `truncated = true`. Truncated bundles are still valid results — never
/// discarded silently.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WalkBudget {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub timeout: Duration,
}

impl WalkBudget {
    /// A generous default useful for tests and small walks.
    pub fn default_generous() -> Self {
        Self {
            max_nodes: 1024,
            max_edges: 4096,
            timeout: Duration::from_secs(30),
        }
    }
}

/// The four MVP walk strategies from the brainstorm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum WalkStrategy {
    /// BFS over reverse call edges from the root: who calls this function?
    CallersTransitive { max_depth: u8 },
    /// BFS over forward call edges from the root: what does this function call?
    CalleesTransitive { max_depth: u8 },
    /// Enumerate all functions in the same `(crate, module_path)` as the root.
    ModuleSiblings,
    /// Find files that co-change with the root's file in git history,
    /// filtered by `min_commits_shared`.
    GitCochange { min_commits_shared: u32 },
}

impl WalkStrategy {
    fn name(self) -> &'static str {
        match self {
            WalkStrategy::CallersTransitive { .. } => "callers_transitive",
            WalkStrategy::CalleesTransitive { .. } => "callees_transitive",
            WalkStrategy::ModuleSiblings => "module_siblings",
            WalkStrategy::GitCochange { .. } => "git_cochange",
        }
    }
}

/// The walker. Owns a resolver and a lazily-computed reverse adjacency,
/// borrows the index.
pub struct Walker<'a> {
    index: &'a ProjectIndex,
    resolver: CallSiteResolver<'a>,
}

impl<'a> Walker<'a> {
    /// Construct a walker over an existing index.
    pub fn new(index: &'a ProjectIndex) -> Self {
        Self {
            index,
            resolver: CallSiteResolver::new(index),
        }
    }

    /// Walk the graph starting from a fully-qualified free-function path
    /// like `"ix_math::eigen::jacobi"`.
    ///
    /// Returns a [`ContextBundle`] that is always well-formed even when
    /// truncated, the root is not found, or git history is unavailable.
    /// Failures manifest as bundles with zero edges and an explanatory
    /// trace step, not errors.
    pub fn walk_from_free_fn(
        &self,
        qualified: &str,
        strategy: WalkStrategy,
        budget: WalkBudget,
    ) -> ContextBundle {
        let start = Instant::now();
        let strategy_name = strategy.name().to_string();

        let Some(root_def) = self
            .index
            .symbols
            .get(&SymbolKey::FreeFn(qualified.to_string()))
            .cloned()
        else {
            // Unknown root — return an empty bundle with an explanatory trace.
            return ContextBundle {
                root: format!("unresolved:{qualified}"),
                strategy: strategy_name,
                nodes: Vec::new(),
                edges: Vec::new(),
                unresolved_count: 0,
                walk_trace: vec![WalkStep {
                    step: 0,
                    node_id: format!("unresolved:{qualified}"),
                    action: WalkAction::BudgetSkip {
                        reason: format!("root {qualified} not found in symbol table"),
                    },
                }],
                truncated: false,
            };
        };

        let root_name = qualified
            .rsplit("::")
            .next()
            .unwrap_or(qualified)
            .to_string();
        let root_id = root_def.to_fn_id(&root_name);

        let mut bundle = ContextBundle {
            root: root_id.clone(),
            strategy: strategy_name,
            nodes: Vec::new(),
            edges: Vec::new(),
            unresolved_count: 0,
            walk_trace: vec![WalkStep {
                step: 0,
                node_id: root_id.clone(),
                action: WalkAction::Start,
            }],
            truncated: false,
        };

        // Seed the bundle with the root node.
        let root_node = function_node_from_def(&root_def, &root_name, qualified, Hexavalent::Unknown);
        bundle.nodes.push(root_node);
        bundle.walk_trace.push(WalkStep {
            step: 1,
            node_id: root_id.clone(),
            action: WalkAction::VisitNode,
        });

        // Dispatch.
        match strategy {
            WalkStrategy::CallersTransitive { max_depth } => {
                self.walk_callers(&root_id, &root_def, &root_name, max_depth, budget, start, &mut bundle);
            }
            WalkStrategy::CalleesTransitive { max_depth } => {
                self.walk_callees(&root_id, &root_def, &root_name, max_depth, budget, start, &mut bundle);
            }
            WalkStrategy::ModuleSiblings => {
                self.walk_siblings(&root_id, &root_def, &root_name, budget, start, &mut bundle);
            }
            WalkStrategy::GitCochange { min_commits_shared } => {
                self.walk_cochange(&root_id, &root_def, min_commits_shared, budget, start, &mut bundle);
            }
        }

        if !bundle.truncated {
            let step_n = bundle.walk_trace.len();
            bundle.walk_trace.push(WalkStep {
                step: step_n,
                node_id: root_id,
                action: WalkAction::Complete,
            });
        }

        bundle.unresolved_count = bundle
            .edges
            .iter()
            .filter(|e| {
                matches!(
                    e.to,
                    ResolvedOrAmbiguous::Ambiguous { .. } | ResolvedOrAmbiguous::Unresolved { .. }
                )
            })
            .count();

        bundle
    }

    // ── Callers-transitive ──────────────────────────────────────────────

    fn walk_callers(
        &self,
        root_id: &str,
        _root_def: &DefSite,
        root_name: &str,
        max_depth: u8,
        budget: WalkBudget,
        start: Instant,
        bundle: &mut ContextBundle,
    ) {
        // Compute the global reverse adjacency: target_fn_id -> Vec<(caller_fn_id, call_site_line)>
        let reverse = self.build_reverse_adjacency_for_name(root_name);

        // BFS from root outward over reverse edges.
        let mut frontier: VecDeque<(String, u8)> = VecDeque::new();
        frontier.push_back((root_id.to_string(), 0));
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(root_id.to_string());

        while let Some((node_id, depth)) = frontier.pop_front() {
            if self.budget_exceeded(&budget, start, bundle) {
                return;
            }
            if depth >= max_depth {
                continue;
            }
            let Some(callers) = reverse.get(&node_id) else {
                continue;
            };
            for (caller_id, caller_def, caller_name, caller_qualified, line) in callers {
                if self.budget_exceeded(&budget, start, bundle) {
                    return;
                }

                // Add edge from caller -> current node (a call into the node).
                let edge = ContextEdge {
                    from: caller_id.clone(),
                    to: ResolvedOrAmbiguous::Resolved {
                        id: node_id.clone(),
                    },
                    provenance: EdgeProvenance::AstCall {
                        call_site_line: *line,
                    },
                    weight: 1.0,
                    belief: Hexavalent::Unknown,
                };
                bundle.edges.push(edge);
                let step_n = bundle.walk_trace.len();
                bundle.walk_trace.push(WalkStep {
                    step: step_n,
                    node_id: caller_id.clone(),
                    action: WalkAction::TraverseEdge {
                        target: ResolvedOrAmbiguous::Resolved {
                            id: node_id.clone(),
                        },
                    },
                });

                if visited.insert(caller_id.clone()) {
                    let node = function_node_from_def(
                        caller_def,
                        caller_name,
                        caller_qualified,
                        Hexavalent::Unknown,
                    );
                    bundle.nodes.push(node);
                    let step_n = bundle.walk_trace.len();
                    bundle.walk_trace.push(WalkStep {
                        step: step_n,
                        node_id: caller_id.clone(),
                        action: WalkAction::VisitNode,
                    });
                    frontier.push_back((caller_id.clone(), depth + 1));
                }
            }
        }
    }

    // ── Callees-transitive ──────────────────────────────────────────────

    fn walk_callees(
        &self,
        root_id: &str,
        root_def: &DefSite,
        root_name: &str,
        max_depth: u8,
        budget: WalkBudget,
        start: Instant,
        bundle: &mut ContextBundle,
    ) {
        // For each node in the frontier, find its file, look up its
        // outgoing calls in the per-file call graph, and resolve each.
        let mut frontier: VecDeque<(String, DefSite, String, u8)> = VecDeque::new();
        frontier.push_back((
            root_id.to_string(),
            root_def.clone(),
            root_name.to_string(),
            0,
        ));
        let mut visited: HashSet<String> = HashSet::new();
        visited.insert(root_id.to_string());

        while let Some((node_id, def, name, depth)) = frontier.pop_front() {
            if self.budget_exceeded(&budget, start, bundle) {
                return;
            }
            if depth >= max_depth {
                continue;
            }
            let Some(file_index) = self.index.files.get(&def.file) else {
                continue;
            };
            for edge in &file_index.call_graph.edges {
                if edge.caller != name {
                    continue;
                }
                if self.budget_exceeded(&budget, start, bundle) {
                    return;
                }
                let resolved = self.resolver.resolve(
                    &edge.callee_hint,
                    &def.file,
                    &file_index.crate_name,
                    &file_index.module_path,
                );

                let ctx_edge = ContextEdge {
                    from: node_id.clone(),
                    to: resolved.clone(),
                    provenance: EdgeProvenance::AstCall {
                        call_site_line: edge.call_site_line,
                    },
                    weight: 1.0,
                    belief: Hexavalent::Unknown,
                };
                bundle.edges.push(ctx_edge);
                let step_n = bundle.walk_trace.len();
                bundle.walk_trace.push(WalkStep {
                    step: step_n,
                    node_id: node_id.clone(),
                    action: WalkAction::TraverseEdge {
                        target: resolved.clone(),
                    },
                });

                if let ResolvedOrAmbiguous::Resolved { id } = resolved {
                    if visited.insert(id.clone()) {
                        if let Some((target_def, target_name, qualified)) =
                            lookup_def_by_fn_id(self.index, &id)
                        {
                            let node = function_node_from_def(
                                &target_def,
                                &target_name,
                                &qualified,
                                Hexavalent::Unknown,
                            );
                            bundle.nodes.push(node);
                            let step_n = bundle.walk_trace.len();
                            bundle.walk_trace.push(WalkStep {
                                step: step_n,
                                node_id: id.clone(),
                                action: WalkAction::VisitNode,
                            });
                            frontier.push_back((id, target_def, target_name, depth + 1));
                        }
                    }
                }
            }
        }
    }

    // ── Module siblings ─────────────────────────────────────────────────

    fn walk_siblings(
        &self,
        root_id: &str,
        root_def: &DefSite,
        root_name: &str,
        budget: WalkBudget,
        start: Instant,
        bundle: &mut ContextBundle,
    ) {
        for (path, fi) in &self.index.files {
            if fi.crate_name != root_def.crate_name || fi.module_path != root_def.module_path {
                continue;
            }
            for (name, def) in &fi.free_fns {
                if self.budget_exceeded(&budget, start, bundle) {
                    return;
                }
                // Skip the root itself.
                if path == &root_def.file && name == root_name {
                    continue;
                }
                let qualified = build_qualified_path(&fi.crate_name, &fi.module_path, name);
                let sibling_id = def.to_fn_id(name);
                let node = function_node_from_def(def, name, &qualified, Hexavalent::Unknown);
                bundle.nodes.push(node);
                let step_n = bundle.walk_trace.len();
                bundle.walk_trace.push(WalkStep {
                    step: step_n,
                    node_id: sibling_id.clone(),
                    action: WalkAction::VisitNode,
                });

                let edge = ContextEdge {
                    from: root_id.to_string(),
                    to: ResolvedOrAmbiguous::Resolved {
                        id: sibling_id.clone(),
                    },
                    provenance: EdgeProvenance::Sibling {
                        parent_module: fi.module_path.join("::"),
                    },
                    weight: 1.0,
                    belief: Hexavalent::Unknown,
                };
                bundle.edges.push(edge);
                let step_n = bundle.walk_trace.len();
                bundle.walk_trace.push(WalkStep {
                    step: step_n,
                    node_id: root_id.to_string(),
                    action: WalkAction::TraverseEdge {
                        target: ResolvedOrAmbiguous::Resolved { id: sibling_id },
                    },
                });
            }
        }
    }

    // ── Git co-change ───────────────────────────────────────────────────

    fn walk_cochange(
        &self,
        root_id: &str,
        root_def: &DefSite,
        min_commits_shared: u32,
        budget: WalkBudget,
        start: Instant,
        bundle: &mut ContextBundle,
    ) {
        // Best-effort git-history walk. `git2::Repository::open` walks up
        // to find a .git/ directory; if there isn't one, we gracefully
        // emit a BudgetSkip and exit.
        let repo = match git2::Repository::open(&self.index.workspace_root) {
            Ok(r) => r,
            Err(e) => {
                let step_n = bundle.walk_trace.len();
                bundle.walk_trace.push(WalkStep {
                    step: step_n,
                    node_id: root_id.to_string(),
                    action: WalkAction::BudgetSkip {
                        reason: format!("git2::open failed: {e}"),
                    },
                });
                return;
            }
        };

        // Collect all files co-changed with the root's file. We walk at
        // most `max_nodes * 4` commits to stay within budget regardless of
        // repo history depth.
        let target_file = &root_def.file;
        let mut cochange_counts: HashMap<String, u32> = HashMap::new();
        let mut commit_count = 0usize;
        let commit_cap = budget.max_nodes.saturating_mul(4).max(64);

        let Ok(mut revwalk) = repo.revwalk() else {
            return;
        };
        let _ = revwalk.set_sorting(git2::Sort::TIME);
        if revwalk.push_head().is_err() {
            return;
        }

        for oid_result in revwalk {
            if commit_count >= commit_cap {
                break;
            }
            if self.budget_exceeded(&budget, start, bundle) {
                return;
            }
            let Ok(oid) = oid_result else {
                continue;
            };
            let Ok(commit) = repo.find_commit(oid) else {
                continue;
            };
            let Ok(tree) = commit.tree() else {
                continue;
            };
            let parent_tree = if commit.parent_count() > 0 {
                commit
                    .parent(0)
                    .ok()
                    .and_then(|p| p.tree().ok())
            } else {
                None
            };
            let diff = match repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), None) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let mut touched: HashSet<String> = HashSet::new();
            let _ = diff.foreach(
                &mut |delta, _| {
                    if let Some(path) = delta.new_file().path().or_else(|| delta.old_file().path()) {
                        if let Some(p) = path.to_str() {
                            touched.insert(p.replace('\\', "/"));
                        }
                    }
                    true
                },
                None,
                None,
                None,
            );

            if touched.contains(target_file) {
                for other in touched.iter().filter(|p| *p != target_file) {
                    *cochange_counts.entry(other.clone()).or_insert(0) += 1;
                }
            }
            commit_count += 1;
        }

        // Emit edges for co-changed files that cleared the threshold.
        for (other_file, count) in cochange_counts {
            if count < min_commits_shared {
                continue;
            }
            if self.budget_exceeded(&budget, start, bundle) {
                return;
            }
            let confidence = (count as f64 / commit_count.max(1) as f64).min(1.0);
            let other_id = crate::model::file_id(&other_file);
            let other_node = ContextNode::File(crate::model::FileNode {
                meta: NodeMeta {
                    id: other_id.clone(),
                    label: other_file.clone(),
                    belief: Hexavalent::Unknown,
                    provenance: NodeProvenance::GitHistory {
                        commit: "revwalk".to_string(),
                    },
                },
                path: other_file.clone(),
            });
            bundle.nodes.push(other_node);
            let step_n = bundle.walk_trace.len();
            bundle.walk_trace.push(WalkStep {
                step: step_n,
                node_id: other_id.clone(),
                action: WalkAction::VisitNode,
            });

            let edge = ContextEdge {
                from: root_id.to_string(),
                to: ResolvedOrAmbiguous::Resolved {
                    id: other_id.clone(),
                },
                provenance: EdgeProvenance::GitCochange {
                    commits_shared: count,
                    confidence,
                },
                weight: count as f64,
                belief: Hexavalent::Unknown,
            };
            bundle.edges.push(edge);
        }
    }

    // ── Reverse adjacency ───────────────────────────────────────────────

    /// Build a reverse adjacency scoped to a single target name. We filter
    /// at edge-emission time so the map stays small; the MVP doesn't need
    /// a full workspace-wide reverse graph.
    ///
    /// The value is a list of `(caller_id, caller_def, caller_name,
    /// caller_qualified, line)` tuples.
    #[allow(clippy::type_complexity)]
    fn build_reverse_adjacency_for_name(
        &self,
        target_name: &str,
    ) -> HashMap<String, Vec<(String, DefSite, String, String, usize)>> {
        let mut out: HashMap<String, Vec<(String, DefSite, String, String, usize)>> =
            HashMap::new();

        for (file_path, file_index) in &self.index.files {
            for edge in &file_index.call_graph.edges {
                // Filter by bare callee name — cheaper than running the
                // resolver on every edge in the workspace.
                if edge.callee_hint.name() != target_name {
                    continue;
                }

                // Look up the caller's DefSite. For MVP, only free-fn
                // callers are handled. Methods-as-callers fall through.
                let Some(caller_def) = file_index.free_fns.get(&edge.caller).cloned() else {
                    continue;
                };
                let caller_name = edge.caller.clone();
                let caller_qualified = build_qualified_path(
                    &file_index.crate_name,
                    &file_index.module_path,
                    &caller_name,
                );
                let caller_id = caller_def.to_fn_id(&caller_name);

                // Resolve to get the target's fn_id. This ensures the
                // reverse edge is indexed against the right canonical ID.
                let resolved = self.resolver.resolve(
                    &edge.callee_hint,
                    file_path,
                    &file_index.crate_name,
                    &file_index.module_path,
                );
                let targets: Vec<String> = match resolved {
                    ResolvedOrAmbiguous::Resolved { id } => vec![id],
                    ResolvedOrAmbiguous::Ambiguous { candidates } => candidates,
                    ResolvedOrAmbiguous::Unresolved { .. } => continue,
                };
                for t in targets {
                    out.entry(t).or_default().push((
                        caller_id.clone(),
                        caller_def.clone(),
                        caller_name.clone(),
                        caller_qualified.clone(),
                        edge.call_site_line,
                    ));
                }
            }
        }

        out
    }

    // ── Budget ──────────────────────────────────────────────────────────

    fn budget_exceeded(
        &self,
        budget: &WalkBudget,
        start: Instant,
        bundle: &mut ContextBundle,
    ) -> bool {
        let reason = if bundle.nodes.len() >= budget.max_nodes {
            Some(format!("max_nodes={} reached", budget.max_nodes))
        } else if bundle.edges.len() >= budget.max_edges {
            Some(format!("max_edges={} reached", budget.max_edges))
        } else if start.elapsed() >= budget.timeout {
            Some(format!("timeout={:?} reached", budget.timeout))
        } else {
            None
        };

        if let Some(reason) = reason {
            if !bundle.truncated {
                bundle.truncated = true;
                let step_n = bundle.walk_trace.len();
                bundle.walk_trace.push(WalkStep {
                    step: step_n,
                    node_id: bundle.root.clone(),
                    action: WalkAction::Truncated { reason },
                });
            }
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_qualified_path(crate_name: &str, module_path: &[String], name: &str) -> String {
    if module_path.is_empty() {
        format!("{crate_name}::{name}")
    } else {
        format!("{crate_name}::{}::{name}", module_path.join("::"))
    }
}

fn function_node_from_def(
    def: &DefSite,
    name: &str,
    qualified: &str,
    belief: Hexavalent,
) -> ContextNode {
    let id = def.to_fn_id(name);
    ContextNode::Function(FunctionNode {
        meta: NodeMeta {
            id,
            label: name.to_string(),
            belief,
            provenance: NodeProvenance::TreeSitter {
                file: def.file.clone(),
                span: (def.line_start, def.line_end),
            },
        },
        qualified: qualified.to_string(),
        file: def.file.clone(),
        span: Some((def.line_start, def.line_end)),
    })
}

/// Find the DefSite for a given stable fn_id by linear scan.
///
/// Returns `(def, bare_name, qualified_path)`. Returns `None` if the ID
/// does not match any symbol. O(workspace_size) — acceptable for MVP
/// walks which are budget-bounded.
fn lookup_def_by_fn_id(index: &ProjectIndex, id: &str) -> Option<(DefSite, String, String)> {
    for (key, def) in &index.symbols {
        let candidate_name = match key {
            SymbolKey::FreeFn(path) => path.rsplit("::").next().unwrap_or(path).to_string(),
            SymbolKey::InherentMethod { method, .. } => method.clone(),
            SymbolKey::TraitMethod { method, .. } => method.clone(),
        };
        if def.to_fn_id(&candidate_name) == id {
            let qualified = match key {
                SymbolKey::FreeFn(path) => path.clone(),
                SymbolKey::InherentMethod { ty, method } => format!("{ty}::{method}"),
                SymbolKey::TraitMethod {
                    trait_name,
                    ty,
                    method,
                } => format!("{trait_name}::{ty}::{method}"),
            };
            return Some((def.clone(), candidate_name, qualified));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_fixture(dir: &std::path::Path, files: &[(&str, &str)]) {
        for (rel, contents) in files {
            let path = dir.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create_dir_all");
            }
            fs::write(&path, contents).expect("write fixture");
        }
    }

    fn build_index(files: &[(&str, &str)]) -> (tempfile::TempDir, ProjectIndex) {
        let tmp = tempfile::tempdir().expect("tempdir");
        write_fixture(tmp.path(), files);
        let idx = ProjectIndex::build(tmp.path()).expect("build");
        (tmp, idx)
    }

    // ── Callers-transitive ─────────────────────────────────────────────

    #[test]
    fn callers_transitive_finds_single_caller() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
pub mod eigen;
pub fn consumer() {
    crate::eigen::jacobi();
}
"#,
            ),
            ("crates/mini/src/eigen.rs", "pub fn jacobi() {}\n"),
        ];
        let (_tmp, idx) = build_index(files);
        let walker = Walker::new(&idx);
        let bundle = walker.walk_from_free_fn(
            "mini::eigen::jacobi",
            WalkStrategy::CallersTransitive { max_depth: 3 },
            WalkBudget::default_generous(),
        );
        assert!(!bundle.truncated);
        // Root + consumer
        assert!(bundle.nodes.len() >= 2, "expected >= 2 nodes, got {}: {:#?}", bundle.nodes.len(), bundle.nodes);
        // At least one edge into jacobi
        assert!(
            bundle.edges.iter().any(|e| matches!(&e.to, ResolvedOrAmbiguous::Resolved { id } if id.contains("jacobi"))),
            "no edge into jacobi, edges: {:#?}",
            bundle.edges
        );
    }

    #[test]
    fn callers_transitive_unknown_root_returns_empty_bundle() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("crates/mini/src/lib.rs", "fn main() {}\n"),
        ];
        let (_tmp, idx) = build_index(files);
        let walker = Walker::new(&idx);
        let bundle = walker.walk_from_free_fn(
            "mini::no_such_fn",
            WalkStrategy::CallersTransitive { max_depth: 3 },
            WalkBudget::default_generous(),
        );
        assert!(bundle.nodes.is_empty());
        assert!(bundle.edges.is_empty());
        // Trace should record the reason.
        assert!(bundle
            .walk_trace
            .iter()
            .any(|s| matches!(&s.action, WalkAction::BudgetSkip { reason } if reason.contains("not found"))));
    }

    // ── Callees-transitive ─────────────────────────────────────────────

    #[test]
    fn callees_transitive_finds_direct_callee() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
pub fn helper() {}
pub fn caller() { helper(); }
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let walker = Walker::new(&idx);
        let bundle = walker.walk_from_free_fn(
            "mini::caller",
            WalkStrategy::CalleesTransitive { max_depth: 2 },
            WalkBudget::default_generous(),
        );
        assert!(!bundle.truncated);
        assert!(
            bundle.edges.iter().any(|e| matches!(&e.to, ResolvedOrAmbiguous::Resolved { id } if id.contains("helper"))),
            "no edge caller -> helper, edges: {:#?}",
            bundle.edges
        );
    }

    // ── Module siblings ────────────────────────────────────────────────

    #[test]
    fn siblings_enumerates_same_module_free_fns() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
pub fn alpha() {}
pub fn beta() {}
pub fn gamma() {}
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let walker = Walker::new(&idx);
        let bundle = walker.walk_from_free_fn(
            "mini::alpha",
            WalkStrategy::ModuleSiblings,
            WalkBudget::default_generous(),
        );
        // Should have root + beta + gamma, alpha itself skipped from sibling output.
        let has_beta = bundle
            .nodes
            .iter()
            .any(|n| n.meta().label == "beta");
        let has_gamma = bundle
            .nodes
            .iter()
            .any(|n| n.meta().label == "gamma");
        assert!(has_beta, "beta sibling missing: {:#?}", bundle.nodes);
        assert!(has_gamma, "gamma sibling missing: {:#?}", bundle.nodes);
        // Sibling edges carry the right provenance
        assert!(bundle.edges.iter().all(|e| matches!(
            &e.provenance,
            EdgeProvenance::Sibling { .. }
        )));
    }

    // ── Budget enforcement ─────────────────────────────────────────────

    #[test]
    fn budget_truncation_flags_bundle() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
pub fn alpha() {}
pub fn beta() {}
pub fn gamma() {}
pub fn delta() {}
pub fn epsilon() {}
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let walker = Walker::new(&idx);
        let budget = WalkBudget {
            max_nodes: 2,
            max_edges: 10,
            timeout: Duration::from_secs(10),
        };
        let bundle = walker.walk_from_free_fn(
            "mini::alpha",
            WalkStrategy::ModuleSiblings,
            budget,
        );
        assert!(bundle.truncated, "expected truncated bundle");
        // Trace should contain a Truncated step
        assert!(bundle
            .walk_trace
            .iter()
            .any(|s| matches!(&s.action, WalkAction::Truncated { .. })));
    }

    // ── Git co-change ──────────────────────────────────────────────────

    #[test]
    fn cochange_without_git_repo_returns_budget_skip() {
        // Build an index over a tempdir that is NOT a git repo.
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("crates/mini/src/lib.rs", "pub fn target() {}\n"),
        ];
        let (_tmp, idx) = build_index(files);
        let walker = Walker::new(&idx);
        let bundle = walker.walk_from_free_fn(
            "mini::target",
            WalkStrategy::GitCochange {
                min_commits_shared: 1,
            },
            WalkBudget::default_generous(),
        );
        // No edges, but a trace step explaining why
        assert!(bundle.edges.is_empty());
        assert!(bundle
            .walk_trace
            .iter()
            .any(|s| matches!(&s.action, WalkAction::BudgetSkip { reason } if reason.contains("git2"))));
    }

    #[test]
    fn cochange_with_real_git_repo_detects_cochanges() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // Initialize a real git repo and stage fixture files.
        let repo = git2::Repository::init(root).expect("git init");
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("crates/mini/src/lib.rs", "pub fn target() {}\n"),
            ("crates/mini/src/other.rs", "pub fn other() {}\n"),
        ];
        write_fixture(root, files);

        // Programmatic commit 1: initial add of everything.
        let mut sig = git2::Signature::now("test", "test@test").expect("sig");
        let mut idx_git = repo.index().expect("index");
        idx_git.add_all(["."].iter(), git2::IndexAddOption::DEFAULT, None).expect("add_all");
        idx_git.write().expect("write index");
        let tree_id = idx_git.write_tree().expect("write_tree");
        let tree = repo.find_tree(tree_id).expect("find_tree");
        let commit_1 = repo
            .commit(Some("HEAD"), &sig, &sig, "initial", &tree, &[])
            .expect("commit");

        // Commit 2: modify both files together.
        fs::write(root.join("crates/mini/src/lib.rs"), "pub fn target() { let _ = 1; }\n")
            .unwrap();
        fs::write(root.join("crates/mini/src/other.rs"), "pub fn other() { let _ = 2; }\n").unwrap();
        sig = git2::Signature::now("test", "test@test").expect("sig");
        let mut idx_git = repo.index().expect("index");
        idx_git.add_all(["."].iter(), git2::IndexAddOption::DEFAULT, None).expect("add_all");
        idx_git.write().expect("write index");
        let tree_id = idx_git.write_tree().expect("write_tree");
        let tree = repo.find_tree(tree_id).expect("find_tree");
        let parent = repo.find_commit(commit_1).expect("parent");
        repo.commit(Some("HEAD"), &sig, &sig, "co-change", &tree, &[&parent])
            .expect("commit");

        let idx = ProjectIndex::build(root).expect("build");
        let walker = Walker::new(&idx);
        let bundle = walker.walk_from_free_fn(
            "mini::target",
            WalkStrategy::GitCochange { min_commits_shared: 1 },
            WalkBudget::default_generous(),
        );

        // At least one co-change edge
        let has_cochange_edge = bundle.edges.iter().any(|e| {
            matches!(
                e.provenance,
                EdgeProvenance::GitCochange { commits_shared: n, .. } if n >= 1
            )
        });
        assert!(
            has_cochange_edge,
            "expected co-change edge to other.rs, edges: {:#?}",
            bundle.edges
        );
    }
}
