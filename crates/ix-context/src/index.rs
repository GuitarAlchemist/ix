//! Project-wide symbol table built by a workspace-scale tree-sitter walk.
//!
//! This is the primary source of truth that [`crate::resolve`] consults
//! when answering *"what function does this call site actually refer to?"*.
//!
//! # Scope
//!
//! MVP extracts:
//! - Free functions — `fn foo() { ... }` at module scope
//! - Inherent methods — `impl Type { fn foo() { ... } }`
//! - Trait-impl methods — `impl Trait for Type { fn foo() { ... } }`
//! - File-level `use` aliases — `use crate::X::Y as Z;`
//!
//! MVP deliberately does NOT handle:
//! - `#[cfg(...)]` gates — assume all cfg are enabled (trust tree-sitter)
//! - Inline `mod { ... }` blocks — only file-based modules
//! - Re-exports (`pub use`) — deferred to v2
//! - Macro-generated items — tree-sitter won't see them anyway
//! - Generic type parameters in impl blocks — stripped to bare type names
//!
//! Ambiguity and incompleteness are preserved as signal rather than hidden.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use ix_code::semantic::{extract_call_graph, CallGraph};
use serde::{Deserialize, Serialize};
use streaming_iterator::StreamingIterator;
use tree_sitter::{Node, Parser, Query, QueryCursor};

use crate::model;

// ---------------------------------------------------------------------------
// Symbol key + def site
// ---------------------------------------------------------------------------

/// The key under which a definition is filed in [`ProjectIndex::symbols`].
///
/// Different symbol flavors use different keys so the resolver can answer
/// questions like *"is there an inherent method `foo` on `Type`?"* without
/// scanning the whole table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SymbolKey {
    /// A free function at `crate::module::name`, stored as a full path
    /// joined with `::`.
    FreeFn(String),
    /// A method on an inherent impl block: `impl Type { fn method() }`.
    /// `ty` is the bare type name with generic parameters stripped
    /// (e.g., `Vec` from `impl<T> Vec<T>`).
    InherentMethod { ty: String, method: String },
    /// A method on a trait impl block: `impl Trait for Type { fn method() }`.
    /// Both `trait_name` and `ty` are bare type/trait names without
    /// generic parameters.
    TraitMethod {
        trait_name: String,
        ty: String,
        method: String,
    },
}

impl SymbolKey {
    /// The final method/function name regardless of flavor.
    pub fn name(&self) -> &str {
        match self {
            SymbolKey::FreeFn(path) => path.rsplit("::").next().unwrap_or(path),
            SymbolKey::InherentMethod { method, .. } => method,
            SymbolKey::TraitMethod { method, .. } => method,
        }
    }
}

/// Where a symbol is defined in the source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DefSite {
    /// Workspace-relative path to the file.
    pub file: String,
    /// Name of the crate this file belongs to (from its `Cargo.toml`).
    pub crate_name: String,
    /// Module path within the crate, excluding the crate root itself
    /// (e.g., `["eigen"]` for `crates/ix-math/src/eigen.rs`).
    pub module_path: Vec<String>,
    /// 1-based line range of the definition.
    pub line_start: usize,
    pub line_end: usize,
}

impl DefSite {
    /// Build a stable function ID from this site using [`model::fn_id`].
    pub fn to_fn_id(&self, name: &str) -> String {
        let module_refs: Vec<&str> = self.module_path.iter().map(String::as_str).collect();
        model::fn_id(
            &self.crate_name,
            &module_refs,
            name,
            &self.file,
            self.line_start,
            self.line_end,
        )
    }
}

// ---------------------------------------------------------------------------
// Per-file index + project index
// ---------------------------------------------------------------------------

/// What we know about a single file after pass 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FileIndex {
    /// Workspace-relative path.
    pub path: String,
    /// Crate this file belongs to.
    pub crate_name: String,
    /// Module path within the crate, excluding the crate root itself.
    pub module_path: Vec<String>,
    /// Free-function definitions found in this file, keyed by name.
    pub free_fns: HashMap<String, DefSite>,
    /// Inherent methods, keyed by `(type_name, method_name)`.
    pub inherent_methods: HashMap<(String, String), DefSite>,
    /// Trait-impl methods, keyed by `(trait_name, type_name, method_name)`.
    pub trait_methods: HashMap<(String, String, String), DefSite>,
    /// `use` aliases at file scope: short name → full path segments.
    /// `use crate::eigen::jacobi as j;` yields `j -> ["crate", "eigen", "jacobi"]`.
    /// Plain imports `use crate::eigen::jacobi;` yield the final segment
    /// `jacobi -> ["crate", "eigen", "jacobi"]`.
    pub use_aliases: HashMap<String, Vec<String>>,
    /// Per-file call graph extracted via [`ix_code::semantic::extract_call_graph`].
    /// Used by the walker for caller/callee traversal — forward edges only.
    /// Reverse edges are computed lazily at walk time by scanning all files.
    pub call_graph: CallGraph,
}

/// Workspace-wide symbol table — the primary artifact of pass 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectIndex {
    /// Absolute path to the workspace root (where the top-level
    /// `Cargo.toml` lives).
    pub workspace_root: PathBuf,
    /// Per-file state, keyed by workspace-relative path.
    pub files: HashMap<String, FileIndex>,
    /// Flat symbol lookup, keyed by [`SymbolKey`].
    pub symbols: HashMap<SymbolKey, DefSite>,
}

/// Errors raised while building a [`ProjectIndex`].
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("failed to set tree-sitter language")]
    TreeSitterLanguage,
    #[error("failed to parse {path}")]
    ParseFailed { path: PathBuf },
    #[error("failed to build tree-sitter query: {0}")]
    QueryBuild(String),
    #[error("could not determine crate name for {path} (no Cargo.toml found in any parent)")]
    CrateNotFound { path: PathBuf },
}

impl ProjectIndex {
    /// Build an index by walking all `*.rs` files under `workspace_root`
    /// (excluding `target/`, `node_modules/`, `.git/`, and hidden
    /// directories).
    pub fn build(workspace_root: impl AsRef<Path>) -> Result<Self, IndexError> {
        let workspace_root = workspace_root.as_ref().to_path_buf();
        let mut files: HashMap<String, FileIndex> = HashMap::new();
        let mut symbols: HashMap<SymbolKey, DefSite> = HashMap::new();

        let rs_files = collect_rs_files(&workspace_root)?;
        for abs_path in rs_files {
            let rel_path = abs_path
                .strip_prefix(&workspace_root)
                .unwrap_or(&abs_path)
                .to_string_lossy()
                .replace('\\', "/");

            let Some(crate_info) = find_crate_for(&abs_path, &workspace_root)? else {
                // File outside any crate — skip rather than error.
                continue;
            };

            let source = match fs::read_to_string(&abs_path) {
                Ok(s) => s,
                Err(source) => {
                    return Err(IndexError::Io {
                        path: abs_path.clone(),
                        source,
                    });
                }
            };

            let module_path =
                infer_module_path(&abs_path, &crate_info.src_dir, &crate_info.crate_name);

            let file_index = parse_file(
                &rel_path,
                &crate_info.crate_name,
                &module_path,
                &source,
            )?;

            // Promote file-level definitions into the flat symbol table.
            for (name, def) in &file_index.free_fns {
                let qualified = if module_path.is_empty() {
                    format!("{}::{}", crate_info.crate_name, name)
                } else {
                    format!(
                        "{}::{}::{}",
                        crate_info.crate_name,
                        module_path.join("::"),
                        name
                    )
                };
                symbols.insert(SymbolKey::FreeFn(qualified), def.clone());
            }
            for ((ty, method), def) in &file_index.inherent_methods {
                symbols.insert(
                    SymbolKey::InherentMethod {
                        ty: ty.clone(),
                        method: method.clone(),
                    },
                    def.clone(),
                );
            }
            for ((trait_name, ty, method), def) in &file_index.trait_methods {
                symbols.insert(
                    SymbolKey::TraitMethod {
                        trait_name: trait_name.clone(),
                        ty: ty.clone(),
                        method: method.clone(),
                    },
                    def.clone(),
                );
            }

            files.insert(rel_path, file_index);
        }

        Ok(ProjectIndex {
            workspace_root,
            files,
            symbols,
        })
    }

    /// Look up a free function by its fully-qualified path, e.g.
    /// `"ix_math::eigen::jacobi"`.
    pub fn find_free_fn(&self, qualified: &str) -> Option<&DefSite> {
        self.symbols.get(&SymbolKey::FreeFn(qualified.to_string()))
    }

    /// Look up an inherent method by `(type_name, method)`.
    pub fn find_inherent_method(&self, ty: &str, method: &str) -> Option<&DefSite> {
        self.symbols.get(&SymbolKey::InherentMethod {
            ty: ty.to_string(),
            method: method.to_string(),
        })
    }

    /// Return all trait-impl methods matching `(bare_method_name)` across
    /// the workspace — used by the resolver when a method call can't be
    /// narrowed to a single type and needs to surface ambiguity.
    pub fn find_trait_methods_by_name(&self, method: &str) -> Vec<(&SymbolKey, &DefSite)> {
        self.symbols
            .iter()
            .filter(|(k, _)| match k {
                SymbolKey::TraitMethod { method: m, .. } => m == method,
                _ => false,
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// File walk + crate inference
// ---------------------------------------------------------------------------

struct CrateInfo {
    crate_name: String,
    src_dir: PathBuf,
}

/// Walk a single directory tree, returning all `*.rs` files (skipping
/// `target/`, hidden dirs, and the submodule `governance/`).
fn collect_rs_files(root: &Path) -> Result<Vec<PathBuf>, IndexError> {
    let mut out = Vec::new();
    walk_recursive(root, &mut out)?;
    Ok(out)
}

fn walk_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), IndexError> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // unreadable dir — skip rather than error
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            walk_recursive(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            // Skip build.rs when it sits directly next to a Cargo.toml —
            // it's a Cargo build script, not a source module. Indexing
            // it as a module `build` would produce bogus symbol table
            // entries. A build.rs nested deeper (e.g., inside src/) is
            // unusual but still indexed.
            if name == "build.rs" {
                let parent_is_crate_root = path
                    .parent()
                    .map(|p| p.join("Cargo.toml").exists())
                    .unwrap_or(false);
                if parent_is_crate_root {
                    continue;
                }
            }
            out.push(path);
        }
    }
    Ok(())
}

/// Find the crate that owns `file_path` by walking up to the nearest
/// `Cargo.toml` containing a `[package] name = "..."` entry.
fn find_crate_for(file_path: &Path, workspace_root: &Path) -> Result<Option<CrateInfo>, IndexError> {
    let mut cursor = file_path.parent();
    while let Some(dir) = cursor {
        if !dir.starts_with(workspace_root) && dir != workspace_root {
            return Ok(None);
        }
        let cargo = dir.join("Cargo.toml");
        if cargo.exists() {
            if let Ok(contents) = fs::read_to_string(&cargo) {
                if let Some(name) = extract_package_name(&contents) {
                    return Ok(Some(CrateInfo {
                        crate_name: name,
                        src_dir: dir.join("src"),
                    }));
                }
            }
        }
        if dir == workspace_root {
            return Ok(None);
        }
        cursor = dir.parent();
    }
    Ok(None)
}

/// Extract `name = "..."` from the `[package]` section of a `Cargo.toml`
/// without pulling in `toml` as a dependency. Returns `None` for workspace
/// Cargo.toml files that only carry `[workspace]`.
fn extract_package_name(contents: &str) -> Option<String> {
    let mut in_package = false;
    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.starts_with('[') {
            in_package = line.starts_with("[package");
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(rest) = line.strip_prefix("name") {
            let rest = rest.trim_start().trim_start_matches('=').trim();
            let rest = rest.trim_matches('"').trim();
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    None
}

/// Infer the module path within a crate from a file path.
///
/// - `src/lib.rs` or `src/main.rs` → `[]` (crate root)
/// - `src/eigen.rs` → `["eigen"]`
/// - `src/eigen/mod.rs` → `["eigen"]`
/// - `src/eigen/jacobi.rs` → `["eigen", "jacobi"]`
///
/// The `crate_name` is used only to detect when the file is outside the
/// crate's `src/` tree (tests, benches, examples) — those are tagged with
/// a single segment matching the subdirectory name.
fn infer_module_path(file: &Path, src_dir: &Path, _crate_name: &str) -> Vec<String> {
    let Ok(rel) = file.strip_prefix(src_dir) else {
        // Not under src/ — likely tests/, benches/, examples/. Use the
        // first path segment as a coarse module name.
        let crate_dir = src_dir.parent().unwrap_or(src_dir);
        let Ok(rel) = file.strip_prefix(crate_dir) else {
            return Vec::new();
        };
        let components: Vec<String> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
            .collect();
        // Drop the final `.rs` suffix from the filename component.
        return components
            .iter()
            .enumerate()
            .map(|(i, seg)| {
                if i + 1 == components.len() {
                    seg.strip_suffix(".rs").unwrap_or(seg).to_string()
                } else {
                    seg.clone()
                }
            })
            .collect();
    };

    let components: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    if components.is_empty() {
        return Vec::new();
    }

    let last = components.last().copied().unwrap_or("");
    if matches!(last, "lib.rs" | "main.rs") {
        return Vec::new();
    }

    let mut out: Vec<String> = Vec::with_capacity(components.len());
    for (i, seg) in components.iter().enumerate() {
        let is_last = i + 1 == components.len();
        if is_last {
            if *seg == "mod.rs" {
                // Drop mod.rs — the parent dir already contributed.
                break;
            }
            out.push(seg.strip_suffix(".rs").unwrap_or(seg).to_string());
        } else {
            out.push((*seg).to_string());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// tree-sitter parsing
// ---------------------------------------------------------------------------

/// Parse a single file and extract its definitions, inherent methods, trait
/// methods, and `use` aliases. Pure function — no side effects, no I/O.
fn parse_file(
    rel_path: &str,
    crate_name: &str,
    module_path: &[String],
    source: &str,
) -> Result<FileIndex, IndexError> {
    let mut parser = Parser::new();
    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    parser
        .set_language(&language)
        .map_err(|_| IndexError::TreeSitterLanguage)?;
    let Some(tree) = parser.parse(source, None) else {
        return Err(IndexError::ParseFailed {
            path: PathBuf::from(rel_path),
        });
    };

    let bytes = source.as_bytes();
    let mut file_index = FileIndex {
        path: rel_path.to_string(),
        crate_name: crate_name.to_string(),
        module_path: module_path.to_vec(),
        free_fns: HashMap::new(),
        inherent_methods: HashMap::new(),
        trait_methods: HashMap::new(),
        use_aliases: HashMap::new(),
        call_graph: extract_call_graph(source).unwrap_or_default(),
    };

    // Query 1: free function definitions at module scope.
    // `function_item` nodes whose parent is NOT an impl body.
    extract_free_functions(tree.root_node(), bytes, rel_path, crate_name, module_path, &mut file_index)?;

    // Query 2: impl blocks (inherent + trait).
    extract_impl_methods(tree.root_node(), bytes, rel_path, crate_name, module_path, &mut file_index)?;

    // Query 3: use aliases.
    extract_use_aliases(&tree, bytes, &mut file_index)?;

    Ok(file_index)
}

fn extract_free_functions(
    root: Node,
    bytes: &[u8],
    rel_path: &str,
    crate_name: &str,
    module_path: &[String],
    file_index: &mut FileIndex,
) -> Result<(), IndexError> {
    const QUERY_SRC: &str = r#"(function_item name: (identifier) @fn.name) @fn.item"#;
    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    let query = Query::new(&language, QUERY_SRC).map_err(|e| IndexError::QueryBuild(e.to_string()))?;
    let name_idx = query.capture_index_for_name("fn.name");
    let item_idx = query.capture_index_for_name("fn.item");

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, root, bytes);
    while let Some(m) = matches.next() {
        let mut name_node: Option<Node> = None;
        let mut item_node: Option<Node> = None;
        for cap in m.captures {
            if Some(cap.index) == name_idx {
                name_node = Some(cap.node);
            } else if Some(cap.index) == item_idx {
                item_node = Some(cap.node);
            }
        }
        let (Some(name_node), Some(item_node)) = (name_node, item_node) else {
            continue;
        };

        // Skip if the function is inside an impl block — extract_impl_methods
        // handles that case.
        if is_inside_impl(item_node) {
            continue;
        }

        let Ok(name) = name_node.utf8_text(bytes) else {
            continue;
        };
        let start = item_node.start_position().row + 1;
        let end = item_node.end_position().row + 1;

        let def = DefSite {
            file: rel_path.to_string(),
            crate_name: crate_name.to_string(),
            module_path: module_path.to_vec(),
            line_start: start,
            line_end: end,
        };
        file_index.free_fns.insert(name.to_string(), def);
    }
    Ok(())
}

fn extract_impl_methods(
    root: Node,
    bytes: &[u8],
    rel_path: &str,
    crate_name: &str,
    module_path: &[String],
    file_index: &mut FileIndex,
) -> Result<(), IndexError> {
    // Walk all impl_item nodes. For each, inspect its trait/type fields
    // and extract the function_item children inside the body.
    walk_impls(root, bytes, rel_path, crate_name, module_path, file_index);
    Ok(())
}

fn walk_impls(
    node: Node,
    bytes: &[u8],
    rel_path: &str,
    crate_name: &str,
    module_path: &[String],
    file_index: &mut FileIndex,
) {
    if node.kind() == "impl_item" {
        // `impl Trait for Type { ... }` has both `trait` and `type` fields.
        // `impl Type { ... }` has only `type`.
        let trait_name = node
            .child_by_field_name("trait")
            .and_then(|n| bare_type_name(n, bytes));
        let ty = node
            .child_by_field_name("type")
            .and_then(|n| bare_type_name(n, bytes));

        if let Some(ty) = ty {
            // Iterate fn items in the impl body.
            if let Some(body) = node.child_by_field_name("body") {
                let mut cursor = body.walk();
                for child in body.children(&mut cursor) {
                    if child.kind() == "function_item" {
                        let Some(name_node) = child.child_by_field_name("name") else {
                            continue;
                        };
                        let Ok(method) = name_node.utf8_text(bytes) else {
                            continue;
                        };
                        let start = child.start_position().row + 1;
                        let end = child.end_position().row + 1;
                        let def = DefSite {
                            file: rel_path.to_string(),
                            crate_name: crate_name.to_string(),
                            module_path: module_path.to_vec(),
                            line_start: start,
                            line_end: end,
                        };
                        if let Some(tname) = trait_name.as_ref() {
                            file_index
                                .trait_methods
                                .insert((tname.clone(), ty.clone(), method.to_string()), def);
                        } else {
                            file_index
                                .inherent_methods
                                .insert((ty.clone(), method.to_string()), def);
                        }
                    }
                }
            }
        }
        // Don't descend further into this impl — the body has been handled.
        return;
    }

    // Recurse into children.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_impls(child, bytes, rel_path, crate_name, module_path, file_index);
    }
}

/// Reduce a `type_identifier`, `generic_type`, or `scoped_type_identifier`
/// node to its bare type name (generic parameters stripped).
fn bare_type_name(node: Node, bytes: &[u8]) -> Option<String> {
    match node.kind() {
        "type_identifier" => node.utf8_text(bytes).ok().map(str::to_string),
        "generic_type" => {
            let inner = node.child_by_field_name("type")?;
            bare_type_name(inner, bytes)
        }
        "scoped_type_identifier" => {
            let name = node.child_by_field_name("name")?;
            name.utf8_text(bytes).ok().map(str::to_string)
        }
        _ => node.utf8_text(bytes).ok().map(str::to_string),
    }
}

/// `true` iff `node` is transitively inside an `impl_item` (as opposed to
/// at module scope). Used to exclude impl-block methods from the free-fn
/// query.
fn is_inside_impl(node: Node) -> bool {
    let mut cursor = node.parent();
    while let Some(parent) = cursor {
        if parent.kind() == "impl_item" {
            return true;
        }
        cursor = parent.parent();
    }
    false
}

fn extract_use_aliases(
    tree: &tree_sitter::Tree,
    bytes: &[u8],
    file_index: &mut FileIndex,
) -> Result<(), IndexError> {
    // Walk `use_declaration` nodes. For each, handle the three common
    // shapes:
    //   1. `use foo::bar::baz;`              → `baz` -> ["foo","bar","baz"]
    //   2. `use foo::bar::baz as qux;`       → `qux` -> ["foo","bar","baz"]
    //   3. `use foo::bar::{baz, quux};`      → each leaf becomes its own alias
    walk_use_decls(tree.root_node(), bytes, file_index);
    Ok(())
}

fn walk_use_decls(node: Node, bytes: &[u8], file_index: &mut FileIndex) {
    if node.kind() == "use_declaration" {
        if let Some(arg) = node.child_by_field_name("argument") {
            collect_use_paths(arg, &[], bytes, file_index);
        }
        return;
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_use_decls(child, bytes, file_index);
    }
}

fn collect_use_paths(node: Node, prefix: &[String], bytes: &[u8], file_index: &mut FileIndex) {
    match node.kind() {
        "scoped_identifier" => {
            // foo::bar — split into prefix + final name.
            let mut segments: Vec<String> = prefix.to_vec();
            walk_scoped_for_use(node, bytes, &mut segments);
            if let Some(name) = segments.last().cloned() {
                file_index.use_aliases.insert(name, segments);
            }
        }
        "identifier" | "self" | "super" | "crate" => {
            if let Ok(text) = node.utf8_text(bytes) {
                let mut segments: Vec<String> = prefix.to_vec();
                segments.push(text.to_string());
                file_index
                    .use_aliases
                    .insert(text.to_string(), segments);
            }
        }
        "use_as_clause" => {
            // path as alias
            let path_node = node.child_by_field_name("path");
            let alias_node = node.child_by_field_name("alias");
            if let (Some(path_node), Some(alias_node)) = (path_node, alias_node) {
                let mut segments: Vec<String> = prefix.to_vec();
                walk_scoped_for_use(path_node, bytes, &mut segments);
                if let Ok(alias) = alias_node.utf8_text(bytes) {
                    file_index.use_aliases.insert(alias.to_string(), segments);
                }
            }
        }
        "scoped_use_list" => {
            // foo::bar::{a, b, c}
            let path_node = node.child_by_field_name("path");
            let list_node = node.child_by_field_name("list");
            let mut new_prefix: Vec<String> = prefix.to_vec();
            if let Some(path) = path_node {
                walk_scoped_for_use(path, bytes, &mut new_prefix);
            }
            if let Some(list) = list_node {
                let mut cursor = list.walk();
                for child in list.children(&mut cursor) {
                    if child.kind() == "," || child.kind() == "{" || child.kind() == "}" {
                        continue;
                    }
                    collect_use_paths(child, &new_prefix, bytes, file_index);
                }
            }
        }
        _ => {
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                collect_use_paths(child, prefix, bytes, file_index);
            }
        }
    }
}

fn walk_scoped_for_use(node: Node, bytes: &[u8], out: &mut Vec<String>) {
    match node.kind() {
        "scoped_identifier" => {
            if let Some(path) = node.child_by_field_name("path") {
                walk_scoped_for_use(path, bytes, out);
            }
            if let Some(name) = node.child_by_field_name("name") {
                if let Ok(t) = name.utf8_text(bytes) {
                    out.push(t.to_string());
                }
            }
        }
        "identifier" | "self" | "super" | "crate" => {
            if let Ok(t) = node.utf8_text(bytes) {
                out.push(t.to_string());
            }
        }
        _ => {
            if let Ok(t) = node.utf8_text(bytes) {
                out.push(t.to_string());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── extract_package_name ────────────────────────────────────────────

    #[test]
    fn extract_package_name_from_simple_cargo_toml() {
        let toml = r#"
[package]
name = "ix-math"
version = "0.1.0"
"#;
        assert_eq!(extract_package_name(toml), Some("ix-math".to_string()));
    }

    #[test]
    fn extract_package_name_returns_none_for_workspace() {
        let toml = r#"
[workspace]
members = ["crates/ix-math"]
"#;
        assert_eq!(extract_package_name(toml), None);
    }

    #[test]
    fn extract_package_name_handles_inline_package() {
        let toml = r#"[package]
name="foo"
"#;
        assert_eq!(extract_package_name(toml), Some("foo".to_string()));
    }

    // ── infer_module_path ────────────────────────────────────────────────

    #[test]
    fn module_path_for_lib_rs_is_empty() {
        let src = PathBuf::from("/w/crates/foo/src");
        let file = PathBuf::from("/w/crates/foo/src/lib.rs");
        assert_eq!(infer_module_path(&file, &src, "foo"), Vec::<String>::new());
    }

    #[test]
    fn module_path_for_main_rs_is_empty() {
        let src = PathBuf::from("/w/crates/foo/src");
        let file = PathBuf::from("/w/crates/foo/src/main.rs");
        assert_eq!(infer_module_path(&file, &src, "foo"), Vec::<String>::new());
    }

    #[test]
    fn module_path_for_flat_module_file() {
        let src = PathBuf::from("/w/crates/foo/src");
        let file = PathBuf::from("/w/crates/foo/src/eigen.rs");
        assert_eq!(infer_module_path(&file, &src, "foo"), vec!["eigen".to_string()]);
    }

    #[test]
    fn module_path_for_mod_rs_uses_parent_dir() {
        let src = PathBuf::from("/w/crates/foo/src");
        let file = PathBuf::from("/w/crates/foo/src/eigen/mod.rs");
        assert_eq!(infer_module_path(&file, &src, "foo"), vec!["eigen".to_string()]);
    }

    #[test]
    fn module_path_for_nested_module_file() {
        let src = PathBuf::from("/w/crates/foo/src");
        let file = PathBuf::from("/w/crates/foo/src/eigen/jacobi.rs");
        assert_eq!(
            infer_module_path(&file, &src, "foo"),
            vec!["eigen".to_string(), "jacobi".to_string()]
        );
    }

    // ── parse_file: free functions ──────────────────────────────────────

    #[test]
    fn parse_file_extracts_free_function() {
        let src = r#"
fn add(a: i32, b: i32) -> i32 { a + b }
"#;
        let fi = parse_file("a.rs", "demo", &["m".to_string()], src).expect("parse");
        assert!(fi.free_fns.contains_key("add"));
        let def = &fi.free_fns["add"];
        assert_eq!(def.crate_name, "demo");
        assert_eq!(def.module_path, vec!["m".to_string()]);
        assert!(def.line_start >= 1);
    }

    #[test]
    fn parse_file_extracts_multiple_free_functions() {
        let src = r#"
fn a() {}
fn b() {}
fn c() {}
"#;
        let fi = parse_file("x.rs", "demo", &[], src).expect("parse");
        assert_eq!(fi.free_fns.len(), 3);
    }

    #[test]
    fn parse_file_does_not_conflate_impl_methods_with_free_fns() {
        let src = r#"
struct S;
impl S {
    fn foo(&self) {}
}
fn bar() {}
"#;
        let fi = parse_file("y.rs", "demo", &[], src).expect("parse");
        // `foo` is a method, NOT a free fn.
        assert!(!fi.free_fns.contains_key("foo"));
        assert!(fi.free_fns.contains_key("bar"));
        // `foo` lives as an inherent method on `S`.
        assert!(fi
            .inherent_methods
            .contains_key(&("S".to_string(), "foo".to_string())));
    }

    // ── parse_file: impl methods ────────────────────────────────────────

    #[test]
    fn parse_file_extracts_inherent_method() {
        let src = r#"
struct Widget;
impl Widget {
    fn new() -> Self { Widget }
    fn name(&self) -> &str { "w" }
}
"#;
        let fi = parse_file("w.rs", "demo", &[], src).expect("parse");
        assert!(fi
            .inherent_methods
            .contains_key(&("Widget".to_string(), "new".to_string())));
        assert!(fi
            .inherent_methods
            .contains_key(&("Widget".to_string(), "name".to_string())));
    }

    #[test]
    fn parse_file_extracts_trait_impl_method() {
        let src = r#"
trait Greet { fn greet(&self) -> String; }
struct Hello;
impl Greet for Hello {
    fn greet(&self) -> String { "hi".to_string() }
}
"#;
        let fi = parse_file("g.rs", "demo", &[], src).expect("parse");
        assert!(fi.trait_methods.contains_key(&(
            "Greet".to_string(),
            "Hello".to_string(),
            "greet".to_string()
        )));
        // It should NOT also show up as inherent.
        assert!(!fi
            .inherent_methods
            .contains_key(&("Hello".to_string(), "greet".to_string())));
    }

    #[test]
    fn parse_file_strips_generic_parameters_from_impl_type() {
        let src = r#"
struct Container<T>(T);
impl<T> Container<T> {
    fn push(&mut self, _x: T) {}
}
"#;
        let fi = parse_file("c.rs", "demo", &[], src).expect("parse");
        // Generic params stripped — bare type name "Container".
        assert!(fi
            .inherent_methods
            .contains_key(&("Container".to_string(), "push".to_string())));
    }

    // ── parse_file: use aliases ─────────────────────────────────────────

    #[test]
    fn parse_file_captures_plain_use() {
        let src = r#"
use crate::eigen::jacobi;
fn main() { let _ = jacobi; }
"#;
        let fi = parse_file("u.rs", "demo", &[], src).expect("parse");
        let segs = fi.use_aliases.get("jacobi").expect("jacobi alias missing");
        assert_eq!(
            segs,
            &vec!["crate".to_string(), "eigen".to_string(), "jacobi".to_string()]
        );
    }

    #[test]
    fn parse_file_captures_use_as_alias() {
        let src = r#"
use crate::eigen::jacobi as j;
"#;
        let fi = parse_file("u.rs", "demo", &[], src).expect("parse");
        let segs = fi.use_aliases.get("j").expect("j alias missing");
        assert_eq!(
            segs,
            &vec!["crate".to_string(), "eigen".to_string(), "jacobi".to_string()]
        );
    }

    #[test]
    fn parse_file_captures_use_list() {
        let src = r#"
use crate::eigen::{jacobi, lanczos};
"#;
        let fi = parse_file("u.rs", "demo", &[], src).expect("parse");
        assert!(fi.use_aliases.contains_key("jacobi"));
        assert!(fi.use_aliases.contains_key("lanczos"));
    }

    // ── SymbolKey::name() ───────────────────────────────────────────────

    #[test]
    fn symbol_key_name_for_free_fn_is_final_segment() {
        let k = SymbolKey::FreeFn("ix_math::eigen::jacobi".to_string());
        assert_eq!(k.name(), "jacobi");
    }

    #[test]
    fn symbol_key_name_for_inherent_method() {
        let k = SymbolKey::InherentMethod {
            ty: "Widget".to_string(),
            method: "new".to_string(),
        };
        assert_eq!(k.name(), "new");
    }

    #[test]
    fn symbol_key_name_for_trait_method() {
        let k = SymbolKey::TraitMethod {
            trait_name: "Greet".to_string(),
            ty: "Hello".to_string(),
            method: "greet".to_string(),
        };
        assert_eq!(k.name(), "greet");
    }

    // ── DefSite::to_fn_id ───────────────────────────────────────────────

    #[test]
    fn def_site_produces_stable_fn_id() {
        let def = DefSite {
            file: "crates/ix-math/src/eigen.rs".to_string(),
            crate_name: "ix_math".to_string(),
            module_path: vec!["eigen".to_string()],
            line_start: 42,
            line_end: 103,
        };
        assert_eq!(
            def.to_fn_id("jacobi"),
            "fn:ix_math::eigen::jacobi@crates/ix-math/src/eigen.rs#L42-L103"
        );
    }

    // ── Full ProjectIndex build over a fixture tree ─────────────────────

    fn write_fixture(dir: &std::path::Path, files: &[(&str, &str)]) {
        for (rel, contents) in files {
            let path = dir.join(rel);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).expect("create_dir_all");
            }
            fs::write(&path, contents).expect("write fixture");
        }
    }

    #[test]
    fn project_index_builds_over_mini_crate() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();

        // Minimal workspace: one crate with lib.rs and a submodule.
        write_fixture(
            root,
            &[
                (
                    "Cargo.toml",
                    r#"[workspace]
members = ["crates/mini"]
"#,
                ),
                (
                    "crates/mini/Cargo.toml",
                    r#"[package]
name = "mini"
version = "0.1.0"
edition = "2021"
"#,
                ),
                (
                    "crates/mini/src/lib.rs",
                    r#"pub mod eigen;
pub fn top_level() {}
"#,
                ),
                (
                    "crates/mini/src/eigen.rs",
                    r#"pub fn jacobi() {}
pub fn lanczos() {}

pub struct Solver;
impl Solver {
    pub fn new() -> Self { Solver }
}
"#,
                ),
            ],
        );

        let idx = ProjectIndex::build(root).expect("build");

        // Both free functions should be indexed with fully-qualified paths.
        assert!(
            idx.find_free_fn("mini::top_level").is_some(),
            "free fn mini::top_level missing. symbols: {:?}",
            idx.symbols.keys().collect::<Vec<_>>()
        );
        assert!(
            idx.find_free_fn("mini::eigen::jacobi").is_some(),
            "free fn mini::eigen::jacobi missing"
        );
        assert!(
            idx.find_free_fn("mini::eigen::lanczos").is_some(),
            "free fn mini::eigen::lanczos missing"
        );

        // Inherent method on Solver should also be present.
        assert!(
            idx.find_inherent_method("Solver", "new").is_some(),
            "inherent method Solver::new missing"
        );

        // The symbol table should NOT contain Solver as a free function.
        assert!(idx.find_free_fn("mini::eigen::Solver").is_none());

        // Each file should know its own crate and module path.
        let eigen = idx
            .files
            .values()
            .find(|f| f.path.ends_with("eigen.rs"))
            .expect("eigen file missing");
        assert_eq!(eigen.crate_name, "mini");
        assert_eq!(eigen.module_path, vec!["eigen".to_string()]);
    }

    #[test]
    fn project_index_trait_methods_findable_by_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        write_fixture(
            root,
            &[
                (
                    "Cargo.toml",
                    "[workspace]\nmembers = [\"crates/mini\"]\n",
                ),
                (
                    "crates/mini/Cargo.toml",
                    "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
                ),
                (
                    "crates/mini/src/lib.rs",
                    r#"
pub trait Greet { fn greet(&self) -> String; }
pub struct Hello;
pub struct World;
impl Greet for Hello { fn greet(&self) -> String { "hello".into() } }
impl Greet for World { fn greet(&self) -> String { "world".into() } }
"#,
                ),
            ],
        );

        let idx = ProjectIndex::build(root).expect("build");
        let matches = idx.find_trait_methods_by_name("greet");
        assert_eq!(
            matches.len(),
            2,
            "expected 2 greet impls, got {}: {:?}",
            matches.len(),
            matches.iter().map(|(k, _)| k).collect::<Vec<_>>()
        );
        for (key, _) in &matches {
            match key {
                SymbolKey::TraitMethod { trait_name, ty, method } => {
                    assert_eq!(trait_name, "Greet");
                    assert_eq!(method, "greet");
                    assert!(ty == "Hello" || ty == "World");
                }
                other => panic!("expected TraitMethod, got {:?}", other),
            }
        }
    }
}
