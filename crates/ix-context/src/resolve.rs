//! Cross-file call-site resolver.
//!
//! Consumes [`ix_code::semantic::CalleeHint`] records and produces
//! [`crate::model::ResolvedOrAmbiguous`] outcomes by walking the
//! [`crate::index::ProjectIndex`] in a fixed, ordered-heuristic sequence:
//!
//! ```text
//! For a Bare name:
//!   1. Local scope             — free fn in the caller's own file
//!   2. Use aliases              — alias in the caller's file's use_aliases
//!   3. Same module              — free fn in any file sharing the
//!                                 caller's (crate, module_path)
//!   4. Crate root               — free fn at `crate::name`
//!   5. Trait method candidates  — any trait methods with this bare name
//!                                 → Ambiguous if multiple, Resolved if one
//!   6. Inherent method fallback — any inherent method with this bare name
//!   7. Unresolved { hint }
//!
//! For a Scoped path `a::b::c`:
//!   1. Rewrite `crate::…` → `<caller_crate>::…`
//!   2. Lookup as FreeFn(joined)
//!   3. If the last two segments look like `Type::method`, lookup as
//!      InherentMethod { Type, method }
//!   4. Unresolved { hint }
//!
//! For a MethodCall { receiver_hint, method }:
//!   1. If receiver_hint resolves to a known type → InherentMethod
//!   2. Otherwise, collect all trait methods with this name
//!      → Ambiguous(N) for N≥2, Resolved for N=1
//!   3. Otherwise, collect all inherent methods with this name
//!      → Ambiguous / Resolved
//!   4. Unresolved { hint: method }
//! ```
//!
//! **Ambiguity is signal.** When multiple candidates match, the resolver
//! emits [`crate::model::ResolvedOrAmbiguous::Ambiguous`] with the full
//! candidate list; it NEVER arbitrarily picks one. A walker that silently
//! chose would lie to Demerzel about the agent's informational state.

use ix_code::semantic::CalleeHint;

use crate::index::{ProjectIndex, SymbolKey};
use crate::model::ResolvedOrAmbiguous;

/// Resolver over a pre-built [`ProjectIndex`]. Cheap to construct; borrow
/// semantics only.
pub struct CallSiteResolver<'a> {
    index: &'a ProjectIndex,
}

impl<'a> CallSiteResolver<'a> {
    /// Wrap a reference to an existing index.
    pub fn new(index: &'a ProjectIndex) -> Self {
        Self { index }
    }

    /// Resolve a single call site.
    ///
    /// `caller_file` is the workspace-relative path of the file where the
    /// call appears; `caller_crate` is the crate name that file belongs to;
    /// `caller_module` is the module path of that file within the crate.
    pub fn resolve(
        &self,
        hint: &CalleeHint,
        caller_file: &str,
        caller_crate: &str,
        caller_module: &[String],
    ) -> ResolvedOrAmbiguous {
        match hint {
            CalleeHint::Bare { name } => {
                self.resolve_bare(name, caller_file, caller_crate, caller_module)
            }
            CalleeHint::Scoped { segments } => {
                self.resolve_scoped(segments, caller_crate)
            }
            CalleeHint::MethodCall {
                method,
                receiver_hint,
            } => self.resolve_method(method, receiver_hint.as_deref()),
        }
    }

    // ── Bare names ──────────────────────────────────────────────────────

    fn resolve_bare(
        &self,
        name: &str,
        caller_file: &str,
        caller_crate: &str,
        caller_module: &[String],
    ) -> ResolvedOrAmbiguous {
        // 1. Local scope
        if let Some(file_index) = self.index.files.get(caller_file) {
            if let Some(def) = file_index.free_fns.get(name) {
                return ResolvedOrAmbiguous::Resolved {
                    id: def.to_fn_id(name),
                };
            }
            // 2. Use aliases in this file
            if let Some(segments) = file_index.use_aliases.get(name) {
                // Rewrite and fall through to scoped resolution.
                if let ResolvedOrAmbiguous::Resolved { id } =
                    self.resolve_scoped(segments, caller_crate)
                {
                    return ResolvedOrAmbiguous::Resolved { id };
                }
            }
        }

        // 3. Same module (different files in the same crate::module_path).
        // Collect into a Vec then sort — HashMap iteration is not stable
        // across processes, and Ambiguous candidate order must be
        // deterministic for replay identity.
        let mut same_module_hits: Vec<String> = self
            .index
            .files
            .iter()
            .filter(|(_, fi)| fi.crate_name == caller_crate && fi.module_path == caller_module)
            .filter_map(|(_, fi)| fi.free_fns.get(name).map(|def| def.to_fn_id(name)))
            .collect();
        same_module_hits.sort();
        match same_module_hits.len() {
            0 => {}
            1 => {
                return ResolvedOrAmbiguous::Resolved {
                    id: same_module_hits.into_iter().next().unwrap(),
                };
            }
            _ => {
                return ResolvedOrAmbiguous::Ambiguous {
                    candidates: same_module_hits,
                };
            }
        }

        // 4. Crate root — `<crate>::<name>`
        let crate_root_key = format!("{}::{}", caller_crate, name);
        if let Some(def) = self.index.find_free_fn(&crate_root_key) {
            return ResolvedOrAmbiguous::Resolved {
                id: def.to_fn_id(name),
            };
        }

        // 5. Trait methods matching this bare name. Sort for stable
        // Ambiguous candidate order.
        let mut trait_hits: Vec<String> = self
            .index
            .find_trait_methods_by_name(name)
            .into_iter()
            .filter_map(|(key, def)| match key {
                SymbolKey::TraitMethod {
                    trait_name,
                    ty,
                    method,
                } => Some(def.to_fn_id(&format!("{trait_name}::{ty}::{method}"))),
                _ => None,
            })
            .collect();
        trait_hits.sort();

        match trait_hits.len() {
            0 => {}
            1 => {
                return ResolvedOrAmbiguous::Resolved {
                    id: trait_hits.into_iter().next().unwrap(),
                };
            }
            _ => {
                return ResolvedOrAmbiguous::Ambiguous {
                    candidates: trait_hits,
                };
            }
        }

        // 6. Inherent methods matching this bare name. Sorted for stability.
        let mut inherent_hits: Vec<String> = self
            .index
            .symbols
            .iter()
            .filter_map(|(k, def)| match k {
                SymbolKey::InherentMethod { ty, method } if method == name => {
                    Some(def.to_fn_id(&format!("{ty}::{method}")))
                }
                _ => None,
            })
            .collect();
        inherent_hits.sort();

        match inherent_hits.len() {
            0 => {}
            1 => {
                return ResolvedOrAmbiguous::Resolved {
                    id: inherent_hits.into_iter().next().unwrap(),
                };
            }
            _ => {
                return ResolvedOrAmbiguous::Ambiguous {
                    candidates: inherent_hits,
                };
            }
        }

        // 7. Fallback
        ResolvedOrAmbiguous::Unresolved {
            hint: name.to_string(),
        }
    }

    // ── Scoped paths ────────────────────────────────────────────────────

    fn resolve_scoped(&self, segments: &[String], caller_crate: &str) -> ResolvedOrAmbiguous {
        if segments.is_empty() {
            return ResolvedOrAmbiguous::Unresolved {
                hint: String::new(),
            };
        }

        // Rewrite a leading `crate` segment to the caller's crate name, so
        // `crate::eigen::jacobi` becomes `ix_math::eigen::jacobi`.
        let normalized: Vec<String> = if segments[0] == "crate" {
            std::iter::once(caller_crate.to_string())
                .chain(segments.iter().skip(1).cloned())
                .collect()
        } else {
            segments.to_vec()
        };

        let joined = normalized.join("::");

        // Try direct free-function lookup first.
        if let Some(def) = self.index.find_free_fn(&joined) {
            let name = normalized.last().cloned().unwrap_or_default();
            return ResolvedOrAmbiguous::Resolved {
                id: def.to_fn_id(&name),
            };
        }

        // Try `Type::method` interpretation: the second-to-last segment as
        // type, the last as method.
        if normalized.len() >= 2 {
            let method = normalized.last().cloned().unwrap_or_default();
            let ty = normalized[normalized.len() - 2].clone();
            if let Some(def) = self.index.find_inherent_method(&ty, &method) {
                return ResolvedOrAmbiguous::Resolved {
                    id: def.to_fn_id(&format!("{ty}::{method}")),
                };
            }
        }

        ResolvedOrAmbiguous::Unresolved { hint: joined }
    }

    // ── Method calls ────────────────────────────────────────────────────

    fn resolve_method(
        &self,
        method: &str,
        receiver_hint: Option<&str>,
    ) -> ResolvedOrAmbiguous {
        // If the receiver hint looks like a known type, try InherentMethod
        // lookup directly. The receiver_hint from CalleeHint::MethodCall is
        // the raw source text of the receiver — it may be a variable name,
        // a type path, or an expression. We do a best-effort type match.
        if let Some(receiver) = receiver_hint {
            // Strip trailing `()` or `.method()` patterns from the hint to
            // recover a bare type name in simple cases like `Widget::new()`
            // or `foo.bar()`.
            let bare = receiver.split(|c: char| !c.is_alphanumeric() && c != '_').next().unwrap_or("");
            if !bare.is_empty() {
                if let Some(def) = self.index.find_inherent_method(bare, method) {
                    return ResolvedOrAmbiguous::Resolved {
                        id: def.to_fn_id(&format!("{bare}::{method}")),
                    };
                }
            }
        }

        // Collect all trait methods with this name across the workspace.
        // Sorted for stable Ambiguous candidate order across processes.
        let mut trait_hits: Vec<String> = self
            .index
            .find_trait_methods_by_name(method)
            .into_iter()
            .filter_map(|(key, def)| match key {
                SymbolKey::TraitMethod {
                    trait_name,
                    ty,
                    method,
                } => Some(def.to_fn_id(&format!("{trait_name}::{ty}::{method}"))),
                _ => None,
            })
            .collect();
        trait_hits.sort();

        match trait_hits.len() {
            0 => {}
            1 => {
                return ResolvedOrAmbiguous::Resolved {
                    id: trait_hits.into_iter().next().unwrap(),
                };
            }
            _ => {
                return ResolvedOrAmbiguous::Ambiguous {
                    candidates: trait_hits,
                };
            }
        }

        // Fall back to inherent methods matching by name. Sorted.
        let mut inherent_hits: Vec<String> = self
            .index
            .symbols
            .iter()
            .filter_map(|(k, def)| match k {
                SymbolKey::InherentMethod { ty, method: m } if m == method => {
                    Some(def.to_fn_id(&format!("{ty}::{m}")))
                }
                _ => None,
            })
            .collect();
        inherent_hits.sort();

        match inherent_hits.len() {
            0 => ResolvedOrAmbiguous::Unresolved {
                hint: method.to_string(),
            },
            1 => ResolvedOrAmbiguous::Resolved {
                id: inherent_hits.into_iter().next().unwrap(),
            },
            _ => ResolvedOrAmbiguous::Ambiguous {
                candidates: inherent_hits,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::ProjectIndex;
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

    // ── Bare name resolution ───────────────────────────────────────────

    #[test]
    fn bare_resolves_local_scope() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
fn helper() {}
fn caller() { helper(); }
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::Bare {
            name: "helper".to_string(),
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Resolved { id } => {
                assert!(id.contains("helper"), "resolved id: {id}");
                assert!(id.contains("crates/mini/src/lib.rs"));
            }
            other => panic!("expected Resolved, got {:?}", other),
        }
    }

    #[test]
    fn bare_resolves_via_use_alias() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"pub mod eigen;
use crate::eigen::jacobi;
fn caller() { let _ = jacobi; }
"#,
            ),
            (
                "crates/mini/src/eigen.rs",
                r#"pub fn jacobi() {}
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::Bare {
            name: "jacobi".to_string(),
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Resolved { id } => {
                assert!(id.contains("jacobi"));
                assert!(
                    id.contains("eigen.rs"),
                    "resolved id should point to eigen.rs: {id}"
                );
            }
            other => panic!("expected Resolved via alias, got {:?}", other),
        }
    }

    #[test]
    fn bare_unknown_identifier_is_unresolved() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("crates/mini/src/lib.rs", "fn caller() {}\n"),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::Bare {
            name: "nonexistent".to_string(),
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Unresolved { hint } => {
                assert_eq!(hint, "nonexistent");
            }
            other => panic!("expected Unresolved, got {:?}", other),
        }
    }

    // ── Scoped path resolution ─────────────────────────────────────────

    #[test]
    fn scoped_path_resolves_with_crate_rewrite() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                "pub mod eigen;\nfn main() { let _ = crate::eigen::jacobi(); }\n",
            ),
            ("crates/mini/src/eigen.rs", "pub fn jacobi() {}\n"),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::Scoped {
            segments: vec!["crate".to_string(), "eigen".to_string(), "jacobi".to_string()],
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Resolved { id } => {
                assert!(id.contains("jacobi"));
                assert!(id.contains("eigen.rs"));
            }
            other => panic!("expected Resolved from crate rewrite, got {:?}", other),
        }
    }

    #[test]
    fn scoped_path_falls_back_to_type_method_shape() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
struct Widget;
impl Widget { fn new() -> Self { Widget } }
fn caller() { let _ = Widget::new(); }
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::Scoped {
            segments: vec!["Widget".to_string(), "new".to_string()],
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Resolved { id } => {
                assert!(id.contains("Widget::new"), "id: {id}");
            }
            other => panic!("expected Resolved Type::method, got {:?}", other),
        }
    }

    #[test]
    fn scoped_path_unknown_is_unresolved() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("crates/mini/src/lib.rs", "fn main() {}\n"),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::Scoped {
            segments: vec!["no".into(), "such".into(), "thing".into()],
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Unresolved { hint } => {
                assert_eq!(hint, "no::such::thing");
            }
            other => panic!("expected Unresolved, got {:?}", other),
        }
    }

    // ── Method call resolution — the ambiguity-preservation acid test ──

    #[test]
    fn method_call_with_two_trait_impls_emits_ambiguous() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
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
impl Greet for Hello { fn greet(&self) -> String { "hi".into() } }
impl Greet for World { fn greet(&self) -> String { "world".into() } }

fn caller() {
    let h = Hello;
    let _ = h.greet();
}
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);

        // MethodCall with a bare receiver hint that doesn't match a type
        // name — resolver must fall through to trait search and surface
        // BOTH impls as Ambiguous.
        let hint = CalleeHint::MethodCall {
            receiver_hint: Some("h".to_string()),
            method: "greet".to_string(),
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Ambiguous { candidates } => {
                assert_eq!(
                    candidates.len(),
                    2,
                    "expected 2 candidates, got {:?}",
                    candidates
                );
                let joined = candidates.join(" ");
                assert!(joined.contains("Hello"), "missing Hello impl: {joined}");
                assert!(joined.contains("World"), "missing World impl: {joined}");
            }
            other => panic!(
                "expected Ambiguous for greet with 2 impls, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn method_call_with_receiver_hint_resolves_inherent() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            (
                "crates/mini/src/lib.rs",
                r#"
pub struct Rx;
impl Rx { fn send(&self, _msg: i32) {} }
"#,
            ),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::MethodCall {
            receiver_hint: Some("Rx".to_string()),
            method: "send".to_string(),
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Resolved { id } => {
                assert!(id.contains("Rx::send"), "id: {id}");
            }
            other => panic!("expected Resolved via receiver hint, got {:?}", other),
        }
    }

    #[test]
    fn method_call_unknown_is_unresolved() {
        let files = &[
            ("Cargo.toml", "[workspace]\nmembers = [\"crates/mini\"]\n"),
            (
                "crates/mini/Cargo.toml",
                "[package]\nname = \"mini\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
            ),
            ("crates/mini/src/lib.rs", "fn main() {}\n"),
        ];
        let (_tmp, idx) = build_index(files);
        let resolver = CallSiteResolver::new(&idx);
        let hint = CalleeHint::MethodCall {
            receiver_hint: Some("unknown".to_string()),
            method: "nope".to_string(),
        };
        let out = resolver.resolve(&hint, "crates/mini/src/lib.rs", "mini", &[]);
        match out {
            ResolvedOrAmbiguous::Unresolved { hint } => {
                assert_eq!(hint, "nope");
            }
            other => panic!("expected Unresolved, got {:?}", other),
        }
    }
}
