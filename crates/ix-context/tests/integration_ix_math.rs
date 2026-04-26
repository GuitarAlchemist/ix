//! Integration test: build a [`ProjectIndex`] over the real `ix-math`
//! crate and walk from `ix_math::eigen::symmetric_eigen`.
//!
//! This is the end-to-end validation demanded by the MVP done-criteria in
//! `docs/brainstorms/2026-04-10-context-dag.md`. It uses the real Rust
//! source living in `crates/ix-math/src/eigen.rs` as the fixture, not a
//! synthetic test workspace, so the resolver and walker are exercised
//! against production code and its idioms (tree-sitter queries, use
//! aliases, nested modules, trait impls).
//!
//! # Deliberate scope
//!
//! The brainstorm originally mentioned "callers include MDS, Kernel PCA,
//! LDA" — those callers are real, but they live inside `impl` blocks
//! (`LinearDiscriminantAnalysis::fit`, etc.). The MVP walker handles
//! **free-function callers only**; impl-method callers are a documented
//! v2 limitation in `walk.rs`.
//!
//! So this test focuses on callers and callees that ARE free functions:
//! - `symmetric_eigen` calls `symmetric_eigen_with_opts` (same file,
//!   free-fn → free-fn) — validated by the callees-transitive walk
//! - `symmetric_eigen` is called by several test functions inside
//!   `#[cfg(test)] mod tests` — validated by the callers-transitive walk

use std::path::PathBuf;

use ix_context::{
    index::ProjectIndex,
    model::ResolvedOrAmbiguous,
    walk::{WalkBudget, WalkStrategy, Walker},
};

/// Locate the real ix-math crate relative to this integration test's
/// working directory. At `cargo test` time the current dir is the ix
/// workspace root, so `crates/ix-math` is the target.
fn ix_math_root() -> PathBuf {
    let mut path = std::env::current_dir().expect("cwd");
    // If we're running from inside crates/ix-context/, walk up to the
    // workspace root.
    while !path.join("crates/ix-math/Cargo.toml").exists() {
        if !path.pop() {
            panic!(
                "could not locate crates/ix-math from cwd {:?}",
                std::env::current_dir()
            );
        }
    }
    path.join("crates/ix-math")
}

#[test]
fn build_index_over_real_ix_math() {
    let root = ix_math_root();
    let idx = ProjectIndex::build(&root).expect("index build");

    // Sanity: the index should have a non-trivial number of files and
    // symbols. ix-math has ~20 files at the time of writing.
    assert!(
        idx.files.len() >= 5,
        "expected >= 5 files in ix-math index, got {}",
        idx.files.len()
    );
    assert!(
        idx.symbols.len() >= 10,
        "expected >= 10 symbols in ix-math index, got {}",
        idx.symbols.len()
    );

    // The target symbol must be present under its fully-qualified name.
    assert!(
        idx.find_free_fn("ix-math::eigen::symmetric_eigen")
            .is_some()
            || idx
                .find_free_fn("ix_math::eigen::symmetric_eigen")
                .is_some(),
        "symmetric_eigen not found — symbol names: {:?}",
        idx.symbols
            .keys()
            .filter_map(|k| match k {
                ix_context::index::SymbolKey::FreeFn(p) if p.contains("eigen") => Some(p.clone()),
                _ => None,
            })
            .take(10)
            .collect::<Vec<_>>()
    );
}

/// Determine the crate-name-as-indexed: the package name in ix-math's
/// Cargo.toml is `ix-math` but Rust module paths typically use `ix_math`.
/// The index preserves the Cargo.toml name verbatim.
fn expected_crate_name() -> &'static str {
    "ix-math"
}

#[test]
fn callees_from_symmetric_eigen_finds_with_opts_variant() {
    let root = ix_math_root();
    let idx = ProjectIndex::build(&root).expect("index build");
    let walker = Walker::new(&idx);

    let target = format!("{}::eigen::symmetric_eigen", expected_crate_name());
    let bundle = walker.walk_from_free_fn(
        &target,
        WalkStrategy::CalleesTransitive { max_depth: 2 },
        WalkBudget::default_generous(),
    );

    assert_eq!(bundle.strategy, "callees_transitive");
    assert!(
        !bundle.nodes.is_empty(),
        "expected at least the root node, got 0"
    );

    // symmetric_eigen is a thin wrapper around symmetric_eigen_with_opts.
    // The callees walk should surface that edge.
    let has_with_opts = bundle.edges.iter().any(|e| match &e.to {
        ResolvedOrAmbiguous::Resolved { id } => id.contains("symmetric_eigen_with_opts"),
        _ => false,
    });
    assert!(
        has_with_opts,
        "expected symmetric_eigen -> symmetric_eigen_with_opts edge. Got edges: {:?}",
        bundle
            .edges
            .iter()
            .map(|e| format!("{:?}", e.to))
            .collect::<Vec<_>>()
    );
}

#[test]
fn callers_from_symmetric_eigen_finds_test_functions() {
    let root = ix_math_root();
    let idx = ProjectIndex::build(&root).expect("index build");
    let walker = Walker::new(&idx);

    let target = format!("{}::eigen::symmetric_eigen", expected_crate_name());
    let bundle = walker.walk_from_free_fn(
        &target,
        WalkStrategy::CallersTransitive { max_depth: 3 },
        WalkBudget::default_generous(),
    );

    assert_eq!(bundle.strategy, "callers_transitive");

    // The eigen.rs file has several #[cfg(test)] fn test_*() { ... } that
    // call symmetric_eigen. These are free functions from tree-sitter's
    // perspective (they're inside a `mod tests` but not inside an impl
    // block), so the MVP walker should pick them up.
    //
    // We assert a weaker condition than "exactly N" because test functions
    // come and go — we only require that at least one free-fn caller is
    // resolved. If this asserts zero, the walker has regressed or the
    // fixture has changed dramatically.
    let resolved_callers: Vec<String> = bundle
        .edges
        .iter()
        .filter_map(|e| match &e.to {
            ResolvedOrAmbiguous::Resolved { id } if id.contains("symmetric_eigen") => {
                Some(e.from.clone())
            }
            _ => None,
        })
        .collect();

    assert!(
        !resolved_callers.is_empty(),
        "expected at least one resolved caller of symmetric_eigen, found none. \
         bundle: nodes={} edges={} walk_trace_steps={}",
        bundle.nodes.len(),
        bundle.edges.len(),
        bundle.walk_trace.len()
    );
}

#[test]
fn walk_trace_is_bit_identical_across_replay() {
    // The governance-instrument contract: two walks with the same
    // inputs must produce bit-identical traces. This is what lets
    // Demerzel replay an agent action and verify its informational
    // state.
    let root = ix_math_root();
    let idx = ProjectIndex::build(&root).expect("index build");
    let walker = Walker::new(&idx);

    let target = format!("{}::eigen::symmetric_eigen", expected_crate_name());
    let bundle1 = walker.walk_from_free_fn(
        &target,
        WalkStrategy::CalleesTransitive { max_depth: 2 },
        WalkBudget::default_generous(),
    );
    let bundle2 = walker.walk_from_free_fn(
        &target,
        WalkStrategy::CalleesTransitive { max_depth: 2 },
        WalkBudget::default_generous(),
    );

    assert_eq!(
        bundle1.walk_trace,
        bundle2.walk_trace,
        "replay walk_trace diverged: {} steps vs {} steps",
        bundle1.walk_trace.len(),
        bundle2.walk_trace.len()
    );
    assert_eq!(bundle1.nodes, bundle2.nodes, "replay nodes diverged");
    assert_eq!(bundle1.edges, bundle2.edges, "replay edges diverged");
}

#[test]
fn bundle_json_roundtrip_preserves_structure() {
    let root = ix_math_root();
    let idx = ProjectIndex::build(&root).expect("index build");
    let walker = Walker::new(&idx);

    let target = format!("{}::eigen::symmetric_eigen", expected_crate_name());
    let original = walker.walk_from_free_fn(
        &target,
        WalkStrategy::CalleesTransitive { max_depth: 2 },
        WalkBudget::default_generous(),
    );

    let json = serde_json::to_string(&original).expect("serialize bundle");
    let back: ix_context::model::ContextBundle =
        serde_json::from_str(&json).expect("deserialize bundle");

    assert_eq!(back, original);
    assert_eq!(back.walk_trace, original.walk_trace);
    assert_eq!(back.nodes.len(), original.nodes.len());
    assert_eq!(back.edges.len(), original.edges.len());
}

#[test]
fn module_siblings_from_eigen_enumerates_same_file_fns() {
    let root = ix_math_root();
    let idx = ProjectIndex::build(&root).expect("index build");
    let walker = Walker::new(&idx);

    let target = format!("{}::eigen::symmetric_eigen", expected_crate_name());
    let bundle = walker.walk_from_free_fn(
        &target,
        WalkStrategy::ModuleSiblings,
        WalkBudget::default_generous(),
    );

    // eigen.rs has at least symmetric_eigen and symmetric_eigen_with_opts
    // as module-level free fns, so the siblings walk should include at
    // least the with_opts variant.
    let has_with_opts = bundle
        .nodes
        .iter()
        .any(|n| n.meta().label == "symmetric_eigen_with_opts");
    assert!(
        has_with_opts,
        "siblings walk should include symmetric_eigen_with_opts. \
         Got nodes: {:?}",
        bundle
            .nodes
            .iter()
            .map(|n| n.meta().label.as_str())
            .collect::<Vec<_>>()
    );
}
