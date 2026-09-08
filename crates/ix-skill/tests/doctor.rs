//! `ix doctor` tests (ix#185).
//!
//! The orphan-trait tests seed a violation into a throwaway workspace and
//! assert the check catches it — a check nobody has watched fail is not a
//! check. The boundary cases (bound-only use, multi-line impl, derive) pin
//! down the scanner's deliberate bias: it must never invent a finding.

use ix_skill::doctor::orphan_traits::{self, Allowlist, AllowEntry};
use ix_skill::doctor::registry_snapshot::{self, Live, Snapshot, Surface, SCHEMA_VERSION};
use ix_skill::doctor::{self, Status};
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// The real IX workspace root, for the checks that must hold on `main`.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

/// Build a throwaway workspace containing `crates/seed/src/lib.rs`.
fn seeded_repo(lib_rs: &str) -> TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("crates/seed/src");
    std::fs::create_dir_all(&src).expect("mkdir");
    std::fs::write(src.join("lib.rs"), lib_rs).expect("write lib.rs");
    dir
}

fn census(lib_rs: &str, allowlist: &Allowlist) -> orphan_traits::Census {
    let dir = seeded_repo(lib_rs);
    orphan_traits::scan_with_allowlist(dir.path(), allowlist).expect("scan")
}

fn allow(trait_name: &str, reason: &str) -> Allowlist {
    Allowlist {
        note: String::new(),
        allow: vec![AllowEntry {
            trait_name: trait_name.to_string(),
            reason: reason.to_string(),
            issue: None,
        }],
    }
}

// ---------------------------------------------------------------- orphan scan

#[test]
fn seeded_orphan_trait_is_reported() {
    let c = census(
        "pub trait Ghost {\n    fn haunt(&self);\n}\n",
        &Allowlist::default(),
    );
    let names: Vec<&str> = c.unlisted_orphans.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["Ghost"], "seeded orphan should be reported");
    let ghost = &c.unlisted_orphans[0];
    assert_eq!(ghost.impls, 0);
    assert_eq!(ghost.bounds, 0);
    assert_eq!(ghost.file, "crates/seed/src/lib.rs");
    assert_eq!(ghost.line, 1, "reports the declaration line");
}

#[test]
fn trait_with_an_implementor_is_not_an_orphan() {
    let c = census(
        "pub trait Ghost { fn haunt(&self); }\n\
         pub struct House;\n\
         impl Ghost for House { fn haunt(&self) {} }\n",
        &Allowlist::default(),
    );
    assert!(c.unlisted_orphans.is_empty(), "{:?}", c.unlisted_orphans);
    let ghost = c.traits.iter().find(|t| t.name == "Ghost").expect("Ghost");
    assert_eq!(ghost.impls, 1);
}

#[test]
fn trait_used_only_as_a_generic_bound_is_not_an_orphan() {
    // The load-bearing case for the "reliable for zeros" rule: a trait with no
    // in-tree implementor but a live bound is an open extension point, not
    // dead surface. Flagging it would be the false positive that gets the
    // whole check switched off.
    let c = census(
        "pub trait Ghost { fn haunt(&self); }\n\
         pub fn spook<G: Ghost>(g: &G) { g.haunt() }\n",
        &Allowlist::default(),
    );
    assert!(c.unlisted_orphans.is_empty(), "{:?}", c.unlisted_orphans);
    let ghost = c.traits.iter().find(|t| t.name == "Ghost").expect("Ghost");
    assert_eq!(ghost.impls, 0);
    assert_eq!(ghost.bounds, 1);
}

#[test]
fn trait_object_use_counts_as_a_bound() {
    let c = census(
        "pub trait Ghost { fn haunt(&self); }\n\
         pub struct Attic { tenant: Box<dyn Ghost> }\n",
        &Allowlist::default(),
    );
    assert!(c.unlisted_orphans.is_empty(), "{:?}", c.unlisted_orphans);
}

#[test]
fn impl_header_split_across_lines_is_still_an_impl() {
    // rustfmt wraps long impl headers; a line-oriented scanner would miss this
    // and report a false orphan.
    let c = census(
        "pub trait Ghost { fn haunt(&self); }\n\
         pub struct House<T>(T);\n\
         impl<T: Clone>\n    Ghost\n    for House<T>\n{\n    fn haunt(&self) {}\n}\n",
        &Allowlist::default(),
    );
    assert!(c.unlisted_orphans.is_empty(), "{:?}", c.unlisted_orphans);
    let ghost = c.traits.iter().find(|t| t.name == "Ghost").expect("Ghost");
    assert_eq!(ghost.impls, 1, "multi-line impl header must be counted");
}

#[test]
fn generic_bound_inside_an_impl_header_is_not_an_impl_of_that_trait() {
    // `impl<T: Ghost> Other for Wrapper<T>` implements `Other`, not `Ghost`.
    // Miscounting it would hide a genuine orphan behind an unrelated impl.
    let c = census(
        "pub trait Ghost { fn haunt(&self); }\n\
         pub trait Other {}\n\
         pub struct Wrapper<T>(T);\n\
         impl<T: Ghost> Other for Wrapper<T> {}\n",
        &Allowlist::default(),
    );
    let ghost = c.traits.iter().find(|t| t.name == "Ghost").expect("Ghost");
    assert_eq!(ghost.impls, 0, "not an impl of Ghost");
    assert_eq!(ghost.bounds, 1, "but it is a bound use");
}

#[test]
fn derive_backed_trait_is_not_an_orphan() {
    // A derive macro implements the trait with no textual `impl` header.
    let c = census(
        "pub trait Ghost {}\n\
         #[derive(Debug, Ghost)]\n\
         pub struct House;\n",
        &Allowlist::default(),
    );
    assert!(c.unlisted_orphans.is_empty(), "{:?}", c.unlisted_orphans);
}

#[test]
fn trait_declared_in_a_doc_comment_is_not_a_declaration() {
    let c = census(
        "/// Example:\n/// pub trait Phantom {}\npub struct Real;\n",
        &Allowlist::default(),
    );
    assert!(
        !c.traits.iter().any(|t| t.name == "Phantom"),
        "doc-comment example must not register as surface: {:?}",
        c.traits
    );
}

// ----------------------------------------------------------------- allowlist

#[test]
fn allowlist_entry_exempts_a_seeded_orphan() {
    let c = census(
        "pub trait Ghost { fn haunt(&self); }\n",
        &allow("Ghost", "kept for the 0.2 I/O rework; tracked in ix#000"),
    );
    assert!(c.unlisted_orphans.is_empty(), "{:?}", c.unlisted_orphans);
    assert_eq!(c.allowed_orphans, ["Ghost"]);
    assert!(c.stale_allowlist.is_empty());
}

#[test]
fn allowlist_entry_without_a_reason_is_rejected() {
    // An exemption nobody justified is a silencer, not a decision.
    let c = census("pub trait Ghost { fn haunt(&self); }\n", &allow("Ghost", "   "));
    assert_eq!(c.reasonless_allowlist, ["Ghost"]);
}

#[test]
fn allowlist_entry_for_a_non_orphan_is_stale() {
    let c = census(
        "pub trait Ghost { fn haunt(&self); }\n\
         pub struct House;\n\
         impl Ghost for House { fn haunt(&self) {} }\n",
        &allow("Ghost", "no longer needed"),
    );
    assert_eq!(
        c.stale_allowlist,
        ["Ghost"],
        "an exemption whose trait gained an implementor must be flagged"
    );
}

// ---------------------------------------------------------- registry snapshot

fn snapshot_of(skills: &[&str], tools: &[&str], gated: &[&str]) -> Snapshot {
    Snapshot {
        schema_version: SCHEMA_VERSION,
        note: String::new(),
        skills: Surface::from_names(skills.iter().map(|s| (*s).to_string())),
        mcp_tools: Surface::from_names(tools.iter().map(|s| (*s).to_string())),
        feature_gated_tools: gated.iter().map(|s| (*s).to_string()).collect(),
    }
}

fn live_of(skills: &[&str], tools: &[&str]) -> Live {
    Live {
        skills: Surface::from_names(skills.iter().map(|s| (*s).to_string())),
        mcp_tools: Surface::from_names(tools.iter().map(|s| (*s).to_string())),
    }
}

#[test]
fn snapshot_matching_the_live_build_is_clean() {
    let snap = snapshot_of(&["a.fit"], &["ix_a"], &[]);
    let live = live_of(&["a.fit"], &["ix_a"]);
    assert!(registry_snapshot::diff(&snap, &live).is_clean());
}

#[test]
fn seeded_registry_drift_names_the_capability_in_both_directions() {
    // This is the `mesh_correlate` incident in miniature: a tool appears in the
    // build and not in the oracle.
    let snap = snapshot_of(&["a.fit"], &["ix_a", "ix_gone"], &[]);
    let live = live_of(&["a.fit", "b.fit"], &["ix_a", "ix_mesh_correlate"]);
    let drift = registry_snapshot::diff(&snap, &live);
    assert!(!drift.is_clean());
    assert_eq!(drift.skills_added, ["b.fit"]);
    assert!(drift.skills_removed.is_empty());
    assert_eq!(drift.tools_added, ["ix_mesh_correlate"]);
    assert_eq!(drift.tools_removed, ["ix_gone"]);
}

#[test]
fn feature_gated_tools_do_not_count_as_drift_either_way() {
    // The default build and a `--features maintain-gate` build must both stay
    // green against one snapshot.
    let snap = snapshot_of(&[], &["ix_a"], &["ix_maintain_gate"]);
    assert!(registry_snapshot::diff(&snap, &live_of(&[], &["ix_a"])).is_clean());
    assert!(
        registry_snapshot::diff(&snap, &live_of(&[], &["ix_a", "ix_maintain_gate"])).is_clean(),
        "the feature-gated tool must be tolerated when present"
    );
}

#[test]
fn snapshot_write_then_load_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    let live = live_of(&["a.fit", "b.fit"], &["ix_a"]);
    let written = registry_snapshot::snapshot_of(&live, vec!["ix_gated".to_string()]);
    written.write(dir.path()).expect("write");

    let loaded = Snapshot::load(dir.path()).expect("load");
    assert_eq!(loaded.skills, written.skills);
    assert_eq!(loaded.mcp_tools, written.mcp_tools);
    assert_eq!(loaded.feature_gated_tools, ["ix_gated"]);
    assert!(registry_snapshot::diff(&loaded, &live).is_clean());
}

#[test]
fn a_missing_snapshot_says_how_to_generate_one() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = Snapshot::load(dir.path()).expect_err("no snapshot present");
    assert!(
        err.contains("doctor --write"),
        "error must be actionable, got: {err}"
    );
}

// ------------------------------------------------------------ the live repo

#[test]
fn doctor_is_green_on_this_workspace() {
    // The gate itself. `registry-snapshot` failing here means the committed
    // snapshot no longer matches the build; `orphan-traits` failing means a
    // new public trait has no implementor, no bound, and no allowlist entry.
    let report = doctor::run(&repo_root(), doctor::Options::default());
    let bad: Vec<&doctor::CheckResult> = report
        .checks
        .iter()
        .filter(|c| c.status == Status::Fail)
        .collect();
    assert!(
        bad.is_empty(),
        "ix doctor failing on this workspace:\n{}",
        bad.iter()
            .map(|c| format!(
                "  [{}] {}\n    -> {}",
                c.name,
                c.summary,
                c.remedy.as_deref().unwrap_or("(no remedy)")
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn every_non_ok_check_carries_a_remedy() {
    // The issue asked for actionable messages, not raw test noise.
    let report = doctor::run(&repo_root(), doctor::Options::default());
    for c in &report.checks {
        if matches!(c.status, Status::Warn | Status::Fail) {
            assert!(
                c.remedy.as_deref().is_some_and(|r| !r.trim().is_empty()),
                "check `{}` is {:?} but offers no remedy",
                c.name,
                c.status
            );
        }
    }
}

#[test]
fn the_committed_allowlist_documents_every_exemption() {
    let allowlist = Allowlist::load(&repo_root()).expect("load allowlist");
    for entry in &allowlist.allow {
        assert!(
            !entry.reason.trim().is_empty(),
            "allowlist entry `{}` has no reason",
            entry.trait_name
        );
    }
}

#[test]
fn find_repo_root_walks_up_from_a_nested_directory() {
    let root = repo_root();
    let nested = root.join("crates/ix-skill/src");
    assert_eq!(
        doctor::find_repo_root(&nested).expect("root found"),
        root,
        "doctor must work from anywhere inside the repo"
    );
}
