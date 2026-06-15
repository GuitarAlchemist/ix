//! `ix-value` — the Business-Value Scorecard CLI.
//!
//! - `ix-value catalog` — regenerate `state/value/catalog.jsonl` from all roots.
//! - `ix-value check`   — exit non-zero if the committed catalog drifts from manifests.
//!
//! (No `campus`/render command — the UI is GA-side, reading the catalog via
//! `/dev-data/value`.)

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ix_value::{check, default_roots, from_jsonl, ingest::ingest, to_jsonl};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "ix-value", about = "Business-Value Scorecard — federate RICE manifests into a catalog")]
struct Cli {
    /// Path to the ix repo root (sibling repos are discovered beside it).
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Regenerate the value catalog from all manifests.
    Catalog,
    /// Fail if the committed catalog is stale relative to the manifests.
    Check,
}

fn catalog_path(root: &Path) -> PathBuf {
    root.join("state/value/catalog.jsonl")
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let roots = default_roots(&cli.repo_root);

    match cli.cmd {
        Cmd::Catalog => {
            let rep = ingest(&roots);
            let path = catalog_path(&cli.repo_root);
            if let Some(p) = path.parent() {
                std::fs::create_dir_all(p)?;
            }
            std::fs::write(&path, to_jsonl(&rep.records))
                .with_context(|| format!("write {}", path.display()))?;
            eprintln!(
                "value: {} records ({} skipped); roots seen {:?}, missing {:?}",
                rep.records.len(),
                rep.skipped,
                rep.roots_seen,
                rep.roots_missing
            );
        }
        Cmd::Check => {
            let rep = ingest(&roots);
            let path = catalog_path(&cli.repo_root);
            let committed = from_jsonl(
                &std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?,
            );
            let d = check::drift(&rep.records, &committed, &rep.roots_seen);
            if d.is_clean() {
                eprintln!("ix-value: catalog is fresh ({} records)", committed.len());
            } else {
                eprintln!("ix-value: catalog DRIFT — regenerate with `ix-value catalog`:");
                if !d.missing.is_empty() {
                    eprintln!("  uncatalogued ({}): {:?}", d.missing.len(), d.missing);
                }
                if !d.extra.is_empty() {
                    eprintln!("  stale/removed ({}): {:?}", d.extra.len(), d.extra);
                }
                if !d.changed.is_empty() {
                    eprintln!("  changed ({}): {:?}", d.changed.len(), d.changed);
                }
                std::process::exit(1);
            }
        }
    }
    Ok(())
}
