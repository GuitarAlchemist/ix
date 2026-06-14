//! `streeling` — the Streeling University CLI.
//!
//! - `streeling catalog` — regenerate `state/streeling/catalog.jsonl` from all roots.
//! - `streeling campus`  — render `docs/streeling/README.md` from the catalog.
//! - `streeling check`   — exit non-zero if the committed catalog drifts from sources.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use ix_streeling::{campus, check, default_roots, from_jsonl, ingest::ingest, to_jsonl};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "streeling", about = "Streeling University — learnings catalog + campus index")]
struct Cli {
    /// Path to the ix repo root (sibling repos are discovered beside it).
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Regenerate the learnings catalog from all roots.
    Catalog,
    /// Render the campus index from the catalog.
    Campus,
    /// Fail if the committed catalog is stale relative to the sources.
    Check,
}

fn catalog_path(root: &Path) -> PathBuf {
    root.join("state/streeling/catalog.jsonl")
}
fn campus_path(root: &Path) -> PathBuf {
    root.join("docs/streeling/README.md")
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
                "catalog: {} records ({} skipped); roots seen {:?}, missing {:?}",
                rep.records.len(),
                rep.skipped,
                rep.roots_seen,
                rep.roots_missing
            );
        }
        Cmd::Campus => {
            let path = catalog_path(&cli.repo_root);
            let raw = std::fs::read_to_string(&path)
                .with_context(|| format!("read {} (run `streeling catalog` first)", path.display()))?;
            let records = from_jsonl(&raw);
            let out = campus_path(&cli.repo_root);
            if let Some(p) = out.parent() {
                std::fs::create_dir_all(p)?;
            }
            let existing = std::fs::read_to_string(&out).ok();
            std::fs::write(&out, campus::render(&records, existing.as_deref()))?;
            eprintln!("campus: wrote {} ({} records)", out.display(), records.len());
        }
        Cmd::Check => {
            let rep = ingest(&roots);
            let path = catalog_path(&cli.repo_root);
            let committed = from_jsonl(
                &std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?,
            );
            let d = check::drift(&rep.records, &committed, &rep.roots_seen);
            if d.is_clean() {
                eprintln!("streeling: catalog is fresh ({} records)", committed.len());
            } else {
                eprintln!("streeling: catalog DRIFT — regenerate with `streeling catalog`:");
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
