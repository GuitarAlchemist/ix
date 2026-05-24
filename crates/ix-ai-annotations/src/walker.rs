//! File-system walker that streams [`Annotation`]s out of a workspace.

use crate::parser::{parse_line, ParsedMarker};
use crate::types::{annotation_id, Annotation, Location, Source, SCHEMA_VERSION};
use crate::Error;
use ignore::WalkBuilder;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// File extensions we'll bother scanning. Tuned to the GuitarAlchemist ecosystem
/// languages plus common config/docs.
const SCANNABLE_EXTS: &[&str] = &[
    "rs", "cs", "fs", "fsx", "fsi", "ts", "tsx", "js", "jsx", "py", "rb", "go", "java", "swift",
    "c", "h", "cpp", "hpp", "lua", "sql", "hs", "sh", "ps1", "psm1", "yml", "yaml", "toml",
    "html", "htm", "xml", "md",
];

fn has_scannable_ext(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| SCANNABLE_EXTS.iter().any(|w| w.eq_ignore_ascii_case(e)))
        .unwrap_or(false)
}

/// Extract every `@ai:` annotation under `workspace`. Uses `ignore::WalkBuilder`
/// so we honor `.gitignore`, `.ignore`, and skip hidden dirs by default.
pub fn extract(workspace: &Path) -> Result<Vec<Annotation>, Error> {
    let now = chrono::Utc::now().to_rfc3339();
    let mut out = Vec::new();

    let mut walker = WalkBuilder::new(workspace);
    walker
        .hidden(false) // .claude/, .github/ should be visible — they may have annotations
        .git_ignore(true)
        .git_exclude(true)
        .filter_entry(|e| {
            let name = e.file_name().to_string_lossy();
            // Hard-exclude these regardless of .gitignore
            !matches!(
                name.as_ref(),
                "target" | "node_modules" | ".git" | "dist" | "build" | ".next" | ".venv" | "venv"
            )
        });

    for result in walker.build() {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };
        if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            continue;
        }
        let path = entry.path();
        if !has_scannable_ext(path) {
            continue;
        }
        scan_file(workspace, path, &now, &mut out)?;
    }

    Ok(out)
}

fn scan_file(
    workspace: &Path,
    path: &Path,
    now: &str,
    out: &mut Vec<Annotation>,
) -> Result<(), Error> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return Ok(()), // Skip files we can't open
    };
    let reader = BufReader::new(file);
    let rel = path
        .strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/");

    for (idx, line_res) in reader.lines().enumerate() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => return Ok(()), // Likely binary or invalid UTF-8
        };
        let lineno = (idx + 1) as u32;
        if let Some(marker) = parse_line(&line) {
            out.push(promote(marker, &rel, lineno, now));
        }
    }
    Ok(())
}

fn promote(marker: ParsedMarker, rel_path: &str, lineno: u32, now: &str) -> Annotation {
    let ParsedMarker {
        kind,
        claim,
        truth_value,
        certainty,
        confidence,
        evidence,
    } = marker;
    let id = annotation_id(rel_path, lineno, kind, &claim);
    Annotation {
        schema_version: SCHEMA_VERSION,
        id,
        kind,
        claim,
        truth_value,
        certainty,
        confidence: confidence.unwrap_or(0.5),
        source: Source {
            author: "auto".to_string(),
            model: None,
            evidence,
        },
        location: Location {
            path: rel_path.to_string(),
            line_start: lineno,
            line_end: lineno,
        },
        created_at: now.to_string(),
        updated_at: now.to_string(),
        stale: false,
        reconciliation: None,
    }
}
