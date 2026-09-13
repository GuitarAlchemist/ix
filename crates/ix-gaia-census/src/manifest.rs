//! Parses the declared inventory out of the evidence manifest.
//!
//! The manifest is a Markdown document. Its §2 `Inventory` section carries one
//! table row per declared file — relative path, byte count, SHA-256 — and the
//! manifest never lists itself. Only that section is read; every other table in
//! the document is ignored. There must be exactly one such section: a manifest
//! declaring its inventory twice is refused, wherever the second declaration
//! sits.

use crate::CensusRefusal;

/// One row of the declared inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeclaredRow {
    pub(crate) name: String,
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

const SECTION_HEADING: &str = "## 2. Inventory";

/// Parse the declared inventory. Refuses rather than guessing.
pub(crate) fn parse(manifest_bytes: &[u8]) -> Result<Vec<DeclaredRow>, CensusRefusal> {
    let text = std::str::from_utf8(manifest_bytes).map_err(|e| CensusRefusal::ManifestUnparseable {
        line: 0,
        detail: format!("manifest is not valid UTF-8: {e}"),
    })?;

    let mut rows = Vec::new();
    let mut in_section = false;
    let mut section_line: Option<usize> = None;

    // The whole document is scanned. Stopping at the end of the first inventory
    // section would let a second one — a wholly contradictory re-declaration —
    // sit below it unread and collapse into a pass.
    for (index, raw_line) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();

        if line.starts_with("## ") {
            if line != SECTION_HEADING {
                in_section = false;
                continue;
            }
            // Two inventory sections are two contradictory declarations of the
            // same thing. Both locations are retained in the refusal, and
            // neither declaration is chosen over the other.
            if let Some(first) = section_line {
                return Err(CensusRefusal::DuplicateInventorySection {
                    first,
                    second: line_number,
                });
            }
            section_line = Some(line_number);
            in_section = true;
            continue;
        }
        if !in_section || !line.starts_with('|') {
            continue;
        }

        let cells = split_row(line);
        if is_separator(&cells) || is_header(&cells) {
            continue;
        }
        rows.push(parse_row(&cells, line_number)?);
    }

    if rows.is_empty() {
        return Err(CensusRefusal::ManifestDeclaresNoRows);
    }
    if let Some(name) = first_duplicate(&rows) {
        return Err(CensusRefusal::DuplicateManifestRow { name });
    }
    Ok(rows)
}

/// The ordinally first name declared more than once, if any.
///
/// A name declared twice is refused whether or not the two declarations agree:
/// the disagreement is retained in the refusal, never collapsed into a pass and
/// never silently resolved to one of the two.
fn first_duplicate(rows: &[DeclaredRow]) -> Option<String> {
    let mut seen: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for row in rows {
        *seen.entry(row.name.as_str()).or_insert(0) += 1;
    }
    seen.into_iter()
        .find(|(_, count)| *count > 1)
        .map(|(name, _)| name.to_string())
}

/// Split a Markdown table row into its cells, dropping the leading and
/// trailing pipe delimiters.
fn split_row(line: &str) -> Vec<String> {
    let inner = line
        .strip_prefix('|')
        .unwrap_or(line)
        .strip_suffix('|')
        .unwrap_or(line);
    inner
        .split('|')
        .map(|cell| cell.trim().trim_matches('`').trim().to_string())
        .collect()
}

fn is_separator(cells: &[String]) -> bool {
    !cells.is_empty()
        && cells
            .iter()
            .all(|cell| !cell.is_empty() && cell.chars().all(|c| c == '-' || c == ':'))
}

fn is_header(cells: &[String]) -> bool {
    cells.first().map(String::as_str) == Some("Relative path")
}

fn parse_row(cells: &[String], line: usize) -> Result<DeclaredRow, CensusRefusal> {
    if cells.len() != 3 {
        return Err(CensusRefusal::ManifestUnparseable {
            line,
            detail: format!("expected 3 cells, found {}", cells.len()),
        });
    }

    let name = cells[0].clone();
    if name.is_empty() {
        return Err(CensusRefusal::ManifestUnparseable {
            line,
            detail: "empty relative path".to_string(),
        });
    }

    // A declared name is a plain file name inside the evidence root. Anything
    // carrying a path separator, a drive or stream colon, or a `.`/`..` segment
    // is refused before any read is attempted, so no read leaves the root.
    if name.contains('/') || name.contains('\\') || name.contains(':') || name == "." || name == ".."
    {
        return Err(CensusRefusal::UnsafeManifestName { name });
    }

    let digits: String = cells[1].chars().filter(|c| *c != ',').collect();
    let bytes = digits
        .parse::<u64>()
        .map_err(|_| CensusRefusal::ManifestUnparseable {
            line,
            detail: format!("byte count is not a non-negative integer: {:?}", cells[1]),
        })?;

    let sha256 = cells[2].clone();
    if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()) {
        return Err(CensusRefusal::ManifestUnparseable {
            line,
            detail: format!("digest is not 64 lowercase hex characters: {sha256:?}"),
        });
    }

    Ok(DeclaredRow {
        name,
        bytes,
        sha256,
    })
}
