//! Parses the declared inventory, and every corroborating claim site, out of
//! the evidence manifest.
//!
//! Two different things are read, and they are kept apart:
//!
//! * the **declaration of record** — the manifest's §2 `Inventory` section, one
//!   table row per declared file: relative path, byte count, SHA-256. There
//!   must be exactly one such section; a manifest declaring its inventory twice
//!   is refused wherever the second declaration sits, because two declarations
//!   of the same thing are a contradiction and neither may be chosen over the
//!   other;
//! * a **corroborating claim site** — any *other* table row in the document
//!   whose first cell is a declared file name and whose second cell parses as a
//!   byte count. It is a second site claiming the same quantity.
//!
//! When a corroborating site disagrees with the declaration of record, the two
//! claim sites are **both retained** and the affected row is reported
//! `Hexavalent::Contradictory`. This is the `210`-versus-`214` shape the
//! approved evidence documents and deliberately declines to re-stamp: the
//! disagreement is a fact about the document and collapsing it to either value
//! would be agreement bought by rewriting history.

use std::collections::BTreeMap;

use crate::AdvisoryRefusal;

/// One row of the declared inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeclaredRow {
    pub(crate) name: String,
    /// unit: bytes
    pub(crate) bytes: u64,
    pub(crate) sha256: String,
}

/// Everything the manifest declares about the bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Declaration {
    /// Ordinal by name.
    pub(crate) rows: Vec<DeclaredRow>,
    /// Byte counts claimed for a declared name by a site outside the inventory.
    pub(crate) corroborated_bytes: BTreeMap<String, u64>,
}

const SECTION_HEADING: &str = "## 2. Inventory";

/// Parse the manifest. Refuses rather than guessing.
pub(crate) fn parse(manifest_bytes: &[u8]) -> Result<Declaration, AdvisoryRefusal> {
    let text =
        std::str::from_utf8(manifest_bytes).map_err(|e| AdvisoryRefusal::ManifestUnparseable {
            line: 0,
            detail: format!("manifest is not valid UTF-8: {e}"),
        })?;

    let mut rows = Vec::new();
    let mut in_section = false;
    let mut section_line: Option<usize> = None;

    // The whole document is scanned. Stopping at the end of the first inventory
    // section would let a second, wholly contradictory re-declaration sit below
    // it unread and collapse into a pass.
    for (index, raw_line) in text.lines().enumerate() {
        let line_number = index + 1;
        let line = raw_line.trim();

        if line.starts_with("## ") {
            if line != SECTION_HEADING {
                in_section = false;
                continue;
            }
            if let Some(first) = section_line {
                return Err(AdvisoryRefusal::DuplicateInventorySection {
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
        return Err(AdvisoryRefusal::ManifestDeclaresNoRows);
    }
    if let Some(name) = first_duplicate(&rows) {
        return Err(AdvisoryRefusal::DuplicateManifestRow { name });
    }
    rows.sort_by(|a, b| a.name.as_bytes().cmp(b.name.as_bytes()));

    let declared_names: std::collections::BTreeSet<&str> =
        rows.iter().map(|row| row.name.as_str()).collect();
    Ok(Declaration {
        corroborated_bytes: corroborating_sites(text, &declared_names),
        rows,
    })
}

/// Every claim site outside the inventory that states a byte count for a
/// declared name. The ordinally first site for a name wins; a document that
/// states the same figure twice states it once.
fn corroborating_sites(
    text: &str,
    declared_names: &std::collections::BTreeSet<&str>,
) -> BTreeMap<String, u64> {
    let mut out: BTreeMap<String, u64> = BTreeMap::new();
    let mut in_section = false;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.starts_with("## ") {
            in_section = line == SECTION_HEADING;
            continue;
        }
        if in_section || !line.starts_with('|') {
            continue;
        }
        let cells = split_row(line);
        if cells.len() < 2 || !declared_names.contains(cells[0].as_str()) {
            continue;
        }
        let digits: String = cells[1].chars().filter(|c| *c != ',').collect();
        if digits.is_empty() {
            continue;
        }
        if let Ok(bytes) = digits.parse::<u64>() {
            out.entry(cells[0].clone()).or_insert(bytes);
        }
    }
    out
}

/// The ordinally first name declared more than once, if any.
///
/// A name declared twice is refused whether or not the two declarations agree:
/// the disagreement is retained in the refusal, never collapsed into a pass.
fn first_duplicate(rows: &[DeclaredRow]) -> Option<String> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for row in rows {
        *seen.entry(row.name.as_str()).or_insert(0) += 1;
    }
    seen.into_iter()
        .find(|(_, count)| *count > 1)
        .map(|(name, _)| name.to_string())
}

/// Split a Markdown table row into its cells, dropping the delimiting pipes.
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

fn parse_row(cells: &[String], line: usize) -> Result<DeclaredRow, AdvisoryRefusal> {
    if cells.len() != 3 {
        return Err(AdvisoryRefusal::ManifestUnparseable {
            line,
            detail: format!("expected 3 cells, found {}", cells.len()),
        });
    }

    let name = cells[0].clone();
    if name.is_empty() {
        return Err(AdvisoryRefusal::ManifestUnparseable {
            line,
            detail: "empty relative path".to_string(),
        });
    }

    // A declared name is a plain file name inside the evidence root. Anything
    // carrying a path separator, a drive or stream colon, or a `.`/`..` segment
    // is refused before any read is attempted, so no read leaves the root.
    if name.contains('/')
        || name.contains('\\')
        || name.contains(':')
        || name == "."
        || name == ".."
    {
        return Err(AdvisoryRefusal::UnsafeManifestName { name });
    }

    let digits: String = cells[1].chars().filter(|c| *c != ',').collect();
    let bytes = digits
        .parse::<u64>()
        .map_err(|_| AdvisoryRefusal::ManifestUnparseable {
            line,
            detail: format!("byte count is not a non-negative integer: {:?}", cells[1]),
        })?;

    let sha256 = cells[2].clone();
    if sha256.len() != 64
        || !sha256
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase())
    {
        return Err(AdvisoryRefusal::ManifestUnparseable {
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
