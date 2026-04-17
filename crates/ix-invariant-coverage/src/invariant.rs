//! Invariant catalog model + markdown parser.
//!
//! Parses the markdown table from `docs/methodology/invariants-catalog.md`. The table format is:
//!
//! ```text
//! | # | Domain | Invariant | Artifact | Status |
//! |---|---|---|---|---|
//! | 1 | enum | Every `ChordQuality` enum value ... | GA.Domain.Core | C |
//! ```
//!
//! Status letters: T (tested), C (claimed, untested), N (latent, not claimed), FAIL (known broken).

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::coverage::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Invariant {
    pub id: u32,
    pub domain: String,
    pub description: String,
    pub artifact: String,
    pub status: InvariantStatus,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum InvariantStatus {
    Tested,
    Claimed,
    Latent,
    Failing,
}

impl InvariantStatus {
    pub fn short(&self) -> &'static str {
        match self {
            Self::Tested => "T",
            Self::Claimed => "C",
            Self::Latent => "N",
            Self::Failing => "FAIL",
        }
    }
}

pub fn parse_catalog(path: &Path) -> Result<Vec<Invariant>, Error> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| Error::Io(format!("reading {}: {e}", path.display())))?;
    parse_catalog_str(&content)
}

pub fn parse_catalog_str(content: &str) -> Result<Vec<Invariant>, Error> {
    let mut invariants = Vec::new();
    let mut in_table = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('|') {
            in_table = false;
            continue;
        }
        // Skip header/separator rows
        if trimmed.contains("---") || trimmed.to_lowercase().contains("| # |") {
            in_table = true;
            continue;
        }
        if !in_table {
            continue;
        }

        let cells: Vec<&str> = trimmed
            .split('|')
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .collect();
        if cells.len() < 5 {
            continue;
        }

        // First cell must parse as an id; otherwise this is a different table
        let Ok(id) = cells[0].parse::<u32>() else {
            continue;
        };

        let status = match cells[4] {
            s if s.starts_with('T') => InvariantStatus::Tested,
            s if s.starts_with('C') => InvariantStatus::Claimed,
            s if s.starts_with('N') => InvariantStatus::Latent,
            s if s.contains("FAIL") => InvariantStatus::Failing,
            other => return Err(Error::Parse(format!("unknown status '{other}' for invariant {id}"))),
        };

        invariants.push(Invariant {
            id,
            domain: cells[1].to_string(),
            description: cells[2].to_string(),
            artifact: cells[3].to_string(),
            status,
        });
    }

    if invariants.is_empty() {
        return Err(Error::Parse("no invariant rows parsed from catalog".into()));
    }
    Ok(invariants)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
Some prose above the table.

| # | Domain | Invariant | Artifact | Status |
|---|---|---|---|---|
| 1 | enum | Every `ChordQuality` enum value appears in YAML | GA.Domain.Core | C |
| 2 | enum | Every `HarmonicFunction` value is emitted | HarmonicFunction.cs | T |
| 25 | embedding | Cross-instrument STRUCTURE equality | optick.index | **FAIL** (56% leak) |

Another section.

| # | Unrelated | Note |
|---|---|---|
| M1 | code | something |
";

    #[test]
    fn parses_five_column_rows_and_skips_unrelated_tables() {
        let invs = parse_catalog_str(SAMPLE).unwrap();
        assert_eq!(invs.len(), 3);
        assert_eq!(invs[0].id, 1);
        assert_eq!(invs[0].domain, "enum");
        assert_eq!(invs[0].status, InvariantStatus::Claimed);
        assert_eq!(invs[1].status, InvariantStatus::Tested);
        assert_eq!(invs[2].id, 25);
        assert_eq!(invs[2].status, InvariantStatus::Failing);
    }

    #[test]
    fn status_short_codes_round_trip() {
        assert_eq!(InvariantStatus::Tested.short(), "T");
        assert_eq!(InvariantStatus::Claimed.short(), "C");
        assert_eq!(InvariantStatus::Latent.short(), "N");
        assert_eq!(InvariantStatus::Failing.short(), "FAIL");
    }

    #[test]
    fn empty_input_is_error() {
        let err = parse_catalog_str("no tables here").unwrap_err();
        assert!(matches!(err, Error::Parse(_)));
    }
}
