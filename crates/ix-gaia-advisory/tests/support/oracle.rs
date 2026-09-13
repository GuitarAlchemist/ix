//! The label oracle — ground truth, and nothing else.
//!
//! It is deliberately blind. Its input is **a directory of bytes and the frozen
//! original manifest**; it never learns which family produced a case, which
//! site was perturbed, or which rule will be scored against it. That blindness
//! is what makes a label non-circular: the label of a `F-CRLF` case is not
//! "`F` because it is `F-CRLF`", it is whatever the oracle measures over the
//! staged bytes.
//!
//! The frozen original manifest is held **outside** every staged root. When a
//! case perturbs the manifest itself, the oracle still compares against the
//! frozen original — otherwise a perturbation could rewrite its own ground
//! truth. The frozen bundle is all fourteen approved files, so the manifest's
//! own bytes are part of the declaration of record, not merely the source of it.

use std::collections::BTreeMap;
use std::path::Path;

use ix_gaia_advisory::Hexavalent;

use super::digest::sha256_hex;

/// One declared row of the frozen original manifest.
struct Declared {
    bytes: u64,
    sha256: String,
}

/// Label a staged root.
///
/// Returns `T` iff every declared row of the frozen original manifest
/// reconciles against the staged root by fresh SHA-256 and byte length, the
/// staged root's name set equals the frozen bundle's name set, and the staged
/// manifest is byte-identical to the frozen original. Otherwise `F`.
pub fn label(
    staged_root: &Path,
    frozen_manifest_name: &str,
    frozen_manifest_bytes: &[u8],
) -> Hexavalent {
    let declared = parse_inventory(frozen_manifest_bytes);

    let mut frozen_names: Vec<String> = declared.keys().cloned().collect();
    frozen_names.push(frozen_manifest_name.to_string());
    frozen_names.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));

    let mut staged_names: Vec<String> = std::fs::read_dir(staged_root)
        .expect("staged root is readable")
        .map(|entry| {
            entry
                .expect("entry is readable")
                .file_name()
                .to_str()
                .expect("staged names are UTF-8")
                .to_string()
        })
        .collect();
    staged_names.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));

    if staged_names != frozen_names {
        return Hexavalent::False;
    }

    for (name, row) in &declared {
        let bytes = match std::fs::read(staged_root.join(name)) {
            Ok(bytes) => bytes,
            Err(_) => return Hexavalent::False,
        };
        if bytes.len() as u64 != row.bytes || sha256_hex(&bytes) != row.sha256 {
            return Hexavalent::False;
        }
    }

    match std::fs::read(staged_root.join(frozen_manifest_name)) {
        Ok(bytes) if bytes == frozen_manifest_bytes => Hexavalent::True,
        _ => Hexavalent::False,
    }
}

/// The §2 `Inventory` rows of the frozen original manifest.
fn parse_inventory(manifest_bytes: &[u8]) -> BTreeMap<String, Declared> {
    let text = std::str::from_utf8(manifest_bytes).expect("the frozen manifest is UTF-8");
    let mut out = BTreeMap::new();
    let mut in_section = false;
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.starts_with("## ") {
            in_section = line == "## 2. Inventory";
            continue;
        }
        if !in_section || !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.trim().trim_matches('`').trim().to_string())
            .collect();
        if cells.len() != 3 || cells[0] == "Relative path" {
            continue;
        }
        let digits: String = cells[1].chars().filter(|c| *c != ',').collect();
        let Ok(bytes) = digits.parse::<u64>() else {
            continue;
        };
        if cells[2].len() != 64 {
            continue;
        }
        out.insert(
            cells[0].clone(),
            Declared {
                bytes,
                sha256: cells[2].clone(),
            },
        );
    }
    assert!(!out.is_empty(), "the frozen manifest declares an inventory");
    out
}
