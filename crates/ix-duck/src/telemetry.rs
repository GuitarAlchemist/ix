//! Shared telemetry-ingestion plumbing for the analyst **lenses** (routing, loops,
//! ood, chatbot, maintain).
//!
//! Every lens reads JSON/JSONL telemetry off disk before it gets to its real work,
//! and they all owe the same contract: **an absent directory degrades to "no signal"
//! (skip), but a present-but-unreadable directory fails closed** (a broken mount or a
//! permissions error must never be silently read as "no telemetry"). That invariant —
//! which `maintain::evaluate` trusts when it fuses the lenses — used to live in four
//! byte-identical copies (`routing::eval_files`, `loops::ledger_files`,
//! `ood::embedding_files`, plus the flat half of `chatbot::collect`). Here it lives
//! once, with one test.
//!
//! Each lens keeps its own public `*Error` type (the deepest lens, `maintain`,
//! distinguishes *which* lens failed — guardrail vs convergence vs drift — via
//! distinct `From` impls), so [`LensError`] is `pub(crate)` and converts into each.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

/// Ingestion error: directory I/O vs DuckDB. `pub(crate)` — each lens's public
/// `*Error` adds `From<LensError>`, so a lens `?`-propagates this into its own type
/// and `maintain` still sees per-lens error context.
#[derive(Debug)]
pub(crate) enum LensError {
    Io(std::io::Error),
    Duck(duckdb::Error),
}

impl std::fmt::Display for LensError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LensError::Io(e) => write!(f, "telemetry I/O error: {e}"),
            LensError::Duck(e) => write!(f, "duckdb error: {e}"),
        }
    }
}
impl std::error::Error for LensError {}
impl From<std::io::Error> for LensError {
    fn from(e: std::io::Error) -> Self {
        LensError::Io(e)
    }
}
impl From<duckdb::Error> for LensError {
    fn from(e: duckdb::Error) -> Self {
        LensError::Duck(e)
    }
}

/// Files in `dir` whose file name satisfies `keep`, sorted. **Absent dir → empty
/// (skip); any other read error on a present dir → surfaced (fail-closed).**
///
/// This is the flat, one-level scan shared by the routing/loops/ood lenses.
/// (`chatbot` walks a two-level `golden-traces/*/leaf` corpus and keeps its own
/// nested enumerator; it shares only [`sql_list`] and [`fnv1a64`].)
pub(crate) fn collect_dir(
    dir: &Path,
    keep: impl Fn(&str) -> bool,
) -> Result<Vec<PathBuf>, LensError> {
    let rd = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    };
    let mut out = Vec::new();
    for entry in rd {
        let p = entry?.path();
        if let Some(n) = p.file_name().and_then(|n| n.to_str()) {
            if keep(n) {
                out.push(p);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// Render paths as a DuckDB list literal `['p1', 'p2', …]`, POSIX-slashed and
/// single-quote-escaped. Paths never contain `://` (validated by the caller).
pub(crate) fn sql_list(paths: &[PathBuf]) -> String {
    let items: Vec<String> = paths
        .iter()
        .map(|p| {
            let s = p.to_string_lossy().replace('\\', "/").replace('\'', "''");
            format!("'{s}'")
        })
        .collect();
    format!("[{}]", items.join(", "))
}

/// FNV-1a (64-bit) over a byte stream, formatted as `fnv1a64:{:016x}`. The one
/// content-hash used for the maintain-gate's evidence: `chatbot::baseline_hash`
/// feeds it the joined `k=v;` baseline pairs, `maintain` feeds it file bytes — both
/// must stay byte-comparable, so they share this definition.
pub(crate) fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn absent_dir_degrades_to_empty() {
        let missing = Path::new("definitely/not/a/real/dir/xyzzy");
        let got = collect_dir(missing, |_| true).unwrap();
        assert!(got.is_empty(), "absent dir must skip, not error");
    }

    #[test]
    fn collect_dir_filters_and_sorts() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["b.jsonl", "a.jsonl", "skip.txt"] {
            std::fs::File::create(dir.path().join(name)).unwrap();
        }
        let got = collect_dir(dir.path(), |n| n.ends_with(".jsonl")).unwrap();
        let names: Vec<_> = got
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.jsonl", "b.jsonl"], "filtered + sorted");
    }

    #[test]
    fn sql_list_escapes_and_posix_slashes() {
        let paths = [PathBuf::from(r"C:\a\b.json"), PathBuf::from("d/e'f.json")];
        let got = sql_list(&paths);
        assert_eq!(got, "['C:/a/b.json', 'd/e''f.json']");
        assert_eq!(sql_list(&[]), "[]");
    }

    #[test]
    fn fnv1a64_is_stable_and_prefixed() {
        // Known FNV-1a 64-bit vector: empty input is the offset basis.
        assert_eq!(fnv1a64(std::iter::empty()), "fnv1a64:cbf29ce484222325");
        // Same bytes from different sources hash identically (the cross-lens contract).
        let from_str = fnv1a64(b"abc".iter().copied());
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(b"abc").unwrap();
        let from_file = fnv1a64(std::fs::read(tmp.path()).unwrap());
        assert_eq!(from_str, from_file);
    }
}
