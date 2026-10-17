//! Loops on disk: what loops propose, and the library a person accepts them into.
//!
//! `<root>/proposals/<id>.json` holds one proposal; `<root>/library/` holds accepted loops, as
//! `<name>.lua` or `<name>.pipeline.json`. A loop can propose, list the library and load from it.
//! Only [`Store::accept`] moves a proposal into the library, and no loop can reach it: the
//! `ix-loop accept` command calls it, for a person.

use serde_json::{json, Value as Json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Bytes a proposal's name may hold.
pub const MAX_NAME_BYTES: usize = 40;
/// Bytes a proposal's body may hold.
pub const MAX_BODY_BYTES: usize = 256 << 10;
/// Bytes a proposal's note may hold.
pub const MAX_NOTE_BYTES: usize = 4 << 10;

/// What a loop proposes.
#[derive(Debug, Clone, PartialEq)]
pub struct Draft {
    /// 1 to [`MAX_NAME_BYTES`] ASCII letters, digits or `-`.
    pub name: String,
    /// `lua` or `pipeline`.
    pub kind: String,
    pub body: String,
    pub note: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Store {
    root: PathBuf,
}

static NEXT: AtomicU32 = AtomicU32::new(0);

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn valid_name(name: &str) -> bool {
    (1..=MAX_NAME_BYTES).contains(&name.len())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
}

fn valid_id(id: &str) -> bool {
    match id.strip_prefix('p').and_then(|rest| rest.split_once('-')) {
        Some((ms, n)) => {
            !ms.is_empty()
                && !n.is_empty()
                && ms.chars().all(|c| c.is_ascii_digit())
                && n.chars().all(|c| c.is_ascii_digit())
        }
        None => false,
    }
}

/// A library file name: a valid name and `.lua` or `.pipeline.json`.
fn valid_file(file: &str) -> bool {
    let stem = file
        .strip_suffix(".pipeline.json")
        .or_else(|| file.strip_suffix(".lua"));
    stem.is_some_and(valid_name)
}

impl Store {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// `IX_LOOP_STORE`, else `~/.cache/ix-loop`.
    pub fn default_root() -> PathBuf {
        if let Some(p) = std::env::var_os("IX_LOOP_STORE") {
            return PathBuf::from(p);
        }
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        home.join(".cache").join("ix-loop")
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Writes a proposal and returns its id. It changes nothing in the library.
    ///
    /// The body and the note are bounded before the document is built. Escaping grows a byte
    /// to at most six (`\u001f`), so a proposal file stays under 2 MB.
    // @ai:invariant a proposal file written by propose is under 2 MB [T:test conf:0.9 src:store::tests::a_proposal_is_bounded_before_it_is_written]
    pub fn propose(&self, draft: &Draft, proposed_by: &str) -> Result<String, String> {
        if !valid_name(&draft.name) {
            return Err(format!(
                "propose: a name is 1 to {MAX_NAME_BYTES} letters, digits or -"
            ));
        }
        if draft.kind != "lua" && draft.kind != "pipeline" {
            return Err("propose: kind is \"lua\" or \"pipeline\"".into());
        }
        if draft.body.len() > MAX_BODY_BYTES {
            return Err(format!(
                "propose: a body holds at most {MAX_BODY_BYTES} bytes"
            ));
        }
        if draft
            .note
            .as_ref()
            .is_some_and(|n| n.len() > MAX_NOTE_BYTES)
        {
            return Err(format!(
                "propose: a note holds at most {MAX_NOTE_BYTES} bytes"
            ));
        }
        let dir = self.root.join("proposals");
        std::fs::create_dir_all(&dir).map_err(|e| format!("propose: {e}"))?;
        let ts_ms = now_ms();
        let id = format!("p{ts_ms}-{}", NEXT.fetch_add(1, Ordering::Relaxed));
        let doc = json!({
            "id": id, "name": draft.name, "kind": draft.kind, "body": draft.body,
            "note": draft.note, "proposed_by": proposed_by, "ts_ms": ts_ms,
        });
        let text = serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())?;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.join(format!("{id}.json")))
            .and_then(|mut f| f.write_all(text.as_bytes()))
            .map_err(|e| format!("propose: {e}"))?;
        Ok(id)
    }

    /// The pending proposals, oldest first.
    pub fn proposals(&self) -> Vec<Json> {
        let mut out: Vec<Json> = std::fs::read_dir(self.root.join("proposals"))
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .filter_map(|t| serde_json::from_str::<Json>(&t).ok())
            .collect();
        out.sort_by_key(|d| (d["ts_ms"].as_u64(), d["id"].as_str().map(String::from)));
        out
    }

    /// The library's file names, sorted.
    pub fn library(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.root.join("library"))
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| valid_file(n))
            .collect();
        names.sort();
        names
    }

    /// One library file's text, by its file name.
    pub fn load(&self, file: &str) -> Option<String> {
        if !valid_file(file) {
            return None;
        }
        std::fs::read_to_string(self.root.join("library").join(file)).ok()
    }

    /// Moves proposal `id` into the library and returns the library file. For a person only.
    pub fn accept(&self, id: &str) -> Result<PathBuf, String> {
        if !valid_id(id) {
            return Err(format!("accept: not a proposal id: {id}"));
        }
        let path = self.root.join("proposals").join(format!("{id}.json"));
        let text =
            std::fs::read_to_string(&path).map_err(|e| format!("accept: no proposal {id}: {e}"))?;
        let doc: Json = serde_json::from_str(&text).map_err(|e| format!("accept: {e}"))?;
        let name = doc["name"].as_str().filter(|n| valid_name(n));
        let ext = match doc["kind"].as_str() {
            Some("lua") => "lua",
            Some("pipeline") => "pipeline.json",
            _ => return Err(format!("accept: proposal {id} has no valid kind")),
        };
        let name = name.ok_or_else(|| format!("accept: proposal {id} has no valid name"))?;
        let lib = self.root.join("library");
        std::fs::create_dir_all(&lib).map_err(|e| format!("accept: {e}"))?;
        let file = lib.join(format!("{name}.{ext}"));
        std::fs::write(&file, doc["body"].as_str().unwrap_or(""))
            .map_err(|e| format!("accept: {e}"))?;
        std::fs::remove_file(&path).map_err(|e| format!("accept: {e}"))?;
        Ok(file)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_proposal_reaches_the_library_only_through_accept() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let draft = Draft {
            name: "count-up".into(),
            kind: "lua".into(),
            body: "return 1".into(),
            note: None,
        };
        let id = store.propose(&draft, "test").unwrap();
        assert!(store.library().is_empty());
        assert_eq!(store.proposals().len(), 1);
        store.accept(&id).unwrap();
        assert_eq!(store.library(), ["count-up.lua"]);
        assert_eq!(store.load("count-up.lua").as_deref(), Some("return 1"));
        assert!(store.proposals().is_empty());
    }

    #[test]
    fn names_and_ids_cannot_leave_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let bad = Draft {
            name: "../x".into(),
            kind: "lua".into(),
            body: String::new(),
            note: None,
        };
        assert!(store.propose(&bad, "test").is_err());
        assert!(store.accept("../x").is_err());
        assert_eq!(store.load("../Cargo.toml"), None);
    }

    #[test]
    fn a_proposal_is_bounded_before_it_is_written() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let draft = |body: String, note: Option<String>| Draft {
            name: "big".into(),
            kind: "lua".into(),
            body,
            note,
        };
        let over_body = draft("x".repeat(MAX_BODY_BYTES + 1), None);
        let over_note = draft(String::new(), Some("x".repeat(MAX_NOTE_BYTES + 1)));
        assert!(store
            .propose(&over_body, "test")
            .unwrap_err()
            .contains("body"));
        assert!(store
            .propose(&over_note, "test")
            .unwrap_err()
            .contains("note"));
        assert!(store.proposals().is_empty());

        // At the limits, with every byte escaped to six.
        let worst = draft(
            "\u{1}".repeat(MAX_BODY_BYTES),
            Some("\u{1}".repeat(MAX_NOTE_BYTES)),
        );
        let id = store.propose(&worst, "test").unwrap();
        let file = dir.path().join("proposals").join(format!("{id}.json"));
        let size = std::fs::metadata(file).unwrap().len();
        assert!(size > 6 * MAX_BODY_BYTES as u64, "not escaped: {size}");
        assert!(size < 2 << 20, "{size}");
    }
}
