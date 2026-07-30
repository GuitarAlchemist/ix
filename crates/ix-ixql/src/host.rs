//! The executor's contact with the outside world: reads, writes, and the clock.
//!
//! Everything effectful goes through [`Host`] so a pipeline can be run offline
//! and deterministically. [`MemoryHost`] is that offline form — a fixed clock
//! and an in-memory filesystem — and is what the fixture tests use; [`FsHost`]
//! is the real one, rooted at a directory so a pipeline cannot write outside
//! the repo it was launched in.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use chrono::{DateTime, TimeZone, Utc};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HostError {
    #[error("{path}: {message}")]
    Io { path: String, message: String },
    #[error("{path} is not valid JSON: {message}")]
    Parse { path: String, message: String },
    #[error("{path} escapes the pipeline root")]
    Escapes { path: String },
}

/// Reads, writes, and time — the only ways an IXQL run touches anything.
pub trait Host: Send + Sync {
    /// `Ok(None)` means "absent", which is what makes `→ default(…)` work.
    /// Absence is not an error; unreadable or malformed content is.
    fn read(&self, path: &str) -> Result<Option<Value>, HostError>;

    fn write(&self, path: &str, value: &Value) -> Result<(), HostError>;

    fn now(&self) -> DateTime<Utc>;
}

/// In-memory host with a frozen clock.
pub struct MemoryHost {
    files: Mutex<BTreeMap<String, Value>>,
    now: DateTime<Utc>,
}

impl MemoryHost {
    /// Build a host whose clock is frozen at the given UTC instant.
    pub fn at(now: DateTime<Utc>) -> Self {
        Self {
            files: Mutex::new(BTreeMap::new()),
            now,
        }
    }

    /// A fixed, arbitrary instant — `2026-01-02T03:04:05Z` — for tests that
    /// only need the clock to be stable, not to be a particular value.
    pub fn frozen() -> Self {
        Self::at(Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap())
    }

    /// Seed a file the pipeline will read.
    pub fn seed(&self, path: &str, value: Value) {
        self.files
            .lock()
            .expect("MemoryHost lock")
            .insert(normalize(path), value);
    }

    /// Everything written so far, keyed by path.
    pub fn files(&self) -> BTreeMap<String, Value> {
        self.files.lock().expect("MemoryHost lock").clone()
    }
}

impl Host for MemoryHost {
    fn read(&self, path: &str) -> Result<Option<Value>, HostError> {
        Ok(self
            .files
            .lock()
            .expect("MemoryHost lock")
            .get(&normalize(path))
            .cloned())
    }

    fn write(&self, path: &str, value: &Value) -> Result<(), HostError> {
        self.files
            .lock()
            .expect("MemoryHost lock")
            .insert(normalize(path), value.clone());
        Ok(())
    }

    fn now(&self) -> DateTime<Utc> {
        self.now
    }
}

/// Real filesystem, confined to `root`.
pub struct FsHost {
    root: PathBuf,
}

impl FsHost {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Reject `..` and absolute paths before touching the disk: a pipeline
    /// builds its own write paths by interpolation, so a value read from
    /// elsewhere could otherwise steer a write out of the repo.
    fn resolve(&self, path: &str) -> Result<PathBuf, HostError> {
        let candidate = Path::new(path);
        let unsafe_component = candidate.components().any(|c| {
            matches!(
                c,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        });
        if unsafe_component {
            return Err(HostError::Escapes {
                path: path.to_string(),
            });
        }
        Ok(self.root.join(candidate))
    }
}

impl Host for FsHost {
    fn read(&self, path: &str) -> Result<Option<Value>, HostError> {
        let full = self.resolve(path)?;
        let text = match std::fs::read_to_string(&full) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(HostError::Io {
                    path: path.to_string(),
                    message: e.to_string(),
                })
            }
        };
        serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| HostError::Parse {
                path: path.to_string(),
                message: e.to_string(),
            })
    }

    fn write(&self, path: &str, value: &Value) -> Result<(), HostError> {
        let full = self.resolve(path)?;
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).map_err(|e| HostError::Io {
                path: path.to_string(),
                message: e.to_string(),
            })?;
        }
        let mut text = serde_json::to_string_pretty(value).map_err(|e| HostError::Parse {
            path: path.to_string(),
            message: e.to_string(),
        })?;
        text.push('\n');
        std::fs::write(&full, text).map_err(|e| HostError::Io {
            path: path.to_string(),
            message: e.to_string(),
        })
    }

    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

fn normalize(path: &str) -> String {
    path.replace('\\', "/")
}

/// Translate a .NET-style format string — the spelling `now_utc(…)` uses in
/// Demerzel's pipelines — into a `chrono` format string.
///
/// Only the tokens the pipelines actually use are translated; anything else is
/// passed through as a literal, with `%` escaped so a stray percent sign in the
/// author's text cannot become a format directive.
pub fn dotnet_format_to_chrono(fmt: &str) -> String {
    const TOKENS: [(&str, &str); 8] = [
        ("yyyy", "%Y"),
        ("MM", "%m"),
        ("dd", "%d"),
        ("HH", "%H"),
        ("mm", "%M"),
        ("ss", "%S"),
        ("fff", "%3f"),
        ("%", "%%"),
    ];

    let mut out = String::with_capacity(fmt.len());
    let mut rest = fmt;
    'outer: while !rest.is_empty() {
        for (token, replacement) in TOKENS {
            if let Some(tail) = rest.strip_prefix(token) {
                out.push_str(replacement);
                rest = tail;
                continue 'outer;
            }
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn absent_file_reads_as_none_so_default_can_fire() {
        let host = MemoryHost::frozen();
        assert_eq!(None, host.read("state/missing.json").unwrap());
    }

    #[test]
    fn writes_are_readable_back_and_path_separators_normalise() {
        let host = MemoryHost::frozen();
        host.write("state\\a\\b.json", &json!({"k": 1})).unwrap();
        assert_eq!(Some(json!({"k": 1})), host.read("state/a/b.json").unwrap());
    }

    #[test]
    fn the_verdict_id_format_from_qa_architect_cycle_translates() {
        assert_eq!(
            "%Y-%m-%dT%H-%M-%SZ",
            dotnet_format_to_chrono("yyyy-MM-ddTHH-mm-ssZ")
        );
    }

    #[test]
    fn minutes_and_months_are_not_confused() {
        // `MM` is the month, `mm` the minute; a case-insensitive replacement
        // would silently emit the month twice.
        let stamp = MemoryHost::frozen()
            .now()
            .format(&dotnet_format_to_chrono("yyyy-MM-ddTHH-mm-ssZ"))
            .to_string();
        assert_eq!("2026-01-02T03-04-05Z", stamp);
    }

    #[test]
    fn a_literal_percent_cannot_become_a_directive() {
        let out = MemoryHost::frozen()
            .now()
            .format(&dotnet_format_to_chrono("100% dd"))
            .to_string();
        assert_eq!("100% 02", out);
    }

    #[test]
    fn fs_host_refuses_to_write_outside_its_root() {
        let host = FsHost::new(std::env::temp_dir().join("ix-ixql-root-test"));
        let err = host.write("../escape.json", &json!(1)).unwrap_err();
        assert!(matches!(err, HostError::Escapes { .. }));
    }
}
