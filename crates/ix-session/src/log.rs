//! [`SessionLog`] — the top-level handle to a JSONL-backed session
//! event log.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use ix_agent_core::SessionEvent;

use crate::errors::{ReloadError, SessionError};
use crate::sink::SessionSink;

/// A JSONL-backed append-only session event log.
///
/// Owns the write side of the file via a `Mutex<BufWriter<File>>`.
/// Ordinal numbering is tracked in memory, initialized from the
/// existing file on open. Clone is deliberately not implemented —
/// each log has exclusive write ownership of its file.
pub struct SessionLog {
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
    /// Next ordinal to assign. Initialized from the file on open;
    /// incremented (under the writer mutex) on every successful
    /// emit.
    next_ordinal: Mutex<u64>,
    /// Errors from the most recent reload, if any. Consumers can
    /// inspect these via [`SessionLog::reload_errors`] to decide
    /// whether the file is usable.
    reload_errors: Vec<ReloadError>,
}

impl SessionLog {
    /// Open an existing session log or create a new one.
    ///
    /// If `path` exists, the file is scanned line by line to count
    /// events and determine the starting ordinal. Corrupted lines
    /// are recorded in [`reload_errors`](Self::reload_errors) but
    /// do NOT abort the open — consumers decide how to react.
    ///
    /// If `path` does not exist, the parent directory is created
    /// (if missing) and a fresh file is opened in append mode.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SessionError> {
        let path = path.as_ref().to_path_buf();

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|source| SessionError::Directory {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }

        // Count existing events + collect reload errors by scanning
        // the file (if it exists) before opening the writer.
        let (starting_ordinal, reload_errors) = if path.exists() {
            Self::scan_existing(&path)?
        } else {
            (0, Vec::new())
        };

        // Open the writer in append mode. O_CREAT | O_APPEND gives us
        // atomic appends on POSIX and ordered appends on Windows.
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| SessionError::OpenFile {
                path: path.clone(),
                source,
            })?;

        Ok(Self {
            path,
            writer: Mutex::new(BufWriter::new(file)),
            next_ordinal: Mutex::new(starting_ordinal),
            reload_errors,
        })
    }

    /// The filesystem path backing this log.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Current next ordinal. Thread-safe but not stable under
    /// concurrent writes — callers that want a stable snapshot
    /// should call `next_ordinal` on their [`SessionSink`] inside
    /// a single emit operation.
    pub fn next_ordinal(&self) -> u64 {
        *self
            .next_ordinal
            .lock()
            .expect("next_ordinal mutex poisoned")
    }

    /// Errors captured during the most recent reload. Empty means
    /// every line in the on-disk file parsed cleanly.
    pub fn reload_errors(&self) -> &[ReloadError] {
        &self.reload_errors
    }

    /// Build a [`SessionSink`] bound to this log. The sink borrows
    /// the log for its lifetime; multiple sinks can exist
    /// concurrently because they all funnel through the same
    /// mutex.
    pub fn sink(&self) -> SessionSink<'_> {
        SessionSink::new(self)
    }

    /// Iterate over all events currently in the on-disk file.
    ///
    /// Each iterator item is `Result<SessionEvent, ReloadError>`
    /// because individual lines may be corrupt. The iterator
    /// opens a fresh read handle on the file, so it sees events
    /// up to the point the buffered writer has flushed (i.e., not
    /// events currently sitting in the `BufWriter`'s in-memory
    /// buffer). Call [`SessionLog::flush`] before iterating if
    /// you need to read your own recent writes.
    pub fn events(&self) -> Result<EventIter, SessionError> {
        let file = File::open(&self.path).map_err(|source| SessionError::Read {
            path: self.path.clone(),
            source,
        })?;
        Ok(EventIter {
            reader: BufReader::new(file),
            line_number: 0,
        })
    }

    /// Flush the underlying `BufWriter` so recent emits are
    /// visible to readers. The sink flushes on every emit by
    /// default, so this is usually a no-op — call it explicitly
    /// only if you need to synchronize with a separate reader
    /// thread.
    pub fn flush(&self) -> Result<(), SessionError> {
        let mut writer = self.writer.lock().expect("writer mutex poisoned");
        writer.flush().map_err(|source| SessionError::Write {
            path: self.path.clone(),
            source,
        })
    }

    /// INTERNAL: lock the writer for an append operation. Only the
    /// [`SessionSink`] uses this.
    pub(crate) fn writer_lock(&self) -> MutexGuard<'_, BufWriter<File>> {
        self.writer.lock().expect("writer mutex poisoned")
    }

    /// INTERNAL: assign and increment the next ordinal. Returns the
    /// value to use for the current emit.
    pub(crate) fn claim_ordinal(&self) -> u64 {
        let mut n = self
            .next_ordinal
            .lock()
            .expect("next_ordinal mutex poisoned");
        let value = *n;
        *n = n.saturating_add(1);
        value
    }

    /// Scan an existing log file once to compute the starting
    /// ordinal and collect reload errors. Returns `(ordinal,
    /// errors)`.
    fn scan_existing(path: &Path) -> Result<(u64, Vec<ReloadError>), SessionError> {
        let file = File::open(path).map_err(|source| SessionError::Read {
            path: path.to_path_buf(),
            source,
        })?;
        let reader = BufReader::new(file);
        let mut count: u64 = 0;
        let mut errors: Vec<ReloadError> = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line_num = idx + 1;
            let text = match line {
                Ok(t) => t,
                Err(_) => {
                    errors.push(ReloadError::InvalidUtf8 { line: line_num });
                    continue;
                }
            };
            if text.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<SessionEvent>(&text) {
                Ok(_) => count += 1,
                Err(source) => {
                    errors.push(ReloadError::BadJson {
                        line: line_num,
                        source,
                    });
                }
            }
        }
        Ok((count, errors))
    }
}

/// Iterator over events on disk, lazily parsing one line at a time.
pub struct EventIter {
    reader: BufReader<File>,
    line_number: usize,
}

impl Iterator for EventIter {
    type Item = Result<SessionEvent, ReloadError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut buf = String::new();
            match self.reader.read_line(&mut buf) {
                Ok(0) => return None, // EOF
                Ok(_) => {
                    self.line_number += 1;
                    let trimmed = buf.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    return Some(
                        serde_json::from_str::<SessionEvent>(trimmed).map_err(|source| {
                            ReloadError::BadJson {
                                line: self.line_number,
                                source,
                            }
                        }),
                    );
                }
                Err(_) => {
                    self.line_number += 1;
                    return Some(Err(ReloadError::InvalidUtf8 {
                        line: self.line_number,
                    }));
                }
            }
        }
    }
}
