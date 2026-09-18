//! [`SessionSink`] — the [`EventSink`] implementation that writes to
//! a [`SessionLog`]'s JSONL file.

use std::io::Write;

use ix_agent_core::{EventSink, SessionEvent};

use crate::log::SessionLog;

/// Append-only event sink backed by a [`SessionLog`].
///
/// Every [`emit`](Self::emit) serializes the event to JSON and
/// appends it as a single line, followed by a flush of the
/// underlying buffered writer. The flush is deliberate: it trades
/// throughput for durability, because the governance-instrument
/// contract assumes events are persistent the moment they're
/// recorded.
///
/// Error handling: `emit` cannot return a `Result` because the
/// [`EventSink`] trait is infallible. Write failures are stored in
/// the sink's internal error slot; callers that care can inspect it
/// via [`SessionSink::last_error`].
pub struct SessionSink<'a> {
    log: &'a SessionLog,
    last_error: Option<String>,
}

impl<'a> SessionSink<'a> {
    pub(crate) fn new(log: &'a SessionLog) -> Self {
        Self {
            log,
            last_error: None,
        }
    }

    /// Returns the most recent error from a failed [`emit`](Self::emit)
    /// call, if any. Cleared implicitly on the next successful emit —
    /// this is a one-level error slot, not an error queue.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Clear the stored error (if any). Use this after handling an
    /// error if you want subsequent `last_error()` calls to reflect
    /// only newer failures.
    pub fn clear_error(&mut self) {
        self.last_error = None;
    }

    /// Attempt to emit an event and return a `Result`. This is the
    /// fallible version of [`EventSink::emit`] — use it when you
    /// need to handle write failures at the call site.
    pub fn try_emit(&mut self, event: SessionEvent) -> Result<(), String> {
        let json = serde_json::to_string(&event).map_err(|e| format!("serialize event: {e}"))?;
        let mut writer = self.log.writer_lock();
        writer
            .write_all(json.as_bytes())
            .map_err(|e| format!("write event: {e}"))?;
        writer
            .write_all(b"\n")
            .map_err(|e| format!("write newline: {e}"))?;
        writer.flush().map_err(|e| format!("flush: {e}"))?;
        Ok(())
    }
}

impl<'a> EventSink for SessionSink<'a> {
    fn emit(&mut self, event: SessionEvent) {
        match self.try_emit(event) {
            Ok(()) => {
                // Successful emit advances the ordinal. The ordinal
                // the caller passed inside the event is preserved
                // verbatim on disk — we just increment our counter
                // so `next_ordinal()` reflects the new state.
                self.log.claim_ordinal();
                self.last_error = None;
            }
            Err(e) => {
                self.last_error = Some(e);
            }
        }
    }

    fn next_ordinal(&self) -> u64 {
        self.log.next_ordinal()
    }
}

// SessionSink is Send + Sync because all mutation goes through
// SessionLog's internal mutexes, and the inner &SessionLog is
// shared-borrow safe.
// NB: the `Mutex<BufWriter<File>>` ensures writer exclusion;
// `SessionSink` itself has no interior mutability beyond its
// `last_error` field which is owned (not shared).
// We do NOT impl Send/Sync manually — the derive is automatic
// because all fields are Send/Sync.
