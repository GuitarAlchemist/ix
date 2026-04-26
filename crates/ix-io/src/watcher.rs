//! File system watcher — react to data file changes.
//!
//! Use case: watch a CSV/JSON file, re-trigger model training when data changes.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use crate::error::IoError;

/// Events emitted by the file watcher.
#[derive(Debug, Clone)]
pub enum FileEvent {
    Created(PathBuf),
    Modified(PathBuf),
    Removed(PathBuf),
    Renamed { from: PathBuf, to: PathBuf },
}

/// Synchronous file watcher that yields events via channel.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    receiver: mpsc::Receiver<Result<Event, notify::Error>>,
}

impl FileWatcher {
    /// Start watching a path (file or directory).
    pub fn new(path: &Path, recursive: bool) -> Result<Self, IoError> {
        let (tx, rx) = mpsc::channel();

        let mut watcher = RecommendedWatcher::new(
            move |res| {
                let _ = tx.send(res);
            },
            Config::default(),
        )?;

        let mode = if recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };

        watcher.watch(path, mode)?;

        Ok(Self {
            _watcher: watcher,
            receiver: rx,
        })
    }

    /// Block until the next file event, with timeout.
    pub fn next_event(&self, timeout: Duration) -> Option<FileEvent> {
        match self.receiver.recv_timeout(timeout) {
            Ok(Ok(event)) => translate_event(event),
            _ => None,
        }
    }

    /// Non-blocking: get all pending events.
    pub fn pending_events(&self) -> Vec<FileEvent> {
        let mut events = Vec::new();
        while let Ok(Ok(event)) = self.receiver.try_recv() {
            if let Some(fe) = translate_event(event) {
                events.push(fe);
            }
        }
        events
    }
}

fn translate_event(event: Event) -> Option<FileEvent> {
    let path = event.paths.first()?.clone();

    match event.kind {
        EventKind::Create(_) => Some(FileEvent::Created(path)),
        EventKind::Modify(_) => Some(FileEvent::Modified(path)),
        EventKind::Remove(_) => Some(FileEvent::Removed(path)),
        _ => None,
    }
}

/// Watch a file and call a callback on every modification.
/// Blocks the current thread. Use in a spawned thread.
pub fn watch_file_loop<F>(path: &Path, debounce: Duration, mut callback: F) -> Result<(), IoError>
where
    F: FnMut(&Path),
{
    let watcher = FileWatcher::new(path, false)?;
    let mut last_trigger = std::time::Instant::now() - debounce;

    loop {
        if let Some(event) = watcher.next_event(Duration::from_secs(1)) {
            match event {
                FileEvent::Modified(p) | FileEvent::Created(p) => {
                    let now = std::time::Instant::now();
                    if now.duration_since(last_trigger) >= debounce {
                        callback(&p);
                        last_trigger = now;
                    }
                }
                FileEvent::Removed(_) => {
                    break; // File deleted, stop watching
                }
                _ => {}
            }
        }
    }

    Ok(())
}

/// Async version: watch a directory and send events through a tokio channel.
pub async fn watch_async(
    path: PathBuf,
    recursive: bool,
) -> Result<tokio::sync::mpsc::Receiver<FileEvent>, IoError> {
    let (tx, rx) = tokio::sync::mpsc::channel(100);

    let (notify_tx, notify_rx) = mpsc::channel();
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            let _ = notify_tx.send(res);
        },
        Config::default(),
    )?;

    let mode = if recursive {
        RecursiveMode::Recursive
    } else {
        RecursiveMode::NonRecursive
    };
    watcher.watch(&path, mode)?;

    // Bridge sync notify -> async tokio channel
    tokio::spawn(async move {
        let _watcher = watcher; // Keep alive
        loop {
            match notify_rx.recv_timeout(Duration::from_millis(100)) {
                Ok(Ok(event)) => {
                    if let Some(fe) = translate_event(event) {
                        if tx.send(fe).await.is_err() {
                            break;
                        }
                    }
                }
                Ok(Err(_)) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
            }
        }
    });

    Ok(rx)
}
