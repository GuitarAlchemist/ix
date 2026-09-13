//! # ix-io
//!
//! Data ingestion and output for the machin ML toolkit.
//!
//! - **watcher**: File system watching (react to data changes)
//! - **csv_io**: CSV read/write to ndarray
//! - **json_io**: JSON streaming and batch
//! - **pipe**: Named pipes (Windows) / FIFO (Unix) for IPC
//! - **tcp**: TCP server/client for data streaming
//! - **http**: HTTP client for REST API data sources
//! - **websocket**: WebSocket for real-time data feeds
//! - **protocol**: `DataSource` / `DataSink`, the synchronous record interface
//!   the backends are pumped through
//!
//! ## Which backends implement the protocol
//!
//! `csv_io` and `json_io` (NDJSON) implement [`protocol::DataSource`] and
//! [`protocol::DataSink`] directly. The `async` backends — `http`, `tcp`,
//! `websocket` — acquire a batch and serve it through
//! [`protocol::BatchSource`]. `pipe`, `watcher` and `trace_bridge` implement
//! neither, and each module doc says why. [`protocol`] carries the full table.

pub mod csv_io;
pub mod error;
pub mod http;
pub mod json_io;
pub mod pipe;
pub mod protocol;
pub mod tcp;
pub mod trace_bridge;
pub mod watcher;
pub mod websocket;
