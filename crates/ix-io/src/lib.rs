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
//! - **protocol**: Common trait for all data sources/sinks

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
