//! WebSocket client for real-time data streaming.
//!
//! Use case: connect to live data feeds (market data, sensor streams, etc.)
//! and pipe into models in real-time.
//!
//! # Reaching the synchronous protocol
//!
//! Nothing here implements [`DataSource`](crate::protocol::DataSource), for
//! the same reason as [`crate::tcp`]: the reads are `async`. A WebSocket is
//! also unbounded in time — there is no point at which it is exhausted — so
//! the conversion has to name a cutoff. [`collect_ws`] is that cutoff: it
//! takes `n_messages` and returns a [`DataBatch`](crate::protocol::DataBatch),
//! which [`crate::protocol::BatchSource`] then serves through the trait.

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use crate::error::IoError;
use crate::protocol::{DataBatch, DataRecord};

/// Connect to a WebSocket and stream records through a channel.
pub async fn stream_ws(
    url: &str,
) -> Result<
    (
        tokio::sync::mpsc::Receiver<DataRecord>,
        tokio::sync::mpsc::Sender<String>,
    ),
    IoError,
> {
    let (ws_stream, _) = connect_async(url)
        .await
        .map_err(|e| IoError::Connection(format!("WebSocket connection failed: {}", e)))?;

    let (mut write, mut read) = ws_stream.split();
    let (data_tx, data_rx) = tokio::sync::mpsc::channel(100);
    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel::<String>(10);

    // Read task: WS -> channel
    tokio::spawn(async move {
        while let Some(Ok(msg)) = read.next().await {
            let record = match msg {
                Message::Text(text) => {
                    // Try parse as JSON array of f64
                    if let Ok(row) = serde_json::from_str::<Vec<f64>>(&text) {
                        DataRecord::Row(row)
                    } else {
                        DataRecord::Text(text.to_string())
                    }
                }
                Message::Binary(bin) => DataRecord::Bytes(bin.to_vec()),
                _ => continue,
            };
            if data_tx.send(record).await.is_err() {
                break;
            }
        }
    });

    // Write task: channel -> WS
    tokio::spawn(async move {
        while let Some(msg) = cmd_rx.recv().await {
            if write.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    Ok((data_rx, cmd_tx))
}

/// Connect to a WebSocket, collect N messages, return as DataBatch.
pub async fn collect_ws(url: &str, n_messages: usize) -> Result<DataBatch, IoError> {
    let (mut rx, _tx) = stream_ws(url).await?;
    let mut batch = DataBatch::new();

    for _ in 0..n_messages {
        if let Some(record) = rx.recv().await {
            batch.push(record);
        } else {
            break;
        }
    }

    Ok(batch)
}

/// WebSocket data source with reconnection.
pub struct WebSocketSource {
    pub url: String,
    pub subscribe_message: Option<String>,
}

impl WebSocketSource {
    pub fn new(url: &str) -> Self {
        Self {
            url: url.to_string(),
            subscribe_message: None,
        }
    }

    /// Set a message to send after connecting (e.g., subscription request).
    pub fn with_subscribe(mut self, msg: &str) -> Self {
        self.subscribe_message = Some(msg.to_string());
        self
    }

    /// Start streaming with auto-reconnect.
    pub async fn stream_with_reconnect(
        self,
    ) -> Result<tokio::sync::mpsc::Receiver<DataRecord>, IoError> {
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let url = self.url.clone();
        let sub_msg = self.subscribe_message.clone();

        tokio::spawn(async move {
            loop {
                match stream_ws(&url).await {
                    Ok((mut data_rx, cmd_tx)) => {
                        // Send subscription message if configured
                        if let Some(ref msg) = sub_msg {
                            let _ = cmd_tx.send(msg.clone()).await;
                        }

                        // Forward all messages
                        while let Some(record) = data_rx.recv().await {
                            if tx.send(record).await.is_err() {
                                return; // Receiver dropped
                            }
                        }
                    }
                    Err(_) => {
                        // Wait before reconnecting
                        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                    }
                }
            }
        });

        Ok(rx)
    }
}
