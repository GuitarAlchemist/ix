//! TCP server and client for streaming data.
//!
//! Use case: expose a model as a TCP service, or consume
//! streaming data from an external source.

use std::net::SocketAddr;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};

use crate::error::IoError;
use crate::protocol::{DataBatch, DataRecord};

/// Simple TCP data server: accepts connections and sends data as JSON lines.
pub struct TcpDataServer {
    pub addr: SocketAddr,
}

impl TcpDataServer {
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    /// Start serving. Sends batch as JSON lines to each connected client.
    pub async fn serve_batch(&self, batch: &DataBatch) -> Result<(), IoError> {
        let listener = TcpListener::bind(self.addr).await?;

        let (mut stream, _) = listener.accept().await?;

        for record in &batch.records {
            if let DataRecord::Row(row) = record {
                let json =
                    serde_json::to_string(row).map_err(|e| IoError::Connection(e.to_string()))?;
                stream.write_all(json.as_bytes()).await?;
                stream.write_all(b"\n").await?;
            }
        }

        stream.flush().await?;
        Ok(())
    }

    /// Start a persistent server that calls handler for each connection.
    pub async fn serve<F, Fut>(addr: SocketAddr, handler: F) -> Result<(), IoError>
    where
        F: Fn(TcpStream) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), IoError>> + Send + 'static,
    {
        let listener = TcpListener::bind(addr).await?;

        loop {
            let (stream, _) = listener.accept().await?;
            let handler = &handler;
            // Note: in production you'd spawn this
            handler(stream).await?;
        }
    }
}

/// TCP data client: connect and receive data as JSON lines.
pub struct TcpDataClient;

impl TcpDataClient {
    /// Connect to a TCP server and read all data as a DataBatch.
    pub async fn read_batch(addr: &str) -> Result<DataBatch, IoError> {
        let stream = TcpStream::connect(addr).await?;
        let reader = BufReader::new(stream);
        let mut lines = reader.lines();
        let mut batch = DataBatch::new();

        while let Some(line) = lines.next_line().await? {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            // Try parsing as array of f64
            if let Ok(row) = serde_json::from_str::<Vec<f64>>(&line) {
                batch.push(DataRecord::Row(row));
            } else {
                batch.push(DataRecord::Text(line));
            }
        }

        Ok(batch)
    }

    /// Stream data: connect and yield records one at a time via channel.
    pub async fn stream(addr: &str) -> Result<tokio::sync::mpsc::Receiver<DataRecord>, IoError> {
        let stream = TcpStream::connect(addr).await?;
        let reader = BufReader::new(stream);
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        tokio::spawn(async move {
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                let record = if let Ok(row) = serde_json::from_str::<Vec<f64>>(&line) {
                    DataRecord::Row(row)
                } else {
                    DataRecord::Text(line)
                };
                if tx.send(record).await.is_err() {
                    break;
                }
            }
        });

        Ok(rx)
    }
}

/// Quick utility: send a DataBatch over TCP to a given address.
pub async fn send_batch(addr: &str, batch: &DataBatch) -> Result<(), IoError> {
    let mut stream = TcpStream::connect(addr).await?;

    for record in &batch.records {
        if let DataRecord::Row(row) = record {
            let json =
                serde_json::to_string(row).map_err(|e| IoError::Connection(e.to_string()))?;
            stream.write_all(json.as_bytes()).await?;
            stream.write_all(b"\n").await?;
        }
    }

    stream.flush().await?;
    Ok(())
}
