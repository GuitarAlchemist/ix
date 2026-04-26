//! Named pipes for inter-process communication.
//!
//! - Windows: CreateNamedPipe (\\.\pipe\ix_xxx)
//! - Unix: mkfifo
//!
//! Use case: connect machin to external processes (Python scripts, other tools)
//! for real-time data exchange without network overhead.

use crate::error::IoError;
use crate::protocol::{DataBatch, DataRecord};

/// Named pipe server configuration.
pub struct PipeConfig {
    pub name: String,
    pub buffer_size: usize,
}

impl PipeConfig {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            buffer_size: 65536,
        }
    }

    /// Get the platform-specific pipe path.
    pub fn path(&self) -> String {
        if cfg!(windows) {
            format!(r"\\.\pipe\ix_{}", self.name)
        } else {
            format!("/tmp/ix_{}", self.name)
        }
    }
}

/// Write data to a named pipe (synchronous).
/// Creates the pipe if it doesn't exist (Unix only for now).
#[cfg(unix)]
pub fn write_to_pipe(config: &PipeConfig, data: &[u8]) -> Result<(), IoError> {
    use std::io::Write;
    use std::path::Path;

    let path = config.path();
    let pipe_path = Path::new(&path);

    // Create FIFO if it doesn't exist
    if !pipe_path.exists() {
        std::process::Command::new("mkfifo")
            .arg(&path)
            .status()
            .map_err(|e| IoError::Pipe(format!("Failed to create FIFO: {}", e)))?;
    }

    let mut file = std::fs::OpenOptions::new().write(true).open(pipe_path)?;
    file.write_all(data)?;
    file.flush()?;
    Ok(())
}

/// Write data to a named pipe (Windows).
#[cfg(windows)]
pub fn write_to_pipe(config: &PipeConfig, data: &[u8]) -> Result<(), IoError> {
    use std::fs::OpenOptions;
    use std::io::Write;

    let path = config.path();
    // On Windows, we open the pipe as a regular file for client-side writes
    let mut file = OpenOptions::new()
        .write(true)
        .open(&path)
        .map_err(|e| IoError::Pipe(format!("Failed to open pipe {}: {}", path, e)))?;
    file.write_all(data)?;
    file.flush()?;
    Ok(())
}

/// Read data from a named pipe (synchronous, blocking).
pub fn read_from_pipe(config: &PipeConfig, buf_size: usize) -> Result<Vec<u8>, IoError> {
    use std::io::Read;

    let path = config.path();
    let mut file = std::fs::File::open(&path)
        .map_err(|e| IoError::Pipe(format!("Failed to open pipe {}: {}", path, e)))?;

    let mut buffer = vec![0u8; buf_size];
    let n = file.read(&mut buffer)?;
    buffer.truncate(n);
    Ok(buffer)
}

/// Write a DataBatch as JSON lines to a named pipe.
pub fn write_batch_to_pipe(config: &PipeConfig, batch: &DataBatch) -> Result<(), IoError> {
    let mut lines = String::new();
    for record in &batch.records {
        if let DataRecord::Row(row) = record {
            let json = serde_json::to_string(row)
                .map_err(|e| IoError::Pipe(format!("Serialization error: {}", e)))?;
            lines.push_str(&json);
            lines.push('\n');
        }
    }
    write_to_pipe(config, lines.as_bytes())
}

/// Async named pipe server using tokio (Windows).
#[cfg(windows)]
pub mod async_pipe {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::windows::named_pipe::{NamedPipeServer, ServerOptions};

    use super::PipeConfig;
    use crate::error::IoError;

    /// Create an async named pipe server.
    pub async fn create_server(config: &PipeConfig) -> Result<NamedPipeServer, IoError> {
        let path = config.path();
        ServerOptions::new()
            .first_pipe_instance(true)
            .create(&path)
            .map_err(|e| IoError::Pipe(format!("Failed to create pipe server: {}", e)))
    }

    /// Read a message from the pipe.
    pub async fn read_message(
        server: &mut NamedPipeServer,
        buf_size: usize,
    ) -> Result<Vec<u8>, IoError> {
        server
            .connect()
            .await
            .map_err(|e| IoError::Pipe(format!("Client connect failed: {}", e)))?;

        let mut buffer = vec![0u8; buf_size];
        let n = server.read(&mut buffer).await?;
        buffer.truncate(n);
        Ok(buffer)
    }

    /// Write a message to the pipe.
    pub async fn write_message(server: &mut NamedPipeServer, data: &[u8]) -> Result<(), IoError> {
        server.write_all(data).await?;
        server.flush().await?;
        Ok(())
    }
}

/// Async named pipe using Unix FIFO with tokio.
#[cfg(unix)]
pub mod async_pipe {
    use tokio::fs;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::PipeConfig;
    use crate::error::IoError;

    /// Read from a FIFO asynchronously.
    pub async fn read_fifo(config: &PipeConfig, buf_size: usize) -> Result<Vec<u8>, IoError> {
        let path = config.path();
        let mut file = fs::File::open(&path).await?;
        let mut buffer = vec![0u8; buf_size];
        let n = file.read(&mut buffer).await?;
        buffer.truncate(n);
        Ok(buffer)
    }

    /// Write to a FIFO asynchronously.
    pub async fn write_fifo(config: &PipeConfig, data: &[u8]) -> Result<(), IoError> {
        let path = config.path();
        let mut file = fs::OpenOptions::new().write(true).open(&path).await?;
        file.write_all(data).await?;
        file.flush().await?;
        Ok(())
    }
}
