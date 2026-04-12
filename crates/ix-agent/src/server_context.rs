//! Shared server context for bidirectional JSON-RPC.
//!
//! Enables tool handlers to send server-initiated requests (e.g. MCP
//! `sampling/createMessage`) and await the correlated client response
//! without blocking the main reader loop.
//!
//! # Shape
//!
//! - A single **writer thread** owns stdout and drains an mpsc channel
//!   of outbound JSON strings. Any thread can enqueue a message via
//!   [`ServerContext::write_outbound`].
//! - A **pending-request table** maps outbound request id → oneshot
//!   `mpsc::Sender<Value>`. The main reader thread routes inbound
//!   JSON-RPC responses into this table by id.
//! - Tool handlers call [`ServerContext::sample`] to issue a
//!   `sampling/createMessage` request and block (with a timeout) on
//!   the correlated response.
//!
//! # Concurrency model (std only, no tokio)
//!
//! Everything here uses `std::sync::{Arc, Mutex}` and `std::sync::mpsc`.
//! Handlers that call `sample` run on worker threads spawned by the
//! dispatcher so they can block without stalling the reader.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Default timeout for server-initiated sampling requests.
pub const SAMPLING_TIMEOUT: Duration = Duration::from_secs(30);

/// A slot in the pending-request table. The handler thread holds the
/// `Receiver`; the reader thread uses the `Sender` to deliver the
/// client's response envelope.
type PendingSlot = Sender<Value>;

/// Shared context passed to context-aware tool handlers.
///
/// Cloneable: the inner state is an `Arc`, so clones share the same
/// writer queue and pending map.
#[derive(Clone)]
pub struct ServerContext {
    inner: Arc<Inner>,
}

struct Inner {
    // `mpsc::Sender` is not `Sync` on all Rust versions — wrap in a
    // `Mutex` so `&ServerContext` can safely enqueue from any thread.
    outbound: Mutex<Sender<String>>,
    pending: Mutex<HashMap<i64, PendingSlot>>,
    next_id: AtomicI64,
}

impl ServerContext {
    /// Build a new context plus the receiver half of the outbound queue.
    /// The caller is expected to spawn a writer thread that drains the
    /// receiver and writes each string as a line to stdout.
    pub fn new() -> (Self, Receiver<String>) {
        let (tx, rx) = mpsc::channel::<String>();
        let ctx = Self {
            inner: Arc::new(Inner {
                outbound: Mutex::new(tx),
                pending: Mutex::new(HashMap::new()),
                next_id: AtomicI64::new(1),
            }),
        };
        (ctx, rx)
    }

    /// Allocate a fresh outbound request id. Server-initiated request
    /// ids use a dedicated high range (starts at 1, monotonic) so they
    /// cannot collide with client-supplied request ids in practice —
    /// the JSON-RPC spec only requires uniqueness within the sender.
    fn next_id(&self) -> i64 {
        self.inner.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Enqueue a serialized JSON message for the writer thread to emit.
    /// Silently drops on channel close (server shutting down).
    pub fn write_outbound(&self, line: String) {
        if let Ok(tx) = self.inner.outbound.lock() {
            let _ = tx.send(line);
        }
    }

    /// Serialize and enqueue a JSON value as a single line.
    pub fn write_value(&self, value: &Value) {
        match serde_json::to_string(value) {
            Ok(s) => self.write_outbound(s),
            Err(e) => eprintln!("[ix-mcp] failed to serialize outbound value: {}", e),
        }
    }

    /// Route an inbound response envelope (one that carries an `id`
    /// matching an outstanding server-initiated request) to the waiting
    /// handler. Returns `true` if the id matched a pending slot.
    pub fn deliver_response(&self, id: i64, envelope: Value) -> bool {
        let slot = {
            let mut map = self.inner.pending.lock().unwrap();
            map.remove(&id)
        };
        match slot {
            Some(tx) => {
                let _ = tx.send(envelope);
                true
            }
            None => false,
        }
    }

    /// Issue a `sampling/createMessage` request to the client and block
    /// (up to [`SAMPLING_TIMEOUT`]) waiting for the response. Returns
    /// the concatenated text content on success.
    pub fn sample(
        &self,
        user_text: &str,
        system_prompt: &str,
        max_tokens: u32,
    ) -> Result<String, String> {
        let id = self.next_id();
        let (tx, rx) = mpsc::channel::<Value>();

        // Register the pending slot *before* writing, so a very fast
        // client cannot respond before we are listening.
        {
            let mut map = self.inner.pending.lock().unwrap();
            map.insert(id, tx);
        }

        let envelope = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "sampling/createMessage",
            "params": {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": user_text,
                        }
                    }
                ],
                "maxTokens": max_tokens,
                "systemPrompt": system_prompt,
            }
        });

        self.write_value(&envelope);

        let result = match rx.recv_timeout(SAMPLING_TIMEOUT) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => {
                // Clean up pending slot on timeout.
                let mut map = self.inner.pending.lock().unwrap();
                map.remove(&id);
                return Err(format!(
                    "sampling/createMessage timed out after {:?}",
                    SAMPLING_TIMEOUT
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err("sampling/createMessage channel disconnected".into());
            }
        };

        // Response shape: { jsonrpc, id, result: { role, content: { type, text }, ... } }
        // or an error envelope.
        if let Some(err) = result.get("error") {
            return Err(format!("sampling/createMessage error: {}", err));
        }

        let text = result
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(|c| {
                // Content may be a single object or an array of parts.
                if let Some(arr) = c.as_array() {
                    // Concatenate any text parts.
                    let mut out = String::new();
                    for part in arr {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(t);
                        }
                    }
                    if out.is_empty() {
                        None
                    } else {
                        Some(out)
                    }
                } else {
                    c.get("text").and_then(|v| v.as_str()).map(str::to_string)
                }
            })
            .ok_or_else(|| format!("sampling response missing text content: {}", result))?;

        Ok(text)
    }

    /// Issue a `sampling/createMessage` request with both text AND
    /// an image. The MCP sampling spec supports multimodal content
    /// arrays — this method sends a `[{type: image}, {type: text}]`
    /// pair so the client (Claude) can analyze a screenshot alongside
    /// a text prompt.
    ///
    /// Primary consumer: the Sentinel's rendering-audit adapter,
    /// which captures a Prime Radiant screenshot via the GA API and
    /// asks Claude to analyze it for rendering correctness.
    ///
    /// Per the MCP sampling spec:
    /// <https://modelcontextprotocol.info/docs/concepts/sampling/>
    ///
    /// ```json
    /// { "messages": [{
    ///     "role": "user",
    ///     "content": [
    ///       { "type": "image", "data": "base64...", "mimeType": "image/png" },
    ///       { "type": "text", "text": "What do you see?" }
    ///     ]
    /// }] }
    /// ```
    pub fn sample_with_image(
        &self,
        user_text: &str,
        image_base64: &str,
        mime_type: &str,
        system_prompt: &str,
        max_tokens: u32,
    ) -> Result<String, String> {
        let id = self.next_id();
        let (tx, rx) = mpsc::channel::<Value>();

        {
            let mut map = self.inner.pending.lock().unwrap();
            map.insert(id, tx);
        }

        let envelope = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "sampling/createMessage",
            "params": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "data": image_base64,
                                "mimeType": mime_type,
                            },
                            {
                                "type": "text",
                                "text": user_text,
                            }
                        ]
                    }
                ],
                "maxTokens": max_tokens,
                "systemPrompt": system_prompt,
            }
        });

        self.write_value(&envelope);

        let result = match rx.recv_timeout(SAMPLING_TIMEOUT) {
            Ok(v) => v,
            Err(RecvTimeoutError::Timeout) => {
                let mut map = self.inner.pending.lock().unwrap();
                map.remove(&id);
                return Err(format!(
                    "sampling/createMessage (with image) timed out after {:?}",
                    SAMPLING_TIMEOUT
                ));
            }
            Err(RecvTimeoutError::Disconnected) => {
                return Err("sampling/createMessage channel disconnected".into());
            }
        };

        if let Some(err) = result.get("error") {
            return Err(format!("sampling/createMessage error: {}", err));
        }

        // Same response extraction as text-only `sample`.
        let text = result
            .get("result")
            .and_then(|r| r.get("content"))
            .and_then(|c| {
                if let Some(arr) = c.as_array() {
                    let mut out = String::new();
                    for part in arr {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(t);
                        }
                    }
                    if out.is_empty() { None } else { Some(out) }
                } else {
                    c.get("text").and_then(|v| v.as_str()).map(str::to_string)
                }
            })
            .ok_or_else(|| format!("sampling response missing text content: {}", result))?;

        Ok(text)
    }
}
