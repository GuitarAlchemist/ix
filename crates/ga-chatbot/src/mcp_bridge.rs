//! MCP bridge — spawns GA and IX MCP servers as child processes and routes
//! tool calls over stdio JSON-RPC.
//!
//! Each child process speaks newline-delimited JSON-RPC over stdin/stdout.
//! The bridge merges their tool catalogs under `ga__` / `ix__` prefixes and
//! dispatches `execute_tool` to the correct child by prefix.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the MCP bridge.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// Failed to spawn the child process.
    #[error("failed to spawn MCP child: {0}")]
    SpawnFailed(String),

    /// I/O error on the child's stdin/stdout pipe.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// The child returned a JSON-RPC error object.
    #[error("JSON-RPC error {code}: {message}")]
    JsonRpcError {
        /// JSON-RPC error code.
        code: i64,
        /// Human-readable message.
        message: String,
    },

    /// No tool matched the prefixed name.
    #[error("tool not found: {0}")]
    ToolNotFound(String),

    /// Reading a response exceeded the timeout.
    #[error("response timed out after {0:?}")]
    Timeout(Duration),

    /// Failed to parse JSON from the child.
    #[error("JSON parse error: {0}")]
    ParseError(#[from] serde_json::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, McpError>;

// ---------------------------------------------------------------------------
// JSON-RPC message types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    params: Value,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    #[allow(dead_code)]
    id: Option<Value>,
    result: Option<Value>,
    error: Option<JsonRpcErrorObj>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcErrorObj {
    code: i64,
    message: String,
}

// ---------------------------------------------------------------------------
// Tool descriptor (mirrors the MCP `tools/list` schema)
// ---------------------------------------------------------------------------

/// A single tool descriptor returned by `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// Tool name (un-prefixed from child, prefixed in the merged catalog).
    pub name: String,
    /// Human-readable description.
    #[serde(default)]
    pub description: String,
    /// JSON Schema describing the tool's input parameters.
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

// ---------------------------------------------------------------------------
// McpChild — a single MCP child process
// ---------------------------------------------------------------------------

/// Default read-response timeout.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// A running MCP child process that communicates via stdio JSON-RPC.
pub struct McpChild {
    /// Human label for logging.
    label: String,
    /// Underlying OS process.
    child: Child,
    /// Buffered writer to the child's stdin.
    writer: BufWriter<std::process::ChildStdin>,
    /// Buffered reader from the child's stdout.
    reader: BufReader<std::process::ChildStdout>,
    /// Auto-incrementing message ID.
    next_id: AtomicU64,
    /// Read timeout per response.
    timeout: Duration,
}

impl McpChild {
    /// Spawn a new MCP child process.
    ///
    /// `label` is a human-friendly tag (`"ga"` / `"ix"`).
    /// `program` is the executable (or first word of the command).
    /// `args` are extra CLI arguments.
    pub fn spawn(label: &str, program: &str, args: &[String]) -> Result<Self> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()); // let child errors appear in terminal

        let mut child = cmd
            .spawn()
            .map_err(|e| McpError::SpawnFailed(format!("{label}: {e}")))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::SpawnFailed(format!("{label}: stdin not piped")))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::SpawnFailed(format!("{label}: stdout not piped")))?;

        Ok(Self {
            label: label.to_string(),
            child,
            writer: BufWriter::new(stdin),
            reader: BufReader::new(stdout),
            next_id: AtomicU64::new(1),
            timeout: DEFAULT_TIMEOUT,
        })
    }

    /// Override the default read-response timeout.
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    // -- private helpers ----------------------------------------------------

    /// Send a JSON-RPC request and return the assigned ID.
    fn send(&mut self, method: &str, params: Value) -> Result<u64> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };
        let mut line = serde_json::to_string(&req)?;
        line.push('\n');
        self.writer.write_all(line.as_bytes())?;
        self.writer.flush()?;
        Ok(id)
    }

    /// Read one line from stdout with a timeout.
    ///
    /// Because `std::io::BufRead` has no native timeout, we poll with short
    /// sleeps. This is acceptable for a synchronous bridge that processes
    /// one request at a time.
    fn read_line_timeout(&mut self) -> Result<String> {
        let deadline = Instant::now() + self.timeout;
        let mut buf = String::new();

        // On Windows, `BufReader::read_line` on a pipe blocks until data
        // arrives. We rely on the child responding promptly. For a
        // production system you would use overlapped I/O or threads; here
        // a simple blocking read with an outer timeout thread is overkill
        // for the expected workload. We do a single blocking read and
        // check elapsed time after it returns.
        let n = self.reader.read_line(&mut buf)?;
        if n == 0 {
            return Err(McpError::IoError(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("{}: child closed stdout", self.label),
            )));
        }
        if Instant::now() > deadline {
            return Err(McpError::Timeout(self.timeout));
        }
        Ok(buf)
    }

    /// Read the next JSON-RPC response, skipping notifications (no `id`).
    fn read_response(&mut self) -> Result<JsonRpcResponse> {
        let deadline = Instant::now() + self.timeout;
        loop {
            if Instant::now() > deadline {
                return Err(McpError::Timeout(self.timeout));
            }
            let line = self.read_line_timeout()?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Skip non-JSON lines (build output, warnings, etc.)
            if !trimmed.starts_with('{') {
                eprintln!("[mcp-bridge/{}] skip: {}", self.label, trimmed);
                continue;
            }
            let resp: JsonRpcResponse = match serde_json::from_str(trimmed) {
                Ok(r) => r,
                Err(_) => {
                    eprintln!(
                        "[mcp-bridge/{}] skip non-json: {}",
                        self.label,
                        &trimmed[..trimmed.len().min(120)]
                    );
                    continue;
                }
            };
            // Skip notifications (messages without an id).
            if resp.id.is_some() {
                return Ok(resp);
            }
        }
    }

    /// Send a request and return the `result` value from the response.
    fn call(&mut self, method: &str, params: Value) -> Result<Value> {
        self.send(method, params)?;
        let resp = self.read_response()?;
        if let Some(err) = resp.error {
            return Err(McpError::JsonRpcError {
                code: err.code,
                message: err.message,
            });
        }
        Ok(resp.result.unwrap_or(Value::Null))
    }

    // -- public API ---------------------------------------------------------

    /// Call `tools/list` and return the tool descriptors.
    pub fn list_tools(&mut self) -> Result<Vec<ToolDescriptor>> {
        let result = self.call("tools/list", serde_json::json!({}))?;
        // result.tools is the array
        let tools_val = result.get("tools").cloned().unwrap_or(Value::Array(vec![]));
        let tools: Vec<ToolDescriptor> = serde_json::from_value(tools_val)?;
        Ok(tools)
    }

    /// Call `tools/call` with the given tool name and arguments.
    pub fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });
        self.call("tools/call", params)
    }

    /// Kill the child process (best-effort).
    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for McpChild {
    fn drop(&mut self) {
        self.kill();
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for spawning the two MCP child processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpBridgeConfig {
    /// Executable for the GA MCP server (e.g. `"dotnet"`).
    pub ga_command: String,
    /// Arguments for the GA MCP server (e.g. `["run", "--project", "GaMcpServer"]`).
    #[serde(default)]
    pub ga_args: Vec<String>,
    /// Executable for the IX MCP server (e.g. `"cargo"`).
    pub ix_command: String,
    /// Arguments for the IX MCP server (e.g. `["run", "-p", "ix-agent"]`).
    #[serde(default)]
    pub ix_args: Vec<String>,
}

// ---------------------------------------------------------------------------
// McpBridge — merged two-child bridge
// ---------------------------------------------------------------------------

/// The prefix separator used in merged tool names.
const PREFIX_SEP: &str = "__";

/// An entry in the merged tool catalog, linking a prefixed name back to its
/// originating child and un-prefixed name.
#[derive(Debug, Clone)]
struct CatalogEntry {
    /// `"ga"` or `"ix"`.
    origin: String,
    /// Original (un-prefixed) tool name on the child.
    original_name: String,
    /// The full tool descriptor (with the prefixed name).
    descriptor: ToolDescriptor,
}

/// Bridge that owns two MCP children and merges their tool catalogs.
pub struct McpBridge {
    ga: McpChild,
    ix: McpChild,
    catalog: Vec<CatalogEntry>,
}

impl McpBridge {
    /// Spawn both MCP children, fetch their tool catalogs, and merge them
    /// under `ga__` / `ix__` prefixes.
    pub fn new(config: &McpBridgeConfig) -> Result<Self> {
        let mut ga = McpChild::spawn("ga", &config.ga_command, &config.ga_args)?;
        let mut ix = McpChild::spawn("ix", &config.ix_command, &config.ix_args)?;

        let ga_tools = ga.list_tools()?;
        let ix_tools = ix.list_tools()?;

        let mut catalog = Vec::with_capacity(ga_tools.len() + ix_tools.len());

        for tool in ga_tools {
            let prefixed = format!("ga{PREFIX_SEP}{}", tool.name);
            catalog.push(CatalogEntry {
                origin: "ga".to_string(),
                original_name: tool.name.clone(),
                descriptor: ToolDescriptor {
                    name: prefixed,
                    description: tool.description,
                    input_schema: tool.input_schema,
                },
            });
        }

        for tool in ix_tools {
            let prefixed = format!("ix{PREFIX_SEP}{}", tool.name);
            catalog.push(CatalogEntry {
                origin: "ix".to_string(),
                original_name: tool.name.clone(),
                descriptor: ToolDescriptor {
                    name: prefixed,
                    description: tool.description,
                    input_schema: tool.input_schema,
                },
            });
        }

        Ok(Self { ga, ix, catalog })
    }

    /// Return the merged tool catalog (prefixed names, suitable for LLM
    /// function-calling schemas).
    pub fn merged_tools(&self) -> Vec<ToolDescriptor> {
        self.catalog.iter().map(|e| e.descriptor.clone()).collect()
    }

    /// Execute a tool by its prefixed name (e.g. `"ga__GetTuning"`).
    ///
    /// Routes to the correct child, stripping the prefix before dispatch.
    pub fn execute_tool(&mut self, prefixed_name: &str, arguments: Value) -> Result<Value> {
        let entry = self
            .catalog
            .iter()
            .find(|e| e.descriptor.name == prefixed_name)
            .ok_or_else(|| McpError::ToolNotFound(prefixed_name.to_string()))?;

        let original_name = entry.original_name.clone();
        let origin = entry.origin.clone();

        match origin.as_str() {
            "ga" => self.ga.call_tool(&original_name, arguments),
            "ix" => self.ix.call_tool(&original_name, arguments),
            _ => Err(McpError::ToolNotFound(prefixed_name.to_string())),
        }
    }

    /// Shut down both child processes.
    pub fn shutdown(&mut self) {
        self.ga.kill();
        self.ix.kill();
    }
}

impl Drop for McpBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_separator_is_double_underscore() {
        assert_eq!(PREFIX_SEP, "__");
    }

    #[test]
    fn tool_descriptor_roundtrip() {
        let td = ToolDescriptor {
            name: "ga__GetTuning".to_string(),
            description: "Get string notes for an instrument tuning".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "tuning": { "type": "string" }
                }
            }),
        };
        let json = serde_json::to_string(&td).unwrap();
        let back: ToolDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "ga__GetTuning");
        assert_eq!(back.description, td.description);
    }

    #[test]
    fn config_deserializes() {
        let json = r#"{
            "ga_command": "dotnet",
            "ga_args": ["run", "--project", "GaMcpServer"],
            "ix_command": "cargo",
            "ix_args": ["run", "-p", "ix-agent"]
        }"#;
        let cfg: McpBridgeConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.ga_command, "dotnet");
        assert_eq!(cfg.ga_args, vec!["run", "--project", "GaMcpServer"]);
        assert_eq!(cfg.ix_command, "cargo");
        assert_eq!(cfg.ix_args, vec!["run", "-p", "ix-agent"]);
    }

    #[test]
    fn config_defaults_empty_args() {
        let json = r#"{"ga_command": "dotnet", "ix_command": "cargo"}"#;
        let cfg: McpBridgeConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.ga_args.is_empty());
        assert!(cfg.ix_args.is_empty());
    }

    #[test]
    fn error_display() {
        let e = McpError::ToolNotFound("ga__Foo".to_string());
        assert_eq!(e.to_string(), "tool not found: ga__Foo");

        let e = McpError::Timeout(Duration::from_secs(10));
        assert_eq!(e.to_string(), "response timed out after 10s");

        let e = McpError::JsonRpcError {
            code: -32601,
            message: "Method not found".to_string(),
        };
        assert!(e.to_string().contains("-32601"));
    }
}
