//! MCP (Model Context Protocol) server for ix ML algorithms.
//!
//! Communicates over stdio using JSON-RPC 2.0. This dispatcher is
//! **bidirectional**: tool handlers can issue server-initiated requests
//! (e.g. `sampling/createMessage`) and await the correlated client
//! response without blocking the inbound reader.
//!
//! # Architecture
//!
//! ```text
//!          stdin                                    stdout
//!            │                                         ▲
//!            ▼                                         │
//!     ┌────────────┐   requests   ┌──────────────┐     │
//!     │ reader     │─────────────▶│ worker pool  │     │
//!     │ (main)     │              │ (tools/call) │─────┤
//!     │            │              └──────────────┘     │
//!     │            │                                    │
//!     │            │   responses (to server-initiated)  │
//!     │            │─────▶ ServerContext::deliver ─▶ ... │
//!     └────────────┘                                    │
//!                                                        │
//!                               ┌────────────────┐      │
//!                               │ writer thread  │──────┘
//!                               │ (drains queue) │
//!                               └────────────────┘
//! ```
//!
//! - **writer thread**: owns stdout, drains an mpsc queue of outbound
//!   JSON lines. Any other thread can enqueue via `ServerContext`.
//! - **worker threads**: one spawned per `tools/call`; they may block
//!   on `ctx.sample()` without stalling the reader.
//! - **reader (main)**: parses each inbound line as a generic `Value`,
//!   branches on shape (request vs server-initiated response), and
//!   either dispatches to a worker or routes into the pending map.

use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::thread;

use ix_agent::scopes::Scope;
use ix_agent::server_context::ServerContext;
use ix_agent::tools::ToolRegistry;

const JSONRPC_PARSE_ERROR: i64 = -32700;
const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
const JSONRPC_INTERNAL_ERROR: i64 = -32603;

fn main() {
    let registry = Arc::new(ToolRegistry::new());
    let (ctx, outbound_rx) = ServerContext::new();

    // Active scope for this MCP session. Resolved at startup from the
    // IX_MCP_SCOPE env var, then potentially overridden by the
    // `initialize` request's `clientInfo.scope` field. Wrapped in a
    // Mutex because the reader thread reads it on every tools/list and
    // the initialize handler writes to it once on connect.
    let initial_scope = Scope::resolve(None);
    let scope = Arc::new(Mutex::new(initial_scope));
    eprintln!(
        "[ix-mcp] startup scope = {:?} (override via IX_MCP_SCOPE or clientInfo.scope)",
        initial_scope
    );

    // Writer thread: single owner of stdout. All outbound JSON-RPC
    // messages (responses, notifications, and server-initiated
    // requests) pass through this channel.
    let writer = thread::Builder::new()
        .name("ix-mcp-writer".into())
        .spawn(move || {
            let mut stdout = io::stdout();
            for line in outbound_rx {
                if writeln!(stdout, "{}", line).is_err() {
                    break;
                }
                let _ = stdout.flush();
            }
        })
        .expect("failed to spawn writer thread");

    eprintln!("[ix-mcp] MCP server started (bidirectional dispatcher), listening on stdin...");

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[ix-mcp] stdin read error: {}", e);
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let value: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                let resp = error_response(
                    Value::Null,
                    JSONRPC_PARSE_ERROR,
                    format!("Parse error: {}", e),
                );
                ctx.write_value(&resp);
                continue;
            }
        };

        // Demux: is this an inbound request/notification (has `method`)
        // or a response to a server-initiated request (has `id` and
        // `result`/`error`, no `method`)?
        let has_method = value.get("method").is_some();
        if !has_method {
            // Likely a response envelope for a sampling/createMessage call.
            if let Some(id) = value.get("id").and_then(|v| v.as_i64()) {
                if ctx.deliver_response(id, value) {
                    continue;
                }
                eprintln!(
                    "[ix-mcp] response id={} did not match any pending request; dropping",
                    id
                );
            } else {
                eprintln!("[ix-mcp] inbound message has neither method nor numeric id; dropping");
            }
            continue;
        }

        // Inbound request or notification.
        let method = value
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let id = value.get("id").cloned().unwrap_or(Value::Null);
        let params = value.get("params").cloned();

        eprintln!("[ix-mcp] method={}", method);

        match method.as_str() {
            "initialize" => {
                // Allow the client to nominate a scope via
                // params.clientInfo.scope. If absent, keep whatever
                // env-derived scope is already in the slot.
                if let Some(hint) = params
                    .as_ref()
                    .and_then(|p| p.get("clientInfo"))
                    .and_then(|ci| ci.get("scope"))
                    .and_then(|v| v.as_str())
                {
                    let resolved = Scope::resolve(Some(hint));
                    let mut slot = scope.lock().expect("scope mutex poisoned");
                    if *slot != resolved {
                        eprintln!(
                            "[ix-mcp] scope override from clientInfo: {:?} -> {:?}",
                            *slot, resolved
                        );
                        *slot = resolved;
                    }
                }
                let resp = handle_initialize(id, registry.as_ref());
                ctx.write_value(&resp);
            }
            "notifications/initialized" => {
                // Notification — no response required.
            }
            "tools/list" => {
                let active = *scope.lock().expect("scope mutex poisoned");
                let resp = handle_tools_list(id, registry.as_ref(), active);
                ctx.write_value(&resp);
            }
            "tools/call" => {
                // Run on a worker so handlers that block on sampling
                // do not stall the reader loop.
                let registry = Arc::clone(&registry);
                let ctx_clone = ctx.clone();
                thread::Builder::new()
                    .name("ix-mcp-worker".into())
                    .spawn(move || {
                        let resp = handle_tools_call(id, params, &registry, &ctx_clone);
                        ctx_clone.write_value(&resp);
                    })
                    .expect("failed to spawn worker thread");
            }
            other => {
                let resp = error_response(
                    id,
                    JSONRPC_METHOD_NOT_FOUND,
                    format!("Method not found: {}", other),
                );
                ctx.write_value(&resp);
            }
        }
    }

    eprintln!("[ix-mcp] Server shutting down.");
    // Dropping `ctx` here closes the outbound channel so the writer
    // thread can exit cleanly.
    drop(ctx);
    let _ = writer.join();
}

fn success_response(id: Value, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn error_response(id: Value, code: i64, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": {
            "code": code,
            "message": message,
        }
    })
}

fn handle_initialize(id: Value, _registry: &ToolRegistry) -> Value {
    success_response(
        id,
        json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "sampling": {}
            },
            "serverInfo": {
                "name": "ix-mcp",
                "version": env!("CARGO_PKG_VERSION"),
            },
        }),
    )
}

fn handle_tools_list(id: Value, registry: &ToolRegistry, scope: Scope) -> Value {
    success_response(id, registry.list_scoped(scope))
}

fn handle_tools_call(
    id: Value,
    params: Option<Value>,
    registry: &ToolRegistry,
    ctx: &ServerContext,
) -> Value {
    let params = match params {
        Some(p) => p,
        None => {
            return error_response(
                id,
                JSONRPC_INTERNAL_ERROR,
                "Missing params for tools/call".into(),
            );
        }
    };

    let tool_name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => {
            return error_response(
                id,
                JSONRPC_INTERNAL_ERROR,
                "Missing 'name' in tools/call params".into(),
            );
        }
    };

    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    eprintln!("[ix-mcp] tools/call name={}", tool_name);

    match registry.call_with_ctx(&tool_name, arguments, ctx) {
        Ok(result) => success_response(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&result).unwrap_or_default(),
                }],
            }),
        ),
        Err(e) => success_response(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error: {}", e),
                }],
                "isError": true,
            }),
        ),
    }
}
