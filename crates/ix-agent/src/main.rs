//! MCP (Model Context Protocol) server for machin ML algorithms.
//!
//! Communicates over stdio using JSON-RPC 2.0.
//! Reads requests from stdin, writes responses to stdout, logs to stderr.

mod handlers;
mod ml_pipeline;
mod tools;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{self, BufRead, Write};

use tools::ToolRegistry;

/// JSON-RPC 2.0 request.
#[derive(Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    method: String,
    id: Option<Value>,
    params: Option<Value>,
}

/// JSON-RPC 2.0 response.
#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

/// JSON-RPC 2.0 error object.
#[derive(Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

const JSONRPC_PARSE_ERROR: i64 = -32700;
const JSONRPC_METHOD_NOT_FOUND: i64 = -32601;
const JSONRPC_INTERNAL_ERROR: i64 = -32603;

fn main() {
    let registry = ToolRegistry::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    eprintln!("[ix-mcp] MCP server started, listening on stdin...");

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

        let request: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0".into(),
                    id: Value::Null,
                    result: None,
                    error: Some(JsonRpcError {
                        code: JSONRPC_PARSE_ERROR,
                        message: format!("Parse error: {}", e),
                        data: None,
                    }),
                };
                write_response(&mut stdout, &resp);
                continue;
            }
        };

        let id = request.id.clone().unwrap_or(Value::Null);

        eprintln!("[ix-mcp] method={}", request.method);

        let response = match request.method.as_str() {
            "initialize" => handle_initialize(id, &registry),
            "notifications/initialized" => {
                // Client acknowledgement — no response needed for notifications
                continue;
            }
            "tools/list" => handle_tools_list(id, &registry),
            "tools/call" => handle_tools_call(id, request.params, &registry),
            _ => JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: JSONRPC_METHOD_NOT_FOUND,
                    message: format!("Method not found: {}", request.method),
                    data: None,
                }),
            },
        };

        write_response(&mut stdout, &response);
    }

    eprintln!("[ix-mcp] Server shutting down.");
}

fn write_response(stdout: &mut io::Stdout, response: &JsonRpcResponse) {
    let json = serde_json::to_string(response).expect("Failed to serialize response");
    let _ = writeln!(stdout, "{}", json);
    let _ = stdout.flush();
}

fn handle_initialize(id: Value, registry: &ToolRegistry) -> JsonRpcResponse {
    let tool_list = registry.list();
    let _tools_arr = tool_list.get("tools").cloned().unwrap_or(Value::Array(vec![]));

    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "ix-mcp",
                "version": env!("CARGO_PKG_VERSION"),
            },
        })),
        error: None,
    }
}

fn handle_tools_list(id: Value, registry: &ToolRegistry) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(registry.list()),
        error: None,
    }
}

fn handle_tools_call(id: Value, params: Option<Value>, registry: &ToolRegistry) -> JsonRpcResponse {
    let params = match params {
        Some(p) => p,
        None => {
            return JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: JSONRPC_INTERNAL_ERROR,
                    message: "Missing params for tools/call".into(),
                    data: None,
                }),
            };
        }
    };

    let tool_name = match params.get("name").and_then(|v| v.as_str()) {
        Some(n) => n.to_string(),
        None => {
            return JsonRpcResponse {
                jsonrpc: "2.0".into(),
                id,
                result: None,
                error: Some(JsonRpcError {
                    code: JSONRPC_INTERNAL_ERROR,
                    message: "Missing 'name' in tools/call params".into(),
                    data: None,
                }),
            };
        }
    };

    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    eprintln!("[ix-mcp] tools/call name={}", tool_name);

    match registry.call(&tool_name, arguments) {
        Ok(result) => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&result).unwrap_or_default(),
                }],
            })),
            error: None,
        },
        Err(e) => JsonRpcResponse {
            jsonrpc: "2.0".into(),
            id,
            result: Some(serde_json::json!({
                "content": [{
                    "type": "text",
                    "text": format!("Error: {}", e),
                }],
                "isError": true,
            })),
            error: None,
        },
    }
}
