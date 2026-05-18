# ix-agent

MCP (Model Context Protocol) server that exposes ix's ML and governance
primitives as Claude Code tools. Binary: `ix-mcp`.

This README is intentionally short — it documents only the things that
matter when **operating** the server (start, env vars, scopes). For the
internal architecture, read `src/main.rs` (dispatcher), `src/tools.rs`
(registry), and `src/registry_bridge.rs` (skill ↔ MCP adapter).

## Quick start

```bash
cargo run --bin ix-mcp
```

The process reads JSON-RPC 2.0 frames on stdin and writes responses on
stdout. Standard MCP — wire it up to Claude Code, agent-blackbox,
Demerzel, etc. via your client's MCP server configuration.

## Tool scopes (per-consumer subsets)

By default `ix-mcp` advertises its full tool surface — ~68 tools. Real
consumers usually only need a handful. **Scopes** let a consumer
opt into a curated subset:

| Scope             | Tools                                | Notes                                |
| ----------------- | ------------------------------------ | ------------------------------------ |
| `default`         | all registered tools (the full set)  | Backward-compatible — implicit default. |
| `agent-blackbox`  | 10 tools (see `src/scopes.rs`)       | The risk-report + autoresearch slice. |

### Selecting a scope

Two opt-in mechanisms, in precedence order (highest first):

1. **`clientInfo.scope` inside the `initialize` request**:

   ```json
   {
     "jsonrpc": "2.0", "id": 1, "method": "initialize",
     "params": { "clientInfo": { "name": "agent-blackbox", "scope": "agent-blackbox" } }
   }
   ```

2. **`IX_MCP_SCOPE` environment variable** before launching the binary:

   ```bash
   IX_MCP_SCOPE=agent-blackbox cargo run --bin ix-mcp
   ```

3. If neither is set, the scope is `default` and every tool is
   advertised — same behavior as today.

### Backward compatibility

Existing consumers that connect without setting a scope see the full
tool surface. No behavior change.

### Boundary

Scopes are an **advertisement filter**, not a capability boundary.
`tools/list` returns only the subset for the active scope, but
`tools/call` is not gated — once a client discovers a tool name
(out-of-band, in another scope, or in source), it can invoke it.
Layering real auth on top is a wave-2 concern.

### Adding a new scope

Three steps in `src/scopes.rs`:

1. Add a variant to the `Scope` enum.
2. Define a `const NEW_SCOPE_TOOLS: &[&str] = &["ix_foo", ...]`.
3. Add a row to `SCOPES`: `(Scope::NewScope, NEW_SCOPE_TOOLS)`.

Then add a test pair to `tests/scope_advertisement.rs` modelled on the
agent-blackbox pair: a size ceiling and a subset-of-default assertion.
The subset assertion is what catches the regression where someone
renames a tool in `register_*` and forgets to update the scope table.

## Environment variables

- `IX_MCP_SCOPE` — see above.
- `IX_SESSION_LOG` — path to a JSONL session log for governance
  middleware. See `src/registry_bridge.rs::session_log_slot` for the
  bootstrap order.
