# ixmux mcp: the MCP adapter

[Français](MCP.fr.md)

`ixmux mcp` is a local MCP server on stdio that hands mandates to the sessions an `ixmux serve`
manages, and reads their results. It is the same binary as `ixmux serve`, and it talks to it
through its HTTP API, on loopback only.

## Transport and command

```
ixmux mcp --state <dir>
```

- `<dir>` is the `--state` directory of the `ixmux serve` to talk to. At each tool call, the
  adapter reads the address in `ixmux.json` and the token in `ixmux.token` there. It never takes
  the token as an argument or a variable, and never writes it: not to stdout, not to stderr, not in
  a tool answer.
- JSON-RPC 2.0 over stdin and stdout, one message per line (`\n`, a `\r` before it is accepted).
  stdout carries JSON-RPC only; stderr gets the usage message of a bad command line, and nothing
  else.
- Exit codes: 0 once stdin is closed; 2 for a bad command line.

To register it in Claude Code (cmd.exe, with the state of the README's real-session runbook):

```
claude mcp add ixmux -- %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe mcp --state %USERPROFILE%\.cache\ixmux-claude\state
```

Demonstration without a client, from PowerShell, once `ixmux serve` runs on that state:

```
'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"demo","version":"0"}}}',
'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ixmux_submit","arguments":{"target":"c","mandate_id":"demo-1","text":"Dis bonjour."}}}',
'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ixmux_result","arguments":{"mandate_id":"demo-1"}}}' |
  & "$env:USERPROFILE\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe" mcp --state "$env:USERPROFILE\.cache\ixmux-claude\state"
```

## Methods

| Method | Answer |
|---|---|
| `initialize` | `protocolVersion`: the one asked if the adapter speaks it (`2025-11-25`, `2025-06-18`, `2025-03-26`, `2024-11-05`), else `2025-11-25`. `capabilities.tools.listChanged` is false. `serverInfo` is `ixmux` and its version. It needs no ixmux. |
| `notifications/initialized` | A notification: no answer. Any other notification is ignored. |
| `ping` | `{}`. |
| `tools/list` | The three tools below, with their `inputSchema`. |
| `tools/call` | `{content: [{type: "text", text}], structuredContent, isError}`. `text` is the JSON of `structuredContent`. |

Protocol errors are JSON-RPC errors, and the adapter goes on reading:

| Code | When | `data.reason` |
|---|---|---|
| -32700 | a line that is not JSON | |
| -32600 | a line over 1 MiB (read to its end and dropped), a batch, no `jsonrpc: "2.0"` or no `method` | `line-too-long` for the first |
| -32601 | an unknown method | |
| -32602 | an unknown tool, or `params` of `tools/call` that do not decode | `unknown-tool`, `bad-arguments` |

## Tools

A tool that fails answers `isError: true` with `structuredContent` `{error, detail}`. `error` is a
typed reason: one of the adapter's (below) or the refusal code of ixmux as it is
(`mandate-id-reuse`, `stale-incarnation`, `unknown-target`, `uncertain-needs-reconcile`,
`in-flight`, `target-busy`, `busy`, `unknown-mandate`, `text-too-long`, `engine-closed`, ...).
`detail` is for a person; never parse it.

| Reason of the adapter | When |
|---|---|
| `ixmux-unavailable` | no `ixmux.json` or no token in `<dir>`, `ixmux.json` says `stopped`, its address is not loopback, ixmux does not answer, or refuses the token |
| `bad-arguments` | an unknown argument, a wrong type, a missing `target` |
| `bad-mandate-id` | a `mandate_id` that is empty, over 128 bytes, not made of letters, digits, `.`, `_` and `-`, or starting with `.` |
| `ixmux-error` | an answer of ixmux the adapter cannot read |

### `ixmux_status`

Input: `{}`. Output:

```
{"version": "...", "epoch": "...",
 "targets": [{"name": "c", "kind": "claude", "incarnation": "<uuid>", "alive": true,
              "in_flight": [{"mandate_id": "m1", "state": "delivering", "still_running": true}]}]}
```

Metadata only: no text, no answer, no path, no token. `in_flight` lists the target's deliveries
that are `accepted` or `delivering`, or whose adapter call has not returned.

### `ixmux_submit`

Input: `{target, mandate_id, text, incarnation?}`, required `target`, `mandate_id`, `text` (64 KiB
at most, else `text-too-long`). Without `incarnation`, the adapter takes the target's current one
from `/v1/status`. **Give the original incarnation when submitting again after a restart**: the
same mandate under another incarnation is another mandate, refused `mandate-id-reuse`.

Output, the receipt, within 2 s even when the turn lasts 10 minutes:

```
{"mandate_id": "m1", "target": "c", "incarnation": "<uuid>",
 "accepted": true, "state": "delivering", "replayed": false, "receipt": "<session>:m1"}
```

- `accepted: true` only once the mandate's `accepted` line is written and synced in ixmux's command
  log: either ixmux answered the POST, or `GET /v1/mandates/{id}` shows it `accepted` or
  `delivering`, which the engine shows only after the line is synced.
- `state` is `accepted` or `delivering` (the turn runs on), `delivered` (it was quick, or it is a
  replay: then `replayed: true` and no new turn), or `refused` (the target certainly did not take
  it).
- `accepted: false, state: "submitted"`: ixmux did not confirm within 1.5 s. Nothing is promised;
  `ixmux_result` tells, and the same submission again is safe (at most once per `mandate_id`).
- `receipt` is the target's receipt, once delivered.

Why this design: `POST /v1/mandates` waits for the whole turn. The adapter sends it and lets it
run, polls `GET /v1/mandates/{id}` every 25 ms, and answers at the first proof, or after 1.5 s.
ixmux delivers whether or not the adapter still waits: a killed adapter does not cancel the turn.
It is the smallest change that keeps the promise: no new mode on the server.

### `ixmux_result`

Input: `{mandate_id}`. Output:

```
{"mandate_id": "m1", "target": "c", "state": "delivered", "last": "delivering->delivered",
 "still_running": false, "reason": "...",
 "receipt": "<session>:m1", "text": "ACK m1\n...", "ack_seen": true, "truncated": false}
```

- `state`: `accepted`, `delivering`, `delivered`, `refused` or `uncertain`. An `uncertain` mandate
  is never replayed: a new submission is refused `uncertain-needs-reconcile`.
- `receipt`, `text`, `ack_seen` and `truncated` appear once delivered. `text` is at most 4 KiB;
  `truncated: true` says the answer was longer and its end is missing.
- It is read from the command log of ixmux (`GET /v1/mandates/{id}`): it survives a restart of the
  adapter and a restart of ixmux.
- An unknown id is `unknown-mandate`.

## Stop

stdin closed: the adapter finishes the call under way (2 s at most for `ixmux_submit`) and exits 0.
A POST still running is dropped by the adapter; ixmux goes on and finishes the delivery. ixmux keeps
running in every case.

## What the adapter does not do

- It does not stop ixmux: `/v1/shutdown` is not a tool. It does not reconcile an uncertain mandate
  either (`/v1/mandates/{id}/reconcile` is not a tool).
- No shell, no file access outside `ixmux.json` and `ixmux.token`, no WMUX session control, no
  tunnel, no service, no network beyond the loopback address of `ixmux.json`.
- It keeps no state of its own: everything it answers comes from ixmux.
- No resources, prompts, sampling or notifications of its own.

## Limits

- A long `ixmux_submit` holds the next calls: the adapter answers one call at a time.
- The receipt's 2 s are 1.5 s of wait plus the time of one HTTP read on loopback; a machine stalled
  beyond that answers later.
- Tested against the fake claude only (`claudestream/internal/fakeclaude`); no real model call.
