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
  a tool answer. It sends it only to a server that proved it holds it: for each request, it dials
  one TCP connection, posts `/v1/hello` with a fresh 32-byte nonce, checks the `mac` with
  `hmac.Equal` against HMAC-SHA-256(HMAC-SHA-256(token, `ixmux/hello/v1`), nonce), and only then
  writes the authenticated request on that same connection. A missing, malformed or wrong `mac`
  (a replay included) is `ixmux-unverified`, and the token is not sent. A process that took the
  port without the token never gets it; a process that can read the token file is outside this
  model.
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
| `initialize` | `params` needs `protocolVersion` (a string), `capabilities` (an object) and `clientInfo` `{name, version}`, else -32602. `protocolVersion`: the one asked if the adapter holds it (`2025-11-25`, `2025-06-18`), else `2025-11-25`. Older versions are not offered: their batches are refused here. `capabilities.tools.listChanged` is false. `serverInfo` is `ixmux` and its version. It needs no ixmux. |
| `notifications/initialized` | A notification: no answer. Any other notification is ignored. |
| `ping` | `{}`. |
| `tools/list` | The four tools below, with their `inputSchema`. |
| `tools/call` | `{content: [{type: "text", text}], structuredContent, isError}`. `text` is the JSON of `structuredContent`. |

Protocol errors are JSON-RPC errors, and the adapter goes on reading:

| Code | When | `data.reason` |
|---|---|---|
| -32700 | a line that is not JSON | |
| -32600 | a line over 1 MiB (read to its end and dropped), a batch, a value that is not an object, an `id` that is not a string or a number (`null` included: MCP forbids it), no `jsonrpc: "2.0"`, a `method` that is not a string, `params` that are not an object or an array. Answered even without an `id`, with the request's `id` when it is valid, else `null` | `line-too-long` for the first |
| -32601 | an unknown method | |
| -32602 | an unknown tool; `initialize` without its fields; `tools/call` without a string `name`, or with `arguments` that are not an object | `unknown-tool`, `bad-arguments` |

A request without an `id` member is a notification and gets no answer, unless it is invalid.

## Tools

A tool that fails answers `isError: true` with `structuredContent` `{error, detail}`. `error` is a
typed reason: one of the adapter's (below) or a known refusal code of ixmux (`mandate-id-reuse`,
`stale-incarnation`, `unknown-target`, `uncertain-needs-reconcile`, `in-flight`, `target-busy`,
`busy`, `unknown-mandate`, `text-too-long`, `engine-closed`, `log-full`, ...). A code the adapter does not
know is `ixmux-error`. `detail` is the adapter's own text, for a person; never parse it: for an
ixmux refusal, the adapter writes one message per code. No free text of ixmux (a refusal's detail,
a mandate's reason, a target's error) reaches a tool answer.

| Reason of the adapter | When |
|---|---|
| `ixmux-unavailable` | no `ixmux.json` or no token in `<dir>`, `ixmux.json` says `stopped`, its address is not loopback, ixmux does not answer, or refuses the token |
| `bad-arguments` | an unknown argument, a wrong type, a missing `target` |
| `bad-mandate-id` | a `mandate_id` that is empty, over 128 bytes, not made of letters, digits, `.`, `_` and `-`, or starting with `.` |
| `submits-in-flight` | 4 submissions still wait for ixmux's answer (their POSTs run past the receipt); try again once one ends |
| `ixmux-answer-too-large` | an HTTP answer of ixmux over 4 MiB, or its headers over 64 KiB: the connection is closed, the rest is not read |
| `ixmux-unverified` | the server at the address of `ixmux.json` did not answer the `/v1/hello` challenge with the mac of the token, or answered it with more than 4 KiB (headers included) or a body over 1 KiB: the token was not sent, the rest is not read |
| `ixmux-error` | an answer of ixmux the adapter cannot read, or a refusal without a known code |

What passes from ixmux is data only: identifiers (names, kinds, incarnations, mandate ids,
states, receipts) and a target's answer (`text`), as ixmux gives them. Nothing is scanned; those
fields come only from a server that proved it holds the token.

### `ixmux_status`

Input: `{}`. Output:

```
{"version": "...", "epoch": "...",
 "targets": [{"name": "c", "kind": "claude", "incarnation": "<uuid>", "alive": true,
              "in_flight": [{"mandate_id": "m1", "state": "delivering", "still_running": true}]}]}
```

Metadata only: no text, no answer, no path, no token, no error text. `lost: true` says the
target's kill failed: its process may still run (ixmux's `/v1/status` keeps its `pid`), and it is
not relaunched. `in_flight` lists the target's deliveries that are `accepted` or `delivering`, or
whose adapter call has not returned.

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
  log: either ixmux answered the POST, or `GET /v1/mandates/{id}` shows it in a durable state
  (`accepted`, `delivering`, `delivered`, `refused`, `uncertain`), which the engine shows only after
  the line is synced. A POST cut at the transport proves nothing: the entry is read on until the
  bound.
- `state` is `accepted` or `delivering` (the turn runs on), `delivered` (it was quick, or it is a
  replay: then `replayed: true` and no new turn), `refused` (the target certainly did not take
  it), or `uncertain`. Read from the entry, it may be the state of an earlier submission of the
  same `mandate_id`.
- `accepted: false, state: "submitted"`: ixmux did not confirm within 1.5 s. Nothing is promised;
  `ixmux_result` tells, and the same submission again is safe (at most once per `mandate_id`).
- `receipt` is the target's receipt, once delivered.

Why this design: `POST /v1/mandates` waits for the whole turn. The adapter sends it and lets it
run, polls `GET /v1/mandates/{id}` every 25 ms, and answers at the first proof, or after 1.5 s.
ixmux delivers whether or not the adapter still waits: a killed adapter does not cancel the turn.
It is the smallest change that keeps the promise: no new mode on the server. A POST left running is
bounded: at most 4 at once, each within ixmux's delivery deadline (from `/v1/status`) plus 15 s.

### `ixmux_result`

Input: `{mandate_id}`. Output:

```
{"mandate_id": "m1", "target": "c", "state": "delivered", "last": "delivering->delivered",
 "still_running": false,
 "receipt": "<session>:m1", "text": "ACK m1\n...", "ack_seen": true, "truncated": false}
```

- `state`: `accepted`, `delivering`, `delivered`, `refused` or `uncertain`, and `last` the
  transition that led there. ixmux's reason is not passed on (free text). An `uncertain` mandate is
  never replayed: a new submission is refused `uncertain-needs-reconcile`. Nor may a client send its
  work again under another id: `uncertain` means the target may have taken it. Nothing is
  relaunched automatically; a person decides, from the session's output or a reconcile.
- `cause`, on an `uncertain` mandate, says why, from a closed set: `deadline` (no answer within
  ixmux's delivery deadline), `adapter-error` (the target answered an error, `is_error` included),
  `restart` (ixmux stopped, cleanly or not, or restarted with the delivery under way), `log-write-failed` (its final state
  could not be written). A mandate past its deadline whose call then ends in an error while ixmux
  stops keeps `deadline`, the error added to its `reason`. Any other value from ixmux is left out. Saturation is no state: it is a
  refused submission, `target-busy` or `busy`.
- `receipt`, `text`, `ack_seen` and `truncated` appear once delivered with an answer. `text` is at
  most 4 KiB; `truncated: true` says the answer was longer and its end is missing. A mandate made
  `delivered` by a reconcile (the target said it received it) has none of them: no answer was read.
- A turn longer than ixmux's delivery deadline is `uncertain` with `still_running: true` while it
  runs. Its answer, if it comes before TurnMax, makes it `delivered` with its result, unless the
  result is an error (`is_error`) or acknowledges another mandate: then it stays `uncertain`. Past
  TurnMax the child is killed; if the kill fails, the turn is `uncertain`, unless its result had
  already come.
- It is read from the command log of ixmux (`GET /v1/mandates/{id}`): it survives a restart of the
  adapter and a restart of ixmux.
- An unknown id is `unknown-mandate`: never submitted, or finished (`delivered` or `refused`) more
  than ixmux's deduplication window ago (`--dedup-window`, 720 h by default) and forgotten; that
  id is then a new mandate. An `uncertain` mandate is never forgotten.

### `ixmux_inventory`

No arguments. What can be found again after a crash or a restart, read from the files of the state
directory, so it also answers while ixmux is down (`ixmux_status` then fails `ixmux-unavailable`).
The report (`ixmux.inventory.mcp.v1`) is a reduced view of `ixmux inventory --json`. It holds:

- the ixmux entry, with its `incarnation` (`epoch:…`) and its `status`. `liveness` is `proven` only
  when a live process has the pid of `ixmux.json` and was created when ixmux recorded its start,
  within a minute either way (a clock step). `absent` or `pid-reused` while `ixmux.json` says
  `running` gives `unreachable`: ixmux did not stop cleanly. `unknown` with "read again" means
  `ixmux.json` changed during the read;
- each unfinished mandate of the command log: `mandate_id`, `target`, `incarnation`, the recorded
  `state`, `cause` (from the closed set of `ixmux_result`), and its `status`:
  - `running`: accepted or delivering under a live ixmux;
  - `uncertain`: the target may have acted, and a person decides. That covers a mandate left
    delivering by an ixmux that is gone, which its next start records `uncertain`, cause `restart`;
  - `done`: accepted under an ixmux that is gone, so never sent (its next start records it
    `refused`);
  - `unknown`: its ixmux could not be checked, or started during the read;
- `finished`: the number of delivered and refused mandates. They are not listed: `ixmux_result`
  answers for each id;
- `last_activity` and `age_s`, the last transition. `resume` says what a person may do; the tool does
  nothing;
- `gaps`: what the inventory does not read.

It carries no path, pid, address, exit code or creation time, no Claude Code session of the machine,
and no `reason` (text that may come from a target). It reads the command log without its lock. On
Windows a compaction of a running ixmux fails during that read: no harm, the log is kept as it was
and the compaction is tried again a minute later.

## Stop

stdin closed: the adapter finishes the call under way (2 s at most for `ixmux_submit`) and exits 0.
A POST still running is dropped by the adapter; ixmux goes on and finishes the delivery. ixmux keeps
running in every case.

## What the adapter does not do

- It does not stop ixmux: `/v1/shutdown` is not a tool. It does not reconcile an uncertain mandate
  either (`/v1/mandates/{id}/reconcile` is not a tool).
- No shell, no file access outside `ixmux.json`, `ixmux.token` and, read-only for
  `ixmux_inventory`, `commands.jsonl`; no WMUX session control, no
  tunnel, no service, no network beyond the loopback address of `ixmux.json`.
- It keeps no durable state of its own: what it says of a mandate comes from ixmux. Its own
  answers are `initialize`, the tool schemas, the protocol and validation errors, the
  `submitted` receipt of a submission ixmux did not confirm in time, and the message of each
  refusal code.
- No resources, prompts, sampling or notifications of its own.

## Limits

- A long `ixmux_submit` holds the next calls: the adapter answers one call at a time.
- `ixmux_status` and `ixmux_result` are one request, so one connection, per call. `ixmux_submit`
  makes several (its POST holds a connection for the whole turn while it polls): each on its own
  connection, each after its own challenge.
- The receipt's 2 s are 1.5 s of wait plus the time of one HTTP read on loopback; a machine stalled
  beyond that answers later.
- Tested against the fake claude only (`claudestream/internal/fakeclaude`); no real model call.
