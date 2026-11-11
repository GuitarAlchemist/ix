# ixmux

[Français](README.fr.md)

`ixmux` is one durable Go process that serves the `journal` and `deliver` packages over HTTP and
WebSocket, on loopback only. A client that loses its connection reconnects and resumes without
running a mandate twice. On a clean stop the process closes everything properly, and on restart it
picks up its state from the command log.

```
ixmux serve --addr 127.0.0.1:0 --state <dir> [--journal name=path ...] [--target name=path ...]
            [--claude-target name ...] [--claude-bin path] [--claude-env NAME ...]
            [--timeout 30s] [--max-conns 16] [--max-http-conns 64] [--read-timeout 10s] [--queue 256] [--poll 200ms]
ixmux mcp --state <dir>
ixmux version
```

## Build

```
go -C %USERPROFILE%\.cache\ix-ixmux\go\ixmux build -ldflags "-X main.version=<git describe>" -o %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe ./cmd/ixmux
```

`bin/` is ignored by git. `ixmux.json` records the SHA-256 of the running executable.

## API (Bearer token, Host on loopback, no Origin)

| Route | What it does |
|---|---|
| `POST /v1/hello` `{nonce}` | The only route without the token (Host on loopback and no Origin still hold, the body is bounded). The body is one JSON value, `{nonce}`, with `nonce` 32 bytes in lower-case hex; anything else (a second value included) is 400 `bad-nonce`. The answer is `{mac}`: the hex of HMAC-SHA-256(HMAC-SHA-256(token, `ixmux/hello/v1`), the nonce's 32 bytes). A client checks it before it sends the token, on the same connection: then the token goes only to a server that proved it holds it. `ixmux mcp` always does; the runbook below and the other clients send the Bearer alone, and may run the challenge the same way. The derived key signs nothing else, and the journal cursors are signed by another key, random per process. |
| `POST /v1/mandates` `{mandate_id, target, incarnation, text}` | Delivers at most once. The answer gives `state` and `replayed`. A refusal comes back as `{error, detail}`, with 400, 404, 409, 413 or 503. |
| `GET /v1/mandates/{id}` | The state of one delivery and, once delivered, its result: `ack` (`receipt`, `text`, `ack_seen`, `truncated`), read from the command log, so also after a restart. |
| `POST /v1/mandates/{id}/reconcile` | Asks the target whether an uncertain delivery arrived. |
| `GET /v1/status` | Targets and their incarnations, deliveries (age, `still_running`), connections, bounds. |
| `GET /v1/journal/{name}?from=&incarnation=&max_bytes=&cursor=` | One page of `journal.ReadCursor`. Pass back the `cursor` it returns; ixmux signs it with HMAC. A cursor that is unsigned or badly signed is ignored (`cursor_ignored`), and the read starts again from `from`. |
| `GET /v1/stream[?since=<seq>&epoch=<epoch>]` | WebSocket. The first message is `hello`. Then come either the missed events, or a `gap` followed by the state: `state-page` messages, then one `state-end`. Then live events: `mandate`, `journal`, `journal-reset`. |
| `POST /v1/shutdown` | Clean stop. |

How the stream works:
- Events are numbered by `seq` within an `epoch`. The epoch changes each time ixmux starts.
- To resume, pass `since=<seq of the last event>&epoch=<epoch>`.
- A `gap` has one of four reasons:
  - `epoch`: ixmux restarted;
  - `ring`: the events are no longer kept;
  - `ahead`: `since` is past the last event;
  - `slow`: your queue overflowed. The connection then closes with 1008; resume with `since=last_seq`.
- A message that cannot be written within 10 s (the client stopped reading) closes the connection at once, with no `gap`: nothing more can reach that client. Resume with `since`, as after any cut.
- After a `gap`, each `state-page` holds `deliveries` (the last transition of each, its reason cut at 1 KiB), and `state-end` gives `pages`, `deliveries`, `targets` and `journals`. Their `seq` is the last event they cover: the next event is `seq+1`, and no transition they hold comes again as an event.
- Clients must raise their WebSocket read limit to `max_stream_message` (1 MiB). Every message fits in it, the state included.

HTTP bounds: a request's headers and body are read within `--read-timeout`, and its answer is written within 10 s, or within the delivery deadline plus 10 s for a delivery or a reconcile. ixmux holds at most `--max-http-conns` TCP connections, stream connections included; one more is answered 503 `too-many-connections` at once. `--max-conns` bounds the stream connections alone.

## Files in `<state>`

- `commands.jsonl`: the command log of deliveries. While it runs, ixmux holds `commands.jsonl.lock`. It is bounded at 64 MiB, the size a start reads back: no append takes it past that bound. **Each delivery holds room in the log for the records it may still write**: once accepted, its delivering record, its final record (with the answer: a record is at most 32 KiB, and an answer's text past that is cut and said `truncated`) and a late answer; once `uncertain`, the record that settles it (a late answer, or a reconcile's). No other record takes that room, so an accepted delivery always records its outcome, however many run at once. A new mandate that cannot be given its room is refused `log-full`, before anything is sent. A mandate whose id, target and incarnation take more than half of a 1 KiB record is refused `bad-mandate`. A record that still does not fit (in a log written by an older ixmux, or under a lowered bound) is no refusal: the delivery is `uncertain`, cause `log-write-failed`, and the request fails `internal`; so is a delivery whose final record finds the log broken by another one's failed write. **Retention**: a finished mandate (`delivered` or `refused`) is remembered, and so deduplicated (replayed, or refused `mandate-id-reuse`), at least `--dedup-window` after its last transition (720 h, 30 days, by default). The first compaction past that forgets it: its id is then a new mandate, and `GET /v1/mandates/{id}` answers 404. An `uncertain`, `accepted` or `delivering` mandate is never forgotten. The log is compacted when a new mandate finds no room, and at a start past three quarters of its bound or with no room for a new mandate, if that makes it smaller: one record per mandate, in its current state, with its fingerprint, incarnation, receipt and cause. A running ixmux compacts at most once a minute. The compacted log is written to `commands.jsonl.compact`, synced, then renamed over the log (on Windows with write-through). If the restart's own transitions do not fit, they are applied in memory: the log already makes them at every start. The log still fills if the mandates finished within the window do (about 14,000 with 4 KiB answers): a new mandate is then refused `log-full` until older ones leave the window; shorten the window to free it. Never delete the log. The window reads the system clock: a clock set forward forgets sooner. At each start, stderr says what was found and done: `ixmux: command log: N records replayed[, a torn last record of B bytes cut], U made uncertain, R made refused[ (K of them in memory only: the log is full)][, compacted from X to Y bytes[ (F finished mandates past the deduplication window forgotten)]][, compaction not done (the log is kept as read): why]; a finished mandate is remembered at least 720h0m0s`.
- `ixmux.json`: address, PID, start time, version, SHA-256 of the executable, epoch, and `state` (`running` or `stopped`, then `exit_code`).
- `ixmux.token`: the Bearer token. It is rewritten at each start and removed at a clean stop. On Windows, its DACL is protected and grants the current user alone; elsewhere its mode is 0600. If ixmux cannot create it that way, it refuses to start.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Clean stop: no adapter call left, log closed, lock released. |
| 1 | Serving failed. |
| 2 | Bad command line, including a `--claude-target` name given twice (refused before any child is launched). |
| 3 | Start refused: the address is not a loopback IP, the state is held by another ixmux, or the token file failed. |
| 4 | Stopped while deliveries, reconciles or adapter calls were still under way: the engine's `Close` refused with `calls-in-flight`. |
| 5 | Stopped, but the token file could not be removed or `ixmux.json` could not be marked stopped. The error is on stderr; it names the file, never the token. The next start rewrites the token. |

A stop exits 1 if a transition could not be written to the command log (`log-broken`).

A stop (Ctrl+C, Ctrl+Break or `POST /v1/shutdown`) refuses new deliveries and reconciles at once (`engine-closed`), and waits for those under way until their final state is written. It takes at most the delivery deadline (`--timeout`) plus 10 s, plus up to 20 s per claude target (10 s for the child to exit, 10 s more after a kill).

## Runbook (cmd.exe, absolute paths)

The session in this runbook is a fake one: the JSONL file `%USERPROFILE%\.cache\ixmux-fake\session.jsonl`, created at start if missing. Its incarnation names the path and the file there: it is the same across restarts, and another one if another file is put at the path.

1. Start ixmux in its own window:

   ```
   start "ixmux" %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47011 --state %USERPROFILE%\.cache\ixmux-fake\state --target fake=%USERPROFILE%\.cache\ixmux-fake\session.jsonl
   ```

2. Check the status. Note the `incarnation` of the target `fake`.

   ```
   set /p IXMUX_TOKEN=<%USERPROFILE%\.cache\ixmux-fake\state\ixmux.token
   C:\Windows\System32\curl.exe -s -H "Authorization: Bearer %IXMUX_TOKEN%" http://127.0.0.1:47011/v1/status
   ```

3. Send Hello. Replace `<INC>` with that incarnation. The file `session.jsonl` should then hold one line, and the answer should say `"state":"delivered"`. Send it again: the answer says `"replayed":true`, and the file still holds one line.

   ```
   C:\Windows\System32\curl.exe -s -H "Authorization: Bearer %IXMUX_TOKEN%" -H "Content-Type: application/json" -d "{\"mandate_id\":\"hello-1\",\"target\":\"fake\",\"incarnation\":\"<INC>\",\"text\":\"hello\"}" http://127.0.0.1:47011/v1/mandates
   ```

4. Stop it, with Ctrl+C in its window or:

   ```
   C:\Windows\System32\curl.exe -s -X POST -H "Authorization: Bearer %IXMUX_TOKEN%" http://127.0.0.1:47011/v1/shutdown
   ```

   `%USERPROFILE%\.cache\ixmux-fake\state\ixmux.json` then says `"state": "stopped"` and `"exit_code": 0`.

**Rollback:** stop ixmux (step 4), then delete the fake session's state and file:

```
rmdir /s /q %USERPROFILE%\.cache\ixmux-fake
```

Nothing else is written: no service, no registry key, no file outside `--state` and the `--target` files.

## Real session (`--claude-target`)

`--claude-target c` keeps one `claude -p` child open in stream-json, with no tools:

```
claude -p --input-format stream-json --output-format stream-json --verbose --tools "" --strict-mcp-config --session-id <uuid>
```

- Never `--bare`, `--dangerously-skip-permissions` or `--permission-mode bypassPermissions`. `--setting-sources` is left alone, so your hooks apply as they are. ixmux sets no permission mode: the child runs in the mode of your configuration (`auto` here). What protects is that it has no tool.
- The incarnation of the target is the `session_id` of the child's `system/init`. The real claude sends `init` only after a first user message, so ixmux sends a priming message ("ixmux : ouverture de la session. Réponds seulement : OK.") at each start. That costs one model turn per start, and the target is ready about 28 s after the start.
- If `init` lists any tool, has no `tools` array (absent or `null`), or names another session, ixmux refuses to deliver to that target. The check and the write on the child's stdin happen under one lock: an `init` that arrives in between refuses the mandate, and nothing is written.
- Each mandate is one user message: `ixmux mandate_id: <id>`, then the instruction to begin with `ACK <id>`, then the text. The answer carries `ack.receipt` (`<session>:<id>`), `ack.text` (at most 4 KiB; `ack.truncated` is true when the answer was longer and cut) and `ack.ack_seen`.
- A `result` counts only if its `session_id` is the target's incarnation and a turn is pending. Any other `result` is journaled (`{"ixmux":"ignored-result"}`) and ignored.
- The ACK is exact: the answer begins with `ACK <id>`, then a space or the end. `ACK m2-extra` does not acknowledge `m2`.
  - `ack_seen: true`: the answer acknowledges this mandate.
  - `ack_seen: false`: the answer has no ACK. The mandate is still `delivered`, since the result is of this session and this turn, but the model did not confirm which mandate it answers.
  - An ACK naming **another** mandate makes the delivery `uncertain`, never `delivered`, and is journaled (`{"ixmux":"ack-mismatch"}`).
- A reconcile reads `stream.jsonl`: a mandate sent and answered by a `result` of this incarnation becomes `delivered`, with no ack (no answer is read back). A `result` with `is_error: true` proves nothing: the mandate stays `uncertain`.
- The child's environment is a whitelist, not ixmux's whole environment. It holds what Windows, a Unix system and claude need to start (`SystemRoot`, `ComSpec`, `PATH`, `PATHEXT`, `USERPROFILE`, `HOMEDRIVE`, `HOMEPATH`, `APPDATA`, `LOCALAPPDATA`, `TEMP`, `TMP`, `HOME` and a few others, see `claudestream.BaseEnv`), then each variable named with `--claude-env NAME` (repeatable). No variable is added by pattern: a `*_TOKEN` or `*_KEY` reaches the child only when named. Your Claude login comes from your profile.
- A turn that gets no result by the delivery deadline makes the mandate `uncertain`, but the turn stays reserved: no other mandate is sent while it may still answer (those are refused `target-busy`, before anything is written). The delivery's call keeps waiting (`still_running: true` in `/v1/status`), and an answer that comes later makes the mandate `delivered` with its ack, unless it is an error result or acknowledges another mandate (then it stays `uncertain`). The reservation lasts at most `TurnMax`, 10 minutes, plus the relaunch. Past `TurnMax`, ixmux kills the child through its Job and waits up to `StopWait` (10 s) for it to exit; then it starts a new one, under a new incarnation, and the target is free again. A relaunch takes at most `InitWait` (90 s), and one that fails is stopped within 2×`StopWait` more: the reservation lasts at most `TurnMax` + `StopWait` + `InitWait` + 2×`StopWait`. A stop ends the reservation sooner: it stops the child (killed past 10 s), and a replacement launched at the instant of the stop is killed by that stop, within `StopWait` (3×`StopWait` if its kills fail). If a kill fails, or the child does not exit within `StopWait` of it, the incarnation is lost (`kill-failed` in `stream.jsonl`): its turn is `uncertain`, unless its result had already come; the reservation is freed; and nothing replaces the child. `/v1/status` then shows the target with `"state": "lost"` and its `pid` (the process may still run), and a mandate for it is refused 409 `stale-incarnation`. The mandate cannot be sent again under the new incarnation (409 `stale-incarnation`). With the real claude, that relaunch costs one priming turn.
- Each incarnation may add at most 256 MiB of stdout to `stream.jsonl` and 16 MiB to `stderr.log`. Past either cap, the child is killed and the pending turn becomes `uncertain` with the reason `stream-too-large` (or `stderr-too-large`). A stdout line over 8 MiB is cut in the journal, and the pending turn becomes `uncertain`. These caps end the incarnation: the child is not relaunched.
- ixmux waits at most 90 s (`InitWait`) for the child's `init` and its priming answer; past that, the start fails.
- The working directory of the child is `<state>\claude-<name>\`, which ixmux creates. It holds `stream.jsonl` (the raw stdout of each of the target's children in turn, plus `{"ixmux":...}` lines for start, send, result, exit, stop and the cases above) and `stderr.log`. `stream.jsonl` is served as the journal `claude-<name>`.
- The child runs in a kill-on-close Job Object: if ixmux dies, killed or not, the child dies with it. A restart starts a new child, so a new incarnation: a mandate that was delivering becomes `uncertain`, its reconcile answers 409 `stale-incarnation`, and resending it answers 409.
- One real turn often takes longer than the default `--timeout 30s`. Pass `--timeout 120s`.
- A stop also waits up to 10 s for the child to exit once its stdin is closed, then kills the job and waits up to 10 s more. `stream.jsonl` traces it: `{"ixmux":"stop","step":"stdin-closed"}`, then `exited`, `killed` or `kill-failed` with `after_ms`. A child still running its SessionEnd hooks at 10 s is killed with them.

Runbook (cmd.exe), with a throwaway state:

```
start "ixmux" %USERPROFILE%\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47012 --state %USERPROFILE%\.cache\ixmux-claude\state --claude-target c --timeout 120s
```

ixmux writes `ixmux.json` and answers only once the child is ready (about 30 s). `/v1/status` then shows the target `c` with its `incarnation` and `"alive":true`. Send Hello as in step 3, with `"target":"c"`. Stop it as in step 4.

**Pitfall:** a `start` run inside a `cmd /c` whose output is captured inherits the pipe. The caller then waits for ixmux, and for its child, to exit. Run `start` from a real console window.

**Rollback:** stop ixmux, check that no `claude` child is left (`tasklist /fi "imagename eq claude.exe"`), then delete the throwaway state:

```
rmdir /s /q %USERPROFILE%\.cache\ixmux-claude
```

The Claude session itself is kept by claude under your profile, as any `claude -p` session is. ixmux writes nothing else.

## MCP adapter (`ixmux mcp`)

`ixmux mcp --state <dir>` is a local MCP server on stdio over the ixmux serve of that state
directory. It reads the address and the token there at each call, so it follows an ixmux restarted
since, and has four tools: `ixmux_status`, `ixmux_submit` (a receipt within 2 s, even when the
turn lasts longer), `ixmux_result` (read from the command log) and `ixmux_inventory` (the inventory
below, of that state directory alone, read from its files). It has no tool that stops ixmux.
It sends the token only to a server that answered its `/v1/hello` challenge, on the same
connection; a process that took the port without the token gets nothing (`ixmux-unverified`). A
process that can read the token file is outside this model. The contract, with the exact command,
the schemas and the errors: [MCP.md](MCP.md).

## Inventory after a restart (`ixmux inventory`)

`ixmux inventory [--state <dir> ...] [--claude-home <dir>] [--no-claude] [--json]` reads files and
process creation times only. It sends, starts, replays, locks and writes nothing. After a crash, a
power loss or a Windows restart it lists:

- **the local Claude Code sessions**: Claude Code writes a record for each running process in
  `<claude-home>/sessions/<pid>.json` (`--claude-home` defaults to `%CLAUDE_CONFIG_DIR%`, else
  `~/.claude`). That file is undocumented: an unreadable or incomplete record is reported
  `unreadable`, never guessed. A session's `id` is its session id. Its `incarnation` is
  `pid:<pid>@<creation time>`. It is `running` or `idle` only when proven: a live process has that
  pid and was created at that time, exactly on Windows (`procStart`), or within a minute of
  `startedAt` elsewhere. A record left behind by a crash is `unreachable`, with the
  `claude --resume <id>` to run. A session id held by two records is listed under `duplicates`.
  While one of them is live, the others say "do not resume";
- **each `--state` ixmux** and its mandates, as `ixmux_inventory` gives them ([MCP.md](MCP.md)).

`last_activity` is the latest of the record's times and of the transcript file's modification time
(its content is never read). `declared` keeps what a file says apart from what was proven. A bound
(256 session records, 64 KiB per record, the 50 newest finished mandates per ixmux, the log's
64 MiB) is never silent: `omitted` counts what it left out. `gaps` names what is not read: cloud
sessions, which no local file lists and none of which is shown as active, wmux's pane states, and
Windows services (`ixctl resume check`). Exit code 0 whenever the report is printed.

## Limits of this slice

- The fake session (a JSONL file) and one real Claude session in stream-json, with no tools, are the only adapters.
- The real session was run by hand, within 20 model turns. The test suite never starts a real claude: it uses a fake one (`claudestream/internal/fakeclaude`).
- Followed journals start at offset 0 at each start, under a new epoch, and the stream carries their pages with their lines.
- A journal's incarnation names its file (device and inode on Unix, volume and file index on Windows) and its first record. A file system may give a new file the identity of a deleted one: Linux reuses inodes. Two files put one after the other at the same path are then told apart by their first record only, and a checkpoint of an incomplete first record may be read from 0 on the new file instead of being refused. An empty journal names no file: every empty journal has the same incarnation, and a checkpoint taken on one is read from 0 on whatever file is at the path, which loses and repeats nothing (`promoted_from` says so).
- A connection counts as `slow` only once its queue overflows, which needs TCP back-pressure. A client that reads slowly but keeps up is never cut.
- Not tested here:
  - `-race`, because there is no gcc on the machine: it runs in CI only (`go-modules`, ubuntu);
  - the Unix code paths (token mode 0600, `flock`), which run in CI only;
  - Ctrl+C and Ctrl+Break, which only the runbook covers, while the tests stop ixmux through `/v1/shutdown`.
- The runbook was run step by step from PowerShell with these same absolute paths. `start` and `set /p` are cmd.exe built-ins and were not run here.
