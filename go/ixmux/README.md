# ixmux

[Français](README.fr.md)

`ixmux` is one durable Go process that serves the `journal` and `deliver` packages over HTTP and
WebSocket, on loopback only. A client that loses its connection reconnects and resumes without
running a mandate twice. On a clean stop the process closes everything properly, and on restart it
picks up its state from the command log.

```
ixmux serve --addr 127.0.0.1:0 --state <dir> [--journal name=path ...] [--target name=path ...]
            [--claude-target name ...] [--claude-bin path]
            [--timeout 30s] [--max-conns 16] [--max-http-conns 64] [--read-timeout 10s] [--queue 256] [--poll 200ms]
ixmux version
```

## Build

```
go -C C:\Users\spare\.cache\ix-ixmux\go\ixmux build -ldflags "-X main.version=<git describe>" -o C:\Users\spare\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe ./cmd/ixmux
```

`bin/` is ignored by git. `ixmux.json` records the SHA-256 of the running executable.

## API (Bearer token, Host on loopback, no Origin)

| Route | What it does |
|---|---|
| `POST /v1/mandates` `{mandate_id, target, incarnation, text}` | Delivers at most once. The answer gives `state` and `replayed`. A refusal comes back as `{error, detail}`, with 400, 404, 409, 413 or 503. |
| `GET /v1/mandates/{id}` | The state of one delivery. |
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
- After a `gap`, each `state-page` holds `deliveries` (the last transition of each, its reason cut at 1 KiB), and `state-end` gives `pages`, `deliveries`, `targets` and `journals`. Their `seq` is the last event they cover: the next event is `seq+1`, and no transition they hold comes again as an event.
- Clients must raise their WebSocket read limit to `max_stream_message` (1 MiB). Every message fits in it, the state included.

HTTP bounds: a request's headers and body are read within `--read-timeout`, and its answer is written within 10 s, or within the delivery deadline plus 10 s for a delivery or a reconcile. ixmux holds at most `--max-http-conns` TCP connections, stream connections included; one more is answered 503 `too-many-connections` at once. `--max-conns` bounds the stream connections alone.

## Files in `<state>`

- `commands.jsonl`: the command log of deliveries. While it runs, ixmux holds `commands.jsonl.lock`.
- `ixmux.json`: address, PID, start time, version, SHA-256 of the executable, epoch, and `state` (`running` or `stopped`, then `exit_code`).
- `ixmux.token`: the Bearer token. It is rewritten at each start and removed at a clean stop. On Windows, its DACL is protected and grants the current user alone; elsewhere its mode is 0600. If ixmux cannot create it that way, it refuses to start.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Clean stop: no adapter call left, log closed, lock released. |
| 1 | Serving failed. |
| 2 | Bad command line. |
| 3 | Start refused: the address is not a loopback IP, the state is held by another ixmux, or the token file failed. |
| 4 | Stopped while deliveries, reconciles or adapter calls were still under way: the engine's `Close` refused with `calls-in-flight`. |
| 5 | Stopped, but the token file could not be removed or `ixmux.json` could not be marked stopped. The error is on stderr; it names the file, never the token. The next start rewrites the token. |

A stop exits 1 if a transition could not be written to the command log (`log-broken`).

A stop (Ctrl+C, Ctrl+Break or `POST /v1/shutdown`) refuses new deliveries and reconciles at once (`engine-closed`), and waits for those under way until their final state is written. It takes at most the delivery deadline (`--timeout`) plus 10 s, plus up to 10 s per claude target.

## Runbook (cmd.exe, absolute paths)

The session in this runbook is a fake one: the JSONL file `C:\Users\spare\.cache\ixmux-fake\session.jsonl`, created at start if missing. Its incarnation names the path and the file there: it is the same across restarts, and another one if another file is put at the path.

1. Start ixmux in its own window:

   ```
   start "ixmux" C:\Users\spare\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47011 --state C:\Users\spare\.cache\ixmux-fake\state --target fake=C:\Users\spare\.cache\ixmux-fake\session.jsonl
   ```

2. Check the status. Note the `incarnation` of the target `fake`.

   ```
   set /p IXMUX_TOKEN=<C:\Users\spare\.cache\ixmux-fake\state\ixmux.token
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

   `C:\Users\spare\.cache\ixmux-fake\state\ixmux.json` then says `"state": "stopped"` and `"exit_code": 0`.

**Rollback:** stop ixmux (step 4), then delete the fake session's state and file:

```
rmdir /s /q C:\Users\spare\.cache\ixmux-fake
```

Nothing else is written: no service, no registry key, no file outside `--state` and the `--target` files.

## Real session (`--claude-target`)

`--claude-target c` keeps one `claude -p` child open in stream-json, with no tools:

```
claude -p --input-format stream-json --output-format stream-json --verbose --tools "" --strict-mcp-config --session-id <uuid>
```

- Never `--bare`, `--dangerously-skip-permissions` or `--permission-mode bypassPermissions`. `--setting-sources` is left alone, so your hooks apply as they are. ixmux sets no permission mode: the child runs in the mode of your configuration (`auto` here). What protects is that it has no tool.
- The incarnation of the target is the `session_id` of the child's `system/init`. The real claude sends `init` only after a first user message, so ixmux sends a priming message ("ixmux : ouverture de la session. Réponds seulement : OK.") at each start. That costs one model turn per start, and the target is ready about 28 s after the start.
- If `init` lists any tool, or names another session, ixmux refuses to deliver to that target.
- Each mandate is one user message: `ixmux mandate_id: <id>`, then the instruction to begin with `ACK <id>`, then the text. The answer carries `ack.receipt` (`<session>:<id>`), `ack.text` (at most 4 KiB) and `ack.ack_seen`.
- The working directory of the child is `<state>\claude-<name>\`, which ixmux creates. It holds `stream.jsonl` (the child's raw stdout, plus `{"ixmux":...}` lines for start, send, result and exit) and `stderr.log`. `stream.jsonl` is served as the journal `claude-<name>`.
- The child runs in a kill-on-close Job Object: if ixmux dies, killed or not, the child dies with it. A restart starts a new child, so a new incarnation: a mandate that was delivering becomes `uncertain`, its reconcile answers 409 `stale-incarnation`, and resending it answers 409.
- One real turn often takes longer than the default `--timeout 30s`. Pass `--timeout 120s`.
- A stop also waits up to 10 s for the child to exit once its stdin is closed, then kills the job. `stream.jsonl` traces it: `{"ixmux":"stop","step":"stdin-closed"}`, then `exited` or `killed` with `after_ms`. A child still running its SessionEnd hooks at 10 s is killed with them.

Runbook (cmd.exe), with a throwaway state:

```
start "ixmux" C:\Users\spare\.cache\ix-ixmux\go\ixmux\bin\ixmux.exe serve --addr 127.0.0.1:47012 --state C:\Users\spare\.cache\ixmux-claude\state --claude-target c --timeout 120s
```

ixmux writes `ixmux.json` and answers only once the child is ready (about 30 s). `/v1/status` then shows the target `c` with its `incarnation` and `"alive":true`. Send Hello as in step 3, with `"target":"c"`. Stop it as in step 4.

**Pitfall:** a `start` run inside a `cmd /c` whose output is captured inherits the pipe. The caller then waits for ixmux, and for its child, to exit. Run `start` from a real console window.

**Rollback:** stop ixmux, check that no `claude` child is left (`tasklist /fi "imagename eq claude.exe"`), then delete the throwaway state:

```
rmdir /s /q C:\Users\spare\.cache\ixmux-claude
```

The Claude session itself is kept by claude under your profile, as any `claude -p` session is. ixmux writes nothing else.

## Limits of this slice

- The fake session (a JSONL file) and one real Claude session in stream-json, with no tools, are the only adapters.
- The real session was run by hand, within 20 model turns. The test suite never starts a real claude: it uses a fake one (`claudestream/internal/fakeclaude`).
- Followed journals start at offset 0 at each start, under a new epoch, and the stream carries their pages with their lines.
- A connection counts as `slow` only once its queue overflows, which needs TCP back-pressure. A client that reads slowly but keeps up is never cut.
- Not tested here:
  - `-race`, because there is no gcc on the machine;
  - the Unix code paths (token mode 0600, `flock`), which are compiled but not run;
  - Ctrl+C and Ctrl+Break, which only the runbook covers, while the tests stop ixmux through `/v1/shutdown`.
- The runbook was run step by step from PowerShell with these same absolute paths. `start` and `set /p` are cmd.exe built-ins and were not run here.
