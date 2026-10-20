# ixctl — IX's control plane

French: [`README.fr.md`](README.fr.md).

`ixctl` is a small Go process that supervises the services a spec declares, reconciles them
toward a desired-state document the way a local Terraform would (`plan` says what would change,
`apply` bumps a generation and acts), bounds each run (wall clock, memory) and reports what every
run cost. It serves one loopback HTTP/WebSocket face and federates with peer instances. The Rust
and IXQL client is [`crates/ix-ctl`](../../crates/ix-ctl) (`ix.ctl.status`, `ix.ctl.plan`,
`ix.ctl.apply`, `ix.ctl.stop`, `ix.ctl.wait`, `ix.ctl.events`).

The three formats it defines (the `ixctl.v1` envelope, the desired-state document, the journal)
are **provisional**: see the one-way-doors page,
[`docs/plans/2026-10-17-ixctl-one-way-doors.md`](../../docs/plans/2026-10-17-ixctl-one-way-doors.md),
for their exact shape, what freezing each would lock in, and the review rounds that shaped them.
Nothing outside `go/ixctl` and `crates/ix-ctl` should depend on them until a door is signed.

## Build and run

```sh
cd go/ixctl
go build -o ixctl .            # Go 1.27; deps: coder/websocket, x/sys
go test ./...                  # the unit suite (fake services, in-memory sessions)
./ixctl up spec.json [--addr 127.0.0.1:0] [--name NAME] [--state DIR]
./ixctl bench spec.json        # latency distributions: start, stop, restart, status RTT, fan-out
./ixctl version
```

`up` prints one JSON line when it listens, then runs until SIGINT/SIGTERM or `POST /v1/shutdown`:

```json
{"event":"ready","name":"local","addr":"127.0.0.1:52011","pid":4242,"state_file":".../ixctl/local.json","ms_to_ready":3.1}
```

The listener is loopback only (`--addr` must resolve to 127.0.0.1, `::1` or `localhost`). One
instance per name per state directory: a lock file is held for the instance's life, and a state
file `<name>.json` (`{name, addr, pid, token}`, mode 0600) tells clients where to connect and with
which bearer token. The default state directory is `%LOCALAPPDATA%\ixctl` on Windows, else
`$XDG_STATE_HOME/ixctl` or `~/.local/state/ixctl`. The journal `<name>.journal.jsonl` sits beside
the state file.

## The spec

The spec is the declared world of one instance: the only services it can ever start, and the
peers it federates with. Unknown fields are refused.

```json
{
  "name": "local",
  "services": [
    {
      "name": "pod-a",
      "command": "bin/pod",
      "args": ["--nodes", "4"],
      "dir": ".",
      "env": {"RUST_LOG": "info"},
      "restart": "on-failure",
      "max_restarts": 5,
      "autostart": false,
      "limits": {"wall_ms": 120000, "memory_mb": 2048, "cpu_ms": 60000}
    },
    {
      "name": "job",
      "command": "bin/ix",
      "args": ["pipeline", "run", "--file", "job/ix.yaml", "--json"],
      "restart": "never"
    }
  ],
  "peers": [
    {"name": "b", "state_file": "state/b.json"}
  ]
}
```

| Field | Meaning |
|---|---|
| `name` | Instance or service name: `^[a-z0-9][a-z0-9-]{0,62}$`, unique within the spec. |
| `command`, `args`, `dir`, `env` | What to run. Relative `command`, `dir` and peer `state_file` resolve against the spec file's folder. On Windows a command without extension that does not exist as written is tried with `.exe`. |
| `restart` | `never`, `on-failure` (default) or `always`. A **job** is a service with `restart: never`: it runs to its end and the plane leaves it `stopped`, its `last_exit` saying how. |
| `max_restarts` | Restarts allowed inside a 60 s window (default 5); past it the service is `failed` ("too many restarts"). Backoff doubles from 200 ms to 10 s. |
| `autostart` | Start the service when the plane starts, as generation 1. |
| `limits.wall_ms` | Longest a run may last, from its launch. 0 or absent: no limit. |
| `limits.memory_mb` | Most memory the run's whole process tree may hold, in MiB. 0 or absent: no limit. |
| `limits.cpu_ms` | Most processor time, user and kernel, the whole tree may use, in milliseconds. 0 or absent: no limit. |
| `peers[].state_file` | The state file of a peer instance; the link reads its address and token there and reconnects with backoff (200 ms to 5 s). |

### Limits

A run past a limit is killed, its whole tree with it, and its exit names the limit: `reason` is
`wall_limit`, `memory_limit` or `cpu_limit` on `service.exited` and in `last_exit`, beside the
limit that was set on the event (`wall_ms`, `memory_mb`, or `cpu_limit_ms`, since `cpu_ms` is
what the run used). The restart policy then applies as to any failure, so a job so ended stays
`stopped` and a service restarts with backoff.

- **Wall** is a timer armed at launch.
- **Memory** on Windows is a job-object *notification limit* at `memory_mb` (the message the
  kernel guarantees to deliver) plus a hard cap an eighth above it (at least one MiB), so the
  plane's kill on the notification enforces the limit and the cap bounds how far a single commit
  can overshoot. On Linux the plane polls the process group's resident set from `/proc` every
  100 ms. A run's exit is classified only once its monitor has been drained (one second at most),
  so a timely exit is never misread as a limit.
- **Processor time** is read every 100 ms from the same accounting the exit reports as
  `cpu_ms` (the job's user+kernel time on Windows; the group's `/proc/<pid>/stat` times on
  Linux, waited-for children included), so a run is killed once the figure it would report goes
  past the limit, at most a poll late. The kernel's own job time limit on Windows counts user
  time alone and ends the job itself; the plane does not use it.

Every exit also carries what the run cost: `peak_memory_bytes` (Windows: the job's peak commit;
Linux: the larger of the polled peak and the leader's `ru_maxrss`) and `cpu_ms` (Windows: the
job's user+kernel time; Linux: the larger of the group's polled time and the leader's
`ru_utime + ru_stime`), when the platform could measure them.

## Desired state, plan and apply

A desired-state document names the state each service should be in. Services it does not name
are left as they are. It says *what* is wanted, never *where*: routing (`peer`) travels beside it,
so a journaled document holds no address.

```json
{"schema": "ixctl-desired/0", "services": {"pod-a": {"state": "running"}, "job": {"state": "stopped"}}}
```

- `plan` is pure: it returns `{generation, actions: [{op: start|stop, service}], noop: [...]}`
  against the observed state, with the generation an apply would get.
- `apply` bumps the generation, journals the document, returns `{generation, epoch, actions}` at
  once and acts in the background. Status reports `generation` (last applied),
  `observed_generation` (last whose actions were all seen through) and `settled_generation`
  (last whose services have each reached their desired state); `diverged` names the generation
  the plane gave up on, with the service and reason, and `failed` keeps the recent launch
  failures by generation so a later apply cannot hide an earlier failure.
- `epoch` names the boot these generations belong to. A client that waits on a generation
  should name the epoch too, so a plane restart ends the wait instead of letting the new boot's
  generation of the same number pass for the old one.
- `stop` stops one service out of band and answers once its tree is gone:
  `{stopped, generation, exit}`. A requested stop closes stdin, waits two seconds, then kills.

A job's `desired` stays `running` after it ends, so applying the same document again runs it
again; the client waits on the end of the run *a generation started*
(`last_exit.generation`), not on the last exit the plane saw.

## HTTP and WebSocket

Every request passes the same checks before any route: the `Host` must be the listener's own
(DNS rebinding), an `Origin`, when present, must be the loopback origin of this port, and every
route but `GET /health` needs `Authorization: Bearer <token>` (the token from the state file).
Bodies are read strictly: one JSON value, no unknown field, nothing after it, one MiB at most.

| Route | Does |
|---|---|
| `GET /health` | `{ok, name, version, epoch, uptime_ms, ops}`; no token. |
| `GET /v1/status[?peer=NAME]` | The status reply (above), or that peer's. |
| `POST /v1/plan[?peer=]`, `POST /v1/apply[?peer=]` | Body: a desired-state document. |
| `POST /v1/stop` | Body `{"service": NAME, "peer"?: NAME}`. |
| `POST /v1/shutdown` | Stops every tree, then the transports; the state file is removed. |
| `GET /v1/events?ch=PREFIX&since=SEQ&epoch=E` | Server-sent events: `id: epoch:seq`, `event: <kind>`, `data: <envelope>`. `Last-Event-ID` works like `since`. A reader that falls off the ring gets a `gap` event and the stream ends. |
| `GET /v1/ws` | The `ixctl.v1` WebSocket (below). |

Errors are `{"error": "..."}` with 400 (refused request, undeclared name), 401 (token),
403 (host or origin), 502 (peer down).

### `ixctl.v1` over the WebSocket

Client frames are requests: `{id, op, ...}` with `op` one of `status`, `ping`, `plan`, `apply`,
`stop`, `subscribe`, `unsubscribe`, and the op's fields (`desired`, `service`, `peer`, `ch`,
`since`, `epoch`, `sub`). Every request gets one reply frame `{re: id, ok, data | error}`.
Subscriptions are set up in line (their reply precedes their first event); other ops run on their
own so a slow stop never holds the connection.

Events arrive as envelopes:

```json
{"ch":"svc/pod-a","seq":17,"epoch":"8c1f…","origin":"local","t":1760990000123,
 "ev":{"kind":"stats","t":1760990000120,"data":{"nodes":[...]}}}
```

`seq` counts from 1 at each boot; `epoch` names the boot; `origin` is the instance that emitted
it. A subscriber with `since` replays from the ring (4096 envelopes per instance) and gets
`{sub, gap}` when it cannot: `since-too-old`, `overflow`, `epoch-changed`, `since-ahead`.

### Channels and events

| Channel | Carries |
|---|---|
| `ctl` | The plane's own events: `service.starting`, `service.started` (`pid`, `restarts`, `generation`), `service.exited` (`pid`, `code`, `requested`, `generation`, and when set `reason` + `wall_ms`/`memory_mb`/`cpu_limit_ms`, `peak_memory_bytes`, `cpu_ms`), `service.restarting` (`attempt`, `in_ms`), `service.stopped`, `service.failed` (`reason`), `converged` / `diverged` (`generation`), `peer.up` / `peer.down`. |
| `svc/<name>` | The service's output. A stdout line that is a JSON object with a string `kind` is forwarded as that event (`t` in Unix ms on the service's clock; a `stats` event's `nodes` array becomes the service's `nodes` in status). Any other stdout line is a `log` event `{line}`; stderr lines are `stderr` events. |
| `peer/<name>/<ch>` | A peer's `ctl` and `svc/` channels, mirrored. `subscribe` is always local; ops carry `peer` to be forwarded. |

## Journal

`<name>.journal.jsonl` appends, as JSON lines, a `boot` header per start (`schema:
"ixctl-journal/0"`, `name`, `epoch`), every applied document with its generation and actions, and
every `ctl` event's envelope. It is an audit log, not a store a program should read yet (door 3).

## Gate

`e2e/e2e.py` is the end-to-end gate CI runs on Windows and Linux: the real plane, the real Rust
client through its `ixql-ctl` example, ix-stream's `pod` example as two federated instances, the
real `ix pipeline run` as a job, and runs killed at their wall, memory and processor-time
limits. 35 checks, E1..E13, a restart loop and the M1 measurement (what a wait on the stream
would gain over the 25 ms poll).

```sh
python go/ixctl/e2e/e2e.py --build            # build ixctl, pod, ixql-ctl and ix, then run
python go/ixctl/e2e/e2e.py --bin <dir>        # use binaries already built there
python go/ixctl/e2e/e2e.py --build --work D   # keep binaries, state, runs and results.json in D
```

The job reads the Demerzel constitution from the `governance/demerzel` submodule
(`IX_GOVERNANCE_DIR` overrides it). The gate never prints a token and scans every file it wrote
for one. CI also runs `go vet` and `go test -race` on Linux.

## Known limits

- Windows-first: the gate's liveness check and the process code were written for Windows; the
  Linux memory watch is a poll, and a hard death on Linux is contained for the leader only.
- A single commit refused past the Windows memory cap still depends on the kernel's best-effort
  message; the notification at `memory_mb` is the guaranteed path.
- The processor-time limit is a poll: a run may use up to one poll period of the machine's
  cores past it before the kill. No process-count limit yet.
