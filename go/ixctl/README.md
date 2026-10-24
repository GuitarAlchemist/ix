# ixctl — IX's control plane

French: [`README.fr.md`](README.fr.md).

`ixctl` is a small Go process that supervises the services a spec declares, reconciles them
toward a desired-state document the way a local Terraform would (`plan` says what would change,
`apply` bumps a generation and acts), bounds each run (wall clock, memory) and reports what every
run cost. It serves one loopback HTTP/WebSocket face and federates with peer instances. The Rust
and IXQL client is [`crates/ix-ctl`](../../crates/ix-ctl) (`ix.ctl.status`, `ix.ctl.plan`,
`ix.ctl.apply`, `ix.ctl.stop`, `ix.ctl.wait`, `ix.ctl.events`, and one `ix.ctl.probe.*`
capability per probe op).

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
./ixctl up spec.json [--addr 127.0.0.1:0] [--name NAME] [--state DIR] [--bookmarks disk|memory] [--probe-rate N] [--probe-tree-rate N]
./ixctl bench spec.json        # latency distributions: start, stop, restart, status RTT, fan-out
./ixctl probes                 # the probe kinds this binary carries and the system's limits (JSON, offline)
./ixctl version
```

`up` prints one JSON line when it listens, then runs until SIGINT/SIGTERM or `POST /v1/shutdown`:

```json
{"event":"ready","name":"local","addr":"127.0.0.1:52011","pid":4242,"state_file":".../ixctl/local.json","ms_to_ready":3.1}
```

The listener is loopback only (`--addr` must be a loopback IP, 127.0.0.0/8 or `::1`; a hostname,
`localhost` included, is refused). One
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
      "limits": {"wall_ms": 120000, "memory_mb": 2048, "cpu_ms": 60000, "processes": 16}
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
| `limits.processes` | Most processes the tree may hold at once. 0 or absent: no limit. |
| `peers[].state_file` | The state file of a peer instance; the link reads its address and token there and reconnects with backoff (200 ms to 5 s). |
| `peers[].probes` | The peer's probe nodes whose pipes this instance wants mirrored as `peer/<name>/probe/<node>`, by exact name (D4, ix#402). A pipe crosses only while the peer marks the node `export: true`; the status says what each wanted name is. Empty or absent: no probe pipe crosses. |

### Limits

A run past a limit is killed, its whole tree with it, and its exit names the limit: `reason` is
`wall_limit`, `memory_limit`, `cpu_limit` or `process_limit` on `service.exited` and in
`last_exit`, beside the limit that was set on the event (`wall_ms`, `memory_mb`, `processes`, or
`cpu_limit_ms`, since `cpu_ms` is what the run used). The restart policy then applies as to any
failure, so a job so ended stays `stopped` and a service restarts with backoff.

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
- **Processes** are counted by the same 100 ms poll (the job's active processes on Windows, the
  group's live members on Linux) and reported as `peak_processes`; a run whose count goes past
  the limit is killed. On Windows the job object also refuses a process past a cap an eighth
  above the limit (at least one), so a fork storm is bounded between two polls, and the
  kernel's message on that refusal (best effort, like the memory cap's) names the limit too.
  The tree is read once more as it is killed, so a peak the watch had no tick for (the cap's
  message arrives within milliseconds of the spawn) is still reported.

Every exit also carries what the run cost: `peak_memory_bytes` (Windows: the job's peak commit;
Linux: the larger of the polled peak and the leader's `ru_maxrss`), `cpu_ms` (Windows: the job's
user+kernel time; Linux: the larger of the group's polled time and the leader's
`ru_utime + ru_stime`) and, under a processor-time or process limit, `peak_processes`, when the
platform could measure them. A limit read from a reported figure (`cpu_ms`, `peak_processes`)
is applied to that figure at the exit too: a run that went past it and ended before a poll could
kill it still carries the reason.

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

Status announces no consumer contract yet: it will carry `contract: "run/1"` once every slice of
that contract has landed (#443), and a consumer relies on none of run/1 before it reads that.
One generation can carry several runs when a service restarts automatically:

| Status field | Meaning |
|---|---|
| `services[].run` | The latest run announced by `service.starting`. It is the active run in `starting` (once announced) and `running`; in restart backoff (`restarting`) it is the run whose exit caused the backoff, until the successor is announced; `stopped` and `failed` retain the last run. Before the first announcement it is absent; a queued start may already show `starting` while retaining the preceding run, or no run before the first. |
| `services[].last_exit.run` | The exact run that produced `last_exit`; after a restart starts it may differ from `services[].run`. |

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
| `POST /v1/shutdown` | Stops every tree and every probe, then the transports; the state file is removed. |
| `GET /v1/probes[?peer=]` | The probe kinds and the system's limits (`probe.kinds`, below). |
| `GET /v1/probe[?peer=][&with_cursor=1]` | The probe tree (`probe.list`); with `with_cursor=1`, bound to a cursor of the plane's stream. |
| `POST /v1/probe/create[?peer=]` | Body: a node (`probe.create`). |
| `POST /v1/probe/start`, `stop`, `touch`, `remove` | Body `{"node": NAME, "peer"?: NAME}`. |
| `GET /v1/bookmark[?peer=]` | The bookmarks (`bookmark.list`, below). |
| `POST /v1/bookmark/set` | Body `{"bookmark": NAME, "since"?: SEQ, "epoch"?: E, "note"?: TEXT, "peer"?: NAME}`. |
| `POST /v1/bookmark/remove` | Body `{"bookmark": NAME, "peer"?: NAME}`. |
| `GET /v1/events?ch=PREFIX&since=SEQ&epoch=E`, or `?ch=PREFIX&bookmark=NAME` | Server-sent events: `id: epoch:seq`, `event: <kind>`, `data: <envelope>`. `Last-Event-ID` works like `since`; `bookmark` replays from a bookmark instead (not beside `since`). A reader that falls behind the ring while reading gets a `gap` event (`overflow`) and the stream ends. A resume point the ring no longer holds (`since-too-old`), or one from another boot (`epoch-changed`, `since-ahead`), gets a `gap` event first, then the stream goes on from the oldest envelope the ring still holds. |
| `GET /v1/ws` | The `ixctl.v1` WebSocket (below). |

Errors are `{"error": "..."}` with 400 (refused request, undeclared name), 401 (token),
403 (host or origin), 502 (peer down).

### `ixctl.v1` over the WebSocket

Client frames are requests: `{id, op, ...}` with `op` one of `status`, `ping`, `plan`, `apply`,
`stop`, `subscribe`, `unsubscribe`, the seven `probe.*` ops and the four `bookmark.*` ops (below),
and the op's fields (`desired`, `service`, `peer`, `ch`, `since`, `epoch`, `sub`, `probe`, `node`,
`bookmark`, `note`). Every request gets one reply frame `{re: id, ok, data | error}`.
Subscriptions are set up in line (their reply precedes their first event); other ops run on their
own so a slow stop never holds the connection.

Events arrive as envelopes:

```json
{"ch":"svc/pod-a","seq":17,"epoch":"8c1f…","origin":"local",
 "run":{"origin":"local","epoch":"8c1f…","seq":12},"t":1760990000123,
 "ev":{"kind":"stats","t":1760990000120,"data":{"nodes":[...]}}}
```

`seq` counts from 1 at each boot; `epoch` names the boot; `origin` is the instance that emitted
it; a mirrored envelope (a `peer/` channel) also carries `src_seq` and `src_epoch`, the
source's own seq and boot, so a reader here keys and dedups as one on the origin would (S2
slice 2). A subscriber with `since` replays from the ring (4096 envelopes per instance) and gets
`{sub, gap}` when it cannot: `since-too-old`, `overflow`, `epoch-changed`, `since-ahead`.

`run = {origin, epoch, seq}` names one launch attempt; its `seq` is that attempt's
`service.starting` sequence on its origin plane. It is fixed before output, immutable, and unique
within an origin and epoch. Every `svc/<name>` envelope of the attempt and every `ctl` service
event attributable to it carries the same `run`. An automatic restart gets another run while
keeping the generation that requested the service. A mirrored envelope (`peer/<name>/…`) does
not carry it yet (#452).

### Channels and events

| Channel | Carries |
|---|---|
| `ctl` | The plane's own events: `service.starting` (`restart`, `generation`), `service.started` (`pid`, `restarts`, `generation`), `service.exited` (`pid`, `code`, `requested`, `generation`, and when set `reason` + `wall_ms`/`memory_mb`/`cpu_limit_ms`/`processes`, `peak_memory_bytes`, `cpu_ms`, `peak_processes`), `service.restarting` (`attempt`, `in_ms`), `service.stopped`, `service.failed` (`reason`), `converged` / `diverged` (`generation`), `peer.up` / `peer.down`. Service events attributable to a run carry it in the envelope. |
| `svc/<name>` | The service's output. Every envelope carries its run. A stdout line that is a JSON object with a string `kind` becomes that event, of which only `kind`, `t` and `data` are kept: any other top-level field is dropped (`t` in Unix ms on the service's clock, rounded, and dropped when it is not a finite number; a `null` `data` is no data; a `stats` event's `nodes` array becomes the service's `nodes` in status). Any other stdout line is a `log` event `{line}`; stderr lines are `stderr` events. |
| `probe/<node>` | A probe node's pipe: `sample` events (below). Mirrored to a peer only when this node is created with `export: true` and that peer's spec names it (D4, ix#402). |
| `peer/<name>/<ch>` | A peer's `ctl` and `svc/` channels, mirrored, plus the probe pipes `probe/<node>` this instance asked for in `peers[].probes` while the peer exports them; and the link's own `link.gap` envelopes, the continuity markers (below). `subscribe` is always local; ops carry `peer` to be forwarded. |

**Probe pipes across the link (D4, ix#402, resolved 2026-10-21; S2 ix#408, slice 1).** What crosses
is the intersection of two declarations: the child marks a node exportable at create (`export:
true`, immutable, false by default: an availability, not a request) and the parent's peer spec
names exactly the nodes it wants (`peers[].probes`). The link subscribes to each wanted node's
pipe by its exact channel name (`probe/cpu` does not cover `probe/cpu-extra`), from the peer's
`probe.list` when the link comes up and from the mirrored `probe.created` / `probe.removed`
after; `status.peers[].probes` says what each wanted name is: `mirrored`, `not-exportable`,
`opening` (subscribing), or `absent` (no such node, or the link down). A pipe starts live on
every session and after every drop, never from a seq: the same channel name can carry another
incarnation's records; since slice 4 the link knows which incarnation it mirrors (the cursor-bound
list below, `created_seq`, on `status.peers[].probes[]`), so a resume from the last seq mirrored
would now be sound and is left for later; slice 2 brought the identity, slice 3 the marker for what
a live start skips (contract G8 closed) and the bounded catch-up. Aggregation lives with the reader: in V1 the reader
is the consumer of `peer/<name>/probe/<node>` (IXQL `ix.ctl.events`), no local node names a
remote source. Export is a federation rule, not a confidentiality barrier: direct subscription to
a pipe is unchanged. Slice 2 adds the source's `src_seq` / `src_epoch` on every mirrored
envelope. Slice 3 marks what the mirror skips, in the mirrored stream itself: a `link.gap`
envelope the parent publishes (its own origin, no provenance, never a count) on the pipe at
every live start (`reason: live-start`, `src_epoch` the boot now read, `last_src_seq` /
`last_src_epoch` the last frame mirrored on it before, when any) and on `peer/<name>/ctl` when
the peer came back as a new instance (`epoch-changed`, `last_src_epoch` the boot before) or
answered a gap on the link's resumed `ctl` / `svc/` (`prefix`, `reason`, `from`, `oldest` as it
said: its history expired); a local gap stays a gap frame. The link's queues are bounded per
class (criterion 6 of ix#408): `ctl` and `svc/` in their own queue, never shed by the link and read first (a control frame waits at most one pipe frame's apply; the peer's own overflow of those subscriptions reopens them in the session, resumed or live as below, and a replay its ring no longer holds is a gap, marked); the
wanted pipes in a queue bounded in records (1024) and in bytes (4 MiB) that their forwarders
never wait on for a record (the one frame a forwarder waits to queue is its subscription's
overflow gap, its last), so a parent slower than the offered load drops pipe records rather than
grow or hold the lifecycle back; each run of drops is counted on `status.peers[].probes[].dropped` and
marked on the pipe (`link.gap`, `reason: dropped`, `dropped` the link's own exact count) before
the record that followed, and a pipe subscription the peer itself dropped (`overflow`) is marked
and reopened live without ending the session. A catch-up of `ctl` / `svc/` (a resume after an
outage or after the peer dropped the subscription) is bounded by an allowance the link owns
(criterion 7): 8192 frames and 32 MiB per link, refilled to full over a minute, at most 30 s per
catch-up, never reset by a reconnect. A catch-up begins only with a frame and a byte of it left
and ends at the first frame it cannot cover: the link then subscribes live instead and marks the
jump on `peer/<name>/ctl` (`link.gap`, `reason: jumped`, `prefix`, `from` the last seq mirrored on
that prefix, `to` the peer's head the live subscription starts from, `src_epoch`, dated by `t`),
so a parent that cannot keep up replays once, bounded, then jumps: it replays at most one
allowance per minute of refill, reconnects for none of it (a dropped control subscription is
reopened in the same session) and does not grow. A live start that misses nothing (the peer's
head is the last seq mirrored on the prefix, in the same boot) is no jump and marks nothing; the
span a marker names is of the head every channel shares, so it may hold frames of other channels
only (the marker over-reports, never under). The parent's `status.peers[]` tells `live` (the link is up and both control
subscriptions are past their replay: the link asks the peer to say so, `subscribe` with
`live: true` answered by one `{"sub", "live": true}` frame once the replay is past the head the
subscription started from, at once when nothing was to be replayed; no other subscriber gets it)
from `reconciled` (the link is up, the peer's probe list was applied this session and no wanted
pipe is still opening), lifts `degraded` only when both hold, and counts the catch-ups given up in
`jumps`. A jump of `ctl` loses the lifecycle events of its span: the state is no longer known
reconciled (`reconciled` false, `degraded`, from the step that gives the catch-up up) until the
link reads the list again, which it does on its session loop right after the live subscription
stands, as the snapshot below; see the resolution on ix#402.

**The list as a snapshot (S2 slice 4, criteria 3 and 4 of ix#408).** `probe.list` taken with
`with_cursor` (`GET /v1/probe?with_cursor=1`, IXQL `ix.ctl.probe.list(with_cursor: true)`) says
the point of the plane's own stream it stands at, `cursor: {seq, epoch}` (the head and the boot),
read under the tree's lock with the nodes, where every lifecycle event is published: which nodes
exist, their kind, parent and export, which are running, are exactly what `ctl` says up to that
seq and nothing of what follows, so a reader can fold the stream from the cursor on without a
hole or a repeat (what no event says, `stopping`, `expires_in_ms`, the counters, is the tree's own
reading at the cursor). Every node carries `created_seq`, the seq of
its `probe.created`: the identity of the incarnation under that name (a same-name recreation is a
new one). A leaf whose stop is in progress says `state: "stopping"` (its `probe.stopped` is not out
yet: `ctl` still says running, a start is refused until it is; the event and the end of the stop
are one step, so the list never says `stopping` once `ctl` said stopped). The link reads its list
so: a lifecycle event at or before the cursor is already reflected and not followed; a list that
is no snapshot (no cursor, a wanted node without `created_seq`) is refused and ends the session;
each wanted pipe carries the identity it mirrors (`status.peers[].probes[].created_seq` /
`src_epoch`, from the list or from the followed `probe.created`), so a consumer that reads the
status, then the pipe, keys what it reads from then on `origin:src_epoch:node:created_seq` without
having read that event (the status is a reading, not a point of the stream: a frame read before
it is told apart only by the mirrored `probe.created` or the list with its cursor); after a jump
on `ctl` the link reads the list again on its session loop, holding the control frames that
arrive meanwhile (bounded as at session start) and applying them after it, and reconciles to it:
a wanted node gone from the list is dropped, a name whose identity changed is reopened as the new
incarnation (its live start marked as any), and `reconciled` holds again. Tested on a churning
tree (25 snapshots folded from the stream, leaves stopping under the list), on the link's
identities and skips, on the held frames, on `reconciled` falling at the jump, and on a flood of
`ctl` that makes the link jump and read the list again; gate E23 on the two planes.

`ctl` also carries the probe lifecycle: `probe.created` (`node`, `kind`, `parent`, `export` when set),
`probe.started` (`expires_in_ms` when leased), `probe.stopped` and `probe.removed` with a `reason`
(`requested`, `expired`, `removed`, `parent-removed`, `shutdown`).

## Probes

A probe is a sampler that runs **inside the plane's process** on a timer and publishes what it
reads on its own channel; a tree of probes filters and multiplexes those samples toward a root.
The plane carries its probe **kinds**, versioned and compiled in (`name@version`), and knows its
own limits: `probe.kinds` lists both, every refusal names the bound it hit, and `probe.list` shows
the usage. The design, the decisions it assumes and the ones still open are on the probes page,
[`docs/plans/2026-10-20-ixctl-probes.md`](../../docs/plans/2026-10-20-ixctl-probes.md), and the
work is tracked under epic ix#397. The surface is provisional, like the three doors.

| Kind | Fields | Reads |
|---|---|---|
| `heartbeat@1` | `n` | Its own tick count, from 1. The kind tests and demonstrations use. |
| `cpu@1` | `busy_pct`, `cores` | The host's processor busy share over the last interval (`GetSystemTimes`; `/proc/stat`). The first interval emits nothing. |
| `mem@1` | `total_bytes`, `avail_bytes`, `used_pct` | The host's physical memory (`GlobalMemoryStatusEx`; `/proc/meminfo`). |
| `delta@1` | `event`, `journal_seq`, `t_ms`, `t`, `pid`, `offset`, and the event's own (`utterance_id`, `text`, `asr_ms`, `route`, `triage`, `triage_ms`, `parrot`, `promise`, `say_id`, `by`, `request`, `cost`, `muted`, `version`, `replaced_by`) | The JSONL journal of Delta, the voice robot, named by `ixctl up --delta-journal FILE`: one line per tick, every top-level field of the line as it is, plus `offset`, the line's byte position. Without the flag the kind is listed and refused at create. |
| `mux@1` | (its children's) | No sampler: it carries what its children pass it. |

`delta@1` is the plane *reading* a file another program writes (brickyard's `brain.exe`, which
journals what Delta heard and said); nothing is pushed into the plane. One line per tick keeps
the node's declared rate (1000 / `every_ms`) the bound whatever the journal's pace: a burst
drains at that rate and nothing is dropped. A line that is not a JSON object, or is longer than
1 MiB (`max_line`), is an error record naming its offset, and the lines after it flow; a missing
journal is said once and waited for; a journal shorter than what was read is a new one under the
same name, said once and read again from offset 0. `node`, `kind` and `seq` are the tree's
stamps and win over a line field of the same name. A consumer dedups on the writer's pair
(`pid`, `journal_seq`), which restarts with each brain, and resumes on `offset`, which belongs
to the file.

A **node** is one instance of a kind in the tree:

```json
{"name": "hb", "kind": "heartbeat@1", "every_ms": 20, "parent": "agg", "ttl_ms": 1500,
 "filter": {"where": [{"field": "n", "op": ">", "value": 2}], "keep": ["n"], "every": 2}}
```

`every_ms` is the sampling interval (a `mux@1` has none). `parent` names a running or stopped
`mux@1` node; a sample a node emits goes on its own pipe `probe/<name>` and, when its parent is
running, up to the parent, which applies its own filter and publishes it on `probe/<parent>`
stamped with `src` (the child's name). Propagation is synchronous: nothing is queued and nothing
is dropped on the way up; the only loss is the ring's, and it is a `gap` the reader sees.
A `filter` is applied before a sample is emitted, in this order: `where` (every term must hold;
`op` is one of `==`, `!=`, `<`, `<=`, `>`, `>=`; `value` is a number, a string or a bool, a bool
with `==` or `!=` only, anything else refused at creation; a missing field or a type mismatch is
false), `every` (one sample in N of those that passed), `keep` (the fields kept;
`node`/`kind`/`seq`, a mux's `src` and a failed sample's `error` always). A sample is `{node, kind, seq, src?, ...fields}`; a read error becomes
`{node, kind, seq, error}`, counts in the node's `errors`, and reaches the pipe whatever the
filter (`where` and `every` do not apply to it). A node's records reach its pipe in `seq` order,
and `probe.stopped` is the last thing its pipe says until it starts again; `ctl`'s lifecycle
events come in the order of the state changes they report, whatever the interleaving of the ops.

`ttl_ms` is a **lease**: counted from `probe.start`, renewed by `probe.touch`, and when it runs
out the node is removed with its whole subtree (`probe.removed`, reason `expired`). A node
without `ttl_ms` has no lease and `probe.touch` refuses it. `probe.remove` takes the subtree too
(descendants are stopped and removed with reason `parent-removed`); `probe.stop` stops one node
and leaves its children running (their samples stay on their own pipes). A node a remove has
claimed refuses a start, a touch or a new child until it is gone; a node whose stop is still
waiting for its last tick refuses a start until `probe.stopped` is out. `POST /v1/shutdown`
closes admission (no create or start from then on) and stops every running node, reason
`shutdown`.

| Op | Fields | Reply |
|---|---|---|
| `probe.kinds` | | `{kinds: [{kind, version, fields, doc}], limits, version}` |
| `probe.create` | `probe` (a node) | `{node, kind, state: "stopped", depth}` (`depth`: the muxes above it) |
| `probe.list` | `with_cursor?` | `{nodes: [{name, kind, state, every_ms, ttl_ms, seq, errors, bytes, children, rate_per_s, created_seq, max_rate_per_s?, export?, parent?, filter?, expires_in_ms?}], count, max_nodes, rate_per_s, max_rate_per_s, tree_rate_per_s, cursor?: {seq, epoch}}` (`state` also `stopping` for a leaf whose stop is in progress; `cursor` with `with_cursor`) |
| `probe.start` | `node` | `{node, state: "running", expires_in_ms?}` (a noop on a running node) |
| `probe.stop` | `node` | `{node, state: "stopped"}` |
| `probe.touch` | `node` | `{node, state, ttl_ms, expires_in_ms?}` (the time left, on a running node) |
| `probe.remove` | `node` | `{node, removed}` (nodes removed, the subtree counted) |

The limits `probe.kinds` reports (and `ixctl probes` prints offline): `max_nodes` 256 in one
plane's tree, `max_depth` 8 (a leaf under at most this many muxes), `min_every_ms` 10,
`max_every_ms` 3 600 000, `max_ttl_ms` (what a duration holds), `ring` 4096 (the hub's, shared
with every channel), `max_where` 16 terms, `max_keep` 32 fields. A create past one of them is
refused with its name (`every_ms 5 is under min_every_ms 10`). Every `probe.*` op is forwardable
to a peer (`peer`); the peer's `probe.*` lifecycle comes back mirrored on `peer/<name>/ctl`,
its pipes do not (D4). From IXQL, each op is one capability named after it
(`ix.ctl.probe.create(probe: {…})`, `ix.ctl.probe.start(node: "agg")`, …) and a pipe is read
with `ix.ctl.events(channel: "probe/<node>")`; there is no wait on a sample (D7, ix#405).
[`crates/ix-ctl/examples/probe.ixql`](../../crates/ix-ctl/examples/probe.ixql) is the program:
a CPU probe under a mux, ten records, the tree removed.

Since D10 (ix#416) the limits also carry the plane's **budget**: `max_rate_per_s` 1 000, what one
subtree (a mux and its descendants, or a root leaf alone) may declare to emit in records per
second, and `tree_rate_per_s` 2 500 for the whole tree; `ixctl up --probe-rate N
--probe-tree-rate N` sets them (the defaults are about a quarter of a core, extrapolating M2).
The declared rate is the worst case, summed at `create` whatever the nodes' state: 1000 /
`every_ms` per leaf. The filter does not lower it: a failing sampler emits an error record every
interval, and errors bypass `where` and `every`. The sum is exact (rationals, not floats or floored
integers): twelve leaves at 12 ms fill 1 000 to the unit and are admitted; 32 leaves at 31 999 ms
are over a budget of 1 by a hair and refused. The budget is set before any node
exists and frozen from the first.
A mux's `max_rate_per_s` narrows the budget its subtree is held to and never widens its parent's
or the plane's; it is immutable (remove and recreate); a leaf carries none. A create that would
cross a bound is refused in advance with the subtree's name and the bound (`subtree "agg" would
declare 1100 records/s with "cpu-10"; its max_rate_per_s is 1000`), like `max_nodes`: nothing is
slowed or stopped for budget, and no event is published. `probe.list` says the usage: each node's
`rate_per_s`, each mux's `max_rate_per_s`, the tree's `rate_per_s` against `tree_rate_per_s`,
and `bytes`, the encoded records each node emitted, measured and not bounded. CPU and RSS are the
process's and are not attributed to probes: one process cannot separate them honestly.

## Bookmarks and replay

A **bookmark** names one point in the instance's sequence: `{name, epoch, seq, t, note}`. The
hub's `seq` is shared by every channel, so one bookmark covers a whole probe tree's pipes, the
services' pipes and the `ctl` lifecycle at the same instant: a replay from it gives back what
every reader saw after that point, in the order it happened. Set `before`, start the tree, set
`after` when something looks wrong, and replay `probe/` (the pipes) or `` (everything) from
`before`: what the pipeline saw, after the fact.

The plane owns the bookmarks. `ixctl up --bookmarks disk` (the default) keeps them in
`<name>.bookmarks.json` beside the journal, rewritten whole on every change (a temporary file,
then a rename; mode 0600), so they survive a restart; `--bookmarks memory` keeps them for the boot
only. A file the plane cannot read as bookmarks is refused at startup, not emptied. The replay
source is the ring (4096 envelopes): a bookmark older than it answers `since-too-old` first, then
what the ring still holds, never a silently partial replay; a bookmark of an earlier boot answers
`epoch-changed` first, then the new boot's ring. The journal is not replayed; durable restoration
would be a separate contract that announces itself (D9, ix#414; door 3 untouched).

| Op | Fields | Reply |
|---|---|---|
| `bookmark.set` | `bookmark`, `since?`, `epoch?`, `note?` | `{bookmark, epoch, seq, t, note, head}`. Without `since` the point is the head ("here"); `since` must be at or below the head and `epoch`, when given, this boot's. An existing name is moved; a `note` given replaces its note (an empty one clears it), no `note` keeps it. |
| `bookmark.get` | `bookmark` | `{name, epoch, seq, t, note}` |
| `bookmark.list` | | `{bookmarks: [{name, epoch, seq, t, note, live, retained}], count, store, head, oldest, max_bookmarks, max_note_len, ring}`; `live` says this boot set it; `retained` says the ring still holds everything after its point (of this boot, and `seq + 1 >= oldest`, the oldest seq held), so a replay from it opens with no gap. Both are read at one instant; the ring keeps turning. |
| `bookmark.remove` | `bookmark` | `{bookmark, removed: true}` |

`subscribe` (`ixctl.v1`) and `GET /v1/events` take `bookmark` in place of `since`/`epoch`; both
together are refused ("bookmark or since"), except that a `Last-Event-ID` header beside a
`bookmark` query is a reconnect of a stream already past the bookmark (an `EventSource` sends
its original URL back with the header) and resumes from the header. Precedence on
`/v1/events`: the header wins over a query `since` and over a `bookmark` (a bare seq in the
header takes the query `epoch`, `epoch:seq` carries its own); without the header, `bookmark`
and `since`/`epoch` are two points and refused together. A stream's headers go out only once
its cursor exists: a reader that has them misses nothing published from then on. The replay is everything
after the bookmark's point on the prefix asked, then live. Bounds, reported by `bookmark.list`:
`max_bookmarks` 64 per plane, a name under the instance-name rule, `max_note_len` 256 characters;
a refusal names what it hit, and a file whose records break them is refused at startup. Every
`bookmark.*` op is forwardable to a peer (`peer`), whose own sequence the bookmark then points
into; the replay stays local to that peer.

## Journal

`<name>.journal.jsonl` appends, as JSON lines, a `boot` header per start (`schema:
"ixctl-journal/0"`, `name`, `epoch`), every applied document with its generation and actions, and
every `ctl` event's envelope. The plane never replays it at startup: desired state does not
survive a restart; the new boot draws a new `epoch`, starts at generation 0 with every declared
service stopped, then applies `autostart` services as generation 1. Applied documents are partial,
appends are not `fsync`ed, write errors are ignored, and a crash can cut the last line (the next
boot's header then starts a line of its own), so no record is a commit point; durable restoration would be a separate contract that announces itself
(door 3).

## Gate

`e2e/e2e.py` is the end-to-end gate CI runs on Windows and Linux: the real plane, the real Rust
client through its `ixql-ctl` example, ix-stream's `pod` example as two federated instances, the
real `ix pipeline run` as a job, runs killed at their wall, memory, processor-time and
process limits, a probe tree under the plane (E15: kinds and limits, filter and mux, lease
expiry, the host's cpu and mem) and bookmarks (E16: a replay, seq for seq, of what a probe tree's
pipes and the `ctl` lifecycle carried after a point the operator named; bounds, refusals, a
peer's bookmark; survival of a restart) and the IXQL probe surface (E17: `probe.ixql` through
the Rust client, a CPU probe under a mux, ten records, the tree removed) and probe budgets (E18:
the plane's budget in `limits`, a leaf past its mux's `max_rate_per_s` refused naming both, a leaf
carries no budget, a mux cannot widen, rates and bytes on the list; E19: only a probe pipe the peer exports and the parent asks for crosses the link, by exact name, the status saying what each wanted name is; E20: a mirrored envelope carries the source's `src_seq` / `src_epoch`, naming the envelope on the peer that carries the same record; E21: the link marks what the mirror skips, a `link.gap` on the pipe at a live start and on the mirrored `ctl` at the peer's restart; M3: eight 10 ms pipes across the link, the records mirrored, the drops counted and marked, and the time a `probe.created` takes to cross under the load; E22: around b's restart the parent's status tells a live flow from a reconciled state and lifts `degraded` only when both hold; E23: the identity of a probe agrees on b's list, a's routed list and a's status, and b's list folded from its own stream to the cursor is exactly the list, no gap; E24: the totals the two READMEs, the gate's own docstring and the CI workflow's comment (`.github/workflows/ci.yml`) state match the checks that ran, every statement of each text, so a check added without the texts following it fails the gate). 52 checks, E1..E24, a
restart loop and two measurements: M1
(what a wait on the stream would gain over the 25 ms poll) and M2 (what one mux over sixteen
heartbeats at 10 ms delivers, and what it costs the plane).

```sh
python go/ixctl/e2e/e2e.py --build            # build ixctl, pod, ixql-ctl and ix, then run
python go/ixctl/e2e/e2e.py --bin <dir>        # use binaries already built there
python go/ixctl/e2e/e2e.py --build --work D   # keep binaries, state, runs and results.json in D
```

The job reads the Demerzel constitution from the `governance/demerzel` submodule
(`IX_GOVERNANCE_DIR` overrides it). The gate never prints a token and scans every file it wrote
for one. CI also runs `go vet` and `go test -race` on Linux; from Windows,
`pwsh scripts/ixctl-test-linux.ps1` runs the same in a WSL distro before a push (the suite can pass
here and fail there: a test that needs a subscription to overflow sets the server's `writeDelay`
knob, since Linux loopback buffers absorb a flood the Windows ones do not).

## Known limits

- Windows-first: the gate's liveness check and the process code were written for Windows; the
  Linux memory watch is a poll, and a hard death on Linux is contained for the leader only.
- A single commit refused past the Windows memory cap still depends on the kernel's best-effort
  message; the notification at `memory_mb` is the guaranteed path.
- The processor-time and process limits are polls: a run may use up to one poll period of the
  machine's cores past the first before the kill, and on Linux (no cap) spawn freely for one
  poll period past the second.
