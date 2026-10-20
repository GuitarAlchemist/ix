---
title: "ixctl slice 1: the three one-way doors"
type: decision
status: awaiting sign-off (none of the three doors is signed)
date: 2026-10-17
branch: exp/ixctl at f77b31e (pushed, no PR)
reversibility: two-way today for all three (nothing merged, no consumer outside this branch); each becomes one-way at the trigger named in its section
revisit-trigger: a program outside go/ixctl and crates/ix-ctl reads the protocol, sends the document or parses the journal; or exp/ixctl is proposed for merge
---

# ixctl slice 1: the three one-way doors

French: [`docs/fr/plans/2026-10-17-ixctl-one-way-doors.md`](../fr/plans/2026-10-17-ixctl-one-way-doors.md).

Slice 1 of the control plane (commit `f77b31e`: a Go plane in `go/ixctl`, a Rust client in
`crates/ix-ctl`) defines three formats that other programs will build on. Its commit message
calls them drafts until they are signed off. This page puts each one in front of the user for a
decision:

1. the `ixctl.v1` envelope and protocol;
2. the desired-state document;
3. the journal format.

For each door it gives the exact shape today, read from the code (line numbers at `f77b31e`),
who writes and reads it, what freezing it locks in, the options, a recommendation and the line
to sign. A last section weighs the provisional next slice: durable pipeline-run identity keyed by
a client `run_id`.

Nothing here changes code. The findings marked **F** come from reading the code. None was
reproduced by a test, and none is fixed on this branch.

## How to sign

Copy a door's sign-off line, keep or change its option, and add a name and a date. A door that
is not signed stays provisional: any commit may change it, and nothing outside `exp/ixctl`
should depend on it.

## Summary

| Door | Read today by (outside `go/ixctl`) | Recommendation | Settle before any freeze |
|---|---|---|---|
| 1. `ixctl.v1` | `crates/ix-ctl`, HTTP replies only | **B**: provisional; the subprotocol name is already the version field | F1.1: a `since` from an earlier boot is misread without a word. F1.2: `ev.t` has no unit |
| 2. Desired state | written by `crates/ix-ctl` and IXQL `ix.ctl.plan` / `ix.ctl.apply` | **B**: provisional behind a `schema` field; keep it partial; drop the "later generation supersedes" rule | F2.1: a later generation can hide a failed one. F2.2: strict on HTTP, lax on WebSocket |
| 3. Journal | no program (three tests grep it) | **B**: provisional behind a per-boot header; an audit log only | F3.1: generations and seqs repeat across boots in one file. F3.2: it embeds door 1 verbatim |

The doors are coupled. The journal copies door 1's envelopes byte for byte, so door 3 cannot be
frozen before door 1. Door 2's generations travel in door 1's status reply and `ctl` events. One
addition serves all three: an **instance epoch**, an id chosen at each boot (F1.1, F2.4, F3.1).

---

## Door 1: the `ixctl.v1` envelope and protocol

### Shape today

**Transports.** One plane serves three faces, and every request passes the same checks first
(`go/ixctl/http.go:44-59`, `:64-86`):

- JSON over HTTP: `GET /health` (no token), `GET /v1/status`, `POST /v1/plan`,
  `POST /v1/apply`, `POST /v1/stop` with `{service, peer?}`, and `POST /v1/shutdown`. A refusal
  is `{"error": "..."}` with status 400, 401, 403, 500, 502 or 503 (`http.go:134-136`,
  `plane.go:295`, `peers.go:54-62`).
- SSE: `GET /v1/events?ch=PREFIX&since=SEQ[&epoch=EPOCH]`, or `Last-Event-ID`. Each event is
  `id: <epoch>:<seq>` (a bare `<seq>` was the shape before the F1.1 fix below),
  `event: <ev.kind>`, `data: <envelope>`. A gap is `event: gap` with the gap object, and
  `: ping` comes after 15 s of silence (`http.go:138-207`).
- WebSocket: `GET /v1/ws`. Subprotocol `ixctl.v1` is required; without it the connection is
  closed with a policy violation (`ws.go:14-29`, `client.go:16`). The server's read limit is
  1 MiB, the Go client's 16 MiB (`ws.go:29`, `client.go:77`).

**Envelope.** Every event on every transport has this shape (`hub.go:17-33`; the encoding is
built by hand at `hub.go:131-146`):

```json
{"ch":"svc/pod-a","seq":42,"epoch":"3f9a1c0b7d2e","origin":"local","t":1760700000000,
 "ev":{"kind":"stats","t":1760699999980,"data":{"nodes":[{"name":"src","emitted":20}],"final":false}}}
```

(The pod's `stats` data is shortened here; each node carries ten fields,
`crates/ix-stream/src/lib.rs:125-134`. `ev.t` is Unix milliseconds on the pod's own clock,
F1.2 fix.)

`seq` belongs to one instance and starts at 1 at each boot (`hub.go:100`, `:134`); `epoch`
names that boot (12 hex characters drawn at start, F1.1 fix). `t` is Unix milliseconds at
publish, or the peer's own `t` for a mirrored envelope (`hub.go:109-130`). The replay ring
holds 4096 envelopes (`hub.go:15`).

**Client frames** (WebSocket). One `Request` shape serves every op (`hub.go:50-61`):

```json
{"id":7,"op":"subscribe","ch":"ctl","since":40}
{"id":8,"op":"apply","desired":{"services":{"pod-a":{"state":"running"}}}}
{"id":9,"op":"stop","service":"pod-a","peer":"b"}
{"id":10,"op":"unsubscribe","sub":7}
```

The ops are `subscribe` and `unsubscribe` (`session.go:59-63`), then `status`, `ping`, `plan`,
`apply`, `stop` and `shutdown` (`plane.go:109-133`). `status`, `ping`, `plan`, `apply` and
`stop` may carry `peer`, and are then forwarded one hop (`plane.go:136-155`). A `subscribe`
with `peer` is refused: a peer's channels are read as mirrors (`session.go:102-104`).

**Server frames** (`session.go:72-77`, `:198-214`; the Go client reads them as one union,
`client.go:28-39`):

```json
{"re":7,"ok":true,"data":{"sub":7,"head":41,"epoch":"3f9a1c0b7d2e"}}
{"re":8,"ok":false,"error":"unknown service \"pod-z\""}
{"sub":7,"env":{"ch":"ctl","seq":42,"epoch":"3f9a1c0b7d2e","origin":"local","t":1760700000000,"ev":{}}}
{"sub":7,"gap":{"reason":"since-too-old","from":12,"oldest":310}}
{"sub":7,"gap":{"reason":"epoch-changed","from":1,"oldest":1,"epoch":"3f9a1c0b7d2e"}}
```

A gap is `since-too-old` (a replay asked for more than the ring holds; the subscription goes on),
`overflow` (a reader fell behind; the subscription ends), or, since the F1.1 fix,
`epoch-changed` (the `since` named another boot's epoch) or `since-ahead` (a `since` with no
epoch is past this boot's head); those two carry the current `epoch`, and the subscription goes
on with everything the ring still holds (`hub.go:43-52`, `session.go:151-196`). Replay waits
for room in the connection's queue; live events never wait (`session.go:151-153`).

**Channels and event kinds.**

- `ctl` carries the plane's own events, each with `service` where it applies:
  `service.starting {restart}`, `service.started {pid, restarts}`,
  `service.exited {pid, code, requested}`, `service.stopped {pid?, code?, ms?}`,
  `service.restarting {attempt, in_ms}`, `service.failed {reason, restarts_in_window?, code?}`
  (`supervisor.go:121-124`, `:168-198`, `:300-348`); then `converged {generation}` and
  `diverged {generation, service, reason}` (`plane.go:379`, `:387`); then
  `peer.up {peer, addr, origin}` and `peer.down {peer, reason}` (`peers.go:78`, `:154`).
- `svc/<name>` carries one event per line of the service's output. A stdout line that is a
  JSON object with a string `kind` passes through as `{kind, t?, data}`, `t` being Unix
  milliseconds on the service's clock, rounded to an integer (F1.2 fix); any other stdout line
  becomes `log {line}`, and a stderr line `stderr {line}` (`supervisor.go:351-414`). A
  `stats {nodes: [...]}` event also sets the service's `nodes` in status
  (`supervisor.go:375-384`).
- `peer/<peer>/<ch>` carries a peer's `ctl` and `svc/` envelopes, mirrored with their origin
  and never forwarded again (`peers.go:121-131`, `:187-197`).

**Replies.** Status is `{name, generation, observed_generation, settled_generation, diverged,
failed[{generation, service, reason}], services[{name, desired, state, pid, restarts,
started_at, last_exit, nodes}], peers[{name, state, services}]}` (`plane.go:157-181`;
`settled_generation` and `failed` since the F2.1 fix below). Plan is `{generation, actions[{op, service}], noop}`, apply is
`{generation, actions}`, stop is `{stopped, generation, exit}` (`plane.go:202-216`,
`:392-396`).

**Access.** The plane listens on loopback only (`http.go:209-220`). It checks `Host` and
`Origin` against allow-lists and wants a bearer token; it never sends a CORS header
(`http.go:64-86`). The address and the token come from the state file
`{name, addr, pid, token}` (`main.go:178`, `client.go:228-233`). `ixctl up` prints one ready
line, `{event: "ready", name, addr, pid, state_file, ms_to_ready}` (`main.go:129-136`).

### Who writes it, who reads it

- **Writes:** the plane (hub, sessions, SSE), and every supervised service through its stdout
  lines. Today the only service is the `ix-stream` pod (`crates/ix-stream/examples/pod.rs:7-12`).
- **Reads the WebSocket side:** only `go/ixctl` itself: its peer links (`peers.go:93-180`) and
  `ixctl bench` (`bench.go:169-379`).
- **Reads the SSE side:** tests only (`http_test.go`).
- **Reads the HTTP side:** `crates/ix-ctl`. It reads `settled_generation` and `failed[]` when
  present, else `observed_generation` and
  `diverged.{generation, service, reason}`, `services[].{name, state, nodes}`, `generation` from
  apply and `error` from a refusal (`crates/ix-ctl/src/lib.rs:215-317`), and `addr` and `token`
  from the state file (`lib.rs:142-164`). Its waits poll `/v1/status`; they do not subscribe yet
  (`lib.rs:30-31`).

So no program outside one Go binary reads the WebSocket or SSE side. The only contract across
languages today is the HTTP replies.

### Findings

- **Fixed (F1.1), this slice:** the hub draws an **instance epoch** at boot (12 hex characters)
  and stamps it on every envelope (`epoch`), the subscribe reply (`data.epoch`), `/health`, every
  SSE id (`<epoch>:<seq>`) and every gap. A `subscribe` may say which boot its `since` belongs
  to (`epoch`; `Last-Event-ID` carries it as `<epoch>:<seq>`, or `?epoch=` beside `?since=`). A
  `since` of another boot gets a gap `epoch-changed`, a bare `since` past the head a gap
  `since-ahead`, both naming the current epoch, then everything the ring still holds. A bare
  `since` below the head is still trusted: nothing can tell it apart, so a client that wants the
  guarantee sends the epoch. The peer links send the epoch of the last mirrored envelope. The
  journal lines now carry `epoch`, which tells one boot's seqs from the next (F3.1).
  Tests: `TestHubSinceFromAnotherBoot`, `TestSSEReplayWithLastEventID` (stale id case),
  `TestHubEnvelopeShape`.
- **Fixed (F1.2), this slice:** `ev.t` is **Unix milliseconds on the emitter's own clock**, or
  absent when it gave none; the envelope's `t` stays the plane's clock at publish. The
  supervisor rounds a service's `t` instead of truncating it and drops one that is not a finite
  number (`supervisor.go`, `parseEventLine`). The plane's own `ctl` events now carry `t`
  (`plane.go`, `event`). The pod pins its bus's birth to the wall clock once and writes Unix
  milliseconds (`crates/ix-stream/examples/pod.rs`); `bus.rs` is untouched and still counts
  seconds since the bus was made. `ixctl bench` already read `ev.t` as Unix milliseconds for
  its fan-out latency (`bench.go:251-252`), so it now gets a true number from the pod. Test:
  `TestParseEventLine` (rounding; a non-number is dropped).
- **F1.1 A `since` from an earlier boot is read as one of this boot.** `Subscribe` replays only
  when `since < head`, and starts live when `since >= head` (`hub.go:174-186`). After a restart,
  a client that resumes with its last seq, as a standard SSE reconnect does with
  `Last-Event-ID`, is misread either way. Below the new head it replays from the wrong place; at
  or above it, it starts live and misses what the new boot published before it reconnected. In
  both cases nothing tells it. A WebSocket client could compare `since` with the subscribe
  reply's `head` (`session.go:117`); an SSE client gets no reply. The peer links avoid the trap
  by treating a new token as a new instance (`peers.go:116-119`). Nothing in the envelope says
  which boot a seq belongs to.
- **F1.2 `ev.t` has no defined unit.** The plane's own events carry none (`plane.go:435-438`).
  A service's `t` is parsed as a float and truncated to an integer (`supervisor.go:406-409`).
  The pod writes seconds since its bus was made, rounded to the millisecond
  (`crates/ix-stream/src/bus.rs:19-26`), so 0.2 s becomes 0, which `omitempty` then drops, and
  1.4 s becomes 1. The envelope's own `t` is Unix milliseconds.
- **F1.3 The name says v1, the build says 0.1.0.** The subprotocol is `ixctl.v1` and the paths
  start with `/v1/`, while `/health` reports `version: "0.1.0"` (`http.go:16`, `:88-91`) and the
  contract is not signed. Elsewhere in the ecosystem a draft is spelt `/0`
  (`ix-recover-intent/0`) or v0.1.x (CLAUDE.md, cross-repo contracts).

### What freezing locks in

The envelope's five fields and their meaning. `seq` as the only key to resume, scoped to a boot
that has no name (F1.1). The two gap reasons, and the rule that an overflow ends a
subscription. Every op name and reply shape above. The channel names `ctl`, `svc/` and
`peer/<peer>/`. The `ctl` kinds and their data fields, which both the plane's `await`
(`plane.go:350-371`) and the Rust waits match on. The stdout line contract `{kind, t?, data}`
for every future service. Forwarding limited to one hop.

### Options

- **A. Freeze `ixctl.v1` as it is.** Cheapest now. F1.1 and F1.2 then stay forever, or each
  needs a `v2`.
- **B. Stay provisional behind the version field it already has.** The subprotocol string and
  the `/v1/` prefix are explicit version fields: a later plane can accept `ixctl.v1` and
  `ixctl.v2` side by side, because `websocket.Accept` takes a list (`ws.go:18-20`). Until it is
  signed, `ixctl.v1` means "draft", and any commit may break it. Fix F1.1 (an instance epoch in
  the envelope and the subscribe reply, and a gap when `since` is ahead of `head`) and F1.2
  before any freeze. Optionally rename the draft `ixctl.v0`, so the wire says what this page
  says.
- **C. Freeze a core and keep the vocabulary open.** After the F1.1 and F1.2 fixes, freeze the
  envelope, the `seq` / `since` / gap semantics and the frame union. Declare the op names,
  channel names and `ev.kind` values additive: new ones may appear, existing ones keep their
  meaning.

### Recommendation

**B now, then C at the trigger.** No program outside one Go binary reads the WebSocket or SSE
side, so a freeze buys nothing today and would make F1.1 permanent. Freeze the core (C) when
the first consumer outside `go/ixctl` subscribes, which the Rust client's own comment plans
(`lib.rs:30-31`), or before `exp/ixctl` is proposed for merge, whichever comes first. Keep the
name `ixctl.v1`.

> **Door 1 sign-off:** `ixctl.v1` stays provisional (option B) until a consumer outside
> go/ixctl subscribes or exp/ixctl goes to review; before it is frozen as option C, the
> envelope gains an instance epoch and `ev.t` gets a defined unit.
> Signed: ______ Date: ______

---

## Door 2: the desired-state document

### Shape today

```json
{"services": {"pod-a": {"state": "running"}, "pod-b": {"state": "stopped"}},
 "peer": "b"}
```

(`go/ixctl/spec.go:161-175`)

- `services` maps a declared service name to `{state}`, and `state` is `running` or `stopped`.
  An empty map, an undeclared name or another state is a 400 (`plane.go:234-249`).
- **The document is partial.** "Services it does not name are left as they are"
  (`spec.go:161-162`). The plane's own callers rely on it: autostart applies only the autostart
  services (`plane.go:83-91`), and `stop` applies `{name: stopped}` (`plane.go:403`).
- **`peer` is routing, not state.** When it is set, the whole op goes to that peer, with `peer`
  removed (`plane.go:101-108`, `:148-153`, `http.go:99`).
- **Generations.** `apply` bumps a counter of the instance and answers it at once
  (`plane.go:298-313`); `plan` is pure and answers the current one (`plane.go:251-262`). The
  counter lives in memory and starts at 0 at each boot (`plane.go:29`, `:41-47`). Each apply's
  outcome is published on `ctl` as `converged {generation}` or
  `diverged {generation, service, reason}` (`plane.go:373-388`). Status keeps
  `observed_generation`, the highest generation that converged, and `diverged`, the highest that
  diverged (`plane.go:374-386`). The rule (`plane.go:218-219`,
  `crates/ix-ctl/src/lib.rs:203-214`): generation g is done once `observed_generation >= g`,
  and has failed while `observed_generation < g` and `diverged.generation >= g`. A test pins
  "a later generation that converges supersedes the failed one" (`plane_test.go:196-199`).
- **Decoding.** HTTP refuses unknown fields (`http.go:119-126`). The WebSocket path uses plain
  `json.Unmarshal` (`session.go:54-58`), so unknown fields, inside `desired` too, are dropped.
- **Fixed (F2.2), commit `09193b7`:** both transports go through one strict decoder
  (`decodeStrict`); an unknown field gets the same refusal on HTTP and WebSocket. Envelope and
  schema unchanged.

### Who writes it, who reads it

- **Writes:** `crates/ix-ctl` (`Ctl::plan` and `Ctl::apply`, `lib.rs:175-191`); any IXQL
  program through `ix.ctl.plan` and `ix.ctl.apply` (`crates/ix-ctl/src/ixql.rs:30-38`,
  `:81-96`, which only checks that `services` is a record); the plane itself (autostart and
  `stop`); a peer that forwards an op.
- **Reads:** the plane's `validate`, `plan` and `Apply` (`plane.go:234-314`), and the journal,
  which records each applied document (door 3).

### Findings

- **Fixed (F2.1), this slice:** status gained `settled_generation` (every generation at or
  below it has ended, converged or diverged; it only moves past an unbroken run of ended
  generations) and `failed` (every diverged apply since boot, lowest generation first).
  `crates/ix-ctl`'s `wait_generation(g)` fails when `failed` names g, is done once
  `settled_generation >= g`, and otherwise waits; with an older plane it falls back on the old
  rule. `observed_generation` and `diverged` are unchanged. Tests:
  `TestALaterGenerationDoesNotHideAFailedOne` (plane) and
  `a_later_generation_does_not_hide_a_failed_one` (client). The finding below is kept as
  written; the concurrent case (B converges while A still starts) is covered by the same rule
  but has no test, since `service.started` follows the spawn within milliseconds.
- **F2.1 A later generation can hide a failed one.** The supersede rule comes from a total
  desired state (the `observedGeneration` of Kubernetes), where generation g+1 restates
  everything g asked for. Here the document is partial. Say g starts A and g+1 starts B. If B
  converges first, `observed_generation` becomes g+1 and `wait_generation(g)` answers done
  while A is still starting; if A then fails, `diverged` names g, but `observed_generation >= g`
  already, so no wait on status ever reports it. In sequence it is the same: g fails on A,
  g+1 starts only B and converges, and `wait_generation(g)` answers done. The `ctl` events are
  exact, since `converged` and `diverged` carry their own generation; the summary in status is
  not. Read, not reproduced: no test applies two documents that name different services.
- **F2.2 Strict on HTTP, lax on WebSocket.** The same document with one extra field is a 400 on
  `POST /v1/apply`, and is accepted with the field dropped as a WebSocket `apply`. Because HTTP
  is strict, a version field added later breaks a new client against an older plane.
- **F2.3 Routing sits inside the state.** `peer` belongs to the document, so a stored or
  journaled document says where it was sent as well as what it wants. `stop` already carries
  `peer` beside `service` (`http.go:49-56`).
- **F2.4 A generation means nothing across boots.** A client that holds g from before a restart
  waits until its timeout (`lib.rs:248-253`), or, once the new boot reaches g, takes another
  document's outcome for its own.

### What freezing locks in

The partial semantics and the two states. `peer` inside the document. No version field, with
strict HTTP decoding making one hard to add later. The generation as a counter that restarts at
each boot. The supersede rule, in every client that waits on status, IXQL's
`ix.ctl.wait(generation:)` included.

### Options

- **A. Freeze it as it is.** Document F2.1 as a limit: a wait on status is sound only for a
  single writer whose documents all name the same services.
- **B. Stay provisional behind an explicit version field.** Add an optional
  `"schema": "ixctl-desired/0"` now, while no plane outside this branch would refuse it, and
  decode the same way on both transports. Keep the document partial. Before any freeze, replace
  the supersede rule with each generation's own outcome (status keeps the outcomes of recent
  generations, or the waits read `converged` and `diverged` on `ctl`), and scope generations to
  an instance epoch (F2.4).
- **C. Make the document total.** Every apply names every service, and an absent one means
  `stopped`. A later generation then really supersedes an earlier one, and F2.1 goes away. The
  cost: `stop` and autostart become a read-modify-write that races between writers, and a
  forwarded apply has to restate the peer's whole state.

### Recommendation

**B.** Keep the document partial: `stop`, autostart and forwarding are partial by nature, and C
would push a race onto every writer. Drop the rule instead: a generation's outcome is its own
`converged` or `diverged`, not the highest number seen so far. Moving `peer` out of the document
(F2.3) is worth doing in the same change, but it is not a precondition.

> **Door 2 sign-off:** the desired-state document stays provisional (option B) behind
> `"schema": "ixctl-desired/0"`, and stays partial (a service it does not name is left as it
> is); before any freeze, a generation's outcome is its own `converged` or `diverged` (the
> "later generation supersedes" rule is dropped), and generations are scoped to an instance
> epoch.
> Signed: ______ Date: ______

---

## Door 3: the journal format

### Shape today

There is one file per instance name, `<state dir>/<name>.journal.jsonl`. It is opened
append-only with mode 0600, and kept from one boot to the next (`main.go:170`,
`plane.go:446-452`). It holds two kinds of line (`plane.go:48-60`, `:465-472`):

```json
{"actions":[{"op":"start","service":"pod-a"}],"desired":{"services":{"pod-a":{"state":"running"}}},"generation":1,"kind":"apply","t":1760700000000}
{"ch":"ctl","seq":3,"origin":"local","t":1760700000412,"ev":{"kind":"service.started","data":{"pid":4242,"restarts":0,"service":"pod-a"}}}
```

- An `apply` record, written once the generation is bumped and before any service acts. Its
  keys come out sorted, because it is marshalled from a Go map.
- Every envelope whose channel is exactly `ctl`, byte for byte as door 1 sends it. `svc/` lines
  and mirrored `peer/<peer>/ctl` envelopes are not journaled.
- Each line is one `Write` on an `O_APPEND` file (`plane.go:454-463`). There is no fsync, and
  write errors are ignored. The `ctl` lines are written while the hub's lock is held
  (`hub.go:147-150`).
- Nothing reads the file back. At boot the plane starts at generation 0 and applies autostart as
  generation 1 (`plane.go:73-92`). The journal is an audit trail, not a recovery log.

### Who writes it, who reads it

- **Writes:** the plane only.
- **Reads:** no program. Three tests grep it: for `"kind":"apply"`, `"generation":2` and three
  `ctl` kinds (`plane_test.go:90-99`), and to check that no token appears in it
  (`peers_test.go:65-72`, `http_test.go:211-213`).

### Findings

- **F3.1 Identities repeat inside one file.** `generation` and `seq` restart at each boot while
  the file keeps growing, so `generation: 1` and `seq: 3` occur once per boot, and only line
  order tells them apart. No line marks a boot.
- **F3.2 The journal is door 1 under another name.** Its `ctl` lines are door 1's envelopes
  verbatim, so freezing the journal also freezes the envelope and the `ctl` vocabulary.
- **F3.3 The two kinds of line share the name `kind` at different depths.** An `apply` record
  has a top-level `kind`; an envelope has `ch` and `ev.kind`. A reader tells them apart by which
  keys are present.

### What freezing locks in

An audit format with no version and no boot boundary. Door 1's envelope (F3.2). The promise that
only `ctl` is kept. The absence of any durability promise, which readers would come to rely on.

### Options

- **A. Freeze it as it is.**
- **B. Stay provisional behind an explicit version field.** Open each boot with a header line,
  `{"kind":"boot","schema":"ixctl-journal/0","name":"...","epoch":"...","t":...}`, carrying the
  instance epoch of F1.1; change nothing else. Lines then read as (epoch, generation) and
  (epoch, seq).
- **C. Make it the plane's recovery log.** Replay the last desired state at boot, keep
  generations across boots, and fsync each `apply` record. That makes ixctl a third durable
  store, beside ix-recover's and Gaia's.

### Recommendation

**B.** It is the cheapest door to keep open: no program reads it. Leave C out: durable identity
belongs to the existing pieces weighed below, not to the service supervisor. Freeze the journal
only after door 1, and only once a program starts to parse it (a Gaia evidence importer, a
DuckDB query over it).

> **Door 3 sign-off:** the journal stays provisional (option B): an audit log that is never
> replayed, opened at each boot by a header `{"kind":"boot","schema":"ixctl-journal/0",
> "epoch":...}`; it is frozen only after door 1, once a program first parses it.
> Signed: ______ Date: ______

### If all three are signed as recommended

These are later commits, not this one:

1. An instance epoch, chosen at boot: in the state file, the envelope or the subscribe reply,
   the journal's boot header, and as the scope of generations (F1.1, F2.4, F3.1).
2. A gap when `since` is ahead of `head` (F1.1).
3. A defined unit for `ev.t`, or the producer's number passed through untouched (F1.2).
4. Each generation's own outcome in place of the supersede rule, in the plane and in
   `crates/ix-ctl` (F2.1).
5. The `schema` field of the document, and the same decoding on both transports (F2.2).

---

## Provisional next slice: durable pipeline-run identity keyed by a client `run_id`

The question: when a caller starts a pipeline run under an id it chooses, how is that run found
again after a kill, with its result kept and never published twice? ixctl's generations do not
answer it: they number applies of service state, and only within one boot (F2.4). The pieces
that exist, weighed before proposing anything:

| Piece | What it does today | Can it carry a client run id? |
|---|---|---|
| `ix-pipeline` executor | Runs a lowered DAG in process, level by level, with threads inside a level (`crates/ix-pipeline/src/executor.rs:115-123`). Its only memory is `PipelineCache`, keyed `pipeline:<node>:<FNV-64 of the inputs' Debug text>` (`executor.rs:97-113`, `:309-316`): a per-stage memo, not a durable content address. ix-lock/v2 has a `run_id`, but it is new at every execution and deliberately not reproducible (`lock.rs:37-40`, `:155-167`). | No. And the name `run_id` is already taken by a per-execution id. |
| `StageGate` | Vets each stage's resolved args just before its skill runs (`crates/ix-pipeline/src/gate.rs:18-29`). Plain `lower()` uses `AllowAll`, which allows everything (`gate.rs:31-39`, `lower.rs:35-39`); `ix pipeline run` lowers with the Demerzel `ConstitutionGate` instead (`crates/ix-skill/src/verbs/pipeline.rs:546-547`). | No. It sees `(stage_id, skill, args)` and no run, and it runs too late to refuse a run before a store is opened. |
| `ix-ixql` compile | Turns a program into the shape of a `PipelineSpec`, but "compiling is not lowering": `lower()` fails with `UnknownSkill` on host calls such as `ix.ctl.*`, until someone defines a mapping from host functions to registry skills (`crates/ix-ixql/src/compile.rs:18-24`). | No. An IXQL program cannot be a recoverable run today. |
| `ix-stream` SQL nodes | `query` and `Window` start one `duckdb` process per window (`crates/ix-stream/src/sql.rs:46-66`); `ParquetSink` overwrites its file in place when the stream ends (`sql.rs:148-203`). These are streams, not runs. | No. Subprocesses and in-place writes are exactly what ix-recover keeps out of an attempt. |
| `ix-recover` (branch `exp/ix-recover` at `3f2e935`, not on `exp/ixctl`) | `Store::begin(run_id, bindings)` answers `Fresh`, `Done` or `Repaired`. A run id has at most one committed attempt, by `hard_link`; the owner lock is `File::try_lock`; the schemas are `ix-recover-{intent,manifest,receipt}/0` (`crates/ix-recover/src/lib.rs:63-65`, `:294-354`). `ix pipeline run --run-id` binds the spec and the `ix` binary, admits only `stats` and `distance`, and replays the recorded stdout (`crates/ix-skill/src/verbs/pipeline.rs:18`, `:28`, `:587`, `:697-726` on that branch). Astra approved the source of `2b83590` for this narrow use, provisionally (ix#386, still open). | **Yes.** For pure stages, it is the slice. |
| Gaia | I found no run-state tracker in `gaia/src`. What ships is `gaia-lineage-receipt/1`, with closed field sets (`src/lineage-receipt.mjs:82`). The split between a durable Work Identity and replaceable Generation Identities comes from the gaia#76 continuity design (issue closed on 2026-09-21; the receipt draft is not committed). Astra's guidance on ix#386: hand Gaia the store's logical id and the attempt id, never the ix.lock `run_id` nested inside. | Not yet. It is the natural recorder, not the store. |

**Proposal: wire, do not build.** The durable identity exists already. The client's run id is
the Work Identity, the committed attempt id is the Generation Identity, and ix-recover's store
holds both. The slice is a tracer through what is there:

1. A caller (Gaia, or a script standing in for it) runs
   `ix pipeline run --run-id <id> --runs-dir <dir>` on a `stats` / `distance` spec.
2. It reads back the run id, the attempt id and the receipt's digest, and records them. Nothing
   else, and never ix.lock's `run_id`.
3. A kill before the commit, then a rerun, gives a new attempt and one record. A kill after the
   commit, then a rerun, gives the same attempt and executes nothing. ix-recover's tests already
   prove the store side (`crates/ix-skill/tests/pipeline_kill.rs` and `pipeline_recover.rs` on
   `exp/ix-recover`). The new check is that the recorder ends with exactly one entry per run id.

**Left out on purpose:** a run concept inside ixctl (services are long-lived and restart by
policy; a one-shot run would need a new state in door 2, one more reason to keep door 2
provisional); a new executor; an invented IXQL lowering; ix-stream SQL nodes inside a
recoverable run; ix.lock's `run_id` as the key.

**Gated on:** the user's go for the Gaia consumer, and ix-recover staying provisional (its `/0`
schemas), like the three doors above. `exp/ixctl` and `exp/ix-recover` share no commit beyond
`main`, so `exp/ix-recover` has to land, or be rebased, first.

Once signed, these decisions belong in an ADR under `docs/adr/`.
