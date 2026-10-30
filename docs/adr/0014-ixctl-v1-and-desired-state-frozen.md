# ixctl: `ixctl.v1` freezes a core and keeps its vocabulary open; the desired-state document is frozen as `ixctl-desired/0`; the journal stays provisional

Status: accepted (2026-10-25)

French: [`docs/fr/adr/0014-ixctl-v1-and-desired-state-frozen.md`](../fr/adr/0014-ixctl-v1-and-desired-state-frozen.md).

## Context

The control plane (`go/ixctl`, with its Rust and IXQL client `crates/ix-ctl`) defines three
formats that other programs build on: the `ixctl.v1` protocol, the desired-state document and the
journal. The one-way-doors page,
[`docs/plans/2026-10-17-ixctl-one-way-doors.md`](../plans/2026-10-17-ixctl-one-way-doors.md),
weighed each one and named what had to be fixed before any freeze. Those fixes have landed:

- an instance epoch (F1.1, F2.4, F3.1);
- a unit for `ev.t` (F1.2);
- each generation's own outcome (F2.1);
- the same strict decoding on both transports (F2.2);
- routing beside the document (F2.3).

The run/1 consumer contract (#443, #444) is built on doors 1 and 2, so it cannot be announced
before they are signed. Its announcement is a separate decision, with its own evidence.

## Decision

### Door 1: `ixctl.v1`, option C, a frozen core and an open vocabulary

#### 1.1 Frozen guarantees

**Access.**

- The plane listens on loopback only.
- Every request passes the same checks:
  - its `Host` must be the listener's;
  - an `Origin`, when present, must be this port's local origin;
  - every route but `GET /health` requires `Authorization: Bearer <token>`.
- The state file gives `addr` and `token`.

**Transports.** Three faces:

- JSON over HTTP under `/v1/`;
- server-sent events on `GET /v1/events`;
- the WebSocket `GET /v1/ws`, whose subprotocol `ixctl.v1` is required.

`GET /health` answers without a token: `{ok, name, epoch, …}`.

**Request bodies.** They are read strictly: one JSON value, nothing after it, 1 MiB at most. An
unknown field is refused (400), the same way on HTTP and on the WebSocket.

**Errors.** An error the plane answers is `{"error": TEXT}`, with one of these statuses:

- 400: a refused request, or an undeclared name;
- 401: the token;
- 403: the host or the origin;
- 429: the wait slots are full;
- 500: an internal failure, such as a stop that did not converge;
- 502: an unreachable peer;
- 503: shutting down.

TEXT is meant for a person and is not part of the contract. A route or a method the plane does
not serve gets the HTTP server's own 404 or 405, which is not part of the contract either.

**WebSocket frames.**

- A request is `{id, op, …the op's fields}`, and every request gets exactly one reply
  `{re: id, ok, data | error}`.
- The server's other frames are:
  - `{sub, env}`, an event;
  - `{sub, gap}`, a gap;
  - `{sub, live: true}`, sent only to a subscriber that asked for it.
- A subscription's reply comes before its first event.

**Envelope.**

```json
{"ch":"svc/pod-a","seq":17,"epoch":"8c1f…","origin":"local",
 "run":{"origin":"local","epoch":"8c1f…","seq":12},"t":1760990000123,
 "ev":{"kind":"stats","t":1760990000120,"data":{}}}
```

- `seq` belongs to one instance and starts again at 1 at each boot.
- `epoch` names that boot.
- `origin` is the instance that emitted the envelope.
- `t` is the publication time, in Unix milliseconds.
- `ev.kind` is a string. `ev.t`, when present, is in Unix milliseconds on the emitter's clock.
  `ev.data` is the payload.
- An envelope mirrored from a peer also carries `src_seq` and `src_epoch`, the source's. These
  are the `peer/<peer>/…` channels.

**Resume and gaps.**

- A subscription may name `since`, and `epoch` beside it.
- On SSE, the `id` is `epoch:seq`, and `Last-Event-ID` stands for `since`.
- The gap reasons are a **closed set**. A gap always carries `reason` and `from` (the first seq
  not received), plus `oldest` and `epoch` when they apply:
  - `since-too-old`: the replay goes on from the oldest envelope kept;
  - `overflow`: the subscription ends;
  - `epoch-changed`: the replay goes on from the oldest envelope kept;
  - `since-ahead`: only for a `since` with no `epoch`, past the head; the replay goes on from the
    oldest envelope kept.
- A `since` that names this boot's epoch is trusted, even past the head: no gap, and the
  subscription starts live.

**Federation.**

- An op that carries `peer` is forwarded to that peer, one hop.
- `subscribe` is always local. A peer's channels are read as mirrors under `peer/<peer>/`.
- On `peer/<peer>/ctl`, a `link.gap` envelope marks continuity. Its `reason` is `epoch-changed`,
  `jumped`, or one of the four gap reasons the peer answered (`prefix` then names the
  subscription).

**Channels and events.** These channels keep their meaning:

- `ctl`;
- `svc/<name>`;
- `peer/<peer>/<channel>`.

These `ctl` kinds keep their meaning and their data fields:

- `service.starting`, `service.started`, `service.exited`, `service.restarting`,
  `service.stopped`, `service.failed`;
- `converged`, `diverged`;
- `peer.up`, `peer.down`.

**Service line.**

- A stdout line that is a JSON object with a string `kind` becomes that event. Only `kind`, `t`
  and `data` are kept (D17, #448).
- Any other stdout line is `log {line}`, and a stderr line is `stderr {line}`.
- A `kind` that starts with the reserved prefix `ixctl.` is rewritten as `log`.

#### 1.2 Extensions allowed under the same name

- New ops, new routes, new channels and new values of `ev.kind`. Those that exist keep their
  meaning.
- New **optional** fields in a request. A request that does not carry them keeps exactly its
  former meaning; this is how D16 (#447) can come.
- New fields in a reply, an envelope or an event's data. **A client ignores the fields it does
  not know.**
- The bounds that are not contract values (the ring's size, the caps) may change. A client reads
  those that status announces, and assumes none of the others.

#### 1.3 What takes a new name: the migration rule

These take a new name:

- removing or renaming a frozen element;
- changing its meaning or its type;
- adding a gap reason;
- making an optional field required;
- refusing a request that was valid.

The migration goes this way:

- such a change ships under the subprotocol `ixctl.v2` and the paths `/v2/`, **served beside**
  `ixctl.v1` (`websocket.Accept` takes a list);
- a plane stops serving `ixctl.v1` only in a release that announces it, once no known client uses
  it any more.

#### 1.4 Outside door 1: stays provisional

- The probe surface: the `probe.*` ops, the `/v1/probe*` routes, the `probe/<node>` channels, the
  `probe.*` kinds on `ctl`, and the `link.gap` markers of probe pipes (`live-start`, `dropped`).
- Bookmarks and replay: the `bookmark.*` ops and `/v1/bookmark*`.
- `ixctl bench`, the command-line options, and the fields of `/health` other than `ok`, `name`
  and `epoch`.
- `status.peers[]`, except what run/1 says of it.

### Door 2: the desired-state document, frozen as `ixctl-desired/0`

Option B, recommended at the time, left it provisional. This decision goes further: it freezes
`/0` and leaves what comes next to a new value of `schema`.

#### 2.1 Frozen guarantees

**Shape.**

```json
{"schema": "ixctl-desired/0", "services": {"pod-a": {"state": "running"}, "job": {"state": "stopped"}}}
```

`schema` is optional, and its absence means `ixctl-desired/0`. Each of these is a 400, on both
transports:

- another value of `schema`;
- an unknown field, in the document or in a service;
- an empty `services`;
- an undeclared service;
- another state.

**Partial.** A service the document does not name is left as it is.

**Routing beside it.** `peer` travels beside the document (`?peer=` on HTTP, the frame's `peer`
field on the WebSocket), never inside it.

**`plan`.**

- It is pure.
- It answers `{generation, actions: [{op: start|stop, service}], noop}`.
- Its `generation` is the current generation.

**`apply`.**

- It increments the generation.
- It answers at once: `{generation, epoch, actions, …}`.
- It acts afterwards, in the background.

**Generations.**

- The generation is a counter that belongs to the boot and starts at 0; `epoch` places it.
- A generation's outcome is its own, `converged` or `diverged` on `ctl`. A later generation never
  hides it.
- Status carries `generation`, `observed_generation`, `settled_generation`, `diverged` and
  `failed`, with the meaning [`go/ixctl/README.md`](../../go/ixctl/README.md) gives them.

#### 2.2 Extensions allowed under `ixctl-desired/0`

- New fields in the **replies** of `plan` and `apply`. A client ignores those it does not know.
- Nothing in the document itself: strict decoding would refuse a new field, so any addition to the
  document is a new schema.

#### 2.3 The migration rule

- A change of the document ships under `"schema": "ixctl-desired/1"`.
- The plane accepts `/0` and `/1` side by side.
- It stops accepting `/0` only in a release that announces it, once no known client writes it any
  more.

### Door 3: the journal stays provisional

Unchanged: an audit log, opened at each boot by a header
`{"kind":"boot","schema":"ixctl-journal/0","epoch":…}`. It is never replayed (D14, #445), and it
is frozen only once a program reads it.

## Consequences

- A program outside `go/ixctl` and `crates/ix-ctl` may build on the frozen core of `ixctl.v1` and
  on `ixctl-desired/0`. What 1.4 lists, and the journal, stay drafts.
- A change that breaks a frozen guarantee is served beside the old one (`ixctl.v2`,
  `ixctl-desired/1`), never in its place.
- The run/1 announcement (#444) is a separate decision. Doors 1 and 2 are its first condition;
  the others are an independent validation (#454) on the announced commit, and the list of the
  cases no independent validator reached.

## Sign-off

Signed by the operator on 2026-10-25, on the French draft of this text. One omission of the draft
was corrected against the code before this commit: its error list lacked status 500
(`go/ixctl/plane.go`, `Stop`; `hub.go`, `statusOf`; `http.go`, an SSE stream that cannot be
flushed), and it did not say that a route or a method the plane does not serve gets the HTTP
server's own 404 or 405.
