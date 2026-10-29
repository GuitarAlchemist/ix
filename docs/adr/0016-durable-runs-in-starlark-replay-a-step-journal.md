# Durable runs: a Starlark script, re-executed and answered step by step from a write-ahead journal

Status: proposed (2026-10-29), revision 3 after Codex rounds 1 and 2 and the Starlark prototype

Number 0016: ADR-0015 is taken by `docs/adr/0015-local-isolated-jobs.md` on the unmerged branch
`feat/local-isolated-execution`. Related: ADR-0017 (agent DSLs) and ADR-0018 (composition and
packaging) build on this one.

## Context

The operator asked for real durable IX agents (2026-10-29). A durable agent must do four things:

- survive a kill and resume at the last finished step, without asking the model again;
- never repeat an effect it cannot prove did not happen;
- wait hours for an event or an approval with no process alive;
- keep one identity and one budget across all of that.

That is the durable-execution model of Temporal, Restate and DBOS. AutoKitteh ran durable
workflows written in Starlark on Temporal; the project is archived (2026), so it counts here as a
precedent for the design, not as a maintained proof. IX has the pieces but nothing composes them:

- **`ix-loop` (ADR-0011):** sandboxed Lua loops, but nothing survives the process. Its `Ledger`
  is best effort, written after the call and never synced.
- **`ix-recover` (#389):** an owner lock and reconciliation, but run-level, and an outside effect
  is "neither undone nor reconciled".
- **`ixctl` run/1 (ADR-0014):** supervision, but no notion of a step.
- **`ixrelay` (#473, not built yet):** a durable queue to local Claude sessions, whose request
  states (`QUEUED`, `ACCEPTED`, `REPLIED`, `UNKNOWN`) and refusal to resend blindly are the same
  rules as this journal's.

Re-execution needs a script that computes the same thing twice. The VM cannot be snapshotted, so
a resumed run must re-run its script and get the same calls. A first draft kept `ix-loop`'s Lua.
Codex round 1 inventoried what a Lua loop can observe that changes between processes:

- the string-hash seed, which mixes `time(NULL)` and addresses (`lua-5.4.7/lstate.c:71`), so
  `pairs` order changes;
- `math.random`, seeded from the time and an address (`lmathlib.c:623`);
- `tostring` of a table, a function or a thread, which prints an address;
- weak tables, which depend on the garbage collector;
- the Lua-to-JSON conversion, which iterates raw pairs (`sandbox.rs:86`);
- `budget()` and child results, which expose live counters and seconds (`sandbox.rs:365`, `:425`).

Fixing the hash seed needs a fork of `lua-src`, which exposes no define hook to `mlua`. Addresses
and the collector cannot be fixed at all. A per-step check would not catch a branch taken after
the last call, so a Lua durable mode would rest on a guard Codex showed unsound.

A throwaway prototype then ran a durable host on `starlark-rust` 0.14.2. It is kept on branch
`proto/starlark-durable`, results in its `RESULTS.md`:

- three processes gave the same dict order, `hash`, `json.encode` and `repr`;
- no clock and no random number generator exist;
- `while` does not parse, and recursion is allowed;
- a deterministic tick limit stopped a ten-million-iteration loop;
- kills at all 15 step boundaries resumed to the same outcome, with one proposal and no
  repeated send;
- a cut send stopped `UNKNOWN`, and a person settled it;
- an edited script was refused;
- Sonnet wrote a valid agent from the capability contract on its first try, and that agent
  survived a kill.

Codex round 2 read the interpreter's source and found no address or heap-layout observable in the
Standard-plus-JSON environment:

- `repr` of a function prints its name or signature;
- there is no `id()`;
- `hash()` takes strings only and is the Java string hash;
- dicts keep insertion order, and `dir()` sorts.

Starlark is the language Bazel and Buck2 chose for hermetic, reproducible evaluation. It has two
maintained interpreters, in Rust (Meta) and in Go (Google). It has no exceptions, which suits
the railway-style results below.

## Decision

**1. Durable runs are Starlark scripts, in a new experimental crate `crates/ix-durable`.** It
depends on `starlark` (pure Rust), and on `ix-loop` for the semantic node and the store.
`ix-loop`'s Lua loops are untouched. Whether they move to Starlark is a later decision, on
evidence from this crate. The dialect is Standard plus top-level statements, and `load` is off
(ADR-0018 turns it on for locked modules). The interpreter's tick and heap limits are always set,
but `starlark-rust` documents both as best effort: they are the deterministic, cheap budget. The
hard bounds come from the operating system (decision 10).

**2. A run is created by one atomic publication, and resumed by its id alone.** `ix-durable run
--run-id ID <script> [--input file.json] [--grant grant.json]` writes one file, `start.json`.
It holds:

- the source text and the input;
- their digests;
- the `ix-durable` executable digest and the `starlark` version;
- the grant (decision 9).

It is written to a temporary file in `runs/<ID>/`, synced, then hard-linked to
`runs/<ID>/start.json`. One link is atomic, so a run either exists whole or does not exist.

- A `runs/<ID>/` with no `start.json` was never created. `run` completes it; `resume` answers
  `NOT_CREATED`. Leftover temporary files are ignored and removed by the next `run`.
- `run` with an existing id and a different source, input or grant is `CONFLICT`. The same bytes
  are not an error: the run is already created.
- `ix-durable resume ID` reads `start.json` back. A different executable or interpreter is
  refused (`CONFLICT`), since replay is proved only on the code that wrote the journal.
- One process owns a run, through `File::try_lock` on `owner.lock`; a second one is `BUSY`.
- The crate sets `rust-version = "1.89"` for `try_lock`, as `ix-recover` does.

**3. The journal is write-ahead, and it stops the run when it cannot be written.** It lives at
`runs/<ID>/journal.jsonl`, one LF-terminated JSON record per line:

- `intent` for each step: `seq`, `cap`, canonical args and their digest;
- `attempt`, before every live execution of a step;
- `result`, holding the full value, on either rail, and the cumulative tick count at that point;
- `activation`, opening each process's work;
- `heartbeat`, the activation's ticks and active milliseconds so far, written every 50,000 ticks
  or every second of active time, whichever comes first;
- `resolution`, written by a person;
- `done` or `failed(reason)`.

Each record is flushed and `sync_data`ed before its value reaches the script, or before its
effect starts. A failed write ends the run in `failed(journal)`. On open, every LF-terminated
record must parse and keep the `seq` and phase order: no gap, no duplicate, no `result` without
its `intent`. Only a final fragment without an LF may be dropped, and it is truncated and synced
before anything is appended. Durability is claimed for process kills only. No power-loss claim
is made, as in `ix-recover`.

**4. Capabilities answer on two rails; emergency stops are not values.** Every capability returns
`{"ok": True, "value": …}` or `{"ok": False, "err": {"code": …, "message": …}}`. Scripts branch on
`err.code`, a stable identifier. They can read `err.message` too. That is harmless for replay,
because the whole result is journaled, but the message is not a contract. A failure is journaled
and replayed like a success. The emergency stops are Rust errors, and Starlark has no `try`, so no
script can catch one:

- `UNKNOWN(seq)`, `NONDETERMINISTIC(seq)`, `CONFLICT`;
- budget exhausted, the tick or heap limit, an operating-system bound, `failed(journal)`.

**5. The capability table is closed; what it does not list does not exist in a durable run.**
First slice:

| Capability | Class | Replay | Cut mid-step (intent, no result) |
|---|---|---|---|
| `claude(prompt, opts)` | recorded | the journaled value | run again, after a new `attempt` |
| `propose(kind, name, body)` | effect, with evidence | the journaled value | the host reads proposal `d<run>-<seq>` back and checks its content digest against the intent: present and equal means done; absent means run it; different means `UNKNOWN` |
| `say(text)` | effect, no evidence | the journaled value | `UNKNOWN`: the run stops until a person resolves it |
| `wait(name)` | wait | the journaled signal | look for the signal again; none means `waiting(name)` and exit 0 |
| `print(…)` | host output | suppressed | not applicable |

`budget()`, timings, `run` (child scripts), `tasks.*`, `gh.*`, `wasm.*` and `loops.list/load`
are absent from durable runs. Each one comes back only through an amendment that classes it and
says how replay rebuilds the state it touches.

`propose` goes through the existing `Store`, so the operator accepts a durable proposal with the
same `accept` as a loop's. Two changes to `ix-loop`'s store make that possible:

- **A caller-id publication.** A new `Store::publish_as(id, draft, by)` writes atomically
  (temporary file, sync, hard link) and refuses an existing id with different content. Today
  `Store::propose` picks its own id and writes the final file directly (`store.rs:97`, `:120`).
- **A second id shape.** `valid_id` (`store.rs:51`) also admits `d<12 hex>-<digits>`. The hex is
  the first 12 digits of the run id's SHA-256, so any run id gives a safe file name. `kind` keeps
  the store's values (`lua`, `pipeline`).

**6. Replay checks itself on three things.** At each journaled step, the replay must ask for:

- the same `cap`;
- the same args digest;
- the same cumulative tick count as the `result` record.

The tick count is deterministic for the same executable, source, input and recorded answers.
Codex round 2 confirmed it in the source: ticks count calls and loop back edges, and the
1,000-tick check period only moves the counter. It is a control-flow counter, not a state proof:

- Two branches with the same number of calls and back edges pass it.
- A divergence after the last journaled result goes undetected if the process dies before
  `done`, since no later step compares it.

Both residuals are accepted. With the closed globals above, nothing that varies between processes
was found to feed such a branch. A mismatch stops `NONDETERMINISTIC(seq)` and names both sides.
With the source and the executable bound (decision 2), a mismatch means a bug in IX, not an edited
script.

**7. A person resolves `UNKNOWN`; the script decides what follows.** `ix-durable resolve ID seq
--performed <value.json> | --not-performed` appends a `resolution` record. No script can reach
that command (as with `accept`, ADR-0011 §6). After `--performed`, the step returns that value
and the effect is never repeated. After `--not-performed`, the step returns
`{"ok": False, "err": {"code": "NOT_PERFORMED"}}`, and the script may call the capability again
under a new `seq`.

**8. Host state is a fold over the journal, never memory.** At each activation, the host rebuilds
from the journal before running a line:

- calls spent, which is the number of `attempt` records;
- ticks and active time spent (decision 9);
- proposals and their bytes;
- `say` utterances;
- notional cost, summed from `result` records only. An interrupted attempt has no known cost.

The live tail then sees the run it would have seen without the kill.

**9. One budget, fixed at creation and cumulative over the run.** The grant lives in
`start.json`. A later `resume` cannot change it, whatever the environment says. Raising it
means a new run.

- **Calls:** one per `attempt`, so a call killed repeatedly is charged every time.
- **Activations:** each process that opens an `activation` record is charged one, before any
  replay. Every resume is therefore charged, killed or not.
- **Ticks and active time:** summed over activations, from each activation's last `heartbeat`. A
  kill loses at most one heartbeat interval of accounting. The work that escapes the budget is
  therefore at most the activation grant times one interval (50,000 ticks or one second), however
  often the run is killed.
- **Replay margin:** replaying up to the frontier spends ticks again. The interpreter's limit is
  set once, at the frontier's tick count plus a fixed margin plus the live ticks left. Past the
  frontier, the host checks the live ticks itself, at every capability call and every heartbeat.
  Live work can therefore borrow at most the margin.
- Time spent waiting is not counted.

**10. Hard bounds and process ownership come from the operating system.** `ix-durable run` and
`resume` start a worker process. The worker evaluates the script, holds the owner lock and writes
the journal. The launcher puts it in a kill-on-close Windows Job Object with a process memory limit
and a CPU-time limit. On Linux it uses a process group, a parent-death signal (as
`go/ixctl/pdeathsig_linux.go` does) and `setrlimit` for address space and CPU. A wall-clock
watchdog backs both up.

- A `claude` subprocess (`node.rs:133`) starts inside the worker's job. A killed worker leaves no
  call running beside the next attempt.
- When a bound kills the worker, the launcher takes the owner lock and appends `failed(limit)`.
  A single expensive native operation, such as `sorted(range(100_000_000))`, then ends the run.
  It cannot get past the tick check.

**11. States are readable without the owner.** `ix-durable status ID` reports one of these:

- `not_created`;
- `running`, when the owner lock is held;
- `waiting(name)`, `unknown(seq)`, `nondeterministic(seq)`;
- `done`, where the `done` record holds the outcome; a later `resume` prints it and runs
  nothing;
- `failed(reason)`.

**12. A signal is published atomically.** `ix-durable signal ID name value.json` uses a temporary
file, a sync and a hard link into `inbox/<name>`. A second, different signal is `CONFLICT`, and the
same bytes again are not an error. No timeout in the first slice. `ixctl` resuming a run when a
signal lands, or after a crash, is a later slice; the engine stays a command line.

## Tracer slice (test first, each test seen red)

One script: `claude`, `propose`, `say`, then `wait("approve")`. Done when every check below holds:

1. **Kill matrix.** A real process kill before and after every `intent`, `attempt`, effect,
   `result` and `done` write, and every death resumed:
   - the model is asked once per `attempt` record, and never again for a journaled result;
   - exactly one proposal, which the existing `accept` moves into the library;
   - `say` happens at most once;
   - every run ends with the same outcome.
2. **Creation.** A kill before and after the `start.json` link: `resume` answers `NOT_CREATED`,
   then `run` completes the run. A second `run` with another grant is `CONFLICT`.
3. **Repeated kills of one Claude call.** Each kill adds an `attempt` and a charge; the grant runs
   out and the run stops; no descendant process survives the owner.
4. **Repeated kills during local computation.** Kills between two steps, with unclosed
   activations. The activation and tick charges grow with every resume, and the run stops within
   the bound of decision 9.
5. **A native operation past the bounds.** `sorted(range(100_000_000))` ends in `failed(limit)`,
   and the machine keeps its memory.
6. **Both resolutions.** After an `UNKNOWN` `say`, `--performed` never speaks again, and
   `--not-performed` speaks exactly once on the script's retry.
7. **Proposal evidence.** A kill before, during and after publication, and a proposal whose
   content differs from the intent, which must give `UNKNOWN`.
8. **Divergence and determinism.**
   - A host built with one capability made deliberately nondeterministic stops `NONDETERMINISTIC`
     at the next step.
   - An edited source, another executable and a different input are refused.
   - Two independent processes give the same per-step tick vector on a script past 1,000 ticks.
9. **Journal damage.**
   - A torn final fragment is truncated, and the run resumes.
   - A malformed record that ends in an LF, interior or last, is `Corrupt`, and so are a sequence
     gap, a duplicate and a `result` without its `intent`.
   - Injected write and sync failures give `failed(journal)`.
10. **Budget across waits.** A script that spends most of its ticks before the wait still finishes
    after it, and live work after the frontier never gets more than the margin.
11. **State rebuilt.** After each resume, every field of decision 8 equals what an unkilled run
    holds, proposal bytes and notional cost included.
12. **Ownership and signals.** A second process on a running run is `BUSY`; a second, different
    signal is `CONFLICT`.
13. **Suffix proof.** After each resume, the host asserts it replayed exactly the journal's steps
    before going live, rather than returning early.

## Next slices (decided elsewhere)

- **Agent DSLs** (ADR-0017): a `parse(dsl, text)` capability, recorded, over a grammar locked by
  digest.
- **Composition and packaging** (ADR-0018): `load` of locked modules, child runs, versioned
  modules released on GitHub.
- **A send with evidence** (#473): once `ixrelay` exists, `send(target, text)` replaces `say` for
  messages to a session. The request id is `d<run>-<seq>`, and ixrelay refuses the same id with
  different content. `ixrelay request <id>` gives the evidence that the proposal rule uses:
  - `ACCEPTED` or `REPLIED` with the same content means done;
  - absent means submit it;
  - ixrelay's own `UNKNOWN` means a person resolves it, as here.

## Rejected

- **Lua for durable runs:** the sources of drift listed above, two of which (addresses, the
  collector) cannot be fixed, and a hash seed fixable only by forking `lua-src`.
- **Snapshotting a VM:** Pluto- or Eris-style persistence is C, unmaintained for 5.4, and breaks
  on host userdata. Starlark offers no snapshot either.
- **Temporal, Restate or DBOS as the engine:** a server or a database outside IX for one local
  operator. The archived AutoKitteh showed that Starlark on Temporal can work; IX keeps the
  journal local instead.
- **`starlark-go`:** the agent's capabilities (the semantic node, the store, and later the
  oracles, `ix-wasm` and `ix-origami`) are Rust. Go keeps the control plane (`ixctl`) and
  delivery (`ixrelay`). Scripts stay portable between the two interpreters if they keep to the
  common dialect. A wasm build of `starlark-go` exists (`starlark-webasm` on npm), so the ga
  editor could check a script in the browser without running it durably.
- **Trusting the interpreter's limits as hard bounds:** its own documentation says they are best
  effort and miss native allocations.
- **Three identity files published one by one:** a kill between two links leaves a run that
  neither `run` nor `resume` can settle. One file, one link.
- **`ix-recover` as the outcome store:** it is run-level and has its own owner lock; two
  authorities over one run would open two crash windows. The `done` record is the only authority
  (Codex, open question 1).
- **Turning `Ledger` into the journal:** it is written after the call, unsynced, and must never
  stop a loop. Both stay.
- **Redoing an interrupted effect:** that is the double send this model exists to prevent.
- **A deterministic model** (temperature 0, a seed): Claude Code headless promises neither. The
  journal records answers instead.

## Reversibility

**Two-way door** while `ix-durable` is experimental and its journals are local files nothing else
reads. It becomes one-way when another repository reads the journal or the states (Gaia, the ga
editor), or when a run must outlive a release. Either freezes the record schema as
`ix-durable-journal/0`. Decision 2 already refuses to replay under another executable; lifting
that refusal is the same one-way step. The second store id shape (decision 5) is two-way: it
only widens what `accept` takes.

## Revisit triggers

- **A divergence that the three-part check of decision 6 misses:** then journal a digest of the
  full Starlark module state at each step.
- **Scripts need `run`, `tasks.judge`, `gh` or `wasm`:** add them one at a time, each with its
  replay rule and the state it rebuilds.
- **The Lua loops are needed in durable runs:** decide then whether to migrate them to Starlark,
  with this crate's numbers as evidence.
- **Runs wait longer than the operator tolerates stale state:** add a run deadline after which
  `resume` ends it in `failed(expired)`.
- **A run must move to another machine:** the operating-system math runtime is not bound by the
  executable digest, so float formatting is proved on one platform only.
