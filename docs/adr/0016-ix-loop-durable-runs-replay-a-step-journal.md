# `ix-loop` durable runs: re-execute the loop, answer each step from a write-ahead journal

Status: proposed (2026-10-29)

Number 0016: ADR-0015 is taken by `docs/adr/0015-local-isolated-jobs.md` on the unmerged branch
`feat/local-isolated-execution`.

## Context

The operator asked for real durable IX agents (2026-10-29). Such an agent:

- survives a kill and resumes at the last finished step, without asking the model again;
- never repeats an effect it cannot prove did not happen;
- waits hours for an event or an approval with no process alive;
- keeps one identity and a budget across all of that.

This is the durable-execution model of Temporal, Restate and DBOS. IX already has most of the
pieces, but none of them composes them:

| Piece | What it gives | What it lacks |
|---|---|---|
| `ix-loop` (ADR-0011) | loops in a fresh Lua 5.4 sandbox, reaching the world only through host capabilities, on carved budgets | nothing survives the process: the `Host` holds its counters in memory, and the `Ledger` is best effort, written after the fact, never synced, and "never stops a loop" |
| `ix-recover` (#389) | one committed attempt per run id, an owner lock (`File::try_lock`), reconciliation from evidence after a kill | it is run-level: one result per run; an effect outside the attempt's files is "neither undone nor reconciled" (its README, Scope) |
| `ixctl` run/1 (ADR-0014) | supervision, restart, journals, single instance | no notion of a step inside a job |
| `ix-agent-core` | session events replayed into beliefs | events project beliefs; they do not answer calls |

Two facts about the sandbox shape the design:

- **Lua state cannot be saved.** `mlua` on Lua 5.4 has no way to serialize closures, coroutines or
  upvalues. A loop can only be resumed by running it again.
- **Lua 5.4 is not deterministic across processes by default.** In `lua-src` 547.0.0
  (`lua-5.4.7/lstate.c:71`), `luai_makeseed` mixes `time(NULL)` with heap and stack addresses
  into the string-hash seed, so `pairs` over string keys can visit them in another order in
  another process. `math.random` is seeded from `time(NULL)` too (`lmathlib.c:628`), and the
  sandbox loads `math` whole. The `Host`'s own draws are seeded from `now_ms()`
  (`lib.rs`, `Host::new`).

## Decision

**1. A durable run is a root loop under a run id the caller chooses.** `ix-loop run --run-id ID`
writes an intent under `<store>/runs/<ID>/`: the source digest, the grant, the input digest and
one 64-bit seed drawn once. The same id with a different source, grant or input is refused
(`CONFLICT`), as in `ix-recover`. One process owns a run at a time, through `File::try_lock` on
`owner.lock`; a second owner is refused (`BUSY`).

**2. Resume means re-execution.** A resumed run starts the same source again in a fresh sandbox.
Each capability call is answered from the journal, in order, until the journal runs out; from
there the run continues live. The VM is never snapshotted.

**3. The journal is write-ahead and stops the run when it cannot be written.** It lives at
`runs/<ID>/journal.jsonl`, one line per event:

```json
{"seq":7,"cap":"claude","args_sha256":"…","phase":"intent"}
{"seq":7,"phase":"result","value":{…}}
```

Each line is synced (`sync_data`) before its value reaches Lua, or before the effect starts.
A failed write ends the run in `failed(journal)`; unlike `Ledger`, it never continues unjournaled.
Only a torn **last** line, from a kill mid-write, is dropped. A torn line anywhere else is
`Corrupt`.

**4. Replay checks itself.** At step `n`, the capability name and its canonical-JSON args digest
must equal the journal's step `n`. Otherwise the run stops in `nondeterministic(n)`, naming both
calls. This catches an edited loop and a loop that branched on something the journal does not
hold. Durable mode removes the known sources of drift:

- `math.random` and `math.randomseed` are replaced by a generator seeded from the intent's seed;
- the `Host`'s draws use that same seed;
- the string-hash seed is fixed (`luai_makeseed` defined at build time), if `lua-src` allows it.
  If it does not, decision 4's check is the only guard and the tracer must prove it trips.

**5. Each capability has one of three classes, fixed by the host, never by the loop.**

- **Recorded** (`claude`, `gh.*` reads, `tasks.*`, `wasm.run`): the result is journaled, and
  replay returns it without calling. A step with an intent and no result was cut mid-call. It
  touched nothing outside, so it is run again, charged to the budget again.
- **Effect** (`say`, `loops.propose`, and every future write): the intent is synced before the
  effect starts. On resume, an effect with an intent and no result is `UNKNOWN`. The run stops
  in `unknown(seq)`, and the effect is never redone blindly. A person settles it with
  `ix-loop resolve <ID> <seq> --performed <value.json> | --not-performed`, appended to the journal;
  no loop can reach that command (as with `accept`, ADR-0011 §6). An effect whose evidence the
  host can read settles itself: a proposal carries the id `<ID>-<seq>`, and the store says
  whether it exists.
- **Wait** (new: `wait(name, { timeout_s })`): if the journal holds the signal, it is returned.
  If not, the run writes `waiting(name)` and the process exits 0, so nothing runs and nothing is
  spent. `ix-loop signal <ID> <name> <value.json>` writes `runs/<ID>/inbox/<name>` with
  `create_new` (a second, different signal is `CONFLICT`), and `ix-loop resume <ID>` continues.

A child loop (`run`) is replayed inside its parent. Its steps take the next numbers of the same
journal, depth first, so a tree has one sequence.

**6. Budgets count across activations.** Calls and notional cost are summed from the journal:
a replayed step counts as spent and is not spent again. Fuel and seconds bound each activation.
Replay burns fuel again, which costs CPU only, and a wait spends no seconds.

**7. A run's states are readable without a process.** The states are `running`,
`waiting(name)`, `unknown(seq)`, `nondeterministic(seq)`, `done` and `failed(reason)`.
`ix-loop status <ID>` derives them from the journal. `done` is the last journal line and holds
the outcome; a later `resume` prints it and runs nothing.

**8. Supervision is a client, not part of the engine.** `ixctl` restarting a killed run, or
resuming one when a signal lands, means running `ix-loop resume`. That wiring is a later slice.

## Tracer slice (first slice, test first)

A loop with three steps: `claude` (Scripted), `loops.propose`, then `wait("approve")`. The run
is killed at every step boundary: before and after each intent, and before and after each
result. Each kill uses a real process kill, as in `crates/ix-recover/tests/kill.rs`, and every
death is resumed. Done when all of the following hold:

- **Model calls:** zero repeat calls once a result is journaled (the Scripted node counts asks).
- **Effects:** the proposal exists exactly once, or the run stops in `unknown(seq)` and
  `resolve` settles it.
- **Waits:**
  - the wait leaves no process alive;
  - after a `signal`, `resume` finishes the run, and a second `resume` prints `done` without
    running anything.
- **Determinism:** an edited source, and a loop that branches on `pairs` order over string
  keys, both stop in `nondeterministic(n)` and never return a different outcome.

## Rejected

- *Snapshotting the Lua VM* (Pluto/Eris-style persistence): it is C, unmaintained for 5.4, and
  breaks on every userdata the host hands in.
- *Temporal, Restate or DBOS as the engine*: each needs a server or a database outside IX for one
  local operator, and none runs a Lua sandbox; IX keeps local-first and pure-Rust-plus-`mlua`
  (ADR-0011 §7).
- *Turning `Ledger` into the journal*: the ledger is written after the call, unsynced, and must
  never stop a loop. The journal must precede the call and must stop the run. Both stay.
- *Redoing an interrupted effect after a crash*: that is the double send durable execution
  exists to prevent.
- *Making the model deterministic* (temperature 0, a fixed seed): Claude Code headless promises
  neither. The journal records answers instead.
- *Reusing `ix-agent-core` session events as the journal*: those events feed belief projection
  and have no request/answer pairing. Revisit if `hari` should project a run's journal.

## Open questions for review

- Should a finished run publish its outcome through `ix-recover` (`Store::begin` then `publish`)
  rather than in a final journal line? That would compose the existing publish-once store, at
  the cost of a dependency on a provisional crate.
- Can `lua-src` take a `luai_makeseed` definition through `mlua`'s build without a fork?

## Reversibility

**Two-way door** while `ix-loop` is experimental and journals are local files that nothing else
reads. Two events make it a one-way door:

- another repository reads the journal or the states (Gaia, the ga editor);
- a run outlives a release, so an old journal must still replay.

Either one freezes the line schema as `ix-loop-journal/0`.

## Revisit triggers

- **A replay diverges without tripping decision 4**: then the args digest is too weak, or a
  capability leaks nondeterminism; journal its full args, not a digest.
- **An effect cannot be classed, or its evidence cannot be read back** (a send to a peer
  session): then it needs an idempotency key on the receiver's side (the ixrelay case, #473).
- **Runs wait longer than the operator tolerates for stale state**: then give a run an absolute
  deadline, after which `resume` ends it in `failed(expired)`.
- **Two capabilities need to run concurrently**: one journal sequence orders everything; a
  concurrent capability needs its own sub-sequence.
