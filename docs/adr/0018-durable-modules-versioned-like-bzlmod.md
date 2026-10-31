# Durable modules: composed by `load` and child runs, versioned and locked like bzlmod, shipped as GitHub release assets

Status: proposed (2026-10-29)

Builds on ADR-0016 (durable runs) and ADR-0017 (agent DSLs).

## Context

The operator asked for durable agents that are composable, versioned and packaged, "a bit like
Maven, Nexus and poms", without reinventing the wheel (2026-10-28). Three needs:

- **Composition:** shared Starlark libraries, and agents that run other agents (composites).
- **Versioning:** a module has a version and declares the versions it depends on, and resolving
  them is reproducible.
- **Distribution:** a version is shipped as a file anyone can fetch and check, from a GitHub
  release or any other server.

Each need has a well-tried answer, and the closest one is for Starlark itself:

- **Bazel's bzlmod.** A `MODULE.bazel` declares `module(name, version)` and `bazel_dep(name,
  version)`, like a pom. Versions are resolved by Minimal Version Selection (MVS), the algorithm
  of Go modules: the highest version any dependent asks for, and never a newer one. The result is
  reproducible and needs no solver. A lockfile pins it. A registry is a directory of metadata
  (the Bazel Central Registry is a Git repository), where each version's `source.json` gives an
  archive URL and its integrity hash. Modules with different `compatibility_level`s cannot be
  mixed.
- **starlark-rust** gives `load` through a `FileLoader` that the host writes
  (`eval/runtime/file_loader.rs:28`, `Evaluator::set_loader`). The host decides what a `load` can
  reach.
- **Temporal's child workflows:** a workflow starts another with its own history, and the parent
  records the child's result.
- **GitHub releases** with a `SHA256SUMS` file and an artifact attestation
  (`actions/attest-build-provenance`, checked by `gh attestation verify`). IX already plans this
  for `ixrelay` (#473, slice S5) and the ComfyUI pack (ADR-0009), and Claude skills already come
  from versioned plugins (ADR-0008).

For durable runs, versioning matters more than anywhere else. Replay re-executes the script, so
every line a run loads must be the same line on every resume (ADR-0016 §2).

## Decision

**1. A module is a directory with a `MODULE.star`.** That file is evaluated in a dialect of its
own, which gives it two functions and nothing else:

```python
module(name = "fold-planner", version = "1.2.0", compatibility_level = 1)
dep(name = "origami-dsl", version = "0.3.1")
```

A module holds Starlark libraries (`*.star`), agents (scripts a run can start) and grammars
(`*.ebnf`, ADR-0017). Names are 1 to 40 letters, digits or `-`, and versions are SemVer.

**2. Versions are resolved by MVS, ahead of any run.** `ix-durable lock <module dir>` resolves the
dependency graph by MVS. Within one `compatibility_level` it takes the highest version any
dependent asks for. Two levels of one module in a graph are an error. The command writes
`ix-durable.lock`, which records for each module:

- its name, version and source URL;
- the SHA-256 of its archive;
- the SHA-256 of each of its files.

Resolving needs the network, so it is a person's command, like `accept`. No run and no capability
resolves anything.

**3. A run carries its whole closure.** `start.json` (ADR-0016 §2) also holds the lock and the
bytes of every file the run may load, within a fixed size limit (4 MiB). A later `resume` reads
nothing else. A run started on `fold-planner` 1.2.0 therefore finishes on 1.2.0, even after the
lock moves to 1.3.0, and replay stays proved.

**4. `load` reaches locked files only.** The host's `FileLoader` serves two kinds of label:

- `load("@origami-dsl//parse.star", "plan")` loads a file of a dependency;
- `load(":util.star", "x")` loads a file of the same module.

Anything else is a load error: a path, a URL, or a file missing from the closure. A loaded file
runs with the same globals as the script. Its top-level code may define values but not call a
capability. A call at load time is one too, `LOAD_EFFECT`, so loading has no effect for
replay to reconcile. A load error ends the run in the terminal record `failed(load)`, with its
code, before any step: the closure and the source are bound (ADR-0016 §2), so a `resume` would
fail the same way. The grammars ADR-0017's `parse` takes are the closure's `*.ebnf` files,
named the same way.

**5. A composite is a parent run with child runs.** This amends ADR-0016 §5's closed table (it
says `run` comes back only through an amendment):

| Capability | Class | Replay | Cut mid-step (intent, no result) |
|---|---|---|---|
| `child(agent, input, grant)` | effect, with evidence | the journaled outcome | the child's id is fixed, `<parent ID>.<seq>`: its `start.json` present and equal means it exists, so resume it; absent means create it, then resume it; different means `UNKNOWN` on the parent's step |

- `agent` is a label of the closure, such as `@fold-planner//plan.star`. The child's
  `start.json` takes the bytes from the parent's closure.
- The child's id is the only one with a dot, so it cannot collide with an id the operator chose
  (ADR-0016 §2). Runs nest at most four deep, and a run's `seq` stays below 10^15 (a run that
  reaches it ends in `failed(limit)`), so each level adds at most 16 characters. Four levels on a
  64-character root give at most 128, `ix-recover`'s segment limit (`lib.rs:156`). A deeper
  `child` returns `{"ok": False, "err": {"code": "TOO_DEEP"}}`.
- **The child's grant must fit what the parent has left.** Each cumulative field (`calls`,
  `ticks`, `active_seconds`, `activations`) must be at most the parent's remainder after its
  reservations and after the call that the `child` step's own `attempt`, written after the
  intent, will charge the parent. A
  parent with one call left therefore cannot give a child a call. `memory_mb` must be at most the parent's own, since it is a ceiling, not a
  quantity spent. The fit is computed from the fold, before the intent is written. A grant that
  does not fit gives an intent like any other, with its canonical args, the grant included, and
  the mark `"refused": "GRANT_EXCEEDED"`. The step's result is
  `{"ok": False, "err": {"code": "GRANT_EXCEEDED"}}`. The fold subtracts the grant of every
  `child` intent except a refused one and one resolved `--not-performed`, so a refused one
  reserves nothing. A crash between that
  intent and its result replays the refusal the intent records, without computing the fit
  again, since the fold has moved on. Replay checks the same args digest and writes the same
  result. No child is created.
- **The grant is reserved in the intent.** The parent's `intent` record for a `child` step holds
  the child's grant, and the fold of ADR-0016 §8 subtracts the grant of every child intent from
  what the parent has left, except a refused one and one resolved `--not-performed`. That record is written before the child exists, so
  a crash leaves either the intent with no child, which replay then creates under the same
  reservation, or both. A refused intent never creates a child, on replay either.
  - **A reservation is never returned to the parent,** except through a person. A parent that ends
    (`done`, or `failed`, `failed(budget)` included) before it creates the child keeps the
    reservation spent, with the rest of its grant. That errs toward under-spending, never over.
  - **`--not-performed` on a `child` step releases its reservation.** By that resolution, a person
    says the intended child does not exist. A retry under a new `seq` then reserves once again,
    and no grant is deducted twice.
  - #416 asks the same question for ixctl's probe trees; for runs, this ADR answers it by carving
    the grant up front.
- **A present child is resumed, not awaited.** The parent's worker runs `ix-durable resume
  <child>` and reads what it says. While it waits, the parent writes no lease and its watchdog
  is paused: that time is spent on the child's grant, which the parent reserved, not on its own
  `active_seconds`. A child created just before the parent died is therefore
  started too:
  - a running child answers `BUSY`, and the parent stays `waiting(child <id>)`;
  - a finished child prints its outcome, which the parent journals as the step's result.
- A child's state maps to its parent's step:
  - `done`: the child's outcome is the step's result, on the rail it came on;
  - `failed(reason)`, whatever the reason (`budget`, `limit`, `journal`): the parent ends
    `failed(child)` too. A child's script reports its own failures through `done`, on the
    failure rail, so `failed` only ever comes from an emergency stop (ADR-0016 §4). Passing it on
    as a value would let a script start the agent again, and the child may have done an effect
    that a new child would repeat. A `failed(journal)` that never became durable leaves the
    child in the state of its last valid record, and the parent resumes it like any present
    child (ADR-0016 §3);
  - `nondeterministic(seq)`: the parent stops too, as `nondeterministic(<its seq>)`, where
    `<its seq>` is the parent's `child` step, and its status names the child's id and the
    child's own `seq`. That is an emergency stop, not a value, for the same reason. Both runs
    stay stopped, since the child's executable is bound in its `start.json` (ADR-0016 §6), and
    the cause is a bug in IX;
  - `stopped` (the child's activation ended without a terminal record, ADR-0016 §11): the
    parent's activation ends too, writing nothing, so the parent is `stopped` as well, and its
    next `resume` resumes the child;
  - `waiting` or `unknown(seq)`: the parent stays `waiting(child <id>)`. Its status names the
    child and the child's state. A signal to the child, or a person's resolution of the child's
    step (ADR-0016 §7), lets the child go on, and the parent's next resume reads the result.
- A `different` child is the parent's own `UNKNOWN`, resolved on the parent's step (ADR-0016 §7).

**6. A version is shipped as an archive with its sum, and a registry is a directory.**

- `ix-durable module pack <dir>` writes `<name>-<version>.tar.gz` and its `SHA256SUMS`, with no
  timestamps and in a fixed file order, so the same module packs to the same bytes.
- A release is tagged `<name>-v<version>` on GitHub, with the archive, `SHA256SUMS` and an
  artifact attestation as its assets. That is the convention `ixrelay` uses (#473, slice S5), so
  both use one release workflow.
- A registry is a directory, as in bzlmod: `modules/<name>/<version>/MODULE.star` plus
  `source.json`, which holds `url` and `sha256`. It can be a Git repository or a local path. The
  URL can point anywhere HTTPS reaches: a GitHub release asset, a raw repository in Nexus or
  Artifactory, an internal web server. A `file://` URL serves a local registry. IX needs no
  registry server.
- `ix-durable lock` checks every archive against its `sha256` and refuses a mismatch. With
  `--verify-attestation`, it also runs `gh attestation verify`, and the lock records that it did.

**7. External code enters through a person.** A module from outside the operator's repositories
enters a run only through a lock that a person wrote. An agent proposes single files through the
store (`lua`, `pipeline`, `ebnf`), but not modules: a store proposal is one body and `accept`
writes one file (`store.rs:172`), while a module is a directory. A person assembles, locks and
releases a module. Tagging a release stays a person's act.

The lock and the closure are part of `start.json`, so ADR-0016 §2's rule covers them: `run` with
an existing id and another lock is `CONFLICT`, never a silent reuse of the old closure.

## Tracer slice (test first, each test seen red)

1. **Two local modules, one durable run.** A library module defines a function that calls
   `claude` and holds a grammar; an agent module loads both. The lock comes from a local directory
   registry with `file://` URLs. ADR-0016's kill matrix holds with the loaded code.
2. **A run keeps its version.** After a run starts, the lock is moved to a new version of the
   library. The next `resume` replays on the old bytes and finishes. A new run gets the new
   version.
3. **MVS.**
   - The diamond A → B 1.0 and C 1.1, B → D 1.0, C → D 1.1 resolves D to 1.1, and never to a
     published D 1.2.
   - Two `compatibility_level`s of D in one graph are an error.
4. **Load fences.** Each of these stops the run: a `load` of a file outside the closure; a path; a
   URL; a capability called at load time (`LOAD_EFFECT`).
5. **Child runs.**
   - A parent with one child, under the kill matrix across the boundary: exactly one child is
     created, and the parent finishes with the child's outcome.
   - The child's grant is charged to the parent once, including when the parent dies between
     its intent and the child's creation, and between the creation and the first resume.
   - A child created just before the parent died is started by the parent's next resume.
   - A child stopped `UNKNOWN` leaves the parent `waiting`, and resolving the child resumes the
     parent. A child `start.json` that differs from the intent makes the parent's step `UNKNOWN`.
     `--not-performed` on that step releases the reservation, and the retry reserves exactly
     once.
   - A child grant larger than the parent's remainder in any field is `GRANT_EXCEEDED`, and
     reserves nothing, also when the parent dies between the refused intent and its result: a
     later child that fits is then created. A fifth level of nesting is `TOO_DEEP`. A parent with
     20 active seconds left that grants all 20 to its child is not stopped while the child spends
     them.
   - A child that ends `failed(journal)` after a journaled `say` ends the parent
     `failed(child)`, and no second child is created. A child that
     stops `NONDETERMINISTIC` after a journaled `say` stops the parent `nondeterministic` at its
     own `child` step, its status names the child's id and the child's `seq`, and no second child
     is created. A child whose `failed(journal)` could not be written is resumed by the parent.
   - A parent with one call left asking for a child with one call gets `GRANT_EXCEEDED`.
   - A `seq` of 10^15 − 1 gives a child id of at most 128 characters on a 64-character root, and
     the run that reaches 10^15 ends in `failed(limit)`.
6. **Integrity.**
   - Packing a module twice gives the same bytes.
   - An archive whose bytes differ from its `sha256` is refused at lock time.

## Rejected

- **A registry server of our own** (Nexus-style): bzlmod and Go show a directory of metadata plus
  archive URLs is enough. Any server that serves files already fits behind a URL.
- **Maven poms, npm or Cargo as the package manager:** they version their own ecosystems' code.
  None knows a Starlark closure or a durable run's need to keep its bytes for every resume.
- **Version ranges resolved at run time:** a run would not replay if a newer version appeared
  between two resumes.
- **`load` from a URL or a path:** an unlocked byte in a run breaks replay and lets code in
  without a person.
- **OCI artifacts first** (ORAS to `ghcr.io`): a second tool and a second protocol. GitHub
  release assets are what `ixrelay` ships with. OCI stays a revisit trigger.
- **A shared global module cache that runs read from:** a run reads only its own `start.json`,
  so deleting a cache never breaks a waiting run.

## Reversibility

**Two-way door** while modules stay in the operator's repositories. It becomes one-way at the
first module released for others to lock. That freezes `MODULE.star`'s two functions as
`ix-module/0`, the lockfile as `ix-durable-lock/0` and the tag convention `<name>-v<version>`.
The child capability's id rule (`<parent ID>.<seq>`) is frozen with ADR-0016's journal schema.

## Revisit triggers

- **A second repository publishes modules:** move the registry into its own repository (as the
  Bazel Central Registry is).
- **Someone needs OCI** (a Nexus or Artifactory OCI repository, `ghcr.io`): add an `oci://` source
  type fetched by ORAS, with the same `sha256` rule.
- **Modules from third parties:** require the attestation check, or signatures (sigstore), at
  lock time.
- **Agents need to propose whole modules:** give the store a bundle kind that `accept` unpacks
  into a directory by one rename.
- **A closure outgrows 4 MiB:** store the closure as files beside `start.json`, published by one
  directory rename, and keep `start.json` as its digest.
