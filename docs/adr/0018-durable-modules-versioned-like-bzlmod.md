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
capability. A call at load time stops the run with `LOAD_EFFECT`, so loading has no effect for
replay to reconcile. The grammars ADR-0017's `parse` takes are the closure's `*.ebnf` files,
named the same way.

**5. A composite is a parent run with child runs.** This amends ADR-0016 §5's closed table (it
says `run` comes back only through an amendment):

| Capability | Class | Replay | Cut mid-step (intent, no result) |
|---|---|---|---|
| `child(agent, input, grant)` | effect, with evidence | the journaled outcome | the child's id is fixed, `<parent ID>.<seq>`: its `start.json` present and equal means it exists, so wait for its outcome; absent means create it; different means `UNKNOWN` |

- `agent` is a label of the closure, such as `@fold-planner//plan.star`. The child's
  `start.json` takes the bytes from the parent's closure.
- The child's grant is charged to the parent when the child is created, so a tree of runs keeps
  one budget. #416 asks the same question for ixctl's probe trees; for runs, this ADR answers it
  by carving the grant up front.
- A child that is `waiting` or `unknown` leaves its parent `waiting(child <id>)`. The parent's
  status names the child, and a person resolves the child, not the parent.
- The child's outcome comes back on the two rails, like any capability.

**6. A version is shipped as an archive with its sum, and a registry is a directory.**

- `ix-durable module pack <dir>` writes `<name>-<version>.tar.gz` and its `SHA256SUMS`, with no
  timestamps and in a fixed file order, so the same module packs to the same bytes.
- A release is tagged `<name>-v<version>` on GitHub, with the archive, `SHA256SUMS` and an
  artifact attestation as its assets. That is the convention `ixrelay` uses (#473, slice S5), so
  both use one release workflow.
- A registry is a directory, as in bzlmod: `modules/<name>/<version>/MODULE.star` plus
  `source.json`, which holds `url` and `sha256`. It can be a Git repository or a local path. The
  URL can point anywhere HTTPS reaches: a GitHub release asset, a raw repository in Nexus or
  Artifactory, an internal web server. IX needs no registry server.
- `ix-durable lock` checks every archive against its `sha256` and refuses a mismatch. With
  `--verify-attestation`, it also runs `gh attestation verify`, and the lock records that it did.

**7. External code enters through a person.** A module from outside the operator's repositories
enters a run only through a lock that a person wrote. An agent may draft a module and propose it
(`propose("module", name, body)`, ADR-0016 §5, the store adding the kind `module` as it adds
`ebnf` in ADR-0017), but it can neither lock nor publish one. Tagging a
release stays a person's act.

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
   - The child's grant is charged to the parent once.
   - A child stopped `UNKNOWN` leaves the parent `waiting`, and resolving the child resumes the
     parent.
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
- **A closure outgrows 4 MiB:** store the closure as files beside `start.json`, published by one
  directory rename, and keep `start.json` as its digest.
