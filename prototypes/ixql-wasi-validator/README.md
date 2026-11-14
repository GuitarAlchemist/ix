# PROTOTYPE — IXQL → Go WASI validator, and a WASI 0.3 feasibility spike

Not a workspace member, not shipped, not wired into `ix-agent` or `ix-cli`. Delete this folder
to remove it. Written 2026-11-03.

## What already existed (not claimed here)

| Proven before this prototype | Where |
|---|---|
| Go source → `GOOS=wasip1` module, content-addressed cache, no module fetch | `crates/ix-wasm/src/build.rs`, `tests/go.rs` |
| WASI **preview 1** sandbox in wasmi: stdio only, no files/sockets, seeded clock/random, fuel/memory/output caps, deterministic reruns | `crates/ix-wasm/src/{lib,wasi}.rs`, `tests/sandbox.rs` |
| WASI stages as `wasm.run` nodes of an `ix.yaml` pipeline | `crates/ix-wasm/tests/pipeline.rs` |
| IXQL capability dispatch and `→ when T >= x:` verdict gates | `crates/ix-ixql/src/capability.rs`, `tests/verdict_gate_tests.rs` |

Before this, nothing in the repo connected **IXQL** to a WASI module, no Go validator stage
existed, and there was no WASI 0.2, 0.3 or Component Model work.

## What this prototype adds

- `src/lib.rs`: `WasiValidator`, an IXQL `Capability`. It sends the call's value to an `ix_wasm::Program`
  as JSON on stdin, reads `{"valid": bool, ...}` from stdout, and returns `T 1.0` or `F 1.0`.
  A trap, running out of fuel, a non-zero exit or a malformed report is an `Err`, which fails the run.
- `validator/main.go`: a **fixture validator**. It uses only the standard library and checks a
  small, invented "sandbox candidate" shape. **It is not the previously validated Go package**,
  which is not in this repository. Its input must be exactly one JSON value: trailing data such
  as `{…} {…}` or `{…}junk` exits 2. Before 2026-11-14 it judged only the first value and ignored
  the rest; the WasmEdge comparison below ran on that earlier version (none of its 5 cases has
  trailing data).
- `validator/main_test.go`: Go tests of the validator's stdin → stdout/stderr path (field errors,
  unknown fields, error order over 200 runs, empty, malformed and trailing input, and the
  invalid fixture's output against `p3-spike/expected/validator-invalid.stdout`). Standard library
  only; `validator/go.mod` exists for `go test` and is not used by the ix-wasm build, which
  compiles `main.go` alone.
- `fixtures/`: `pipeline.ixql`, one valid candidate and one invalid candidate (sandbox data only).
- `tests/pipeline.rs`: 3 end-to-end tests: valid → T → written; invalid → F → alert only, with
  the Go errors passed through verbatim; fuel exhaustion → run error, nothing written.
- `p3-spike/`: the WASI 0.3 feasibility spike. `run.sh` reproduces the spike and follow-ups,
  except the manual trace and core dump noted in follow-up 2. Steps 6–8 assert their results
  (`lib/check.sh`, `expected/`); steps 1–5 print evidence only. `wasmedge-compare.sh` reproduces
  the WasmEdge comparison.
  - `guest/`: a 30-line Rust p3 echo guest.
  - `async-guest/`: follow-up 1, async behaviour.
  - `adapter-trap/`: follow-up 2, the Go adapter trap — controlled WAT tests, a stress
    program, and a diagnostic Go runtime diff.

```bash
cd prototypes/ixql-wasi-validator && cargo test   # needs go 1.21+ on PATH (or IX_GO); Cargo.lock is gitignored repo-wide
```

```bash
cd prototypes/ixql-wasi-validator/validator && go test ./... && go vet ./... && gofmt -l .
```

```bash
prototypes/ixql-wasi-validator/p3-spike/run.sh /some/tmp/dir  # downloads pinned tools, checks sha256
```

**CI** (`.github/workflows/ixql-wasi-prototype.yml`, experimental, not a required gate): one
`ubuntu-latest` job, on pull requests that touch this folder or that file. It runs `cargo test`
here with Go 1.24.7, then `run.sh` with `P3_SKIP_RUST_GUESTS=1`. That executes steps 1, 2 and 4
(printed only) and asserts steps 6–8; steps 3 and 5 (the WASI 0.3 Rust guests, which need the
`wasm32-wasip2` target and crates.io, and assert nothing) are skipped. It does not run
`wasmedge-compare.sh`, and it does not trigger on changes to the path-dependency crates
(`ix-ixql`, `ix-wasm`, `ix-types` and theirs).

## Pins

| Piece | Exact version | Source |
|---|---|---|
| Go | go1.24.7 linux/amd64 (`GOOS=wasip1 GOARCH=wasm`) | preinstalled |
| Rust | 1.97.0, target `wasm32-wasip2` (Rust has no `wasm32-wasip3` target) | rustup |
| Prototype sandbox | wasmi 2.0.0 via `ix-wasm`, WASI **preview 1** only | workspace |
| Wasmtime CLI | 50.0.1 (c307ec81d 2026-10-29), sha256 `685466…498a5` | GitHub release |
| Host WIT in Wasmtime 50.0.1 | `wasi:*@0.3.0` (p3) and `wasi:*@0.2.12` (p2), read from the `wasmtime-wasi-50.0.1` crate | crates.io |
| wasm-tools | 1.263.0, sha256 `21c036…65431` | GitHub release |
| p1→component adapter | `wasi_snapshot_preview1.command.wasm` from Wasmtime v50.0.1, sha256 `0b4fe4…2a3c4` | GitHub release |
| Guest bindings, 0.3.0 | `wasip3 =0.9.0+wasi-0.3.0` → wit-bindgen 0.62.0 | crates.io |
| Guest bindings, 0.3.2 | `wasip3 =0.10.0+wasi-0.3.2` → wit-bindgen 0.63.0 | crates.io |
| Async guest | `wasip3 =0.9.0`, `wit-bindgen =0.62.0`, `futures =0.3.34` | crates.io |
| WasmEdge (Preview 1 comparison only, then removed) | 0.18.1 `manylinux_2_28_x86_64`, sha256 `ec97cc…a67c9` | GitHub release |

## WASI 0.3 spike: results (each one run in this session)

1. **The Go fixture validator works under Wasmtime 50.0.1 as a core p1 module.** For the invalid
   candidate it prints the same report that the ix-wasm/wasmi test asserts. That was compared as
   JSON, not byte for byte between the two runtimes.
2. **Go → component does not work.** Adapting the Go module with the preview-1 command adapter
   gives a valid `wasi:cli/run@0.2.12` component, a **0.2** component, not a 0.3 one. It **traps**
   at start (`assertion failed at adapter line 2857`, in `args_sizes_get` called from Go's
   `runtime.schedinit`). The trap is the same with `-S p3=n`, so p3 does not cause it. The same
   adapter works on the repo's minimal `ECHO` WAT stage, so the failure is specific to Go's output.
   The cause is diagnosed in follow-up 2 below.
3. **Native WASI 0.3 guests run on Wasmtime 50.0.1** with default flags. A guest built against
   `@0.3.0` echoes stdin to stdout. A guest built against `@0.3.2` also links against the host's
   `@0.3.0` and runs, through semver-compatible import matching. Both fail with `-S p3=n`, which
   confirms the p3 host serves them. Two caveats:
   - The bundled 0.3.0 and 0.3.2 WIT differ **only in version strings**, so this shows the 0.3.2
     guest is structurally compatible here. It does not show 0.3.x guests are compatible in general.
   - Each guest is a **hybrid**: it imports `wasi:cli/*@0.3.x` for its own calls and
     `wasi:*@0.2.9` for Rust std on the `wasm32-wasip2` target.
4. **A mismatched type fails at link time.** A component that imports
   `wasi:cli/environment@0.3.0` with `get-arguments -> list<u32>` is refused:
   `instance export get-arguments has the wrong type`. The real `list<string>` links.

**WASI 0.3.1:** the official release page says it needs extra Component Model features
(`map<K,V>`, `implements`, `external-id`). Wasmtime 50.0.1 has flags for them
(`-W component-model-map`, `-W component-model-implements`), but its bundled WIT is `0.3.0`.
No guest using those features was tested.

**WasmEdge.** Its docs on GitHub (`WasmEdge/docs` main, `extensions/proposals.md`) list the
Component Model behind `--enable-component` as "Loader and validator (since 0.17.0)", and they
list no WASI 0.2 or 0.3 runtime support. wasmedge.org could not be reached from this session.
WasmEdge was therefore compared only on **WASI Preview 1**; see the section below.

## WasmEdge vs Wasmtime on the Preview 1 fixture (run 2026-11-03, then uninstalled)

`p3-spike/wasmedge-compare.sh <tools-dir>` reproduces this.
- **What was run.** It is **only the small fixture validator** (`validator/main.go`, ~70 lines,
  invented candidate shape). It is **not** the complete, previously validated Go package, which
  is not in this repository. Nothing here says anything about how that package behaves.
- **Download.** The script fetches the official asset
  `github.com/WasmEdge/WasmEdge/releases/download/0.18.1/WasmEdge-0.18.1-manylinux_2_28_x86_64.tar.gz`
  into a temporary directory. That is the newest release found: 0.18.2 and 0.19.x return 404,
  and the release page itself is blocked by this session's proxy, so the search was not
  exhaustive.
- **Verification.** The tarball is checked against the release's own `SHA256SUM` and against the
  digest recorded here, `ec97ccfc…fa67c9`. A checksum served from the same origin proves
  integrity, not authorship; no signature was checked.
- **How it runs.** WasmEdge runs from the temporary directory with `LD_LIBRARY_PATH` set for its
  commands only, then the directory is deleted.
  - The official installer was **not** used, because by default it also fetches the
    `wasi_nn-ggml` AI plugin.
  - `wasmedge --version` prints `0.18.1` and the built-in `wasi_logging` plugin 0.1.0.0. No
    plugin was loaded from disk, and no GPU or AOT feature was used, so WasmEdge ran in its
    default (interpreter) mode.
- **Module.** go1.24.7 `GOOS=wasip1` (ix-wasm flags), sha256 `d3dd3a38…bb1a3`. The script builds
  it once and runs that same file under both runtimes. It is the same build as follow-up 2's
  "stock" module. Result 1 above used an earlier build (`3df0ee54…`) made before a comment-only
  `gofmt` change. No `--dir` preopens and no environment were passed to either runtime.

| Case | Wasmtime 50.0.1 exit | WasmEdge 0.18.1 exit | stdout | stderr |
|---|---|---|---|---|
| valid candidate | 0 | 0 | identical: `"valid":true,"errors":[]` | identical, empty |
| invalid candidate | 0 | 0 | identical: same 4 sorted errors as the ix-wasm test | identical, empty |
| `[1,2,3]` (not an object) | 0 | 0 | identical: `candidate: must be an object` | identical, empty |
| failure: `nope` (not JSON) | 2 | 2 | identical, empty | identical: `not JSON: invalid character 'o' in literal null (expecting 'u')` |
| failure: empty stdin | 2 | 2 | identical, empty | identical: `not JSON: EOF` |

Result: all 5 cases are byte-identical on stdout and stderr, with the same exit codes.

**Removal check** (run after the script):
- `find / -xdev -iname '*wasmedge*'` matches only this script;
- `wasmedge` is not on PATH;
- there is no `~/.wasmedge`, no `libwasmedge` in `ldconfig -p`, no `WASMEDGE*` environment
  variable, and no change to any shell profile.

**Limits:**
- **Only core WASI Preview 1 on x86_64 Linux was compared**, with no file or network access.
- **WasmEdge AOT (`wasmedge compile`) was not tested.**
- **WasmEdge cannot be a component or WASI 0.3 host here**, per its own docs, so it does not
  change the 0.3 verdict above.

## Feasibility verdict for *this* pipeline

- **Works today:** IXQL → WASI **preview 1** → Go. It is built, tested and deterministic.
- **WASI 0.3 host:** feasible with Wasmtime ≥ 46 (tested on 50.0.1) for guests built natively for 0.3.
  - **Go cannot produce such a guest:** standard Go has only `wasip1`, and the p1 adapter route
    traps (result 2). The trap comes from a memory-ownership conflict in Go's runtime
    (follow-up 2), and the adapter would only produce a 0.2 component anyway.
  - The host sandbox would also have to move from wasmi, which runs core modules and preview 1
    only, to Wasmtime's component runtime. That means re-establishing what ix-wasm already
    guarantees: fuel, memory and output caps, a deterministic clock and random source, and no
    files or network.
- **Recommendation:** keep the validator on p1 in ix-wasm. Re-open the 0.3 question when a Go
  toolchain emits 0.3 components, or when Go's wasm runtime tolerates `memory.grow` calls it did
  not make (follow-up 2). Treat it as a two-way door: the capability seam (`WasiValidator`) does
  not depend on the WASI version.

## Integrating the previously validated Go package

The package is not in this repo. To swap it in for the fixture:

1. Vendor its source as a `main` package that uses only the standard library. ix-wasm builds with
   `GOPROXY=off`, `GOFLAGS=-mod=mod` and no network. External modules would need vendoring plus a
   build-path change, and that change is not made here.
2. Match the stdio contract: one JSON value on stdin, `{"valid": bool, ...}` on stdout, exit 0.
   Exit non-zero only for unusable input.
3. Remove nondeterminism from its output. Sort anything built from map iteration, as `main.go`
   does. Go seeds its map order from WASI `random_get`. ix-wasm seeds that source and Wasmtime
   and native runs do not, so map order can change between runs there (inferred from the
   runtimes' design, not measured here). The output must not depend on it.
4. Re-run its existing test suite natively, then the 3 tests here against its real fixtures.
   Budget fuel from a measured run: `Output::fuel_used` is returned in the report.

## Follow-up 1 — real WASI 0.3 async: futures, streams, and sync vs async (built and run)

`p3-spike/async-guest`: Rust 1.97.0, target `wasm32-wasip2`, `wasip3 =0.9.0+wasi-0.3.0` (WIT
`0.3.0`, the same version as the Wasmtime 50.0.1 host), `wit-bindgen =0.62.0`, `futures =0.3.34`.
Runtime: Wasmtime 50.0.1 with default flags. Step 5 of `run.sh`; timestamps come from
`wasi:clocks/monotonic-clock@0.3.0#now` inside the guest:

| Mode | What runs | Observed (ms) | Reading |
|---|---|---|---|
| `concurrent` | `join!(wait-for 300, wait-for 100)` | fast done 101.9, slow done 301.3, total **301** | two host-side async calls really overlap |
| `sequential` | the same, awaited in turn | total **403** | control: the sum |
| `blocking` | `join!(std::thread::sleep 300, wait-for 100)` | async timer *starts* at 301.1, total **402** | a sync WASI 0.2 call blocks the whole instance; `join!` cannot help |
| `stream` | guest `stream<u8>`, 3 chunks 100 ms apart | chunks read at 0 / 100.7 / 202.5, `Dropped` at 304.5 | incremental delivery; each write completes when the reader takes it |
| `stdin` | host `read-via-stream` → (`stream<u8>`, `future<result<_, error-code>>`), plus a 100 ms timer; input `a`, then 0.3 s, then `b`, then EOF | `a` 0.2, timer 102.0, `b` 293.9, `Dropped` and future `Ok(())` at 495.5 | host-backed stream and future; the timer finishes while a read is pending |
| `future-intra` | guest `future<result<_, error-code>>`, written and read in one instance | **refused**: `cannot read from and write to intra-component future/stream with non-numeric payload` | a limit in Wasmtime 50.0.1; an intra-component `stream<u8>` (numeric payload) works |

ABI evidence (from `wasm-tools print`):
- `wait-for` is imported as `[async-lower]wait-for`.
- The export is `[async-lift]wasi:cli/run@0.3.0#run` with a `[callback]` entry.
- Streams and futures use the canonical builtins (`stream.new/read/write`, `future.new/read/write`,
  `waitable-set.*`, `subtask.*`, `task.return`).
- The blocking path is Rust std's sync-lowered `wasi:io/poll@0.2.0` `[method]pollable.block`.
- With `-W component-model-async=n`, Wasmtime refuses to load the component:
  `` `stream` requires the component model async feature ``.

Limits:
- **Timings come from a shared cloud container:** absolute times vary with load. The ordering
  and the max-vs-sum totals were the same in both runs recorded here, but two runs are not a
  statistical result.
- **Futures across two components are not tested:** this is one component talking to the host,
  and the composed-components case that WASI 0.3 is designed for is not covered.
- **The `stdin` future is resolved by the host.** No guest-to-guest future with a non-numeric
  payload was run, because of the `future-intra` refusal.

## Follow-up 2 — Go Preview 1 → 0.2 adapter trap (diagnosed, with causal confirmation)

**The failing check:** `wasi_snapshot_preview1` adapter (Wasmtime v50.0.1),
[`lib.rs:2857`](https://github.com/bytecodealliance/wasmtime/blob/v50.0.1/crates/wasi-preview1-component-adapter/src/lib.rs#L2857):
`assert_eq!(state_ref.magic1, MAGIC)` in `State::with`. `MAGIC` is `b"ugh!"`.

**Why it fails.** The adapter keeps two private pages in the *main module's* linear memory. Go
does not export `cabi_realloc`, so wit-component substitutes a `cabi_realloc` that only does
`memory.grow(1)`. It uses it twice:
- for the adapter **stack**, from the adapter's `start` function, before Go's `_start`;
- for the adapter **`State`**, lazily, at the first WASI call that needs it.

Go 1.24.7's wasm runtime breaks two assumptions the adapter needs:
1. [`runtime/os_wasm.go:16`](https://github.com/golang/go/blob/go1.24.7/src/runtime/os_wasm.go#L16):
   `osinit` sets `blocMax = currentMemory() * 64KiB`. That claims every page that exists at
   `_start`, including the adapter's stack page, as Go heap space.
2. [`runtime/mem_wasm.go`](https://github.com/golang/go/blob/go1.24.7/src/runtime/mem_wasm.go):
   `sbrk` grows by `(bl + n - blocMax)` pages, ignores where `memory.grow` actually put them,
   and sets `blocMax = bl + n`. Go therefore assumes it is the only caller of `memory.grow`, so
   its next heap range covers the adapter's `State` page.

The invariant that fails: *pages obtained with `memory.grow` by the adapter are not used by
anyone else.* Go's runtime assumes it owns the whole memory and every grow.

**Evidence.** Each item was run here. The controlled tests and the patch test are steps 6 and 7
of `run.sh`. Those steps **assert** each case's exact exit code, its stdout against hand-written
files in `p3-spike/expected/`, and the trap marker on stderr. Step 8 holds negative controls:
deliberately wrong expectations against the same real artifacts, all of which must be rejected.
`run.sh` exits 1 if any of the 24 assertions fails. Steps 1–5 only print evidence. The call trace and the core dump were one-off manual commands and are *not* in
`run.sh`. To repeat them on the stock `validator.p2.wasm` that `run.sh` step 2 produces:
`echo '{}' | WASMTIME_LOG=wasmtime_wasi=trace wasmtime run validator.p2.wasm` and
`echo '{}' | wasmtime run -D coredump=go-trap.coredump validator.p2.wasm`. The page bytes were
then read from the core dump's data section with a short throwaway script that is not included.
- **Call trace** (`WASMTIME_LOG=wasmtime_wasi=trace`): `monotonic-clock.now` ×2 and
  `random.get-random-bytes` succeed. Both go through `State::with` (lib.rs:714, :2400), so the
  canaries were intact. Then ~13 ms of Go runtime work runs with no host calls, and the next
  `State::with` (`args_sizes_get`, from `runtime.goenvs`) traps.
- **Core dump at the trap** (`wasmtime run -D coredump=…`):
  - memory is 129 pages (Go started at 40);
  - adapter globals: `__stack_pointer` = 2686896, `internal_state_ptr` = 2686976 (page 41),
    `allocation_state` = 4 (`StateAllocated`);
  - the 64 KiB `State` page reads **all zeros**, so the canaries and every other field are gone;
  - page 40 (the adapter stack) holds only 41 live bytes.
- **Controlled tests without Go** (`adapter-trap/*.wat`, same adapter):

  | Test | Core p1 | Component |
  |---|---|---|
  | `c0-control` | 0 | 0 |
  | `c1-go-sbrk`: Go-style "grow 1 page but clear `[old brk, +64KiB)`" | 0 | **134, `assertion failed at adapter line 2857`** |
  | `c2-honest-grow`: same, but clear the page `memory.grow` returned | 0 | 0 |
  | `c3-heap-on-stack-page`: heap data in the top 256 bytes below the initial break | 0 | **non-zero** |

  `c1` reproduces the exact Go trap. `c3` shows the reverse, silent corruption: adapter stack
  frames overwrite what Go counts as heap. (The adapter maps any non-zero `proc_exit` to exit 1,
  so `c3` reports "changed", not a count.)
- **Causal fix test.** `adapter-trap/go1.24.7-runtime-diagnostic.diff` is diagnostic only. It is
  derived from Go's BSD-3-Clause runtime source, and its header records provenance. It is applied
  to scratch copies with `go build -overlay`; GOROOT is not modified, and nothing outside
  `run.sh` step 7 references it. It drops the `osinit` claim, and makes `sbrk` skip
  pages it did not grow (`if currentMemory()*64KiB != blocMax { bl = blocMax = currentMemory()*64KiB }`).
  - The validator and a stress program run as 0.2 components. The stress program grows the heap
    300 × 256 KiB, with clock and env calls in between.
  - Their outputs are byte-identical to stock core p1: all four builds are compared with
    `cmp` against `expected/validator-invalid.stdout` or `expected/stress.stdout`.
  - Stock components still trap: exit 134, with `assertion failed at adapter line 2857` on stderr.
  - Core p1 behaviour is unchanged.
  - `run.sh` also asserts that the patched build differs from the stock build, so the overlay
    really was applied.
  - Two mutation runs on scratch copies made `run.sh` exit 1, as they should:
    - corrupting one expected-output file produced 3 FAILs;
    - replacing the patch with a comment-only no-op produced 3 FAILs, because both patched
      components trapped and one negative control became wrong.
  - Exit code 134 is SIGABRT as Wasmtime reports a trap on Linux x86_64, the only platform
    checked.

**Limits:**
- **The patch is diagnostic only**, not a fix to ship. It wastes the unused initial pages, it was
  not tested in ix-wasm/wasmi or with GC under load, and it was not reported upstream.
- **Exit codes are lost through the adapter.** `proc_exit(2)` becomes exit 1, because
  `wasi:cli/exit@0.2` `exit(result)` carries only ok/err.
- **No upstream issue was checked.** Searching GitHub issues would need the GitHub API, which
  this session's proxy blocks.
- **Real fixes would need one of:**
  - a Go runtime that tolerates foreign `memory.grow` calls;
  - an adapter with memory separate from the main module's;
  - a Go toolchain that emits components natively.

  None of these is available in Go 1.24.7.

**Sources:**
- [WASI 0.3 release notes](https://github.com/bytecodealliance/wasi.dev/blob/main/docs/releases/wasi-p3.md)
- [adapter source v50.0.1](https://github.com/bytecodealliance/wasmtime/tree/v50.0.1/crates/wasi-preview1-component-adapter)
- [`wasip3` 0.9.0](https://crates.io/crates/wasip3/0.9.0+wasi-0.3.0), whose `examples/cli-command.rs` is the base pattern used here
- Go runtime sources read from the local go1.24.7 toolchain (`/usr/local/go1.24.7/src/runtime`)
