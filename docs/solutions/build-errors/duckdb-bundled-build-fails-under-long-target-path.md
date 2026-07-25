---
title: "ix-duck `duck` feature fails with a bogus missing-header error under a long CARGO_TARGET_DIR (Windows MAX_PATH)"
category: build-errors
date: 2026-07-24
tags: [windows, max-path, cargo, duckdb, libduckdb-sys, cargo-target-dir, cc, msvc]
symptom: "cargo build -p ix-duck --features duck fails with `fatal error C1083: Cannot open include file: 'duckdb/common/insertion_order_preserving_map.hpp': No such file or directory` even though the header exists on disk"
root_cause: "Windows MAX_PATH (260 chars). The bundled DuckDB C++ amalgamation nests deeply under CARGO_TARGET_DIR; a long target path pushes include resolution past the limit and cl.exe reports the header as missing rather than as a path-length error"
---

# ix-duck `duck` feature fails under a long CARGO_TARGET_DIR

## Problem

Building the optional bundled-DuckDB feature:

```bash
cargo build -p ix-duck --features duck
```

fails during the `libduckdb-sys` build script with:

```
warning: libduckdb-sys@1.10503.1: ToolExecError: command did not execute successfully
  (status code exit code: 2): "...\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe" ...
...\out\duckdb\src\include\duckdb/common/types/value.hpp(21): fatal error C1083:
  Cannot open include file: 'duckdb/common/insertion_order_preserving_map.hpp':
  No such file or directory
```

The trap: **`insertion_order_preserving_map.hpp` exists.** It is present in the
extracted amalgamation, on the include path, spelled correctly. Chasing it as a
vendoring bug, a bad `duckdb` version pin, or a corrupted download is wasted
effort — none of those are the cause.

## Root cause

Windows `MAX_PATH` (260 characters).

`libduckdb-sys` extracts the DuckDB C++ amalgamation *underneath* the cargo
target directory, and the resulting tree is already deep before any user path
is prepended:

```
<CARGO_TARGET_DIR>\debug\build\libduckdb-sys-<hash>\out\duckdb\src\include\duckdb\common\types\value.hpp
```

That suffix alone is ~110 characters. With a default in-repo `target/` the
total stays comfortably under the limit. With a relocated *and* long
`CARGO_TARGET_DIR` it does not. The failing invocation in this case used a
scratchpad path, giving a ~240-character resolved header path:

```
C:/Users/<user>/AppData/Local/Temp/claude/<project-slug>/<session-uuid>/scratchpad/zig-spike/target-baseline\debug\build\libduckdb-sys-<hash>\out\...
```

`cl.exe` does not report path-length failures distinctly — it surfaces them as
`C1083: Cannot open include file`, identical to a genuinely absent header. That
is what makes this expensive to diagnose: the error message actively points away
from the real cause.

Note this is a *different* failure mode from
[Windows Application Control blocking cargo test binaries](windows-app-control-blocks-cargo-test-binaries.md),
which also gets triggered by relocating `CARGO_TARGET_DIR` — but that one fails
at *execution* with `os error 4551`, not at *compilation* with `C1083`.

## Fix

Use a short target path when building the `duck` feature:

```bash
CARGO_TARGET_DIR=C:/ixsp/t cargo build -p ix-duck --features duck
```

Or simply build into the repo's default `target/`, which is short enough.

Verified both ways on `feat/ix-bracelet-fourier` @ `6a2850d`:

| `CARGO_TARGET_DIR` | Result |
|---|---|
| ~130-char scratchpad path | `C1083`, exit 101 |
| `C:/ixsp/t` | **exit 0, 1m51s** |

The `duck` feature itself is healthy. If you hit `C1083` here, the build is
fine and your path is too long.

### Alternative: enable long paths system-wide

Setting `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`
raises the limit, but only for applications whose manifest opts in. MSVC's
`cl.exe` historically does not, so this is **not** a reliable fix for this
specific failure — prefer the short target path.

## Why this matters beyond ix-duck

Any workflow that relocates `CARGO_TARGET_DIR` to a temp or session-scoped
directory — CI scratch dirs, sandboxed agent sessions, parallel worktree builds
— can trip this on **any** crate that unpacks a deep C/C++ source tree into
`OUT_DIR`. `libduckdb-sys` is simply the deepest such tree in this workspace.
When isolating a build into a scratch target dir, keep the path short by
default.

## Discovered during

A timeboxed spike (2026-07-24) evaluating `zig cc` / `zig c++` as a hermetic C
toolchain for IX's native dependencies. The spike's own conclusion was negative
and is not repeated here; this path-length gotcha was incidental to it, and is
the only durable finding.

For the record, so the Zig question is not re-litigated: `zig c++` **cannot**
target the MSVC ABI at all. It compiles its bundled `libcxxabi` against MSVC's
`vcruntime_typeinfo.h` and dies on redefinitions of `bad_cast` / `bad_typeid` /
`type_info`. It works for the `-gnu` ABI, which is incompatible with this
workspace's `x86_64-pc-windows-msvc` host. `zig cc` (C only) does work, but
`libduckdb-sys` is the only crate in the workspace that compiles native code,
and it is C++ — so the intersection of "IX compiles it" and "Zig can compile
it" is empty. (`ort`/ONNX consumes prebuilt binaries and compiles nothing:
`ix-skill --features embeddings` builds in ~73s with no native step.)
