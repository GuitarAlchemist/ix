---
title: "Windows Application Control blocks cargo test/build-script binaries (OS error 4551)"
category: build-errors
date: 2026-04-10
tags: [windows, app-control, wdac, cargo, rust, test-execution]
symptom: "cargo test fails with 'An Application Control policy has blocked this file. (os error 4551)' on freshly compiled test binaries or build scripts"
root_cause: "Windows Defender Application Control (WDAC) / App Control for Business policy blocks execution of unsigned binaries under target/ and temp paths"
---

# Windows Application Control blocks cargo test binaries

## Problem

Running `cargo test -p <crate>` on a Rust crate fails immediately after linking
the test binary:

```
Finished `test` profile [unoptimized + debuginfo] target(s)
Running unittests src\lib.rs (target\debug\deps\<crate>-<hash>.exe)
error: test failed, to rerun pass `-p <crate> --lib`

Caused by:
  could not execute process `.../target/debug/deps/<crate>-<hash>.exe` (never executed)

Caused by:
  An Application Control policy has blocked this file. (os error 4551)
```

The same error appears on build scripts for any dependency that has one
(`tree-sitter-language`, `serde_json`, `proc-macro2`, `cc`, etc.) when the
target directory is relocated (e.g. via `CARGO_TARGET_DIR=/tmp/...`).

This blocks the **entire cargo test pipeline** on affected machines — not
just the final test binary.

## Root cause

Windows Defender Application Control (WDAC, also branded "App Control for
Business") enforces a policy that prevents unsigned or unknown executables
from running. A recent Windows 11 policy update appears to apply this
restriction to executables produced under *developer* paths like
`target/debug/deps/`, `target/release/deps/`, and `C:/Users/<user>/AppData/Local/Temp/`.

This is **not**:
- A Rust bug
- A Cargo bug
- An MSVC linker issue (that would show up at link time, not run time)
- The LNK1318 PDB size limit (different symptom, different error code)

It **is**:
- A Windows security policy rejecting unsigned binaries from "untrusted"
  directories at exec time
- Triggered by WDAC or SmartScreen-adjacent policies, typically configured
  by IT / Group Policy / Intune
- Triggered *after* successful compilation — the binary builds fine, it
  just cannot run
- Equally applicable to debug and release builds
- Equally applicable to the default `target/` and relocated `CARGO_TARGET_DIR`

## Diagnosing

Three fast checks:

1. **Does `cargo check` pass?** If yes, the source compiles; you are hitting
   execution policy, not a code bug.
2. **Does `cargo clippy -- -D warnings` pass?** Clippy runs the compiler
   frontend only — it does not execute any built binaries. If clippy passes
   and `cargo test` fails with error 4551, it is definitively an execution
   policy problem.
3. **Does `CARGO_TARGET_DIR=/tmp/... cargo test` also fail?** If yes with
   the same error, the policy covers temp paths too, which rules out the
   "just relocate target" workaround.

## Working solutions

In order of operational impact, not preference:

**1. Ask IT to allow-list developer binary paths**

The cleanest fix: request an exception for `target/**/*.exe` under developer
working directories. Requires IT cooperation; may not be possible on locked
machines. Once allow-listed, the error disappears with no code changes.

**2. Sign the test binaries**

WDAC can be configured to allow signed binaries from a trusted CA. Sign
every test binary before execution. This is viable for CI but tedious for
day-to-day `cargo test` loops.

**3. Relocate the target dir to an allow-listed path**

If WDAC has a permitted path for user binaries (common: `~/dev/bin/`,
`~/.cargo/bin/`), set `CARGO_TARGET_DIR` to a subdirectory under it. Does
*not* always work — on some policy configurations the restriction is
content-based, not path-based.

**4. Run tests on a machine without the policy**

WSL2, a Linux VM, or a non-corporate build host. Tests run there; commits
come back via git. The ecosystem-integration solution for teams that cannot
get (1) or (2) approved.

**5. Accept compile-only verification for affected sessions**

For isolated refactors where the type system carries most of the weight
(struct-field renames, enum additions, API shape changes), `cargo check`
plus `cargo clippy -D warnings` is a meaningful verification signal even
without test execution. The code does not run, but the signature stays
consistent and the lint bar is met. Document this clearly in the commit
message so reviewers know the verification level.

## Prevention

1. **Before any development session on Windows, run `cargo test -p <any-crate>`
   in the repo.** If error 4551 appears, stop and resolve policy before
   writing code that depends on runtime verification.
2. **CI on a Linux runner is mandatory** for any workspace that cannot
   guarantee a WDAC-free dev environment. Windows CI that exercises the
   same policy will hit the same wall.
3. **When hitting this mid-session, commit cargo-check-verified code in a
   marked commit** (e.g. `refactor(foo): <change> (compile-only, App Control)`)
   so the git log records the verification gap honestly.
4. **Do not silently loop retrying `cargo test`.** The error is not a
   transient flake — every rebuild produces a fresh binary with a new
   hash that the policy will block again. Retry spirals burn time and
   context without progress.

## Related

- Windows docs: https://learn.microsoft.com/en-us/windows/security/application-security/application-control/windows-defender-application-control/
- Prior session compound: 95+ tests previously ran in this repo, meaning
  the policy was applied between then and now — likely a Windows Update
  or a Group Policy refresh. If tests worked yesterday and fail today,
  check Windows Update history and `Get-AppLockerPolicy -Effective` in an
  elevated PowerShell.
- Distinct from `docs/solutions/build-errors/windows-lnk1318-pdb-size-limit.md`
  (that one is linker-side, fires before the binary is ever produced).
