---
title: "A Rust test that hands a canonicalized Windows path to Git bash hangs for minutes: the `\\\\?\\` prefix reads as the UNC share `//?/`"
category: test-failures
date: 2026-10-22
tags: [windows, msys, git-bash, canonicalize, verbatim-prefix, unc, hang, ix-skill, doctor, fixtures]
symptom: "cargo test for ix-skill's doctor suites took 629 s and 1220 s on Windows with every test green; a bash fixture spawned by the test sat idle, and its child `bash \"$HOOK\"` with `dirname \"$0\"` never returned until the 120 s timeout"
root_cause: "Path::canonicalize on Windows returns a verbatim path, `\\\\?\\C:\\...`; replacing backslashes with slashes turns it into `//?/C:/...`, which msys bash parses as a UNC path on a host named `?` and tries to resolve over the network before giving up"
---

# msys bash hangs on a canonicalized Windows path

## Problem

The `hooks` check of `ix doctor` runs the behavioural fixtures under `.claude/hooks/test-*.sh`
through Git bash. Its tests resolve the repo root with `canonicalize()` (the pattern every doctor
test uses) and the check converted that path to forward slashes before handing it to bash. The
two test suites passed, but took 629 s and 1220 s instead of a minute; the process list showed
the fixture's bash and its child idle for minutes, each hitting the 120 s fixture timeout in
turn.

## Root cause

`canonicalize()` on Windows yields `\\?\C:\Users\...` (the verbatim prefix). A naive
backslash-to-slash conversion gives `//?/C:/Users/...`. To msys bash a leading `//host/` is a UNC
share, so `?` becomes a host name: `bash //?/C:/.../test-x.sh` and the `dirname "$0"` inside the
fixture both wait on name resolution before falling back. The scripts did run eventually, which
is why every test stayed green while the wall clock exploded.

## Fix

`crates/ix-skill/src/doctor/hooks.rs`, `posix()`: strip `\\?\` (and `\\?\UNC\` back to `\\`)
before replacing the separators. Suites back to 33 s and 56 s; `ix doctor` 40 s with the
fixtures included. PR #424.

## How to find it again

A Windows-only slowdown of a test that spawns bash, python or any msys tool with a path built
from `canonicalize()`, every test still green:

```powershell
# the idle children tell the story: a bash whose argument starts with //?/
Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Select-Object ProcessId, ParentProcessId, CommandLine
```

`std::path::absolute` (Rust 1.79+) gives a path without the prefix when canonical symlink
resolution is not needed; otherwise strip the prefix as `posix()` does.
