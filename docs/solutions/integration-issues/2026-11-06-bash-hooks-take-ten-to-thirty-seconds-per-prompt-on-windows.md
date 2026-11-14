---
title: "Every prompt waits 10 to 30 seconds before Claude starts, in ix on Windows"
category: integration-issues
date: 2026-11-06
tags: [claude-code, hooks, bash, git-bash, windows, latency, process-spawn, ix-hooks]
symptom: "UserPromptSubmit hooks correction-detector.sh took 26.3 s median (31.9 s max) and digest-staleness-nudge.sh 10.5 s, measured on the same stdin with CPU at 57-90 %"
root_cause: "On a loaded Windows machine each new process costs 1 to 2.5 s (git rev-parse alone: 2.5 s median through Git\\cmd\\git.exe). A bash hook spawns several per run (bash, git, grep, jq, date, stat), so the cost multiplies on every prompt and every edit."
---

## Problem

The two UserPromptSubmit hooks of `.claude/settings.json` delayed every prompt. Indigo flagged it after Gaia's NO-BASH-R1 note. Timing them on a loaded machine (other sessions and wmux kept the CPU at 57–90 %) gave:

| Hook | bash (median / max) | Go `ix-hooks` |
|---|---|---|
| correction-detector | 26.3 s / 31.9 s | 0.2 s |
| digest-staleness-nudge | 10.5 s / 19.3 s | 1.0 s |

The Go port's first build was slower than bash on the nudge: 6.5 s median. It still started `git rev-parse --show-toplevel` to find the repository root, and that one process cost as much as a whole script.

**Measurement trap.** `git show origin/main:<script> | Set-Content -NoNewline <file>` joins every line into one. The result is a script that is a single comment, which runs in about 300 ms and prints nothing. The first bash baseline was that empty script. `bash -x <file>` printing no trace exposed it.

## Root cause

On Windows, process creation dominates: a no-op Go binary took 0.1 s idle and up to 2 s under load. `Git\cmd\git.exe` is itself a launcher that starts `mingw64\bin\git.exe`.

Hook latency is therefore roughly the number of processes a hook starts. It is not the work the hook does.

## Fix

PR #512 (`feat/claude-hooks-go`) does two things:

- **One binary, no shell.** The bash hooks become subcommands of the Go binary `go/claudehooks/cmd/ix-hooks`.
- **No git process for the root.** `system.toplevel` / `findTopLevel` finds the repository root by walking up to the nearest `.git` directory or worktree file. `TestFindTopLevel` binds it, and a mutant without the walk-up is killed.

The prompt hooks now start no process at all.

## How to find it again

```powershell
$p = '{"prompt":"non, pas ce fichier"}'
1..5 | % { (Measure-Command { $p | & 'C:\Program Files\Git\bin\bash.exe' .claude/hooks/correction-detector.sh | Out-Null }).TotalMilliseconds }
1..5 | % { (Measure-Command { git rev-parse --show-toplevel | Out-Null }).TotalMilliseconds }
```

Extract a script from git with `(git show <rev>:<path>) -join "`n"` and write it with `[IO.File]::WriteAllText`, never with `Set-Content -NoNewline`. Run `bash -x` once to see that it really executes.
