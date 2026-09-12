# IX-pi — Pi as an independent second agent over this repository

An experiment. Not wired into CI, not depended on by any crate, and safe to
delete: `rm -rf IX-pi` removes it entirely.

[Pi](https://github.com/earendil-works/pi) is a separate agent harness (MIT,
TypeScript, ~104k stars). This directory installs it and uses it for one narrow
purpose.

## Why this, and not the obvious alternatives

Two other ideas were considered first and rejected:

- **An MCP bridge, so Pi could call IX's tools.** Pi speaks no MCP — verified:
  no `@modelcontextprotocol/*` dependency in `pi-coding-agent`, `pi-agent`, or
  `pi-ai`; the only hits are transitive lockfile entries. Building the bridge is
  real work whose payoff is *capability Claude Code already has* over the same
  federation. It answers no question we cannot already answer.
- **Porting Pi's comparative eval harness.** Worth doing, and genuinely the best
  idea in their tree — `packages/evals` reports pass-rate lift in percentage
  points alongside token, latency and cost deltas. But it does not require Pi to
  be installed, so it is not this experiment.

What Pi uniquely offers is **independence**: a different runtime, a different
provider, a different system prompt, and no governance layer. An answer produced
that way is *evidence* about the repository rather than an echo of the governed
lane's own reasoning. That is the one thing the ecosystem cannot manufacture for
itself, and it is what this prototype harvests.

## The slice

One prompt, end to end, touching every layer:

```
prompt file ──▶ pi (ephemeral git worktree, --mode json)
                          │
                    raw JSONL events  ── kept verbatim, the evidence
                          │
                  to-contract.mjs
                          │
              state/<date>-<slug>.second-opinion.json
```

```bash
cd IX-pi
npm install --ignore-scripts          # once
bash probe/run-probe.sh probe/prompts/silent-skips.md
node probe/to-contract.mjs 2026-09-12-silent-skips
```

Provider readiness is Pi's own: `npx pi auth check --provider openai`. On this
machine `openai` is ready; `anthropic` and `google` are not.

## Containment

Pi ships **no permission system** — its own README says so, and recommends
containerisation. It does have a tool allowlist (`--tools`), which is real:
`-t read` was verified to deny `bash`, `write` and `edit`.

That is not sufficient on its own, for two reasons:

1. Pi has no `grep` or `glob` tool, so any *search* task needs `bash`, and bash
   can write.
2. A misspelled tool name in `--tools` is **accepted silently** and yields a run
   with no tools at all rather than an error. The flag cannot be trusted on its
   own.

So containment is structural, not declarative. Probes run in an ephemeral
`git worktree` at a pinned commit; the real working tree is never the cwd. After
each run the worktree is diffed and the modified-path count is recorded in the
artifact, so "it did not write" is a measurement (`worktree_unmodified`) rather
than an assumption.

## Admissibility

Every artifact carries an `admissible` boolean and the checks behind it. The one
that matters: **`used_tools`**. An agent that answers without opening a single
file has reported its prior, not a finding about this repository — and it will
sound exactly as confident either way. Such a run is marked inadmissible and
`to-contract.mjs` exits non-zero. Verified by feeding it a synthetic run whose
sole output was a fluent "There are no such tests."

`no_tool_errors` is reported but deliberately does **not** invalidate a probe;
agents recover from a failed `grep` routinely.

## First result: the probe found something the governed lane missed

Question: *which tests pass without asserting anything, because they return
early when an external tool is absent?* Ground truth was known — five such
duckdb sites had just been found and fixed by hand in ix#294 / PR #319.

Pi, given only the prompt, found **all five**, with the correct file paths, test
names and mechanism. Then it named a sixth:

> `crates/ix-sentrux-annotations/tests/end_to_end.rs` —
> `live_sentrux_check_against_ix_worktree` — `if !exe.exists() { eprintln!(...); return; }`

Verified, and worse than the duckdb cases. The path comes from
`crates/ix-sentrux-annotations/src/lib.rs:68`:

```rust
pub const DEFAULT_SENTRUX_EXE: &str = "C:/Users/spare/bin/sentrux.exe";
```

A hardcoded absolute path into one developer's home directory, in a `pub const`
in library code. The binary exists on that machine, so the test asserts there
and silently returns on every CI runner and every other checkout. The duckdb
tests could be fixed by installing duckdb; this one cannot be fixed by anyone
but that user. There is also a second silent `return` at line 106 that Pi did
not mention.

Cost of the run that found it: 26,539 tokens, **$0.02**, 2m53s, 26 tool calls,
zero files modified.

## Honest limits

- One probe, one model (`openai/gpt-5.2`), one question. Nothing here
  establishes a hit rate; the calibrated agreement on five known sites plus one
  verified new one is a promising single data point, not a measurement.
- Every Pi finding still needs verifying by hand. The sixth one was real; that
  is not a reason to trust the seventh.
- `node_modules/`, `worktree/`, `sessions/` and `.pi/` are git-ignored. Session
  transcripts carry prompts and file contents and must not be committed.
- ix stays a pure Rust workspace: no root `package.json`, no Cargo member, no
  crate depends on this.

## Where it could go

The artifact shape (`ix-pi.second-opinion/v0.1`) is deliberately close to the
ecosystem's JSON-on-disk convention, because the interesting next step is not
more probes — it is **disagreement**. Two harnesses answering the same question
with different tools and no shared prompt produce exactly the multi-source
disagreement the belief substrate is starved of, and which is currently
generated and discarded. Persisting it is the unlock.
