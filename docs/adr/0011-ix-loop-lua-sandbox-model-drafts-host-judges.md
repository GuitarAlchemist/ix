# `ix-loop` — loops in sandboxed Lua; the model drafts, the host judges, a person promotes

Status: accepted (2026-10-16)

## Context

The wish is second-order agentic loops: loops that run other loops, judge them and write new
ones. Such a loop needs a semantic node, a place where it asks a model, and that node should be
Claude Code on the person's Max plan, not the API. ga already has a loop editor (a graph of
`{steps}` validated by `ix_pipeline_validate`), but its state lives in the Vite process's memory
and it has no headless use.

An experiment on 2026-10-16 (a throwaway host, `mlua` 0.10 with Lua 5.4) ran such loops. Three
things came out of it:

- **Sandboxed Lua is a good loop language.** Sonnet wrote working Lua on its first try, and a
  fresh state per loop with only data crossing kept loops apart.
- **Budgets must be carved, not shared.** The runs made twelve calls in all (4 Sonnet, 8 Haiku),
  each tree inside its root's budget, because each child's share was taken from its parent and
  refunded unspent.
- **A writer judged on what it saw cheats.** In run 1 the writer, shown one instance, returned
  that instance's answer instead of a loop. Judged on instances it never saw, the trick fails.

## Decision

**1. A loop is a Lua 5.4 chunk run in a fresh sandbox.** The crate is `crates/ix-loop`
(experimental). Only `table`, `string`, `math`, `coroutine` and `utf8` are loaded; `io`, `os`,
`require`, `debug`, `dofile`, `loadfile`, `collectgarbage` and `string.dump` are gone, `load`
takes text only, and a state may use 64 MB. Only JSON data crosses between loops: no function,
no coroutine, no shared state. A loop reaches the world through `claude`, `run`, `tasks` and
`loops`, nothing else.

**2. The engine is behind a seam.** `sandbox::Engine` runs one loop; `Lua54` is the only engine.
Luau, which has a native interrupt, could replace it without touching budgets, the node or the
store.

**3. Budgets are carved from parent to child.** A budget counts fuel (an instruction hook),
semantic-node calls, a deadline and depth (at most 3). A child gets the lesser of what it asks
for and what its parent has left, at once, and what it leaves goes back. The root's budget bounds
the tree. Calls are counted, not dollars: `--max-budget-usd` applies to the API only, and calls
are what spends the plan's usage windows. A caught error, a message handler, a coroutine or a
finalizer cannot hide a spent budget: `pcall`, `xpcall`, `coroutine.resume` and
`coroutine.close` raise it again; `xpcall`'s handler, which Lua runs with the hooks still off
when the hook raised the error, is skipped once the budget is spent; a coroutine is hooked while
it runs; and `__gc` is refused.

**4. The semantic node is Claude Code headless, on the plan.** `ClaudeCode` runs
`claude -p --safe-mode --tools "" --no-session-persistence --output-format json --model <m>`,
with `--json-schema` and `--system-prompt` when asked, the prompt on stdin, in the store's
folder. It never passes `--bare`, which reads only an API key. `Scripted` is the test double, so
CI never needs `claude`.

**5. The model drafts; the host's oracles judge.** Oracles are task families the host owns
(`lipogram`, `acrostic`) and checks in Rust. `tasks.judge` runs a draft on instances the host
draws and has handed to no loop in the run, and reads only the answer: never the draft's own
claim that it passed.

**6. A loop can propose; only a person promotes.** The store is on disk (`IX_LOOP_STORE`, default
`~/.cache/ix-loop`): `proposals/`, `library/` and `ledger.jsonl`. A loop can `propose`, `list`
and `load`. Only `Store::accept`, called by the `ix-loop accept` command, moves a proposal into
the library, and no loop can reach it.

**7. `mlua` compiles C.** It builds Lua 5.4 from vendored C sources (`lua-src`, through `cc`),
an exception to the pure-Rust convention like the bundled DuckDB engine behind `ix-duck`'s
`duck` feature, signed off by the user on 2026-10-16. The exception is this crate's: no other
crate may depend on `mlua` without its own decision.

**8. Headless first.** IX runs and stores loops; the ga editor becomes a client of the store
later. The MCP surface will be one tool, `ix_loop_run`.

**Rejected:**

- *Rhai, or a Lua written in Rust* — no C, but models know Lua far better than Rhai, and a
  young Lua reimplementation is a weaker sandbox than the reference VM; the experiment's
  first-try Lua was the point.
- *Loops as `{steps}` JSON only* — fine for the editor's graph, poor for control logic (retry,
  branch, judge). The graph will get a `loop` node whose body is Lua.
- *Budgets in dollars* — on the plan nothing is billed per call, so a dollar cap would never
  stop a loop; calls do.
- *`claude --bare`* — it needs an API key, so it would leave the plan.
- *Judging on the instance the writer saw, or trusting the draft's report* — run 1 showed both
  are gamed.
- *Letting a loop accept its own proposal* — the loop would be writer and judge of what runs
  next.

## Reversibility

**Two-way door while proposed and while the crate is experimental**: no other crate depends on
it, and the store's files are local. The proposal and library layout becomes one-way once the ga
editor reads it (slice B4), and the Lua capabilities once loops written outside this repo depend
on them. Adding a capability is backward compatible; renaming or removing one breaks loops.

## Revisit triggers

- **A loop escapes its budget** — a way to run Lua without the hook, or a C call that runs long
  (a string pattern that backtracks): then run each root loop in a child process with a wall-clock
  kill, or move to Luau's interrupt.
- **`ix_loop_run` lands (B2)** — then decide which IX tools a loop may call through MCP; the
  experiment's `ix()` capability is left out of B1.
- **The ga editor reads the store (B4)** — then freeze the proposal and library format.
- **Plan usage windows run out in practice** — then calls need a rate, not only a count.
- **Answers come back in the person's language despite `--safe-mode`** — the experiment's drafts
  came back in French; if it matters for oracles, pass a language in the system prompt.
