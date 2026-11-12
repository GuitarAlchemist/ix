# #484 provider evidence: Claude Code's programmatic mode, pinned

Research note for [#484](https://github.com/GuitarAlchemist/ix/issues/484) (map
[#483](https://github.com/GuitarAlchemist/ix/issues/483)). Local, not published. It records
evidence; it decides nothing (the operator decides #484). No model call was made for it.

Status words: **source-read** (a document or code was read), **tested** (a command ran here),
**indeterminate** (neither settles it).

## Pinned release

- Installed: `2.1.314 (Claude Code)` at `~/.local/bin/claude.exe` (tested: `claude --version`).
- Help of that release: `evidence/claude-2.1.314-help.txt` (tested: `claude --help`, 313 lines).
- Docs: <https://code.claude.com/docs/en/headless>, read 2026-10-30. The page does not pin itself
  to a release; it cites minimums (v2.1.259, v2.1.286, v2.1.302, ...), all below 2.1.314.

## What the release offers a caller (help, tested)

`-p/--print` with `--output-format text|json|stream-json`; `--input-format text|stream-json`
("realtime streaming input", print only); `--include-partial-messages`; `--replay-user-messages`
(stream-json in and out); `--permission-prompts host|none`; `--permission-mode` (`acceptEdits`,
`auto`, `bypassPermissions`, `manual`, `dontAsk`, `plan`); `--json-schema`; `--max-budget-usd`;
`--no-session-persistence`; `--session-id <uuid>`; `--resume`.

## What the docs say about each event class #484 names (source-read)

| #484 class | Documented signal | Status |
|---|---|---|
| request acceptance | none documented: `system/init` says the session started, not that a request was accepted | indeterminate |
| output delta | `stream_event` with `event.delta.type == "text_delta"` (needs `--include-partial-messages`, `--verbose`) | source-read |
| tool activity | `assistant` / `user` messages with `tool_use` / `tool_result`; `parent_tool_use_id` ties a subagent's messages to its call | source-read |
| permission wait | with a host, the run waits for it; `--permission-prompts none` denies, emits `permission_denied` system messages, lists `permission_denials` in the result | source-read |
| final result | the stream's last line is a `result` message; `structured_output` is absent when no valid output was produced | source-read |
| end of input | not specified for stream-json input beyond "background work waits after stdin closes" | indeterminate |
| termination | exit 0 success, non-zero failure; a failure inside the run is printed as the result on stdout; SIGTERM → exit 143, the turn in progress is left unfinished with no result | source-read |
| retries | `system/api_retry` with `attempt`, `max_retries`, `error` category | source-read |

Also documented: `system/init` carries a `capabilities` array (feature detection, e.g.
`interrupt_receipt_v1`); piped stdin is capped at 10 MB; since v2.1.302 a `-p` run starts in
Manual mode when nothing sets a mode; costs in the result are client-side estimates.

**Not on the page:** the schema of a stream-json *input* line and the list of `result` subtypes.
Both point to the Agent SDK TypeScript reference, not read yet. Until then a consumer must treat
an unknown `type`/`subtype`, a malformed or truncated line, and a stream with no `result` line as
not-a-success, which is #484's own rule.

## Seams IX already has (source-read, origin/main 99f2f869)

- `crates/ix-loop/src/node.rs`, `ClaudeCode` (a `SemanticNode`): one process per ask; `-p
  --safe-mode --tools "" --no-session-persistence --output-format json --model <m>`, optional
  `--json-schema`, `--system-prompt`; prompt on stdin; a deadline. All eight flags exist in
  2.1.314 (tested against the saved help).
- `crates/ix-baml/src/claude_cli.rs`, the IXQL `BamlOperation`: `--print --output-format json
  --json-schema ... --permission-mode manual --strict-mcp-config`; prompt on stdin. Its module doc
  records a measurement on 2026-07-29 (CLI 2.1.220): a trivial call cost $0.28, 45,840
  cache-creation input tokens of default system prompt, tool definitions and CLAUDE.md.
- No code in `crates/` or `go/` uses `--input-format stream-json` or `--output-format
  stream-json` (tested: `git grep stream-json -- crates go`).

## What this means for #484's alternatives

- **One process per task** exists twice already (ix-loop, ix-baml), both with `json` output: one
  envelope, no events. It proves process exit and envelope shape, not business completion.
- **Reused session** (stream-json in and out, several turns in one process) has no IX
  implementation. The ix-baml measurement is the argument for it: the per-call fixed cost is the
  default context, paid again by every process. Not measured on 2.1.314; the argument against is
  state leaking between tasks of one session.
- **SDK-backed adapter**: not evaluated here (no SDK package in the workspace).
- `--bare` would trim the fixed context but reads no OAuth login and needs an API key: excluded
  (no new key, operator rule).

## Next bounded step (offline, no model call)

A fixture test of a stream-json line classifier: valid run, malformed line, truncated last line,
unknown `type`, no `result` line, `permission_denied`, exit 143 with no result. Each must come out
as one of #484's classes or as not-a-success. Fixtures are hand-written from the documented shapes
above, so they are labelled synthetic until a recorded stream replaces them, and a recorded
stream needs a separately admitted live run.

**Done (tested), still synthetic:** `crates/ix-router-spike/src/bin/claude_stream.rs` (binary
`claude-stream`, 12 tests, each rule seen red under a mutant). What the caller can prove from one
process's stdout: which of the turns it sent ended, in order, on a `result` line of its own
session. Only `subtype: "success"` with `is_error: false` is a success, and it carries the count of
`permission_denials`. Everything else is a typed non-success: `result_error`, `malformed` (any
unparsable line voids the open turn, even before a success result), `truncated` (a last line without
newline, even when it parses), `limit_exceeded`, `terminated` (non-zero exit before the result, 143
included), `no_result`. Lines of another session are counted `stale` and ignored; a result that does
not carry the expected `session_id` as a string is `unattributed_result` and closes no turn; results
beyond the turns sent are `unexpected_result`. Limits: 1 MiB per line, 100,000 lines, 8 diagnostics of at most
200 bytes, the rest counted. Request acceptance and end of input stay unclassified: no documented
signal. Correlation is by order only, since no documented field carries a caller's task id.

French: [`docs/fr/research/agent-composition/2026-10-30-484-claude-code-programmatic-evidence.md`](../../fr/research/agent-composition/2026-10-30-484-claude-code-programmatic-evidence.md).
