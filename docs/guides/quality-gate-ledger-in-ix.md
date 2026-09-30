# The quality-gate ledger in IX

**Contract:** [`docs/contracts/2026-05-24-quality-gate-ledger.contract.md`](../contracts/2026-05-24-quality-gate-ledger.contract.md)
**French:** [`docs/fr/guides/registre-des-portes-qualite.md`](../fr/guides/registre-des-portes-qualite.md)

## What it is

One append-only JSONL file per repo, at `state/quality/gate-ledger.jsonl`. Each
line is one quality gate's verdict at one moment: who ran, what they measured,
whether it passed.

```
{"schema_version":1,"schema":"quality-gate-ledger-v1","id":"01a0b19d-…",
 "run_at":"2026-09-17T23:04:24Z","source":"ix-doctor","domain":"harness",
 "decision":"pass","metric":{"name":"doctor_checks_failing","value":0.0,"threshold":0.0},
 "extra":{"verdict":"T","checks":{"registry-snapshot":"ok", …}}}
```

## Who writes it

**`ix doctor`** — the pre-PR gate. Every run appends exactly one row:

```bash
cargo run -p ix-skill --bin ix -- doctor
```

One row per run, not per check. The gate's answer is its aggregate verdict —
that is what the exit code reports — so `metric` is the failing-check count
against a threshold of zero, and the per-check breakdown rides in `extra`
alongside the run mode (`--write` / `--full`). A row produced without `--full`
skipped the clippy and test checks, so comparing it with a `--full` row without
reading `extra.mode` would compare two different gates.

The human output ends with the path it wrote. If the append failed, it says so
on stderr and the gate's own verdict is unchanged — a ledger the filesystem
refused must not turn a green repo red.

`ix-sentrux-gate-writer` (`crates/ix-quality-trend/src/bin/sentrux_gate_writer.rs`)
is a second producer, still dormant: it wraps `sentrux gate` and needs a
`sentrux` binary on PATH, which this machine does not have.

## Who reads it

The `ix_quality_gate_history` MCP tool:

```jsonc
{ "source": "ix-doctor", "decision": "fail", "since": "2026-09-01T00:00:00Z", "limit": 20 }
```

The default path resolves against the **ix workspace root**, not the process
working directory. Pass `ledger_path` to read a sibling repo (ga writes its own
ledger).

### Read `ledger_status` before `count`

```jsonc
{ "ledger_status": "absent", "count": 0, "note": "no ledger at this path — …" }
```

`count: 0` on its own is ambiguous, and one of its two meanings is dangerous:

| `ledger_status` | What `count: 0` means |
|---|---|
| `absent` | No gate has ever recorded a run here. **Not** evidence that gates passed. |
| `empty` | File exists, no rows. Same weight as `absent`. |
| `present` | Gates ran; your filter excluded them all. This one *is* evidence. |

Before this was wired, the tool answered every query from an absent file, and
`count: 0` looked exactly like a clean history. That is the failure mode the
repo calls green-but-dead: a reassuring answer with nothing behind it.

## Practical notes

- **Gitignored in ix**, committed in ga. `ix doctor` runs on every
  contributor's machine, so tracking the file would put an append-only JSONL on
  the merge path of every PR. ga commits its ledger because a CI script, not
  each contributor, writes it. So: ix's ledger is per-checkout history, and a
  fresh clone reads `absent` until the gate runs once.
- **Bounded.** At 4 MiB (~10k rows) the live file rotates to
  `gate-ledger.1.jsonl`; one generation is kept and consumers read the live file
  only. Anything that must outlive rotation belongs in a dated snapshot under
  `state/quality/`.
- **Crash-safe per row.** Each row is one `write_all` on an `O_APPEND` handle,
  so concurrent producers cannot interleave half-lines. Readers skip blank lines
  and degrade a malformed v1 line to legacy v0, so the worst case is a dropped
  row rather than an unreadable file.
- **Legacy v0 rows coexist.** ga's older chatbot-PR-shaped rows have no
  `schema_version`. They are excluded from `rows` but still count as history, so
  a v0-only ledger reads `present`, not `empty`.
