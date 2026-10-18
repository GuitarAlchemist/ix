# IX control plane feeds — prototype

[Français](README.fr.md)

The scripts behind two views of the IX fleet, as they ran on 2026-10-18:

- the **IX Control Room**, a private dashboard Artifact of IX's pull requests,
  main-branch runs, issues, crate tiers and Indigo's work in progress;
- the **control snapshot** the 3D Observatory scene reads: a private
  `snapshot.json` for the user's own view, and a `public.json` that the public
  stream may show.

Kept here as the primary source for the design, not proposed for main as it
stands: the paths and the Artifact's asset ids are this machine's.

## Files

| File | What it does |
| --- | --- |
| `gather.py` | Reads GitHub through `gh` and origin/main through git; writes one JSON file of row objects per dataset. |
| `gather_indigo.py` | One row per Indigo work item. Heads and PR states are read; `verdict`, `review` and `next` are hand-kept notes. |
| `refresh.py` | `plan` runs both gatherers into `data-<stamp>/`, prints which datasets changed by sha256, then rewrites the snapshot. `record` notes each new asset in `state.json`. The uploads themselves are Claude's Artifact tool calls. |
| `make_docs.py` | Writes the dashboard's store documents, one per dataset, for one batch write. |
| `index.html` | The dashboard page. It derives stages, counts and weeks from the rows, as calculations its readers can inspect. |
| `control_snapshot.py` | Writes `snapshot.json` and `public.json` under `%LOCALAPPDATA%\ix-control`, atomically; `--every 5` rewrites them every 5 s. |
| `demo-plan.json` | The work in progress and the next demo, hand-edited as the work moves. |

`IX_CLONE` names the ix clone (default `~/source/repos/ix`). `IX_DASH_SCRATCH`
and `IX_DASH_TASK` name the folders that hold the result files and review
bundles `gather_indigo.py` recomputes counts from.

## The two snapshot files

`snapshot.json`, schema `ix-control-snapshot/0`, is for the user alone. It
names sessions, quotes claims-ledger notes and gives the local paths of the
DuckDB databases, so it never goes on the stream. It holds:

- open PRs with CI and Codex P0/P1 counts, and the latest run per workflow on
  main;
- live sessions, from `~/.claude/sessions/*.json` (never the `*.key` files),
  and active lanes from `~/.agents/claims.jsonl`;
- the decisions waiting on the user, and how long the working lanes say they
  can run without them;
- every DuckDB database under the repos' `state/` and `code/` folders, with
  its tables, read with `duckdb -readonly` every 5 minutes. A database another
  process holds open shows as `busy`.

`public.json`, schema `ix-public/0`, is the only part the public stream may
show:

- the busy and waiting sessions, by name and repo, and a count of idle ones;
- the number of decisions waiting on the user;
- the demo plan's titles and progress, and the next demo's time;
- the machine gauges from the probe host (prototype/probes, PR #390). A stale
  or missing sample gives `null`, so the scene shows no gauge rather than an
  old one;
- `viewers`, the stream's viewer count, which the player page's front door
  writes to `viewers.json` every 5 s. A missing count, or one older than 30 s,
  gives `null`.

Both schemas are provisional: a reader keys on `schema` and ignores fields it
does not know.

Every title in `demo-plan.json` is `{en, fr, es}`, so the scene can show the
whole board in its own language. `public.json` carries them as `titles`, and
keeps `title` in French for older readers. Process gauges carry `labels` the
same way.

A failed rewrite is logged to `snapshot-loop.log` and the loop carries on. On
2026-10-18 the loop had died at 15:47 on a half-written `data-*` folder, which
hid the gauges during a demo.

## Showing a demo in the scene

A demo is a one-shot script, started at the demo's time, that writes the
lines the bot is to say to the scene's public speech channel, and moves
`next_demo` in the plan. The channel holds one line of `"<lang>: <text>"`, at
most 400 characters, written atomically: the bot says it on the stream. Only
that channel reaches the stream, never the private captions.

## Not done

- The Artifact's asset ids are written into `make_docs.py`.
- Nothing checks that `public.json` stays free of names and paths; it rests on
  `public()` building it field by field.
- Windows only.
