# Probes — prototype

[Français](README.fr.md)

A host process that runs small Go programs, *probes*, each collecting one kind
of measurement from the machine. Edit a probe's source while the host runs and
the host rebuilds it and swaps the running process, without restarting itself.
Everything the probes print lands in one JSONL file that DuckDB reads.

This is a prototype on a `prototype/probes` branch: it answers whether the
loop *edit a probe → it runs → DuckDB sees it* works on Windows, and what it
costs. It is not wired into IX.

## The contract

A probe is a directory under `probes/` holding a Go `main` package. It prints
one JSON object per line on stdout:

```json
{"ts":"2026-10-07T22:47:41.123456Z","metric":"ram_free_mb","value":11133.4,"unit":"MB"}
```

`metric` (a string) and `value` (a number) are required; `ts` and `unit` are
optional. The host adds `probe` (the directory's name) and `build` (the hash
of its source), and a `ts` if the probe gave none. A line that is not such an
object becomes a `bad_line` event instead. What a probe writes on stderr
becomes `stderr` events. The host passes the sampling interval to the probe as
`PROBE_INTERVAL_MS`.

The host writes:

- `out/probes.jsonl` — the records;
- `out/host-events.jsonl` — its own events: `host_started`, `built`,
  `build_failed` (with `kept_running`, the build still running), `started`,
  `exited`, `stopped`, `start_failed`, `bad_line`, `stderr`, `host_stopped`;
- `out/bin/<probe>-<hash>.exe` — one executable per build.

## How the hot reload works

Go's `plugin` package does not exist on Windows, and Windows locks a running
executable. So each build is a new process: every 500 ms the host hashes each
probe's `.go` files (contents, not modification times); when a hash changes it
runs `go build` to an executable named after the hash, starts it, and stops the
old one only once the new one has printed its first line (or died, or 30 s
have passed), so a swap leaves no gap. If the new source does not build, the old build keeps
running and the host does not retry that source until it changes again. A
probe that crashes is restarted, at most every 2 s. A probe whose directory is
removed is stopped.

## Running it

Needs Go 1.27 on PATH and, for the aggregation, the DuckDB CLI.

```powershell
cd prototypes/probes
go build -o out/bin/host.exe ./host
out/bin/host.exe -for 2m          # or no -for: until Ctrl+C
duckdb -c ".read aggregate.sql"   # per metric, then per build
```

Flags: `-probes` (default `probes`), `-out` (`out`), `-poll` (`500ms`),
`-interval` (`1s`), `-for` (`0`, until interrupted), `-rotate-bytes` (64 MiB:
each JSONL file is renamed `<name>-<UTC time>.jsonl` at that size, `0` never)
and `-keep` (`4` renamed files of each). A file named `stop` in the output
directory stops the host and its probes, as Ctrl+C does.

To run it in the background as the 3D scene's board expects, writing to
`%LOCALAPPDATA%\ix-probes` at a 5 s interval with the request watcher beside it:

```powershell
pwsh -File start.ps1   # builds the host if needed; does nothing for what already runs
pwsh -File stop.ps1    # through the stop file, then the watcher
```

`out/probes.jsonl` is plain JSONL with typed timestamps, so any DuckDB query
works on it, for example the CPU load in 10-second buckets:

```sql
SELECT time_bucket(INTERVAL 10 SECOND, ts) AS t, build, round(avg(value), 1) AS cpu
FROM read_json_auto('out/probes.jsonl') WHERE metric = 'cpu_load_pct'
GROUP BY ALL ORDER BY t;
```

## The first probe: `sysinfo`

WMI, through `github.com/yusufpapurcu/wmi` (MIT): `Win32_OperatingSystem` for
`ram_total_mb` and `ram_free_mb`, `Win32_Processor` for `cpu_logical` and
`cpu_load_pct` (the mean `LoadPercentage` over processors, left out while WMI
has no sample yet).

## Checked, and measured

`check_hot_reload.py` runs the whole loop and checks it. It starts a host of
its own on a copy of `sysinfo` under `out/check-hr`, compares a sample with
`Get-CimInstance`, adds a metric to the copy's source, then breaks it, stops
the host with Ctrl+Break and runs `aggregate.sql`. A host already running on
`probes/` is left alone. Rerun on 2026-10-18 with the gap-free swap: PASS, the
longest stretch without a sample 1.38 s at a 1 s interval.

```powershell
python -B check_hot_reload.py     # from prototypes/probes, after building the host
```

On 2026-10-07, on a 24-thread Windows 11 machine whose CPU load was 90–100%
throughout (other work running), it passed:

| | What | Measured |
| --- | --- | --- |
| S1 | first sample after the probe starts, ≤ 5 s | 1.3 s; plus the first build, 12.8 s with a warm Go cache |
| S2 | RAM total and logical CPUs equal `Get-CimInstance`; free RAM within 10% | equal; free RAM within 1% |
| S3 | a source edit is picked up, the host keeps running | the new metric 15–19 s after the edit; one `host_started` |
| S4 | a source that does not build keeps the old build running | `build_failed` with `kept_running` set, samples continue from the old build |
| S5 | DuckDB aggregates the output | both queries in `aggregate.sql` run; `ts` reads as `timestamp` |

Also seen:

- **An edit takes 15–20 s to show up, nearly all of it `go build`** (8–14 s
  under that load). Starting a new executable takes another 3–5 s, which looks
  like the antivirus scanning it; that is why the new build starts before the
  old one stops.
- **The swap left a 2–3.5 s gap** in the samples: the time between the new
  process starting and its first WMI sample. Closed on 2026-10-18 (see below).
- The sampling period was 1.2–1.3 s at a 1 s interval, the WMI queries being
  slow under load.

`check_host.py` checks what came after, on a throwaway probe of its own under
`out/check-probes`, so a host already running on `probes/` is left alone:

```powershell
python -B check_host.py [host.exe]   # out/bin/host.exe by default
```

On 2026-10-18 it passed:

| | What | Measured |
| --- | --- | --- |
| S6 | the JSONL is renamed at `-rotate-bytes`, only `-keep` renamed files remain | 2 kept at keep 2, each 20 150 bytes for a 20 000-byte limit |
| S7 | a swap leaves no gap over 1 s; the old build stops after the new one's first sample | longest gap 51 ms at a 50 ms interval; stopped 50 ms after |
| S8 | the stop file stops the host and its probes within 5 s | 0.1 s, no probe left running |

A host that stops the old build at once turns S7 red: the old build stopped
before the new one's first sample. The gap alone would not show it with this
probe, which samples at once; the 2–3.5 s gap came from WMI's first query.

## Probes from templates, and requests

`new_probe.py` writes a probe from a fixed template, never from free code. Its
only template today is `process`: the CPU, memory and count of the processes
whose WMI name matches a `LIKE` pattern. Every parameter must match a strict
pattern before it reaches the source:

```powershell
python -B new_probe.py process scene "godot.windows.editor%"   # probes/proc-scene
```

`probe_requests.py --every 2` lets the 3D scene, a voice command or an agent
ask for one without running anything. A request is a JSON file in
`%LOCALAPPDATA%\ix-probes\requests\`, written whole (a `.part` renamed into
place) and named by its id:

```json
{"template": "process", "name": "obs", "match": "obs64%"}
```

The watcher hands it to `new_probe.py` and writes `<id>.result.json` beside it,
`{"ok": true, "probe": "proc-obs"}` or `{"ok": false, "error": "..."}`. A probe
that exists is reported, not rewritten, and there are at most 8 template
probes. On 2026-10-18 it refused `obs; rm` as a name, `x' OR 1=1` as a pattern
and a `shell` template. The `obs` request's first sample arrived about 20 s
later.

## Limits, and what next

- Windows only as written: `sysinfo` reads WMI, and the host has only run
  there. A WSL host would need its probes to read `/proc` instead.
- The output is bounded by rotation, at most 5 files of 64 MiB each for the
  samples; there is no retention by age.
- A probe is trusted code that the host builds and runs. Only the template
  probes can be asked for from outside, and they take no code.
- Nothing restarts the host after a reboot; `start.ps1` has to be run.

Next, in the order that seems useful: route the records into the IX DuckDB
bench (`ix-duck`) so they can meet the IX UDFs; start at logon; a WSL host.
