"""End-to-end check of the probe prototype, run from prototypes/probes after
`go build -o out/bin/host.exe ./host`, with `go` on PATH.

It runs the host and checks five things:
  S1  the probe's first sample arrives within 5 s of the probe starting;
  S2  RAM total and logical CPUs match Get-CimInstance exactly, free RAM within 10%;
  S3  an edit to the probe's source is picked up without restarting the host,
      and samples keep coming while the new build replaces the old one;
  S4  a probe that no longer compiles keeps its last good build running;
  S5  DuckDB aggregates the JSONL.
It stops the host with Ctrl+Break, and restores the probe's source whatever happens.
"""
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
PROBE = os.path.join(ROOT, "probes", "sysinfo", "main.go")
BACKSTOP = 180  # the host's own -for, in case the check dies before stopping it
WAIT = 60  # for one build, under load


def lines(name):
    """The complete lines of out/<name>: a line the host is still writing is left out."""
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f.read().split("\n")[:-1] if l.strip()]


def at(rec):
    return datetime.fromisoformat(rec["ts"])


def wait_for(pred, timeout, what):
    end = time.time() + timeout
    while time.time() < end:
        got = pred()
        if got:
            return got
        time.sleep(0.25)
    raise SystemExit(f"FAIL: timed out after {timeout}s waiting for {what}")


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def cim(query):
    ps = ["powershell", "-NoProfile", "-Command", query + " | ConvertTo-Json -Compress"]
    return json.loads(subprocess.run(ps, capture_output=True, text=True, check=True).stdout)


def samples(metric, build=None, after=None):
    return [r for r in lines("probes.jsonl") if r["metric"] == metric
            and (build is None or r["build"] == build) and (after is None or at(r) > after)]


def events(kind):
    return [e for e in lines("host-events.jsonl") if e["event"] == kind]


def main():
    for name in ("probes.jsonl", "host-events.jsonl"):
        p = os.path.join(OUT, name)
        if os.path.exists(p):
            os.remove(p)
    with open(PROBE, encoding="utf-8") as f:
        original = f.read()
    marker = '\t\t{ts, "cpu_logical", float64(logical), "count"},\n'
    assert original.count(marker) == 1, "probe source changed shape; update the check"
    edited = original.replace(marker, marker + (
        '\t\t{ts, "ram_used_pct", 100 * float64(system[0].TotalVisibleMemorySize-system[0].FreePhysicalMemory) /'
        ' float64(system[0].TotalVisibleMemorySize), "%"},\n'))

    host = subprocess.Popen([os.path.join(OUT, "bin", "host.exe"), "-for", f"{BACKSTOP}s"], cwd=ROOT,
                            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    t0 = time.time()
    try:
        # S1
        first = wait_for(lambda: samples("ram_total_mb"), WAIT, "a first sample")
        started = events("started")[0]
        s1 = (at(first[0]) - at(started)).total_seconds()
        built = events("built")[0]
        print(f"S1 first sample {s1:.1f}s after the probe started; {time.time() - t0:.1f}s after the host "
              f"started, of which the first build took {built['ms'] / 1000:.1f}s")

        # S2
        os_ = cim("Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory")
        cpu = cim("Get-CimInstance Win32_Processor | Select-Object NumberOfLogicalProcessors")
        cpu = cpu if isinstance(cpu, list) else [cpu]
        last = {r["metric"]: r["value"] for r in lines("probes.jsonl")}
        total_ok = abs(last["ram_total_mb"] - os_["TotalVisibleMemorySize"] / 1024) < 1e-6
        cpus_ok = last["cpu_logical"] == sum(c["NumberOfLogicalProcessors"] for c in cpu)
        free_ref = os_["FreePhysicalMemory"] / 1024
        free_ok = abs(last["ram_free_mb"] - free_ref) <= 0.10 * free_ref
        print(f"S2 total {last['ram_total_mb']:.1f} MB match={total_ok}; logical {last['cpu_logical']:.0f} "
              f"match={cpus_ok}; free {last['ram_free_mb']:.0f} vs {free_ref:.0f} MB within10%={free_ok}; "
              f"load {last.get('cpu_load_pct', 'n/a')}%")

        # S3
        first_build = first[0]["build"]
        t_edit = time.time()
        write_atomic(PROBE, edited)
        new = wait_for(lambda: samples("ram_used_pct"), WAIT, "the edited probe's new metric")
        new_build = new[0]["build"]
        print(f"S3 new metric ram_used_pct={new[0]['value']:.1f}% {time.time() - t_edit:.1f}s after the edit; "
              f"build {first_build} -> {new_build}")

        # S4
        t_break = time.time()
        write_atomic(PROBE, edited + "\nfunc broken( {\n")
        failed = wait_for(lambda: events("build_failed"), WAIT, "a build_failed event")[0]
        kept = failed.get("kept_running")
        after = wait_for(lambda: len(samples("ram_total_mb", after=at(failed))) >= 2 and
                         samples("ram_total_mb", after=at(failed)), 15, "samples after the failed build")
        still = all(r["build"] == new_build for r in after)
        print(f"S4 build_failed {time.time() - t_break:.1f}s after the break, kept_running={kept}; "
              f"{len(after)} samples since, all from {new_build}: {still}")

        # Back to the running build's source: same hash, so nothing rebuilds.
        write_atomic(PROBE, edited)
        time.sleep(2)
    finally:
        host.send_signal(signal.CTRL_BREAK_EVENT)
        try:
            host.wait(timeout=30)
        except subprocess.TimeoutExpired:
            host.kill()
            host.wait()
        write_atomic(PROBE, original)

    ev = lines("host-events.jsonl")
    starts = [e for e in ev if e["event"] == "host_started"]
    rebuilt = [e for e in ev if e["event"] == "built" and e["build"] == new_build]
    leftovers = subprocess.run(["tasklist", "/FI", "IMAGENAME eq sysinfo-*"], capture_output=True,
                               text=True).stdout
    print(f"host started {len(starts)} time(s), stopped={bool(events('host_stopped'))}, exit code "
          f"{host.returncode}; probe processes left: {'none' if 'sysinfo-' not in leftovers else leftovers}")
    print("events:", ", ".join(e["event"] + (f"[{e['build']}]" if e.get("build") else "") for e in ev))

    # The longest stretch without a sample while the host ran, swap included.
    ticks = sorted(at(r) for r in lines("probes.jsonl") if r["metric"] == "ram_total_mb")
    gaps = [(b - a).total_seconds() for a, b in zip(ticks, ticks[1:])]
    print(f"{len(ticks)} sampling rounds; median gap {sorted(gaps)[len(gaps) // 2]:.2f}s, longest {max(gaps):.2f}s")

    # S5
    sql = subprocess.run(["duckdb", "-c", ".read aggregate.sql"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8")
    print("S5 duckdb exit", sql.returncode)
    print(sql.stdout or sql.stderr)

    ok = (s1 <= 5 and total_ok and cpus_ok and free_ok and new_build != first_build and kept == new_build
          and still and len(rebuilt) == 1 and len(starts) == 1 and host.returncode == 0
          and "sysinfo-" not in leftovers and sql.returncode == 0)
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
