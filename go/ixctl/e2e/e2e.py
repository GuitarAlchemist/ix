"""End-to-end gate for the ixctl control plane: the real Go plane (`go/ixctl`), the real Rust
client (`crates/ix-ctl`, through its `ixql-ctl` example) and ix-stream's `pod` example, as two
federated instances on loopback, plus the `ix` binary run as a job (E10) and runs killed at their
wall, memory, processor-time and process limits (E11..E14), and a probe tree under the plane
(E15: kinds and limits, filter and mux, lease expiry, the host's cpu and mem) and bookmarks
(E16: a replay, seq for seq, of what a probe tree's pipes and the ctl lifecycle carried after
a point the operator named; bounds, refusals, a peer's bookmark; survival of a restart) and the
IXQL probe surface (E17: `probe.ixql` through the Rust client, a CPU probe under a mux, ten
records, the tree removed) and probe budgets (E18: `limits` say the plane's budget, a leaf
past its mux's `max_rate_per_s` is refused naming both, a leaf carries no budget, a mux cannot
widen, the list says rates and bytes). 44 checks, E1..E18, a restart loop and the M1 and M2
measurements.

    python go/ixctl/e2e/e2e.py --build            # build the four binaries, then run the gate
    python go/ixctl/e2e/e2e.py --bin <dir>        # use ixctl, pod, ixql-ctl and ix already built there
    python go/ixctl/e2e/e2e.py --build --work D   # keep binaries, state, runs and results.json in D

Paths are relative to this file: the repo root is three levels up, `pods.ixql` and `job.ixql`
come from `crates/ix-ctl/examples`, and the job's `ix pipeline run` reads the Demerzel
constitution from `governance/demerzel` (the submodule; `IX_GOVERNANCE_DIR` overrides it).
Everything the gate writes goes under the work folder (by default `ixctl-e2e` in the system temp
folder): `bin/`, `state/` (the planes' state files and journals), `runs/` (each IXQL program
with its stdout and stderr, each plane's output), `job/` (the job's `ix.yaml`, its `ix.lock` and
provenance) and `results.json`.
Tokens are read from the state files and kept in memory only: nothing here prints one, and the
last check scans every file written for them.

Windows only for now: `alive` asks the kernel, and the plane's process code is Windows-first.
"""

import argparse
import ctypes
import http.client
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
PODS_IXQL = REPO / "crates" / "ix-ctl" / "examples" / "pods.ixql"
JOB_IXQL = REPO / "crates" / "ix-ctl" / "examples" / "job.ixql"
PROBE_IXQL = REPO / "crates" / "ix-ctl" / "examples" / "probe.ixql"
# Absolute: the job runs from WORK/job, so a relative override would resolve against that.
GOV = pathlib.Path(os.environ.get("IX_GOVERNANCE_DIR") or REPO / "governance" / "demerzel").resolve()
EXE = ".exe" if os.name == "nt" else ""

# The pipeline the job runs: what `ix pipeline new` scaffolds, one `stats` stage.
JOB_PIPELINE = """version: '1'
params: {}
stages:
  load:
    skill: stats
    args:
      data:
      - 1.0
      - 2.0
      - 3.0
      - 4.0
      - 5.0
    deps: []
    cache: null
"""

# Set by main() from the arguments.
WORK = BIN = STATE = RUNS = JOBDIR = None
IXCTL = POD = IXQLCTL = IX = None

results = {"checks": {}, "measures": {}, "notes": []}


def check(key, ok, **evidence):
    results["checks"][key] = {"ok": bool(ok), **evidence}
    print(f"[{'PASS' if ok else 'FAIL'}] {key}: {json.dumps(evidence, default=str)[:400]}")
    return ok


def measure(key, value):
    results["measures"][key] = value


# --- processes -------------------------------------------------------------------------------

if os.name == "nt":
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]


def alive(pid):
    """True while `pid` runs. Read-only: it opens the process for query only."""
    if not pid:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        try:
            # A zombie (exited, not yet reaped by a PID 1 that does not reap) is not alive.
            stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
            return stat[stat.rindex(")") + 2] != "Z"
        except (OSError, ValueError, IndexError):
            return True
    h = k32.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def wait_until(pred, timeout, step=0.01):
    t0 = time.perf_counter()
    while True:
        v = pred()
        if v:
            return v, (time.perf_counter() - t0) * 1000
        if time.perf_counter() - t0 > timeout:
            return v, None
        time.sleep(step)


# --- building ---------------------------------------------------------------------------------


def build():
    """Build the plane, the pod, the IXQL client and the `ix` CLI from this tree into BIN."""
    t0 = time.perf_counter()
    subprocess.run(["go", "build", "-o", str(BIN / ("ixctl" + EXE)), "."],
                   cwd=str(REPO / "go" / "ixctl"), check=True)
    env = {**os.environ, "CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_INCREMENTAL": "0"}
    r = subprocess.run(["cargo", "build", "-q", "-p", "ix-stream", "--example", "pod",
                        "-p", "ix-ctl", "--example", "ixql-ctl", "-p", "ix-skill", "--bin", "ix",
                        "--message-format=json"],
                       cwd=str(REPO), env=env, check=True, capture_output=True)
    copied = []
    for line in r.stdout.decode("utf-8", "replace").splitlines():
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get("reason") == "compiler-artifact" and m.get("executable"):
            shutil.copy2(m["executable"], BIN)
            copied.append(pathlib.Path(m["executable"]).name)
    measure("build_ms", round((time.perf_counter() - t0) * 1000))
    print(f"built ixctl{EXE} and copied {sorted(copied)} into {BIN}")


# --- the plane over HTTP (no redirects: http.client never follows one) ------------------------


def call(st, method, path, body=None, timeout=15):
    host, port = st["addr"].rsplit(":", 1)
    c = http.client.HTTPConnection(host.strip("[]"), int(port), timeout=timeout)
    headers = {"Authorization": "Bearer " + st["token"], "Host": st["addr"]}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=headers)
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, json.loads(raw) if raw else None
    except ValueError:
        return r.status, {"raw": raw.decode("utf-8", "replace")[:300]}


def status(st):
    code, s = call(st, "GET", "/v1/status")
    assert code == 200, (code, s)
    return s


def svc(s, name):
    return next(x for x in s["services"] if x["name"] == name)


def journal_lines(name):
    """A plane's journal, parsed line by line, in order."""
    text = (STATE / f"{name}.journal.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line]


def journal_applies(name):
    """The apply records of a plane's journal, in order."""
    return [m for m in journal_lines(name) if m.get("kind") == "apply"]


def peer(s, name):
    return next((x for x in s.get("peers", []) if x["name"] == name), None)


SSE_GAPS = []  # every gap event a sse() reader got, in order: {ch, reason, from, ...}


def sse(st, ch, seconds, query=None, ready=None):
    """The envelopes /v1/events gives for `ch` (prefix) from the start, read for `seconds`.
    `query` replaces the whole query string (`ch=...&bookmark=...` for a replay from a bookmark,
    `ch=...` alone for a live reader). `ready`, a threading.Event, is set once the stream's
    headers are in: from then on the reader is on the pipe and nothing published is missed.
    Gap events (a ring that no longer holds seq 1, a reader that fell behind) are not
    envelopes: they go to SSE_GAPS, with the prefix asked, and the envelopes come back alone."""
    host, port = st["addr"].rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=seconds)
    req = (
        f"GET /v1/events?{query or f'ch={ch}&since=0'} HTTP/1.1\r\nHost: {st['addr']}\r\n"
        f"Authorization: Bearer {st['token']}\r\nAccept: text/event-stream\r\n\r\n"
    )
    sock.sendall(req.encode())
    buf = b""
    end = time.perf_counter() + seconds
    sock.settimeout(0.2)
    while time.perf_counter() < end:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            continue
        if not chunk:
            break
        buf += chunk
        if ready is not None and not ready.is_set() and b"\r\n\r\n" in buf:
            ready.set()
    sock.close()
    if ready is not None:
        ready.set()
    text = buf.decode("utf-8", "replace")
    head, _, body = text.partition("\r\n\r\n")
    envs = []
    if not head.startswith("HTTP/1.1 200"):
        # A refused stream (a bookmark nobody set, a bookmark beside since) is one JSON error.
        try:
            SSE_GAPS.append({"ch": ch, "reason": "refused", **json.loads(body)})
        except ValueError:
            SSE_GAPS.append({"ch": ch, "reason": "refused", "raw": body[:200]})
        return head.split("\r\n")[0], envs
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                rec = json.loads(line[6:])
            except ValueError:
                continue
            if "ch" in rec:
                envs.append(rec)
            else:
                SSE_GAPS.append({"ch": ch, **rec})
    return head.split("\r\n")[0], envs


def sse_async(st, ch, seconds, query=None, ready=None):
    """sse() on a thread: the reader is on the pipe before what it must see is published.
    Returns the thread and a dict whose `r` is sse()'s result once the thread is joined."""
    out = {}

    def go():
        out["r"] = sse(st, ch, seconds, query=query, ready=ready)

    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th, out


def proc_cpu_ms(pid):
    """Processor time a process has used so far, user plus kernel, in ms; None when unreadable."""
    try:
        if os.name == "nt":
            k32.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
            h = k32.OpenProcess(0x1000, 0, pid)
            if not h:
                return None
            c, e, k, u = (ctypes.c_ulonglong() for _ in range(4))
            ok = k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u))
            k32.CloseHandle(h)
            return (k.value + u.value) / 10000 if ok else None
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
        f = stat[stat.rindex(")") + 2:].split()
        return (int(f[11]) + int(f[12])) * 1000 / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


# --- running things ---------------------------------------------------------------------------

LIVE = []  # every plane started by up(); reap() kills what a failed run leaves behind


def reap():
    """Kill the processes still running (a plane's pods die with it) and drop the state files,
    which hold the bearer tokens, and any temporary copy of one. A run that ends in finish()
    leaves nothing for this to do. One process that will not die does not stop the rest."""
    for p in LIVE:
        try:
            if p.poll() is None:
                p.kill()
                p.wait(5)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"reap: pid {p.pid}: {e}", file=sys.stderr)
    if STATE is not None:
        for pattern in ("*.json", "*.json.tmp"):
            for f in STATE.glob(pattern):
                try:
                    f.unlink()
                except OSError as e:
                    print(f"reap: {f.name}: {e}", file=sys.stderr)


def up(name, spec, *extra):
    out = open(RUNS / f"plane-{name}.stdout", "wb", buffering=0)
    err = open(RUNS / f"plane-{name}.stderr", "wb", buffering=0)
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [str(IXCTL), "up", str(spec), "--name", name, "--state", str(STATE), *extra],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, cwd=str(WORK))
    LIVE.append(p)
    line = p.stdout.readline()
    parent_ms = (time.perf_counter() - t0) * 1000
    out.write(line)
    threading.Thread(target=shutil.copyfileobj, args=(p.stdout, out), daemon=True).start()
    ready = json.loads(line)
    st = json.loads((STATE / f"{name}.json").read_text())
    return p, ready, st, parent_ms


def ixql(tag, program, state_file, timeout=60):
    t0 = time.perf_counter()
    r = subprocess.run([str(IXQLCTL), str(program), "--state", str(state_file)],
                       capture_output=True, cwd=str(WORK), timeout=timeout)
    ms = (time.perf_counter() - t0) * 1000
    (RUNS / f"ixql-{tag}.stdout").write_bytes(r.stdout)
    (RUNS / f"ixql-{tag}.stderr").write_bytes(r.stderr)
    out = None
    if r.returncode == 0:
        out = json.loads(r.stdout)
    return r.returncode, out, r.stderr.decode("utf-8", "replace").strip(), ms


def write_ixql(name, text):
    path = RUNS / name
    path.write_text(text, encoding="utf-8")
    return path


def spec(name, services, peers):
    path = WORK / f"spec-{name}.json"
    path.write_text(json.dumps({"name": name, "services": services, "peers": peers}, indent=2))
    return path


def pod(service, label, nodes, rate=20, extra=()):
    return {"name": service, "command": str(POD),
            "args": ["--name", label, "--nodes", str(nodes), "--rate", str(rate), *extra],
            "restart": "on-failure"}


def job(service, pipeline):
    """A job: `ix pipeline run` on `pipeline`, run once (`restart: never`) in JOBDIR, where its
    `ix.lock` and provenance land. The constitution gate reads GOV. Bounded, as every job under
    the factory is meant to be: two minutes, two GiB."""
    return {"name": service, "command": str(IX),
            "args": ["pipeline", "run", "--file", str(pipeline), "--json"],
            "dir": str(JOBDIR), "env": {"IX_GOVERNANCE_DIR": str(GOV), "IX_ROOT": str(REPO)},
            "restart": "never", "limits": {"wall_ms": 120_000, "memory_mb": 2048, "cpu_ms": 60_000, "processes": 16}}


def main():
    try:
        return run()
    finally:
        reap()


def run():
    global WORK, BIN, STATE, RUNS, JOBDIR, IXCTL, POD, IXQLCTL, IX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build ixctl, pod and ixql-ctl from this tree first")
    ap.add_argument("--bin", type=pathlib.Path, help="folder holding prebuilt ixctl, pod and ixql-ctl (default: <work>/bin)")
    ap.add_argument("--work", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()) / "ixctl-e2e",
                    help="where state, runs, results.json and built binaries go")
    args = ap.parse_args()

    WORK = args.work.resolve()
    BIN = (args.bin or WORK / "bin").resolve()
    STATE, RUNS, JOBDIR = WORK / "state", WORK / "runs", WORK / "job"
    IXCTL, POD, IXQLCTL = BIN / ("ixctl" + EXE), BIN / ("pod" + EXE), BIN / ("ixql-ctl" + EXE)
    IX = BIN / ("ix" + EXE)

    for d in (STATE, RUNS, JOBDIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    if args.build:
        BIN.mkdir(parents=True, exist_ok=True)
        build()
    for exe in (IXCTL, POD, IXQLCTL, IX):
        assert exe.exists(), f"missing {exe} (build it, or pass --build)"
    for f in (PODS_IXQL, JOB_IXQL):
        assert f.exists(), f"missing {f}"
    constitution = GOV / "constitutions" / "default.constitution.md"
    assert constitution.exists(), (f"missing {constitution}: init the governance/demerzel "
                                   "submodule, or point IX_GOVERNANCE_DIR at a checkout")
    (JOBDIR / "ix.yaml").write_text(JOB_PIPELINE, encoding="utf-8")

    spec_b = spec("b", [pod("pod-a", "b/pod-a", 3)],
                  [{"name": "a", "state_file": str(STATE / "a.json")}])
    spec_a = spec("a", [
        pod("pod-a", "a/pod-a", 4),
        pod("pod-b", "a/pod-b", 4),
        {**pod("crashy", "a/crashy", 2, 5, ["--crash-after", "0.15"]), "max_restarts": 2},
        {"name": "missing", "command": str(BIN / ("does-not-exist" + EXE)), "restart": "never"},
        job("job", JOBDIR / "ix.yaml"),
        job("job-bad", JOBDIR / "does-not-exist.yaml"),
        # A job that would run for a minute, bounded to 1.5 s (long enough for E11 to read its
        # pid from the status on a loaded runner before the kill).
        {"name": "sleepy", "command": sys.executable, "args": ["-c", "import time; time.sleep(60)"],
         "restart": "never", "limits": {"wall_ms": 1500}},
        # A job that would hold 256 MiB (written, so resident; taken 4 MiB at a time, so the
        # tree goes past the limit rather than having one commit refused) for a minute,
        # bounded to 64 MiB.
        {"name": "hog", "command": sys.executable,
         "args": ["-c", "c = [b'x' * (4 << 20) for _ in range(64)]; import time; time.sleep(60)"],
         "restart": "never", "limits": {"memory_mb": 64}},
        # A job that would spin for a minute, bounded to 300 ms of processor time.
        {"name": "spinner", "command": sys.executable,
         "args": ["-c", "import time; end = time.monotonic() + 60\nwhile time.monotonic() < end: pass"],
         "restart": "never", "limits": {"cpu_ms": 300}},
        # A job that would hold four sleeping children (five processes) for a minute, bounded to
        # three processes. A spawn the Windows cap refuses is kept going with fewer: the count
        # stays past the limit for the watch to see.
        {"name": "spawner", "command": sys.executable,
         "args": ["-c", "import subprocess, sys, time\n"
                  "ps = []\n"
                  "for _ in range(4):\n"
                  "    try: ps.append(subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']))\n"
                  "    except OSError as e: print('spawn refused:', e, file=sys.stderr); break\n"
                  "time.sleep(60)"],
         "restart": "never", "limits": {"processes": 3}},
    ], [{"name": "b", "state_file": str(STATE / "b.json")}])

    pb, ready_b, sb, ms_b = up("b", spec_b)
    # Plane a carries the operator's probe budget (D10): M2 declares exactly 1 600 records/s under
    # one mux, the compiled default being 1 000; E18 reads these values back from `limits`.
    pa, ready_a, sa, ms_a = up("a", spec_a, "--probe-rate", "1600", "--probe-tree-rate", "3200")
    tokens = [sa["token"], sb["token"]]
    measure("ready_ms", {"a": {"parent": round(ms_a, 1), "internal": ready_a.get("ms_to_ready")},
                         "b": {"parent": round(ms_b, 1), "internal": ready_b.get("ms_to_ready")}})
    check("E3.distinct_instances",
          sa["addr"] != sb["addr"] and sa["token"] != sb["token"] and sa["name"] == "a"
          and sb["name"] == "b" and pa.pid != pb.pid and status(sa)["name"] == "a"
          and status(sb)["name"] == "b",
          a_addr=sa["addr"], b_addr=sb["addr"], tokens_differ=sa["token"] != sb["token"])
    _, ms = wait_until(lambda: (peer(status(sa), "b") or {}).get("state") == "up"
                       and (peer(status(sb), "a") or {}).get("state") == "up", 10)
    check("E3.peers_up", ms is not None, ms=ms, a_sees_b=peer(status(sa), "b"),
          b_sees_a=(peer(status(sb), "a") or {}).get("state"))

    # E1: pods.ixql through the Rust client, end to end.
    code, out, err, ms = ixql("pods", PODS_IXQL, STATE / "a.json")
    measure("pods_ixql_ms", round(ms, 1))
    if not check("E1.pods_ixql_runs", code == 0, code=code, stderr=err[:300]):
        return finish(pa, pb, sa, sb, tokens)
    g0 = out["planned"]["generation"]
    starts = [{"op": "start", "service": "pod-a"}, {"op": "start", "service": "pod-b"}]
    check("E1.plan_is_the_diff", out["planned"]["actions"] == starts, planned=out["planned"])
    check("E1.apply_ack_generation",
          out["applied"]["generation"] == g0 + 1 and out["applied"]["actions"] == starts,
          applied=out["applied"])
    # The document the Rust client sent reached the journal with its schema field: a client
    # that dropped it would still be accepted (no schema reads as this one), so ask the record.
    # The stops that pods.ixql goes on to make are the plane's own documents, without one.
    applies = journal_applies("a")
    check("E1.schema_journaled",
          applies and applies[0]["generation"] == out["applied"]["generation"]
          and applies[0]["desired"].get("schema") == "ixctl-desired/0",
          first_apply={k: applies[0].get(k) for k in ("generation", "desired")} if applies else None)
    check("E1.converged_observed",
          out["converged"]["observed_generation"] >= out["applied"]["generation"],
          observed=out["converged"]["observed_generation"])
    st_services = {s["name"]: s for s in out["status"]["services"]}
    pa_pid, pb_pid = st_services["pod-a"]["pid"], st_services["pod-b"]["pid"]
    check("E1.pod_identities",
          out["pod_a"]["state"] == "running" and len(out["pod_a"]["nodes"]) == 4
          and st_services["pod-b"]["state"] == "running" and pa_pid and pb_pid and pa_pid != pb_pid,
          pod_a_pid=pa_pid, pod_b_pid=pb_pid,
          pod_a_nodes=[n["name"] for n in out["pod_a"]["nodes"]])
    check("E1.replan_empty", out["replanned"]["actions"] == []
          and sorted(out["replanned"]["noop"]) == ["pod-a", "pod-b"], replanned=out["replanned"])
    check("E5.stopped_trees_gone",
          out["stopped_a"]["stopped"] == "pod-a" and out["stopped_b"]["stopped"] == "pod-b"
          and not alive(pa_pid) and not alive(pb_pid),
          stopped_a=out["stopped_a"], stopped_b=out["stopped_b"])

    # E7: re-applying a document already in force starts nothing (a state property only).
    doc = {"services": {"pod-a": {"state": "running"}}}
    _, r1 = call(sa, "POST", "/v1/apply", doc)
    _, ms = wait_until(lambda: status(sa)["observed_generation"] >= r1["generation"], 10)
    _, ms_nodes = wait_until(lambda: len(svc(status(sa), "pod-a")["nodes"] or []) == 4, 10)
    pid1 = svc(status(sa), "pod-a")["pid"]
    _, r2 = call(sa, "POST", "/v1/apply", doc)
    wait_until(lambda: status(sa)["observed_generation"] >= r2["generation"], 10)
    pid2 = svc(status(sa), "pod-a")["pid"]
    check("E7.reapply_starts_nothing",
          r2["generation"] == r1["generation"] + 1 and r2["actions"] == [] and pid1 == pid2
          and alive(pid1), first=r1, second=r2, pid=pid1)
    measure("apply_to_converged_ms", ms)

    # E5: a stop answers once the tree is gone; the pod saw stdin close and exited 0.
    t0 = time.perf_counter()
    code, stop = call(sa, "POST", "/v1/stop", {"service": "pod-a"})
    stop_ms = (time.perf_counter() - t0) * 1000
    measure("stop_ms", round(stop_ms, 1))
    check("E5.stop_bounded", code == 200 and not alive(pid1) and stop_ms < 2500
          and (stop.get("exit") or {}).get("code") == 0, ms=round(stop_ms, 1), reply=stop)

    # E2: a command that does not exist ends the apply diverged, and the Rust wait says so.
    prog = write_ixql("diverge.ixql",
                      'desired <- { services: { "missing": { state: "running" } } }\n'
                      "applied <- desired → ix.ctl.apply\n"
                      "done <- ix.ctl.wait(generation: applied.generation, timeout_ms: 10000)\n")
    code, out2, err, ms = ixql("diverge", prog, STATE / "a.json")
    s = status(sa)
    div = s.get("diverged") or {}
    check("E2.rust_wait_diverged_promptly",
          code == 1 and "diverged on missing" in err and ms < 3000, code=code, ms=round(ms, 1),
          stderr=err[:300])
    check("E2.status_diverged",
          div.get("service") == "missing" and div.get("generation", 0) > s["observed_generation"],
          diverged=div, generation=s["generation"], observed=s["observed_generation"],
          missing_state=svc(s, "missing")["state"])
    _, ctl = sse(sa, "ctl", 1.0)
    kinds = [e["ev"]["kind"] for e in ctl]
    check("E2.ctl_says_diverged", "diverged" in kinds and "service.failed" in kinds,
          kinds_tail=kinds[-8:])
    # E9: the Rust client reads the same stream (chunked SSE, keep-alive comment) and stops at
    # `max:`; what it read is what the bare socket read, seq for seq.
    n = min(len(ctl), 6)
    prog = write_ixql("events.ixql",
                      f'seen <- ix.ctl.events(channel: "ctl", since: 0, max: {n}, timeout_ms: 2000)\n')
    code, out_ev, err, ms = ixql("events", prog, STATE / "a.json")
    rust_seen = [(e["seq"], e["kind"]) for e in ((out_ev or {}).get("seen") or {}).get("events", [])]
    check("E9.rust_reads_the_event_stream",
          code == 0 and rust_seen == [(e["seq"], e["ev"]["kind"]) for e in ctl[:n]]
          and (out_ev["seen"]["ended"] is False) and out_ev["seen"]["gap"] is None,
          code=code, ms=round(ms, 1), n=n, rust_seen=rust_seen[:4], stderr=err[-160:])

    # M1: what a wait built on the stream would gain over the polling one. A live reader on
    # `ctl` (no `since`) and a 25 ms status poll (the Rust client's POLL) watch the same apply,
    # a real start (pod-a was stopped at E5); each reports when it saw the generation settle,
    # measured from the apply's return. The pod is stopped again afterwards.
    seen_converged = {}
    reader_done = threading.Event()

    def live_reader():
        host, port = sa["addr"].rsplit(":", 1)
        sock = socket.create_connection((host, int(port)), timeout=5)
        sock.sendall((f"GET /v1/events?ch=ctl HTTP/1.1\r\nHost: {sa['addr']}\r\n"
                      f"Authorization: Bearer {sa['token']}\r\nAccept: text/event-stream\r\n\r\n")
                     .encode())
        sock.settimeout(0.05)
        buf = b""
        while not reader_done.is_set():
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            now = time.perf_counter()
            for line in buf.decode("utf-8", "replace").splitlines():
                if line.startswith("data: ") and '"converged"' in line:
                    try:
                        e = json.loads(line[6:])
                    except ValueError:
                        continue
                    seen_converged.setdefault(e["ev"]["data"]["generation"], now)
        sock.close()

    threading.Thread(target=live_reader, daemon=True).start()
    time.sleep(0.15)  # the subscription has no reply; give the plane time to register it
    t_apply = time.perf_counter()
    _, r = call(sa, "POST", "/v1/apply", {"services": {"pod-a": {"state": "running"}}})
    t_reply = time.perf_counter()
    g = r["generation"]
    polled = None
    while time.perf_counter() - t_reply < 5:
        if status(sa).get("settled_generation", 0) >= g:
            polled = time.perf_counter()
            break
        time.sleep(0.025)
    wait_until(lambda: g in seen_converged, 5)
    reader_done.set()
    call(sa, "POST", "/v1/stop", {"service": "pod-a"})
    check("M1.stream_and_poll_both_see_the_generation", g in seen_converged and polled is not None,
          generation=g, actions=len(r["actions"]),
          apply_reply_ms=round((t_reply - t_apply) * 1000, 1),
          stream_after_reply_ms=round((seen_converged[g] - t_reply) * 1000, 1) if g in seen_converged else None,
          poll_25ms_after_reply_ms=round((polled - t_reply) * 1000, 1) if polled else None)

    # E10: a job. A service declared with `restart: never` runs to its end: here the real
    # `ix pipeline run --json` (one `stats` stage, through the Demerzel constitution gate) under
    # the plane, driven by job.ixql through the Rust client. The start's generation converging
    # says launched; the exit wait says ended, with the exit code; the NDJSON the job printed
    # reached `svc/job` as `log` events and is read back from the ring.
    code, out, err, ms = ixql("job", JOB_IXQL, STATE / "a.json", timeout=180)
    measure("job_ixql_ms", round(ms, 1))
    exited = (out or {}).get("exited") or {}
    output = (out or {}).get("output") or {}
    printed = []
    for e in output.get("events", []):
        if e.get("kind") == "log":
            try:
                printed.append(json.loads(e["data"]["line"]))
            except (ValueError, KeyError, TypeError):
                pass
    events = [p.get("event") for p in printed if isinstance(p, dict)]
    applied_g = ((out or {}).get("applied") or {}).get("generation")
    check("E10.job_runs_to_its_end",
          code == 0 and exited.get("state") == "stopped" and exited.get("desired") == "running"
          and (exited.get("last_exit") or {}).get("code") == 0 and exited.get("pid") is None
          and applied_g and (exited.get("last_exit") or {}).get("generation") == applied_g
          and exited.get("generation") == applied_g
          and exited.get("restarts") == 0 and "start" in events and events[-1:] == ["done"]
          and output.get("gap") is None and (JOBDIR / "ix.lock").exists()
          # what the run cost, as the OS counted it, with no limit reached. `cpu_ms` is recorded,
          # not required: Windows charges processor time in ~15.6 ms ticks, and a run this short
          # may end inside one; the Go suite burns CPU for that assertion.
          and (exited.get("last_exit") or {}).get("peak_memory_bytes", 0) > 0
          and "reason" not in (exited.get("last_exit") or {}),
          code=code, ms=round(ms, 1), cpu_ms=(exited.get("last_exit") or {}).get("cpu_ms"),
          exited={k: exited.get(k) for k in ("state", "desired", "generation", "last_exit", "restarts")},
          applied_generation=applied_g, printed_events=events, gap=output.get("gap"),
          lock=(JOBDIR / "ix.lock").exists(), stderr=err[-200:])
    # A job that fails (its pipeline file does not exist: `ix` exits 10) is left stopped with
    # that code and is not restarted. Its start converges all the same: converged means launched.
    # The exit wait is scoped to the apply's generation and boot, so no earlier exit can answer.
    _, r = call(sa, "POST", "/v1/apply", {"services": {"job-bad": {"state": "running"}}})
    _, settled_ms = wait_until(lambda: status(sa)["settled_generation"] >= r["generation"], 15)
    prog = write_ixql("job-bad.ixql",
                      f'exited <- ix.ctl.wait(service: "job-bad", exit: true, generation: {r["generation"]}, '
                      f'epoch: "{r["epoch"]}", timeout_ms: 60000)\n')
    code, out, err, ms = ixql("job-bad", prog, STATE / "a.json")
    bad = (out or {}).get("exited") or {}
    failed = [d["generation"] for d in status(sa)["failed"]]
    check("E10.failed_job_stays_stopped",
          code == 0 and settled_ms is not None and r["generation"] not in failed
          and bad.get("state") == "stopped" and (bad.get("last_exit") or {}).get("code") == 10
          and (bad.get("last_exit") or {}).get("generation") == r["generation"]
          and bad.get("restarts") == 0,
          code=code, ms=round(ms, 1), settled_ms=settled_ms, generation=r["generation"],
          failed_generations=failed,
          exited={k: bad.get(k) for k in ("state", "desired", "last_exit", "restarts")},
          stderr=err[-200:])

    # E11: a wall limit. A run that lasts past `limits.wall_ms` (1.5 s here) is killed by the
    # plane, its tree with it, and its exit names the limit (`last_exit.reason`); a job so ended
    # stays stopped, never restarted. The exit wait, scoped to the start's generation, answers it.
    t_apply = time.monotonic()  # before the request: the launch, so the limit, may start in it
    _, r = call(sa, "POST", "/v1/apply", {"services": {"sleepy": {"state": "running"}}})
    sleepy_pid, _ = wait_until(lambda: svc(status(sa), "sleepy").get("pid"), 10)
    prog = write_ixql("sleepy.ixql",
                      f'exited <- ix.ctl.wait(service: "sleepy", exit: true, generation: {r["generation"]}, '
                      f'epoch: "{r["epoch"]}", timeout_ms: 60000)\n')
    code, out, err, ms = ixql("sleepy", prog, STATE / "a.json")
    killed_after_ms = round((time.monotonic() - t_apply) * 1000, 1)
    sl = (out or {}).get("exited") or {}
    le = sl.get("last_exit") or {}
    gone = not alive(sleepy_pid)
    check("E11.wall_limit_kills_the_run",
          code == 0 and sleepy_pid and gone and sl.get("state") == "stopped"
          and sl.get("desired") == "running" and le.get("reason") == "wall_limit"
          and le.get("generation") == r["generation"] and le.get("code") not in (None, 0)
          and sl.get("restarts") == 0 and sl.get("pid") is None
          and 1500 <= killed_after_ms < 15000,  # not before its limit, not long after
          code=code, ms=round(ms, 1), killed_after_ms=killed_after_ms, pid=sleepy_pid,
          process_gone=gone,
          exited={k: sl.get(k) for k in ("state", "desired", "generation", "last_exit", "restarts")},
          stderr=err[-200:])

    # E12: a memory limit. A run that holds more than `limits.memory_mb` is killed (on Windows
    # the job's guaranteed notification says the tree went past it; on Linux the plane reads
    # the group's resident set from /proc), its exit names the limit and the peak the OS
    # counted; the job stays stopped.
    _, r = call(sa, "POST", "/v1/apply", {"services": {"hog": {"state": "running"}}})
    t_apply = time.monotonic()
    prog = write_ixql("hog.ixql",
                      f'exited <- ix.ctl.wait(service: "hog", exit: true, generation: {r["generation"]}, '
                      f'epoch: "{r["epoch"]}", timeout_ms: 60000)\n')
    code, out, err, ms = ixql("hog", prog, STATE / "a.json")
    killed_after_ms = round((time.monotonic() - t_apply) * 1000, 1)
    hg = (out or {}).get("exited") or {}
    le = hg.get("last_exit") or {}
    check("E12.memory_limit_kills_the_run",
          code == 0 and hg.get("state") == "stopped" and hg.get("desired") == "running"
          and le.get("reason") == "memory_limit" and le.get("generation") == r["generation"]
          and le.get("code") not in (None, 0) and (le.get("peak_memory_bytes") or 0) > 0
          and hg.get("restarts") == 0 and hg.get("pid") is None and killed_after_ms < 15000,
          code=code, ms=round(ms, 1), killed_after_ms=killed_after_ms,
          exited={k: hg.get(k) for k in ("state", "desired", "generation", "last_exit", "restarts")},
          stderr=err[-200:])

    # E13: a processor-time limit. A run that uses more than `limits.cpu_ms` is killed (the
    # plane reads every 100 ms the same accounting the exit reports as `cpu_ms`: the job's on
    # Windows, the group's /proc stat on Linux), its exit names the limit and the time used is
    # past it by construction; the job stays stopped.
    _, r = call(sa, "POST", "/v1/apply", {"services": {"spinner": {"state": "running"}}})
    t_apply = time.monotonic()
    prog = write_ixql("spinner.ixql",
                      f'exited <- ix.ctl.wait(service: "spinner", exit: true, generation: {r["generation"]}, '
                      f'epoch: "{r["epoch"]}", timeout_ms: 60000)\n')
    code, out, err, ms = ixql("spinner", prog, STATE / "a.json")
    killed_after_ms = round((time.monotonic() - t_apply) * 1000, 1)
    sp = (out or {}).get("exited") or {}
    le = sp.get("last_exit") or {}
    check("E13.cpu_limit_kills_the_run",
          code == 0 and sp.get("state") == "stopped" and sp.get("desired") == "running"
          and le.get("reason") == "cpu_limit" and le.get("generation") == r["generation"]
          and le.get("code") not in (None, 0) and (le.get("cpu_ms") or 0) > 300
          and sp.get("restarts") == 0 and sp.get("pid") is None and killed_after_ms < 15000,
          code=code, ms=round(ms, 1), killed_after_ms=killed_after_ms,
          exited={k: sp.get(k) for k in ("state", "desired", "generation", "last_exit", "restarts")},
          stderr=err[-200:])

    # E14: a process limit. A run that holds more processes at once than `limits.processes` is
    # killed (the plane counts them every 100 ms; on Windows the job's cap, an eighth above,
    # refuses the spawn past it and the kernel's message, best effort, says so too), its exit
    # names the limit and the most processes it held; the job stays stopped.
    _, r = call(sa, "POST", "/v1/apply", {"services": {"spawner": {"state": "running"}}})
    t_apply = time.monotonic()
    prog = write_ixql("spawner.ixql",
                      f'exited <- ix.ctl.wait(service: "spawner", exit: true, generation: {r["generation"]}, '
                      f'epoch: "{r["epoch"]}", timeout_ms: 60000)\n')
    code, out, err, ms = ixql("spawner", prog, STATE / "a.json")
    killed_after_ms = round((time.monotonic() - t_apply) * 1000, 1)
    sw = (out or {}).get("exited") or {}
    le = sw.get("last_exit") or {}
    check("E14.process_limit_kills_the_run",
          code == 0 and sw.get("state") == "stopped" and sw.get("desired") == "running"
          and le.get("reason") == "process_limit" and le.get("generation") == r["generation"]
          and le.get("code") not in (None, 0) and (le.get("peak_processes") or 0) > 3
          and sw.get("restarts") == 0 and sw.get("pid") is None and killed_after_ms < 15000,
          code=code, ms=round(ms, 1), killed_after_ms=killed_after_ms,
          exited={k: sw.get(k) for k in ("state", "desired", "generation", "last_exit", "restarts")},
          stderr=err[-200:])

    # A real pod that keeps crashing: restarts with backoff, then failed. A pod reports its
    # nodes as soon as it runs, so the crash loop is watched on status, and the Rust wait is
    # asked afterwards: a failed service ends it at once.
    _, r = call(sa, "POST", "/v1/apply", {"services": {"crashy": {"state": "running"}}})
    seen = []

    def crashed_out():
        c = svc(status(sa), "crashy")
        if not seen or seen[-1] != (c["state"], c["restarts"]):
            seen.append((c["state"], c["restarts"]))
        return c["state"] == "failed"

    _, fail_ms = wait_until(crashed_out, 15)
    c = svc(status(sa), "crashy")
    prog = write_ixql("crashy.ixql",
                      'pod <- ix.ctl.wait(service: "crashy", nodes: 2, timeout_ms: 15000)\n')
    code, _, err, ms = ixql("crashy", prog, STATE / "a.json")
    check("restart.backoff_then_failed",
          fail_ms is not None and c["restarts"] == 2 and code == 1 and "failed" in err
          and ms < 3000, apply=r, ms_to_failed=fail_ms, transitions=seen,
          rust_wait_ms=round(ms, 1), stderr=err[:200])

    # E3: an apply forwarded from A lands on B only. The peer is named beside the document.
    a_before = svc(status(sa), "pod-a")
    code, fwd = call(sa, "POST", "/v1/apply?peer=b", doc)
    _, ms = wait_until(lambda: len(svc(status(sb), "pod-a")["nodes"] or []) == 3, 10)
    b_pod = svc(status(sb), "pod-a")
    a_after = svc(status(sa), "pod-a")
    check("E3.forwarded_apply_lands_on_b_only",
          code == 200 and b_pod["state"] == "running" and alive(b_pod["pid"])
          and a_before["state"] == a_after["state"] == "stopped" and a_after["pid"] is None,
          reply=fwd, b_pod_pid=b_pod["pid"], b_nodes=len(b_pod["nodes"] or []), a_pod=a_after["state"],
          ms_to_b_nodes=ms)
    # The forwarded reply carries B's generation and epoch, so the wait for it reads B too
    # (`peer:`); asked of A, the same wait compares B's epoch with A's and reports a restart.
    prog = write_ixql("fwd-wait.ixql",
                      f'done <- ix.ctl.wait(generation: {fwd["generation"]}, epoch: "{fwd["epoch"]}", '
                      'peer: "b", timeout_ms: 10000)\n'
                      'pod <- ix.ctl.wait(service: "pod-a", nodes: 3, peer: "b", timeout_ms: 10000)\n'
                      'bst <- ix.ctl.status(peer: "b")\n')
    code, out4, err, ms = ixql("fwd-wait", prog, STATE / "a.json")
    prog = write_ixql("fwd-wait-local.ixql",
                      f'done <- ix.ctl.wait(generation: {fwd["generation"]}, epoch: "{fwd["epoch"]}", '
                      "timeout_ms: 10000)\n")
    code_local, _, err_local, ms_local = ixql("fwd-wait-local", prog, STATE / "a.json")
    code_http, bst = call(sa, "GET", "/v1/status?peer=b")
    code_empty, empty = call(sa, "GET", "/v1/status?peer=")
    check("E3.forwarded_wait_asks_the_peer",
          code == 0 and out4["done"]["epoch"] == fwd["epoch"] and out4["done"]["name"] == "b"
          and len(out4["pod"]["nodes"]) == 3 and out4["bst"]["name"] == "b"
          and code_local == 1 and "restarted" in err_local and ms_local < 3000
          and code_http == 200 and bst["name"] == "b" and bst["epoch"] == fwd["epoch"]
          and code_empty == 400 and "empty" in (empty or {}).get("error", ""),
          ms=round(ms, 1), stderr=err[-160:], local_wait=(code_local, round(ms_local, 1), err_local[-120:]),
          http_status_peer=(code_http, (bst or {}).get("name")), empty_peer=(code_empty, empty))
    _, mirrored = sse(sa, "peer/b/", 1.5)
    chans = sorted({e["ch"] for e in mirrored})
    metas = [e["ev"]["data"].get("name") for e in mirrored
             if e["ch"] == "peer/b/svc/pod-a" and e["ev"]["kind"] == "meta"]
    check("E3.mirrors_keep_origin",
          mirrored and all(e["origin"] == "b" and e["ch"].startswith("peer/b/") for e in mirrored)
          and not any(ch.startswith("peer/b/peer/") for ch in chans) and "b/pod-a" in metas,
          n=len(mirrored), channels=chans[:8], meta_names=metas)
    _, own = sse(sa, "svc/pod-a", 1.0)
    own_metas = [e["ev"]["data"].get("name") for e in own if e["ev"]["kind"] == "meta"]
    check("E3.own_events_origin_a",
          own and all(e["origin"] == "a" for e in own) and set(own_metas) == {"a/pod-a"},
          n=len(own), meta_names=sorted(set(own_metas)))
    prog = write_ixql("fwd-stop.ixql", 'stopped <- ix.ctl.stop(service: "pod-a", peer: "b")\n')
    code, out3, err, ms = ixql("fwd-stop", prog, STATE / "a.json")
    check("E3.forwarded_stop_from_rust",
          code == 0 and not alive(b_pod["pid"]) and svc(status(sb), "pod-a")["state"] == "stopped",
          code=code, ms=round(ms, 1), reply=(out3 or {}).get("stopped"), stderr=err[:200])

    # E15 and M2 run here, after the E3 reads of the ring from seq 1: M2 publishes thousands of
    # envelopes, after which the ring no longer holds the boot's first events.
    # E15: probes. The plane carries its probe kinds and says its limits; a refusal names the
    # bound it hit; a tree (a mux parent, a heartbeat child with a filter) puts the child's
    # filtered samples on the parent's pipe stamped with their source; the real cpu and mem kinds
    # read the host within range; a lease nobody touches expires and takes its node out of the
    # tree, said on ctl.
    code_k, kinds = call(sa, "GET", "/v1/probes")
    kind_names = sorted(k["kind"] for k in (kinds or {}).get("kinds", []))
    limits = (kinds or {}).get("limits") or {}
    code_bad, bad = call(sa, "POST", "/v1/probe/create", {"name": "hb", "kind": "heartbeat@1", "every_ms": 5})
    th, agg_env = sse_async(sa, "probe/agg", 2.5)
    time.sleep(0.2)
    created = [call(sa, "POST", "/v1/probe/create", n)[0] for n in (
        {"name": "agg", "kind": "mux@1"},
        {"name": "hb", "kind": "heartbeat@1", "every_ms": 20, "parent": "agg", "ttl_ms": 1500,
         "filter": {"where": [{"field": "n", "op": ">", "value": 2}], "keep": ["n"], "every": 2}},
        {"name": "cpu", "kind": "cpu@1", "every_ms": 100},
        {"name": "mem", "kind": "mem@1", "every_ms": 100})]
    started = [call(sa, "POST", "/v1/probe/start", {"node": n})[0] for n in ("hb", "agg", "cpu", "mem")]
    _, mid = call(sa, "GET", "/v1/probe")
    mid_nodes = {n["name"]: n for n in (mid or {}).get("nodes", [])}
    th.join()
    _, envs = agg_env["r"]
    agg = [e["ev"]["data"] for e in envs if e["ch"] == "probe/agg" and e["ev"]["kind"] == "sample"]
    ns = [s.get("n") for s in agg]
    check("E15.probe_tree_filters_and_multiplexes",
          code_k == 200 and {"heartbeat@1", "cpu@1", "mem@1", "mux@1"} <= set(kind_names)
          and limits.get("max_nodes") == 256 and limits.get("min_every_ms") == 10
          and code_bad == 400 and "min_every_ms" in (bad or {}).get("error", "")
          and created == [200] * 4 and started == [200] * 4
          and mid_nodes["hb"]["state"] == "running" and mid_nodes["hb"]["parent"] == "agg"
          and (mid_nodes["hb"].get("expires_in_ms") or 0) > 0
          and len(ns) >= 10 and all(isinstance(n, (int, float)) and n > 2 and n % 2 == 0 for n in ns)
          and all(s.get("src") == "hb" and s.get("node") == "agg" and "kind" in s for s in agg),
          kinds=kind_names, limits=limits, refusal=(code_bad, (bad or {}).get("error")),
          agg_samples=len(ns), first_n=ns[:5], mid=mid_nodes.get("hb"))
    _, after = call(sa, "GET", "/v1/probe")
    after_nodes = {n["name"]: n for n in (after or {}).get("nodes", [])}
    _, ctl = sse(sa, "ctl", 0.5)
    lifecycle = [(e["ev"]["kind"], e["ev"]["data"].get("node"), e["ev"]["data"].get("reason"))
                 for e in ctl if e["ch"] == "ctl" and e["ev"]["kind"].startswith("probe.")]
    _, cpu_env = sse(sa, "probe/cpu", 0.5)
    _, mem_env = sse(sa, "probe/mem", 0.5)
    cpu_s = [e["ev"]["data"] for e in cpu_env if e["ev"]["kind"] == "sample"]
    mem_s = [e["ev"]["data"] for e in mem_env if e["ev"]["kind"] == "sample"]
    check("E15.lease_expires_and_host_kinds_read_within_range",
          "hb" not in after_nodes and after_nodes["agg"]["state"] == "running"
          and (after or {}).get("count") == 3
          and ("probe.removed", "hb", "expired") in lifecycle and ("probe.started", "hb", None) in lifecycle
          and len(cpu_s) >= 3 and all(0 <= s["busy_pct"] <= 100 and s["cores"] >= 1 for s in cpu_s)
          and len(mem_s) >= 3 and all(s["total_bytes"] > s["avail_bytes"] > 0 and 0 < s["used_pct"] < 100
                                      for s in mem_s),
          count=(after or {}).get("count"), lifecycle=lifecycle[-8:],
          cpu=cpu_s[-1] if cpu_s else None, mem=mem_s[-1] if mem_s else None)

    # M2: what one mux costs. 16 heartbeat nodes at the shortest interval (10 ms) under one
    # mux, read on the mux's pipe for 2 s: the samples per second that reach a reader (the
    # ceiling is 1600, the subtree budget plane a was started with), and the plane's processor
    # time over the window. The tree is removed
    # afterwards, the mux's removal taking its children.
    n_hb, window = 16, 2.0
    call(sa, "POST", "/v1/probe/create", {"name": "agg2", "kind": "mux@1"})
    for i in range(n_hb):
        call(sa, "POST", "/v1/probe/create", {"name": f"m2-{i}", "kind": "heartbeat@1", "every_ms": 10, "parent": "agg2"})
    th, m2_env = sse_async(sa, "probe/agg2", window + 0.5)
    gaps0 = len(SSE_GAPS)
    time.sleep(0.2)
    cpu0 = proc_cpu_ms(pa.pid)
    t0 = time.perf_counter()
    for i in range(n_hb):
        call(sa, "POST", "/v1/probe/start", {"node": f"m2-{i}"})
    call(sa, "POST", "/v1/probe/start", {"node": "agg2"})
    th.join()
    elapsed = time.perf_counter() - t0
    cpu1 = proc_cpu_ms(pa.pid)
    _, envs = m2_env["r"]
    m2 = [e for e in envs if e["ch"] == "probe/agg2" and e["ev"]["kind"] == "sample"]
    # A reader that fell behind the ring would get an `overflow` gap; `since-too-old` at the
    # start is only the ring no longer holding seq 1 after the earlier checks.
    gaps = [g for g in SSE_GAPS[gaps0:] if g["ch"] == "probe/agg2" and g["reason"] == "overflow"]
    srcs = {e["ev"]["data"].get("src") for e in m2}
    per_s = round(len(m2) / elapsed, 1)
    _, removed = call(sa, "POST", "/v1/probe/remove", {"node": "agg2"})
    for n in ("agg", "cpu", "mem"):
        call(sa, "POST", "/v1/probe/remove", {"node": n})
    measure("m2_mux_16x10ms", {"samples_per_s": per_s, "ceiling_per_s": n_hb * 100, "sources": len(srcs),
                               "plane_cpu_ms": None if None in (cpu0, cpu1) else round(cpu1 - cpu0, 1),
                               "window_s": round(elapsed, 2), "gaps": len(gaps)})
    check("M2.one_mux_carries_its_children",
          len(srcs) == n_hb and per_s > n_hb * 100 * 0.5 and not gaps
          and (removed or {}).get("removed") == n_hb + 1 and (call(sa, "GET", "/v1/probe")[1] or {}).get("count") == 0,
          samples_per_s=per_s, ceiling_per_s=n_hb * 100, sources=len(srcs), gaps=len(gaps),
          plane_cpu_ms=results["measures"]["m2_mux_16x10ms"]["plane_cpu_ms"], removed=removed)

    # E16: bookmarks and replay. A bookmark names one point of the instance's sequence, which
    # every channel shares. The operator sets `before`, a probe tree starts under the plane
    # while a live reader is on its pipes, sets `after` once something looked wrong, and takes
    # the tree down. The replay of the pipes from `before` gives back, seq for seq, what the
    # live reader saw; the replay of everything from `before` is the ctl lifecycle and both
    # pipes in one ordered stream: what the pipeline saw, after the fact.
    code_before, before = call(sa, "POST", "/v1/bookmark/set", {"bookmark": "before", "note": "before the tree"})
    # A live reader (no since): on the pipe, headers in, before the tree exists, so what it
    # collects was delivered live, not replayed.
    live_ready = threading.Event()
    th, live_env = sse_async(sa, "probe/", 2.0, query="ch=probe/", ready=live_ready)
    live_on = live_ready.wait(5)
    tree = [call(sa, "POST", "/v1/probe/create", n)[0] for n in (
        {"name": "agg3", "kind": "mux@1"},
        {"name": "hb3", "kind": "heartbeat@1", "every_ms": 20, "parent": "agg3",
         "filter": {"where": [{"field": "n", "op": ">", "value": 2}], "keep": ["n"], "every": 2}})]
    tree += [call(sa, "POST", "/v1/probe/start", {"node": n})[0] for n in ("agg3", "hb3")]
    time.sleep(1.0)
    code_after, after = call(sa, "POST", "/v1/bookmark/set", {"bookmark": "after", "note": "looked wrong here"})
    _, torn = call(sa, "POST", "/v1/probe/remove", {"node": "agg3"})
    th.join()
    _, live = live_env["r"]
    b_seq, a_seq = (before or {}).get("seq", 0), (after or {}).get("seq", 0)
    live_seqs = [e["seq"] for e in live if b_seq < e["seq"] <= a_seq]
    gaps0 = len(SSE_GAPS)
    _, replay = sse(sa, "probe/", 1.0, query="ch=probe/&bookmark=before")
    replay_seqs = [e["seq"] for e in replay if e["seq"] <= a_seq]
    _, everything = sse(sa, "", 1.0, query="ch=&bookmark=before")
    all_seqs = [e["seq"] for e in everything]
    all_chans = sorted({e["ch"] for e in everything})
    first_kinds = [(e["ch"], e["ev"]["kind"], e["ev"]["data"].get("node")) for e in everything[:2]]
    replay_gaps = SSE_GAPS[gaps0:]
    check("E16.bookmark_replays_what_the_pipeline_saw",
          code_before == 200 and code_after == 200 and before["seq"] == before["head"] and a_seq > b_seq
          and live_on and tree == [200] * 4 and (torn or {}).get("removed") == 2
          and len(live_seqs) >= 20 and replay_seqs == live_seqs and not replay_gaps
          and all_seqs == sorted(all_seqs) and len(set(all_seqs)) == len(all_seqs) and all_seqs[0] > b_seq
          and first_kinds == [("ctl", "probe.created", "agg3"), ("ctl", "probe.created", "hb3")]
          and {"ctl", "probe/agg3", "probe/hb3"} <= set(all_chans),
          before=before, after=after, live_on=live_on, live=len(live_seqs), replayed=len(replay_seqs),
          everything=len(all_seqs), channels=all_chans[:8], first=first_kinds, gaps=replay_gaps[:2])
    # The bounds, said by the plane; a bookmark older than the ring says so (since-too-old, then
    # what the ring still holds) instead of a silently partial replay; refusals name their cause;
    # a bookmark set through A on B is B's: in B's sequence, in B's list, not in A's.
    _, blist = call(sa, "GET", "/v1/bookmark")
    code_old, old = call(sa, "POST", "/v1/bookmark/set", {"bookmark": "old", "since": 1})
    gaps0 = len(SSE_GAPS)
    _, from_old = sse(sa, "probe/", 0.5, query="ch=probe/&bookmark=old")
    old_gaps = SSE_GAPS[gaps0:]
    _, _ = sse(sa, "probe/", 0.3, query="ch=probe/&bookmark=before&since=5")
    _, _ = sse(sa, "probe/", 0.3, query="ch=probe/&bookmark=nope")
    refused = SSE_GAPS[gaps0 + len(old_gaps):]
    code_name, bad_name = call(sa, "POST", "/v1/bookmark/set", {"bookmark": "Not a name"})
    code_ahead, ahead = call(sa, "POST", "/v1/bookmark/set", {"bookmark": "ahead", "since": 1 << 40})
    code_fwd, fwd_bm = call(sa, "POST", "/v1/bookmark/set", {"bookmark": "onb", "peer": "b", "note": "set from a"})
    _, blist_b = call(sa, "GET", "/v1/bookmark?peer=b")
    _, blist_a = call(sa, "GET", "/v1/bookmark")
    names_a = [m["name"] for m in (blist_a or {}).get("bookmarks", [])]
    names_b = [m["name"] for m in (blist_b or {}).get("bookmarks", [])]
    # The list says, without a replay, which bookmarks the ring still holds: `before` and
    # `after` are retained, `old` (seq 1 after the ring turned) is not, and the list's `oldest`
    # is the gap's.
    retained = {m["name"]: m.get("retained") for m in (blist_a or {}).get("bookmarks", [])}
    code_rm, removed_bm = call(sa, "POST", "/v1/bookmark/remove", {"bookmark": "old"})
    check("E16.bookmark_bounds_refusals_and_peers",
          (blist or {}).get("store") == "disk" and blist.get("max_bookmarks") == 64 and blist.get("ring") == 4096
          and [m["name"] for m in blist["bookmarks"]] == ["before", "after"] and all(m["live"] for m in blist["bookmarks"])
          and code_old == 200 and old_gaps and old_gaps[0]["reason"] == "since-too-old" and old_gaps[0]["oldest"] > 1
          and from_old and from_old[0]["seq"] >= old_gaps[0]["oldest"]
          and retained.get("before") is True and retained.get("after") is True and retained.get("old") is False
          and (blist_a or {}).get("oldest") == old_gaps[0]["oldest"] and blist_a.get("head", 0) >= blist_a["oldest"]
          and len(refused) == 2 and "bookmark or since" in refused[0].get("error", "") and "nope" in refused[1].get("error", "")
          and code_name == 400 and "name" in (bad_name or {}).get("error", "")
          and code_ahead == 400 and "head" in (ahead or {}).get("error", "")
          and code_fwd == 200 and fwd_bm["epoch"] == status(sb)["epoch"]
          and "onb" in names_b and "onb" not in names_a
          and code_rm == 200 and (removed_bm or {}).get("removed") is True,
          list=blist, old=(code_old, old_gaps[:1], len(from_old)), refused=refused,
          bad_name=(code_name, bad_name), ahead=(code_ahead, ahead), forwarded=(code_fwd, fwd_bm),
          names=(names_a, names_b), retained=retained, span=((blist_a or {}).get("oldest"), (blist_a or {}).get("head")))

    # E8: plane B restarts under its name. Its lock and state file are free again, the new boot
    # draws another epoch and opens a second section of the same journal, A's peer link comes
    # back up, and a wait scoped to the old boot ends at once with Restarted instead of waiting
    # for the new boot to reach that generation.
    b0 = status(sb)
    # A waiter made from B's state file, polling the old boot for a generation it will never
    # reach, is already running when B goes down: it must find the new boot through the state
    # file and report the restart, not the dead address.
    stale = write_ixql("stale-waiter.ixql",
                       f'done <- ix.ctl.wait(generation: {b0["generation"] + 5}, epoch: "{b0["epoch"]}", '
                       "timeout_ms: 20000)\n")
    ops0 = call(sb, "GET", "/health")[1]["ops"]
    waiter = subprocess.Popen([str(IXQLCTL), str(stale), "--state", str(STATE / "b.json")],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(WORK))
    LIVE.append(waiter)
    t_waiter = time.perf_counter()
    # The old boot counts the ops it serves (/health is not one): the shutdown waits until the
    # waiter has polled it a few times, so the check cannot pass on a waiter that only ever saw
    # the new boot.
    old_polls, polls_ms = wait_until(lambda: call(sb, "GET", "/health")[1]["ops"] - ops0 >= 3, 10)
    call(sb, "POST", "/v1/shutdown", {})
    exit_b = pb.wait(10)
    pb, _, sb, ms_b2 = up("b", spec_b)
    tokens.append(sb["token"])
    b1 = status(sb)
    _, up_ms = wait_until(lambda: (peer(status(sa), "b") or {}).get("state") == "up", 15)
    boots = [m for m in journal_lines("b") if m.get("kind") == "boot"]
    check("E8.restart_under_the_same_name",
          exit_b == 0 and b1["name"] == "b" and b1["epoch"] != b0["epoch"] and b1["generation"] == 0
          and up_ms is not None and len(boots) == 2 and boots[1]["epoch"] == b1["epoch"],
          exit=exit_b, epochs=[b0["epoch"], b1["epoch"]], peer_up_ms=up_ms, boot_lines=len(boots),
          ready_ms=round(ms_b2, 1), old_generation=b0["generation"])
    # E16 (restart): the bookmark set on B before its restart is still listed by the new boot,
    # marked as not live, and a replay from it says epoch-changed before this boot's ring.
    code_bl, bl = call(sb, "GET", "/v1/bookmark")
    onb = next((m for m in (bl or {}).get("bookmarks", []) if m["name"] == "onb"), None)
    gaps0 = len(SSE_GAPS)
    _, replayed = sse(sb, "", 0.5, query="ch=&bookmark=onb")
    rgaps = SSE_GAPS[gaps0:]
    check("E16.bookmark_survives_the_restart",
          code_bl == 200 and onb is not None and onb["epoch"] == b0["epoch"] and onb["live"] is False
          and onb["retained"] is False and onb["note"] == "set from a"
          and rgaps and rgaps[0]["reason"] == "epoch-changed" and rgaps[0]["epoch"] == b1["epoch"]
          and replayed and all(e["epoch"] == b1["epoch"] for e in replayed),
          bookmark=onb, gaps=rgaps[:2], replayed=len(replayed))
    prog = write_ixql("stale-wait.ixql",
                      f'done <- ix.ctl.wait(generation: {b0["generation"]}, epoch: "{b0["epoch"]}", '
                      "timeout_ms: 10000)\n")
    code, _, err, ms = ixql("stale-wait", prog, STATE / "b.json")
    check("E8.stale_wait_ends_at_once", code == 1 and "restarted" in err and ms < 3000,
          code=code, ms=round(ms, 1), stderr=err[-200:])
    try:
        w_out, w_err = waiter.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        waiter.kill()
        w_out, w_err = waiter.communicate()
    w_ms = (time.perf_counter() - t_waiter) * 1000
    (RUNS / "ixql-stale-waiter.stdout").write_bytes(w_out)
    (RUNS / "ixql-stale-waiter.stderr").write_bytes(w_err)
    w_err = w_err.decode("utf-8", "replace").strip()
    check("E8.waiter_follows_the_state_file",
          old_polls and waiter.returncode == 1 and "restarted" in w_err and b1["epoch"] in w_err
          and w_ms < 12000,
          code=waiter.returncode, ms=round(w_ms, 1), old_boot_polls_after_ms=polls_ms,
          stderr=w_err[-200:])

    # E17: the IXQL probe surface (D7, ix#405), end to end through the Rust client: probe.ixql
    # declares a mux and a CPU probe feeding it, starts both, reads ten of the mux's records live
    # and removes the tree. The ten are the cpu's, in the mux's seq order, with no gap and the
    # read stopped at `max:` (not by the idle timeout); the plane's list answers 200 with an empty tree.
    # E15 used the same names earlier in this boot and removed them: a replay from seq 0 would
    # have handed back that tree's records first, which is why the program reads live.
    code, out, err, ms = ixql("probe", PROBE_IXQL, STATE / "a.json", timeout=60)
    measure("probe_ixql_ms", round(ms, 1))
    samples = (out or {}).get("samples") or {}
    recs = samples.get("events") or []
    seqs = [(r.get("data") or {}).get("seq") for r in recs]
    srcs = {(r.get("data") or {}).get("src") for r in recs}
    tree_code, tree_after = call(sa, "GET", "/v1/probe")
    left = {n["name"] for n in (tree_after or {}).get("nodes", [])}
    check("E17.probe_ixql_samples_a_cpu_probe_under_a_mux_and_removes_the_tree",
          code == 0 and len(recs) == 10 and samples.get("ended") is False and samples.get("gap") is None
          and all(r.get("channel") == "probe/agg" and r.get("kind") == "sample" for r in recs)
          and srcs == {"cpu"} and all(isinstance(s, int) for s in seqs)
          and all(a < b for a, b in zip(seqs, seqs[1:]))
          and all("busy_pct" in (r.get("data") or {}) or "error" in (r.get("data") or {}) for r in recs)
          and ((out or {}).get("gone") or {}).get("removed") == 2
          and tree_code == 200 and (tree_after or {}).get("count") == 0 and not ({"agg", "cpu"} & left)
          and any(k["kind"] == "cpu@1" for k in ((out or {}).get("kinds") or {}).get("kinds", [])),
          code=code, ms=round(ms, 1), records=len(recs), seqs=seqs, srcs=sorted(s for s in srcs if s),
          ended=samples.get("ended"), gap=samples.get("gap"), gone=(out or {}).get("gone"),
          tree_status=tree_code, left_in_tree=sorted(left), stderr=err[-200:])

    # E18: probe budgets (D10, ix#416). The plane says its budget in `limits` (here the operator's,
    # `--probe-rate 1600 --probe-tree-rate 3200` at `up`); a subtree's declared
    # rate (1000 / every_ms per leaf, the filter's `every` dividing) is summed at create, and a
    # leaf that would take its mux past `max_rate_per_s` is refused naming the mux and the bound;
    # a leaf carries no budget; a mux cannot widen the plane's; the list says each subtree's rate
    # and budget and counts the bytes a running node emitted. The tree is empty here (E17 removed
    # its own), so the tree's rate is the one subtree's.
    _, kinds_b = call(sa, "GET", "/v1/probes")
    lim = (kinds_b or {}).get("limits") or {}
    bud = [call(sa, "POST", "/v1/probe/create", n)[0] for n in (
        {"name": "bud", "kind": "mux@1", "max_rate_per_s": 150},
        {"name": "bud-0", "kind": "heartbeat@1", "every_ms": 10, "parent": "bud"})]
    code_over, over = call(sa, "POST", "/v1/probe/create", {"name": "bud-1", "kind": "heartbeat@1", "every_ms": 10, "parent": "bud"})
    code_leaf, leaf = call(sa, "POST", "/v1/probe/create", {"name": "solo", "kind": "heartbeat@1", "every_ms": 10, "max_rate_per_s": 10})
    code_wide, wide = call(sa, "POST", "/v1/probe/create", {"name": "wide", "kind": "mux@1", "max_rate_per_s": 100000})
    started_b = [call(sa, "POST", "/v1/probe/start", {"node": n})[0] for n in ("bud", "bud-0")]
    time.sleep(0.3)
    _, tree_b = call(sa, "GET", "/v1/probe")
    nodes_b = {n["name"]: n for n in (tree_b or {}).get("nodes", [])}
    mux_b, leaf_b = nodes_b.get("bud", {}), nodes_b.get("bud-0", {})
    _, gone_b = call(sa, "POST", "/v1/probe/remove", {"node": "bud"})
    check("E18.probe_budget_bounds_the_declared_rate_and_counts_bytes",
          lim.get("max_rate_per_s") == 1600 and lim.get("tree_rate_per_s") == 3200
          and bud == [200, 200] and started_b == [200, 200]
          and code_over == 400 and "bud" in (over or {}).get("error", "") and "max_rate_per_s is 150" in (over or {}).get("error", "")
          and code_leaf == 400 and "leaf" in (leaf or {}).get("error", "")
          and code_wide == 400 and "widen" in (wide or {}).get("error", "")
          and mux_b.get("rate_per_s") == 100 and mux_b.get("max_rate_per_s") == 150
          and leaf_b.get("rate_per_s") == 100 and "max_rate_per_s" not in leaf_b
          and (leaf_b.get("bytes") or 0) > 0 and (mux_b.get("bytes") or 0) >= leaf_b.get("bytes", 0)
          and (tree_b or {}).get("rate_per_s") == 100 and (tree_b or {}).get("tree_rate_per_s") == 3200
          and (gone_b or {}).get("removed") == 2,
          limits={k: lim.get(k) for k in ("max_rate_per_s", "tree_rate_per_s")}, created=bud, started=started_b,
          refusals=[(code_over, (over or {}).get("error")), (code_leaf, (leaf or {}).get("error")),
                    (code_wide, (wide or {}).get("error"))],
          mux=mux_b, leaf=leaf_b, tree_rate=(tree_b or {}).get("rate_per_s"), gone=gone_b)

    # E6: the plane dies hard. Its pods die with it, the client fails fast, B sees A go.
    call(sa, "POST", "/v1/apply", doc)
    _, _ = wait_until(lambda: svc(status(sa), "pod-a")["state"] == "running", 10)
    victim = svc(status(sa), "pod-a")["pid"]
    t0 = time.perf_counter()
    pa.kill()
    pa.wait(5)
    plane_gone_ms = (time.perf_counter() - t0) * 1000
    _, pod_gone_ms = wait_until(lambda: not alive(victim), 5, step=0.001)
    _, down_ms = wait_until(lambda: (peer(status(sb), "a") or {}).get("state") == "down", 10,
                            step=0.005)
    down_after_kill = None if down_ms is None else round((time.perf_counter() - t0) * 1000, 1)
    prog = write_ixql("status.ixql", "s <- ix.ctl.status()\n")
    code, _, err, client_ms = ixql("dead-plane", prog, STATE / "a.json")
    # The same failure without the client's process start: a bare connect to the dead port.
    host, port = sa["addr"].rsplit(":", 1)
    t1 = time.perf_counter()
    try:
        socket.create_connection((host, int(port)), timeout=5).close()
        refused = "connected?!"
    except OSError as e:
        refused = type(e).__name__
    bare_ms = (time.perf_counter() - t1) * 1000
    check("E6.pods_die_with_the_plane", pod_gone_ms is not None, pod_pid=victim,
          plane_exit_ms=round(plane_gone_ms, 1), pod_gone_after_plane_exit_ms=pod_gone_ms)
    check("E6.client_fails_bounded", code == 1 and "unreachable" in err and client_ms < 3000,
          code=code, ms=round(client_ms, 1), bare_connect=refused, bare_connect_ms=round(bare_ms, 1),
          stderr=err[-160:])
    check("E6.peer_down_seen_by_b", down_ms is not None, ms_after_kill=down_after_kill)
    return finish(None, pb, sa, sb, tokens)


def finish(pa, pb, sa, sb, tokens):
    for p, st in ((pa, sa), (pb, sb)):
        if p is None or p.poll() is not None:
            continue
        try:
            call(st, "POST", "/v1/shutdown", {})
        except OSError:
            pass
        try:
            code = p.wait(10)
        except subprocess.TimeoutExpired:
            p.kill()
            code = "killed"
        results["notes"].append(f"plane {st['name']} exit {code}, state file left: "
                                f"{(STATE / (st['name'] + '.json')).exists()}")
    time.sleep(0.3)
    # E4: no token in any output, error text or journal. The state files hold them by design.
    hits = {}
    for f in WORK.rglob("*"):
        if not f.is_file() or f.suffix == ".exe" or BIN in f.parents \
                or (f.parent == STATE and f.name in ("a.json", "b.json")):
            continue
        data = f.read_bytes()
        n = sum(data.count(t.encode()) for t in tokens)
        if n:
            hits[str(f.relative_to(WORK))] = n
    journals = [str(f.relative_to(WORK)) for f in STATE.glob("*.journal.jsonl")]
    check("E4.no_token_in_outputs", not hits, files_with_token=hits, journals_scanned=journals)
    (STATE / "a.json").unlink(missing_ok=True)
    ok = all(c["ok"] for c in results["checks"].values())
    results["ok"] = ok
    (WORK / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print("ALL PASS" if ok else "SOME CHECKS FAILED", "->", WORK / "results.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
