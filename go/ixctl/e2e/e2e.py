"""End-to-end gate for the ixctl control plane: the real Go plane (`go/ixctl`), the real Rust
client (`crates/ix-ctl`, through its `ixql-ctl` example) and ix-stream's `pod` example, as two
federated instances on loopback. 30 checks, E1..E9, a restart loop and the M1 measurement.

    python go/ixctl/e2e/e2e.py --build            # build the three binaries, then run the gate
    python go/ixctl/e2e/e2e.py --bin <dir>        # use ixctl, pod and ixql-ctl already built there
    python go/ixctl/e2e/e2e.py --build --work D   # keep binaries, state, runs and results.json in D

Paths are relative to this file: the repo root is three levels up, `pods.ixql` comes from
`crates/ix-ctl/examples`. Everything the gate writes goes under the work folder (by default
`ixctl-e2e` in the system temp folder): `bin/`, `state/` (the planes' state files and journals),
`runs/` (each IXQL program with its stdout and stderr, each plane's output) and `results.json`.
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
EXE = ".exe" if os.name == "nt" else ""

# Set by main() from the arguments.
WORK = BIN = STATE = RUNS = None
IXCTL = POD = IXQLCTL = None

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
    """Build the plane, the pod and the IXQL client from this tree into BIN."""
    t0 = time.perf_counter()
    subprocess.run(["go", "build", "-o", str(BIN / ("ixctl" + EXE)), "."],
                   cwd=str(REPO / "go" / "ixctl"), check=True)
    env = {**os.environ, "CARGO_PROFILE_DEV_DEBUG": "0", "CARGO_INCREMENTAL": "0"}
    r = subprocess.run(["cargo", "build", "-q", "-p", "ix-stream", "--example", "pod",
                        "-p", "ix-ctl", "--example", "ixql-ctl", "--message-format=json"],
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


def sse(st, ch, seconds):
    """The envelopes /v1/events gives for `ch` (prefix) from the start, read for `seconds`."""
    host, port = st["addr"].rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=seconds)
    req = (
        f"GET /v1/events?ch={ch}&since=0 HTTP/1.1\r\nHost: {st['addr']}\r\n"
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
    sock.close()
    text = buf.decode("utf-8", "replace")
    head, _, body = text.partition("\r\n\r\n")
    envs = []
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                envs.append(json.loads(line[6:]))
            except ValueError:
                pass
    return head.split("\r\n")[0], envs


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


def up(name, spec):
    out = open(RUNS / f"plane-{name}.stdout", "wb", buffering=0)
    err = open(RUNS / f"plane-{name}.stderr", "wb", buffering=0)
    t0 = time.perf_counter()
    p = subprocess.Popen(
        [str(IXCTL), "up", str(spec), "--name", name, "--state", str(STATE)],
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


def main():
    try:
        return run()
    finally:
        reap()


def run():
    global WORK, BIN, STATE, RUNS, IXCTL, POD, IXQLCTL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true", help="build ixctl, pod and ixql-ctl from this tree first")
    ap.add_argument("--bin", type=pathlib.Path, help="folder holding prebuilt ixctl, pod and ixql-ctl (default: <work>/bin)")
    ap.add_argument("--work", type=pathlib.Path, default=pathlib.Path(tempfile.gettempdir()) / "ixctl-e2e",
                    help="where state, runs, results.json and built binaries go")
    args = ap.parse_args()

    WORK = args.work.resolve()
    BIN = (args.bin or WORK / "bin").resolve()
    STATE, RUNS = WORK / "state", WORK / "runs"
    IXCTL, POD, IXQLCTL = BIN / ("ixctl" + EXE), BIN / ("pod" + EXE), BIN / ("ixql-ctl" + EXE)

    for d in (STATE, RUNS):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    if args.build:
        BIN.mkdir(parents=True, exist_ok=True)
        build()
    for exe in (IXCTL, POD, IXQLCTL):
        assert exe.exists(), f"missing {exe} (build it, or pass --build)"
    assert PODS_IXQL.exists(), f"missing {PODS_IXQL}"

    spec_b = spec("b", [pod("pod-a", "b/pod-a", 3)],
                  [{"name": "a", "state_file": str(STATE / "a.json")}])
    spec_a = spec("a", [
        pod("pod-a", "a/pod-a", 4),
        pod("pod-b", "a/pod-b", 4),
        {**pod("crashy", "a/crashy", 2, 5, ["--crash-after", "0.15"]), "max_restarts": 2},
        {"name": "missing", "command": str(BIN / ("does-not-exist" + EXE)), "restart": "never"},
    ], [{"name": "b", "state_file": str(STATE / "b.json")}])

    pb, ready_b, sb, ms_b = up("b", spec_b)
    pa, ready_a, sa, ms_a = up("a", spec_a)
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
