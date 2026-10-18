"""Write the IX control plane's read-only snapshot for the 3D scene.

It reads the newest data-* folder the IX Control Room refresh gathered (PRs, main runs, work
items), the live session registry (~/.claude/sessions/*.json, never the *.key files) and the
claims ledger (~/.agents/claims.jsonl), and writes %LOCALAPPDATA%/ix-control/snapshot.json
atomically. Schema ix-control-snapshot/0 is provisional: a reader keys on `schema` and ignores
fields it does not know.

  control_snapshot.py              write once
  control_snapshot.py --every 5    rewrite every 5 s (sessions and probe samples change faster than the gather)

`sessions`, `lanes`, `decisions` and `horizon` name sessions and quote ledger notes, and
`databases` gives local paths: they are for the user's own view, never for the public stream. The stream reads public.json alone
(schema ix-public/0): running sessions, the demo plan from demo-plan.json, the next demo's time.
"""
import ctypes, glob, json, os, re, shutil, subprocess, sys, time
from datetime import datetime, timedelta, timezone

here = os.path.dirname(os.path.abspath(__file__))
home = os.path.expanduser("~")
LANE_HOURS = 48
ACTIVE = {"in-progress", "in_progress", "claimed", "pr-open", "pr-updated", "awaiting-review",
          "awaiting-ci-and-review", "ready", "local-ready", "ready-local", "handoff-ready",
          "blocked", "review", "in-review", "paused", "committed-local", "local-commit", "live"}
# A ledger note that hands the next step to the user, in the words the lanes actually use.
FOR_USER = re.compile(r"operator|user'?s (go|merge|review|label|ok|decision)|awaiting the (user|operator)"
                      r"|merge go|sign-?off|l'utilisateur|on the user", re.I)
REPOS = os.path.join(home, "source", "repos")
DB_GLOBS = [os.path.join(REPOS, "*", "state", "**", "*.duckdb"), os.path.join(REPOS, "*", "code", "**", "*.duckdb")]
DB_EVERY = 300  # s: listing every file's tables is cheap, but not every 5 s
_dbs = {"at": 0.0, "list": []}


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def alive(pid):
    # os.kill(pid, 0) terminates the process on Windows, so ask the kernel instead.
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def sessions():
    out = []
    for path in glob.glob(os.path.join(home, ".claude", "sessions", "*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError):
            continue
        if not s.get("pid") or not alive(s["pid"]):
            continue
        out.append({"name": s.get("name") or "?", "repo": os.path.basename(s.get("cwd") or ""),
                    "status": s.get("status") or "?",
                    "since": iso(s["statusUpdatedAt"]) if s.get("statusUpdatedAt") else None})
    return sorted(out, key=lambda s: (s["status"] != "waiting", s["name"]))


def lanes(now):
    latest = {}
    try:
        with open(os.path.join(home, ".agents", "claims.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                latest[(r.get("repo"), r.get("lane"))] = r
    except OSError:
        return []
    cut = (now - timedelta(hours=LANE_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
    return [{"repo": r.get("repo"), "lane": r.get("lane"), "status": r.get("status"),
             "at": r.get("ts"), "note": r.get("note"), "until": r.get("until")}
            for r in latest.values() if (r.get("ts") or "") >= cut and r.get("status") in ACTIVE]


def decisions(live, active, prs, work):
    # What waits on the user, oldest first within each kind; a waiting session idles a whole agent.
    out = [{"kind": "session", "what": f"{s['name']} waits for your answer", "where": s["repo"],
            "since": s["since"]} for s in live if s["status"] == "waiting"]
    out += [{"kind": "lane", "what": l["note"], "where": f"{l['repo']} {l['lane']}", "since": l["at"]}
            for l in active if FOR_USER.search(l["note"] or "")]
    for p in prs:
        if p["draft"]:
            continue
        if p["failing"] == ["risk-report"]:
            out.append({"kind": "review", "what": f"#{p['number']} {p['title']}: review, then the agent-blackbox-reviewed label",
                        "where": p["url"], "since": p["updated_at"]})
        elif p["ci"] == "SUCCESS" and not p["codex_open"]:
            out.append({"kind": "merge", "what": f"#{p['number']} {p['title']}: green, ready for your merge",
                        "where": p["url"], "since": p["updated_at"]})
    out += [{"kind": "work", "what": w["next"], "where": w["item"], "since": None}
            for w in work if re.search(r"\buser'?s?\b|on the user's go", w.get("next") or "", re.I)]
    order = ["session", "merge", "review", "lane", "work"]
    return sorted(out, key=lambda d: (order.index(d["kind"]), d["since"] or ""))


def horizon(live, active, now):
    # The latest time a working lane says it can run without the user. Lanes declare it with an
    # optional `until` in their claim; none declared means no estimate, not "now".
    until = [l["until"] for l in active if l.get("until") and l["until"] > now.strftime("%Y-%m-%dT%H:%M:%SZ")]
    count = {k: sum(s["status"] == k for s in live) for k in ("busy", "waiting", "idle")}
    return {**count, "autonomous_until": max(until) if until else None,
            "basis": f"{len(until)} lane(s) declared an until" if until else "no working lane declared an until"}


def databases():
    # The local DuckDB files with their tables and estimated row counts, for the user's private
    # view. Each is opened read-only; one a writer holds is listed as busy, never waited on.
    if time.time() - _dbs["at"] < DB_EVERY:
        return _dbs["list"]
    cli = shutil.which("duckdb")
    out = []
    for path in sorted({p for g in DB_GLOBS for p in glob.glob(g, recursive=True)}):
        entry = {"repo": os.path.relpath(path, REPOS).split(os.sep)[0], "path": path,
                 "mb": round(os.path.getsize(path) / 2**20, 1), "modified": iso(os.path.getmtime(path) * 1000),
                 "tables": []}
        try:
            r = subprocess.run([cli or "duckdb", "-readonly", "-json", path, "-c",
                                "SELECT schema_name AS schema, table_name AS name, estimated_size AS rows,"
                                " column_count AS columns FROM duckdb_tables() ORDER BY 1, 2"],
                               capture_output=True, text=True, timeout=10, creationflags=0x08000000)  # no window
            if r.returncode == 0:
                entry["tables"] = json.loads(r.stdout or "[]")
            else:
                entry["error"] = "busy" if "lock" in r.stderr.lower() else r.stderr.strip()[:200]
        except (OSError, subprocess.TimeoutExpired, ValueError) as e:
            entry["error"] = type(e).__name__
        out.append(entry)
    _dbs.update(at=time.time(), list=out)
    return out


def build():
    dirs = sorted(glob.glob(os.path.join(here, "data-*")))
    if not dirs:
        sys.exit("no data-* folder yet: run refresh.py plan first")
    src = dirs[-1]

    def load(name):
        with open(os.path.join(src, name + ".json"), encoding="utf-8") as f:
            return json.load(f)

    prs = [{
        "number": p["number"], "title": p["title"], "url": p["url"], "draft": p["draft"],
        "ci": p["ci"],  # SUCCESS | FAILURE | PENDING | ...
        "failing": [c for c in (p.get("failing_checks") or "").split(", ") if c],  # gather.py joins the names
        "codex_open": p.get("codex_p0_p1_open") or 0,
        "updated_at": p.get("updated_at"),
    } for p in load("open_prs")]

    latest = {}
    for r in load("main_runs"):
        if r["workflow"] not in latest or r["created_at"] > latest[r["workflow"]]["created_at"]:
            latest[r["workflow"]] = r
    main = [{"workflow": w, "result": r["result"], "at": r["created_at"]} for w, r in sorted(latest.items())]

    work = [{"id": w["id"], "item": w["item"], "verdict": w.get("verdict"), "next": w.get("next")}
            for w in load("indigo_work")]

    now = datetime.now(timezone.utc)
    live, active = sessions(), lanes(now)
    return {
        "schema": "ix-control-snapshot/0",
        "at": os.path.basename(src)[len("data-"):],
        "written": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": "GuitarAlchemist/ix",
        "main": main,
        "prs": prs,
        "work": work,
        "surface": (load("surface") or [{}])[0],
        "sessions": live,
        "lanes": active,
        "decisions": decisions(live, active, prs, work),
        "horizon": horizon(live, active, now),
        "databases": databases(),
    }


def tr(text):
    # A demo-plan title is a French string or {"en": ..., "fr": ..., "es": ...}. The public file
    # keeps `title` in French for older readers and adds `titles`, one per language.
    titles = text if isinstance(text, dict) else {"fr": text}
    return {"title": titles.get("fr"), "titles": titles}


def public(snapshot):
    # The only part the public stream may show: session names and repos, states, the demo plan's
    # titles. No ledger notes, no PR titles, no paths.
    with open(os.path.join(here, "demo-plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    demo = plan.get("next_demo") or {}
    at = datetime.fromisoformat(demo["at"]) if demo.get("at") else None
    steps = plan.get("steps") or []
    return {
        "schema": "ix-public/0",
        "written": snapshot["written"],
        "next_demo": {**tr(demo.get("title")), "at": demo.get("at"),
                      "at_local": at.astimezone().strftime("%H:%M") if at else None},
        "chantier": {**tr(plan.get("chantier")), "done": sum(s["state"] == "done" for s in steps),
                     "total": len(steps), "steps": [{**tr(s["title"]), "state": s["state"]} for s in steps]},
        "sessions": [{"name": s["name"], "repo": s["repo"], "status": s["status"], "since": s["since"]}
                     for s in snapshot["sessions"] if s["status"] in ("busy", "waiting")],
        "idle": sum(s["status"] == "idle" for s in snapshot["sessions"]),
        "waiting_on_human": len(snapshot["decisions"]),
        "autonomous_until_local": local_hm(snapshot["horizon"]["autonomous_until"]),
        "machine": machine(),
        "viewers": viewers(),
    }


def viewers():
    # The stream's viewer count, from the scene cast's front door (viewer_proxy.py, the 3D scene
    # session's file): {"at", "viewers"}, rewritten every 5 s. Missing or older than 30 s gives None.
    path = os.path.join(os.environ["LOCALAPPDATA"], "scene-cast", "viewers.json")
    try:
        with open(path, encoding="utf-8") as f:
            v = json.load(f)
        at = datetime.fromisoformat(v["at"].replace("Z", "+00:00"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if datetime.now(timezone.utc) - at > timedelta(seconds=30) or not isinstance(v.get("viewers"), int):
        return None
    return v["viewers"]


def machine():
    # The probe host's latest samples (prototype/probes, writing to %LOCALAPPDATA%/ix-probes).
    # Stale or missing samples give None, so the scene shows no gauge rather than an old one.
    path = os.path.join(os.environ["LOCALAPPDATA"], "ix-probes", "probes.jsonl")
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 32768))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return None
    last, cpu = {}, []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        last[r.get("metric")] = r
        # Task Manager's % Processor Utility; WMI's LoadPercentage (cpu_load_pct) over-reads.
        if r.get("metric") == "cpu_util_pct":
            cpu.append(round(r["value"]))
    if "ram_free_mb" not in last or "ram_total_mb" not in last:
        return None
    at = last["ram_free_mb"]["ts"]
    if datetime.now(timezone.utc) - datetime.fromisoformat(at.replace("Z", "+00:00")) > timedelta(seconds=30):
        return None
    total, free = last["ram_total_mb"]["value"], last["ram_free_mb"]["value"]
    val = lambda m: last[m]["value"] if m in last else None
    gpu = None
    if val("gpu_mem_total_mb"):
        gpu = {"util_pct": val("gpu_util_pct"), "mem_used_pct": round(100 * val("gpu_mem_used_mb") / val("gpu_mem_total_mb")),
               "mem_total_gb": round(val("gpu_mem_total_mb") / 1024), "temp_c": val("gpu_temp_c"), "power_w": val("gpu_power_w")}
    disks = [{"drive": m[len("disk_"):-len("_free_gb")].upper(), "free_gb": round(last[m]["value"]),
              "total_gb": round(val(m.replace("_free_gb", "_total_gb")) or 0)}
             for m in sorted(last) if m and m.startswith("disk_") and m.endswith("_free_gb")]
    # Probes made from the process template (new_probe.py): proc_<name>_cpu_pct and _ram_mb.
    labels = {"scene": {"en": "3D scene", "fr": "Scène 3D", "es": "Escena 3D"},
              "obs": {"en": "Public stream", "fr": "Flux public", "es": "Transmisión pública"},
              "tunnel": {"en": "Tunnel", "fr": "Tunnel", "es": "Túnel"},
              "relay": {"en": "Video relay", "fr": "Relais vidéo", "es": "Relé de vídeo"},
              "voice": {"en": "Bot's voice", "fr": "Voix du bot", "es": "Voz del bot"}}
    processes = [{"name": m[5:-8], "label": labels.get(m[5:-8], {}).get("fr", m[5:-8]),
                  "labels": labels.get(m[5:-8], {}), "cpu_pct": round(last[m]["value"], 1),
                  "ram_gb": round((val(m.replace("_cpu_pct", "_ram_mb")) or 0) / 1024, 1)}
                 for m in sorted(last) if m and m.startswith("proc_") and m.endswith("_cpu_pct")]
    return {"at": at, "cpu_pct": cpu[-1] if cpu else None, "cpu_history": cpu[-30:], "processes": processes,
            "ram_used_pct": round(100 * (total - free) / total), "ram_total_gb": round(total / 1024),
            "cpu_logical": int(last["cpu_logical"]["value"]) if "cpu_logical" in last else None,
            "gpu": gpu, "disks": disks}


def local_hm(utc):
    return datetime.fromisoformat(utc.replace("Z", "+00:00")).astimezone().strftime("%H:%M") if utc else None


def write(name, data):
    out_dir = os.path.join(os.environ["LOCALAPPDATA"], "ix-control")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name)
    with open(out + ".part", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(out + ".part", out)
    return out


if __name__ == "__main__":
    every = int(sys.argv[2]) if sys.argv[1:2] == ["--every"] else 0
    while True:
        try:
            s = build()
        except Exception as e:  # e.g. a data-* folder the refresh is still writing
            if not every:
                raise
            with open(os.path.join(os.environ["LOCALAPPDATA"], "ix-control", "snapshot-loop.log"), "a",
                      encoding="utf-8") as f:
                f.write(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {type(e).__name__}: {e}\n")
            time.sleep(every)
            continue
        out = write("snapshot.json", s)
        write("public.json", public(s))
        if not every:
            print(out, len(s["prs"]), "PRs", sum(p["ci"] != "SUCCESS" for p in s["prs"]), "not green,",
                  len(s["decisions"]), "decisions,", s["horizon"]["busy"], "sessions busy")
            break
        time.sleep(every)
