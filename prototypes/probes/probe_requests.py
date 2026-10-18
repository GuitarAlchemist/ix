"""Turn probe requests into template probes.

    python -B probe_requests.py              handle the pending requests once
    python -B probe_requests.py --every 2    keep watching

A request is a small JSON file in %LOCALAPPDATA%/ix-probes/requests/, written whole (a .part
renamed into place), its name the request id:

    {"template": "process", "name": "obs", "match": "obs64%"}

The watcher hands the parameters to new_probe.py, which accepts a fixed template list and
strictly patterned values, then writes <id>.result.json beside the request
({"ok": true, "probe": "proc-obs"} or {"ok": false, "error": "..."}) and removes the request.
The running host builds the new probe by itself. The scene, a voice command or an agent can ask
for a probe this way; none of them can supply code.
"""
import json, os, re, subprocess, sys, time

import new_probe

HERE = os.path.dirname(os.path.abspath(__file__))
REQUESTS = os.path.join(os.environ["LOCALAPPDATA"], "ix-probes", "requests")
MAX_TEMPLATE_PROBES = 8  # a request can add a WMI query every interval; keep the total small
RESULT_HOURS = 1


def handle(path):
    with open(path, encoding="utf-8") as f:
        request = json.loads(f.read(1024))
    if not isinstance(request, dict) or request.get("template") != "process":
        return {"ok": False, "error": "templates: process"}
    name, match = str(request.get("name", "")), str(request.get("match", ""))
    if not re.fullmatch(dict(new_probe.TEMPLATES["process"])["name"], name):
        return {"ok": False, "error": "name must be a short lowercase word"}
    if os.path.isdir(os.path.join(HERE, "probes", f"proc-{name}")):
        return {"ok": True, "probe": f"proc-{name}", "existed": True}
    if len([d for d in os.listdir(os.path.join(HERE, "probes")) if d.startswith("proc-")]) >= MAX_TEMPLATE_PROBES:
        return {"ok": False, "error": f"at most {MAX_TEMPLATE_PROBES} template probes"}
    # new_probe.py validates every value against its pattern before it reaches the source.
    r = subprocess.run([sys.executable, "-B", "-I", os.path.join(HERE, "new_probe.py"), "process", name, match],
                       capture_output=True, text=True, timeout=30)
    if r.returncode:
        return {"ok": False, "error": (r.stderr.strip().splitlines() or ["refused"])[-1][:200]}
    return {"ok": True, "probe": f"proc-{name}"}


def sweep():
    os.makedirs(REQUESTS, exist_ok=True)
    for entry in os.listdir(REQUESTS):
        path = os.path.join(REQUESTS, entry)
        if entry.endswith(".result.json"):
            if time.time() - os.path.getmtime(path) > RESULT_HOURS * 3600:
                os.remove(path)
            continue
        if not entry.endswith(".json"):
            continue
        rid = entry[:-len(".json")]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", rid):
            os.remove(path)
            continue
        try:
            result = handle(path)
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            result = {"ok": False, "error": type(e).__name__}
        result["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        part = os.path.join(REQUESTS, rid + ".result.json.part")
        with open(part, "w", encoding="utf-8") as f:
            json.dump(result, f)
        os.replace(part, os.path.join(REQUESTS, rid + ".result.json"))
        os.remove(path)
        print(rid, json.dumps(result), flush=True)


if __name__ == "__main__":
    every = int(sys.argv[2]) if sys.argv[1:2] == ["--every"] else 0
    while True:
        sweep()
        if not every:
            break
        time.sleep(every)
