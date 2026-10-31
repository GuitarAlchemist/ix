"""Prepare and record IX Control Room refreshes.

  refresh.py plan            run gather.py + gather_indigo.py into data-<stamp>/, compare each file's
                             sha256 with state.json, print {stamp, dir, changed: [{id, path, old_blob}],
                             unchanged: [id]} as JSON, then rewrite the 3D scene's control snapshot
  refresh.py record ID BLOB SHA256 ...
                             after the uploads and the store write, record each new asset in state.json

The Artifact uploads, the store batch and the old-asset deletes are tool calls, made by Claude."""
import hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
STATE = HERE / "state.json"
PY = sys.executable
IDS = ["open_prs", "merged_prs", "main_runs", "open_issues", "crate_tiers", "surface", "indigo_work"]


def load():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def plan():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = HERE / f"data-{stamp}"
    for script in ("gather.py", "gather_indigo.py"):
        r = subprocess.run([PY, "-B", str(HERE / script), str(out)], capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            print(json.dumps({"error": f"{script} failed", "stderr": r.stderr.strip()[-2000:]}))
            sys.exit(1)
    state = load()
    changed, unchanged = [], []
    for ds in IDS:
        path = out / f"{ds}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if state.get(ds, {}).get("sha256") == digest:
            unchanged.append(ds)
        else:
            changed.append({"id": ds, "path": str(path), "sha256": digest, "old_blob": state.get(ds, {}).get("blob")})
    print(json.dumps({"stamp": stamp, "at": datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "dir": str(out), "changed": changed, "unchanged": unchanged}, indent=1))
    # The 3D scene's control plane reads the same gather; its output stays off stdout, which is the plan's JSON.
    subprocess.run([PY, "-B", str(HERE / "control_snapshot.py")], capture_output=True)


def record(args):
    state = load()
    for ds, blob, digest in zip(args[0::3], args[1::3], args[2::3]):
        assert ds in IDS and len(blob) == 32 and len(digest) == 64, (ds, blob, digest)
        state[ds] = {"blob": blob, "sha256": digest}
    STATE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print("recorded", len(args) // 3)


if __name__ == "__main__":
    # The cron's stored prompt runs `plan .`; the "." is accepted and ignored.
    plan() if sys.argv[1:] in (["plan"], ["plan", "."]) else record(sys.argv[2:]) if sys.argv[1:2] == ["record"] else sys.exit(__doc__)
