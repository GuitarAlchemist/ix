"""Indigo's work in progress for the IX Control Room: one row per item, as JSON row objects.

Heads and commit dates are read from git (origin), PR state from gh, counts and digests recomputed
from the result files named in each row. `verdict`, `open_findings`, `review` and `next` are hand-kept
notes: a reviewer's verdict as relayed by the user, dated in `review`. Update them here when a verdict
changes, or the automatic refresh keeps publishing the old one.
The page is public: no local paths, no personal data."""
import hashlib, json, os, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

REPO = "GuitarAlchemist/ix"
CLONE = os.environ.get("IX_CLONE") or os.path.expanduser(os.path.join("~", "source", "repos", "ix"))
# Where the canary and tracer result files sit, and the review bundles: local, per machine.
SCR = Path(os.environ.get("IX_DASH_SCRATCH", "."))
TASK = Path(os.environ.get("IX_DASH_TASK", "."))
OUT = Path(sys.argv[1])


def run(args):
    out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if out.returncode:
        raise RuntimeError(f"{args[:3]}: {out.stderr.strip()}")
    return out.stdout.strip()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def branch(name):
    heads = dict(reversed(l.split("\t")) for l in run(["git", "-C", CLONE, "ls-remote", "--heads", "origin", name]).splitlines())
    sha = heads[f"refs/heads/{name}"]
    run(["git", "-C", CLONE, "fetch", "-q", "origin", name])
    prs = json.loads(run(["gh", "pr", "list", "-R", REPO, "--head", name, "--state", "all", "--json", "number,state"]))
    return {
        "head_sha": sha[:7],
        "updated_at": datetime.fromisoformat(run(["git", "-C", CLONE, "log", "-1", "--format=%cI", sha]))
        .astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ahead_of_main": int(run(["git", "-C", CLONE, "rev-list", "--count", f"origin/main..{sha}"])),
        "pr": ", ".join(f"#{p['number']} {p['state'].lower()}" for p in prs) or "none",
    }


def test_counts(log, binaries):
    """Sum `test result` lines of the named test binaries in a cargo test log."""
    passed = failed = 0
    current = None
    for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"Running tests[\\/](\S+)\.rs", line)
        if m:
            current = m.group(1)
            continue
        m = re.search(r"test result: \w+\. (\d+) passed; (\d+) failed", line)
        if m and current in binaries:
            passed += int(m.group(1))
            failed += int(m.group(2))
            current = None
    return passed, failed


def test_counts_split(log, binaries):
    """As test_counts, for a log whose stdout and stderr were captured apart (stdout, then a
    `----- stderr -----` line, then stderr): the n-th binary header on stderr owns the n-th
    `test result` line on stdout."""
    out, sep, err = Path(log).read_text(encoding="utf-8", errors="replace").partition("\n----- stderr -----\n")
    assert sep, f"{log}: not a split log"
    heads = [m.group(1) or "" for m in re.finditer(r"^\s+(?:Running (\S+)|Doc-tests \S+)", err, re.M)]
    results = re.findall(r"^test result: \w+\. (\d+) passed; (\d+) failed", out, re.M)
    assert len(heads) == len(results), f"{log}: {len(heads)} binaries, {len(results)} results"
    passed = failed = 0
    for head, (p, f) in zip(heads, results):
        m = re.fullmatch(r"tests[\\/](\S+)\.rs", head)
        if m and m.group(1) in binaries:
            passed += int(p)
            failed += int(f)
    return passed, failed


rows = []

# Recovery canary: an isolated spike, outside git.
res = SCR / "recovery" / "evidence" / "results-20261017T003105Z.json"
summary = json.loads(res.read_text(encoding="utf-8"))["summary"]
bundle = TASK / "indigo-recovery-canary-20261017.zip"
rows.append({
    "id": "recovery-canary", "item": "Recovery canary (isolated spike)", "where": "session scratchpad, not in git",
    "head_sha": None, "ahead_of_main": None, "pr": "none",
    "updated_at": datetime.fromtimestamp(summary["finished_ns"] / 1e9, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "checks_passed": summary["checks_total"] - summary["checks_failed"], "checks_total": summary["checks_total"],
    "mutants_red": None, "mutants_total": None,
    "evidence": res.name, "evidence_sha256": sha256(res)[:12],
    "bundle": bundle.name, "bundle_sha256": sha256(bundle)[:12],
    "binary_sha256": summary["bin_sha256"].removeprefix("sha256:")[:12],
    "verdict": "code approval pending", "open_findings": None,
    "review": "Astra reviewed the report; code approval waits on the bundle",
    "next": "Astra's code review settles who owns recovery",
})

# ix-recover tracer, with the fixes to Astra's reviews (ix#386).
log = SCR / "r2prov" / "final-debug.log"
p, f = test_counts_split(log, {"kill", "recover", "pipeline_recover", "pipeline_kill"})
mut = json.loads((SCR / "mut" / "mutants-result.json").read_text(encoding="utf-8"))
mut.update(json.loads((SCR / "mut2" / "mutants2-result.json").read_text(encoding="utf-8")))
for base in ("M0-none", "K0-none", "S0-none"):
    assert not mut[base]["failed_tests"] and mut[base]["exit"] == 0, f"{base}: the unmutated build must be green"
mutants = {k: v for k, v in mut.items() if not k.endswith("0-none")}
# R2-1's witness; the K1 rerun in the same file re-checks a mutant already counted.
r2 = json.loads((SCR / "r2prov" / "r2-provenance.json").read_text(encoding="utf-8"))
for step in r2["steps"]:
    if step["name"].startswith("mutant-R"):
        mutants[step["name"].removeprefix("mutant-")] = {"failed_tests": step["failed"], "exit": step["exit"]}
    if step["name"].startswith("final-"):
        assert step["exit"] == 0 and not step["failed"], f"{step['name']}: the clean runs must be green"
bundle = TASK / "ix-recover-r2-2b83590.zip"
rows.append({
    "id": "ix-recover", "item": "ix-recover tracer (ix pipeline run --run-id)", "where": "exp/ix-recover",
    **branch("exp/ix-recover"),
    "checks_passed": p, "checks_total": p + f,
    "mutants_red": sum(bool(v["failed_tests"]) for v in mutants.values()), "mutants_total": len(mutants),
    "evidence": log.name, "evidence_sha256": sha256(log)[:12],
    "bundle": bundle.name, "bundle_sha256": sha256(bundle)[:12], "binary_sha256": None,
    "verdict": "source approved (provisional, stats/distance only)",
    "open_findings": "none blocking; non-blocking: Unix kill -0 check reads an English diagnostic; F4 separate",
    "review": "Astra review 3 on 2b83590 (ix#386, 2026-10-17): R2-1 closed; static source approval, not runtime certification",
    "next": "Hand Astra the check-3 provenance bundle. A Gaia consumer may start within the reviewed contract, on the user's go.",
})

# ixctl control plane, slice 1.
e2e = TASK / "indigo-e2e-results.json"
checks = json.loads(e2e.read_text(encoding="utf-8"))["checks"]
rows.append({
    "id": "ixctl", "item": "ixctl control plane, slice 1", "where": "exp/ixctl",
    **branch("exp/ixctl"),
    "checks_passed": sum(bool(c.get("ok")) for c in checks.values()), "checks_total": len(checks),
    "mutants_red": None, "mutants_total": None,
    "evidence": e2e.name, "evidence_sha256": sha256(e2e)[:12],
    "bundle": None, "bundle_sha256": None, "binary_sha256": None,
    "verdict": "reviewed, findings fixed", "open_findings": None,
    "review": "Codex reviewed; 4 defects fixed",
    "next": "One-way doors unsigned: ixctl.v1 envelope, desired-state document, journal format",
})

(OUT / "indigo_work.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
for r in rows:
    print(r["id"], r["head_sha"], r["pr"], f'{r["checks_passed"]}/{r["checks_total"]}', r["mutants_red"], r["updated_at"])
