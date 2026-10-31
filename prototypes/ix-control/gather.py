"""Snapshot of GuitarAlchemist/ix for the IX Control Room dashboard: one JSON file of row objects per dataset.

Reads GitHub through `gh` and the repo's origin/main through git. Facts only: the page derives stages,
counts and weeks itself, as calculations its readers can inspect."""
import json, os, re, subprocess, sys, tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = "GuitarAlchemist/ix"
CLONE = os.environ.get("IX_CLONE") or os.path.expanduser(os.path.join("~", "source", "repos", "ix"))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
NOW = datetime.now(timezone.utc)


def run(args):
    out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if out.returncode:
        raise RuntimeError(f"{args[:3]}: {out.stderr.strip()}")
    return out.stdout


def gh_json(args):
    return json.loads(run(["gh", *args]))


def save(name, rows):
    (OUT / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {len(rows)} rows")


# Open pull requests: CI, failing checks, unresolved Codex P0/P1 threads, merge state.
Q = """query{repository(owner:"GuitarAlchemist",name:"ix"){pullRequests(states:OPEN,first:100){nodes{
number title url isDraft createdAt updatedAt baseRefName headRefOid mergeable mergeStateStatus reviewDecision
closingIssuesReferences(first:5){nodes{number}}
commits(last:1){nodes{commit{statusCheckRollup{state contexts(first:100){nodes{
 ... on CheckRun{name conclusion status startedAt} ... on StatusContext{context state createdAt}}}}}}}
reviewThreads(first:100){nodes{isResolved comments(first:1){nodes{author{login} body}}}}}}}}"""
prs = []
for p in gh_json(["api", "graphql", "-f", f"query={Q}"])["data"]["repository"]["pullRequests"]["nodes"]:
    roll = p["commits"]["nodes"][0]["commit"]["statusCheckRollup"]
    failing, pending = [], 0
    # A re-run check keeps its earlier attempts in the rollup: only each name's latest attempt counts.
    latest = {}
    for c in (roll["contexts"]["nodes"] if roll else []):
        name = c.get("name") or c.get("context")
        when = c.get("startedAt") or c.get("createdAt") or ""
        if name not in latest or when >= (latest[name].get("startedAt") or latest[name].get("createdAt") or ""):
            latest[name] = c
    for c in latest.values():
        concl = c.get("conclusion") or c.get("state")
        if concl in ("FAILURE", "ERROR", "TIMED_OUT", "CANCELLED", "ACTION_REQUIRED", "STARTUP_FAILURE"):
            failing.append(c.get("name") or c.get("context"))
        elif c.get("status") in ("IN_PROGRESS", "QUEUED", "PENDING", "WAITING") or concl in ("PENDING", "EXPECTED"):
            pending += 1
    p01 = 0
    for t in p["reviewThreads"]["nodes"]:
        first = t["comments"]["nodes"][0] if t["comments"]["nodes"] else None
        if not t["isResolved"] and first and (first["author"] or {}).get("login") == "chatgpt-codex-connector":
            m = re.search(r"(P[0-3]) Badge", first["body"])
            p01 += bool(m and m.group(1) in ("P0", "P1"))
    prs.append({
        "number": p["number"], "title": p["title"], "url": p["url"], "draft": p["isDraft"],
        "created_at": p["createdAt"], "updated_at": p["updatedAt"], "base": p["baseRefName"],
        "head_sha": p["headRefOid"][:7], "mergeable": p["mergeable"], "merge_state": p["mergeStateStatus"],
        "ci": roll["state"] if roll else "NONE", "failing_checks": ", ".join(sorted(set(failing))),
        "pending_checks": pending, "codex_p0_p1_open": p01,
        "closes": ", ".join(f"#{n['number']}" for n in p["closingIssuesReferences"]["nodes"]),
    })
save("open_prs", sorted(prs, key=lambda r: r["number"]))

# Pull requests merged in the last 120 days.
since = (NOW - timedelta(days=120)).strftime("%Y-%m-%d")
merged = gh_json(["pr", "list", "-R", REPO, "--state", "merged", "--limit", "400", "--search", f"merged:>={since}",
                  "--json", "number,title,mergedAt,url"])
save("merged_prs", sorted(({"number": m["number"], "title": m["title"], "merged_at": m["mergedAt"], "url": m["url"]}
                           for m in merged), key=lambda r: r["merged_at"]))

# Workflow runs on main, the latest 150.
runs = gh_json(["run", "list", "-R", REPO, "--branch", "main", "--limit", "150",
                "--json", "databaseId,workflowName,event,status,conclusion,createdAt,url"])
save("main_runs", [{"run_id": r["databaseId"], "workflow": r["workflowName"], "event": r["event"],
                    "result": r["conclusion"] or r["status"], "created_at": r["createdAt"], "url": r["url"]}
                   for r in runs])

# Open issues.
issues = gh_json(["issue", "list", "-R", REPO, "--state", "open", "--limit", "500",
                  "--json", "number,title,labels,createdAt,updatedAt,url"])
save("open_issues", sorted(({"number": i["number"], "title": i["title"],
                             "labels": ", ".join(sorted(l["name"] for l in i["labels"])) or "unlabelled",
                             "created_at": i["createdAt"], "updated_at": i["updatedAt"], "url": i["url"]}
                            for i in issues), key=lambda r: r["number"]))

# origin/main: crate tiers and the registered surface.
main_sha = run(["git", "-C", CLONE, "rev-parse", "--short", "origin/main"]).strip()
tiers = tomllib.loads(run(["git", "-C", CLONE, "show", "origin/main:crate-maturity.toml"]))["crates"]
save("crate_tiers", [{"crate": c, "tier": t} for c, t in sorted(tiers.items())])
snap = json.loads(run(["git", "-C", CLONE, "show", "origin/main:state/registry/skills.snapshot.json"]))
members = tomllib.loads(run(["git", "-C", CLONE, "show", "origin/main:Cargo.toml"]))["workspace"]["members"]
save("surface", [{"commit": main_sha, "registry_skills": len(snap["skills"]["names"]),
                  "mcp_tools": len(snap["mcp_tools"]["names"]), "workspace_members": len(members),
                  "read_at": NOW.strftime("%Y-%m-%dT%H:%M:%SZ")}])
