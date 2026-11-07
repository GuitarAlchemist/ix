"""Write the dashboard's store documents as local JSON files, one per document, for one batch write."""
import json, sys
from pathlib import Path

HERE = Path(sys.argv[1])
DOCS = HERE / "docs"
DOCS.mkdir(exist_ok=True)
READ_AT = json.loads((HERE / "data" / "surface.json").read_text(encoding="utf-8"))[0]["read_at"]
UPDATED = {"at": READ_AT, "by": "Claude"}
SCHEDULE = "Refreshed by Claude on request, from gh and origin/main"

BLOBS = {
    "open_prs": ("d1625c9ee6106ba2437ba744862128d9", "Open pull requests",
                 "Every open pull request on GuitarAlchemist/ix with its CI rollup, failing checks, unresolved Codex "
                 "P0/P1 review threads and merge state. Read from GitHub's GraphQL API through gh."),
    "merged_prs": ("86e7d50ffbab8b76e1954f63955a88f6", "Merged pull requests",
                   "Pull requests merged into GuitarAlchemist/ix in the 120 days before the snapshot. Read through gh."),
    "main_runs": ("31080bf054d0a132ba041ba844cf5164", "Workflow runs on main",
                  "The latest 150 GitHub Actions runs on the main branch of GuitarAlchemist/ix, with each run's "
                  "result. Read through gh."),
    "open_issues": ("4eb83540e0d0f09100180273d69f48a2", "Open issues",
                    "Every open issue on GuitarAlchemist/ix with its labels. Read through gh."),
    "crate_tiers": ("4e146a6147041e11cdf1ee5c1c86c0ba", "Crate maturity tiers",
                    "Each crate's tier as crate-maturity.toml on origin/main states it."),
    "surface": ("5627f82f3f862c8813aa996362eeb07a", "Registered surface on main",
                "Registry skills and MCP tools named in state/registry/skills.snapshot.json, and the workspace "
                "members in Cargo.toml, on origin/main, with the commit read."),
}


def write(name, doc):
    (DOCS / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


write("meta", {"title": "IX Control Room"})
for ds, (blob, title, description) in BLOBS.items():
    write(f"dataset-{ds}", {
        "title": title, "description": description,
        "source": {"kind": "file", "url": f"/_blob/{blob}", "name": f"{ds}.json", "format": "json"},
        "updated": UPDATED, "schedule": SCHEDULE, "staleAfterHours": 48,
    })
write("index", {"text": (HERE / "index.html").read_text(encoding="utf-8")})
print(sorted(p.name for p in DOCS.iterdir()), len((HERE / "index.html").read_bytes()), "bytes of page")
