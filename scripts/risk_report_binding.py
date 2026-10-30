"""Hold the Agent Blackbox risk report to the changeset it gates (ix#316).

`enforce` reads dist/risk-report.json; people read dist/risk-report.md in the PR comment. On
PR #313 the two described different changesets, and the JSON carried a critical finding on
files the repository never had. This check runs before `enforce` and fails closed unless:

- risk-report.json names exactly the files of dist/changed-files.txt;
- its evidence digests for the changed-file list and the diff are those of the files this job
  collected (dist/changed-files.txt, dist/diff.patch), as supplied, not discovered;
- risk-report.md shows the same digests, so the comment and the gate read one report.

Usage: python scripts/risk_report_binding.py --dist dist
Exit 0 when bound, 1 otherwise, each reason printed as a GitHub error annotation.
Standard library only.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

SHOWN = 10  # files listed per side of a mismatch


def normalize(path: str) -> str:
    """The path as `analyze` reads it (its `normalize_path`)."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def lines(path: Path) -> list[str]:
    return [normalize(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problems(dist: Path) -> list[str]:
    try:
        report = json.loads((dist / "risk-report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"risk-report.json cannot be read: {e}"]
    if not isinstance(report, dict):
        return ["risk-report.json is not a JSON object"]

    out = []
    try:
        changed = lines(dist / "changed-files.txt")
    except OSError as e:
        return [f"changed-files.txt cannot be read: {e}"]
    listed = report.get("changed_files")
    if not isinstance(listed, list) or not all(isinstance(f, str) for f in listed):
        out.append("risk-report.json has no changed_files list")
    else:
        extra = sorted(set(listed) - set(changed))
        missing = sorted(set(changed) - set(listed))
        if extra or missing or len(listed) != len(changed):
            out.append(
                f"risk-report.json lists {len(listed)} changed files, this PR changes "
                f"{len(changed)}; not in the PR: {extra[:SHOWN]}; not in the report: "
                f"{missing[:SHOWN]}"
            )

    artifacts = (report.get("evidence_integrity") or {}).get("artifacts") or []
    by_kind = {a.get("kind"): a for a in artifacts if isinstance(a, dict)}
    try:
        md = (dist / "risk-report.md").read_text(encoding="utf-8")
    except OSError as e:
        md = None
        out.append(f"risk-report.md cannot be read: {e}")
    for kind, name in (("changed_files", "changed-files.txt"), ("diff", "diff.patch")):
        a = by_kind.get(kind)
        if a is None:
            out.append(f"risk-report.json records no {kind} evidence digest")
            continue
        if a.get("status") != "present":
            out.append(
                f"risk-report.json's {kind} evidence is '{a.get('status')}', not the {name} "
                "this job supplied"
            )
            continue
        try:
            want = sha256(dist / name)
        except OSError as e:
            out.append(f"{name} cannot be read: {e}")
            continue
        got = a.get("sha256")
        if got != want:
            out.append(f"risk-report.json's {kind} digest {got} is not that of {name}, {want}")
        if md is not None and f"| {kind} | `present` | `{got}` |" not in md:
            out.append(f"risk-report.md does not show the {kind} digest {got} of risk-report.json")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dist", type=Path, required=True)
    found = problems(ap.parse_args().dist)
    for p in found:
        print(f"::error title=risk-report binding::{p}")
    if found:
        print("The risk report does not describe this pull request; not enforcing it (ix#316).")
        return 1
    print("risk-report.json and risk-report.md describe this pull request's changeset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
