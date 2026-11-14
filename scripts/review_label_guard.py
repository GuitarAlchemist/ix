"""Tell whether a push changed the protected paths a PR's review label attests (retro 2026-10-22).

`agent-blackbox-reviewed` says a person reviewed the PR's changes to the paths
agent-blackbox.policy.json lists as blocked or one-way. It stayed on through later pushes. This
compares those paths' content at the PR's head before and after a push. The paths are the ones
the PR changes against the base at either head, matched as `analyze` matches them (fnmatch on
its `normalize_path`), under the BASE's policy, so a PR cannot exempt its own paths.

Usage: python scripts/review_label_guard.py --base <sha> --before <sha> --after <sha>
Exit 0 when the protected content is as it was, 1 when the push changed it (each path printed),
3 when the two heads cannot be compared (the old head was force-pushed away, no policy at the
base). Standard library only; runs git in the current directory.
"""

import argparse
import fnmatch
import json
import subprocess
import sys

POLICY = "agent-blackbox.policy.json"
CANNOT = 3  # not 2: Python exits 2 itself on a missing script or a bad argument


class Cannot(Exception):
    pass


def normalize(path: str) -> str:
    """The path as `analyze` reads it (its `normalize_path`)."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True)


def has_commit(sha: str) -> bool:
    return git("cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def ensure_commit(sha: str) -> None:
    if not has_commit(sha):
        git("fetch", "--no-tags", "--quiet", "origin", sha)
    if not has_commit(sha):
        raise Cannot(f"commit {sha} is not in this clone and cannot be fetched")


def patterns(base: str) -> list[str]:
    out = git("show", f"{base}:{POLICY}")
    if out.returncode != 0:
        raise Cannot(f"no {POLICY} at the base {base}")
    try:
        policy = json.loads(out.stdout.decode("utf-8-sig"))
    except ValueError as e:
        raise Cannot(f"{POLICY} at the base {base} is not JSON: {e}") from e
    found = [
        p
        for key in ("blocked_paths", "one_way_door_paths")
        for p in policy.get(key) or []
        if isinstance(p, str)
    ]
    if not found:
        raise Cannot(f"{POLICY} at the base {base} lists no protected paths")
    return found


def protected_changes(base: str, head: str, pats: list[str]) -> set[str]:
    mb = git("merge-base", base, head)
    if mb.returncode != 0:
        raise Cannot(f"{head} shares no history with the base {base}")
    out = git("diff", "--name-only", "-z", "--no-renames", mb.stdout.decode().strip(), head)
    if out.returncode != 0:
        raise Cannot(f"git diff of {head} failed: {out.stderr.decode(errors='replace').strip()}")
    files = [normalize(f) for f in out.stdout.decode("utf-8", "surrogateescape").split("\0") if f]
    return {f for f in files if any(fnmatch.fnmatch(f, p) for p in pats)}


def blob(head: str, path: str) -> str | None:
    out = git("rev-parse", "--verify", "--quiet", f"{head}:{path}")
    return out.stdout.decode().strip() if out.returncode == 0 else None


def changed(base: str, before: str, after: str) -> list[str]:
    for sha in (base, before, after):
        ensure_commit(sha)
    pats = patterns(base)
    paths = protected_changes(base, before, pats) | protected_changes(base, after, pats)
    return sorted(p for p in paths if blob(before, p) != blob(after, p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("--base", "--before", "--after"):
        ap.add_argument(name, required=True)
    args = ap.parse_args()
    try:
        paths = changed(args.base, args.before, args.after)
    except Cannot as e:
        print(f"::warning title=review label::cannot compare the protected paths across this push: {e}")
        return CANNOT
    for p in paths:
        print(f"changed by this push: {p}")
    if paths:
        return 1
    print("This push leaves the PR's protected paths as they were reviewed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
