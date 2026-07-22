#!/usr/bin/env python3
"""
ix-fleet claims — validate / append / read the cross-session coordination ledger.

The ledger (``~/.agents/claims.jsonl``) is the GuitarAlchemist ecosystem's
fallback coordination channel for concurrent Claude Code sessions: append-only
JSONL, one claim event per line, **latest line per (repo, lane) wins**. It was
born as a hand-appended file — trivially corruptible by a stray keystroke. This
tool is the ix-owned guardrail + interface for it: the "fleet / handoff layer"
the 2026-07-21 loop-engineering review flagged as ix's one structural gap.

Stdlib-only (no third-party deps) so any session can run it anywhere:

    python fleet/claims.py validate
    python fleet/claims.py status [--open]
    python fleet/claims.py append --repo ix --lane my-lane --status claimed \
        --session ix --evidence PR#123 --note "one line"

Ledger path resolution (in order): ``--file``, ``$AGENTS_CLAIMS_FILE``,
``~/.agents/claims.jsonl``.

Exit codes: 0 ok · 1 validation found bad lines · 2 bad append input · 3 no ledger.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

REQUIRED_FIELDS: Tuple[str, ...] = ("ts", "repo", "lane", "status", "session")
VALID_STATUS: Tuple[str, ...] = ("claimed", "in-progress", "done", "released")
# RFC3339 UTC to the second, e.g. 2026-07-21T23:37:36Z
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_ledger_path() -> Path:
    env = os.environ.get("AGENTS_CLAIMS_FILE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".agents" / "claims.jsonl"


def validate_claim(obj: object) -> List[str]:
    """Return human-readable problems with one claim object (empty list == valid)."""
    if not isinstance(obj, dict):
        return ["not a JSON object"]
    problems: List[str] = []
    for f in REQUIRED_FIELDS:
        if f not in obj:
            problems.append(f"missing required field '{f}'")
    status = obj.get("status")
    if status is not None and status not in VALID_STATUS:
        problems.append(f"status {status!r} not in {VALID_STATUS}")
    ts = obj.get("ts")
    if isinstance(ts, str) and not _TS_RE.match(ts):
        problems.append(f"ts {ts!r} is not RFC3339 UTC (YYYY-MM-DDTHH:MM:SSZ)")
    for f in ("repo", "lane", "session"):
        if f in obj and (not isinstance(obj[f], str) or not obj[f].strip()):
            problems.append(f"{f!r} must be a non-empty string")
    if obj.get("evidence") is not None and not isinstance(obj.get("evidence"), str):
        problems.append("'evidence' must be a string or null")
    return problems


def parse_ledger(path: Path) -> List[Tuple[int, object, List[str]]]:
    """Return [(lineno, obj_or_None, problems)] for every non-blank line."""
    rows: List[Tuple[int, object, List[str]]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            rows.append((lineno, None, [f"invalid JSON: {e}"]))
            continue
        rows.append((lineno, obj, validate_claim(obj)))
    return rows


def latest_by_lane(rows: List[Tuple[int, object, List[str]]]) -> "dict[Tuple[str, str], dict]":
    """Latest valid claim per (repo, lane) — later lines overwrite earlier ones."""
    latest: "dict[Tuple[str, str], dict]" = {}
    for _lineno, obj, problems in rows:
        if problems or not isinstance(obj, dict):
            continue
        latest[(obj["repo"], obj["lane"])] = obj
    return latest


# ── subcommands ───────────────────────────────────────────────────────────────

def cmd_validate(path: Path) -> int:
    rows = parse_ledger(path)
    bad = [(ln, p) for ln, _o, p in rows if p]
    if not bad:
        print(f"ok: {len(rows)} claim(s) valid — {path}")
        return 0
    for ln, problems in bad:
        for p in problems:
            print(f"{path}:{ln}: {p}", file=sys.stderr)
    print(f"FAIL: {len(bad)} bad line(s) of {len(rows)}", file=sys.stderr)
    return 1


def cmd_status(path: Path, only_open: bool) -> int:
    latest = latest_by_lane(parse_ledger(path))
    items = sorted(latest.values(), key=lambda o: (o["repo"], o["lane"]))
    if only_open:
        items = [o for o in items if o["status"] in ("claimed", "in-progress")]
    if not items:
        print("(no claims)" if not only_open else "(no open claims)")
        return 0
    w_repo = max(len(o["repo"]) for o in items)
    w_lane = max(len(o["lane"]) for o in items)
    w_stat = max(len(o["status"]) for o in items)
    for o in items:
        ev = o.get("evidence") or ""
        print(f"{o['repo']:<{w_repo}}  {o['lane']:<{w_lane}}  "
              f"{o['status']:<{w_stat}}  {o['session']:<8}  {ev}")
    return 0


def cmd_append(path: Path, args: argparse.Namespace) -> int:
    claim = {
        "ts": now_rfc3339(),
        "repo": args.repo,
        "lane": args.lane,
        "status": args.status,
        "session": args.session,
        "evidence": args.evidence,  # may be None
        "note": args.note or "",
    }
    problems = validate_claim(claim)
    if problems:
        for p in problems:
            print(f"refusing to append: {p}", file=sys.stderr)
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print(json.dumps(claim, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    # --file is accepted both before AND after the subcommand (natural either way).
    # SUPPRESS keeps an unprovided --file from clobbering a value given on the
    # other side of the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--file", type=Path, default=argparse.SUPPRESS,
                        help="ledger path (default: $AGENTS_CLAIMS_FILE or ~/.agents/claims.jsonl)")

    p = argparse.ArgumentParser(prog="claims", parents=[common],
                                description=__doc__.strip().splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", parents=[common], help="check every line against the schema")

    ps = sub.add_parser("status", parents=[common], help="print the current owner per (repo, lane)")
    ps.add_argument("--open", action="store_true", dest="only_open",
                    help="only claimed / in-progress lanes")

    pa = sub.add_parser("append", parents=[common], help="write one well-formed claim (stamps ts)")
    pa.add_argument("--repo", required=True)
    pa.add_argument("--lane", required=True)
    pa.add_argument("--status", required=True, choices=VALID_STATUS)
    pa.add_argument("--session", required=True)
    pa.add_argument("--evidence", default=None)
    pa.add_argument("--note", default=None)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    file_arg = getattr(args, "file", None)
    path = file_arg.expanduser() if file_arg else default_ledger_path()
    if args.cmd in ("validate", "status") and not path.exists():
        print(f"no ledger at {path}", file=sys.stderr)
        return 3
    if args.cmd == "validate":
        return cmd_validate(path)
    if args.cmd == "status":
        return cmd_status(path, args.only_open)
    if args.cmd == "append":
        return cmd_append(path, args)
    return 2  # unreachable (subparser required)


if __name__ == "__main__":
    raise SystemExit(main())
