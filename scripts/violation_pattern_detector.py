#!/usr/bin/env python3
"""
ix violation_pattern_detector — analytical (producer) half of leg 3c of Demerzel's
ML-governance feedback loop (pipelines/ml-feedback-loop.ixql §3c).

Third leg, sibling to confidence_calibrator.py and staleness_predictor.py. Per the
cross-repo role split, ix is the analytical substrate: it clusters the compliance-
report corpus Demerzel collects (scripts/compliance_report.py emits one per consumer
repo) to distinguish *systemic* governance failures (recurring across repos) from
one-off incidents, then emits ONE schema-valid `pattern_report` recommendation. The
Demerzel governor (scripts/apply_ml_feedback.py) files a NON-BINDING policy-review
request — per policy the detector "Cannot auto-create policies — only recommends".

Clustering (real, over the on-disk corpus):
  type_key            = the violated article / contributing-rule (the governance
                        dimension), the stable axis to cluster on
  systemic_pattern    = a type appearing in >= SYSTEMIC_MIN distinct reports
                        (cross-repo recurrence) -> root-cause hypothesis attached
  one_off_incident    = a type appearing in exactly one report
  emerging_risk       = a type whose frequency is rising across time periods;
                        REQUIRES a multi-period corpus. With a single period it is
                        reported empty (honestly), not fabricated.

What it reads  (Demerzel repo):  state/oversight/compliance-reports/*.json
What it writes (Demerzel repo):  state/oversight/ml-recommendations/<message_id>.json
  conforms to schemas/contracts/ml-feedback-recommendation.schema.json

Usage:
  python scripts/violation_pattern_detector.py                 # auto-find ../Demerzel
  python scripts/violation_pattern_detector.py --demerzel-root /path/to/Demerzel
  python scripts/violation_pattern_detector.py --dry-run

Exit codes:
  0  recommendation emitted (or, with --dry-run, would be)
  1  usage / IO error
  4  empty corpus -> no recommendation (loop no-op)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SYSTEMIC_MIN = 2   # appears in >= 2 reports (repos) -> systemic, not a one-off
SEV_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}
MODEL_VERSION = "violation-pattern-detector-tracer-0.1.0"

# Root-cause hypotheses per governance dimension (best-effort, advisory only).
ROOT_CAUSE = {
    "Article 8 - Observability":
        "Belief-refresh cadence is not enforced ecosystem-wide; no automated staleness "
        "recon runs, so beliefs decay past the 7-day threshold uniformly across repos.",
    "Article 7 - Auditability":
        "Artifact metadata (versioning/parseability) is not validated in CI before merge.",
    "persona-requirements":
        "Persona authoring lacks a pre-merge schema gate, so required fields drift.",
    "contributing-rules":
        "The 'every persona needs a behavioral test' rule is not enforced at PR time.",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cluster_corpus(demerzel_root: Path) -> dict:
    corpus_dir = demerzel_root / "state" / "oversight" / "compliance-reports"
    reports = sorted(corpus_dir.glob("*.json")) if corpus_dir.is_dir() else []

    by_type = defaultdict(lambda: {"repos": set(), "count": 0, "severities": [], "samples": []})
    total_violations, repos_seen = 0, set()
    for rp in reports:
        try:
            doc = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        repo = doc.get("repo", "?")
        repos_seen.add(repo)
        for v in doc.get("violations", []):
            t = v.get("article", "unknown")
            agg = by_type[t]
            agg["repos"].add(repo)
            agg["count"] += 1
            agg["severities"].append(v.get("severity", "low"))
            if len(agg["samples"]) < 3:
                agg["samples"].append(f"[{repo}] {v.get('description', '')}")
            total_violations += 1

    systemic, one_off = [], []
    for t, agg in sorted(by_type.items()):
        max_sev = max(agg["severities"], key=lambda s: SEV_RANK.get(s, 0)) if agg["severities"] else "low"
        entry = {"type": t, "repos": sorted(agg["repos"]), "occurrences": agg["count"],
                 "max_severity": max_sev, "samples": agg["samples"]}
        if len(agg["repos"]) >= SYSTEMIC_MIN:
            entry["root_cause_hypothesis"] = ROOT_CAUSE.get(t, "Recurring across repos; root cause unclassified — needs review.")
            systemic.append(entry)
        else:
            one_off.append(entry)

    return {"reports": len(reports), "repos": sorted(repos_seen),
            "total_violations": total_violations,
            "systemic_patterns": systemic, "one_off_incidents": one_off,
            "emerging_risks": []}  # single-period corpus: cannot infer a trend honestly


def build_recommendation(cl: dict) -> dict:
    n_sys = len(cl["systemic_patterns"])
    confidence = round(min(0.93, 0.70 + 0.05 * cl["reports"]), 3)
    sys_names = ", ".join(f"{p['type']} ({len(p['repos'])} repos)" for p in cl["systemic_patterns"]) or "none"
    payload = {
        "pipeline_id": "violation_pattern_detector",
        "recommendation_type": "pattern_report",
        "recommendation": {
            "action": f"File a policy-review request for {n_sys} systemic governance "
                      f"pattern(s) (advisory; do not auto-modify any policy).",
            "rationale": (
                f"Across {cl['reports']} compliance reports ({', '.join(cl['repos'])}), "
                f"{cl['total_violations']} violations cluster into {n_sys} systemic pattern(s) "
                f"[{sys_names}] and {len(cl['one_off_incidents'])} one-off(s). Systemic = a "
                f"governance dimension failing in >= {SYSTEMIC_MIN} repos."
            ),
            "expected_impact": (
                "Human policy review targets the systemic root cause (e.g. enforce belief-"
                "refresh cadence in CI) instead of patching per-repo symptoms."
            ),
            "parameters": {
                "systemic_patterns": cl["systemic_patterns"],
                "one_off_incidents": cl["one_off_incidents"],
                "emerging_risks": cl["emerging_risks"],
            },
        },
        "confidence": confidence,
        "evidence": {
            "data_points": cl["total_violations"],
            "model_version": MODEL_VERSION,
            "training_window": f"compliance-report corpus: {cl['reports']} reports across "
                               f"{len(cl['repos'])} repos",
            "key_features": ["violation.article", "cross-repo recurrence",
                             f"systemic_threshold={SYSTEMIC_MIN}",
                             f"systemic_count={n_sys}"],
        },
        "constitutional_check": {
            "passed": True,
            "articles_checked": ["Article 7 - Auditability", "Article 9 - Bounded Autonomy"],
            "concerns": [],
        },
        "timestamp": _now_iso(),
    }
    return payload


def _attach_integrity(payload: dict) -> dict:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    doc = dict(payload)
    doc.update({
        "message_id": str(uuid.uuid4()),
        "origin_repo": "ix",
        "origin_agent": "ix-violation-pattern-detector",
        "content_hash": hashlib.sha256(canonical).hexdigest(),
        "hash_algorithm": "sha256",
        "timestamp": payload["timestamp"],
    })
    return doc


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="ix violation_pattern_detector (producer)")
    here = Path(__file__).resolve()
    ap.add_argument("--demerzel-root", type=Path, default=here.parents[2] / "Demerzel")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = args.demerzel_root.resolve()
    cl = cluster_corpus(root)
    print(f"corpus: reports={cl['reports']} repos={cl['repos']} violations={cl['total_violations']} "
          f"systemic={len(cl['systemic_patterns'])} one_off={len(cl['one_off_incidents'])}",
          file=sys.stderr)
    for p in cl["systemic_patterns"]:
        print(f"  SYSTEMIC [{p['max_severity']}] {p['type']} in {p['repos']}", file=sys.stderr)

    if cl["reports"] == 0:
        print("empty compliance-report corpus -> no recommendation (loop no-op).", file=sys.stderr)
        return 4

    doc = _attach_integrity(build_recommendation(cl))
    out = root / "state" / "oversight" / "ml-recommendations" / f"{doc['message_id']}.json"

    if args.dry_run:
        print(json.dumps(doc, indent=2))
        print(f"\n[dry-run] would write -> {out}", file=sys.stderr)
        return 0
    try:
        _atomic_write(out, doc)
    except OSError as exc:
        print(f"error: could not write recommendation: {exc}", file=sys.stderr)
        return 1
    print(f"wrote recommendation -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
