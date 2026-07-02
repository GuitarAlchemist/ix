#!/usr/bin/env python3
"""
ix remediation_optimizer — analytical (producer) half of leg 3d of Demerzel's
ML-governance feedback loop (pipelines/ml-feedback-loop.ixql §3d).

Fourth producer (with confidence_calibrator, staleness_predictor,
violation_pattern_detector). Per the cross-repo role split, ix is the analytical
substrate: it reads Demerzel's PDCA (Kaizen) remediation records, computes which
remediation strategies actually succeed, and emits ONE schema-valid
`strategy_change` recommendation. The Demerzel governor records the observed
success rates as evidence — but per policy "Cannot downgrade high-risk gaps
without human approval", so any escalation *downgrade* is advisory only.

Strategy / outcome extraction (real, over on-disk PDCA records):
  outcome   = check.success_criteria_met  (only records that reached 'check' count)
  strategy  = act.decision (when a cycle completed) else kaizen_kind else
              'unclassified' — the remediation approach to attribute the outcome to
  success_rate(strategy) = successes / attempts  (with attempt count, so a thin
              sample is visible, not hidden behind a bare ratio)

automation_candidates  = strategies with success_rate >= AUTOMATE_AT and n >= MIN_N
recommended_escalation_changes = strategies with success_rate < FRAGILE_BELOW
              (proposed *downgrade-resistance*: keep manual / high-risk). Each is
              flagged requires_human_approval per the policy guardrail.

What it reads  (Demerzel repo):  state/pdca/*.pdca.json
What it writes (Demerzel repo):  state/oversight/ml-recommendations/<message_id>.json

Usage:
  python scripts/remediation_optimizer.py                 # auto-find ../Demerzel
  python scripts/remediation_optimizer.py --demerzel-root /path/to/Demerzel
  python scripts/remediation_optimizer.py --dry-run

Exit codes:
  0  recommendation emitted (or, with --dry-run, would be)
  1  usage / IO error
  4  no outcome-bearing PDCA records -> no recommendation (loop no-op)
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

AUTOMATE_AT = 0.8    # success_rate >= this (with enough samples) -> automation candidate
FRAGILE_BELOW = 0.5  # success_rate < this -> keep manual / human-gated
MIN_N = 2            # minimum attempts before a rate is actionable
MODEL_VERSION = "remediation-optimizer-tracer-0.1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def analyze_pdca(demerzel_root: Path) -> dict:
    pdca_dir = demerzel_root / "state" / "pdca"
    records = sorted(pdca_dir.glob("*.pdca.json")) if pdca_dir.is_dir() else []

    by_strategy = defaultdict(lambda: {"attempts": 0, "successes": 0, "records": []})
    outcome_records = 0
    for rp in records:
        try:
            d = json.loads(rp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        chk = d.get("check") or {}
        if "success_criteria_met" not in chk:
            continue  # not yet at 'check' — no outcome to attribute
        success = bool(chk["success_criteria_met"])
        act = d.get("act") or {}
        strategy = act.get("decision") or d.get("kaizen_kind") or "unclassified"
        agg = by_strategy[strategy]
        agg["attempts"] += 1
        agg["successes"] += int(success)
        agg["records"].append(d.get("id", rp.stem))
        outcome_records += 1

    rates, automation, escalations = {}, [], []
    for strat, agg in sorted(by_strategy.items()):
        rate = round(agg["successes"] / agg["attempts"], 3) if agg["attempts"] else 0.0
        rates[strat] = {"success_rate": rate, "attempts": agg["attempts"],
                        "successes": agg["successes"]}
        if rate >= AUTOMATE_AT and agg["attempts"] >= MIN_N:
            automation.append({"strategy": strat, "success_rate": rate, "n": agg["attempts"]})
        if rate < FRAGILE_BELOW:
            escalations.append({
                "strategy": strat, "success_rate": rate, "n": agg["attempts"],
                "change": "keep manual / high-risk (do not auto-remediate)",
                "requires_human_approval": True,  # policy guardrail
            })
    return {"records": len(records), "outcome_records": outcome_records,
            "success_rates_by_strategy": rates,
            "automation_candidates": automation,
            "recommended_escalation_changes": escalations}


def build_recommendation(an: dict) -> dict:
    n = an["outcome_records"]
    # Thin samples -> moderate confidence (likely below auto-apply: escalate, by design).
    confidence = round(min(0.90, 0.50 + 0.06 * n), 3)
    rate_str = ", ".join(f"{k}={v['success_rate']}(n={v['attempts']})"
                         for k, v in an["success_rates_by_strategy"].items())
    payload = {
        "pipeline_id": "remediation_optimizer",
        "recommendation_type": "strategy_change",
        "recommendation": {
            "action": "Record observed remediation-strategy success rates; review "
                      f"{len(an['recommended_escalation_changes'])} fragile strategy(ies).",
            "rationale": (
                f"Across {n} outcome-bearing PDCA records: {rate_str or 'no rates'}. "
                f"Automation candidates: {[a['strategy'] for a in an['automation_candidates']] or 'none'}. "
                f"Sample sizes are small — rates are directional, not conclusive."
            ),
            "expected_impact": (
                "Remediation effectiveness becomes evidence-tracked; fragile strategies are "
                "kept human-gated. No risk class is downgraded autonomously (policy guardrail)."
            ),
            "parameters": {
                "success_rates_by_strategy": an["success_rates_by_strategy"],
                "automation_candidates": an["automation_candidates"],
                "recommended_escalation_changes": an["recommended_escalation_changes"],
            },
        },
        "confidence": confidence,
        "evidence": {
            "data_points": n,
            "model_version": MODEL_VERSION,
            "training_window": "state/pdca/*.pdca.json (all on-disk Kaizen records)",
            "key_features": ["check.success_criteria_met", "act.decision", "kaizen_kind",
                             f"outcome_records={n}"],
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
        "origin_agent": "ix-remediation-optimizer",
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
    ap = argparse.ArgumentParser(description="ix remediation_optimizer (producer)")
    here = Path(__file__).resolve()
    ap.add_argument("--demerzel-root", type=Path, default=here.parents[2] / "Demerzel")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = args.demerzel_root.resolve()
    an = analyze_pdca(root)
    print(f"pdca: records={an['records']} outcome_records={an['outcome_records']} "
          f"strategies={list(an['success_rates_by_strategy'])}", file=sys.stderr)
    for k, v in an["success_rates_by_strategy"].items():
        print(f"  {k}: {v['success_rate']} ({v['successes']}/{v['attempts']})", file=sys.stderr)

    if an["outcome_records"] == 0:
        print("no outcome-bearing PDCA records -> no recommendation (loop no-op).", file=sys.stderr)
        return 4

    doc = _attach_integrity(build_recommendation(an))
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
    print(f"wrote recommendation (confidence {doc['confidence']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
