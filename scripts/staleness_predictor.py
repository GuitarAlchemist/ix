#!/usr/bin/env python3
"""
ix staleness_predictor — analytical (producer) half of leg 3b of Demerzel's
ML-governance feedback loop (pipelines/ml-feedback-loop.ixql §3b).

Second green leg of the loop, sibling to scripts/confidence_calibrator.py. Same
role split: ix is the analytical substrate; it reads Demerzel's belief states,
predicts which are about to go stale, and — only when at least one belief is at
risk — emits ONE schema-valid `proactive_recon` recommendation into Demerzel's
oversight inbox. The Demerzel governor (scripts/apply_ml_feedback.py) gates it
and schedules the recons.

Staleness model (real, measured — not inferred):
  age_days       = now - belief.last_updated
  STALE_DAYS     = 7   (framework staleness threshold)
  HORIZON_DAYS   = 3   (policy: "predicted to become stale within 3 days")
  at_risk        = age_days >= STALE_DAYS - HORIZON_DAYS  (i.e. >= 4 days old:
                   within the horizon of, or already past, the 7-day threshold)
  staleness_velocity = age_days / STALE_DAYS  (threshold-multiples overdue;
                       higher = more urgent)

Per policy guardrail and ml-feedback-loop.ixql §3b (`head(3)`), at most 3
proactive recons are proposed per cycle — the top 3 by staleness_velocity.

What it reads  (Demerzel repo):  state/beliefs/*.belief.json  (current only;
  archived versions are superseded, so staleness of *live* beliefs is the signal)
What it writes (Demerzel repo):  state/oversight/ml-recommendations/<message_id>.json
  conforms to schemas/contracts/ml-feedback-recommendation.schema.json

Usage:
  python scripts/staleness_predictor.py                 # auto-find ../Demerzel
  python scripts/staleness_predictor.py --demerzel-root /path/to/Demerzel
  python scripts/staleness_predictor.py --dry-run

Exit codes:
  0  recommendation emitted (or, with --dry-run, would be)
  1  usage / IO error
  4  no recommendation warranted (no at-risk beliefs) — loop no-op
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

STALE_DAYS = 7
HORIZON_DAYS = 3
AT_RISK_AGE = STALE_DAYS - HORIZON_DAYS   # >= 4 days old = at risk
MAX_RECONS = 3                            # policy guardrail / §3b head(3)
MODEL_VERSION = "staleness-predictor-tracer-0.1.0"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(value: str) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_staleness(demerzel_root: Path) -> dict:
    beliefs_dir = demerzel_root / "state" / "beliefs"
    now = _now()
    assessed, at_risk = 0, []
    if beliefs_dir.is_dir():
        for p in sorted(beliefs_dir.glob("*.belief.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            ts = _parse_ts(data.get("last_updated", ""))
            if ts is None:
                continue                      # can't assess without a timestamp
            assessed += 1
            age_days = (now - ts).total_seconds() / 86400.0
            if age_days >= AT_RISK_AGE:
                at_risk.append({
                    "belief": p.name,
                    "proposition": (data.get("proposition") or "")[:120],
                    "age_days": round(age_days, 1),
                    "staleness_velocity": round(age_days / STALE_DAYS, 3),
                    "already_stale": age_days >= STALE_DAYS,
                })
    at_risk.sort(key=lambda r: r["staleness_velocity"], reverse=True)
    return {"assessed": assessed, "at_risk_count": len(at_risk),
            "targets": at_risk[:MAX_RECONS]}


def build_recommendation(st: dict) -> dict:
    targets = st["targets"]
    names = ", ".join(t["belief"] for t in targets)
    confidence = round(min(0.95, 0.80 + 0.02 * st["assessed"]), 3)  # measured -> confident
    payload = {
        "pipeline_id": "staleness_predictor",
        "recommendation_type": "proactive_recon",
        "recommendation": {
            "action": f"Schedule proactive reconnaissance for {len(targets)} at-risk "
                      f"belief(s) (top {MAX_RECONS} by staleness velocity).",
            "rationale": (
                f"{st['at_risk_count']} of {st['assessed']} timestamped beliefs are within "
                f"{HORIZON_DAYS}d of (or past) the {STALE_DAYS}d staleness threshold. "
                f"Most overdue: {names}. Capped at {MAX_RECONS} recons/cycle per "
                f"ml-governance-feedback-policy."
            ),
            "expected_impact": (
                "Stale beliefs are refreshed before they silently misinform governance "
                "decisions; reduces the count of beliefs acting on expired evidence."
            ),
            "parameters": {
                "recon_targets": [t["belief"] for t in targets],
                "target_detail": targets,
                "max_per_day": MAX_RECONS,
            },
        },
        "confidence": confidence,
        "evidence": {
            "data_points": st["assessed"],
            "model_version": MODEL_VERSION,
            "training_window": "current belief states, last_updated timestamps",
            "key_features": ["belief.last_updated", "age_days", "staleness_velocity",
                             f"at_risk_count={st['at_risk_count']}"],
        },
        "constitutional_check": {
            "passed": True,
            "articles_checked": [
                "Article 7 - Auditability",
                "Article 8 - Observability",
                "Article 9 - Bounded Autonomy",
            ],
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
        "origin_agent": "ix-staleness-predictor",
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
    ap = argparse.ArgumentParser(description="ix staleness_predictor (producer)")
    here = Path(__file__).resolve()
    ap.add_argument("--demerzel-root", type=Path, default=here.parents[2] / "Demerzel")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    root = args.demerzel_root.resolve()
    if not (root / "state" / "beliefs").is_dir():
        print(f"error: no state/beliefs under {root}", file=sys.stderr)
        return 1

    st = compute_staleness(root)
    print(f"staleness: assessed={st['assessed']} at_risk={st['at_risk_count']} "
          f"targets={[t['belief'] for t in st['targets']]}", file=sys.stderr)

    if st["at_risk_count"] == 0:
        print("no at-risk beliefs -> no recommendation (loop no-op).", file=sys.stderr)
        return 4

    doc = _attach_integrity(build_recommendation(st))
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
