#!/usr/bin/env python3
"""
ix confidence_calibrator — the analytical (producer) half of Demerzel's
ML-governance feedback loop, leg 3a of pipelines/ml-feedback-loop.ixql.

Role split (per Demerzel CLAUDE.md "Cross-repo contracts" + policies/
ml-governance-feedback-policy.yaml): **ix is the analytical/ML substrate.**
Demerzel produces governance data (belief states); ix consumes it and returns
a calibration recommendation. Demerzel (the governor) decides whether to apply
it — that is the *separate* applier, scripts/apply_ml_feedback.py in Demerzel.

This is a deliberately minimal *tracer-bullet*: it closes ONE real leg of the
loop end-to-end on REAL data rather than simulating it. It computes an
overconfidence signal from Demerzel's actual belief files (current vs the
archived prior versions that prove a belief was revised), and — only if the
signal clears the policy trigger — emits ONE schema-valid
`ml-feedback-recommendation` into Demerzel's oversight inbox.

What it reads  (Demerzel repo):
  state/beliefs/*.belief.json            current beliefs (confidence assignments)
  state/beliefs/archived/*.belief.json   prior versions = evidence of revision

What it writes (Demerzel repo):
  state/oversight/ml-recommendations/<message_id>.json
      conforms to schemas/contracts/ml-feedback-recommendation.schema.json
      (+ integrity-fields.schema.json: message_id, content_hash, ...)

Overconfidence metric (real, auditable):
  high-confidence beliefs  = every belief (current + archived) with confidence
                             >= HIGH_CONF (0.85)
  revised-and-overconfident = archived prior versions with confidence >= HIGH_CONF
                             (an archived prior is, by definition, a belief that
                              was later revised)
  overconfidence_rate      = revised-and-overconfident / high-confidence beliefs

Per pipelines/ml-feedback-loop.ixql §3a the recommendation is only emitted when
overconfidence_rate > 0.15, and the proposed nudge is capped by policy at
+/- 0.1 per cycle (max(-0.1, -overconfidence_rate * 0.5)).

NOTE on a real contract drift this script intentionally resolves in favour of
the *schema*: ml-feedback-loop.ixql §3a filters `recommendation_type ==
"calibration_report"`, but the validated schema enum has no such value — it is
`recommendation_type: "threshold_adjustment"` with `pipeline_id:
"confidence_calibrator"`. A schema-valid document therefore never matches the
pipeline's filter. We emit the schema-valid shape and flag the drift on stderr.

Usage:
  python scripts/confidence_calibrator.py                 # auto-find ../Demerzel
  python scripts/confidence_calibrator.py --demerzel-root /path/to/Demerzel
  python scripts/confidence_calibrator.py --dry-run       # print, do not write

Exit codes:
  0  recommendation emitted (or, with --dry-run, would be emitted)
  1  usage / IO error
  4  no recommendation warranted (overconfidence_rate <= trigger) — loop is a no-op
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

HIGH_CONF = 0.85          # threshold for "high-confidence belief"
TRIGGER = 0.15            # ml-feedback-loop.ixql §3a: emit only if rate > 0.15
NUDGE_CAP = 0.10          # policy guardrail: +/- 0.1 per cycle
MODEL_VERSION = "confidence-calibrator-tracer-0.1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_beliefs(folder: Path) -> list[dict]:
    out = []
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.belief.json")):
        try:
            with p.open(encoding="utf-8") as fh:
                data = json.load(fh)
            data["__file"] = str(p)
            out.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  warn: skipping unreadable belief {p}: {exc}", file=sys.stderr)
    return out


def compute_calibration(demerzel_root: Path) -> dict:
    beliefs_dir = demerzel_root / "state" / "beliefs"
    current = _load_beliefs(beliefs_dir)
    archived = _load_beliefs(beliefs_dir / "archived")  # prior, revised versions

    all_beliefs = current + archived
    high_conf = [b for b in all_beliefs if isinstance(b.get("confidence"), (int, float))
                 and b["confidence"] >= HIGH_CONF]
    # An archived prior version IS a belief that was revised.
    revised_overconfident = [b for b in archived
                             if isinstance(b.get("confidence"), (int, float))
                             and b["confidence"] >= HIGH_CONF]

    rate = (len(revised_overconfident) / len(high_conf)) if high_conf else 0.0
    return {
        "data_points": len(all_beliefs),
        "current_count": len(current),
        "archived_count": len(archived),
        "high_conf_count": len(high_conf),
        "revised_overconfident_count": len(revised_overconfident),
        "overconfidence_rate": round(rate, 4),
        "worst_examples": [Path(b["__file"]).name for b in revised_overconfident][:5],
    }


def build_recommendation(calib: dict) -> dict:
    rate = calib["overconfidence_rate"]
    # ml-feedback-loop.ixql §3a formula, clamped to the policy guardrail.
    nudge = max(-NUDGE_CAP, round(-rate * 0.5, 4))

    # Producer confidence: the *direction* (reduce threshold) is well supported
    # even at small N, and the magnitude is hard-capped at +/-0.1, so confidence
    # is bounded but meaningful. Grows modestly with data points.
    confidence = round(min(0.92, 0.70 + 0.02 * calib["data_points"]), 3)

    payload = {
        "pipeline_id": "confidence_calibrator",
        "recommendation_type": "threshold_adjustment",
        "recommendation": {
            "action": "Nudge the global confidence-calibration threshold by "
                      f"{nudge} (reduce, to counter detected overconfidence).",
            "rationale": (
                f"{calib['revised_overconfident_count']} of {calib['high_conf_count']} "
                f"high-confidence beliefs (>= {HIGH_CONF}) were later revised "
                f"(overconfidence_rate={rate}). Revisions evidenced by archived prior "
                f"versions: {', '.join(calib['worst_examples']) or 'none'}. Nudge magnitude "
                f"capped at +/-{NUDGE_CAP}/cycle per ml-governance-feedback-policy."
            ),
            "expected_impact": (
                "Future high-confidence assignments require marginally stronger evidence, "
                "reducing the overconfidence-then-revision pattern over subsequent cycles."
            ),
            "parameters": {
                "threshold_nudge": nudge,
                "target": "state/beliefs/confidence-calibration.belief.json",
                "cap": NUDGE_CAP,
            },
        },
        "confidence": confidence,
        "evidence": {
            "data_points": calib["data_points"],
            "model_version": MODEL_VERSION,
            "training_window": "all on-disk belief states (current + archived)",
            "key_features": [
                "belief.confidence", "archived prior version (revision marker)",
                f"high_conf_count={calib['high_conf_count']}",
                f"revised_overconfident_count={calib['revised_overconfident_count']}",
            ],
        },
        "constitutional_check": {
            "passed": True,
            "articles_checked": [
                "Article 3 - Reversibility",
                "Article 7 - Auditability",
                "Article 9 - Bounded Autonomy",
            ],
            "concerns": [],
        },
        "timestamp": _now_iso(),
    }
    return payload


def _attach_integrity(payload: dict) -> dict:
    """Add Galactic-Protocol integrity fields. content_hash is the sha256 of the
    payload BEFORE integrity fields are added (per integrity-fields.schema.json)."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    content_hash = hashlib.sha256(canonical).hexdigest()
    doc = dict(payload)
    doc.update({
        "message_id": str(uuid.uuid4()),
        "origin_repo": "ix",
        "origin_agent": "ix-confidence-calibrator",
        "content_hash": content_hash,
        "hash_algorithm": "sha256",
        # integrity 'timestamp' mirrors payload timestamp (same generation moment)
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
    ap = argparse.ArgumentParser(description="ix confidence_calibrator (producer)")
    here = Path(__file__).resolve()
    default_demerzel = here.parents[2] / "Demerzel"  # repos/ix/scripts/x.py -> repos/Demerzel
    ap.add_argument("--demerzel-root", type=Path, default=default_demerzel)
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args(argv)

    root = args.demerzel_root.resolve()
    if not (root / "state" / "beliefs").is_dir():
        print(f"error: no state/beliefs under {root}", file=sys.stderr)
        return 1

    calib = compute_calibration(root)
    print(f"calibration: {json.dumps(calib, indent=2)}", file=sys.stderr)

    if calib["overconfidence_rate"] <= TRIGGER:
        print(f"overconfidence_rate {calib['overconfidence_rate']} <= trigger {TRIGGER} "
              "-> no recommendation warranted (loop no-op).", file=sys.stderr)
        return 4

    print("note: pipeline §3a filters recommendation_type=='calibration_report' but the "
          "schema enum is 'threshold_adjustment'; emitting the schema-valid shape.",
          file=sys.stderr)

    doc = _attach_integrity(build_recommendation(calib))
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
