"""
Hermetic tests for fleet/claims.py — the ix-owned coordination-ledger tool.

Stdlib-only (unittest + tempfiles), no network, no dependency on the real
~/.agents/claims.jsonl. Run::

    python -m unittest fleet/test_claims.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claims  # noqa: E402


def _valid_line(**over: object) -> str:
    base = {
        "ts": "2026-07-21T23:37:36Z",
        "repo": "ix",
        "lane": "demo",
        "status": "claimed",
        "session": "ix",
        "evidence": None,
        "note": "",
    }
    base.update(over)
    return json.dumps(base)


class ValidateClaimTests(unittest.TestCase):
    def test_good_claim_has_no_problems(self) -> None:
        self.assertEqual(claims.validate_claim(json.loads(_valid_line())), [])

    def test_missing_required_field_flagged(self) -> None:
        obj = json.loads(_valid_line())
        del obj["session"]
        self.assertIn("missing required field 'session'", claims.validate_claim(obj))

    def test_bad_status_flagged(self) -> None:
        obj = json.loads(_valid_line(status="parked"))
        self.assertTrue(any("status" in p for p in claims.validate_claim(obj)))

    def test_bad_timestamp_flagged(self) -> None:
        obj = json.loads(_valid_line(ts="2026-07-21 23:37"))  # not RFC3339 Z
        self.assertTrue(any("RFC3339" in p for p in claims.validate_claim(obj)))

    def test_empty_repo_flagged(self) -> None:
        obj = json.loads(_valid_line(repo="  "))
        self.assertTrue(any("'repo'" in p for p in claims.validate_claim(obj)))

    def test_non_object_flagged(self) -> None:
        self.assertEqual(claims.validate_claim([1, 2, 3]), ["not a JSON object"])


class LedgerFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "claims.jsonl"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_validate_passes_on_clean_ledger(self) -> None:
        self.path.write_text(_valid_line() + "\n" + _valid_line(lane="two") + "\n", encoding="utf-8")
        self.assertEqual(claims.main(["--file", str(self.path), "validate"]), 0)

    def test_validate_fails_on_bad_json(self) -> None:
        self.path.write_text(_valid_line() + "\n{not json}\n", encoding="utf-8")
        self.assertEqual(claims.main(["--file", str(self.path), "validate"]), 1)

    def test_validate_fails_on_schema_violation(self) -> None:
        self.path.write_text(_valid_line(status="nope") + "\n", encoding="utf-8")
        self.assertEqual(claims.main(["--file", str(self.path), "validate"]), 1)

    def test_append_stamps_ts_and_roundtrips(self) -> None:
        rc = claims.main([
            "--file", str(self.path), "append",
            "--repo", "ix", "--lane", "L", "--status", "done",
            "--session", "ix", "--evidence", "PR#1",
        ])
        self.assertEqual(rc, 0)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 1)
        obj = json.loads(lines[0])
        self.assertEqual(obj["lane"], "L")
        self.assertEqual(obj["evidence"], "PR#1")
        self.assertEqual(claims.validate_claim(obj), [])  # what we wrote is valid

    def test_append_refuses_bad_evidence_type_via_api(self) -> None:
        # argparse forces evidence to str|None, so exercise the guard directly.
        obj = {"ts": claims.now_rfc3339(), "repo": "ix", "lane": "L",
               "status": "done", "session": "ix", "evidence": 42}
        self.assertTrue(any("evidence" in p for p in claims.validate_claim(obj)))

    def test_status_latest_line_wins(self) -> None:
        self.path.write_text(
            _valid_line(lane="L", status="claimed") + "\n"
            + _valid_line(lane="L", status="done", ts="2026-07-21T23:40:00Z") + "\n",
            encoding="utf-8",
        )
        latest = claims.latest_by_lane(claims.parse_ledger(self.path))
        self.assertEqual(latest[("ix", "L")]["status"], "done")

    def test_status_open_filter_hides_done(self) -> None:
        self.path.write_text(_valid_line(lane="L", status="done") + "\n", encoding="utf-8")
        # only_open => the done lane is filtered out; command still exits 0.
        self.assertEqual(claims.main(["--file", str(self.path), "status", "--open"]), 0)


if __name__ == "__main__":
    unittest.main()
