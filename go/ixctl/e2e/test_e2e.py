"""Unit tests of the gate's own helpers, without planes or binaries.

    python -I -m unittest discover -s go/ixctl/e2e -p "test_*.py" -v
"""

import unittest

import e2e

# The list E18 read on ubuntu CI (run 40987881936, job 123843050209), the two nodes as listed.
CI_WINDOW = {
    "bud": {"bytes": 0, "children": 1, "created_seq": 7956, "errors": 0, "every_ms": 0, "kind": "mux@1",
            "max_rate_per_s": 150, "name": "bud", "rate_per_s": 100, "seq": 0, "state": "running", "ttl_ms": 0},
    "bud-0": {"bytes": 51, "children": 0, "created_seq": 7957, "errors": 0, "every_ms": 10, "kind": "heartbeat@1",
              "name": "bud-0", "parent": "bud", "rate_per_s": 100, "seq": 1, "state": "running", "ttl_ms": 0},
}


def listing(mux_seq, mux_bytes, child_seq, child_bytes):
    return {"bud": dict(CI_WINDOW["bud"], seq=mux_seq, bytes=mux_bytes),
            "bud-0": dict(CI_WINDOW["bud-0"], seq=child_seq, bytes=child_bytes)}


class CaughtUp(unittest.TestCase):
    def test_the_list_ci_read_is_not_one_to_compare_bytes_on(self):
        self.assertFalse(e2e.caught_up(CI_WINDOW, "bud", "bud-0"))

    def test_a_mux_one_record_behind_later_on_is_not_either(self):
        self.assertFalse(e2e.caught_up(listing(6, 342, 7, 357), "bud", "bud-0"))

    def test_a_mux_that_republished_every_record_is(self):
        self.assertTrue(e2e.caught_up(listing(1, 57, 1, 51), "bud", "bud-0"))

    def test_a_child_that_emitted_nothing_is_not(self):
        self.assertFalse(e2e.caught_up(listing(0, 0, 0, 0), "bud", "bud-0"))
        self.assertFalse(e2e.caught_up({}, "bud", "bud-0"))


if __name__ == "__main__":
    unittest.main()
