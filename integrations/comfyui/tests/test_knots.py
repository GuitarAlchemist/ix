"""Knots and braids through the installed ix-mcp, drawn as control images: the picture shows the knot IX
returned (not its mirror), the rope in front wins each crossing, and inputs are checked before IX
starts. Needs numpy and Pillow, not torch."""
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from ix_comfyui import bridge  # noqa: E402
from ix_comfyui.bridge import IxBridgeError, braid_layout, knot_catalog, knot_layout  # noqa: E402
from ix_comfyui.braid_render import control_images as braid_images  # noqa: E402
from ix_comfyui.nodes import IXBraidControl, IXKnotControl  # noqa: E402
from ix_comfyui.rope_render import control_images, geometry_px, render  # noqa: E402

W, H = 512, 768


class Catalogue(unittest.TestCase):
    def test_lists_the_knots_with_their_names(self):
        entries = {e["id"]: e for e in knot_catalog()}
        self.assertEqual(entries["figure-eight"]["fr"], "Nœud en huit")
        self.assertEqual(entries["overhand"]["closure"], "3_1")

    def test_the_figure_eight_closes_into_4_1(self):
        out = knot_layout("figure-eight")
        self.assertEqual(out["crossings"], 4)
        self.assertEqual(out["jones"]["text"], "t^-2 - t^-1 + 1 - t + t^2")
        self.assertGreaterEqual(out["geometry"]["min_clearance"], 1.0)

    def test_ids_are_checked_before_ix_starts_and_unknown_ones_come_back_as_errors(self):
        for bad in ("", "Figure-Eight", "eight; calc", "x" * 65, 3):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                knot_layout(bad)
        with self.assertRaises(IxBridgeError):
            knot_layout("no-such-knot")


class Picture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = knot_layout("figure-eight")
        cls.paths, cls.radius = geometry_px(cls.out, W, H)
        cls.owner, cls.surface = render(cls.paths, cls.radius, W, H)

    def test_y_is_drawn_upward_so_the_picture_is_not_the_mirror(self):
        # The working end leaves at the bottom right of IX's drawing (y up): it must land in the
        # bottom-right quarter of the image (rows down).
        end = self.out["geometry"]["ropes"][0]["points"][-1]
        start = self.out["geometry"]["ropes"][0]["points"][0]
        self.assertGreater(end[0], start[0])
        self.assertLess(end[1], start[1])
        x, row, _ = self.paths[0][-1]
        self.assertGreater(x, W / 2)
        self.assertGreater(row, H / 2)
        self.assertEqual(self.owner[int(row), int(x)], 0)

    def test_the_rope_in_front_is_the_visible_surface_at_each_crossing(self):
        # Where two pieces of the rope cross in the image, the surface is the higher piece's top.
        p = self.paths[0]
        a, b = p[:-1], p[1:]
        crossings = 0
        for i in range(len(a)):
            d1 = b[i, :2] - a[i, :2]
            for j in range(i + 2, len(a)):
                d2 = b[j, :2] - a[j, :2]
                den = d1[0] * d2[1] - d1[1] * d2[0]
                if abs(den) < 1e-12:
                    continue
                w = a[j, :2] - a[i, :2]
                t = (w[0] * d2[1] - w[1] * d2[0]) / den
                u = (w[0] * d1[1] - w[1] * d1[0]) / den
                if 0 < t < 1 and 0 < u < 1:
                    zi = a[i, 2] + t * (b[i, 2] - a[i, 2])
                    zj = a[j, 2] + u * (b[j, 2] - a[j, 2])
                    x, row = a[i, :2] + t * d1
                    self.assertAlmostEqual(self.surface[int(round(row)), int(round(x))],
                                           max(zi, zj) + self.radius, delta=0.1 * self.radius)
                    crossings += 1
        self.assertEqual(crossings, self.out["crossings"])

    def test_control_images_have_lines_and_two_depth_levels_at_least(self):
        line, deep, owner = control_images(self.out, W, H)
        self.assertEqual(line.shape, (H, W))
        self.assertTrue((line == 255).any())
        self.assertGreater(len(np.unique(deep[owner >= 0])), 2)
        self.assertTrue((deep[owner < 0] == 40).all())


class Nodes(unittest.TestCase):
    def test_registered_and_take_no_path_executable_or_tool_input(self):
        from ix_comfyui import NODE_CLASS_MAPPINGS
        self.assertIs(NODE_CLASS_MAPPINGS["IXKnotControl"], IXKnotControl)
        required = IXKnotControl.INPUT_TYPES()["required"]
        self.assertEqual(set(required), {"knot", "width", "height"})
        self.assertIn("figure-eight", required["knot"][0])
        self.assertEqual(set(IXBraidControl.INPUT_TYPES()["required"]), {"word", "repeat", "width", "height"})

    def test_knot_node_images_and_summary(self):
        line, deep, summary = IXKnotControl().images("overhand", 512, 768)
        self.assertEqual(line.shape, (768, 512))
        self.assertEqual(deep.shape, (768, 512))
        self.assertEqual(summary["closure"], "3_1")
        self.assertEqual(summary["ix_mcp_sha256"], bridge.installed_hash())
        with self.assertRaises(ValueError):
            IXKnotControl().images("overhand", 4096, 768)

    def test_conjugating_a_braid_word_rolls_its_picture(self):
        # s2^-1 s1 repeated is s1 s2^-1 repeated started one crossing later: the same picture moved
        # up one crossing (y is drawn upward), wrapping at the edges.
        out = braid_layout("s1 s2^-1", 6)
        self.assertEqual(out["components"], 3)
        _, deep, _ = braid_images(out, 640, 1536)
        _, rolled, _ = braid_images(braid_layout("s2^-1 s1", 6), 640, 1536)
        per_crossing = 1536 // out["crossings"]
        np.testing.assert_allclose(rolled.astype(int), np.roll(deep, per_crossing, axis=0).astype(int), atol=1, rtol=0)
        _, _, summary = IXBraidControl().images("s1 s2^-1", 2, 640, 512)
        self.assertEqual(summary["jones"]["text"], "t^-2 - t^-1 + 1 - t + t^2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
