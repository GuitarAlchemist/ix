"""Rasterize ix_knot's rope paths into ControlNet control images: lineart and depth.

IX returns each rope's centreline in diagram units (x right, y up, z toward the viewer) and the rope
radius. The drawing is fitted into the image with a margin, y drawn upward so the picture is the knot
and not its mirror, and each rope is a tube: a dome of the radius stamped along its path at height z,
the nearest surface winning.

Where a rope passes over itself, both passages are the same rope, so the rope index alone draws no
outline there and lets the back passage's lay strokes land on the front one; a render then guesses the
crossing. Each visible point therefore also records how far along its rope it is, and two points of one
rope more than PIECE radii apart along it are different pieces: outlined from each other, each with
its own strokes.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MARGIN = 0.06       # of the smaller image side, kept clear around the drawing
LAY_PITCH = 0.65    # rope radii of length between two rope-lay strokes
LAY_ANGLE = 35      # degrees the lay strokes lean from square across the rope
PIECE = 4.0         # rope radii along a rope beyond which two of its visible points are different pieces


def geometry_px(knot_out, width, height):
    """Rope paths in pixels, a list of (n, 3) arrays of x, row, z-height; and the tube radius in pixels."""
    g = knot_out["geometry"]
    radius = float(g["radius"])
    points = [np.asarray(r["points"], dtype=float) for r in g["ropes"]]
    xy = np.concatenate(points)[:, :2]
    lo, hi = xy.min(axis=0) - radius, xy.max(axis=0) + radius
    margin = MARGIN * min(width, height)
    scale = min((width - 2 * margin) / (hi[0] - lo[0]), (height - 2 * margin) / (hi[1] - lo[1]))
    mid = (lo + hi) / 2
    paths = []
    for p, rope in zip(points, g["ropes"]):
        x = width / 2 + (p[:, 0] - mid[0]) * scale
        row = height / 2 - (p[:, 1] - mid[1]) * scale
        q = np.stack([x, row, p[:, 2] * scale], axis=1)
        if rope["closed"]:
            q = np.concatenate([q, q[:1]])
        paths.append(q)
    return paths, radius * scale


def render(paths, radius, width, height):
    """owner: rope index per pixel (-1 off the ropes); surface: tube height, -inf off the ropes."""
    owner, _, surface = stamp(paths, radius, width, height)
    return owner, surface


def stamp(paths, radius, width, height):
    """render's owner and surface, and `along`: how far along its rope (px) each visible point is."""
    owner = np.full((height, width), -1, dtype=np.int16)
    along = np.zeros((height, width))
    surface = np.full((height, width), -np.inf)
    r = int(math.ceil(radius))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    dome = np.where(ox ** 2 + oy ** 2 <= radius ** 2,
                    np.sqrt(np.clip(radius ** 2 - ox ** 2 - oy ** 2, 0, None)), -np.inf)
    for rope, p in enumerate(paths):
        dense = _dense(p, step=0.5)
        for (x, row, z), s in zip(dense, _arc(dense[:, :2])):
            cy, cx = int(round(row)), int(round(x))
            y0, y1, x0, x1 = cy - r, cy + r + 1, cx - r, cx + r + 1
            if y1 <= 0 or y0 >= height or x1 <= 0 or x0 >= width:
                continue
            sy0, sx0 = max(0, -y0), max(0, -x0)
            ty0, ty1, tx0, tx1 = max(0, y0), min(height, y1), max(0, x0), min(width, x1)
            patch = z + dome[sy0:sy0 + ty1 - ty0, sx0:sx0 + tx1 - tx0]
            view = surface[ty0:ty1, tx0:tx1]
            front = patch > view
            view[front] = patch[front]
            owner[ty0:ty1, tx0:tx1][front] = rope
            along[ty0:ty1, tx0:tx1][front] = s
    return owner, along, surface


def _dense(p, step):
    """The polyline resampled at most `step` px apart in the image plane, z interpolated."""
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        k = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step)))
        t = np.arange(1, k + 1)[:, None] / k
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def _arc(xy):
    """Length along a polyline at each of its points, in the image plane."""
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])


class Pieces:
    """How far apart along its rope two points of that rope are, the shorter way round a closed one."""

    def __init__(self, paths, closed, gap):
        self.length = np.array([_arc(p[:, :2])[-1] for p in paths])
        self.closed = np.asarray(closed, dtype=bool)
        self.gap = gap

    def distance(self, rope, a, b):
        d = np.abs(a - b)
        return np.where(self.closed[rope], np.minimum(d, self.length[rope] - d), d)


def lay_lines(paths, owner, radius, along=None, pieces=None):
    """Short slanted strokes across each rope every LAY_PITCH radii of its length, kept where that rope
    is the visible one, so a stroke never lands on the rope in front; with `along` and `pieces`, where
    that piece of it is, so a stroke never lands on the same rope passing in front either."""
    height, width = owner.shape
    pitch = LAY_PITCH * radius
    out = np.zeros((height, width), dtype=np.uint8)
    for rope, p in enumerate(paths):
        layer = Image.new("F", (width, height), -1.0)   # each stroke's arc length, -1 off the strokes
        draw = ImageDraw.Draw(layer)
        xy = p[:, :2]
        arc = _arc(xy)
        for j in np.nonzero(np.diff(np.floor(arc / pitch + 1e-9)) > 0)[0]:
            (x0, y0), (x1, y1) = xy[j], xy[j + 1]
            cy, cx = int(round(y0)), int(round(x0))
            if not (0 <= cy < height and 0 <= cx < width) or owner[cy, cx] != rope:
                continue
            if pieces is not None and pieces.distance(rope, along[cy, cx], arc[j]) > pieces.gap:
                continue
            a = math.atan2(y1 - y0, x1 - x0) + math.radians(90 - LAY_ANGLE)
            dx, dy = 0.85 * radius * math.cos(a), 0.85 * radius * math.sin(a)
            draw.line([(x0 - dx, y0 - dy), (x0 + dx, y0 + dy)], fill=float(arc[j]), width=3)
        at = np.asarray(layer)
        mine = (at >= 0) & (owner == rope)
        if pieces is not None:
            mine &= pieces.distance(rope, along, at) <= pieces.gap
        out[mine] = 255
    return out


def outlines(owner, along=None, pieces=None):
    """Where the visible rope changes; with `along` and `pieces`, also where one rope passes over itself."""
    edge = np.zeros(owner.shape, dtype=bool)
    for a, b in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                 ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        changed = owner[a] != owner[b]
        if pieces is not None:
            rope = np.clip(owner[a], 0, None)
            changed |= (owner[a] >= 0) & (pieces.distance(rope, along[a], along[b]) > pieces.gap)
        edge[a] |= changed
    return edge


def lineart(owner, strokes, along=None, pieces=None):
    """White on black: tube outlines and the rope lay."""
    edge = outlines(owner, along, pieces)
    lines = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5)))
    return np.maximum(lines, strokes)


def depth(surface):
    """Near is bright: the background at 40, the ropes from 90 to 255 by surface height."""
    on = np.isfinite(surface)
    out = np.full(surface.shape, 40.0)
    if on.any():
        lo, hi = surface[on].min(), surface[on].max()
        out[on] = 90 + 165 * (surface[on] - lo) / max(hi - lo, 1e-9)
    return out.astype(np.uint8)


def control_images(knot_out, width, height):
    """(lineart, depth, owner) as uint8 / int16 arrays of shape (height, width)."""
    paths, radius = geometry_px(knot_out, width, height)
    owner, along, surface = stamp(paths, radius, width, height)
    pieces = Pieces(paths, [r["closed"] for r in knot_out["geometry"]["ropes"]], PIECE * radius)
    strokes = lay_lines(paths, owner, radius, along, pieces)
    return lineart(owner, strokes, along, pieces), depth(surface), owner
