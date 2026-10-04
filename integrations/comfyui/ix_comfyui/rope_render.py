"""Rasterize ix_knot's rope paths into ControlNet control images: lineart and depth.

IX returns each rope's centreline in diagram units (x right, y up, z toward the viewer) and the rope
radius. The drawing is fitted into the image with a margin, y drawn upward so the picture is the knot
and not its mirror, and each rope is a tube: a dome of the radius stamped along its path at height z,
the nearest surface winning.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MARGIN = 0.06       # of the smaller image side, kept clear around the drawing
LAY_PITCH = 0.65    # rope radii of length between two rope-lay strokes
LAY_ANGLE = 35      # degrees the lay strokes lean from square across the rope


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
    owner = np.full((height, width), -1, dtype=np.int16)
    surface = np.full((height, width), -np.inf)
    r = int(math.ceil(radius))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    dome = np.where(ox ** 2 + oy ** 2 <= radius ** 2,
                    np.sqrt(np.clip(radius ** 2 - ox ** 2 - oy ** 2, 0, None)), -np.inf)
    for rope, p in enumerate(paths):
        for x, row, z in _dense(p, step=0.5):
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
    return owner, surface


def _dense(p, step):
    """The polyline resampled at most `step` px apart in the image plane, z interpolated."""
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        k = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step)))
        t = np.arange(1, k + 1)[:, None] / k
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def lay_lines(paths, owner, radius):
    """Short slanted strokes across each rope every LAY_PITCH radii of its length, kept where that rope
    is the visible one, so a stroke never lands on the rope in front."""
    height, width = owner.shape
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    pitch = LAY_PITCH * radius
    for rope, p in enumerate(paths):
        xy = p[:, :2]
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        for j in np.nonzero(np.diff(np.floor(arc / pitch + 1e-9)) > 0)[0]:
            (x0, y0), (x1, y1) = xy[j], xy[j + 1]
            cy, cx = int(round(y0)), int(round(x0))
            if not (0 <= cy < height and 0 <= cx < width) or owner[cy, cx] != rope:
                continue
            a = math.atan2(y1 - y0, x1 - x0) + math.radians(90 - LAY_ANGLE)
            dx, dy = 0.85 * radius * math.cos(a), 0.85 * radius * math.sin(a)
            draw.line([(x0 - dx, y0 - dy), (x0 + dx, y0 + dy)], fill=255, width=3)
    strokes = np.asarray(layer)
    out = np.zeros_like(strokes)
    for rope in range(len(paths)):
        out[(strokes > 0) & (owner == rope)] = 255
    return out


def lineart(owner, strokes):
    """White on black: tube outlines (where the visible rope changes) and the rope lay."""
    edge = np.zeros(owner.shape, dtype=bool)
    edge[:, 1:] |= owner[:, 1:] != owner[:, :-1]
    edge[1:, :] |= owner[1:, :] != owner[:-1, :]
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
    owner, surface = render(paths, radius, width, height)
    return lineart(owner, lay_lines(paths, owner, radius)), depth(surface), owner
