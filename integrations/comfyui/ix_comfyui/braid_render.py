"""Rasterize ix_braid's strand layout into ControlNet control images: lineart and depth.

IX lays the braid out in a right-handed frame (x across the strands, y along the braid, one crossing
per unit, z toward the viewer). Image rows grow downward, so y is drawn upward (row = H - y * scale)
and the picture shows the braid itself rather than its mirror. Each strand is a tube: a dome of
RADIUS stamped along its path at height z * DEPTH, the nearest surface winning. Stamps near the top
and bottom edges are repeated one image height away, and the rope-lay strokes are spaced along each
closed component of the braid's closure, so the images repeat vertically without a seam.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

MAX_RADIUS = 34.0       # tube radius, px, as in knots/braid.py
MAX_SPACING = 150.0     # px between neighbouring strand positions: the plait's half-width in braid.py
EDGE = 48               # px kept clear at each side, inside the panel borders
LAY_PITCH = 22.0        # px of strand length between rope-lay strokes
BORDER_INSET = 24       # the panel borders, as in make_control.py


def geometry_px(braid_out, width, height):
    """Strand paths in pixels: {strand start: (n, 3) array of x, row, z-height}; and the tube radius."""
    strands = braid_out["geometry"]["strands"]
    n = len(strands)
    spacing = MAX_SPACING if n < 2 else min(MAX_SPACING, (width - 2 * EDGE - 2 * MAX_RADIUS) / (n - 1))
    radius = MAX_RADIUS if n < 2 else min(MAX_RADIUS, 0.45 * spacing)
    scale = height / braid_out["crossings"]
    paths = {}
    for s in strands:
        p = np.asarray(s["points"], dtype=float)
        x = width / 2 + (p[:, 0] - (n - 1) / 2) * spacing
        row = height - p[:, 1] * scale
        z = 1.2 * radius * p[:, 2]   # depth swing above the radius, so the front tube wins a crossing
        paths[int(s["start"])] = np.stack([x, row, z], axis=1)
    return paths, radius


def render(paths, radius, width, height):
    """owner: strand start per pixel (-1 off the tubes); surface: tube height, -inf off the tubes."""
    owner = np.full((height, width), -1, dtype=np.int16)
    surface = np.full((height, width), -np.inf)
    r = int(math.ceil(radius))
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    dome = np.where(ox ** 2 + oy ** 2 <= radius ** 2,
                    np.sqrt(np.clip(radius ** 2 - ox ** 2 - oy ** 2, 0, None)), -np.inf)
    for strand, p in paths.items():
        for x, row, z in _dense(p, step=0.5):
            for shift in (-height, 0, height):
                cy, cx = int(round(row + shift)), int(round(x))
                y0, y1, x0, x1 = cy - r, cy + r + 1, cx - r, cx + r + 1
                if y1 <= 0 or y0 >= height:
                    continue
                sy0, sx0 = max(0, -y0), max(0, -x0)
                ty0, ty1, tx0, tx1 = max(0, y0), min(height, y1), max(0, x0), min(width, x1)
                patch = z + dome[sy0:sy0 + ty1 - ty0, sx0:sx0 + tx1 - tx0]
                view = surface[ty0:ty1, tx0:tx1]
                front = patch > view
                view[front] = patch[front]
                owner[ty0:ty1, tx0:tx1][front] = strand
    return owner, surface


def _dense(p, step):
    """The polyline resampled at most `step` px apart in the image plane, z interpolated."""
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        k = max(1, int(math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / step)))
        t = np.arange(1, k + 1)[:, None] / k
        out.append(a + (b - a) * t)
    return np.concatenate(out)


def lay_lines(paths, permutation, owner, radius, height):
    """Short slanted strokes across each strand every ~LAY_PITCH px of its length, kept where that strand
    is the visible one. They are spaced along each closed component of the closure (strand after strand
    through the permutation, one image height further each time) with a pitch that divides its length,
    so they continue across the top and bottom edges."""
    width = owner.shape[1]
    out = np.zeros_like(owner, dtype=np.uint8)
    seen = set()
    for first in sorted(paths):
        if first in seen:
            continue
        cycle, s = [], first
        while s not in seen:
            seen.add(s)
            cycle.append(s)
            s = permutation[s]
        pieces = [paths[s][:, :2] + [0, -k * height] for k, s in enumerate(cycle)]
        owners = np.concatenate([np.full(len(p), s) for p, s in zip(pieces, cycle)])
        xy = np.concatenate(pieces)
        arc = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        pitch = arc[-1] / max(1, round(arc[-1] / LAY_PITCH))
        layers = {s: Image.new("L", (width, 3 * height), 0) for s in cycle}
        for j in np.nonzero(np.diff(np.floor(arc / pitch + 1e-9)) > 0)[0]:
            (x0, y0), (x1, y1) = xy[j], xy[j + 1]
            cy, cx = int(round(y0)) % height, int(round(x0))
            if not 0 <= cx < width or owner[cy, cx] != owners[j]:
                continue   # the strand is hidden here: a stroke would land on whatever is in front

            a = math.atan2(y1 - y0, x1 - x0) + math.radians(90 - 35)
            dx, dy = 0.85 * radius * math.cos(a), 0.85 * radius * math.sin(a)
            row = y0 % height + height   # the middle third of a canvas three images high
            for shift in (-height, 0, height):
                ImageDraw.Draw(layers[owners[j]]).line(
                    [(x0 - dx, row + shift - dy), (x0 + dx, row + shift + dy)], fill=255, width=3)
        for s, layer in layers.items():
            out[(np.asarray(layer)[height:2 * height] > 0) & (owner == s)] = 255
    return out


def lineart(owner, strokes):
    """White on black: tube outlines (owner changes, wrapping vertically), the rope lay, panel borders."""
    edge = np.zeros(owner.shape, dtype=bool)
    edge[:, 1:] |= owner[:, 1:] != owner[:, :-1]
    edge |= owner != np.roll(owner, 1, axis=0)
    padded = np.pad((edge * 255).astype(np.uint8), ((2, 2), (0, 0)), mode="wrap")
    lines = np.asarray(Image.fromarray(padded).filter(ImageFilter.MaxFilter(5)))[2:-2]
    img = Image.fromarray(np.maximum(lines, strokes))
    draw = ImageDraw.Draw(img)
    for x in (BORDER_INSET, owner.shape[1] - BORDER_INSET):
        draw.line([(x, 0), (x, owner.shape[0])], fill=255, width=5)
    return np.asarray(img)


def depth(surface):
    """Near is bright: the stone plane at 40, the tubes from 90 to 255 by surface height."""
    on = np.isfinite(surface)
    out = np.full(surface.shape, 40.0)
    if on.any():
        lo, hi = surface[on].min(), surface[on].max()
        out[on] = 90 + 165 * (surface[on] - lo) / max(hi - lo, 1e-9)
    return out.astype(np.uint8)


def control_images(braid_out, width, height):
    """(lineart, depth, owner) as uint8 / int16 arrays of shape (height, width)."""
    paths, radius = geometry_px(braid_out, width, height)
    owner, surface = render(paths, radius, width, height)
    strokes = lay_lines(paths, braid_out["permutation"], owner, radius, height)
    return lineart(owner, strokes), depth(surface), owner
