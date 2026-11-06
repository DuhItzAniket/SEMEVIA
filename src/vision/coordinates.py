"""Canonical box conventions.

Canonical box: float [left, top, right, bottom], zero-based, half-open, in stored-image pixels.
Stanford Cars annotations are 1-based inclusive MATLAB indices; conversion happens once here.
"""
from __future__ import annotations

import math
from typing import Sequence

BBOX_FORMAT_VERSION = "ltrb_halfopen_0based_v1"


def stanford_to_canonical(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """Convert 1-based inclusive pixel indices to zero-based half-open coordinates."""
    return [float(x1) - 1.0, float(y1) - 1.0, float(x2), float(y2)]


def box_valid(box: Sequence[float], width: int, height: int) -> bool:
    l, t, r, b = box
    finite = all(math.isfinite(v) for v in box)
    return finite and 0 <= l < r <= width and 0 <= t < b <= height


def box_area(box: Sequence[float]) -> float:
    l, t, r, b = box
    return max(0.0, r - l) * max(0.0, b - t)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two canonical boxes. Touching boxes give 0."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def letterbox_to_original(box: Sequence[float], scale: float, pad_x: float, pad_y: float) -> list[float]:
    """Invert a resize-by-`scale` then pad transform (detector frame -> original pixels)."""
    l, t, r, b = box
    return [(l - pad_x) / scale, (t - pad_y) / scale, (r - pad_x) / scale, (b - pad_y) / scale]


def expand_box(box: Sequence[float], margin: float, width: int, height: int) -> list[float]:
    """Expand each side by `margin` * box width/height, clipped to the image."""
    l, t, r, b = box
    mw, mh = (r - l) * margin, (b - t) * margin
    return [max(0.0, l - mw), max(0.0, t - mh), min(float(width), r + mw), min(float(height), b + mh)]


def crop_bounds(box: Sequence[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    """Integer array bounds: floor left/top, ceil right/bottom, clip; None for zero area."""
    l = max(0, math.floor(box[0]))
    t = max(0, math.floor(box[1]))
    r = min(width, math.ceil(box[2]))
    b = min(height, math.ceil(box[3]))
    if r <= l or b <= t:
        return None
    return l, t, r, b
