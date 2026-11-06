"""Dataset crop loading in the stored pixel frame (EXIF is never applied to dataset images)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.vision.coordinates import crop_bounds, expand_box


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def annotation_crop(rgb: np.ndarray, box: list[float], margin: float) -> tuple[np.ndarray, list[float]]:
    h, w = rgb.shape[:2]
    eb = expand_box(box, margin, w, h)
    l, t, r, b = crop_bounds(eb, w, h)
    return rgb[t:b, l:r], [float(l), float(t), float(r), float(b)]
