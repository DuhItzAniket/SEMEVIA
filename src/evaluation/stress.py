"""Fixed stress conditions and predeclared evaluation subsets.

Conditions transform the stored-pixel image; the pipeline then reruns detection and
classification on the transformed image. Box-guided occlusion uses the annotation only to place
the synthetic corruption; no ground-truth box is passed to inference.
"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def apply_condition(rgb: np.ndarray, cond: dict, box: list[float], key: str, seed: int, name: str) -> np.ndarray:
    t = cond["type"]
    if t == "none":
        return rgb
    if t == "brightness":
        return np.clip(np.rint(rgb.astype(np.float32) * cond["factor"]), 0, 255).astype(np.uint8)
    if t == "downsample":
        h, w = rgb.shape[:2]
        f = cond["factor"]
        small = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if t == "occlusion":
        l, tp, r, b = box
        bw, bh = r - l, b - tp
        frac = cond["area_frac"]
        ow, oh = bw * np.sqrt(frac), bh * np.sqrt(frac)   # same aspect as the box, area = frac * box area
        rng = np.random.RandomState(_seed(seed, key, name))
        x0 = l + rng.uniform(0, bw - ow)
        y0 = tp + rng.uniform(0, bh - oh)
        out = rgb.copy()
        out[int(round(y0)):int(round(y0 + oh)), int(round(x0)):int(round(x0 + ow))] = 128
        return out
    raise ValueError(f"unknown condition type {t}")


def stratified_subset(keys: list[str], labels: list[int], per_class: int, seed: int) -> list[str]:
    """Seeded selection of up to per_class keys per label; no model outcomes involved."""
    out = []
    for c in sorted(set(labels)):
        ks = sorted(k for k, y in zip(keys, labels) if y == c)
        rng = np.random.RandomState(_seed(seed, "stress", c))
        pick = rng.choice(len(ks), min(per_class, len(ks)), replace=False)
        out.extend(ks[i] for i in sorted(pick))
    return sorted(out)
