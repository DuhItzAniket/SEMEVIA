"""Fully classical vehicle ROI proposal baseline (no learned detector).

Grayscale -> blur -> Canny -> large morphological close -> external contours -> bounding
rectangles, scored by a fixed bounded rule of normalised area and edge support, filtered by a
plausible aspect ratio. Returns a defined failure (empty list) when nothing passes. This is a
baseline; it is not expected to match the learned detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.vision.features import FEATURE_CFG


@dataclass
class Proposal:
    box: list[float]     # canonical ltrb in input-image pixels
    score: float         # bounded [0, 1] ranking score, not a probability
    area_frac: float
    edge_support: float


def propose(rgb: np.ndarray, cfg: dict = FEATURE_CFG["proposals"]) -> list[Proposal]:
    h, w = rgb.shape[:2]
    s = cfg["long_edge"] / max(h, w)
    small = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    k = cfg["close_ksize"]
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_edges = max(int(np.count_nonzero(edges)), 1)
    out = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area_frac = (bw * bh) / float(sw * sh)
        aspect = bw / float(bh) if bh else 0.0
        if not (cfg["min_area_frac"] <= area_frac <= cfg["max_area_frac"]):
            continue
        if not (cfg["min_aspect"] <= aspect <= cfg["max_aspect"]):
            continue
        support = np.count_nonzero(edges[y:y + bh, x:x + bw]) / total_edges
        # bounded score: geometric mean of edge support and a area term peaking near 50% of the frame
        area_term = 1.0 - abs(area_frac - 0.5) / 0.5
        score = float(np.sqrt(max(support, 0.0) * max(area_term, 0.0)))
        if score < cfg["min_score"]:
            continue
        box = [x / s, y / s, min(w, (x + bw) / s), min(h, (y + bh) / s)]
        out.append(Proposal(box=[float(v) for v in box], score=round(score, 6),
                            area_frac=round(area_frac, 6), edge_support=round(float(support), 6)))
    out.sort(key=lambda p: (-p.score, p.box))
    return out[: cfg["max_proposals"]]
