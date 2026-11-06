"""Handcrafted feature extractor (frozen contract in configs/features.yaml).

Vector layout, in order: HOG (1764) | LBP (10) | HSV (32) | geometry (8) | flags (2) = 1816.
Geometry is measured on an aspect-preserving resize, before any square resize, so the crop
aspect ratio is meaningful. Blank or contour-free crops return finite zeros plus a quality
flag; NaN is never produced. Contour measurements describe the projected image shape,
not physical vehicle dimensions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from skimage.feature import hog, local_binary_pattern

ROOT = Path(__file__).resolve().parents[2]
FEATURE_CFG = yaml.safe_load((ROOT / "configs" / "features.yaml").read_text(encoding="utf-8"))
L = FEATURE_CFG["lengths"]
SLICES = {
    "hog": slice(0, L["hog"]),
    "lbp": slice(L["hog"], L["hog"] + L["lbp"]),
    "hsv": slice(L["hog"] + L["lbp"], L["hog"] + L["lbp"] + L["hsv"]),
    "geometry": slice(L["total"] - L["flags"] - L["geometry"], L["total"] - L["flags"]),
    "flags": slice(L["total"] - L["flags"], L["total"]),
}


@dataclass
class ClassicalResult:
    vector: np.ndarray                       # float32, shape (1816,)
    measurements: dict                       # named geometry values and flags
    views: dict = field(default_factory=dict)  # name -> uint8 image (only when requested)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def _resize_long(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_edge / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def extract_classical(crop_rgb: np.ndarray, box: list[float] | None = None,
                      image_size: tuple[int, int] | None = None, cfg: dict = FEATURE_CFG,
                      want_views: bool = False) -> ClassicalResult:
    """crop_rgb: HxWx3 uint8 crop. box/image_size (w, h): crop location in the source image."""
    if crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3 or crop_rgb.size == 0:
        raise ValueError("crop must be a non-empty HxWx3 array")
    ch, cw = crop_rgb.shape[:2]
    gray_full = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)

    # --- HOG and LBP on the square ROI
    rs = cfg["roi_size"]
    roi = cv2.resize(gray_full, (rs, rs), interpolation=cv2.INTER_AREA)
    hc = cfg["hog"]
    v_hog = hog(roi, orientations=hc["orientations"], pixels_per_cell=(hc["pixels_per_cell"],) * 2,
                cells_per_block=(hc["cells_per_block"],) * 2, block_norm=hc["block_norm"], feature_vector=True)
    lc = cfg["lbp"]
    lbp = local_binary_pattern(roi, lc["points"], lc["radius"], method=lc["method"])
    n_lbp = lc["points"] + 2
    v_lbp = np.bincount(lbp.astype(np.int64).ravel(), minlength=n_lbp)[:n_lbp].astype(np.float64)
    v_lbp /= max(v_lbp.sum(), 1.0)

    # --- HSV histograms (OpenCV hue range 0..180), each channel normalised to sum 1
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    hb = cfg["hsv"]
    parts = []
    for c, bins, rng in ((0, hb["h_bins"], 180), (1, hb["s_bins"], 256), (2, hb["v_bins"], 256)):
        hist = cv2.calcHist([hsv], [c], None, [bins], [0, rng]).ravel().astype(np.float64)
        parts.append(hist / max(hist.sum(), 1.0))
    v_hsv = np.concatenate(parts)

    # --- Edges and contours on an aspect-preserving resize
    g = _resize_long(gray_full, cfg["geometry_long_edge"])
    k = cfg["smoothing"]["gaussian_ksize"]
    blurred = cv2.GaussianBlur(g, (k, k), 0)
    edges = cv2.Canny(blurred, cfg["canny"]["low"], cfg["canny"]["high"])
    mk = cfg["morphology"]["close_ksize"]
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gh, gw = g.shape[:2]
    roi_area = float(gh * gw)
    near_uniform = float(gray_full.std()) < cfg["near_uniform_std"]
    largest = None
    if contours:
        # deterministic: largest area, then largest perimeter, then lowest top-left
        largest = max(contours, key=lambda c: (cv2.contourArea(c), cv2.arcLength(c, True),
                                               -cv2.boundingRect(c)[1], -cv2.boundingRect(c)[0]))
        if cv2.contourArea(largest) <= 0:
            largest = None
    empty = largest is None
    if empty:
        area_frac = solidity = extent = 0.0
    else:
        area = cv2.contourArea(largest)
        hull_area = cv2.contourArea(cv2.convexHull(largest))
        bx, by, bw, bh = cv2.boundingRect(largest)
        area_frac = _safe_div(area, roi_area)
        solidity = _safe_div(area, hull_area)
        extent = _safe_div(area, bw * bh)
    edge_density = _safe_div(int(np.count_nonzero(edges)), roi_area)

    if box is not None and image_size is not None:
        iw, ih = image_size
        bwid, bhei = box[2] - box[0], box[3] - box[1]
        crop_aspect = _safe_div(bwid, bhei)
        box_area_frac = _safe_div(bwid * bhei, iw * ih)
        cx = _safe_div((box[0] + box[2]) / 2.0, iw)
        cy = _safe_div((box[1] + box[3]) / 2.0, ih)
    else:
        crop_aspect, box_area_frac, cx, cy = _safe_div(cw, ch), 1.0, 0.5, 0.5
    geometry = [crop_aspect, area_frac, solidity, extent, edge_density, box_area_frac, cx, cy]
    flags = [float(empty), float(near_uniform)]

    vec = np.concatenate([v_hog, v_lbp, v_hsv, np.asarray(geometry), np.asarray(flags)]).astype(np.float32)
    if vec.shape[0] != L["total"]:
        raise RuntimeError(f"feature length {vec.shape[0]} != {L['total']}")
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    meas = dict(zip(cfg["geometry_order"], [round(float(x), 5) for x in geometry]))
    meas.update(dict(zip(cfg["flags_order"], [bool(f) for f in flags])))
    views = {}
    if want_views:
        views["gray"] = g
        views["edges"] = edges
        overlay = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(overlay, contours, -1, (255, 214, 10), 1)
        if largest is not None:
            cv2.drawContours(overlay, [largest], -1, (255, 45, 85), 2)
        views["contours"] = overlay
    return ClassicalResult(vector=vec, measurements=meas, views=views)
