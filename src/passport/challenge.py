"""Six-region contrastive Evidence Challenge.

Freeze the original crop, leading class A and runner-up B. Split the crop into 2 rows x 3
columns (floor-rounded internal boundaries, last boundary = crop size). Each of six variants
starts from the untouched crop and replaces only its own cell with the matching pixels of a
deterministically blurred copy. Every variant goes through the same production scoring
function (which must recompute every branch, including handcrafted features).

delta_r = (zA - zB)_original - (zA - zB)_variant, in raw pre-temperature logit units.
Positive: hiding the cell reduced A's advantage over B. This is perturbation sensitivity, not
proof that a region identifies a manufacturer; cells are not semantic parts.
"""
from __future__ import annotations

import time
from typing import Callable

import cv2
import numpy as np

CELL_LABELS = ["upper left", "upper center", "upper right", "lower left", "lower center", "lower right"]
BLUR_RECIPE = {"type": "gaussian", "sigma_frac": 0.08, "min_sigma": 3.0, "border": "reflect101"}
DIAGNOSTIC_LABEL = "diagnostic stress input; clean calibration does not extend to these variants"


def grid_cells(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Return six (l, t, r, b) cells, row-major. Raises if the crop cannot form six nonempty cells."""
    if width < 3 or height < 2:
        raise ValueError("crop_too_small")
    xs = [0, width // 3, (2 * width) // 3, width]
    ys = [0, height // 2, height]
    return [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(2) for c in range(3)]


def blurred_copy(crop: np.ndarray, recipe: dict = BLUR_RECIPE) -> np.ndarray:
    sigma = max(recipe["min_sigma"], recipe["sigma_frac"] * min(crop.shape[:2]))
    return cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)


def make_variants(crop: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    h, w = crop.shape[:2]
    cells = grid_cells(w, h)
    blur = blurred_copy(crop)
    variants = []
    for (l, t, r, b) in cells:
        v = crop.copy()
        v[t:b, l:r] = blur[t:b, l:r]
        variants.append(v)
    return variants, cells


def summarize(deltas: list[float]) -> tuple[int | None, str]:
    positive = [(d, i) for i, d in enumerate(deltas) if d > 0]
    if not positive:
        return None, "No supporting region found by this test."
    best = max(positive, key=lambda t: (t[0], -t[1]))[1]  # ties -> lower grid index
    return best, (f"Hiding the {CELL_LABELS[best]} region reduced the original A-versus-B margin the most "
                  f"(by {deltas[best]:.3f} logit units) under this blur perturbation.")


def run_challenge(crop: np.ndarray, score_fn: Callable[[list[np.ndarray]], np.ndarray],
                  class_a: int, class_b: int, original_logits: np.ndarray) -> dict:
    """score_fn maps a list of crops to an (N, C) array of raw logits using the production head."""
    t0 = time.perf_counter()
    if class_a == class_b:
        raise ValueError("A and B must differ")
    variants, cells = make_variants(crop)
    z = np.asarray(score_fn(variants), dtype=np.float64)
    if z.shape[0] != 6 or not np.isfinite(z).all():
        raise RuntimeError("challenge scoring returned invalid logits")
    zo = np.asarray(original_logits, dtype=np.float64)
    m0 = float(zo[class_a] - zo[class_b])
    out_cells = []
    deltas = []
    for i, (cell, zr) in enumerate(zip(cells, z)):
        mr = float(zr[class_a] - zr[class_b])
        d = m0 - mr
        deltas.append(d)
        order = sorted(range(zr.shape[0]), key=lambda c: (-zr[c], c))
        e = np.exp(zr - zr.max())
        out_cells.append({"index": i, "label": CELL_LABELS[i], "bounds": list(cell), "m_r": mr, "delta": d,
                          "new_top1": int(order[0]), "new_top1_prob": float(e[order[0]] / e.sum())})
    selected, summary = summarize(deltas)
    return {"state": "complete", "class_a": int(class_a), "class_b": int(class_b), "m0": m0,
            "blur_recipe": BLUR_RECIPE, "cells": out_cells, "selected_cell": selected, "summary": summary,
            "small_margin": abs(m0) < 0.01, "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "label": DIAGNOSTIC_LABEL, "variants": variants}
