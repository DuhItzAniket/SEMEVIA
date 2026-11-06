"""Subgroup summaries and rule-based failure selection from saved evaluator rows.

Failure cases are chosen by fixed rules (not by hand), so the selection cannot favour flattering
examples. Each rule returns at most one row; ties are broken by image key.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.models.heads import softmax
from src.vision.coordinates import iou


def _top(z, k=5):
    z = np.asarray(z)
    return sorted(range(len(z)), key=lambda c: (-z[c], c))[:k]


def subgroup_table(rows: list[dict], gts: dict, labels: dict, zkey: str) -> dict:
    """labels: image_key -> group name. Returns per-group counts and end-to-end top-1/top-5."""
    groups = defaultdict(lambda: {"count": 0, "top1": 0, "top5": 0})
    for r in rows:
        g = labels.get(r["image_key"])
        if g is None:
            continue
        cell = groups[g]
        cell["count"] += 1
        if r.get("sel_box") is None or zkey not in r or iou(r["sel_box"], gts[r["image_key"]]) < 0.5:
            continue
        t5 = _top(r[zkey])
        cell["top1"] += int(t5[0] == r["y"])
        cell["top5"] += int(r["y"] in t5)
    return {g: {**v, "top1_rate": v["top1"] / v["count"], "top5_rate": v["top5"] / v["count"]} for g, v in sorted(groups.items())}


def select_failures(rows: list[dict], gts: dict, classes: dict, zkey: str, T: float = 1.0) -> list[dict]:
    """Rules: (1) localisation failure: no box, else lowest IoU < 0.5; (2) close-category error: correct box,
    wrong top-1 that shares the true make, highest confidence; (3) confident cross-make error: correct box,
    wrong top-1 with a different make, highest confidence."""
    out = []
    loc = [r for r in rows if r.get("sel_box") is None or iou(r["sel_box"], gts[r["image_key"]]) < 0.5]
    if loc:
        pick = sorted(loc, key=lambda r: (r.get("sel_box") is not None,
                                          iou(r["sel_box"], gts[r["image_key"]]) if r.get("sel_box") else 0.0, r["image_key"]))[0]
        out.append({"rule": "localisation failure (no box, or lowest IoU below 0.5)", "row": pick})
    matched = [r for r in rows if r.get("sel_box") is not None and zkey in r and iou(r["sel_box"], gts[r["image_key"]]) >= 0.5]
    wrong = []
    for r in matched:
        p = softmax(np.array([r[zkey]]), T)[0]
        t = int(np.argmax(p))
        if t != r["y"]:
            wrong.append((r, t, float(p[t])))
    same = [w for w in wrong if classes[w[1]]["make"] == classes[w[0]["y"]]["make"]]
    cross = [w for w in wrong if classes[w[1]]["make"] != classes[w[0]["y"]]["make"]]
    for name, pool in (("close-category error (same make), highest confidence", same),
                       ("confident cross-make error, highest confidence", cross)):
        if pool:
            r, t, conf = sorted(pool, key=lambda w: (-w[2], w[0]["image_key"]))[0]
            out.append({"rule": name, "row": r, "predicted": t, "confidence": conf})
    return out
