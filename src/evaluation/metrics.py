"""Evaluation metrics with explicit numerators and denominators.

Every function returns plain dicts carrying the counts used, so reports can print the
population next to each number. A zero denominator yields value None ("not applicable").
"""
from __future__ import annotations

import math

import numpy as np

from src.vision.coordinates import iou


def ratio(num: float, den: float) -> dict:
    return {"numerator": num, "denominator": den, "value": (num / den) if den else None}


def ranked(logits_row: np.ndarray, k: int) -> list[int]:
    z = np.asarray(logits_row)
    return sorted(range(z.shape[0]), key=lambda c: (-z[c], c))[:k]


def classification_topk(logits: np.ndarray, y: np.ndarray, k: int) -> dict:
    hits = sum(int(y[i] in ranked(logits[i], k)) for i in range(len(y)))
    return ratio(hits, len(y))


def end_to_end_topk(selected_boxes: list, gt_boxes: list, topk_ids: list, y: list, k: int, iou_thr: float = 0.5) -> dict:
    """Success needs a selected finite box with IoU >= thr AND the true class in its top-k.
    No detection, wrong target or failed classification count as failures (topk_ids None)."""
    n, hits = len(y), 0
    for sb, gb, ids, yi in zip(selected_boxes, gt_boxes, topk_ids, y):
        if sb is None or ids is None:
            continue
        if iou(sb, gb) >= iou_thr and yi in list(ids)[:k]:
            hits += 1
    return ratio(hits, n)


def localization(selected_boxes: list, gt_boxes: list, iou_thr: float = 0.5) -> dict:
    ious, missing, invalid = [], 0, 0
    for sb, gb in zip(selected_boxes, gt_boxes):
        if sb is None:
            missing += 1
            continue
        if not all(math.isfinite(v) for v in sb) or sb[2] <= sb[0] or sb[3] <= sb[1]:
            invalid += 1
            continue
        ious.append(iou(sb, gb))
    n = len(gt_boxes)
    valid = len(ious)
    wrong = sum(1 for v in ious if v < iou_thr)
    return {
        "population_count": n,
        "mean_iou_valid_selected": ratio(float(np.sum(ious)), valid),
        "mean_iou_zero_for_failures": ratio(float(np.sum(ious)), n),
        "primary_recall_at_iou": ratio(sum(1 for v in ious if v >= iou_thr), n),
        "iou_threshold": iou_thr, "missing": missing, "invalid": invalid, "wrong_target_below_thr": wrong,
    }


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> dict:
    """Count-weighted ECE over equal-width bins [0,0.1),...,[0.9,1]. Empty bins report count 0."""
    c = np.asarray(confidences, dtype=np.float64)
    a = np.asarray(correct, dtype=np.float64)
    n = len(c)
    bins = []
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (c >= lo) & ((c < hi) if b < n_bins - 1 else (c <= hi))
        cnt = int(m.sum())
        if cnt:
            acc, conf = float(a[m].mean()), float(c[m].mean())
            total += cnt / n * abs(acc - conf)
            bins.append({"bin": [lo, hi], "count": cnt, "accuracy": acc, "mean_confidence": conf})
        else:
            bins.append({"bin": [lo, hi], "count": 0, "accuracy": None, "mean_confidence": None})
    return {"ece": total if n else None, "n": n, "bins": bins}


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(probs)[np.arange(len(y)), y], 1e-300, None)
    return float(-np.mean(np.log(p)))


def rejection_contingency(known_accepted: int, known_accepted_correct: int, known_rejected: int,
                          neg_accepted: int, neg_rejected: int) -> dict:
    return {
        "known_retention": ratio(known_accepted, known_accepted + known_rejected),
        "accepted_known_accuracy": ratio(known_accepted_correct, known_accepted),
        "negative_false_acceptance": ratio(neg_accepted, neg_accepted + neg_rejected),
        "note": "End-to-end accuracy over all known inputs is computed from pre-rejection outputs, not from these counts.",
    }


def set_metrics(sets: list[list[int]], y: list[int], specific_flags: list[bool] | None = None) -> dict:
    sizes = np.array([len(s) for s in sets])
    covered = sum(int(yi in set(s)) for s, yi in zip(sets, y))
    singles = [i for i, s in enumerate(sets) if len(s) == 1]
    out = {"coverage": ratio(covered, len(y)), "mean_set_size": float(sizes.mean()) if len(sizes) else None,
           "median_set_size": float(np.median(sizes)) if len(sizes) else None,
           "singleton": ratio(len(singles), len(y)),
           "singleton_accuracy": ratio(sum(int(sets[i][0] == y[i]) for i in singles), len(singles)),
           "empty": ratio(int((sizes == 0).sum()), len(y))}
    if specific_flags is not None:
        out["specific_answer_rate"] = ratio(sum(specific_flags), len(y))
    return out


def coco_ap(gts: list[dict], preds: list[dict], category_ids: list[int]) -> dict:
    """Standard COCO bbox AP via pycocotools. gts: {image_id, category_id, box(ltrb)};
    preds: {image_id, category_id, box(ltrb), score}. Returns AP50 and AP50:95 with settings."""
    import contextlib
    import io

    import pycocotools
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    def xywh(b):
        return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]

    image_ids = sorted({g["image_id"] for g in gts})
    gt = {"images": [{"id": i} for i in image_ids],
          "categories": [{"id": c} for c in category_ids],
          "annotations": [{"id": k + 1, "image_id": g["image_id"], "category_id": g["category_id"], "bbox": xywh(g["box"]),
                           "area": (g["box"][2] - g["box"][0]) * (g["box"][3] - g["box"][1]), "iscrowd": 0}
                          for k, g in enumerate(gts)]}
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO()
        cg.dataset = gt
        cg.createIndex()
        if not preds:
            return {"AP50": 0.0, "AP50_95": 0.0, "num_predictions": 0, "evaluator": f"pycocotools {pycocotools.__version__ if hasattr(pycocotools, '__version__') else ''}"}
        cd = cg.loadRes([{"image_id": p["image_id"], "category_id": p["category_id"], "bbox": xywh(p["box"]),
                          "score": float(p["score"])} for p in preds])
        ev = COCOeval(cg, cd, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"AP50_95": float(ev.stats[0]), "AP50": float(ev.stats[1]), "num_predictions": len(preds),
            "num_ground_truth": len(gts), "settings": {"iouThrs": "0.50:0.05:0.95", "recThrs": 101,
                                                      "maxDets": list(ev.params.maxDets), "areaRng": "all"},
            "evaluator": "pycocotools COCOeval bbox"}
