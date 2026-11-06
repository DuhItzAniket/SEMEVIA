"""Temperature, reject-threshold selection and final conformal scores (pure, fixture-tested).

* Temperature: one positive T minimising NLL on matched-primary tuning rows (runtime box IoU >= 0.5).
* Reject threshold: from a fixed grid, minimise negative false acceptance subject to known retention
  >= target; without a negative set, take the largest threshold meeting the retention target and say so.
* Final conformal scores: s = 1 - p_true on every calibration input through the frozen pipeline; an
  input with no usable detection/prediction gets the frozen uniform fallback (s = 1 - 1/196).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
from scipy.optimize import minimize_scalar

from src.models.heads import softmax


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-300, None))))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y)
    if len(y) == 0:
        raise ValueError("no rows to fit temperature")
    r = minimize_scalar(lambda lt: nll(softmax(z, float(np.exp(lt))), y), bounds=(np.log(0.05), np.log(20.0)),
                        method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def choose_reject_threshold(known_conf: np.ndarray, grid: list[float], retention_target: float,
                            neg_conf: np.ndarray | None = None) -> dict:
    known_conf = np.asarray(known_conf, dtype=np.float64)
    table = []
    for t in grid:
        row = {"threshold": t, "known_retention": float((known_conf >= t).mean())}
        if neg_conf is not None and len(neg_conf):
            row["negative_false_acceptance"] = float((np.asarray(neg_conf) >= t).mean())
        table.append(row)
    feasible = [r for r in table if r["known_retention"] >= retention_target]
    if not feasible:
        best = max(table, key=lambda r: (r["known_retention"], -r["threshold"]))
        return {"threshold": best["threshold"], "feasible": False, "table": table, "negatives_used": neg_conf is not None,
                "rule": "retention target infeasible; highest-retention grid value used"}
    if neg_conf is not None and len(neg_conf):
        best = min(feasible, key=lambda r: (r["negative_false_acceptance"], -r["known_retention"], r["threshold"]))
        rule = "minimise negative false acceptance subject to known retention >= target"
    else:
        best = max(feasible, key=lambda r: r["threshold"])
        rule = "no negative tuning set: largest threshold meeting the known-retention target"
    return {"threshold": best["threshold"], "feasible": True, "table": table, "negatives_used": neg_conf is not None, "rule": rule}


def final_conformal_scores(rows: list[dict], zkey: str, T: float, n_classes: int = 196) -> tuple[np.ndarray, int]:
    """rows: evaluator rows with 'y', 'sel_box' and raw logits under zkey. Returns scores and failure count."""
    scores, failures = [], 0
    for r in rows:
        if r.get("sel_box") is None or zkey not in r:
            scores.append(1.0 - 1.0 / n_classes)
            failures += 1
            continue
        p = softmax(np.array([r[zkey]], dtype=np.float64), T)[0]
        scores.append(1.0 - float(p[int(r["y"])]))
    s = np.array(scores)
    if not np.isfinite(s).all():
        raise ValueError("non-finite calibration scores")
    return s, failures


def model_fingerprint(manifest: dict, production_mode: str, encoder_runtime: str) -> str:
    """Hash of every asset that changes predictions (detector, backbone, heads, ONNX if used)."""
    keep = sorted((a["name"], a["sha256"]) for a in manifest["assets"]
                  if a["name"].endswith((".pt", ".pth", ".joblib")) or (encoder_runtime == "onnx_fp32" and a["name"].endswith(".onnx")))
    return hashlib.sha256(json.dumps({"assets": keep, "mode": production_mode, "runtime": encoder_runtime}).encode()).hexdigest()
