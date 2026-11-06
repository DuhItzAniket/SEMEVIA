"""Passport decision state with deterministic precedence.

Precedence: operational outcomes (handled by the caller) -> detection_failed ->
comparison_only (no matching calibration) -> rejected -> inadequate -> exact_candidate /
ambiguous. All applicable reason codes are kept. The best guess, the reject decision and the
conformal set are independent objects: rejection never removes labels from the set.
"""
from __future__ import annotations

import numpy as np

from src.passport.conformal import ConformalQuantile, candidate_set, failure_set
from src.passport.hierarchy import Hierarchy


def make_passport(probs: np.ndarray | None, detection_ok: bool, calibration: dict | None,
                  cq: ConformalQuantile | None, reject_threshold: float | None, hierarchy: Hierarchy,
                  n_classes: int = 196) -> dict:
    reasons: list[str] = []
    if not detection_ok or probs is None:
        reasons.append("no_eligible_detection" if not detection_ok else "no_prediction")
        cands = failure_set(n_classes)
        spec = hierarchy.specificity(cands, is_failure=True)
        return {"decision_state": "detection_failed", "reason_codes": reasons, "accepted_identity": None,
                "candidate_set": cands, "candidate_count": len(cands), "candidate_set_is_failure_convention": True,
                "specificity": spec.to_dict(), "calibration": calibration, "max_probability": None}
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (n_classes,) or not np.isfinite(p).all():
        raise ValueError("probabilities must be finite with one value per class")
    if calibration is None or cq is None:
        return {"decision_state": "comparison_only", "reason_codes": ["no_matching_calibration_bundle"],
                "accepted_identity": None, "candidate_set": [], "candidate_count": 0,
                "specificity": None, "calibration": None, "max_probability": float(p.max())}
    cands = candidate_set(p, cq)
    spec = hierarchy.specificity(cands)
    top = int(sorted(range(n_classes), key=lambda c: (-p[c], c))[0])
    pmax = float(p[top])
    rejected = reject_threshold is not None and pmax < reject_threshold
    if rejected:
        reasons.append("max_probability_below_reject_threshold")
    if not cands:
        reasons.append("empty_candidate_set")
    elif not hierarchy.is_specific(spec):
        reasons.append("no_shared_make" if spec.level == "root" else f"specificity_{spec.level}")
    if rejected:
        state = "rejected"
    elif not cands or not hierarchy.is_specific(spec):
        state = "inadequate"
    elif len(cands) == 1:
        state = "exact_candidate"
    else:
        state = "ambiguous"
    accepted = None
    if state == "exact_candidate":
        accepted = int(cands[0])
    return {"decision_state": state, "reason_codes": reasons, "accepted_identity": accepted,
            "candidate_set": cands, "candidate_count": len(cands), "candidate_set_is_failure_convention": False,
            "specificity": spec.to_dict(), "calibration": calibration, "max_probability": pmax,
            "best_guess": top}
