"""Split-conformal candidate sets (least-ambiguous set-valued classifier score).

Score convention: s = 1 - p_true, with p the final calibrated probabilities. For fixed alpha
take the one-based order statistic k = ceil((n + 1)(1 - alpha)). If k <= n, q = s_(k)
(ascending sort, no interpolation); if k > n the set is the full label set, serialised as
quantile_mode "full_set" with q = null. A label c is included when 1 - p_c <= q (equality
included). Empty sets stay empty. Detection failure returns the full set with a flag.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

SET_CONVENTION = "lac_1_minus_p_le_q_inclusive_empty_kept_v1"


@dataclass(frozen=True)
class ConformalQuantile:
    alpha: float
    n: int
    k: int
    quantile_mode: str          # "order_statistic" | "full_set"
    q: float | None

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "n": self.n, "k": self.k, "quantile_mode": self.quantile_mode,
                "q": self.q, "set_convention": SET_CONVENTION}


def calibrate(scores: np.ndarray, alpha: float) -> ConformalQuantile:
    scores = np.asarray(scores, dtype=np.float64)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    n = int(scores.shape[0])
    if n == 0:
        raise ValueError("calibration needs at least one score")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite calibration scores")
    k = math.ceil((n + 1) * (1.0 - alpha) - 1e-12)
    if k > n:
        return ConformalQuantile(alpha, n, k, "full_set", None)
    q = float(np.sort(scores, kind="stable")[k - 1])
    return ConformalQuantile(alpha, n, k, "order_statistic", q)


def calibration_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return 1.0 - probs[np.arange(len(y)), np.asarray(y)]


def candidate_set(probs: np.ndarray, cq: ConformalQuantile) -> list[int]:
    """Members sorted by descending probability, then ascending class id."""
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError("probabilities must be a finite 1-D vector")
    if cq.quantile_mode == "full_set":
        idx = np.arange(p.shape[0])
    else:
        idx = np.nonzero(1.0 - p <= cq.q)[0]
    return [int(i) for i in sorted(idx, key=lambda c: (-p[c], c))]


def failure_set(n_classes: int = 196) -> list[int]:
    """Conservative accounting set for a missing detection / unusable prediction."""
    return list(range(n_classes))


def uniform_failure_probs(n_classes: int = 196) -> np.ndarray:
    """Frozen fallback probability vector for calibration inputs without a usable prediction."""
    return np.full(n_classes, 1.0 / n_classes)
