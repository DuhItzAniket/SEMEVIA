"""Linear heads over cached representations, with a canonical 196-class logit order."""
from __future__ import annotations

import numpy as np


def topk_accuracy(logits: np.ndarray, y: np.ndarray, k: int) -> float:
    # deterministic ranking: descending score, ascending class id on ties
    order = np.lexsort((np.broadcast_to(np.arange(logits.shape[1]), logits.shape), -logits), axis=1)
    return float(np.mean([y[i] in order[i, :k] for i in range(len(y))]))


def canonical_logits(clf, X: np.ndarray, n_classes: int = 196) -> np.ndarray:
    """Return logits with columns in class-id order 0..n-1, verifying sklearn's classes_."""
    z = clf.decision_function(X)
    classes = np.asarray(clf.classes_)
    if classes.shape[0] != n_classes:
        raise ValueError(f"head has {classes.shape[0]} classes, expected {n_classes}")
    out = np.empty_like(z)
    out[:, classes] = z
    return out


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    if not (np.isfinite(T) and T > 0):
        raise ValueError("temperature must be finite and positive")
    a = z / T
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)
