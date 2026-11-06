"""Linear heads over cached representations, with a canonical 196-class logit order."""
from __future__ import annotations

import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

N_LEARNED = 1280
HC_GROUPS = {"hog": (0, 1764), "lbp": (1764, 1774), "hsv": (1774, 1806), "geometry": (1806, 1814), "flags": (1814, 1816)}


class FusionFeatures(BaseEstimator, TransformerMixin):
    """Input rows: [learned 1280 | handcrafted 1816]. Output: standardized learned branch and/or
    standardized handcrafted groups -> PCA (<= pca_components, capped by rank) -> standardized.
    branch_balance "equal_variance" rescales the handcrafted branch so both branches carry equal
    total variance; "per_feature" leaves every output feature at unit variance. All transforms fit
    on training rows. Order (fixed): standardize -> PCA -> standardize -> branch scale."""

    def __init__(self, use_learned: bool = True, hc_groups: tuple = (), pca_components: int = 64,
                 branch_balance: str = "equal_variance"):
        self.use_learned = use_learned
        self.hc_groups = hc_groups
        self.pca_components = pca_components
        self.branch_balance = branch_balance   # "equal_variance" | "per_feature"

    def _hc(self, X):
        cols = np.concatenate([np.arange(*HC_GROUPS[g]) + N_LEARNED for g in self.hc_groups])
        return X[:, cols]

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if self.use_learned:
            self.learned_scaler_ = StandardScaler().fit(X[:, :N_LEARNED])
        if self.hc_groups:
            H = self._hc(X)
            self.hc_scaler_ = StandardScaler().fit(H)
            Hs = self.hc_scaler_.transform(H)
            k = int(min(self.pca_components, np.linalg.matrix_rank(Hs[: min(len(Hs), 4000)]), Hs.shape[1]))
            self.pca_ = PCA(n_components=k, random_state=0).fit(Hs)
            self.pca_scaler_ = StandardScaler().fit(self.pca_.transform(Hs))
            if self.branch_balance not in ("equal_variance", "per_feature"):
                raise ValueError("branch_balance must be equal_variance or per_feature")
            equal = self.branch_balance == "equal_variance" and self.use_learned
            self.hc_scale_ = float(np.sqrt(N_LEARNED / k)) if equal else 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        parts = []
        if self.use_learned:
            parts.append(self.learned_scaler_.transform(X[:, :N_LEARNED]))
        if self.hc_groups:
            Z = self.pca_scaler_.transform(self.pca_.transform(self.hc_scaler_.transform(self._hc(X))))
            parts.append(Z * self.hc_scale_)
        return np.concatenate(parts, axis=1)


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
