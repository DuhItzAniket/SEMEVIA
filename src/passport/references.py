"""Curated training-reference index and nearest-reference retrieval.

Up to three training crops per class, nearest (cosine) to that class's normalised-embedding
centroid, ties broken by sample key: at most 588 entries. Retrieval ranks the whole curated
index by cosine similarity (ties by key) and returns the top three. Similarity is not an
identity probability, and a neighbour is not proof of identity. Only training-partition keys
may enter the index; the query and every tuning/calibration/test image are excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PER_CLASS = 3


def _normalise(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.isfinite(X).all() or (n <= 0).any():
        raise ValueError("non-finite or zero-norm embedding")
    return X / n


def build_index(keys: np.ndarray, y: np.ndarray, X: np.ndarray, embedding_version: str, per_class: int = PER_CLASS) -> dict:
    keys = np.asarray(keys).astype(str)
    Xn = _normalise(np.asarray(X, dtype=np.float64))
    sel = []
    for c in sorted(set(int(v) for v in y)):
        idx = np.nonzero(y == c)[0]
        cen = Xn[idx].mean(0)
        cen /= np.linalg.norm(cen)
        sims = Xn[idx] @ cen
        order = sorted(range(len(idx)), key=lambda i: (-sims[i], keys[idx[i]]))[:per_class]
        sel.extend(int(idx[i]) for i in order)
    sel = np.array(sel)
    emb = Xn[sel].astype(np.float16)
    index = {"keys": keys[sel], "labels": np.asarray(y)[sel].astype(np.int16), "emb": emb,
             "embedding_version": embedding_version}
    index["index_id"] = "refidx_" + hashlib.sha256(emb.tobytes() + "|".join(index["keys"]).encode()).hexdigest()[:12]
    return index


def validate_index(index: dict, train_keys: set[str], forbidden_keys: set[str]) -> None:
    keys = set(str(k) for k in index["keys"])
    if keys & forbidden_keys:
        raise ValueError("reference index contains non-training keys: " + ", ".join(sorted(keys & forbidden_keys))[:200])
    if not keys <= train_keys:
        raise ValueError("reference index contains keys outside the training manifest")
    counts = np.bincount(index["labels"].astype(int), minlength=196)
    if counts.max() > PER_CLASS or len(keys) != len(index["keys"]):
        raise ValueError("reference index violates per-class cap or has duplicate keys")


def save_index(index: dict, path: Path) -> None:
    np.savez(path, keys=index["keys"], labels=index["labels"], emb=index["emb"])
    path.with_suffix(".json").write_text(json.dumps({"index_id": index["index_id"], "embedding_version": index["embedding_version"],
                                                    "entries": int(len(index["keys"])), "per_class": PER_CLASS}, indent=1) + "\n",
                                        encoding="utf-8", newline="\n")


def load_index(path: Path) -> dict:
    d = np.load(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return {"keys": d["keys"].astype(str), "labels": d["labels"], "emb": d["emb"], **meta}


def query(index: dict, q: np.ndarray, query_version: str, k: int = 3) -> dict:
    if query_version != index["embedding_version"]:
        return {"state": "unavailable", "reason": "reference index and query encoder versions differ", "items": []}
    q = np.asarray(q, dtype=np.float64).ravel()
    n = np.linalg.norm(q)
    if not np.isfinite(q).all() or n <= 0:
        return {"state": "unavailable", "reason": "query embedding has zero or non-finite norm", "items": []}
    sims = index["emb"].astype(np.float64) @ (q / n)
    order = sorted(range(len(sims)), key=lambda i: (-sims[i], index["keys"][i]))[:k]
    return {"state": "available", "index_id": index["index_id"], "embedding_version": index["embedding_version"],
            "items": [{"rank": r + 1, "sample_key": str(index["keys"][i]), "class_id": int(index["labels"][i]),
                       "cosine_similarity": round(float(sims[i]), 4)} for r, i in enumerate(order)],
            "note": "Nearest in the curated training-reference library (3 per class). Similarity is not a probability."}
