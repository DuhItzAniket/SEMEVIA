import numpy as np
import pytest

from src.passport.references import build_index, query, validate_index


def _data(n_per=6, classes=4, d=16, seed=0):
    rng = np.random.RandomState(seed)
    keys, y, X = [], [], []
    for c in range(classes):
        centre = rng.randn(d)
        for i in range(n_per):
            keys.append(f"train/{c:03d}_{i:02d}.jpg")
            y.append(c)
            X.append(centre + 0.1 * rng.randn(d))
    return np.array(keys), np.array(y), np.array(X)


def test_three_per_class_nearest_centroid_and_query_order():
    k, y, X = _data()
    idx = build_index(k, y, X, "v1")
    assert len(idx["keys"]) == 12 and np.bincount(idx["labels"]).max() == 3
    validate_index(idx, set(k), set())
    r = query(idx, X[0], "v1")
    assert r["state"] == "available" and len(r["items"]) == 3
    sims = [i["cosine_similarity"] for i in r["items"]]
    assert sims == sorted(sims, reverse=True) and r["items"][0]["class_id"] == 0


def test_inserted_non_training_key_fails_validation():
    k, y, X = _data()
    idx = build_index(k, y, X, "v1")
    idx["keys"] = idx["keys"].copy()
    idx["keys"][0] = "test/00001.jpg"
    with pytest.raises(ValueError):
        validate_index(idx, set(k), {"test/00001.jpg"})
    with pytest.raises(ValueError):
        validate_index(idx, set(k), set())          # outside the training manifest


def test_version_mismatch_and_bad_query_disable_retrieval():
    k, y, X = _data()
    idx = build_index(k, y, X, "v1")
    assert query(idx, X[0], "v2")["state"] == "unavailable"
    assert query(idx, np.zeros(16), "v1")["state"] == "unavailable"
    assert query(idx, np.full(16, np.nan), "v1")["state"] == "unavailable"
