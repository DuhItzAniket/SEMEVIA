"""Recognition contract through the real API (skipped when the recognition bundle is absent)."""
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SEMEVIA_DATA_ROOT", "D:/semevia_work/raw/StanfordCars/stanford_cars"))
IMG = DATA_ROOT / "cars_test" / "00046.jpg"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        r = c.get("/readyz").json()
        if not r.get("recognition"):
            pytest.skip("recognition bundle not present in this build")
        yield c


@pytest.mark.skipif(not IMG.exists(), reason="dataset not available")
@pytest.mark.parametrize("mode", ["classical", "deep", "hybrid"])
def test_modes_return_genuine_196_class_output(client, mode):
    j = client.post("/predict", files={"file": ("car.jpg", IMG.read_bytes(), "image/jpeg")}, data={"mode": mode}).json()
    assert j["active_mode"] == mode and j["status"] in ("ok", "uncertain", "no_vehicle")
    if j["status"] == "no_vehicle":
        assert j["prediction"]["top5"] == []
        return
    p = j["prediction"]
    assert p["available"] and len(p["probabilities"]) == 196
    assert abs(sum(p["probabilities"]) - 1) < 1e-3
    top = [c["class_id"] for c in p["top5"]]
    assert len(set(top)) == 5 and top[0] == int(np.argmax(p["probabilities"]))
    assert all(c["original_class_id"] == c["class_id"] + 1 for c in p["top5"])
    pp = j["passport"]
    if mode == j["production_mode"]:
        assert pp["decision_state"] in ("exact_candidate", "ambiguous", "rejected", "inadequate")
        assert pp["candidate_count"] == len(pp["candidate_set"]) and p["calibrated"]
        if pp["decision_state"] in ("rejected", "inadequate", "ambiguous"):
            assert pp["accepted_identity"] is None
    else:
        assert pp["decision_state"] == "comparison_only" and pp["calibration"] is None and not p["calibrated"]
    imgs = j["processing"]["images"]
    for k in ("crop.jpg", "gray.png", "edges.png", "contours.png"):
        assert client.get(imgs[k]).status_code == 200
    if mode == "classical":
        assert "classical_proposal" in j["detection"]


@pytest.mark.skipif(not IMG.exists(), reason="dataset not available")
def test_challenge_six_real_variants_and_cached_repeat(client):
    j = client.post("/predict", files={"file": ("car.jpg", IMG.read_bytes(), "image/jpeg")}, data={"mode": "hybrid"}).json()
    rid = j["request_id"]
    c = client.post("/challenge", json={"request_id": rid}).json()
    assert c["state"] == "complete" and len(c["cells"]) == 6
    assert c["class_a"] == j["challenge"]["class_a"] and c["class_b"] == j["challenge"]["class_b"]
    for cell in c["cells"]:
        assert cell["delta"] == pytest.approx(c["m0"] - cell["m_r"])
        assert client.get(cell["image_url"]).status_code == 200
    again = client.post("/challenge", json={"request_id": rid}).json()
    assert again["cells"] == c["cells"]                              # reuses this request's measured result
    # original prediction still retrievable / unchanged
    assert client.get(j["processing"]["images"]["crop.jpg"]).status_code == 200


def test_challenge_unknown_request_is_expired(client):
    r = client.post("/challenge", json={"request_id": "nope"})
    assert r.status_code == 404 and r.json()["reason"] == "expired"


@pytest.mark.skipif(not IMG.exists(), reason="dataset not available")
def test_references_are_training_only_and_labelled(client):
    ready = client.get("/readyz").json()
    if not ready.get("references"):
        pytest.skip(ready.get("references_reason"))
    j = client.post("/predict", files={"file": ("car.jpg", IMG.read_bytes(), "image/jpeg")}, data={"mode": "hybrid"}).json()
    r = j["references"]
    assert r["state"] == "available" and len(r["items"]) == 3
    assert all(it["sample_key"].startswith("train/") for it in r["items"])
    assert all(it["thumbnail"] is None for it in r["items"])
    sims = [it["cosine_similarity"] for it in r["items"]]
    assert sims == sorted(sims, reverse=True)
