"""End-to-end upload contract tests against the real app and real YOLO11n weights."""
import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SEMEVIA_DATA_ROOT", "D:/semevia_work/raw/StanfordCars/stanford_cars"))
WEIGHTS = ROOT / "artifacts" / "store" / "release" / "yolo11n.pt"

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="release weights not fetched")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 200
        yield c


def _png(arr: np.ndarray, mode=None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _post(client, data: bytes, name="x.jpg"):
    return client.post("/predict", files={"file": (name, data, "application/octet-stream")})


def test_health_and_ready(client):
    assert client.get("/healthz").json()["status"] == "alive"
    r = client.get("/readyz").json()
    assert r["ready"] and isinstance(r["recognition"], bool)
    if not r["recognition"]:
        assert r["modes"] == [] and r["recognition_reason"]


@pytest.mark.skipif(not (DATA_ROOT / "cars_test").is_dir(), reason="dataset not available")
def test_vehicle_image_is_detected(client):
    data = (DATA_ROOT / "cars_test" / "00001.jpg").read_bytes()
    r = _post(client, data)
    assert r.status_code == 200
    j = r.json()
    assert j["schema_version"] == "semevia.v1" and j["status"] == "ok"
    d = j["detection"]
    assert d["state"] == "detected" and d["detector_score"] >= d["threshold"]
    l, t, rr, b = d["selected_box"]
    assert 0 <= l < rr <= j["input"]["width"] and 0 <= t < b <= j["input"]["height"]
    if not j["model_status"]["recognition"]:
        # Without the recognition bundle there is no fabricated fine-grained output
        assert j["prediction"]["available"] is False and j["prediction"]["top5"] == []
        assert j["passport"]["accepted_identity"] is None
    img = client.get(j["processing"]["images"]["crop.jpg"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_non_image_file(client):
    r = _post(client, b"this is not an image at all", "notes.jpg")
    assert r.status_code == 400
    assert r.json()["status"] == "invalid_image" and r.json()["error"]["code"] == "invalid_image"


def test_empty_file(client):
    r = _post(client, b"")
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_file"


def test_oversized_file(client):
    r = _post(client, b"\xff" * (10 * 1024 * 1024 + 10))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_too_many_pixels(client):
    r = _post(client, _png(np.zeros((5100, 5000, 3), np.uint8)), "big.png")
    assert r.status_code == 400 and r.json()["error"]["code"] == "too_many_pixels"


def test_blank_image_no_vehicle(client):
    r = _post(client, _png(np.full((480, 640, 3), 200, np.uint8)), "blank.png")
    j = r.json()
    assert r.status_code == 200 and j["status"] == "no_vehicle"
    assert j["detection"]["selected_box"] is None
    assert j["passport"]["decision_state"] in ("detection_failed", "unavailable")
    assert j["prediction"]["top5"] == []


def test_grayscale_and_transparent_inputs(client):
    g = _post(client, _png(np.full((300, 400), 90, np.uint8), mode="L"), "g.png")
    assert g.status_code == 200 and g.json()["input"]["width"] == 400
    rgba = np.zeros((300, 400, 4), np.uint8)
    rgba[..., 3] = 0
    t = _post(client, _png(rgba, mode="RGBA"), "t.png")
    assert t.status_code == 200


def test_misleading_extension_is_decoded_by_content(client):
    r = _post(client, _png(np.full((200, 300, 3), 10, np.uint8)), "photo.jpg")
    assert r.status_code == 200 and r.json()["input"]["format"] == "PNG"


def test_large_image_resized_long_edge(client):
    r = _post(client, _png(np.full((1200, 3000, 3), 60, np.uint8)), "wide.png")
    j = r.json()
    assert max(j["input"]["width"], j["input"]["height"]) == 1600
    assert j["input"]["resized_from"] == [3000, 1200]


def test_unknown_request_image_is_404(client):
    r = client.get("/request-image/does-not-exist/display.jpg")
    assert r.status_code == 404 and r.json()["error"]["code"] == "expired"
