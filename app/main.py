"""SEMEVIA web service.

P1 build: genuine YOLO11n vehicle detection, bounded uploads and typed error states.
The fine-grained classifier is not part of this build; responses say so explicitly and
contain no category predictions.
"""
from __future__ import annotations

import asyncio
import io
import os
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).parent / "static"
CFG = yaml.safe_load((ROOT / "configs" / "runtime.yaml").read_text(encoding="utf-8"))
SCHEMA_VERSION = "semevia.v1"
VERSION = "0.1.0-p1"

STATE: dict = {"ready": False, "reasons": ["starting"], "detector": None, "bundle": None, "warmup_ms": None}


class RequestStore:
    """In-memory, request-scoped images with expiry. Never written to disk or logged."""

    def __init__(self, ttl: float, max_items: int = 64):
        self.ttl, self.max_items = ttl, max_items
        self._d: OrderedDict[str, dict] = OrderedDict()

    def put(self, rid: str, record: dict) -> None:
        self._expire()
        self._d[rid] = {"t": time.time(), **record}
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)

    def get(self, rid: str) -> dict | None:
        self._expire()
        return self._d.get(rid)

    def _expire(self) -> None:
        now = time.time()
        for k in [k for k, v in self._d.items() if now - v["t"] > self.ttl]:
            del self._d[k]


STORE = RequestStore(ttl=CFG["server"]["request_ttl_seconds"])
ADMISSION = {"active": 0, "waiting": 0}
SEM = asyncio.Semaphore(CFG["server"]["max_active_jobs"])


def _jpeg(rgb: np.ndarray, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _load_models() -> None:
    import torch

    from src.models.detector import VehicleDetector
    from src.release import load_manifest, verify

    torch.set_num_threads(int(os.environ.get("SEMEVIA_TORCH_THREADS", CFG["server"]["torch_threads"])))
    manifest = load_manifest()
    problems = verify(manifest)
    if problems:
        STATE["reasons"] = problems
        return
    det_cfg = CFG["detector"]
    asset = next(a for a in manifest["assets"] if a["name"] == "yolo11n.pt")
    det = VehicleDetector(ROOT / det_cfg["weights"], det_cfg["imgsz"], det_cfg["conf_floor"], det_cfg["nms_iou"],
                          det_cfg["max_det"], det_cfg["vehicle_coco_classes"], asset["sha256"])
    t = time.perf_counter()
    det.detect(np.full((480, 640, 3), 127, np.uint8))  # real warmup inference before readiness
    STATE["warmup_ms"] = round((time.perf_counter() - t) * 1000, 1)
    STATE.update(detector=det, bundle=manifest["bundle_id"], ready=True, reasons=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(_load_models)
    except Exception as exc:  # readiness reports the failure; the process stays alive for diagnosis
        STATE.update(ready=False, reasons=[f"model load failed: {type(exc).__name__}"])
    yield


app = FastAPI(title="SEMEVIA", version=VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _envelope(rid: str, status: str, mode: str, error: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION, "request_id": rid, "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(), "active_mode": mode,
        "production_mode": None, "error": error, "input": None,
        "detection": {"state": "unavailable"},
        "prediction": {"available": False, "unavailable_reason": "fine-grained classifier not available in this development build (P1)",
                       "top5": [], "probabilities": None, "classifier_score": None, "calibrated": False,
                       "temperature": None, "score_type": None},
        "passport": {"decision_state": "unavailable", "reason_codes": ["classifier_not_available"],
                     "accepted_identity": None, "candidate_set": [], "candidate_count": 0},
        "challenge": {"state": "unavailable"}, "references": {"state": "unavailable", "items": []},
        "cnn_diagnostic": {"state": "unavailable"}, "processing": {},
        "provenance": {"bundle_id": STATE["bundle"], "app_version": VERSION,
                       "detector": STATE["detector"].version if STATE["detector"] else None},
        "timings": {},
    }


def _error(rid: str, status: str, mode: str, code: str, message: str, http: int) -> JSONResponse:
    return JSONResponse(status_code=http, content=_envelope(rid, status, mode, {"code": code, "message": message}))


def _run_pipeline(data: bytes, rid: str, mode: str, t_queue: float) -> dict:
    from src.models.detector import select_target
    from src.vision.coordinates import crop_bounds, expand_box
    from src.vision.decode import decode_image

    up = CFG["upload"]
    timings = {"queue_ms": round(t_queue * 1000, 1)}
    t0 = time.perf_counter()
    img = decode_image(data, up["max_bytes"], up["max_pixels"], up["max_long_edge"], up["allowed_formats"])
    timings["decode_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    t1 = time.perf_counter()
    cands, total, invalid = STATE["detector"].detect(img.rgb)
    thr = CFG["target_policy"]["threshold"]
    sel, sel_i = select_target(cands, thr)
    timings["detection_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    resp = _envelope(rid, "ok", mode)
    resp["input"] = {"sha256": img.sha256, "width": img.width, "height": img.height,
                     "orientation_applied": img.orientation_applied,
                     "resized_from": list(img.original_size) if img.resized else None, "crop_sha256": None,
                     "format": img.source_format}
    images = {"display.jpg": _jpeg(img.rgb)}
    resp["detection"] = {
        "state": "detected" if sel else "no_vehicle",
        "selected_box": sel.box if sel else None, "selected_index": sel_i,
        "detector_score": sel.score if sel else None,
        "candidate_count": len(cands),
        "candidates": [{"box": d.box, "score": round(d.score, 4), "coco_class": d.coco_class} for d in cands],
        "raw_detection_count": total, "invalid_box_count": invalid,
        "selection_rule": CFG["target_policy"]["rule"], "threshold": thr,
        "detector_version": STATE["detector"].version,
    }
    if sel is None:
        resp["status"] = "no_vehicle"
        resp["passport"].update(decision_state="detection_failed", reason_codes=["no_eligible_detection"])
    else:
        box = expand_box(sel.box, CFG["crop"]["margin"], img.width, img.height)
        l, t, r, b = crop_bounds(box, img.width, img.height)
        crop = img.rgb[t:b, l:r]
        cj = _jpeg(crop, 90)
        images["crop.jpg"] = cj
        import hashlib
        resp["input"]["crop_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
        resp["processing"] = {"crop_bounds": [l, t, r, b], "crop_margin": CFG["crop"]["margin"]}
    resp["processing"]["images"] = {k: f"/request-image/{rid}/{k}" for k in images}
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000 + timings["queue_ms"], 1)
    resp["timings"] = timings
    STORE.put(rid, {"images": images})
    return resp


@app.get("/healthz")
def healthz():
    return {"status": "alive", "version": VERSION}


@app.get("/readyz")
def readyz():
    if STATE["ready"]:
        return {"ready": True, "bundle_id": STATE["bundle"], "warmup_ms": STATE["warmup_ms"],
                "detector": STATE["detector"].version, "classifier": None}
    return JSONResponse(status_code=503, content={"ready": False, "reasons": STATE["reasons"]})


@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("hybrid")):
    rid = secrets.token_urlsafe(12)
    mode = mode if mode in ("classical", "deep", "hybrid") else "hybrid"
    if not STATE["ready"]:
        return _error(rid, "error", mode, "not_ready", "The service is still starting. Retry in a moment.", 503)
    max_bytes = CFG["upload"]["max_bytes"]
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        return _error(rid, "invalid_image", mode, "file_too_large",
                      f"File exceeds {max_bytes // (1024 * 1024)} MB. Choose a smaller image.", 413)
    if ADMISSION["waiting"] >= CFG["server"]["max_queue"]:
        return _error(rid, "busy", mode, "busy", "The server is busy. Retry in a few seconds.", 503)
    from src.vision.decode import DecodeError

    t_enq = time.perf_counter()
    ADMISSION["waiting"] += 1
    try:
        await SEM.acquire()
    finally:
        ADMISSION["waiting"] -= 1
    ADMISSION["active"] += 1
    try:
        return await run_in_threadpool(_run_pipeline, data, rid, mode, time.perf_counter() - t_enq)
    except DecodeError as e:
        return _error(rid, "invalid_image", mode, e.code, e.message, 413 if e.code == "file_too_large" else 400)
    except Exception as exc:
        return _error(rid, "error", mode, "internal_error",
                      f"Analysis failed ({type(exc).__name__}). Retry or choose another image.", 500)
    finally:
        ADMISSION["active"] -= 1
        SEM.release()


@app.get("/request-image/{rid}/{name}")
def request_image(rid: str, name: str):
    rec = STORE.get(rid)
    if rec is None or name not in rec["images"]:
        return JSONResponse(status_code=404, content={"error": {"code": "expired", "message": "This result has expired. Re-analyze the image."}})
    return Response(content=rec["images"][name], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
