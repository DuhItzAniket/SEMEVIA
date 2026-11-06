"""SEMEVIA web service.

Detection is always live. Fine-grained recognition, the Passport and the Evidence Challenge
are served when the optional recognition bundle (backbone, heads, calibration) is present and
checksum-verified; otherwise responses state that recognition is unavailable and contain no
category output. The current bundle is a DEVELOPMENT bundle (frozen ImageNet backbone,
development calibration) and every response says so.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import secrets
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).parent / "static"
CFG = yaml.safe_load((ROOT / "configs" / "runtime.yaml").read_text(encoding="utf-8"))
SCHEMA_VERSION = "semevia.v1"
VERSION = "0.3.0-dev"
MODES = ("classical", "deep", "hybrid")

STATE: dict = {"ready": False, "reasons": ["starting"], "detector": None, "bundle": None, "warmup_ms": None,
               "predictor": None, "calibration": None, "cq": None, "hierarchy": None, "classes": None,
               "recognition_reason": "not loaded", "asset_sha": {},
               "references": None, "references_reason": "not loaded", "encoder_version": None}


class RequestStore:
    """In-memory, request-scoped records with expiry. Never written to disk or logged."""

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


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _load_recognition(manifest: dict, status: dict) -> None:
    from src.models.inference import Predictor
    from src.passport.conformal import ConformalQuantile
    from src.passport.hierarchy import Hierarchy

    rc = CFG["recognition"]
    by_path = {a["path"]: a for a in manifest["assets"]}
    needed = [rc["backbone"], *rc["heads"].values(), rc["calibration"]]
    missing = [p for p in needed if p not in by_path or status.get(by_path[p]["name"]) != "ok"]
    if missing:
        STATE["recognition_reason"] = "recognition bundle not present in this build: " + ", ".join(Path(m).name for m in missing)
        return
    cal = json.loads((ROOT / rc["calibration"]).read_text(encoding="utf-8"))
    head_sha = by_path[rc["heads"][cal["production_mode"]]]["sha256"]
    if cal["head_sha256"] != head_sha:
        STATE["recognition_reason"] = "calibration does not match the production head; recognition disabled"
        return
    runtime = os.environ.get("SEMEVIA_ENCODER_RUNTIME", rc.get("encoder_runtime", "torch"))
    if runtime == "onnx_fp32" and (rc["onnx_encoder"] not in by_path or status.get(by_path[rc["onnx_encoder"]]["name"]) != "ok"):
        runtime = "torch"   # optimized encoder not in this build; fall back to the verified PyTorch encoder
    pred = Predictor({m: ROOT / p for m, p in rc["heads"].items()}, ROOT / rc["backbone"],
                     int(os.environ.get("SEMEVIA_TORCH_THREADS", CFG["server"]["torch_threads"])),
                     encoder_runtime=runtime, onnx_encoder=ROOT / rc["onnx_encoder"])
    from src.models.inference import CropInput

    dummy = np.full((160, 240, 3), 110, np.uint8)
    for m in pred.modes:  # real warmup of every mode before readiness
        pred.score_crops(m, [CropInput(dummy, [0, 0, 240, 160], (240, 160))])
    STATE["predictor"] = pred
    STATE["calibration"] = cal
    STATE["cq"] = ConformalQuantile(cal["alpha"], cal["n"], cal["k"], cal["quantile_mode"], cal["q"])
    STATE["hierarchy"] = Hierarchy.load()
    STATE["classes"] = STATE["hierarchy"].classes
    STATE["asset_sha"] = {a["name"]: a["sha256"][:12] for a in manifest["assets"]}
    STATE["recognition_reason"] = None
    # Reference index: optional; disabled unless present, verified and version-matched to the encoder
    from src.models.backbone import PREPROCESS_VERSION
    from src.passport.references import load_index

    suffix = "" if pred.encoder_runtime == "torch" else f"|{pred.encoder_runtime}"
    STATE["encoder_version"] = f"effb0_pool1280_{rc['backbone_tag']}|{PREPROCESS_VERSION}|{rc['precision']}" + suffix
    ref = rc.get("reference_index")
    if ref and ref in by_path and status.get(by_path[ref]["name"]) == "ok":
        idx = load_index(ROOT / ref)
        if idx["embedding_version"] == STATE["encoder_version"]:
            STATE["references"] = idx
            STATE["references_reason"] = None
        else:
            STATE["references_reason"] = "reference index built with a different encoder version"
    else:
        STATE["references_reason"] = "reference index not present in this build"


def _load_models() -> None:
    import torch

    from src.models.detector import VehicleDetector
    from src.release import asset_status, load_manifest, verify

    torch.set_num_threads(int(os.environ.get("SEMEVIA_TORCH_THREADS", CFG["server"]["torch_threads"])))
    manifest = load_manifest()
    problems = verify(manifest)
    if problems:
        STATE["reasons"] = problems
        return
    status = asset_status(manifest)
    det_cfg = CFG["detector"]
    asset = next(a for a in manifest["assets"] if a["name"] == "yolo11n.pt")
    det = VehicleDetector(ROOT / det_cfg["weights"], det_cfg["imgsz"], det_cfg["conf_floor"], det_cfg["nms_iou"],
                          det_cfg["max_det"], det_cfg["vehicle_coco_classes"], asset["sha256"])
    t = time.perf_counter()
    det.detect(np.full((480, 640, 3), 127, np.uint8))  # real warmup inference before readiness
    _load_recognition(manifest, status)
    STATE["warmup_ms"] = round((time.perf_counter() - t) * 1000, 1)
    STATE.update(detector=det, bundle=manifest["bundle_id"], ready=True, reasons=[])


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await run_in_threadpool(_load_models)
    except Exception as exc:  # readiness reports the failure; the process stays alive for diagnosis
        STATE.update(ready=False, reasons=[f"model load failed: {type(exc).__name__}: {exc}"])
    yield


app = FastAPI(title="SEMEVIA", version=VERSION, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _class_entry(cid: int, score: float) -> dict:
    c = STATE["classes"][cid]
    return {"class_id": cid, "original_class_id": cid + 1, "name": c["name"], "make": c["make"],
            "family": c["family"], "model": c["model"], "year": c["year"], "score": round(float(score), 6)}


def _envelope(rid: str, status: str, mode: str, error: dict | None = None) -> dict:
    cal = STATE["calibration"]
    recog = STATE["predictor"] is not None
    return {
        "schema_version": SCHEMA_VERSION, "request_id": rid, "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(), "active_mode": mode,
        "production_mode": cal["production_mode"] if cal else None, "error": error, "input": None,
        "detection": {"state": "unavailable"},
        "prediction": {"available": False, "unavailable_reason": STATE["recognition_reason"], "top5": [],
                       "probabilities": None, "classifier_score": None, "calibrated": False,
                       "temperature": None, "score_type": None},
        "passport": {"decision_state": "unavailable", "reason_codes": ["classifier_not_available"] if not recog else [],
                     "accepted_identity": None, "candidate_set": [], "candidate_count": 0},
        "challenge": {"state": "unavailable" if not recog else "not_requested"},
        "references": {"state": "unavailable", "reason": STATE["references_reason"], "items": []},
        "cnn_diagnostic": {"state": "unavailable", "reason": "optional; not implemented"},
        "processing": {},
        "model_status": {"recognition": recog, "backbone": CFG["recognition"]["backbone_status"] if recog else None,
                         "calibration_kind": cal["kind"] if cal else None,
                         "calibration_warning": cal.get("warning") if cal else None},
        "provenance": {"bundle_id": STATE["bundle"], "app_version": VERSION,
                       "detector": STATE["detector"].version if STATE["detector"] else None,
                       "assets": STATE["asset_sha"], "calibration_bundle": cal["bundle_id"] if cal else None},
        "timings": {},
    }


def _error(rid: str, status: str, mode: str, code: str, message: str, http: int) -> JSONResponse:
    return JSONResponse(status_code=http, content=_envelope(rid, status, mode, {"code": code, "message": message}))


def _run_pipeline(data: bytes, rid: str, mode: str, t_queue: float) -> dict:
    from src.models.detector import select_target
    from src.models.heads import softmax
    from src.models.inference import CropInput
    from src.passport.decision import make_passport
    from src.vision.coordinates import crop_bounds, expand_box
    from src.vision.decode import decode_image
    from src.vision.proposals import propose

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
    pred_ok = STATE["predictor"] is not None and mode in STATE["predictor"].modes

    resp = _envelope(rid, "ok", mode)
    resp["input"] = {"sha256": img.sha256, "width": img.width, "height": img.height,
                     "orientation_applied": img.orientation_applied,
                     "resized_from": list(img.original_size) if img.resized else None, "crop_sha256": None,
                     "format": img.source_format}
    images = {"display.jpg": _jpeg(img.rgb)}
    resp["detection"] = {
        "state": "detected" if sel else "no_vehicle", "source": "yolo11n",
        "selected_box": sel.box if sel else None, "selected_index": sel_i,
        "detector_score": sel.score if sel else None, "candidate_count": len(cands),
        "candidates": [{"box": d.box, "score": round(d.score, 4), "coco_class": d.coco_class} for d in cands],
        "raw_detection_count": total, "invalid_box_count": invalid,
        "selection_rule": CFG["target_policy"]["rule"], "threshold": thr, "detector_version": STATE["detector"].version,
    }
    target_box = sel.box if sel else None
    if mode == "classical" and pred_ok:
        # Fully classical path: ROI from the contour proposal baseline, not from YOLO.
        t2 = time.perf_counter()
        props = propose(img.rgb)
        timings["proposal_ms"] = round((time.perf_counter() - t2) * 1000, 1)
        resp["detection"]["classical_proposal"] = {
            "state": "proposed" if props else "no_proposal",
            "box": props[0].box if props else None, "score": props[0].score if props else None,
            "rule": "contour proposal: bounded area/edge-support score (not a probability)"}
        target_box = props[0].box if props else None

    rec: dict = {"images": images, "mode": mode}
    if target_box is None:
        resp["status"] = "no_vehicle"
        if mode == "classical" and pred_ok:
            resp["detection"]["state"] = "no_vehicle"
        resp["passport"] = make_passport(None, False, None, None, None, STATE["hierarchy"]) if pred_ok else resp["passport"]
        resp["challenge"] = {"state": "unavailable", "reason": "no target"}
    else:
        box = expand_box(target_box, CFG["crop"]["margin"], img.width, img.height)
        l, t, r, b = crop_bounds(box, img.width, img.height)
        crop = np.ascontiguousarray(img.rgb[t:b, l:r])
        images["crop.jpg"] = _jpeg(crop, 90)
        resp["input"]["crop_sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
        resp["processing"] = {"crop_bounds": [l, t, r, b], "crop_margin": CFG["crop"]["margin"]}
        if pred_ok:
            t3 = time.perf_counter()
            cin = CropInput(crop, [float(l), float(t), float(r), float(b)], (img.width, img.height))
            pr = STATE["predictor"]
            L = pr.learned([cin]) if pr.has_encoder else None   # one embedding: head + references
            hcX, hc = pr.classical([cin], want_views=True)
            z = pr.score_features(mode, L, hcX)[0]
            timings["features_and_classifier_ms"] = round((time.perf_counter() - t3) * 1000, 1)
            if STATE["references"] is not None and L is not None:
                from src.passport.references import query

                refs = query(STATE["references"], L[0], STATE["encoder_version"])
                for it in refs["items"]:
                    it.update(name=STATE["classes"][it["class_id"]]["name"], thumbnail=None,
                              thumbnail_reason="image rights unresolved; identifier and label only")
                refs["agrees_with_best_guess"] = [it["class_id"] == int(np.argmax(z)) for it in refs["items"]]
                resp["references"] = refs
            cal = STATE["calibration"]
            is_prod = mode == cal["production_mode"]
            T = cal["temperature"] if is_prod else 1.0
            p = softmax(z[None, :], T)[0]
            order = sorted(range(196), key=lambda c: (-p[c], c))
            resp["prediction"] = {
                "available": True, "unavailable_reason": None,
                "score_type": "calibrated softmax (development temperature)" if is_prod else "uncalibrated softmax (comparison mode)",
                "calibrated": is_prod, "temperature": T if is_prod else None,
                "top5": [_class_entry(c, p[c]) for c in order[:5]],
                "probabilities": [round(float(v), 6) for v in p], "classifier_score": round(float(p[order[0]]), 6)}
            if is_prod:
                pp = make_passport(p, True, {"bundle_id": cal["bundle_id"], "kind": cal["kind"], "alpha": cal["alpha"],
                                              "q": cal["q"], "n": cal["n"], "quantile_mode": cal["quantile_mode"]},
                                   STATE["cq"], cal["reject"]["threshold"], STATE["hierarchy"])
            else:
                pp = make_passport(p, True, None, None, None, STATE["hierarchy"])
            if pp.get("accepted_identity") is not None:
                pp["accepted_identity"] = _class_entry(pp["accepted_identity"], p[pp["accepted_identity"]])
            pp["candidates"] = [_class_entry(c, p[c]) for c in pp["candidate_set"]]
            resp["passport"] = pp
            if pp["decision_state"] in ("rejected", "inadequate"):
                resp["status"] = "uncertain"
            v = hc[0].views
            images.update({"gray.png": _png(v["gray"]), "edges.png": _png(v["edges"]), "contours.png": _png(v["contours"])})
            resp["processing"]["measurements"] = hc[0].measurements
            zo = sorted(range(196), key=lambda c: (-z[c], c))
            rec.update(crop=crop, cin_box=cin.box, image_size=cin.image_size, logits=z, class_a=zo[0], class_b=zo[1])
            resp["challenge"] = {"state": "not_requested", "class_a": int(zo[0]), "class_b": int(zo[1])}
        else:
            resp["challenge"] = {"state": "unavailable", "reason": STATE["recognition_reason"] or f"mode {mode} unavailable"}
    resp["processing"]["images"] = {k: f"/request-image/{rid}/{k}" for k in images}
    timings["total_ms"] = round((time.perf_counter() - t0) * 1000 + timings["queue_ms"], 1)
    resp["timings"] = timings
    STORE.put(rid, rec)
    return resp


def _run_challenge(rid: str) -> dict:
    from src.models.inference import CropInput
    from src.passport.challenge import run_challenge

    rec = STORE.get(rid)
    if rec is None:
        return {"request_id": rid, "state": "failed", "reason": "expired", "message": "This result has expired. Re-analyze the image."}
    if "challenge_result" in rec:
        return rec["challenge_result"]
    if "crop" not in rec:
        return {"request_id": rid, "state": "unavailable", "reason": "no recognized target for this request"}
    crop = rec["crop"]
    h, w = crop.shape[:2]
    if w < 3 or h < 2:
        return {"request_id": rid, "state": "unavailable", "reason": "crop_too_small"}
    pred = STATE["predictor"]

    def score(variants):
        return pred.score_crops(rec["mode"], [CropInput(v, rec["cin_box"], rec["image_size"]) for v in variants])[0]

    res = run_challenge(crop, score, rec["class_a"], rec["class_b"], rec["logits"])
    variants = res.pop("variants")
    for i, v in enumerate(variants):
        rec["images"][f"challenge_{i}.jpg"] = _jpeg(v, 90)
        res["cells"][i]["image_url"] = f"/request-image/{rid}/challenge_{i}.jpg"
        res["cells"][i]["new_top1_entry"] = _class_entry(res["cells"][i]["new_top1"], res["cells"][i]["new_top1_prob"])
    res.update(request_id=rid, mode=rec["mode"], class_a_entry=_class_entry(res["class_a"], 0)["name"],
               class_b_entry=_class_entry(res["class_b"], 0)["name"], reason=None)
    rec["challenge_result"] = res
    return res


async def _admit(fn, *args):
    if ADMISSION["waiting"] >= CFG["server"]["max_queue"]:
        return None
    t_enq = time.perf_counter()
    ADMISSION["waiting"] += 1
    try:
        await SEM.acquire()
    finally:
        ADMISSION["waiting"] -= 1
    ADMISSION["active"] += 1
    try:
        return await run_in_threadpool(fn, *args, time.perf_counter() - t_enq) if fn is _run_pipeline \
            else await run_in_threadpool(fn, *args)
    finally:
        ADMISSION["active"] -= 1
        SEM.release()


@app.get("/healthz")
def healthz():
    return {"status": "alive", "version": VERSION}


@app.get("/readyz")
def readyz():
    if STATE["ready"]:
        cal = STATE["calibration"]
        return {"ready": True, "bundle_id": STATE["bundle"], "warmup_ms": STATE["warmup_ms"],
                "detector": STATE["detector"].version,
                "recognition": STATE["predictor"] is not None, "recognition_reason": STATE["recognition_reason"],
                "modes": STATE["predictor"].modes if STATE["predictor"] else [],
                "production_mode": cal["production_mode"] if cal else None,
                "calibration_kind": cal["kind"] if cal else None,
                "backbone": CFG["recognition"]["backbone_status"] if STATE["predictor"] else None,
                "references": STATE["references"] is not None, "references_reason": STATE["references_reason"],
                "encoder_runtime": STATE["predictor"].encoder_runtime if STATE["predictor"] else None}
    return JSONResponse(status_code=503, content={"ready": False, "reasons": STATE["reasons"]})


@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Form("")):
    from src.vision.decode import DecodeError

    rid = secrets.token_urlsafe(12)
    mode = mode if mode in MODES else CFG["recognition"]["default_mode"]
    if not STATE["ready"]:
        return _error(rid, "error", mode, "not_ready", "The service is still starting. Retry in a moment.", 503)
    max_bytes = CFG["upload"]["max_bytes"]
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        return _error(rid, "invalid_image", mode, "file_too_large",
                      f"File exceeds {max_bytes // (1024 * 1024)} MB. Choose a smaller image.", 413)
    try:
        out = await _admit(_run_pipeline, data, rid, mode)
    except DecodeError as e:
        return _error(rid, "invalid_image", mode, e.code, e.message, 413 if e.code == "file_too_large" else 400)
    except Exception as exc:
        return _error(rid, "error", mode, "internal_error",
                      f"Analysis failed ({type(exc).__name__}). Retry or choose another image.", 500)
    if out is None:
        return _error(rid, "busy", mode, "busy", "The server is busy. Retry in a few seconds.", 503)
    return out


@app.post("/challenge")
async def challenge(payload: dict = Body(...)):
    rid = str(payload.get("request_id", ""))
    try:
        out = await _admit(_run_challenge, rid)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"request_id": rid, "state": "failed", "reason": "internal_error",
                                                      "message": f"Challenge failed ({type(exc).__name__}); the original result is unchanged."})
    if out is None:
        return JSONResponse(status_code=503, content={"request_id": rid, "state": "failed", "reason": "busy",
                                                      "message": "The server is busy. Retry the challenge in a few seconds."})
    code = 200 if out["state"] == "complete" else (404 if out.get("reason") == "expired" else 409)
    return JSONResponse(status_code=code, content=out)


@app.get("/request-image/{rid}/{name}")
def request_image(rid: str, name: str):
    rec = STORE.get(rid)
    if rec is None or name not in rec["images"]:
        return JSONResponse(status_code=404, content={"error": {"code": "expired", "message": "This result has expired. Re-analyze the image."}})
    media = "image/png" if name.endswith(".png") else "image/jpeg"
    return Response(content=rec["images"][name], media_type=media, headers={"Cache-Control": "no-store"})


@app.get("/benchmarks")
def benchmarks():
    """Precomputed development diagnostics with their populations; not per-upload accuracy."""
    out = {"note": "Development diagnostics on the tuning split. No test-set results exist yet.", "reports": {}}
    for name in ("classical_tune.json", "heads_tune_imagenet_frozen.json", "detector_tune.json"):
        p = ROOT / "reports" / name
        if p.exists():
            out["reports"][name] = json.loads(p.read_text(encoding="utf-8"))
    return out


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
