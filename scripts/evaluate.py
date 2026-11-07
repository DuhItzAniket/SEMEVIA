"""End-to-end evaluation of the deployed pipeline on a manifest partition.

Usage:
  python scripts/evaluate.py --partition tune --resume
  python scripts/evaluate.py --partition test --resume --confirm-frozen   (P6 only; refuses otherwise)

Uses exactly the release bundle the app loads (detector, backbone, heads, calibration). Images are
read in their stored pixel frame, resized like the runtime when the long edge exceeds the limit,
and boxes are mapped back. Per-image rows are appended to a keyed JSONL so a killed run resumes;
the report refuses partial or duplicated populations.

Paths per mode: classical = contour proposal ROI -> classical head; deep / hybrid = YOLO11n primary
box -> crop -> head. Ground truth is never used at inference; it is used only for scoring.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    coco_ap, ece, end_to_end_topk, localization, nll, ranked, ratio, set_metrics,
)
from src.evaluation.hardware import hardware_label  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.models.detector import select_target  # noqa: E402
from src.models.heads import softmax  # noqa: E402
from src.passport.conformal import ConformalQuantile, candidate_set, failure_set  # noqa: E402
from src.passport.hierarchy import Hierarchy  # noqa: E402
from src.release import load_manifest, verify  # noqa: E402
from src.vision.coordinates import crop_bounds, expand_box, iou  # noqa: E402

MODES = ("classical", "deep", "hybrid")


def load_image(path: Path, max_long: int, transform=None):
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    if transform is not None:
        rgb = transform(rgb)   # stress condition on stored pixels, before any detection
    s = 1.0
    if max(rgb.shape[:2]) > max_long:
        s = max_long / max(rgb.shape[:2])
        rgb = np.asarray(Image.fromarray(rgb).resize((round(rgb.shape[1] * s), round(rgb.shape[0] * s)), Image.BILINEAR))
    return rgb, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="tune", choices=["tune", "calibration", "test"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--confirm-frozen", action="store_true")
    ap.add_argument("--data-root", default=os.environ.get("SEMEVIA_DATA_ROOT"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--subset", default=None, choices=[None, "stress_tune", "stress_test"])
    ap.add_argument("--condition", default="clean")
    args = ap.parse_args()
    if args.subset:
        args.partition = args.subset.split("_", 1)[1]
    rt = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text(encoding="utf-8"))
    from src.release import recognition_config

    rc = recognition_config()
    cal = json.loads((ROOT / rc["calibration"]).read_text(encoding="utf-8"))
    manifest = load_manifest()
    problems = verify(manifest)
    if problems:
        raise SystemExit("; ".join(problems))
    if args.partition in ("calibration", "test"):
        from src.passport.calibration import check_runtime_matches, freeze_head_runtime, model_fingerprint

        fz_path = ROOT / "artifacts/production_freeze.json"
        if not fz_path.exists():
            raise SystemExit(f"the {args.partition} partition is locked until the P5 freeze (scripts/freeze.py)")
        fz = json.loads(fz_path.read_text(encoding="utf-8"))
        check_runtime_matches(rc, fz)
        if model_fingerprint(manifest, fz["production_mode"], fz["encoder_runtime"], freeze_head_runtime(fz)) != fz["model_fingerprint"]:
            raise SystemExit("current model files differ from the frozen pipeline; refusing to touch held-out data")
        if args.partition == "test" and not (args.confirm_frozen and cal["kind"] == "final"
                                             and cal.get("model_fingerprint") == fz["model_fingerprint"]):
            raise SystemExit("refusing the official test set: needs --confirm-frozen and final calibration for the frozen pipeline")
    import torch

    from src.models.detector import VehicleDetector
    from src.models.inference import CropInput, Predictor
    from src.vision.features import extract_classical
    from src.vision.proposals import propose

    torch.set_num_threads(4)
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"][args.partition]
    recs = read_manifest(ROOT / "artifacts/manifests" / part["file"])
    ecfg = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text(encoding="utf-8"))
    cond = ecfg["conditions"][args.condition]
    pop_name = args.partition
    if args.subset:
        sub = json.loads((ROOT / "artifacts/manifests/subsets_index.json").read_text(encoding="utf-8"))["subsets"][args.subset]
        recs = read_manifest(ROOT / "artifacts/manifests" / sub["file"])
        part = {"sha256": sub["sha256"], "file": sub["file"]}
        pop_name = args.subset
    elif args.condition != "clean":
        raise SystemExit("stress conditions run only on a declared stress subset")
    d = rt["detector"]
    asset = next(a for a in manifest["assets"] if a["name"] == "yolo11n.pt")
    det = VehicleDetector(ROOT / d["weights"], d["imgsz"], d["conf_floor"], d["nms_iou"], d["max_det"],
                          d["vehicle_coco_classes"], asset["sha256"])
    pred = Predictor.from_config(rc, ROOT, 4)
    hybrid_rt = pred.head_runtime.get("hybrid", "sklearn")
    thr, margin = rt["target_policy"]["threshold"], rt["crop"]["margin"]
    bundle = manifest["bundle_id"]
    out_dir = store_dir() / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / f"e2e_{pop_name}_{args.condition}_{part['sha256'][:10]}_{bundle}.jsonl"
    done: dict = {}
    if args.resume and jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["image_key"]] = r
        from src.passport.calibration import check_rows_runtime

        check_rows_runtime(list(done.values()), hybrid_rt, f"resume of {jl.name}")
    elif jl.exists():
        raise SystemExit(f"{jl.name} exists; pass --resume or delete it")
    todo = [r for r in recs if r.image_key not in done]
    print(f"{len(done)} done, {len(todo)} to run -> {jl.name}", flush=True)
    t_all = time.time()
    with open(jl, "a", encoding="utf-8") as f:
        for b0 in range(0, len(todo), args.batch):
            batch = todo[b0:b0 + args.batch]
            rows, yolo_crops, cls_crops = [], [], []
            for rec in batch:
                from src.evaluation.stress import apply_condition

                tf = (lambda im, rec=rec: apply_condition(im, cond, rec.box, rec.image_key, ecfg["seed"], args.condition))
                rgb, s = load_image(Path(args.data_root) / rec.rel_path, rt["upload"]["max_long_edge"],
                                    None if cond["type"] == "none" else tf)
                h, w = rgb.shape[:2]
                t0 = time.perf_counter()
                cands, _, _ = det.detect(rgb)
                sel, _ = select_target(cands, thr)
                t_det = (time.perf_counter() - t0) * 1000
                props = propose(rgb)
                row = {"image_key": rec.image_key, "y": rec.class_id, "scale": s, "det_ms": round(t_det, 1),
                       "cands": [{"box": [v / s for v in c.box], "score": round(c.score, 5)} for c in cands],
                       "sel_box": [v / s for v in sel.box] if sel else None, "sel_score": sel.score if sel else None,
                       "prop_box": [v / s for v in props[0].box] if props else None,
                       "prop_score": props[0].score if props else None}
                for key, box, bucket in (("yolo", sel.box if sel else None, yolo_crops),
                                         ("prop", props[0].box if props else None, cls_crops)):
                    if box is None:
                        continue
                    l, t, r_, b = crop_bounds(expand_box(box, margin, w, h), w, h)
                    bucket.append((len(rows), CropInput(np.ascontiguousarray(rgb[t:b, l:r_]), [l, t, r_, b], (w, h))))
                rows.append(row)
            if yolo_crops:
                cins = [c for _, c in yolo_crops]
                t1 = time.perf_counter()
                L = pred.learned(cins)
                H = np.stack([extract_classical(c.crop, c.box, c.image_size).vector for c in cins]).astype(np.float64)
                zd, zh = pred.score_features("deep", L, H), pred.score_features("hybrid", L, H)
                ms = (time.perf_counter() - t1) * 1000 / len(cins)
                for (i, _), a, b in zip(yolo_crops, zd, zh):
                    rows[i]["z_deep"] = [round(float(v), 5) for v in a]
                    rows[i]["z_hybrid"] = [round(float(v), 5) for v in b]
                    if hybrid_rt != "sklearn":
                        rows[i]["hybrid_rt"] = hybrid_rt
                    rows[i]["feat_ms_per_crop"] = round(ms, 1)
            if cls_crops:
                cins = [c for _, c in cls_crops]
                H = np.stack([extract_classical(c.crop, c.box, c.image_size).vector for c in cins]).astype(np.float64)
                zc = pred.score_features("classical", None, H)
                for (i, _), a in zip(cls_crops, zc):
                    rows[i]["z_classical"] = [round(float(v), 5) for v in a]
            for row in rows:
                f.write(json.dumps(row) + "\n")
                done[row["image_key"]] = row
            f.flush()
            print(f"{len(done)}/{len(recs)}", flush=True)
    rows = [done.get(r.image_key) for r in recs]
    if any(r is None for r in rows) or len(done) != len(recs):
        raise SystemExit("incomplete population; rerun with --resume")
    report = build_report(rows, recs, pop_name, part, bundle, cal, rt, jl.name)
    report["condition"] = {"name": args.condition, **cond}
    int8 = next((a["sha256"] for a in manifest["assets"] if a["name"] == "head_hybrid_int8.onnx"), None)
    report["runtime"] = {"encoder": pred.encoder_runtime, "encoder_precision": rc["precision"], "head_runtime": pred.head_runtime,
                         "head_int8_sha256": int8 if hybrid_rt == "onnx_int8" else None}
    report["wall_seconds_this_run"] = round(time.time() - t_all, 1)
    name = f"e2e_{pop_name}_{bundle}.json" if not args.subset else f"e2e_{pop_name}_{args.condition}_{bundle}.json"
    (ROOT / "reports" / name).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report["summary"], indent=1))


def build_report(rows, recs, partition, part, bundle, cal, rt, pred_file) -> dict:
    gts = [r.box for r in recs]
    y = [r.class_id for r in recs]
    hier = Hierarchy.load()
    key_id = {r.image_key: k + 1 for k, r in enumerate(recs)}
    out = {"population": {"partition": partition, "count": len(recs), "manifest_sha256": part["sha256"]},
           "bundle_id": bundle, "predictions_file": pred_file,
           "definitions": {
               "end_to_end_topk": "runtime-selected box IoU >= 0.5 with the annotation AND true class in top-k; misses count as failures; denominator = all images",
               "predicted_crop_topk": "category-only accuracy on images where the path produced a crop (may be a wrong target)",
               "matched_primary_topk": "category accuracy only where the runtime box has IoU >= 0.5",
               "fine_grained_ap": "COCO AP over 196 categories, max one prediction per image (top-1 class), score = localisation score x top-1 probability (ranking score, not a probability)"},
           "modes": {}}
    summary = {}
    for mode in MODES:
        zkey = f"z_{mode}"
        boxkey, scorekey = ("prop_box", "prop_score") if mode == "classical" else ("sel_box", "sel_score")
        boxes = [r[boxkey] for r in rows]
        topk = [ranked(np.array(r[zkey]), 5) if zkey in r else None for r in rows]
        e1 = end_to_end_topk(boxes, gts, topk, y, 1)
        e5 = end_to_end_topk(boxes, gts, topk, y, 5)
        have = [i for i, t in enumerate(topk) if t is not None]
        matched = [i for i in have if iou(boxes[i], gts[i]) >= 0.5]
        m = {"end_to_end_top1": e1, "end_to_end_top5": e5,
             "predicted_crop_top1": ratio(sum(y[i] == topk[i][0] for i in have), len(have)),
             "predicted_crop_top5": ratio(sum(y[i] in topk[i] for i in have), len(have)),
             "matched_primary_top1": ratio(sum(y[i] == topk[i][0] for i in matched), len(matched)),
             "matched_primary_top5": ratio(sum(y[i] in topk[i] for i in matched), len(matched)),
             "localization": localization(boxes, gts),
             "localization_source": "contour proposal" if mode == "classical" else "YOLO11n primary target"}
        preds = []
        for i in have:
            p = softmax(np.array([rows[i][zkey]]))[0]
            c = int(topk[i][0])
            preds.append({"image_id": key_id[recs[i].image_key], "category_id": c + 1, "box": boxes[i],
                          "score": float(rows[i][scorekey]) * float(p[c])})
        g = [{"image_id": key_id[r.image_key], "category_id": r.class_id + 1, "box": r.box} for r in recs]
        m["fine_grained_ap"] = coco_ap(g, preds, list(range(1, 197)))
        out["modes"][mode] = m
        summary[mode] = {"e2e_top1": e1["value"], "e2e_top5": e5["value"],
                         "matched_top1": m["matched_primary_top1"]["value"],
                         "fg_AP50": m["fine_grained_ap"]["AP50"], "loc_recall50": m["localization"]["primary_recall_at_iou"]["value"]}
    # Passport / calibration for the production mode
    pm = cal["production_mode"]
    T = cal["temperature"]
    zkey = f"z_{pm}"
    cq = ConformalQuantile(cal["alpha"], cal["n"], cal["k"], cal["quantile_mode"], cal["q"])
    sets, spec_flags, accepted, acc_correct = [], [], 0, 0
    conf_m, corr_m, P_m, y_m = [], [], [], []
    for i, r in enumerate(rows):
        if zkey not in r or r["sel_box"] is None:
            sets.append(failure_set())
            spec_flags.append(False)
            continue
        p = softmax(np.array([r[zkey]]), T)[0]
        s = candidate_set(p, cq)
        sets.append(s)
        spec_flags.append(hier.is_specific(hier.specificity(s)))
        top = int(np.argmax(p))
        ok_box = iou(r["sel_box"], gts[i]) >= 0.5
        if p[top] >= cal["reject"]["threshold"]:
            accepted += 1
            acc_correct += int(ok_box and top == y[i])
        if ok_box:
            conf_m.append(p.max()); corr_m.append(top == y[i]); P_m.append(p); y_m.append(y[i])
    P_m = np.array(P_m)
    raw_m = [softmax(np.array([rows[i][zkey]]))[0] for i in range(len(rows))
             if zkey in rows[i] and rows[i]["sel_box"] is not None and iou(rows[i]["sel_box"], gts[i]) >= 0.5]
    out["passport_production_mode"] = {
        "mode": pm, "calibration_kind": cal["kind"], "calibration_bundle": cal["bundle_id"],
        "caveat": ("Development calibration was fitted on this same tune split (annotation crops); coverage here is "
                   "in-sample and not a valid conformal guarantee." if cal["kind"] == "development" and partition == "tune" else None),
        "sets_all_inputs": set_metrics(sets, y, spec_flags),
        "accepted": {"retention": ratio(accepted, len(rows)), "accepted_end_to_end_accuracy": ratio(acc_correct, accepted),
                     "threshold": cal["reject"]["threshold"]},
        "calibration_matched_primary": {
            "n": len(y_m), "excluded_miss_or_wrong_target": len(rows) - len(y_m),
            "ece_raw": ece(np.array([p.max() for p in raw_m]), np.array([int(np.argmax(p)) == yy for p, yy in zip(raw_m, y_m)]))["ece"] if raw_m else None,
            "ece_calibrated": ece(np.array(conf_m), np.array(corr_m)) if len(y_m) else None,
            "nll_raw": nll(np.array(raw_m), np.array(y_m)) if raw_m else None,
            "nll_calibrated": nll(P_m, np.array(y_m)) if len(y_m) else None},
    }
    lat = np.array([r["det_ms"] for r in rows])
    feat = np.array([r["feat_ms_per_crop"] for r in rows if "feat_ms_per_crop" in r])
    out["latency_ms"] = {"detector_p50": float(np.percentile(lat, 50)), "detector_p95": float(np.percentile(lat, 95)),
                         "features_and_heads_per_crop_mean_batched": float(feat.mean()) if len(feat) else None,
                         "hardware": hardware_label(4), "note": "offline batch timing, not user-visible latency"}
    out["summary"] = summary
    return out


if __name__ == "__main__":
    main()
