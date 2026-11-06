"""Evaluate the frozen YOLO11n detector on a manifest partition (development: tune only).

Usage: python scripts/evaluate_detector.py --partition tune
Images are used in their stored pixel frame (no EXIF), resized like the runtime when the long
edge exceeds the limit, and detections are mapped back to stored pixels. Results are written
incrementally (JSONL keyed by image) so an interrupted run resumes; the report refuses a
partial population.
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
from src.evaluation.metrics import coco_ap, localization, ratio  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.models.detector import VehicleDetector, select_target  # noqa: E402
from src.release import load_manifest  # noqa: E402
from src.vision.coordinates import iou  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="tune", choices=["tune", "train", "calibration"])
    ap.add_argument("--data-root", default=os.environ.get("SEMEVIA_DATA_ROOT"))
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.25, 0.4, 0.55])
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"][args.partition]
    recs = read_manifest(ROOT / "artifacts/manifests" / part["file"])
    asset = next(a for a in load_manifest()["assets"] if a["name"] == "yolo11n.pt")
    d = cfg["detector"]
    import torch

    torch.set_num_threads(4)
    det = VehicleDetector(ROOT / d["weights"], d["imgsz"], d["conf_floor"], d["nms_iou"], d["max_det"],
                          d["vehicle_coco_classes"], asset["sha256"])
    out_dir = store_dir() / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    jl = out_dir / f"detector_{args.partition}_{part['sha256'][:10]}_{det.version.replace(':', '_')}.jsonl"
    done = {}
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["image_key"]] = r
    max_long = cfg["upload"]["max_long_edge"]
    with open(jl, "a", encoding="utf-8") as f:
        for i, rec in enumerate(recs):
            if rec.image_key in done:
                continue
            with Image.open(Path(args.data_root) / rec.rel_path) as im:
                rgb = np.asarray(im.convert("RGB"))
            s = 1.0
            if max(rgb.shape[:2]) > max_long:
                s = max_long / max(rgb.shape[:2])
                rgb = np.asarray(Image.fromarray(rgb).resize((round(rgb.shape[1] * s), round(rgb.shape[0] * s)), Image.BILINEAR))
            t = time.perf_counter()
            cands, total, invalid = det.detect(rgb)
            ms = (time.perf_counter() - t) * 1000
            row = {"image_key": rec.image_key, "scale": s, "det_ms": round(ms, 2), "raw": total, "invalid": invalid,
                   "cands": [{"box": [v / s for v in c.box], "score": c.score, "cls": c.coco_class} for c in cands]}
            f.write(json.dumps(row) + "\n")
            done[rec.image_key] = row
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(recs)}", flush=True)
    rows = [done[r.image_key] for r in recs]
    if len(rows) != len(recs) or len({r["image_key"] for r in rows}) != len(recs):
        raise SystemExit("incomplete or duplicated detector rows")
    gts = [r.box for r in recs]

    # Generic vehicle AP: all retained vehicle-mapped detections, one category
    key_id = {r.image_key: k + 1 for k, r in enumerate(recs)}
    g = [{"image_id": key_id[r.image_key], "category_id": 1, "box": r.box} for r in recs]
    p = [{"image_id": key_id[row["image_key"]], "category_id": 1, "box": c["box"], "score": c["score"]}
         for row in rows for c in row["cands"]]
    ap_generic = coco_ap(g, p, [1])

    any_recall = sum(1 for row, gt in zip(rows, gts) if any(iou(c["box"], gt) >= 0.5 for c in row["cands"]))
    by_thr = {}
    for thr in args.thresholds:
        from src.models.detector import Detection

        sel = []
        for row in rows:
            cs = [Detection(c["box"], c["score"], c["cls"], True) for c in row["cands"]]
            s, _ = select_target(cs, thr)
            sel.append(s.box if s else None)
        by_thr[str(thr)] = localization(sel, gts)
    lat = np.array([r["det_ms"] for r in rows])
    report = {
        "population": {"partition": args.partition, "count": len(recs), "manifest_sha256": part["sha256"],
                       "frame": "stored pixels, no EXIF; runtime long-edge resize applied then boxes mapped back"},
        "detector": {"version": det.version, "imgsz": d["imgsz"], "conf_floor": d["conf_floor"], "nms_iou": d["nms_iou"],
                     "max_det": d["max_det"], "vehicle_mapping": d["vehicle_coco_classes"]},
        "generic_vehicle_ap": ap_generic,
        "detector_recall_any_candidate_iou50": ratio(any_recall, len(recs)),
        "primary_target_by_threshold": by_thr,
        "latency_ms_detector_only": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                                     "mean": float(lat.mean()), "hardware": "Intel i5-1135G7, 4 torch threads, CPU",
                                     "note": "detector call only, warm process, excludes decode/network"},
        "notes": ["Stanford Cars annotates one target vehicle per image; extra visible vehicles are unannotated, so "
                  "generic AP false positives include real but unlabelled vehicles.",
                  "Development population (tune). Final test evaluation happens in P6 after the configuration freeze."],
        "predictions_file": jl.name,
    }
    (ROOT / "reports" / f"detector_{args.partition}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"AP50": ap_generic["AP50"], "AP50_95": ap_generic["AP50_95"],
                      "recall_any": any_recall / len(recs),
                      "primary_recall": {k: v["primary_recall_at_iou"]["value"] for k, v in by_thr.items()},
                      "missing": {k: v["missing"] for k, v in by_thr.items()},
                      "p50_ms": report["latency_ms_detector_only"]["p50"]}, indent=1))


if __name__ == "__main__":
    main()
