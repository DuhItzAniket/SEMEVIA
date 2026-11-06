"""Evaluate the unsupported-input (negative) suite through the deployed pipeline.

Usage: python scripts/evaluate_negatives.py --partition tune   (feeds scripts/freeze.py --negatives)
       python scripts/evaluate_negatives.py --partition test   (held-out; only after the freeze)
Manifest: artifacts/negative_manifests/negatives_<partition>.csv with columns
  neg_id, rel_path, bucket, source_url, license, author, notes
  bucket in {non_vehicle, two_wheeler, unsupported_vehicle_type, unlisted_car_model}
Images live under SEMEVIA_NEG_ROOT (outside Git). Records raw production-mode logits per image (or
none when no vehicle is detected) and per-bucket whole-system false acceptance with counts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.models.heads import softmax  # noqa: E402
from src.release import load_manifest, recognition_config, verify  # noqa: E402

BUCKETS = ("non_vehicle", "two_wheeler", "unsupported_vehicle_type", "unlisted_car_model")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="tune", choices=["tune", "test"])
    ap.add_argument("--neg-root", default=os.environ.get("SEMEVIA_NEG_ROOT"))
    args = ap.parse_args()
    if args.partition == "test" and not (ROOT / "artifacts/production_freeze.json").exists():
        raise SystemExit("the held-out negative set is locked until the P5 freeze")
    manifest = load_manifest()
    if verify(manifest):
        raise SystemExit("release assets do not verify")
    rc = recognition_config()
    rt = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text(encoding="utf-8"))
    cal = json.loads((ROOT / rc["calibration"]).read_text(encoding="utf-8"))
    mpath = ROOT / f"artifacts/negative_manifests/negatives_{args.partition}.csv"
    with open(mpath, encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    bad = [i["neg_id"] for i in items if i["bucket"] not in BUCKETS or not i["source_url"] or not i["license"]]
    if bad:
        raise SystemExit(f"negative manifest rows without valid bucket/source/licence: {bad[:5]}")
    from src.models.detector import VehicleDetector, select_target
    from src.models.inference import CropInput, Predictor
    from src.vision.coordinates import crop_bounds, expand_box
    from src.vision.decode import decode_image

    d = rt["detector"]
    det_asset = next(a for a in manifest["assets"] if a["name"] == "yolo11n.pt")
    det = VehicleDetector(ROOT / d["weights"], d["imgsz"], d["conf_floor"], d["nms_iou"], d["max_det"],
                          d["vehicle_coco_classes"], det_asset["sha256"])
    pred = Predictor({m: ROOT / p for m, p in rc["heads"].items()}, ROOT / rc["backbone"], 2)
    pm, T, thr = cal["production_mode"], cal["temperature"], cal["reject"]["threshold"]
    up = rt["upload"]
    rows = []
    for it in items:
        data = (Path(args.neg_root) / it["rel_path"]).read_bytes()
        img = decode_image(data, up["max_bytes"], up["max_pixels"], up["max_long_edge"], up["allowed_formats"])
        cands, _, _ = det.detect(img.rgb)
        sel, _ = select_target(cands, rt["target_policy"]["threshold"])
        row = {"neg_id": it["neg_id"], "bucket": it["bucket"], "detected": sel is not None, "z": None}
        if sel is not None:
            l, t, r, b = crop_bounds(expand_box(sel.box, rt["crop"]["margin"], img.width, img.height), img.width, img.height)
            z, _ = pred.score_crops(pm, [CropInput(np.ascontiguousarray(img.rgb[t:b, l:r]), [l, t, r, b], (img.width, img.height))])
            row["z"] = [round(float(v), 5) for v in z[0]]
            p = softmax(z, T)[0]
            row["max_prob"] = float(p.max())
            row["accepted"] = bool(p.max() >= thr)
        else:
            row["accepted"] = False
        rows.append(row)
    per = {}
    for bkt in BUCKETS:
        rs = [r for r in rows if r["bucket"] == bkt]
        acc = sum(r["accepted"] for r in rs)
        per[bkt] = {"count": len(rs), "accepted": acc, "false_acceptance": (acc / len(rs)) if rs else None,
                    "rejected_by_detector": sum(not r["detected"] for r in rs),
                    "reached_classifier": sum(r["detected"] for r in rs)}
    out = {"partition": args.partition, "bundle_id": manifest["bundle_id"], "calibration_kind": cal["kind"],
           "production_mode": pm, "temperature": T, "reject_threshold": thr, "count": len(rows),
           "bucket_counts": dict(Counter(r["bucket"] for r in rows)), "per_bucket": per,
           "overall_false_acceptance": {"numerator": sum(r["accepted"] for r in rows), "denominator": len(rows)},
           "note": ("Whole-system rejection (a detector miss counts as a success); classifier-level open-set ability "
                    "is only measured on images that reached the classifier. Small stress set, not open-world."),
           "rows": rows}
    (ROOT / "reports" / f"negatives_{args.partition}_{manifest['bundle_id']}.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("count", "per_bucket", "overall_false_acceptance")}, indent=1))


if __name__ == "__main__":
    main()
