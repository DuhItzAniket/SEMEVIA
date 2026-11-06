"""P6 analysis from saved predictions: viewpoint subgroups and rule-based failure cases.

Usage: python scripts/analyze_results.py --partition test      (P6, after the test run)
       python scripts/analyze_results.py --partition tune      (tooling check; development only)
Reads the evaluator JSONL for the current bundle. For each selected failure it adds an
annotation-crop diagnostic (same production head on the ground-truth crop) to separate
localisation effects from classifier effects. Causes are written as hypotheses.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.analysis import select_failures, subgroup_table  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.models.heads import softmax  # noqa: E402
from src.release import load_manifest, recognition_config  # noqa: E402
from src.vision.coordinates import iou  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="test", choices=["tune", "test"])
    ap.add_argument("--data-root", default=os.environ.get("SEMEVIA_DATA_ROOT"))
    args = ap.parse_args()
    manifest = load_manifest()
    bundle = manifest["bundle_id"]
    rc = recognition_config()
    cal = json.loads((ROOT / rc["calibration"]).read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"][args.partition]
    pred_dir = store_dir() / "predictions"
    cands = [pred_dir / f"e2e_{args.partition}_clean_{part['sha256'][:10]}_{bundle}.jsonl",
             pred_dir / f"e2e_{args.partition}_{part['sha256'][:10]}_{bundle}.jsonl"]
    jl = next((c for c in cands if c.exists()), None)
    if jl is None:
        raise SystemExit(f"no {args.partition} predictions for bundle {bundle}; run scripts/evaluate.py first")
    rows = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines()]
    recs = {r.image_key: r for r in read_manifest(ROOT / "artifacts/manifests" / part["file"])}
    if len(rows) != len(recs):
        raise SystemExit("prediction file does not cover the whole partition")
    gts = {k: r.box for k, r in recs.items()}
    meta = json.loads((ROOT / "artifacts/class_metadata.json").read_text(encoding="utf-8"))
    classes = {c["class_id"]: c for c in meta["classes"]}
    pm, T = cal["production_mode"], cal["temperature"]
    report = {"partition": args.partition, "bundle_id": bundle, "predictions_file": jl.name,
              "production_mode": pm, "calibration_kind": cal["kind"]}

    if args.partition == "test":
        labels = {}
        with open(ROOT / "reports/viewpoint_labels.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["image_key"]] = row["viewpoint"]
        missing = [k for k in labels if k not in recs]
        if missing:
            raise SystemExit(f"viewpoint keys missing from the test manifest: {missing[:3]}")
        report["viewpoint_subgroups"] = {m: subgroup_table(rows, gts, labels, f"z_{m}") for m in ("deep", "hybrid")}
        report["viewpoint_note"] = "Descriptive; small and imbalanced groups (labelled before results)."

    picks = select_failures(rows, gts, classes, f"z_{pm}", T)
    from src.data.crops import annotation_crop, load_rgb
    from src.models.inference import CropInput, Predictor

    import yaml

    margin = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))["crops"]["margin"]
    pred = Predictor({m: ROOT / p for m, p in rc["heads"].items()}, ROOT / rc["backbone"], 2)
    failures = []
    for f in picks:
        r = f["row"]
        rec = recs[r["image_key"]]
        rgb = load_rgb(Path(args.data_root) / rec.rel_path)
        crop, box = annotation_crop(rgb, rec.box, margin)
        z_or, _ = pred.score_crops(pm, [CropInput(crop, box, (rgb.shape[1], rgb.shape[0]))])
        p_or = softmax(z_or, T)[0]
        entry = {"rule": f["rule"], "image_key": r["image_key"], "true": classes[r["y"]]["name"],
                 "selected_box": r.get("sel_box"), "annotation_box": rec.box,
                 "iou": iou(r["sel_box"], rec.box) if r.get("sel_box") else None, "detector_score": r.get("sel_score")}
        if f"z_{pm}" in r:
            p = softmax(np.array([r[f"z_{pm}"]]), T)[0]
            top = sorted(range(196), key=lambda c: (-p[c], c))[:5]
            entry["runtime_top5"] = [(classes[c]["name"], round(float(p[c]), 4)) for c in top]
        top_or = sorted(range(196), key=lambda c: (-p_or[c], c))[:5]
        entry["annotation_crop_top5"] = [(classes[c]["name"], round(float(p_or[c]), 4)) for c in top_or]
        entry["annotation_crop_correct"] = int(top_or[0]) == r["y"]
        entry["hypothesis"] = ("HYPOTHESIS: " + ("localisation/crop caused the error (annotation crop is classified correctly)"
                                                if entry["annotation_crop_correct"] else
                                                "classifier confusion persists on the annotation crop"))
        failures.append(entry)
    report["failures"] = failures
    out = ROOT / "reports" / f"analysis_{args.partition}_{bundle}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"failures": [(f["rule"], f["image_key"], f["true"], f.get("runtime_top5", [None])[0], f["hypothesis"])
                                   for f in failures]}, indent=1))
    if "viewpoint_subgroups" in report:
        print(json.dumps(report["viewpoint_subgroups"]["hybrid"], indent=1))


if __name__ == "__main__":
    main()
