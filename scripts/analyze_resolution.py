"""Accuracy by native vehicle size and by natural occlusion, from saved test predictions (no model run).

Usage: python scripts/analyze_resolution.py
Reads the evaluator JSONL already written for the released bundle, so the test split is not used again.
Bins are fixed here, before looking at results: the annotated box width in the original image's pixels.
Natural occlusion uses the labelled sample in reports/viewpoint_labels.csv (small; descriptive only).
Writes reports/resolution_test_<bundle>.json.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.analysis import subgroup_table  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.release import load_manifest  # noqa: E402

WIDTH_BINS = [(0, 150, "under 150 px"), (150, 300, "150-299 px"), (300, 500, "300-499 px"), (500, 10 ** 9, "500 px and over")]
MODES = ("deep", "hybrid")   # both use the YOLO box; classical uses its own contour box and is reported in section 2


def main() -> None:
    bundle = load_manifest()["bundle_id"]
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"]["test"]
    pred_dir = store_dir() / "predictions"
    jl = next((c for c in (pred_dir / f"e2e_test_clean_{part['sha256'][:10]}_{bundle}.jsonl",
                           pred_dir / f"e2e_test_{part['sha256'][:10]}_{bundle}.jsonl") if c.exists()), None)
    if jl is None:
        raise SystemExit(f"no saved test predictions for bundle {bundle}")
    rows = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines()]
    recs = {r.image_key: r for r in read_manifest(ROOT / "artifacts/manifests" / part["file"])}
    if len(rows) != len(recs):
        raise SystemExit("prediction file does not cover the whole test partition")
    gts = {k: r.box for k, r in recs.items()}

    def width_bin(box):
        w = box[2] - box[0]
        return next(name for lo, hi, name in WIDTH_BINS if lo <= w < hi)

    by_width = {k: width_bin(r.box) for k, r in recs.items()}
    occl = {}
    with open(ROOT / "reports/viewpoint_labels.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            occl[row["image_key"]] = row["natural_occlusion"]
    order = [name for _, _, name in WIDTH_BINS]
    report = {
        "partition": "test", "bundle_id": bundle, "predictions_file": jl.name,
        "definition": "end-to-end: runtime box IoU >= 0.5 with the annotation and true class in top-k; denominator = all images in the group",
        "width_bins": order,
        "by_box_width": {m: {g: v for g, v in sorted(subgroup_table(rows, gts, by_width, f"z_{m}").items(),
                                                       key=lambda kv: order.index(kv[0]))} for m in MODES},
        "by_natural_occlusion": {m: subgroup_table(rows, gts, occl, f"z_{m}") for m in MODES},
        "occlusion_note": "Labelled sample from reports/viewpoint_labels.csv (team review pending); groups are tiny and descriptive only.",
    }
    out = ROOT / "reports" / f"resolution_test_{bundle}.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({m: {g: (v["count"], round(v["top1_rate"], 4)) for g, v in report["by_box_width"][m].items()} for m in MODES}, indent=1))
    print(json.dumps({m: {g: (v["count"], v["top1"]) for g, v in report["by_natural_occlusion"][m].items()} for m in MODES}))


if __name__ == "__main__":
    main()
