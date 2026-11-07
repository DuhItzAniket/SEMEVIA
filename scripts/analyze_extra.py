"""Extra analysis of the saved test predictions (no new inference, no new use of the test images).

Usage: python scripts/analyze_extra.py [--partition test] [--n-boot 2000]

Reads the per-image prediction rows that scripts/evaluate.py wrote for the current release bundle and produces:
  * bootstrap 95% confidence intervals for end-to-end top-1/top-5 of every mode, and for the paired deep-minus-hybrid
    difference (images resampled with replacement; seed fixed);
  * the most frequent confusion pairs of the production mode among correctly localised images, grouped into
    same family (another model year), same make, badge-engineered twins and different make;
  * reliability bins (accuracy vs confidence) for raw and temperature-scaled probabilities of the production mode.
Before writing anything, the end-to-end counts recomputed from the rows must equal those in
reports/e2e_<partition>_<bundle>.json, otherwise the script stops. Writes reports/extra_<partition>_<bundle>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.evaluation.metrics import ece  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.models.heads import softmax  # noqa: E402
from src.vision.coordinates import iou  # noqa: E402

MODES = ("classical", "deep", "hybrid")
TWINS = [("Chevrolet Express", "GMC Savana"), ("Chevrolet Tahoe", "GMC Yukon"), ("Chevrolet Traverse", "GMC Acadia"),
         ("Dodge Sprinter", "Mercedes-Benz Sprinter"), ("Ram C/V", "Dodge Caravan"), ("Ram C/V", "Chrysler Town and Country"),
         ("Chevrolet Avalanche", "Cadillac Escalade EXT")]


def ranked(z: np.ndarray) -> list[int]:
    return sorted(range(len(z)), key=lambda c: (-z[c], c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="test", choices=["tune", "calibration", "test"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    manifest = json.loads((ROOT / "artifacts/release_manifest.json").read_text(encoding="utf-8"))
    bundle = manifest["bundle_id"]
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"][args.partition]
    report = json.loads((ROOT / "reports" / f"e2e_{args.partition}_{bundle}.json").read_text(encoding="utf-8"))
    jl = store_dir() / "predictions" / report["predictions_file"]
    rows = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines() if line.strip()]
    with open(ROOT / "artifacts/manifests" / part["file"], encoding="utf-8") as f:
        man = {r["image_key"]: r for r in csv.DictReader(f)}
    gts = {k: [float(r["left"]), float(r["top"]), float(r["right"]), float(r["bottom"])] for k, r in man.items()}
    classes = json.loads((ROOT / "artifacts/class_metadata.json").read_text(encoding="utf-8"))["classes"]
    cal = json.loads((ROOT / "artifacts/calibration_final.json").read_text(encoding="utf-8"))
    pm, T = cal["production_mode"], cal["temperature"]
    if len(rows) != report["population"]["count"]:
        raise SystemExit("prediction rows do not match the report population")

    # per-image correctness for every mode (same rule as scripts/evaluate.py)
    hit1, hit5, box_ok = {m: np.zeros(len(rows), bool) for m in MODES}, {m: np.zeros(len(rows), bool) for m in MODES}, {}
    for m in MODES:
        box_key = "prop_box" if m == "classical" else "sel_box"
        ok = np.zeros(len(rows), bool)
        for i, r in enumerate(rows):
            b = r.get(box_key)
            if b is None or f"z_{m}" not in r:
                continue
            ok[i] = iou(b, gts[r["image_key"]]) >= 0.5
            if ok[i]:
                top = ranked(np.asarray(r[f"z_{m}"]))[:5]
                hit1[m][i] = top[0] == r["y"]
                hit5[m][i] = r["y"] in top
        box_ok[m] = ok
        for k, h in (("end_to_end_top1", hit1[m]), ("end_to_end_top5", hit5[m])):
            want = report["modes"][m][k]["numerator"]
            if int(h.sum()) != want:
                raise SystemExit(f"{m} {k}: recomputed {int(h.sum())} != report {want}; refusing to continue")

    rng = np.random.default_rng(args.seed)
    n = len(rows)
    idxs = rng.integers(0, n, size=(args.n_boot, n))
    boot = {}
    for m in MODES:
        for k, h in (("top1", hit1[m]), ("top5", hit5[m])):
            vals = h[idxs].mean(axis=1)
            boot[f"{m}_{k}"] = {"value": float(h.mean()), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}
    d = (hit1["deep"].astype(float) - hit1["hybrid"].astype(float))
    dv = d[idxs].mean(axis=1)
    boot["deep_minus_hybrid_top1"] = {"value": float(d.mean()), "ci95": [float(np.percentile(dv, 2.5)), float(np.percentile(dv, 97.5))],
                                      "includes_zero": bool(np.percentile(dv, 2.5) <= 0 <= np.percentile(dv, 97.5))}

    # confusion pairs of the production mode, among correctly localised images
    pairs = Counter()
    wrong = 0
    for i, r in enumerate(rows):
        if not box_ok[pm][i] or hit1[pm][i]:
            continue
        pred = ranked(np.asarray(r[f"z_{pm}"]))[0]
        pairs[(r["y"], pred)] += 1
        wrong += 1

    def kind(a: int, b: int) -> str:
        ca, cb = classes[a], classes[b]
        if ca["family"] == cb["family"]:
            return "same family, other model year"
        if any((x in ca["name"] and y in cb["name"]) or (y in ca["name"] and x in cb["name"]) for x, y in TWINS):
            return "badge-engineered twin"
        if ca["make"] == cb["make"]:
            return "same make, other model"
        return "different make"

    kinds = Counter()
    for (a, b), c in pairs.items():
        kinds[kind(a, b)] += c
    top_pairs = [{"true": classes[a]["name"], "predicted": classes[b]["name"], "count": c, "kind": kind(a, b)}
                 for (a, b), c in pairs.most_common(15)]

    # reliability of the production mode on correctly localised images
    conf_raw, conf_cal, corr = [], [], []
    for i, r in enumerate(rows):
        if not box_ok[pm][i]:
            continue
        z = np.asarray([r[f"z_{pm}"]])
        p_raw, p_cal = softmax(z)[0], softmax(z, T)[0]
        conf_raw.append(float(p_raw.max()))
        conf_cal.append(float(p_cal.max()))
        corr.append(bool(hit1[pm][i]))
    rel_raw, rel_cal = ece(np.array(conf_raw), np.array(corr)), ece(np.array(conf_cal), np.array(corr))
    want_raw = report["passport_production_mode"]["calibration_matched_primary"]["ece_raw"]
    if abs(rel_raw["ece"] - want_raw) > 1e-6:
        raise SystemExit(f"raw ECE recomputed {rel_raw['ece']} != report {want_raw}")

    out = {"partition": args.partition, "bundle_id": bundle, "predictions_file": report["predictions_file"],
           "population": {"count": n, "manifest_sha256": part["sha256"]}, "production_mode": pm, "temperature": T,
           "consistency": "end-to-end top-1/top-5 numerators of all modes and the raw ECE equal those in "
                          f"reports/e2e_{args.partition}_{bundle}.json",
           "bootstrap": {"resamples": args.n_boot, "seed": args.seed, "method": "image-level percentile bootstrap", **boot},
           "confusions": {"mode": pm, "population": "correctly localised images (box IoU >= 0.5) with a wrong top-1",
                          "wrong": wrong, "by_kind": dict(kinds.most_common()), "top_pairs": top_pairs},
           "reliability": {"mode": pm, "population": "correctly localised images (box IoU >= 0.5)", "n": len(corr),
                           "raw": rel_raw, "calibrated": rel_cal}}
    path = ROOT / "reports" / f"extra_{args.partition}_{bundle}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in boot.items()}, indent=1))
    print("confusions by kind:", dict(kinds.most_common()))
    print("ECE raw %.4f calibrated %.4f" % (rel_raw["ece"], rel_cal["ece"]))
    print("wrote", path.name)


if __name__ == "__main__":
    main()
