"""Per-image comparison of two end-to-end test runs from their saved prediction files (no model run).

Usage: python scripts/compare_runs.py --a ft_ckpt_98b1fb55cc6e_5d755416fd --b <INT8 bundle> [--partition test]
Joins the two JSONL files on image_key and, per mode, reports end-to-end top-1 of each run (box IoU >= 0.5 with the
annotation and the right class), the paired difference with a bootstrap 95% CI (fixed seed), the exact McNemar test
on images correct under only one run, top-1 answer agreement and the maximum logit difference. The deep and classical
heads did not change between r3 and r3-int8, so their agreement is the run-to-run floor. Writes
reports/int8_vs_fp32_<partition>_<b>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.manifest import read_manifest  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.vision.coordinates import iou  # noqa: E402


def load(partition: str, part: dict, bundle: str) -> dict:
    jl = next((c for c in (store_dir() / "predictions" / f"e2e_{partition}_clean_{part['sha256'][:10]}_{bundle}.jsonl",
                           store_dir() / "predictions" / f"e2e_{partition}_{part['sha256'][:10]}_{bundle}.jsonl") if c.exists()), None)
    if jl is None:
        raise SystemExit(f"no saved {partition} predictions for {bundle}")
    return {r["image_key"]: r for r in map(json.loads, jl.read_text(encoding="utf-8").splitlines())}, jl.name


def mcnemar_exact(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference bundle (r3 FP32)")
    ap.add_argument("--b", required=True, help="compared bundle (r3-int8)")
    ap.add_argument("--partition", default="test")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"][args.partition]
    recs = read_manifest(ROOT / "artifacts/manifests" / part["file"])
    gts = {r.image_key: r.box for r in recs}
    A, fa = load(args.partition, part, args.a)
    B, fb = load(args.partition, part, args.b)
    keys = [r.image_key for r in recs]
    if set(A) != set(keys) or set(B) != set(keys):
        raise SystemExit("prediction files do not both cover the whole partition")
    boxes_same = all(A[k].get("sel_box") == B[k].get("sel_box") for k in keys)
    rng = np.random.default_rng(20261107)
    out = {"partition": args.partition, "a": {"bundle": args.a, "predictions_file": fa}, "b": {"bundle": args.b, "predictions_file": fb},
           "selected_boxes_identical": boxes_same, "bootstrap": {"resamples": args.boot, "seed": 20261107}, "modes": {}}
    for m in ("classical", "deep", "hybrid"):
        z = f"z_{m}"
        box = "prop_box" if m == "classical" else "sel_box"

        def correct(R, k):
            r = R[k]
            return bool(r.get(box) is not None and z in r and iou(r[box], gts[k]) >= 0.5 and int(np.argmax(r[z])) == r["y"])

        ca = np.array([correct(A, k) for k in keys])
        cb = np.array([correct(B, k) for k in keys])
        both = [k for k in keys if z in A[k] and z in B[k]]
        agree = float(np.mean([int(np.argmax(A[k][z])) == int(np.argmax(B[k][z])) for k in both])) if both else None
        dmax = float(max(np.abs(np.array(A[k][z]) - np.array(B[k][z])).max() for k in both)) if both else None
        d = cb.astype(float) - ca.astype(float)
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.boot)]
        only_a, only_b = int((ca & ~cb).sum()), int((cb & ~ca).sum())
        out["modes"][m] = {"n": len(keys), "a_top1": float(ca.mean()), "b_top1": float(cb.mean()), "diff": float(d.mean()),
                           "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                           "only_a": only_a, "only_b": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b),
                           "top1_agreement": agree, "max_abs_logit_diff": dmax, "scored_in_both": len(both)}
    path = ROOT / "reports" / f"int8_vs_fp32_{args.partition}_{args.b}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({m: {k: v for k, v in r.items() if k in ("a_top1", "b_top1", "diff", "ci", "only_a", "only_b", "mcnemar_p",
                                                                "top1_agreement", "max_abs_logit_diff")}
                      for m, r in out["modes"].items()}, indent=1), "boxes identical:", boxes_same)


if __name__ == "__main__":
    main()
