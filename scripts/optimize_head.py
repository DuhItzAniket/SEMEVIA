"""Fallback optimization (plan P5): INT8 dense-linear head.

Usage: python scripts/optimize_head.py --mode hybrid --backbone imagenet_frozen
The fitted logistic-regression coefficients are copied into an ONNX MatMul+Add graph (after the
unchanged FP32 FusionFeatures transform; src/models/head_onnx.py). FP32 parity with sklearn is checked
first; the graph is then quantized with onnxruntime dynamic quantization (MatMulInteger, INT8 weights).
Both graphs are scored one row at a time, as served. Tune accuracy, agreement, size and latency are
measured. Writes reports/optimization_head_<mode>_<backbone>.json and a sidecar
<store>/optimized/head_<mode>_<backbone>_int8.json (sha256, provenance, policy result) that
scripts/promote_bundle.py requires before it adds the INT8 head to a release.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from src.evaluation.hardware import hardware_label  # noqa: E402
from src.models.cache_io import cache_meta, load_cache, store_dir  # noqa: E402
from src.models.heads import canonical_logits, softmax, topk_accuracy  # noqa: E402


def lat(fn, x, runs=200):
    for _ in range(10):
        fn(x)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["deep", "hybrid"])
    ap.add_argument("--backbone", default="imagenet_frozen")
    args = ap.parse_args()
    from src.models.head_onnx import build_int8, row_wise_logits, session
    from src.release import recognition_config

    rc = recognition_config()
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    head_path = ROOT / rc["heads"][args.mode]
    pipe = joblib.load(head_path)
    fuse = pipe[:-1]
    _, yl, XL, _ = load_cache(cache_meta("learned", "tune", "annotation", idx, args.backbone))
    _, _, XC, _ = load_cache(cache_meta("classical", "tune", "annotation", idx))
    X = np.concatenate([XL, XC], axis=1)
    F = fuse.transform(X).astype(np.float32)
    z_ref = canonical_logits(pipe, X)

    out = store_dir() / "optimized"
    out.mkdir(parents=True, exist_ok=True)
    p32, p8 = out / f"head_{args.mode}_{args.backbone}_fp32.onnx", out / f"head_{args.mode}_{args.backbone}_int8.onnx"
    built = build_int8(head_path, p8, fp32_out=p32)
    s32, s8 = session(p32, 2), session(p8, 2)
    # Served inference is row-wise (one dynamic activation scale per image); that is what is measured here.
    z32 = row_wise_logits(s32, F)
    z8 = row_wise_logits(s8, F)
    z8_batch = s8.run(None, {"features": F})[0].astype(np.float64)

    def c(z):
        return z - z.mean(1, keepdims=True)

    import onnx

    ops8 = sorted({n.op_type for n in onnx.load(str(p8)).graph.node})
    rpath = ROOT / "reports" / f"optimization_head_{args.mode}_{args.backbone}.json"
    old = json.loads(rpath.read_text(encoding="utf-8")) if rpath.exists() else {}
    superseded = old.get("superseded_batchwise") or (
        {k: old[k] for k in ("accuracy", "int8_vs_fp32", "int8_top1_drop_pp", "int8_passes_policy") if k in old}
        if old and old.get("inference") != "row_wise" else None)
    report = {
        "mode": args.mode, "backbone": args.backbone,
        "population": {"partition": "tune", "count": int(len(yl)), "crops": "annotation"},
        "method": "logistic coefficients copied to ONNX MatMul+Add after the unchanged FP32 fusion transform; "
                  "onnxruntime quantize_dynamic, INT8 weights (MatMulInteger)",
        "inference": "row_wise",
        "int8_graph_ops": ops8,
        "int8_sha256": built["sha256"], "source_head_sha256": built["source_head_sha256"], "recipe": built["recipe"],
        "fp32_parity_vs_sklearn": {"max_abs_centered_logit_diff": float(np.abs(c(z32) - c(z_ref)).max()),
                                   "max_abs_prob_diff": float(np.abs(softmax(z32) - softmax(z_ref)).max()),
                                   "passes_atol_1e-4": bool(np.allclose(c(z32), c(z_ref), atol=1e-4, rtol=1e-4)),
                                   "top1_agreement": float(np.mean(z32.argmax(1) == z_ref.argmax(1)))},
        "accuracy": {"sklearn_fp32": {"tune_top1": topk_accuracy(z_ref, yl, 1), "tune_top5": topk_accuracy(z_ref, yl, 5)},
                     "onnx_fp32": {"tune_top1": topk_accuracy(z32, yl, 1), "tune_top5": topk_accuracy(z32, yl, 5)},
                     "onnx_int8": {"tune_top1": topk_accuracy(z8, yl, 1), "tune_top5": topk_accuracy(z8, yl, 5)}},
        "int8_vs_fp32": {"top1_agreement": float(np.mean(z8.argmax(1) == z_ref.argmax(1))),
                         "max_abs_centered_logit_diff": float(np.abs(c(z8) - c(z_ref)).max()),
                         "mean_abs_centered_logit_diff": float(np.abs(c(z8) - c(z_ref)).mean())},
        "row_vs_batch_int8_max_abs_centered_logit_diff": float(np.abs(c(z8) - c(z8_batch)).max()),
        "size_bytes": {"onnx_fp32": p32.stat().st_size, "onnx_int8": p8.stat().st_size,
                       "sklearn_joblib_whole_head_incl_pca": head_path.stat().st_size},
        "latency_single_row_head_only": {"onnx_fp32": lat(lambda x: s32.run(None, {"features": x}), F[:1]),
                                         "onnx_int8": lat(lambda x: s8.run(None, {"features": x}), F[:1]),
                                         "sklearn_fp32_linear": lat(lambda x: pipe[-1].decision_function(x), F[:1].astype(np.float64)),
                                         "threads": 2, "hardware": hardware_label(2)},
        "note": "The head is a tiny part of end-to-end time; no meaningful end-to-end speed gain is claimed from it. "
                "INT8 results depend on the CPU instruction set; measured on the hardware above.",
    }
    if superseded:
        report["superseded_batchwise"] = {**superseded, "note": "earlier run scored all 815 rows as one tensor "
                                          "(one shared activation scale); superseded by the row-wise run above"}
    drop = report["accuracy"]["sklearn_fp32"]["tune_top1"] - report["accuracy"]["onnx_int8"]["tune_top1"]
    report["int8_top1_drop_pp"] = round(100 * drop, 2)
    report["int8_passes_policy"] = bool(drop <= 0.01) and report["fp32_parity_vs_sklearn"]["passes_atol_1e-4"]
    rpath.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    side = {"sha256": built["sha256"], "bytes": built["bytes"], "source_head_sha256": built["source_head_sha256"],
            "recipe": built["recipe"], "fp32_parity_ok": report["fp32_parity_vs_sklearn"]["passes_atol_1e-4"],
            "int8_passes_policy": report["int8_passes_policy"], "report": rpath.name}
    p8.with_suffix(".json").write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("fp32_parity_vs_sklearn", "accuracy", "int8_vs_fp32", "size_bytes",
                                             "row_vs_batch_int8_max_abs_centered_logit_diff", "int8_top1_drop_pp",
                                             "int8_passes_policy", "int8_sha256")}, indent=1))


if __name__ == "__main__":
    main()
