"""Fallback optimization (plan P5): INT8 dense-linear head.

Usage: python scripts/optimize_head.py --mode hybrid --backbone imagenet_frozen
The fitted logistic-regression coefficients are copied into an ONNX MatMul+Add graph (after the
unchanged FP32 FusionFeatures transform). FP32 parity with sklearn is checked first; the graph is
then quantized with onnxruntime dynamic quantization (MatMulInteger, INT8 weights). Tune accuracy,
agreement, size and latency are measured. Writes reports/optimization_head_<mode>_<backbone>.json.
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
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from src.release import recognition_config

    rc = recognition_config()
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    pipe = joblib.load(ROOT / rc["heads"][args.mode])
    fuse, clf = pipe[0], pipe[-1]
    _, yl, XL, _ = load_cache(cache_meta("learned", "tune", "annotation", idx, args.backbone))
    _, _, XC, _ = load_cache(cache_meta("classical", "tune", "annotation", idx))
    X = np.concatenate([XL, XC], axis=1)
    F = fuse.transform(X).astype(np.float32)
    z_ref = canonical_logits(pipe, X)

    classes = np.asarray(clf.classes_)
    W = np.zeros((196, clf.coef_.shape[1]), np.float32)
    b = np.zeros(196, np.float32)
    W[classes], b[classes] = clf.coef_, clf.intercept_
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["features", "W"], ["mm"]), helper.make_node("Add", ["mm", "b"], ["logits"])],
        "semevia_linear_head",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, W.shape[1]])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 196])],
        [numpy_helper.from_array(W.T.copy(), "W"), numpy_helper.from_array(b, "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10  # onnxruntime 1.31 loads IR <= 13; onnx 1.23 defaults to 14
    onnx.checker.check_model(model)
    out = store_dir() / "optimized"
    out.mkdir(parents=True, exist_ok=True)
    p32, p8 = out / f"head_{args.mode}_{args.backbone}_fp32.onnx", out / f"head_{args.mode}_{args.backbone}_int8.onnx"
    onnx.save(model, str(p32))
    quantize_dynamic(str(p32), str(p8), weight_type=QuantType.QInt8, per_channel=False)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    s32 = ort.InferenceSession(str(p32), so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(p8), so, providers=["CPUExecutionProvider"])
    z32 = s32.run(None, {"features": F})[0].astype(np.float64)
    z8 = s8.run(None, {"features": F})[0].astype(np.float64)

    def c(z):
        return z - z.mean(1, keepdims=True)

    ops8 = sorted({n.op_type for n in onnx.load(str(p8)).graph.node})
    report = {
        "mode": args.mode, "backbone": args.backbone,
        "population": {"partition": "tune", "count": int(len(yl)), "crops": "annotation"},
        "method": "logistic coefficients copied to ONNX MatMul+Add after the unchanged FP32 fusion transform; "
                  "onnxruntime quantize_dynamic, INT8 weights (MatMulInteger)",
        "int8_graph_ops": ops8,
        "fp32_parity_vs_sklearn": {"max_abs_centered_logit_diff": float(np.abs(c(z32) - c(z_ref)).max()),
                                   "max_abs_prob_diff": float(np.abs(softmax(z32) - softmax(z_ref)).max()),
                                   "passes_atol_1e-4": bool(np.allclose(c(z32), c(z_ref), atol=1e-4, rtol=1e-4)),
                                   "top1_agreement": float(np.mean(z32.argmax(1) == z_ref.argmax(1)))},
        "accuracy": {"sklearn_fp32": {"tune_top1": topk_accuracy(z_ref, yl, 1), "tune_top5": topk_accuracy(z_ref, yl, 5)},
                     "onnx_fp32": {"tune_top1": topk_accuracy(z32, yl, 1), "tune_top5": topk_accuracy(z32, yl, 5)},
                     "onnx_int8": {"tune_top1": topk_accuracy(z8, yl, 1), "tune_top5": topk_accuracy(z8, yl, 5)}},
        "int8_vs_fp32": {"top1_agreement": float(np.mean(z8.argmax(1) == z_ref.argmax(1))),
                         "max_abs_centered_logit_diff": float(np.abs(c(z8) - c(z_ref)).max())},
        "size_bytes": {"onnx_fp32": p32.stat().st_size, "onnx_int8": p8.stat().st_size,
                       "sklearn_joblib_whole_head_incl_pca": (ROOT / rc["heads"][args.mode]).stat().st_size},
        "latency_single_row_head_only": {"onnx_fp32": lat(lambda x: s32.run(None, {"features": x}), F[:1]),
                                         "onnx_int8": lat(lambda x: s8.run(None, {"features": x}), F[:1]),
                                         "threads": 2, "hardware": "Intel i5-1135G7 CPU"},
        "note": "The head is a tiny part of end-to-end time; no meaningful end-to-end speed gain is claimed from it.",
    }
    drop = report["accuracy"]["sklearn_fp32"]["tune_top1"] - report["accuracy"]["onnx_int8"]["tune_top1"]
    report["int8_top1_drop_pp"] = round(100 * drop, 2)
    report["int8_passes_policy"] = bool(drop <= 0.01)
    (ROOT / "reports" / f"optimization_head_{args.mode}_{args.backbone}.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("fp32_parity_vs_sklearn", "accuracy", "int8_vs_fp32", "size_bytes",
                                             "latency_single_row_head_only", "int8_top1_drop_pp", "int8_passes_policy", "int8_graph_ops")}, indent=1))


if __name__ == "__main__":
    main()
