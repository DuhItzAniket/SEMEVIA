"""Optimization experiment: ONNX FP32 export parity and static INT8 quantization of the encoder.

Usage: python scripts/optimize.py --backbone imagenet_frozen
Steps: export features+avgpool (1280-d) to ONNX; check FP32 parity on a fixed tune sample; build a
static INT8 (QDQ, per-channel) model calibrated on fixed training crops; evaluate deep and hybrid
heads on all tune crops with each encoder; measure size and single-crop CPU latency at the
deployment thread count. Selection uses tuning data only. Writes reports/optimization_<backbone>.json.
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

from src.data.crops import annotation_crop, load_rgb  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.evaluation.hardware import hardware_label  # noqa: E402
from src.models.cache_io import cache_meta, load_cache, store_dir  # noqa: E402
from src.models.heads import canonical_logits, softmax, topk_accuracy  # noqa: E402


def crops_tensor(recs, data_root, margin):
    from src.models.backbone import crop_to_tensor

    return np.stack([crop_to_tensor(annotation_crop(load_rgb(data_root / r.rel_path), r.box, margin)[0]).numpy() for r in recs]).astype(np.float32)


def latency(fn, x1, runs=40):
    for _ in range(5):
        fn(x1)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(x1)
        ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "runs": runs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet_frozen")
    ap.add_argument("--data-root", default=os.environ.get("SEMEVIA_DATA_ROOT"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--calib-images", type=int, default=128)
    args = ap.parse_args()
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from src.models.inference import _load_backbone

    t_start = time.time()
    from src.release import recognition_config

    rc = recognition_config()
    margin = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))["crops"]["margin"]
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    data_root = Path(args.data_root)
    tune = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["tune"]["file"])
    train = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["train"]["file"])
    rng = np.random.RandomState(20261106)
    calib = [train[i] for i in sorted(rng.choice(len(train), args.calib_images, replace=False))]

    model = _load_backbone(ROOT / rc["backbone"])

    class Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.avgpool(self.m.features(x)).flatten(1)

    enc = Enc(model).eval()
    out = store_dir() / "optimized"
    out.mkdir(parents=True, exist_ok=True)
    fp32_path, pre_path = out / f"encoder_{args.backbone}_fp32.onnx", out / f"encoder_{args.backbone}_pre.onnx"
    torch.onnx.export(enc, torch.zeros(1, 3, 224, 224), str(fp32_path), input_names=["input"], output_names=["embedding"],
                      dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)

    print("loading tune crops...", flush=True)
    Xt = crops_tensor(tune, data_root, margin)
    torch.set_num_threads(4)
    with torch.inference_mode():
        E_torch = np.concatenate([enc(torch.from_numpy(Xt[i:i + 64])).numpy() for i in range(0, len(Xt), 64)])

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    s32 = ort.InferenceSession(str(fp32_path), so, providers=["CPUExecutionProvider"])
    E32 = np.concatenate([s32.run(None, {"input": Xt[i:i + 64]})[0] for i in range(0, len(Xt), 64)])

    calib_x = crops_tensor(calib, data_root, margin)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": calib_x[i:i + 1]} for i in range(len(calib_x))])

        def get_next(self):
            return next(self.it, None)

    from onnxruntime.quantization import CalibrationMethod, quantize_dynamic

    quant_pre_process(str(fp32_path), str(pre_path))
    variants = {}
    # Attempt 1: static QDQ, MinMax calibration. Attempt 2: static QDQ, percentile calibration.
    # Attempt 3: dynamic (weight-only INT8, activations quantized at run time).
    for vname, kind, method in (("onnx_int8", "static", CalibrationMethod.MinMax),
                                ("onnx_int8_percentile", "static", CalibrationMethod.Percentile),
                                ("onnx_int8_dynamic", "dynamic", None)):
        vpath = out / f"encoder_{args.backbone}_{vname}.onnx"
        t_q = time.time()
        try:
            if kind == "static":
                quantize_static(str(pre_path), str(vpath), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=method)
            else:
                quantize_dynamic(str(pre_path), str(vpath), weight_type=QuantType.QUInt8, per_channel=False)
            sess = ort.InferenceSession(str(vpath), so, providers=["CPUExecutionProvider"])
            E = np.concatenate([sess.run(None, {"input": Xt[i:i + 64]})[0] for i in range(0, len(Xt), 64)])
            variants[vname] = {"path": vpath, "session": sess, "E": E, "seconds": round(time.time() - t_q, 1), "error": None}
        except Exception as exc:
            variants[vname] = {"path": vpath, "session": None, "E": None, "seconds": round(time.time() - t_q, 1),
                               "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(vname, "done", variants[vname]["error"], flush=True)
    quant_seconds = sum(v["seconds"] for v in variants.values())

    import joblib

    _, yu, XC, _ = load_cache(cache_meta("classical", "tune", "annotation", idx))
    ku, yl, XL, _ = load_cache(cache_meta("learned", "tune", "annotation", idx, args.backbone))
    assert [r.class_id for r in tune] == list(yl)
    heads = {m: joblib.load(ROOT / rc["heads"][m]) for m in ("deep", "hybrid")}
    res = {}
    encs = [("torch_fp32", E_torch), ("onnx_fp32", E32)] + [(n, v["E"]) for n, v in variants.items() if v["E"] is not None]
    for name, E in encs:
        r = {}
        for m, h in heads.items():
            z = canonical_logits(h, np.concatenate([E.astype(np.float64), XC], axis=1))
            r[m] = {"tune_top1": topk_accuracy(z, yl, 1), "tune_top5": topk_accuracy(z, yl, 5), "logits": z}
        res[name] = r

    def agree(a, b):
        return float(np.mean(a.argmax(1) == b.argmax(1)))

    def centered(z):
        return z - z.mean(1, keepdims=True)

    parity = {}
    for m in heads:
        zt, z32 = res["torch_fp32"][m]["logits"], res["onnx_fp32"][m]["logits"]
        pt, p32 = softmax(zt), softmax(z32)
        parity[m] = {
            "onnx_fp32_vs_torch": {"max_abs_centered_logit_diff": float(np.abs(centered(zt) - centered(z32)).max()),
                                   "max_abs_prob_diff": float(np.abs(pt - p32).max()), "top1_agreement": agree(zt, z32),
                                   "passes_atol_1e-4_logits": bool(np.allclose(centered(zt), centered(z32), atol=1e-4, rtol=1e-4)),
                                   "passes_atol_1e-5_probs": bool(np.allclose(pt, p32, atol=1e-5, rtol=1e-4))},
            **{f"{n}_vs_torch": {"top1_agreement": agree(zt, res[n][m]["logits"]),
                                 "max_abs_centered_logit_diff": float(np.abs(centered(zt) - centered(res[n][m]["logits"])).max())}
               for n in variants if n in res}}
    x1 = Xt[:1]
    with torch.inference_mode():
        torch.set_num_threads(args.threads)
        lat_t = latency(lambda x: enc(torch.from_numpy(x)), x1)
    lat32 = latency(lambda x: s32.run(None, {"input": x}), x1)
    lat_variants = {n: latency(lambda x, ss=v["session"]: ss.run(None, {"input": x}), x1) for n, v in variants.items() if v["session"] is not None}
    pth = ROOT / rc["backbone"]
    report = {
        "backbone": args.backbone, "population": {"partition": "tune", "count": int(len(yl)), "crops": "annotation"},
        "calibration_data": {"partition": "train", "count": len(calib), "seed": 20261106, "selection": "seeded random"},
        "quantization": {"variants": {n: {"method": ("static QDQ per-channel QInt8 weights / QUInt8 activations, " +
                                                    ("MinMax" if n == "onnx_int8" else "Percentile")) if "dynamic" not in n
                                                   else "dynamic weight-only QUInt8", "seconds": v["seconds"], "error": v["error"],
                                                   "size_bytes": v["path"].stat().st_size if v["path"].exists() else None}
                                               for n, v in variants.items()},
                         "opset": 17, "seconds_total": round(quant_seconds, 1)},
        "accuracy": {n: {m: {k: v for k, v in r[m].items() if k != "logits"} for m in r} for n, r in res.items()},
        "parity": parity,
        "embedding_max_abs_diff": {"onnx_fp32": float(np.abs(E32 - E_torch).max()),
                                   **{n: float(np.abs(v["E"] - E_torch).max()) for n, v in variants.items() if v["E"] is not None}},
        "size_bytes": {"torch_state_dict_pth": pth.stat().st_size, "onnx_fp32": fp32_path.stat().st_size,
                       **{n: v["path"].stat().st_size for n, v in variants.items() if v["path"].exists()}},
        "latency_single_crop_encoder_only": {"threads": args.threads, "torch_fp32": lat_t, "onnx_fp32": lat32, **lat_variants,
                                             "hardware": hardware_label(args.threads), "note": "encoder forward only, batch 1, warm"},
        "selection_policy": "keep INT8 only if tune top-1 drop <= 1.0 percentage point for the production head (project choice, not a PS rule)",
        "wall_seconds": round(time.time() - t_start, 1),
    }
    base = report["accuracy"]["torch_fp32"]["hybrid"]["tune_top1"]
    report["int8_hybrid_top1_drop_pp"] = {n: round(100 * (base - report["accuracy"][n]["hybrid"]["tune_top1"]), 2)
                                          for n in variants if n in report["accuracy"]}
    report["int8_passes_policy"] = {n: bool(d <= 1.0) for n, d in report["int8_hybrid_top1_drop_pp"].items()}
    (ROOT / "reports" / f"optimization_{args.backbone}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("accuracy", "size_bytes", "int8_hybrid_top1_drop_pp", "int8_passes_policy")}, indent=1))
    print(json.dumps(report["latency_single_crop_encoder_only"], indent=1))
    print(json.dumps(parity, indent=1))


if __name__ == "__main__":
    main()
