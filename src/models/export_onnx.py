"""Export the EfficientNet-B0 encoder (features + avgpool, 1280-d) to ONNX FP32 with a parity gate.

Usage: python -m src.models.export_onnx --backbone artifacts/store/release/<weights>.pth \
           --out artifacts/store/release/encoder_fp32.onnx
The export is refused (non-zero exit) unless ONNX Runtime reproduces the PyTorch embeddings on a
fixed seeded batch within atol=rtol=1e-4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def export(backbone: Path, out: Path, threads: int = 2) -> dict:
    import onnxruntime as ort
    import torch

    from src.models.inference import _load_backbone

    m = _load_backbone(backbone)

    class Enc(torch.nn.Module):
        def __init__(self, mm):
            super().__init__()
            self.mm = mm

        def forward(self, x):
            return self.mm.avgpool(self.mm.features(x)).flatten(1)

    enc = Enc(m).eval()
    out.parent.mkdir(parents=True, exist_ok=True)
    from src.models.backbone import backbone_input_size

    size = backbone_input_size(backbone)
    torch.onnx.export(enc, torch.zeros(1, 3, size, size), str(out), input_names=["input"], output_names=["embedding"],
                      dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    x = np.random.RandomState(20261106).randn(4, 3, size, size).astype(np.float32)
    with torch.inference_mode():
        ref = enc(torch.from_numpy(x)).numpy()
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    got = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    diff = float(np.abs(got - ref).max())
    ok = bool(np.allclose(got, ref, atol=1e-4, rtol=1e-4))
    info = {"onnx": out.name, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size,
            "source_backbone_sha256": hashlib.sha256(Path(backbone).read_bytes()).hexdigest(),
            "parity_max_abs_diff": diff, "parity_ok": ok, "opset": 17}
    if not ok:
        out.unlink(missing_ok=True)
        raise SystemExit(f"ONNX parity failed (max abs diff {diff})")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    info = export(Path(args.backbone), Path(args.out))
    Path(args.out).with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
