# P5 checkpoint (development backbone): optimization experiment

Date: 2026-11-06. Backbone: frozen ImageNet EfficientNet-B0 (development). All accuracy numbers are on
the 815-image tune split with annotation crops; latency is single-crop, warm, 2 threads, Intel
i5-1135G7 CPU. Sources: `reports/optimization_imagenet_frozen.json`,
`reports/optimization_head_hybrid_imagenet_frozen.json`.

## Encoder (features + pooling, 1280-d)

| Runtime | Hybrid tune top-1 | Size | p50 / p95 latency |
|---|---|---|---|
| PyTorch FP32 (current default) | 0.6491 | 21.4 MB (.pth) | 51.8 / 69.0 ms |
| ONNX FP32 | 0.6491 (top-1 agreement 1.000; centered logits within 1.1e-4, probabilities within 1.5e-5) | 16.0 MB | 20.0 / 27.7 ms |
| ONNX static INT8, MinMax | 0.1742 (fails) | 4.9 MB | 17.5 / 25.7 ms |
| ONNX static INT8, percentile | 0.4613 (fails) | 4.9 MB | 20.2 / 25.0 ms |
| ONNX dynamic INT8 | 0.0074 (fails) | 4.4 MB | 105.2 / 119.6 ms |

Full-network INT8 failed the project policy (at most 1 point top-1 drop) in all three standard
attempts; the 60-minute INT8 cap was used. Hypothesis (not verified): depthwise convolutions and
SiLU/squeeze-excitation activations in EfficientNet quantize poorly with simple per-tensor
activation calibration.

## Fallback: INT8 linear head (plan-defined)

| Hybrid head | Tune top-1 | Tune top-5 | Size | Latency |
|---|---|---|---|---|
| sklearn / ONNX FP32 (parity 5.4e-6) | 0.6491 | 0.8982 | 1.05 MB | 0.06 ms |
| ONNX INT8 (MatMulInteger) | 0.6466 (-0.25 pt, agreement 0.991) | 0.8969 | 0.26 MB | 0.06 ms |

The head is a negligible part of end-to-end time; no speed gain is claimed from it.

## Decision (development)

* Executable optimized variants exist and are measured: ONNX FP32 encoder (~2.6x faster encoder,
  identical predictions) and INT8 head (4x smaller head, -0.25 pt).
* The app keeps the PyTorch FP32 runtime until the P5 freeze on the fine-tuned backbone. Switching
  the released runtime requires refitting temperature / calibration and rebuilding the reference
  index with the released encoder (cache invalidation matrix).
* Rerun on the fine-tuned checkpoint: `python scripts/optimize.py --backbone <tag>` and
  `python scripts/optimize_head.py --mode hybrid --backbone <tag>`.
