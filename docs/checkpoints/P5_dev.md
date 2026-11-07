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

## In-app check (added after integrating the ONNX runtime)

`SEMEVIA_ENCODER_RUNTIME=onnx_fp32` serves the same predictions (hybrid top-1 probability 0.454727 vs
0.454729 on one test photo). Inside a full request, features + classifier took a median of 34.2 ms
with ONNX vs 38.8 ms with PyTorch (5 warm requests, same photo, this laptop): about 12% faster, much
less than the 2.6x isolated-encoder gain, because classical features and other per-request work are a
large share. With the ONNX runtime the reference panel is disabled by design until the index is
rebuilt with that encoder.

## Later outcome on the fine-tuned backbone (added 2026-11-07)

Everything above is the development backbone and stays as it was measured. The same study was re-run on the
fine-tuned r3 backbone `ckpt_98b1fb55cc6e` (`reports/optimization_ckpt_98b1fb55cc6e.json`,
`reports/optimization_head_hybrid_ckpt_98b1fb55cc6e.json`; values in section 4 of `eval_P5_T35.md`):

* INT8 encoder: all three variants failed the policy again. The encoder is not quantised; every release runs
  it in PyTorch FP32 (`precision: fp32`).
* INT8 hybrid head: re-run row-wise, as served (the earlier batch-wise values are kept under
  `superseded_batchwise` in the same report). It passed the policy: tune top-1 94.2% (sklearn FP32) -> 94.2%
  (INT8) (`eval_P5_T35.md`). It is **deployed** since iteration r3-int8 (bundle
  `ft_ckpt_98b1fb55cc6e_f3734d4f40`, decision record in `NEXT.md`): the hybrid head's final logistic layer is
  served as an ONNX INT8 graph derived from the published `head_hybrid.joblib` by `src/models/head_onnx.py`,
  pinned by sha256. Temperature, reject threshold and conformal q were re-derived from INT8 logits on
  tune/calibration data, and the bundle was evaluated once end to end on test. The deep and classical heads stay
  sklearn FP32.
* The ONNX FP32 encoder is still measured, not deployed.
* Only the INT8 head is deployed, and it is a tiny part of request time, so no end-to-end speed gain is
  claimed from it.
