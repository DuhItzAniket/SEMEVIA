USNS_PENDING_TEAM_TO_PROVIDE (first line must list all member USNs, comma-separated)

# SEMEVIA comparative evaluation (PS5, TeamID)

Status: **development**. The backbone is frozen ImageNet EfficientNet-B0 (fine-tuning pending) and calibration is development-only. Final values are filled from the official 8,041-image test run; until then they read *pending*. Tune-split numbers are selection-set diagnostics and are optimistic.

## 1. Classification performance (Top-1 and Top-5)

Final (official test, end-to-end): pending (official test run after the P5 freeze, P6)

Development: tune split (815 images), runtime box rule, misses count as failures; source `reports/e2e_tune_dev_imagenet_frozen_097c99803f.json`.

| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |
|---|---|---|---|---|
| classical | 6.5% | 13.0% | 10.5% | 8.5% |
| deep | 64.5% | 88.8% | 65.0% | 64.9% |
| hybrid | 64.0% | 89.9% | 64.5% | 64.4% |

Oracle-crop diagnostic (annotation crops, tune): classical 18.7% / 38.2%; deep 64.3% / 90.1%; hybrid 64.9% / 89.8% (top-1 / top-5; `reports/heads_tune_imagenet_frozen.json`).

## 2. Object localization (IoU)

Final: pending (official test run after the P5 freeze, P6)

| Path | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | Missing |
|---|---|---|---|---|
| contour proposal | 0.666 | 0.518 | 62.2% | 181 |
| YOLO11n primary target | 0.944 | 0.939 | 99.3% | 4 |

(Development: tune split. One-to-one primary target vs the single annotated car.)

## 3. Object detection (mAP)

Final: pending (official test run after the P5 freeze, P6)

Generic vehicle AP (all YOLO11n vehicle detections, 1 category), tune: AP50 0.847, AP50:95 0.771 (`reports/detector_tune.json`). Unannotated background cars count as false positives.

196-category AP (max one classified prediction per image, score = localisation score x top-1 probability), tune: classical AP50 0.049 / AP50:95 0.036; deep AP50 0.598 / AP50:95 0.561; hybrid AP50 0.593 / AP50:95 0.556

## 4. Model efficiency (model size and inference time)

Encoder, single crop, 2 threads, Intel i5-1135G7 CPU (`reports/optimization_imagenet_frozen.json`):

| Runtime | Size | p50 | p95 | Hybrid tune top-1 |
|---|---|---|---|---|
| PyTorch FP32 | 21.4 MB | 51.8 ms | 69.0 ms | 64.9% |
| ONNX FP32 | 16.0 MB | 20.0 ms | 27.7 ms | 64.9% |
| ONNX INT8 static (MinMax) | 4.9 MB | 17.5 ms | 25.7 ms | 17.4% |
| ONNX INT8 static (percentile) | 4.9 MB | 20.2 ms | 25.0 ms | 46.1% |
| ONNX INT8 dynamic | 4.4 MB | 105.2 ms | 119.6 ms | 0.7% |

INT8 linear head (`reports/optimization_head_hybrid_imagenet_frozen.json`): 1.05 MB -> 0.26 MB, tune top-1 64.9% -> 64.7%.

Deployed end-to-end latency (p50/p95 over fresh uploads on the public host): pending (P7).

## 5. Robustness under varying conditions

Final (392-image test stress subset): pending (official test run after the P5 freeze, P6)

Development: 392-image tune stress subset, conditions applied before detection (`reports/robustness_stress_tune_dev_imagenet_frozen_504165ac73.json`).

| Condition | Mode | E2E top-1 | Change vs clean | Box recall@0.5 |
|---|---|---|---|---|
| clean | classical | 7.4% | +0.0 pt | 63.8% |
| clean | hybrid | 64.8% | +0.0 pt | 99.5% |
| bright_0.5 | classical | 6.4% | -1.0 pt | 70.7% |
| bright_0.5 | hybrid | 63.0% | -1.8 pt | 99.5% |
| bright_1.5 | classical | 6.1% | -1.3 pt | 56.6% |
| bright_1.5 | hybrid | 61.2% | -3.6 pt | 99.2% |
| down_0.50 | classical | 6.6% | -0.8 pt | 67.1% |
| down_0.50 | hybrid | 59.7% | -5.1 pt | 99.5% |
| down_0.25 | classical | 4.6% | -2.8 pt | 51.8% |
| down_0.25 | hybrid | 38.3% | -26.5 pt | 96.4% |
| occl_0.15 | classical | 3.8% | -3.6 pt | 64.3% |
| occl_0.15 | hybrid | 55.9% | -8.9 pt | 98.7% |
| occl_0.30 | classical | 3.3% | -4.1 pt | 65.1% |
| occl_0.30 | hybrid | 34.4% | -30.4 pt | 94.6% |

Natural viewpoint subgroups (60 labelled test photos, `reports/viewpoint_labels.csv`): pending (P6).

## 6. Confidence and threshold analysis

Final: pending (official test run after the P5 freeze, P6)

Development (tune, production mode hybrid, development calibration fitted on this split, so in-sample):

* ECE (10 bins, matched-primary n=809): raw 0.099, temperature-scaled 0.038; NLL 1.369 -> 1.309.
* Reject threshold 0.25: retention 92.4%, accepted end-to-end accuracy 67.5%.
* Passport sets (all inputs): coverage 90.2%, median size 3, singletons 21.5% with singleton accuracy 96.6%.
* Negative (unsupported-input) false acceptance: pending (negative suite not collected).

## Ablations and failure analysis

Ablations (tune, matched heads): see `reports/heads_tune_imagenet_frozen.json`. Failure cases: pending (selected from final test predictions in P6).

