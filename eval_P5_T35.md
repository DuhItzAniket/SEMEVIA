1RUA24CSE0051,1RUA24CSE0097

# SEMEVIA comparative evaluation (PS5, T35)

Status: Bundle `ft_ckpt_2339418979e1_0920511710`: YOLO11n detector + fine-tuned EfficientNet-B0 (effb0_r1, 6 fine-tuning epoch(s), best tune top-1 0.7644); calibration kind `final`. The production pipeline was frozen in project phase P5 (`artifacts/production_freeze.json`: mode `hybrid`, model fingerprint `155e19183d986525`, bundle `ft_ckpt_2339418979e1_7aa5332188`) before any test image was evaluated; the final calibration was then added, which changes the bundle id but not the model fingerprint. Final values below come from the official Stanford Cars test split (8,041 images), evaluated once after the freeze. Tune-split numbers are selection-set diagnostics and are optimistic.

Fine-tuning: run `effb0_r1` (EfficientNet-B0, ImageNet weights), 6 completed epochs with pretrained blocks features.7, features.8 unfrozen and changed, fine-tuning requirement met: True (`reports/training/effb0_r1_summary.json`).

Test protocol: end-to-end on every test image (8041 images, manifest `5eb7f6150857`); a prediction counts only if the runtime-selected box has IoU >= 0.5 with the annotation and the class is right (misses count as failures). Classical mode uses the contour-proposal box; deep and hybrid use the YOLO11n primary target. Source `reports/e2e_test_ft_ckpt_2339418979e1_0920511710.json`.

## 1. Classification performance (Top-1 and Top-5)

Final, official test:

| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |
|---|---|---|---|---|
| classical | 6.3% (507/8041) | 14.8% (1192/8041) | 10.0% (507/5090) | 8.4% (519/6164) |
| deep | 75.1% (6035/8041) | 93.1% (7483/8041) | 75.7% (6035/7971) | 75.6% (6045/7992) |
| hybrid | 74.4% (5984/8041) | 93.2% (7493/8041) | 75.1% (5984/7971) | 75.0% (5994/7992) |

Comparison: on test the highest end-to-end top-1 is **deep** (75.1%); deep minus hybrid is +0.6 pt top-1 and -0.1 pt top-5. Classical-only is far behind (6.3% top-1): its contour box is often wrong (section 2) and handcrafted features alone separate few of the 196 classes. The production mode `hybrid` was fixed on the tune split, where the leader was `hybrid`; it was not changed after seeing test results.

Development, tune split (815 images, same protocol; `reports/e2e_tune_ft_ckpt_2339418979e1_7aa5332188.json`):

| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |
|---|---|---|---|---|
| classical | 6.5% (53/815) | 13.0% (106/815) | 10.5% (53/507) | 8.5% (54/634) |
| deep | 74.5% (607/815) | 92.5% (754/815) | 75.0% (607/809) | 74.8% (607/811) |
| hybrid | 75.6% (616/815) | 93.0% (758/815) | 76.1% (616/809) | 76.0% (616/811) |

Oracle-crop diagnostic (annotation crops, tune, used for head selection): classical 18.7% / 38.2%; deep 75.2% / 93.3%; hybrid 75.3% / 93.7% (top-1 / top-5; `reports/heads_tune_ckpt_2339418979e1.json`, `reports/classical_tune.json`).

## 2. Object localization (IoU)

Final, official test (one-to-one: the runtime-selected box vs the single annotated car):

| Path (mode) | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | No box | Wrong target |
|---|---|---|---|---|---|
| contour proposal (classical) | 0.680 | 0.521 | 63.3% (5090/8041) | 1877 | 1074 |
| YOLO11n primary target (hybrid) | 0.944 | 0.939 | 99.1% (7971/8041) | 49 | 21 |

Development, tune (`reports/e2e_tune_ft_ckpt_2339418979e1_7aa5332188.json`):

| Path (mode) | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | No box | Wrong target |
|---|---|---|---|---|---|
| contour proposal (classical) | 0.666 | 0.518 | 62.2% (507/815) | 181 | 127 |
| YOLO11n primary target (hybrid) | 0.944 | 0.939 | 99.3% (809/815) | 4 | 2 |

## 3. Object detection (mAP)

Final, official test: 196-category COCO AP (pycocotools bbox), at most one classified prediction per image, score = localisation score x top-1 probability:

| Mode | 196-category AP50 | 196-category AP50:95 | Predictions / ground truth |
|---|---|---|---|
| classical | 0.027 | 0.020 | 6164 / 8041 |
| deep | 0.701 | 0.647 | 7992 / 8041 |
| hybrid | 0.694 | 0.641 | 7992 / 8041 |

Generic vehicle AP (all YOLO11n car/bus/truck detections as one category), tune only (the detector evaluator does not run on test): AP50 0.847, AP50:95 0.771 (`reports/detector_tune.json`). Unannotated background cars count as false positives.

## 4. Model efficiency (model size and inference time)

Release bundle files (`artifacts/release_manifest.json`, bundle `ft_ckpt_2339418979e1_0920511710`):

| File | Size |
|---|---|
| yolo11n.pt | 5.6 MB |
| backbone_ckpt_2339418979e1.pth | 17.3 MB |
| head_classical.joblib | 1.5 MB |
| head_deep.joblib | 2.0 MB |
| head_hybrid.joblib | 3.1 MB |
| reference_index.npz | 1.5 MB |
| reference_index.json | 0.2 KB |
| calibration_final.json | 1.6 KB |
| **total** | **31.1 MB** |

Offline test-run timing (Intel(R) Core(TM) i7-14650HX CPU, 4 torch threads, 24 logical cores; offline batch timing, not user-visible latency): detector p50 37.6 ms, p95 89.3 ms per image; features + heads 28.1 ms per crop (mean, batched).

Optimization of the fine-tuned encoder (`reports/optimization_ckpt_2339418979e1.json`). Latency: encoder forward only, batch 1, warm, Intel(R) Core(TM) i7-14650HX CPU, 2 torch threads, 24 logical cores. Accuracy: hybrid head on 815 tune annotation (oracle) crops, so not end-to-end:

| Runtime | Size | p50 | p95 | Hybrid top-1 (tune, annotation crops) |
|---|---|---|---|---|
| PyTorch FP32 (released) | 17.3 MB | 14.4 ms | 16.7 ms | 75.3% |
| ONNX FP32 | 16.0 MB | 7.6 ms | 8.7 ms | 75.3% |
| ONNX INT8 static (MinMax) | 4.9 MB | 5.8 ms | 7.3 ms | 20.0% |
| ONNX INT8 static (percentile) | 4.9 MB | 5.9 ms | 6.9 ms | 55.7% |
| ONNX INT8 dynamic | 4.4 MB | 30.1 ms | 32.2 ms | 1.0% |

Selection policy: keep INT8 only if tune top-1 drop <= 1.0 percentage point for the production head (project choice, not a PS rule). Encoder INT8 variants passing it: none. ONNX FP32 keeps the same tune accuracy at lower encoder latency.

INT8 linear head (`reports/optimization_head_hybrid_ckpt_2339418979e1.json`): the hybrid head's final linear layer exported to ONNX, 1.05 MB (FP32) -> 0.26 MB (INT8), tune top-1 75.3% -> 75.3% (passes the policy: True). The head is a tiny part of end-to-end time; no meaningful end-to-end speed gain is claimed from it.

Released and frozen runtime: encoder `torch`, precision `fp32` (the optimized variants above are measured, not deployed).

Deployed end-to-end latency through the public URL (`reports/release_acceptance.json`, 2026-11-06; Cloudflare quick tunnel (trycloudflare.com) to Docker Desktop on the team laptop (Intel i7-14650HX); container limited to 2 CPUs / 4 GB by compose.yaml. The URL lasts only while the laptop, Docker and the tunnel keep running.): 20 fresh uploads, client wall time including the network, p50 515 ms, p95 788 ms, max 927 ms; 4 simultaneous clients all answered in 758 ms; 6 bad-input cases answered with typed errors.

## 5. Robustness under varying conditions

Final: 392-image predeclared test stress subset (seeded, stratified, up to 2 images per class), conditions applied to stored pixels before detection (`reports/robustness_stress_test_ft_ckpt_2339418979e1_0920511710.json`). Caveat from the report: Synthetic, fixed conditions applied before detection (design choices, not a standard benchmark). Artificial darkening is not real night imagery; downsampling is not distance.

| Condition | Mode | E2E top-1 | Change vs clean | Box recall@0.5 |
|---|---|---|---|---|
| clean | classical | 4.8% | +0.0 pt | 63.3% |
| clean | deep | 73.2% | +0.0 pt | 99.0% |
| clean | hybrid | 72.4% | +0.0 pt | 99.0% |
| bright_0.5 | classical | 4.1% | -0.8 pt | 68.4% |
| bright_0.5 | deep | 71.4% | -1.8 pt | 99.0% |
| bright_0.5 | hybrid | 70.4% | -2.0 pt | 99.0% |
| bright_1.5 | classical | 3.8% | -1.0 pt | 58.2% |
| bright_1.5 | deep | 72.2% | -1.0 pt | 99.5% |
| bright_1.5 | hybrid | 71.4% | -1.0 pt | 99.5% |
| down_0.50 | classical | 4.8% | +0.0 pt | 65.1% |
| down_0.50 | deep | 66.6% | -6.6 pt | 99.2% |
| down_0.50 | hybrid | 65.8% | -6.6 pt | 99.2% |
| down_0.25 | classical | 3.1% | -1.8 pt | 50.8% |
| down_0.25 | deep | 47.2% | -26.0 pt | 97.2% |
| down_0.25 | hybrid | 47.4% | -25.0 pt | 97.2% |
| occl_0.15 | classical | 2.6% | -2.3 pt | 64.0% |
| occl_0.15 | deep | 60.5% | -12.8 pt | 98.7% |
| occl_0.15 | hybrid | 60.2% | -12.2 pt | 98.7% |
| occl_0.30 | classical | 2.0% | -2.8 pt | 63.3% |
| occl_0.30 | deep | 44.9% | -28.3 pt | 95.2% |
| occl_0.30 | hybrid | 43.9% | -28.6 pt | 95.2% |

Natural viewpoint subgroups: 60 seeded random test photos (disjoint from the stress subset), viewpoint labelled by an agent before any test result existed, team review pending (`reports/viewpoint_labels.csv`; `reports/analysis_test_ft_ckpt_2339418979e1_0920511710.json`). Descriptive; small and imbalanced groups (labelled before results).

| Viewpoint | Photos | deep E2E top-1 | deep E2E top-5 | hybrid E2E top-1 | hybrid E2E top-5 |
|---|---|---|---|---|---|
| front | 7 | 85.7% (6) | 100.0% (7) | 85.7% (6) | 100.0% (7) |
| indeterminate | 2 | 50.0% (1) | 50.0% (1) | 50.0% (1) | 50.0% (1) |
| rear | 1 | 100.0% (1) | 100.0% (1) | 100.0% (1) | 100.0% (1) |
| side | 8 | 62.5% (5) | 87.5% (7) | 62.5% (5) | 87.5% (7) |
| three_quarter_front | 36 | 63.9% (23) | 100.0% (36) | 66.7% (24) | 100.0% (36) |
| three_quarter_rear | 6 | 66.7% (4) | 83.3% (5) | 66.7% (4) | 83.3% (5) |

## 6. Confidence and threshold analysis

Final, official test, production mode `hybrid`, calibration `final` (temperature 0.627 from the frozen tune fit, conformal quantile q 0.9571 from the untouched 815-image calibration partition, alpha 0.1):

* Calibration (matched-primary rows, n=7971; 70 misses/wrong targets excluded): ECE (10 bins) raw 0.191, temperature-scaled 0.015; NLL 1.079 -> 0.897.
* Reject threshold 0.3 (max calibrated probability): retention 93.4% (7511/8041), accepted end-to-end accuracy 78.4% (5888/7511).
* Passport candidate sets (all inputs): coverage 91.1% (7324/8041), mean size 3.44, median size 2, singletons 44.1% (3549/8041) with singleton accuracy 95.5% (3389/3549), empty sets 0.0% (0/8041).

Threshold choice (`artifacts/production_freeze.json`, tune known images, before the test run): rule 'no negative tuning set: largest threshold meeting the known-retention target', target known retention 90.0%, chosen threshold 0.3.

| Threshold | Tune known retention |
|---|---|
| 0.05 | 99.5% |
| 0.1 | 99.4% |
| 0.15 | 99.1% |
| 0.2 | 98.2% |
| 0.25 | 96.0% |
| 0.3 | 92.4% |
| 0.4 | 85.8% |
| 0.5 | 78.3% |

* Negative (unsupported-input) false acceptance: pending (negative suite not collected; the reject threshold was chosen on known tune images only).

## Ablations

Component ablations (tune, annotation crops; every variant refit on train with the same C grid and selected by tune top-1; `reports/heads_tune_ckpt_2339418979e1.json`, groups from `configs/heads.yaml`). The handcrafted branch is reduced by PCA to at most 64 components, so removing a group can leave the input size unchanged.

| Variant | Handcrafted groups | Branch balance | Input dim | C | Tune top-1 | Tune top-5 | Tune NLL |
|---|---|---|---|---|---|---|---|
| deep | none | per_feature | 1280 | 0.1 | 75.2% | 93.3% | 0.967 |
| hybrid | hog, lbp, hsv, geometry, flags | per_feature | 1344 | 0.02 | 75.3% | 93.7% | 1.098 |
| hybrid_equal_variance | hog, lbp, hsv, geometry, flags | equal_variance | 1344 | 0.02 | 68.7% | 91.3% | 1.293 |
| hybrid_no_hog | lbp, hsv, geometry, flags | per_feature | 1329 | 0.02 | 75.6% | 93.6% | 1.099 |
| hybrid_no_color_texture | hog, geometry, flags | per_feature | 1344 | 0.1 | 75.6% | 93.6% | 0.969 |
| hybrid_no_geometry | hog, lbp, hsv, flags | per_feature | 1344 | 0.02 | 75.6% | 93.7% | 1.098 |
| classical only (no learned features) | all | - | - | 0.005 | 18.7% | 38.2% | 3.966 |
| hybrid on the frozen ImageNet backbone (no fine-tuning; `reports/heads_tune_imagenet_frozen.json`) | as hybrid | per_feature | 1344 | 0.1 | 64.9% | 89.8% | 1.371 |

Change in tune top-1 versus the full hybrid head: deep: -0.1 pt; hybrid_equal_variance: -6.6 pt; hybrid_no_hog: +0.2 pt; hybrid_no_color_texture: +0.2 pt; hybrid_no_geometry: +0.2 pt; fine-tuning the backbone: +10.4 pt (hybrid, fine-tuned vs frozen). Reading: fine-tuning gives the largest gain; with the fine-tuned features, removing any single handcrafted group changes tune top-1 by well under one point, so the handcrafted branch adds little accuracy here, while equal-variance branch weighting clearly hurts.

## Failure analysis

Selected by fixed rules from the test predictions (`reports/analysis_test_ft_ckpt_2339418979e1_0920511710.json`). Image sizes from `artifacts/manifests/test_m1.csv`; duplicate lists from `reports/data_audit.json`.

1. **localisation failure (no box, or lowest IoU below 0.5)**: `test/00076.jpg`, true class *AM General Hummer SUV 2000*; no vehicle box selected.
   * Runtime top-1: none (no crop); annotation-crop top-1: *AM General Hummer SUV 2000* (p = 0.9999), correct.
   * HYPOTHESIS: localisation/crop caused the error (annotation crop is classified correctly)
   * Evidence: image 96x64 px, annotated car 74x48 px (58% of the image); not listed as a duplicate in the data audit; likely cause: the whole photo is a small thumbnail, so the car has too little detail for YOLO11n to produce a vehicle box above its score floor.
2. **close-category error (same make), highest confidence**: `test/05757.jpg`, true class *Aston Martin V8 Vantage Coupe 2012*; selected box IoU 0.923, detector score 0.896.
   * Runtime top-1: *Aston Martin Virage Coupe 2012* (p = 0.9995); annotation-crop top-1: *Aston Martin Virage Coupe 2012* (p = 0.9992), still wrong.
   * HYPOTHESIS: classifier confusion persists on the annotation crop
   * Evidence: image 300x200 px, annotated car 296x132 px (65% of the image); true class ranked 2 in the runtime top-5; prediction is a sibling model of the same make; not listed as a duplicate in the data audit; likely cause: classifier confusion, not localisation (the box is good and the annotation crop gives the same wrong answer); the very high probability shows over-confidence on visually close classes even after temperature scaling.
3. **confident cross-make error, highest confidence**: `test/02309.jpg`, true class *BMW M3 Coupe 2012*; selected box IoU 0.938, detector score 0.870.
   * Runtime top-1: *Lamborghini Aventador Coupe 2012* (p = 0.9929); annotation-crop top-1: *Lamborghini Aventador Coupe 2012* (p = 0.9928), still wrong.
   * HYPOTHESIS: classifier confusion persists on the annotation crop
   * Evidence: image 1024x640 px, annotated car 751x446 px (51% of the image); true class ranked 2 in the runtime top-5; prediction is a different make; not listed as a duplicate in the data audit; likely cause: classifier confusion, not localisation (the box is good and the annotation crop gives the same wrong answer); the very high probability shows over-confidence on visually close classes even after temperature scaling.

