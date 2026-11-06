1RUA24CSE0051,1RUA24CSE0097

# SEMEVIA comparative evaluation (PS5, T35)

Status: Bundle `ft_ckpt_4f3584f597d5_9662c2e703`: YOLO11n detector + fine-tuned EfficientNet-B0 (effb0_r2, 12 fine-tuning epoch(s), best tune top-1 0.9166); calibration kind `final`. The production pipeline was frozen in project phase P5 (`artifacts/production_freeze.json`: mode `hybrid`, model fingerprint `68e07695555c7579`, bundle `ft_ckpt_4f3584f597d5_c13883508a`) before this bundle's test run; the final calibration was then added, which changes the bundle id but not the model fingerprint. Final values below come from the official Stanford Cars test split (8,041 images), evaluated once for this bundle after its freeze; the same split was evaluated once earlier for the superseded iteration (see Model iterations). Tune-split numbers are selection-set diagnostics and are optimistic.

Fine-tuning: run `effb0_r2` (EfficientNet-B0, ImageNet weights; stages head, full_finetune), 18 completed fine-tuning epochs (early stopping on tune); the released checkpoint is the best one, from fine-tuning epoch 12 (this is the epoch count in the bundle's backbone label); the final pretrained blocks features.7, features.8 are verified changed; best tune top-1 91.7%; fine-tuning requirement met: True (`reports/training/effb0_r2_summary.json`).

Test protocol: end-to-end on every test image (8041 images, manifest `5eb7f6150857`); a prediction counts only if the runtime-selected box has IoU >= 0.5 with the annotation and the class is right (misses count as failures). Classical mode uses the contour-proposal box; deep and hybrid use the YOLO11n primary target. Source `reports/e2e_test_ft_ckpt_4f3584f597d5_9662c2e703.json`.

## Model iterations (each evaluated once on test after its own freeze)

| Iteration | Backbone | Production mode | Hybrid E2E top-1 | Deep E2E top-1 | Hybrid E2E top-5 | Source |
|---|---|---|---|---|---|---|
| r1 (superseded) | fine-tuned EfficientNet-B0 (effb0_r1, 6 fine-tuning epoch(s), best tune top-1 0.7644) | hybrid | 74.4% | 75.1% | 93.2% | `reports/e2e_test_ft_ckpt_2339418979e1_0920511710.json` |
| current | fine-tuned EfficientNet-B0 (effb0_r2, 12 fine-tuning epoch(s), best tune top-1 0.9166) | hybrid | 89.6% | 89.6% | 97.2% | `reports/e2e_test_ft_ckpt_4f3584f597d5_9662c2e703.json` |

How the second iteration came about, in order: the first iteration was frozen and tested once and its test result was known; the second iteration was then designed (full-network instead of last-stage fine-tuning, `configs/train_v2.yaml` says why) and trained on train/tune data. Before its test run, the decision record in `NEXT.md` ("Iteration r2 decision record") fixed that it replaces the first iteration whatever its test result, that hybrid stays the production mode, and that its freeze uses the tune negative suite. So the freeze rule changed too (reject threshold, from the freeze files: r1: threshold 0.3, negatives used False; current: threshold 0.4, negatives used True). Heads, calibration and freeze were redone on tune/calibration data before the single test run of this bundle. Both test results are reported; all numbers below are for the current iteration.

## 1. Classification performance (Top-1 and Top-5)

Final, official test:

| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |
|---|---|---|---|---|
| classical | 6.3% (507/8041) | 14.8% (1192/8041) | 10.0% (507/5090) | 8.4% (519/6164) |
| deep | 89.6% (7207/8041) | 97.3% (7821/8041) | 90.4% (7207/7971) | 90.3% (7218/7992) |
| hybrid | 89.6% (7205/8041) | 97.2% (7819/8041) | 90.4% (7205/7971) | 90.3% (7215/7992) |

Comparison: on test the highest end-to-end top-1 is **deep** (89.6%); deep minus hybrid is +0.02 pt top-1 and +0.02 pt top-5. Classical-only is far behind (6.3% top-1): its contour box is often wrong (section 2) and handcrafted features alone separate few of the 196 classes. The production mode `hybrid` was fixed before this bundle's test run (decision record in `NEXT.md`): on the tune split deep and hybrid were level (end-to-end top-1 732 vs 731 of 815 images), and the project rule keeps hybrid unless deep is clearly better on tune. It was not changed after seeing test results.

Development, tune split (815 images, same protocol; `reports/e2e_tune_ft_ckpt_4f3584f597d5_c13883508a.json`):

| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |
|---|---|---|---|---|
| classical | 6.5% (53/815) | 13.0% (106/815) | 10.5% (53/507) | 8.5% (54/634) |
| deep | 89.8% (732/815) | 97.9% (798/815) | 90.5% (732/809) | 90.5% (734/811) |
| hybrid | 89.7% (731/815) | 97.9% (798/815) | 90.4% (731/809) | 90.4% (733/811) |

Oracle-crop diagnostic (annotation crops, tune, used for head selection): classical 18.7% / 38.2%; deep 90.4% / 98.7%; hybrid 90.4% / 98.7% (top-1 / top-5; `reports/heads_tune_ckpt_4f3584f597d5.json`, `reports/classical_tune.json`).

## 2. Object localization (IoU)

Final, official test (one-to-one: the runtime-selected box vs the single annotated car):

| Path (mode) | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | No box | Wrong target |
|---|---|---|---|---|---|
| contour proposal (classical) | 0.680 | 0.521 | 63.3% (5090/8041) | 1877 | 1074 |
| YOLO11n primary target (hybrid) | 0.944 | 0.939 | 99.1% (7971/8041) | 49 | 21 |

Development, tune (`reports/e2e_tune_ft_ckpt_4f3584f597d5_c13883508a.json`):

| Path (mode) | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | No box | Wrong target |
|---|---|---|---|---|---|
| contour proposal (classical) | 0.666 | 0.518 | 62.2% (507/815) | 181 | 127 |
| YOLO11n primary target (hybrid) | 0.944 | 0.939 | 99.3% (809/815) | 4 | 2 |

## 3. Object detection (mAP)

Final, official test: 196-category COCO AP (pycocotools bbox), at most one classified prediction per image, score = localisation score x top-1 probability:

| Mode | 196-category AP50 | 196-category AP50:95 | Predictions / ground truth |
|---|---|---|---|
| classical | 0.027 | 0.020 | 6164 / 8041 |
| deep | 0.871 | 0.805 | 7992 / 8041 |
| hybrid | 0.870 | 0.804 | 7992 / 8041 |

Generic vehicle AP (all YOLO11n car/bus/truck detections as one category), tune only (the detector evaluator does not run on test): AP50 0.847, AP50:95 0.771 (`reports/detector_tune.json`). Unannotated background cars count as false positives.

## 4. Model efficiency (model size and inference time)

Release bundle files (`artifacts/release_manifest.json`, bundle `ft_ckpt_4f3584f597d5_9662c2e703`):

| File | Size |
|---|---|
| yolo11n.pt | 5.6 MB |
| backbone_ckpt_4f3584f597d5.pth | 17.3 MB |
| head_classical.joblib | 1.5 MB |
| head_deep.joblib | 2.0 MB |
| head_hybrid.joblib | 3.1 MB |
| reference_index.npz | 1.5 MB |
| reference_index.json | 0.2 KB |
| calibration_final.json | 1.6 KB |
| **total** | **31.1 MB** |

Offline test-run timing (Intel(R) Core(TM) i7-14650HX CPU, 4 torch threads, 24 logical cores; offline batch timing, not user-visible latency): detector p50 30.3 ms, p95 47.1 ms per image; features + heads 21.5 ms per crop (mean, batched).

Optimization of the fine-tuned encoder (`reports/optimization_ckpt_4f3584f597d5.json`). Latency: encoder forward only, batch 1, warm, Intel(R) Core(TM) i7-14650HX CPU, 2 torch threads, 24 logical cores. Accuracy: hybrid head on 815 tune annotation (oracle) crops, so not end-to-end:

| Runtime | Size | p50 | p95 | Hybrid top-1 (tune, annotation crops) |
|---|---|---|---|---|
| PyTorch FP32 (released) | 17.3 MB | 13.1 ms | 16.6 ms | 90.4% |
| ONNX FP32 | 16.0 MB | 7.0 ms | 7.8 ms | 90.4% |
| ONNX INT8 static (MinMax) | 4.9 MB | 5.5 ms | 7.5 ms | 61.8% |
| ONNX INT8 static (percentile) | 4.9 MB | 5.5 ms | 6.4 ms | 83.6% |
| ONNX INT8 dynamic | 4.4 MB | 29.2 ms | 37.0 ms | 1.3% |

Selection policy: keep INT8 only if tune top-1 drop <= 1.0 percentage point for the production head (project choice, not a PS rule). Encoder INT8 variants passing it: none. ONNX FP32 keeps the same tune accuracy at lower encoder latency but is measured, not deployed: inside a whole request the encoder is only part of the time (detector, features and Passport also run; see `docs/checkpoints/P5_dev.md`), and switching runtime would change the frozen model fingerprint, so it would need a new freeze.

INT8 linear head (`reports/optimization_head_hybrid_ckpt_4f3584f597d5.json`): the hybrid head's final linear layer exported to ONNX, 1.05 MB (FP32) -> 0.26 MB (INT8), tune top-1 90.4% -> 90.3% (passes the policy: True). The head is a tiny part of end-to-end time; no meaningful end-to-end speed gain is claimed from it.

Released and frozen runtime: encoder `torch`, precision `fp32` (the optimized variants above are measured, not deployed).

Deployed end-to-end latency through the public URL (`reports/release_acceptance.json`, 2026-11-06; Cloudflare quick tunnel (trycloudflare.com) to Docker Desktop on the team laptop (Intel i7-14650HX); container limited to 2 CPUs / 4 GB by compose.yaml. The URL lasts only while the laptop, Docker and the tunnel keep running.): 20 fresh uploads, client wall time including the network, p50 502 ms, p95 871 ms, max 881 ms; 4 simultaneous clients all answered in 817 ms; 5 invalid files rejected with typed errors and 1 no-car image answered `no_vehicle`.

## 5. Robustness under varying conditions

Final: 392-image predeclared test stress subset (seeded, stratified, up to 2 images per class), conditions applied to stored pixels before detection (`reports/robustness_stress_test_ft_ckpt_4f3584f597d5_9662c2e703.json`). Caveat from the report: Synthetic, fixed conditions applied before detection (design choices, not a standard benchmark). Artificial darkening is not real night imagery; downsampling is not distance.

| Condition | Mode | E2E top-1 | Change vs clean | Box recall@0.5 |
|---|---|---|---|---|
| clean | classical | 4.8% | +0.0 pt | 63.3% |
| clean | deep | 87.8% | +0.0 pt | 99.0% |
| clean | hybrid | 88.3% | +0.0 pt | 99.0% |
| bright_0.5 | classical | 4.1% | -0.8 pt | 68.4% |
| bright_0.5 | deep | 87.2% | -0.5 pt | 99.0% |
| bright_0.5 | hybrid | 87.0% | -1.3 pt | 99.0% |
| bright_1.5 | classical | 3.8% | -1.0 pt | 58.2% |
| bright_1.5 | deep | 88.0% | +0.3 pt | 99.5% |
| bright_1.5 | hybrid | 88.0% | -0.3 pt | 99.5% |
| down_0.50 | classical | 4.8% | +0.0 pt | 65.1% |
| down_0.50 | deep | 87.5% | -0.3 pt | 99.2% |
| down_0.50 | hybrid | 87.5% | -0.8 pt | 99.2% |
| down_0.25 | classical | 3.1% | -1.8 pt | 50.8% |
| down_0.25 | deep | 74.5% | -13.3 pt | 97.2% |
| down_0.25 | hybrid | 74.2% | -14.0 pt | 97.2% |
| occl_0.15 | classical | 2.6% | -2.3 pt | 64.0% |
| occl_0.15 | deep | 78.1% | -9.7 pt | 98.7% |
| occl_0.15 | hybrid | 78.1% | -10.2 pt | 98.7% |
| occl_0.30 | classical | 2.0% | -2.8 pt | 63.3% |
| occl_0.30 | deep | 65.3% | -22.4 pt | 95.2% |
| occl_0.30 | hybrid | 65.1% | -23.2 pt | 95.2% |

Natural viewpoint subgroups: 60 seeded random test photos (disjoint from the stress subset), viewpoint labelled by an agent before any test result existed, team review pending (`reports/viewpoint_labels.csv`; `reports/analysis_test_ft_ckpt_4f3584f597d5_9662c2e703.json`). Descriptive; small and imbalanced groups (labelled before results).

| Viewpoint | Photos | deep E2E top-1 | deep E2E top-5 | hybrid E2E top-1 | hybrid E2E top-5 |
|---|---|---|---|---|---|
| front | 7 | 100.0% (7) | 100.0% (7) | 100.0% (7) | 100.0% (7) |
| indeterminate | 2 | 50.0% (1) | 50.0% (1) | 50.0% (1) | 50.0% (1) |
| rear | 1 | 100.0% (1) | 100.0% (1) | 100.0% (1) | 100.0% (1) |
| side | 8 | 87.5% (7) | 87.5% (7) | 87.5% (7) | 87.5% (7) |
| three_quarter_front | 36 | 97.2% (35) | 100.0% (36) | 97.2% (35) | 100.0% (36) |
| three_quarter_rear | 6 | 83.3% (5) | 100.0% (6) | 83.3% (5) | 100.0% (6) |

## 6. Confidence and threshold analysis

Final, official test, production mode `hybrid`, calibration `final` (temperature 0.802 from the frozen tune fit, conformal quantile q 0.7757 from the untouched 815-image calibration partition, alpha 0.1):

* Calibration (matched-primary rows, n=7971; 70 misses/wrong targets excluded): ECE (10 bins) raw 0.069, temperature-scaled 0.030; NLL 0.452 -> 0.406.
* Reject threshold 0.4 (max calibrated probability): retention 94.4% (7592/8041), accepted end-to-end accuracy 93.0% (7057/7592).
* Passport candidate sets (all inputs): coverage 91.7% (7376/8041), mean size 2.23, median size 1, singletons 92.7% (7456/8041) with singleton accuracy 93.3% (6953/7456), empty sets 1.5% (117/8041).

Threshold choice (`artifacts/production_freeze.json`, tune known images, before the test run): rule 'minimise negative false acceptance subject to known retention >= target', target known retention 90.0%, chosen threshold 0.4 (ties on negative false acceptance go to the threshold with higher known retention).

| Threshold | Tune known retention | Tune negative false acceptance |
|---|---|---|
| 0.05 | 99.5% | 42.5% |
| 0.1 | 99.1% | 37.5% |
| 0.15 | 99.0% | 35.0% |
| 0.2 | 98.5% | 32.5% |
| 0.25 | 97.8% | 25.0% |
| 0.3 | 96.8% | 22.5% |
| 0.4 | 94.7% | 12.5% |
| 0.5 | 93.0% | 12.5% |


Unsupported inputs (held-out negative suite, 40 licensed Wikimedia Commons photos, evaluated after the freeze; `reports/negatives_test_ft_ckpt_4f3584f597d5_9662c2e703.json`, sources in `artifacts/negative_manifests/negatives_test.csv`). A photo is falsely accepted when the system returns a confident car identity (calibrated max probability >= 0.4); a detector miss counts as a correct rejection. Whole-system rejection (a detector miss counts as a success); classifier-level open-set ability is only measured on images that reached the classifier. Small stress set, not open-world.

| Bucket | Photos | Reached classifier | Falsely accepted | Whole-system false acceptance | Among photos that reached the classifier |
|---|---|---|---|---|---|
| non_vehicle | 10 | 0 | 0 | 0.0% | n/a |
| two_wheeler | 10 | 0 | 0 | 0.0% | n/a |
| unsupported_vehicle_type | 10 | 5 | 3 | 30.0% | 60.0% |
| unlisted_car_model | 10 | 10 | 5 | 50.0% | 50.0% |
| **all** | 40 | 15 | 8 | 20.0% | 53.3% |

Most non-cars are stopped by the detector (it only accepts car/bus/truck boxes). Once an unsupported vehicle reaches the classifier, the softmax reject rule stops only part of them, especially newer car models that look like supported ones: rejection by a confidence threshold is not an open-set detector.

The reject threshold was chosen on the separate 40-photo tune negative set together with known tune images (`artifacts/production_freeze.json`).

## 7. Ablations

Component ablations (tune, annotation crops; every variant refit on train with the same C grid and selected by tune top-1; `reports/heads_tune_ckpt_4f3584f597d5.json`, classical row from `reports/classical_tune.json`, groups from `configs/heads.yaml`). The handcrafted branch is reduced by PCA to at most 64 components, so removing a group can leave the input size unchanged.

| Variant | Handcrafted groups | Branch balance | Input dim | C | Tune top-1 | Tune top-5 | Tune NLL |
|---|---|---|---|---|---|---|---|
| deep | none | per_feature | 1280 | 0.1 | 90.4% | 98.7% | 0.469 |
| hybrid | hog, lbp, hsv, geometry, flags | per_feature | 1344 | 0.1 | 90.4% | 98.7% | 0.470 |
| hybrid_equal_variance | hog, lbp, hsv, geometry, flags | equal_variance | 1344 | 0.02 | 89.6% | 98.5% | 0.611 |
| hybrid_no_hog | lbp, hsv, geometry, flags | per_feature | 1329 | 0.02 | 90.3% | 98.4% | 0.554 |
| hybrid_no_color_texture | hog, geometry, flags | per_feature | 1344 | 0.1 | 90.4% | 98.7% | 0.470 |
| hybrid_no_geometry | hog, lbp, hsv, flags | per_feature | 1344 | 0.1 | 90.4% | 98.7% | 0.470 |
| classical only (no learned features) | all | - | - | 0.005 | 18.7% | 38.2% | 3.966 |
| hybrid on the frozen ImageNet backbone (no fine-tuning; `reports/heads_tune_imagenet_frozen.json`) | as hybrid | per_feature | 1344 | 0.1 | 64.9% | 89.8% | 1.371 |

Change in tune top-1 versus the full hybrid head: deep: +0.0 pt; hybrid_equal_variance: -0.9 pt; hybrid_no_hog: -0.1 pt; hybrid_no_color_texture: +0.0 pt; hybrid_no_geometry: +0.0 pt; fine-tuning the backbone: +25.5 pt (hybrid, fine-tuned vs frozen). Reading: fine-tuning the backbone gives by far the largest gain. Removing handcrafted feature groups (no_hog -0.1 pt, no_color_texture +0.0 pt, no_geometry +0.0 pt) moves tune top-1 by at most 0.1 pt, so with fine-tuned features the handcrafted branch adds little accuracy; the deep-only head is within 0.0 pt. Equal-variance branch weighting changes top-1 by -0.9 pt and raises tune NLL from 0.470 to 0.611. These are 815-image tune differences without a significance test.

## 8. Failure analysis

Selected by fixed rules from the test predictions (`reports/analysis_test_ft_ckpt_4f3584f597d5_9662c2e703.json`). Image sizes from `artifacts/manifests/test_m1.csv`; duplicate lists from `reports/data_audit.json`.

1. **localisation failure (no box, or lowest IoU below 0.5)**: `test/00076.jpg`, true class *AM General Hummer SUV 2000*; no vehicle box selected.
   * Runtime top-1: none (no crop); annotation-crop top-1: *AM General Hummer SUV 2000* (p = 0.9998), correct.
   * HYPOTHESIS: localisation/crop caused the error (annotation crop is classified correctly)
   * Evidence: image 96x64 px, annotated car 74x48 px (58% of the image); not listed as a duplicate in the data audit; no vehicle box was selected; likely cause: the whole photo is a small thumbnail, so the car carries little detail at the detector's input size.
2. **close-category error (same make), highest confidence**: `test/00294.jpg`, true class *Audi TTS Coupe 2012*; selected box IoU 0.971, detector score 0.900.
   * Runtime top-1: *Audi TT Hatchback 2011* (p = 0.9999); annotation-crop top-1: *Audi TT Hatchback 2011* (p = 0.9999), still wrong.
   * HYPOTHESIS: classifier confusion persists on the annotation crop
   * Evidence: image 1024x640 px, annotated car 893x500 px (68% of the image); true class ranked 3 in the runtime top-5; prediction is a sibling model of the same make; not listed as a duplicate in the data audit; likely cause: classifier confusion, not localisation (the box is good and the annotation crop gives the same wrong answer); the model is very confident and wrong, typical for near-identical sibling classes.
3. **confident cross-make error, highest confidence**: `test/06675.jpg`, true class *Chevrolet Express Cargo Van 2007*; selected box IoU 0.980, detector score 0.887.
   * Runtime top-1: *GMC Savana Van 2012* (p = 0.9989); annotation-crop top-1: *GMC Savana Van 2012* (p = 0.9987), still wrong.
   * HYPOTHESIS: classifier confusion persists on the annotation crop
   * Evidence: image 500x374 px, annotated car 364x207 px (40% of the image); true class ranked 2 in the runtime top-5; prediction is a different make; not listed as a duplicate in the data audit; likely cause: classifier confusion, not localisation (the box is good and the annotation crop gives the same wrong answer); the model is very confident and wrong, typical for near-identical sibling classes; Chevrolet Express and GMC Savana are badge-engineered versions of the same vehicle, so the classes are visually almost identical.

