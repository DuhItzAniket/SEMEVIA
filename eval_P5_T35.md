1RUA24CSE0051,1RUA24CSE0097

# SEMEVIA comparative evaluation (PS5, T35)

Status: Bundle `ft_ckpt_98b1fb55cc6e_5d755416fd`: YOLO11n detector + fine-tuned EfficientNet-B0 (effb0_r3, 21 fine-tuning epoch(s), best tune top-1 0.9485); calibration kind `final`. The production pipeline was frozen in project phase P5 (`artifacts/production_freeze.json`: mode `hybrid`, model fingerprint `5d97761f5b6c726d`, bundle `ft_ckpt_98b1fb55cc6e_f99e386191`) before this bundle's test run; the final calibration was then added, which changes the bundle id but not the model fingerprint. Final values below come from the official Stanford Cars test split (8,041 images), evaluated once end to end for this bundle after its freeze. The same split was used earlier: 2 end-to-end run(s) of superseded iterations and 1 ground-truth-crop run(s) of a superseded frozen bundle (files `reports/e2e_test_*.json`, `reports/oracle_test_*.json`; see Model iterations). Tune-split numbers are selection-set diagnostics and are optimistic.

Fine-tuning: run `effb0_r3` (EfficientNet-B0, ImageNet weights, 320 px input; stages head, full_finetune), 24 completed fine-tuning epochs (full schedule, no early stop); the released checkpoint is the best one, from fine-tuning epoch 21 (this is the epoch count in the bundle's backbone label); the final pretrained blocks features.7, features.8 are verified changed; best tune top-1 94.8%; fine-tuning requirement met: True (`reports/training/effb0_r3_summary.json`).

Test protocol: end-to-end on every test image (8041 images, manifest `5eb7f6150857`); a prediction counts only if the runtime-selected box has IoU >= 0.5 with the annotation and the class is right (misses count as failures). Classical mode uses the contour-proposal box; deep and hybrid use the YOLO11n primary target. Source `reports/e2e_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`.

Dataset: Stanford Cars, 16,185 images in 196 categories (49 makes, model years 1991-2012); official split 8,144 train / 8,041 test. Every image's bounding box and make/model/year label are used. The official train split is divided by a fixed seed into train 6,514 / tune 815 / calibration 815; the test split is only used for the runs listed here (`reports/data_audit.json`).

Preprocessing: uploads are checked (at most 10 MB and 25 MP, formats JPEG, PNG, WEBP, BMP, MPO), EXIF-rotated, flattened to RGB and capped at a 1600 px long edge. The detected box is expanded by 5% and cropped. Classical branch: grayscale, Gaussian blur (kernel 5), Canny (50/150), morphological close, external contours, all on a 256 px aspect-preserving resize; HOG and uniform LBP on a 128x128 grayscale ROI; HSV histograms normalised per channel (1816 features, version `hc_v1`, standardised before the heads). Deep branch: square resize to the training input size (320 px) and ImageNet mean/std normalisation.

Role of the classical branch: it gives the interpretable views and measurements shown in the app, a fully classical mode (contour box plus handcrafted features), and 64 PCA components of the handcrafted features that the hybrid head concatenates with the CNN embedding. On clean test images it adds no accuracy over the fine-tuned CNN (deep vs hybrid in section 1, ablations in section 7); its effect under stress conditions is in section 5.

## Model iterations (each evaluated once on test after its own freeze)

| Iteration | Backbone | Production mode | Hybrid E2E top-1 | Deep E2E top-1 | Hybrid E2E top-5 | Source |
|---|---|---|---|---|---|---|
| r1 (superseded) | fine-tuned EfficientNet-B0 (effb0_r1, 6 fine-tuning epoch(s), best tune top-1 0.7644) | hybrid | 74.4% | 75.1% | 93.2% | `reports/e2e_test_ft_ckpt_2339418979e1_0920511710.json` |
| r2 (superseded) | fine-tuned EfficientNet-B0 (effb0_r2, 12 fine-tuning epoch(s), best tune top-1 0.9166) | hybrid | 89.6% | 89.6% | 97.2% | `reports/e2e_test_ft_ckpt_4f3584f597d5_9662c2e703.json` |
| current | fine-tuned EfficientNet-B0 (effb0_r3, 21 fine-tuning epoch(s), best tune top-1 0.9485) | hybrid | 91.8% | 91.8% | 98.0% | `reports/e2e_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json` |

How the iterations came about, in order (each later iteration was designed after the previous one's test result was known, so each carries a decision record in `NEXT.md` written before its own test run):

* r2: full-network instead of last-stage fine-tuning (`configs/train_v2.yaml` says why). Its decision record ("Iteration r2 decision record") fixed, before its test run, that it replaces r1 whatever its test result, that hybrid stays the production mode, and that its freeze uses the tune negative suite.
* r3: r2's recipe at 320 px input instead of 224 (`configs/train_v3.yaml`). Its decision record ("Iteration r3 decision record") was written before r3 was trained and released r3 only if its tune end-to-end hybrid top-1 beat r2's by at least one point; otherwise r3 would never have touched the test split. The outcome was recorded before the test run (`reports/e2e_tune_<r3 bundle at freeze>.json`).
* The reject rule changed across iterations (from the freeze files: r1: threshold 0.3, negatives used False; r2: threshold 0.4, negatives used True; current: threshold 0.5, negatives used True).

Heads, calibration and freeze were redone on tune/calibration data for every iteration before its single end-to-end test run. All 3 test results are reported; all numbers below are for the current iteration.

## 1. Classification performance (Top-1 and Top-5)

Final, official test:

| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |
|---|---|---|---|---|
| classical | 6.3% (507/8041) | 14.8% (1192/8041) | 10.0% (507/5090) | 8.4% (519/6164) |
| deep | 91.8% (7385/8041) | 97.9% (7870/8041) | 92.6% (7385/7971) | 92.6% (7397/7992) |
| hybrid | 91.8% (7382/8041) | 98.0% (7879/8041) | 92.6% (7382/7971) | 92.5% (7394/7992) |

Standard Stanford Cars protocol on the same frozen model (classify the ground-truth car region, no detector; the usual way published results are reported; `reports/oracle_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`):

| Mode | Top-1 (ground-truth crop) | Top-5 (ground-truth crop) |
|---|---|---|
| classical | 19.6% (1573/8041) | 40.9% (3285/8041) |
| deep | 92.7% (7454/8041) | 98.6% (7932/8041) |
| hybrid | 92.7% (7452/8041) | 98.7% (7939/8041) |

Comparison: on test the highest end-to-end top-1 is **deep** (91.8%); deep minus hybrid is +0.04 pt top-1 and -0.11 pt top-5. Classical-only is far behind (6.3% top-1): its contour box is often wrong (section 2) and handcrafted features alone separate few of the 196 classes. The production mode `hybrid` was fixed before this bundle's test run (decision record in `NEXT.md`): on the tune split deep and hybrid were level (end-to-end top-1 759 vs 759 of 815 images), and the project rule keeps hybrid unless deep is clearly better on tune. It was not changed after seeing test results.

Uncertainty (95% intervals from 2000 image-level bootstrap resamples of the saved test predictions; no new inference; `reports/extra_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`): end-to-end top-1 classical 6.3% [5.8%, 6.8%], deep 91.8% [91.3%, 92.4%], hybrid 91.8% [91.2%, 92.4%]. Paired deep minus hybrid top-1: +0.04 pt [-0.10, +0.16] pt, so no significant difference was detected on test (this is not a test of equivalence).

Development, tune split (815 images, same protocol; `reports/e2e_tune_ft_ckpt_98b1fb55cc6e_f99e386191.json`):

| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |
|---|---|---|---|---|
| classical | 6.5% (53/815) | 13.0% (106/815) | 10.5% (53/507) | 8.5% (54/634) |
| deep | 93.1% (759/815) | 98.2% (800/815) | 93.8% (759/809) | 93.8% (761/811) |
| hybrid | 93.1% (759/815) | 98.2% (800/815) | 93.8% (759/809) | 93.8% (761/811) |

Oracle-crop diagnostic (annotation crops, tune, used for head selection): classical 18.7% / 38.2%; deep 94.1% / 98.8%; hybrid 94.2% / 98.8% (top-1 / top-5; `reports/heads_tune_ckpt_98b1fb55cc6e.json`, `reports/classical_tune.json`).

## 2. Object localization (IoU)

Final, official test (one-to-one: the runtime-selected box vs the single annotated car):

| Path (mode) | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | No box | Wrong target |
|---|---|---|---|---|---|
| contour proposal (classical) | 0.680 | 0.521 | 63.3% (5090/8041) | 1877 | 1074 |
| YOLO11n primary target (hybrid) | 0.944 | 0.939 | 99.1% (7971/8041) | 49 | 21 |

Development, tune (`reports/e2e_tune_ft_ckpt_98b1fb55cc6e_f99e386191.json`):

| Path (mode) | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | No box | Wrong target |
|---|---|---|---|---|---|
| contour proposal (classical) | 0.666 | 0.518 | 62.2% (507/815) | 181 | 127 |
| YOLO11n primary target (hybrid) | 0.944 | 0.939 | 99.3% (809/815) | 4 | 2 |

## 3. Object detection (mAP)

Final, official test: 196-category COCO AP (pycocotools bbox), at most one classified prediction per image, score = localisation score x top-1 probability:

| Mode | 196-category AP50 | 196-category AP50:95 | Predictions / ground truth |
|---|---|---|---|
| classical | 0.027 | 0.020 | 6164 / 8041 |
| deep | 0.896 | 0.828 | 7992 / 8041 |
| hybrid | 0.896 | 0.827 | 7992 / 8041 |

Generic vehicle AP (all YOLO11n car/bus/truck detections as one category), tune only (the detector evaluator does not run on test): AP50 0.847, AP50:95 0.771 (`reports/detector_tune.json`). Unannotated background cars count as false positives.

## 4. Model efficiency (model size and inference time)

Release bundle files (`artifacts/release_manifest.json`, bundle `ft_ckpt_98b1fb55cc6e_5d755416fd`):

| File | Size |
|---|---|
| yolo11n.pt | 5.6 MB |
| backbone_ckpt_98b1fb55cc6e.pth | 17.3 MB |
| head_classical.joblib | 1.5 MB |
| head_deep.joblib | 2.0 MB |
| head_hybrid.joblib | 3.1 MB |
| reference_index.npz | 1.5 MB |
| reference_index.json | 0.2 KB |
| calibration_final.json | 1.6 KB |
| **total** | **31.1 MB** |

Offline test-run timing (Intel(R) Core(TM) i7-14650HX CPU, 4 torch threads, 24 logical cores; offline batch timing, not user-visible latency): detector p50 36.9 ms, p95 49.7 ms per image; features + heads 39.8 ms per crop (mean, batched).

Optimization of the fine-tuned encoder (`reports/optimization_ckpt_98b1fb55cc6e.json`). Latency: encoder forward only, batch 1, warm, Intel(R) Core(TM) i7-14650HX CPU, 2 torch threads, 24 logical cores. Accuracy: hybrid head on 815 tune annotation (oracle) crops, so not end-to-end:

| Runtime | Size | p50 | p95 | Hybrid top-1 (tune, annotation crops) |
|---|---|---|---|---|
| PyTorch FP32 (released) | 17.3 MB | 36.7 ms | 50.3 ms | 94.2% |
| ONNX FP32 | 16.0 MB | 19.6 ms | 24.7 ms | 94.2% |
| ONNX INT8 static (MinMax) | 4.9 MB | 13.3 ms | 23.7 ms | 45.8% |
| ONNX INT8 static (percentile) | 4.9 MB | 15.8 ms | 23.5 ms | 86.1% |
| ONNX INT8 dynamic | 4.4 MB | 59.9 ms | 87.5 ms | 0.5% |

Selection policy: keep INT8 only if tune top-1 drop <= 1.0 percentage point for the production head (project choice, not a PS rule). Encoder INT8 variants passing it: none. ONNX FP32 keeps the same tune accuracy at lower encoder latency but is measured, not deployed: inside a whole request the encoder is only part of the time (detector, features and Passport also run; see `docs/checkpoints/P5_dev.md`), and switching runtime would change the frozen model fingerprint, so it would need a new freeze.

INT8 linear head (`reports/optimization_head_hybrid_ckpt_98b1fb55cc6e.json`): the hybrid head's final linear layer exported to ONNX, 1.05 MB (FP32) -> 0.26 MB (INT8), tune top-1 94.2% -> 94.2% (passes the policy: True). The head is a tiny part of end-to-end time; no meaningful end-to-end speed gain is claimed from it.

Released and frozen runtime: encoder `torch`, precision `fp32` (the optimized variants above are measured, not deployed).

Deployed end-to-end latency through the public URL (`reports/release_acceptance.json`, 2026-11-07; Cloudflare quick tunnel (trycloudflare.com) to Docker Desktop on the team laptop (Intel i7-14650HX); container limited to 2 CPUs / 4 GB by compose.yaml. The URL lasts only while the laptop, Docker and the tunnel keep running.): 20 fresh uploads, client wall time including the network, p50 463 ms, p95 680 ms, max 688 ms; 4 simultaneous clients all answered in 741 ms; 5 invalid files rejected with typed errors and 1 no-car image answered `no_vehicle`.

## 5. Robustness under varying conditions

Final: 392-image predeclared test stress subset (seeded, stratified, up to 2 images per class), conditions applied to stored pixels before detection (`reports/robustness_stress_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`). Caveat from the report: Synthetic, fixed conditions applied before detection (design choices, not a standard benchmark). Artificial darkening is not real night imagery; downsampling is not distance.

| Condition | Mode | E2E top-1 | Change vs clean | Box recall@0.5 |
|---|---|---|---|---|
| clean | classical | 4.8% | +0.0 pt | 63.3% |
| clean | deep | 91.8% | +0.0 pt | 99.0% |
| clean | hybrid | 92.1% | +0.0 pt | 99.0% |
| bright_0.5 | classical | 4.1% | -0.8 pt | 68.4% |
| bright_0.5 | deep | 91.6% | -0.3 pt | 99.0% |
| bright_0.5 | hybrid | 91.6% | -0.5 pt | 99.0% |
| bright_1.5 | classical | 3.8% | -1.0 pt | 58.2% |
| bright_1.5 | deep | 90.6% | -1.3 pt | 99.5% |
| bright_1.5 | hybrid | 90.1% | -2.0 pt | 99.5% |
| down_0.50 | classical | 4.8% | +0.0 pt | 65.1% |
| down_0.50 | deep | 91.6% | -0.3 pt | 99.2% |
| down_0.50 | hybrid | 91.1% | -1.0 pt | 99.2% |
| down_0.25 | classical | 3.1% | -1.8 pt | 50.8% |
| down_0.25 | deep | 77.6% | -14.3 pt | 97.2% |
| down_0.25 | hybrid | 77.0% | -15.1 pt | 97.2% |
| occl_0.15 | classical | 2.6% | -2.3 pt | 64.0% |
| occl_0.15 | deep | 86.2% | -5.6 pt | 98.7% |
| occl_0.15 | hybrid | 86.2% | -5.9 pt | 98.7% |
| occl_0.30 | classical | 2.0% | -2.8 pt | 63.3% |
| occl_0.30 | deep | 70.2% | -21.7 pt | 95.2% |
| occl_0.30 | hybrid | 70.7% | -21.4 pt | 95.2% |

Natural viewpoint subgroups: 60 seeded random test photos (disjoint from the stress subset), viewpoint labelled by an agent before any test result existed, team review pending (`reports/viewpoint_labels.csv`; `reports/analysis_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`). Descriptive; small and imbalanced groups (labelled before results).

| Viewpoint | Photos | deep E2E top-1 | deep E2E top-5 | hybrid E2E top-1 | hybrid E2E top-5 |
|---|---|---|---|---|---|
| front | 7 | 100.0% (7) | 100.0% (7) | 100.0% (7) | 100.0% (7) |
| indeterminate | 2 | 50.0% (1) | 50.0% (1) | 50.0% (1) | 50.0% (1) |
| rear | 1 | 100.0% (1) | 100.0% (1) | 100.0% (1) | 100.0% (1) |
| side | 8 | 75.0% (6) | 87.5% (7) | 75.0% (6) | 87.5% (7) |
| three_quarter_front | 36 | 94.4% (34) | 100.0% (36) | 97.2% (35) | 100.0% (36) |
| three_quarter_rear | 6 | 100.0% (6) | 100.0% (6) | 100.0% (6) | 100.0% (6) |

Native resolution: all 8,041 test images grouped by the width of the annotated car box in the original photo (bins fixed in `scripts/analyze_resolution.py` before computing; from the saved test predictions, no new test run; `reports/resolution_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`).

| Car box width | Photos | deep E2E top-1 | hybrid E2E top-1 |
|---|---|---|---|
| under 150 px | 102 | 78.4% (80) | 78.4% (80) |
| 150-299 px | 1,609 | 91.7% (1476) | 91.8% (1477) |
| 300-499 px | 2,173 | 92.2% (2004) | 92.2% (2003) |
| 500 px and over | 4,157 | 92.0% (3825) | 91.9% (3822) |

Natural occlusion in the labelled sample: deep: none 51/56, partial 4/4; hybrid: none 52/56, partial 4/4. Labelled sample from reports/viewpoint_labels.csv (team review pending); groups are tiny and descriptive only.

## 6. Confidence and threshold analysis

Final, official test, production mode `hybrid`, calibration `final` (temperature 0.659 from the frozen tune fit, conformal quantile q 0.6011 from the untouched 815-image calibration partition, alpha 0.1):

* Calibration (matched-primary rows, n=7971; 70 misses/wrong targets excluded): ECE (10 bins) raw 0.131, temperature-scaled 0.027; NLL 0.450 -> 0.319.
* Reject threshold 0.5 (max calibrated probability): retention 94.1% (7567/8041), accepted end-to-end accuracy 94.9% (7183/7567).
* Passport candidate sets (all inputs): coverage 91.7% (7373/8041), mean size 2.16, median size 1, singletons 95.2% (7654/8041) with singleton accuracy 94.9% (7263/7654), empty sets 3.4% (274/8041).

Threshold choice (`artifacts/production_freeze.json`, tune known images, before the test run): rule 'minimise negative false acceptance subject to known retention >= target', target known retention 90.0%, chosen threshold 0.5 (ties on negative false acceptance go to the threshold with higher known retention).

| Threshold | Tune known retention | Tune negative false acceptance |
|---|---|---|
| 0.05 | 99.5% | 42.5% |
| 0.1 | 99.3% | 40.0% |
| 0.15 | 98.9% | 37.5% |
| 0.2 | 98.3% | 35.0% |
| 0.25 | 98.2% | 22.5% |
| 0.3 | 97.4% | 17.5% |
| 0.4 | 96.1% | 15.0% |
| 0.5 | 94.7% | 10.0% |


Unsupported inputs (held-out negative suite, 40 licensed Wikimedia Commons photos, evaluated after the freeze; `reports/negatives_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`, sources in `artifacts/negative_manifests/negatives_test.csv`). A photo is falsely accepted when the system returns a confident car identity (calibrated max probability >= 0.5); a detector miss counts as a correct rejection. Whole-system rejection (a detector miss counts as a success); classifier-level open-set ability is only measured on images that reached the classifier. Small stress set, not open-world.

| Bucket | Photos | Reached classifier | Falsely accepted | Whole-system false acceptance | Among photos that reached the classifier |
|---|---|---|---|---|---|
| non_vehicle | 10 | 0 | 0 | 0.0% | n/a |
| two_wheeler | 10 | 0 | 0 | 0.0% | n/a |
| unsupported_vehicle_type | 10 | 5 | 4 | 40.0% | 80.0% |
| unlisted_car_model | 10 | 10 | 1 | 10.0% | 10.0% |
| **all** | 40 | 15 | 5 | 12.5% | 33.3% |

Most non-cars are stopped by the detector (it only accepts car/bus/truck boxes). Once an unsupported vehicle reaches the classifier, the softmax reject rule stops only part of them, especially newer car models that look like supported ones: rejection by a confidence threshold is not an open-set detector.

The reject threshold was chosen on the separate 40-photo tune negative set together with known tune images (`artifacts/production_freeze.json`).

Reliability (production mode `hybrid`, 7971 correctly localised test images, `reports/extra_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`, figure `reports/figures/reliability_test.png`): ECE 0.131 with raw softmax, 0.027 after temperature scaling.

| Confidence bin | Images | Raw: mean confidence / accuracy | Calibrated: mean confidence / accuracy |
|---|---|---|---|
| 0.0-0.1 | 89 / 13 | 7.2% / 24.7% (n=89) | 8.6% / 15.4% (n=13) |
| 0.1-0.2 | 237 / 58 | 15.2% / 54.0% (n=237) | 15.7% / 24.1% (n=58) |
| 0.2-0.3 | 228 / 88 | 25.3% / 72.8% (n=228) | 24.8% / 42.0% (n=88) |
| 0.3-0.4 | 271 / 112 | 35.1% / 81.5% (n=271) | 34.7% / 50.0% (n=112) |
| 0.4-0.5 | 326 / 146 | 44.9% / 78.5% (n=326) | 45.8% / 61.6% (n=146) |
| 0.5-0.6 | 366 / 184 | 55.2% / 84.7% (n=366) | 55.1% / 61.4% (n=184) |
| 0.6-0.7 | 444 / 188 | 65.4% / 92.6% (n=444) | 65.3% / 76.1% (n=188) |
| 0.7-0.8 | 668 / 272 | 75.4% / 94.8% (n=668) | 75.1% / 81.6% (n=272) |
| 0.8-0.9 | 1189 / 407 | 85.8% / 97.1% (n=1189) | 85.6% / 89.7% (n=407) |
| 0.9-1.0 | 4153 / 6503 | 95.8% / 98.2% (n=4153) | 98.9% / 97.5% (n=6503) |

## 7. Ablations

Cross-validation of the classical head (5-fold stratified, 6514 training images, C=0.005, `reports/crossval_classical.json`): top-1 17.6% +/- 0.8 pt, top-5 37.4% +/- 0.7 pt (tune top-1 for reference 18.7%). The fine-tuned CNN was trained on the training split, so its features on any training fold are optimistic; cross-validating the deep or hybrid head there would overstate accuracy. Their uncertainty is reported with bootstrap intervals on the test predictions (reports/extra_test_<bundle>.json).

Component ablations (tune, annotation crops; every variant refit on train with the same C grid and selected by tune top-1; `reports/heads_tune_ckpt_98b1fb55cc6e.json`, classical row from `reports/classical_tune.json`, groups from `configs/heads.yaml`). The handcrafted branch is reduced by PCA to at most 64 components, so removing a group can leave the input size unchanged.

| Variant | Handcrafted groups | Branch balance | Input dim | C | Tune top-1 | Tune top-5 | Tune NLL |
|---|---|---|---|---|---|---|---|
| deep | none | per_feature | 1280 | 0.1 | 94.1% | 98.8% | 0.383 |
| hybrid | hog, lbp, hsv, geometry, flags | per_feature | 1344 | 0.02 | 94.2% | 98.8% | 0.462 |
| hybrid_equal_variance | hog, lbp, hsv, geometry, flags | equal_variance | 1344 | 0.02 | 92.5% | 99.0% | 0.499 |
| hybrid_no_hog | lbp, hsv, geometry, flags | per_feature | 1329 | 0.02 | 94.1% | 98.8% | 0.461 |
| hybrid_no_color_texture | hog, geometry, flags | per_feature | 1344 | 0.02 | 94.2% | 98.8% | 0.462 |
| hybrid_no_geometry | hog, lbp, hsv, flags | per_feature | 1344 | 0.02 | 94.2% | 98.8% | 0.462 |
| classical only (no learned features) | all | - | - | 0.005 | 18.7% | 38.2% | 3.966 |
| hybrid on the frozen ImageNet backbone (no fine-tuning, 224 px; `reports/heads_tune_imagenet_frozen.json`) | as hybrid | per_feature | 1344 | 0.1 | 64.9% | 89.8% | 1.371 |

Change in tune top-1 versus the full hybrid head: deep: -0.1 pt; hybrid_equal_variance: -1.7 pt; hybrid_no_hog: -0.1 pt; hybrid_no_color_texture: +0.0 pt; hybrid_no_geometry: +0.0 pt; released fine-tuned backbone vs frozen ImageNet backbone: +29.3 pt (hybrid; fine-tuned at 320 px vs frozen at 224 px, so this mixes fine-tuning and input resolution). Reading: the fine-tuned backbone gives by far the largest gain. Removing handcrafted feature groups (no_hog -0.1 pt, no_color_texture +0.0 pt, no_geometry +0.0 pt) moves tune top-1 by at most 0.1 pt, so with fine-tuned features the handcrafted branch adds little accuracy; the deep-only head is within 0.1 pt. Equal-variance branch weighting changes top-1 by -1.7 pt and raises tune NLL from 0.462 to 0.499. These are 815-image tune differences without a significance test.

## 8. Failure analysis

Where the production mode goes wrong (589 correctly localised test images with a wrong top-1, `reports/extra_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`, figure `reports/figures/confusions_test.png`): same make, other model: 319 (54.2%); different make: 177 (30.1%); badge-engineered twin: 48 (8.1%); same family, other model year: 45 (7.6%).

| True class | Predicted as | Times | Kind |
|---|---|---|---|
| Audi S5 Coupe 2012 | Audi A5 Coupe 2012 | 14 | same make, other model |
| Audi TT Hatchback 2011 | Audi TTS Coupe 2012 | 14 | same make, other model |
| Chevrolet Express Cargo Van 2007 | Chevrolet Express Van 2007 | 12 | same make, other model |
| BMW 6 Series Convertible 2007 | BMW M6 Convertible 2010 | 12 | same make, other model |
| Audi V8 Sedan 1994 | Audi 100 Sedan 1994 | 12 | same make, other model |
| Bentley Continental GT Coupe 2012 | Bentley Continental GT Coupe 2007 | 11 | same family, other model year |
| Chevrolet Express Van 2007 | Chevrolet Express Cargo Van 2007 | 11 | same make, other model |
| Dodge Caliber Wagon 2007 | Dodge Caliber Wagon 2012 | 11 | same family, other model year |
| Bugatti Veyron 16.4 Convertible 2009 | Bugatti Veyron 16.4 Coupe 2009 | 10 | same make, other model |
| Dodge Caliber Wagon 2012 | Dodge Caliber Wagon 2007 | 10 | same family, other model year |

Rule-selected individual cases:

Selected by fixed rules from the test predictions (`reports/analysis_test_ft_ckpt_98b1fb55cc6e_5d755416fd.json`). Image sizes from `artifacts/manifests/test_m1.csv`; duplicate lists from `reports/data_audit.json`.

1. **localisation failure (no box, or lowest IoU below 0.5)**: `test/00076.jpg`, true class *AM General Hummer SUV 2000*; no vehicle box selected.
   * Runtime top-1: none (no crop); annotation-crop top-1: *AM General Hummer SUV 2000* (p = 0.9999), correct.
   * HYPOTHESIS: localisation/crop caused the error (annotation crop is classified correctly)
   * Evidence: image 96x64 px, annotated car 74x48 px (58% of the image); not listed as a duplicate in the data audit; no vehicle box was selected; likely cause: the whole photo is a small thumbnail, so the car carries little detail at the detector's input size.
2. **close-category error (same make), highest confidence**: `test/05135.jpg`, true class *Acura TSX Sedan 2012*; selected box IoU 0.959, detector score 0.893.
   * Runtime top-1: *Acura TL Sedan 2012* (p = 0.9999); annotation-crop top-1: *Acura TL Sedan 2012* (p = 0.9999), still wrong.
   * HYPOTHESIS: classifier confusion persists on the annotation crop
   * Evidence: image 630x378 px, annotated car 387x268 px (44% of the image); true class not in the runtime top-5; prediction is a sibling model of the same make; not listed as a duplicate in the data audit; likely cause: classifier confusion, not localisation (the box is good and the annotation crop gives the same wrong answer); the model is very confident and wrong, typical for near-identical sibling classes.
3. **confident cross-make error, highest confidence**: `test/06675.jpg`, true class *Chevrolet Express Cargo Van 2007*; selected box IoU 0.980, detector score 0.887.
   * Runtime top-1: *GMC Savana Van 2012* (p = 0.9998); annotation-crop top-1: *GMC Savana Van 2012* (p = 0.9997), still wrong.
   * HYPOTHESIS: classifier confusion persists on the annotation crop
   * Evidence: image 500x374 px, annotated car 364x207 px (40% of the image); true class ranked 3 in the runtime top-5; prediction is a different make; not listed as a duplicate in the data audit; likely cause: classifier confusion, not localisation (the box is good and the annotation crop gives the same wrong answer); the model is very confident and wrong, typical for near-identical sibling classes; Chevrolet Express and GMC Savana are badge-engineered versions of the same vehicle, so the classes are visually almost identical.

