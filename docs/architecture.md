# SEMEVIA architecture

Status: reflects the code on branch `semevia/implementation` as of 2026-11-06. Components marked
*(planned)* do not exist yet.

## Runtime flow (one request)

```mermaid
flowchart TD
    U[Upload: POST /predict<br/>file + mode] --> D[decode_image<br/>src/vision/decode.py<br/>10 MB, 25 MP, EXIF, RGB, long edge 1600]
    D --> Y[VehicleDetector YOLO11n<br/>src/models/detector.py<br/>car/bus/truck -> vehicle]
    Y --> S[select_target<br/>largest box, score >= 0.25]
    D --> P[propose<br/>src/vision/proposals.py<br/>Canny + contours<br/>Classical mode only]
    S --> C[crop with 5% margin<br/>src/vision/coordinates.py]
    P --> C
    C --> F1[extract_classical<br/>src/vision/features.py<br/>HOG, LBP, HSV, geometry, flags = 1816]
    C --> F2[EfficientNet-B0 embedding 1280<br/>src/models/backbone.py]
    F1 --> H[Head per mode<br/>src/models/inference.py<br/>classical / deep / hybrid]
    F2 --> H
    H --> T[softmax with temperature<br/>production mode only]
    T --> PP[make_passport<br/>src/passport/decision.py<br/>reject rule + conformal set + hierarchy]
    PP --> R[JSON response v1 + request-scoped images<br/>app/main.py]
    R --> CH[POST /challenge<br/>src/passport/challenge.py<br/>6 blurred cells, same Predictor]
    R --> REF[References <i>(planned)</i>]
```

* One shared `Predictor.score_crops` computes every branch for recognition and for the six
  challenge variants, so the challenge tests exactly the deployed model.
* Readiness (`GET /readyz`) requires checksum-verified assets (`artifacts/release_manifest.json`)
  and a real warmup. Optional recognition assets that are missing disable recognition and the
  responses say so; the detector alone keeps working.
* Uploads and derived images live only in an in-memory store with a 15-minute expiry.

## Training and evaluation flow

```mermaid
flowchart LR
    A[Stanford Cars original layout] --> M[prepare_data.py<br/>audit + frozen manifests]
    M --> TR[train.py<br/>EfficientNet-B0 head then last stage<br/>runs on the GPU PC]
    M --> CF[cache_features.py<br/>classical + learned caches<br/>hash-checked]
    TR -. best_finetuned.pt .-> CF
    CF --> FH[fit_heads.py<br/>classical, deep, hybrid, ablations]
    FH --> CAL[calibrate.py<br/>temperature, reject threshold,<br/>conformal quantile]
    M --> DE[evaluate_detector.py<br/>AP, recall, IoU]
    CAL --> EV[evaluate.py on test <i>(planned)</i>]
    FH --> EV
```

## Data partitions

| Partition | Count | Used for |
|---|---|---|
| train | 6,514 | network training, scalers, PCA, heads, reference index |
| tune | 815 | model/threshold selection, temperature, development calibration |
| calibration | 815 | final conformal quantile only (untouched so far) |
| test | 8,041 | final evaluation only (untouched so far) |

## Current model status

| Component | State |
|---|---|
| Detector | YOLO11n COCO weights, frozen (final) |
| Backbone | **Development:** frozen ImageNet EfficientNet-B0. The fine-tuned checkpoint will replace it |
| Heads | classical, deep, hybrid logistic heads fitted on train, selected on tune |
| Calibration | **Development**, from tuning data |
