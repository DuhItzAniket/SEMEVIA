# SEMEVIA

**Vehicle identity through evidence.** Every identity needs evidence.

> **Status (2026-11-06): development build, P0-P4 partly done.** Detection, three recognition modes,
> the Passport and the Evidence Challenge run end to end locally on a **development model** (frozen
> ImageNet backbone, not yet fine-tuned) with **development calibration**. Fine-grained
> results from the final fine-tuned model and all test-set evaluation are **not available yet**.
> No test accuracy exists yet, and none is claimed.

**Live application URL:** not deployed yet. No public host has been provisioned
(see [docs/hosting_lock.md](docs/hosting_lock.md)).

## Overview

SEMEVIA is our solution to PS5, *Intelligent Fine-Grained Vehicle Recognition and Analysis*, on
the Stanford Cars dataset (196 categories, 16,185 images). The planned system detects a vehicle,
classifies its make, model and category year, and adds an **Identity Evidence Passport**: a
calibrated candidate set summarised at the most specific level the evidence supports (exact
category, family, make, or insufficient evidence), plus a six-region **Evidence Challenge** that
blurs one part of the crop at a time and shows how the margin between the two leading
categories changes.

| Feature | State |
|---|---|
| Upload, validation, typed errors | Implemented, tested |
| Vehicle detection (YOLO11n, COCO car/bus/truck) with primary-target rule | Implemented, evaluated on tune split |
| Verified Stanford Cars manifests and class hierarchy | Implemented, tested |
| Classical features (edges, contours, HOG, LBP, HSV, geometry), classical ROI proposals | Implemented, tested, shown in the app |
| Classical, Deep and Hybrid modes with Top-5 and separate scores | Implemented on the development backbone |
| EfficientNet-B0 fine-tuning | Script ready and smoke-tested; real run pending on the team's GPU PC |
| Identity Evidence Passport (candidate set, specificity, decision states, JSON export) | Implemented; development calibration only |
| Evidence Challenge (six blurred regions, fixed A/B margin) | Implemented, tested |
| Nearest training references | Planned |
| INT8 / ONNX optimization | Planned (P5) |
| Test-set evaluation (Top-1/5, IoU, mAP, efficiency, robustness, confidence) | Metric code tested; runs planned (P6) |

Supported vehicle categories: the 196 Stanford Cars classes (49 makes, model years 1991-2012),
listed in [artifacts/class_metadata.json](artifacts/class_metadata.json).

## Quick start (inference)

Prerequisites: Docker with about 4 GB of free disk for the image.

```bash
docker compose up --build
```

Then open http://localhost:8000. The build downloads the public weights listed in
[artifacts/release_manifest.json](artifacts/release_manifest.json) (YOLO11n and the ImageNet
EfficientNet-B0) and verifies their SHA256; the running container makes no downloads. The trained
heads are not published yet, so a clean checkout currently builds a **detection-only** app; with the
heads present in `artifacts/store/release/heads/` (created by `scripts/promote_bundle.py`) all
modes are served. **Verification status:** the image built successfully
on 2026-11-06; starting it on our machine was blocked by a full system drive, so the
container run itself is not yet verified.

Without Docker (development; this exact install-from-lock sequence has not yet been tested
from a clean virtual environment, while `uvicorn` and the tests were run in our pinned venv):

```bash
python -m venv venv && venv/Scripts/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
python -m src.release fetch
python -m uvicorn app.main:app --port 8000
```

## Usage (current build)

Choose or take a photo, pick a model (Hybrid by default) and press **Analyze**. The page shows:

1. The image with detected boxes; the selected vehicle (largest car/bus/truck with detector score
   >= 0.25) is drawn thick. In Classical mode the region comes from the contour proposal instead.
2. The best guess and Top-5 (make, model, year) with the classification score, separate from the
   detector score.
3. The **Passport**: decision state (one candidate, several remain, rejected, insufficient
   evidence, no vehicle), the supported specificity (category, family, make) and the full
   candidate set. Only the Hybrid production mode has (development) calibration; Deep and
   Classical show "comparison mode".
4. **Run six-region challenge**: blurs each of six crop regions in turn and shows the change in the
   margin between the two leading categories, with the modified crop for any selected region and
   **Restore original**.
5. Intermediate processing: crop, grayscale, Canny edges, contours and geometric measurements.
6. **Download Passport JSON** (hashes, boxes, scores and versions; no image pixels).

Errors (not an image, over 10 MB, over 25 megapixels, empty file) show a message with a recovery
action. The recognition parts need the recognition bundle (see below); without it the app serves
detection only and says so.

## Reproduction (data)

Training and evaluation need the Stanford Cars data; inference does not.

```bash
git clone --depth 1 https://github.com/jhpohovey/StanfordCars.git   # original layout mirror
export SEMEVIA_DATA_ROOT=<path>/StanfordCars/stanford_cars
python scripts/prepare_data.py --config configs/data.yaml
python -m pytest
```

Classical baseline (CPU, about one minute in total on a 4-core laptop):

```bash
export SEMEVIA_ARTIFACT_STORE=<store dir>
python scripts/cache_features.py --features classical --partitions train tune
python scripts/fit_heads.py --head classical
```

Development deep/hybrid heads, calibration and bundle (CPU, about 20 minutes):

```bash
python scripts/cache_features.py --features learned --backbone imagenet_frozen --partitions train tune
python scripts/fit_heads.py --head learned
python scripts/calibrate.py --development --mode hybrid
python scripts/promote_bundle.py --backbone imagenet_frozen
python scripts/evaluate_detector.py --partition tune
```

Deep model training (GPU recommended): see [docs/training_handoff.md](docs/training_handoff.md).

`prepare_data.py` checks the labeled test annotations against the TFDS published SHA256,
decodes all 16,185 images, validates boxes, finds duplicates and writes the split manifests
(reproduced byte-for-byte in our runs). Findings, including 37 duplicate image groups with
conflicting labels, are in [reports/data_audit.json](reports/data_audit.json) and
[docs/checkpoints/P1.md](docs/checkpoints/P1.md).

## Architecture and design

See [docs/architecture.md](docs/architecture.md) (diagrams matching the code) and
[docs/design.md](docs/design.md) (decisions, Passport semantics, limitations, ethics).
Contract: [docs/api_schema.json](docs/api_schema.json).

## Results

No test-set results yet. Development diagnostics so far (every number from a saved report):

| What | Population | Top-1 | Top-5 | Source |
|---|---|---|---|---|
| Classical head (handcrafted features only) | Tune, 815 images, annotation crops | 0.1865 | 0.3816 | [reports/classical_tune.json](reports/classical_tune.json) |
| Deep head, development backbone (frozen ImageNet) | same | 0.6429 | 0.9006 | [reports/heads_tune_imagenet_frozen.json](reports/heads_tune_imagenet_frozen.json) |
| Hybrid head, development backbone | same | 0.6491 | 0.8982 | same file (includes ablations) |

Detector on the same 815 tune images: generic vehicle AP50 0.847, AP50:95 0.771; primary-target
recall at IoU 0.5 is 0.993 ([reports/detector_tune.json](reports/detector_tune.json)).

These are not end-to-end or test accuracy: they use ground-truth crops on the tuning split that
was also used to select the heads, and the backbone is not fine-tuned yet.

## Limitations (known so far)

* Stanford Cars covers 196 car categories up to model year 2012; newer or other vehicles are
  outside its scope.
* The dataset contains duplicate images with conflicting labels (often adjacent model years), so
  some year distinctions are inherently ambiguous.
* The detector is a generic COCO model; it finds cars, buses and trucks, not specific models.

## Configuration

Copy [.env.example](.env.example) to `.env` for local paths. Inference needs no secrets.

## Team and submission

Team member names and USNs: to be provided by the team (not invented here).

## Licences

Code in this repository: to be chosen by the team. YOLO11n and Ultralytics: AGPL-3.0. Stanford
Cars images are not redistributed by this repository or the container.
