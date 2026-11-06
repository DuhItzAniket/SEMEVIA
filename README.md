# SEMEVIA

**Vehicle identity through evidence.** Every identity needs evidence.

> **Status (2026-11-06): early development, phases P0-P2 (partial).** Vehicle *detection* works and a
> classical-feature baseline is trained; fine-grained
> make/model/year recognition in the app, the Identity Evidence Passport and all test-set
> evaluation results are **not implemented yet**. No test accuracy exists yet, and none is claimed.

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
| Vehicle detection (YOLO11n, COCO car/bus/truck) with primary-target rule | Implemented, tested locally |
| Verified Stanford Cars manifests and class hierarchy | Implemented, tested |
| Classical features (edges, contours, HOG, LBP, HSV, geometry), classical ROI proposals | Implemented, tested |
| Classical (handcrafted-only) 196-class head | Trained; tuning diagnostic only, not in the app yet |
| EfficientNet-B0 196-class classifier, fine-tuned | Training script ready and smoke-tested; real run pending on the team's GPU PC |
| Hybrid classical + deep head; three comparison modes | Planned (P3) |
| Passport, Evidence Challenge, references | Planned (P4) |
| INT8 / ONNX optimization | Planned (P5) |
| Evaluation (Top-1/5, IoU, mAP, efficiency, robustness, confidence) | Planned (P6) |

Supported vehicle categories: the 196 Stanford Cars classes (49 makes, model years 1991-2012),
listed in [artifacts/class_metadata.json](artifacts/class_metadata.json).

## Quick start (inference)

Prerequisites: Docker with about 4 GB of free disk for the image.

```bash
docker compose up --build
```

Then open http://localhost:8000. The build downloads the public YOLO11n weights listed in
[artifacts/release_manifest.json](artifacts/release_manifest.json) and verifies their SHA256;
the running container makes no downloads. **Verification status:** the image built successfully
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

Choose or take a photo and press **Analyze**. The page shows the image with every detected
vehicle box, the selected primary vehicle (largest car/bus/truck with detector score >= 0.25),
its crop, and timings. Errors (not an image, over 10 MB, over 25 megapixels, empty file) show a
message with a recovery action. "No supported vehicle target found" is shown when nothing is
detected.

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

Deep model training (GPU recommended): see [docs/training_handoff.md](docs/training_handoff.md).

`prepare_data.py` checks the labeled test annotations against the TFDS published SHA256,
decodes all 16,185 images, validates boxes, finds duplicates and writes the split manifests
(reproduced byte-for-byte in our runs). Findings, including 37 duplicate image groups with
conflicting labels, are in [reports/data_audit.json](reports/data_audit.json) and
[docs/checkpoints/P1.md](docs/checkpoints/P1.md).

## Architecture and design

Planned runtime flow: upload -> validate/decode -> YOLO11n detector -> primary box -> crop ->
classical features + EfficientNet-B0 features -> selected head -> calibrated scores, rejection
and Passport; Evidence Challenge on demand. Contracts: [docs/api_schema.json](docs/api_schema.json).
Architecture and design documents are written as components are built.

## Results

No test-set results yet. Development diagnostics so far (every number from a saved report):

| What | Population | Top-1 | Top-5 | Source |
|---|---|---|---|---|
| Classical head (handcrafted features only, logistic regression) | Tune partition, 815 images, annotation (oracle) crops | 0.1865 | 0.3816 | [reports/classical_tune.json](reports/classical_tune.json) |

This is not end-to-end accuracy and not test accuracy; it uses ground-truth crops on the
development tuning split.

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
