# SEMEVIA

**Vehicle identity through evidence.** SEMEVIA recognises the make, model and year of a car in a photo
(196 Stanford Cars categories) and shows how much of that answer the evidence actually supports.

**Status (2026-11-07):** the released iteration is **r2** (run `effb0_r2`, full-network fine-tuning,
[configs/train_v2.yaml](configs/train_v2.yaml); bundle `ft_ckpt_4f3584f597d5_9662c2e703`, model files in
GitHub Release `model-v2`). It was frozen ([artifacts/production_freeze.json](artifacts/production_freeze.json),
reject threshold chosen with the 40 tune negatives), evaluated once on the official test split, and is the
version served at the live URL. The earlier iteration **r1** (`effb0_r1`, last-stage fine-tuning, bundle
`ft_ckpt_2339418979e1_0920511710`, GitHub Release `model-v1`) is superseded: its freeze, calibration and
config are archived in [artifacts/history/r1/](artifacts/history/r1/) and its test result is kept in the
evaluation file under "Model iterations". r2 was designed after r1's test result was known; the decision that
r2 replaces r1 whatever its result was recorded in [NEXT.md](NEXT.md) ("Iteration r2 decision record") before
r2's test run. The test split was evaluated once per iteration (twice in total).

## Live application

**Live application URL:** <https://readings-agency-respond-bacteria.trycloudflare.com>

Served from the team laptop through a Cloudflare quick tunnel (see [docs/hosting_lock.md](docs/hosting_lock.md)).
Checked through the public URL on 2026-11-07: page, readiness, 20 fresh uploads, 4 simultaneous clients and
six bad-input cases ([reports/release_acceptance.json](reports/release_acceptance.json)). The URL works only
while that laptop, Docker and the tunnel are running. If `cloudflared` restarts, the address changes; this
README and the submission are then updated and the check is rerun.

## What it does

SEMEVIA is our solution to PS5, *Intelligent Fine-Grained Vehicle Recognition and Analysis*, on the Stanford
Cars dataset (196 categories, 16,185 images: 8,144 train and 8,041 test). For one uploaded photo it:

1. **Detects** the vehicle with YOLO11n (COCO car, bus and truck mapped to "vehicle") and selects the primary
   target: the largest box with detector score >= 0.25. In Classical mode the region comes from a classical
   contour proposal instead of YOLO.
2. **Classifies** the crop in one of three modes: **Classical** (1816 handcrafted features: HOG, LBP, HSV,
   geometry, flags), **Deep** (the 1280-d fine-tuned EfficientNet-B0 embedding) or **Hybrid** (the embedding
   plus the handcrafted features reduced by PCA to 64; the production mode). It shows the best guess and Top-5
   with a classification score kept separate from the detector score.
3. Issues an **Identity Evidence Passport**: decision state (one candidate, several remain, rejected,
   insufficient evidence, no vehicle), a calibrated conformal candidate set, and the most specific level the
   set supports (exact category, family, make). Only the production mode (Hybrid) uses the final
   calibration; Deep and Classical are shown as "comparison mode".
4. Runs the **Evidence Challenge** on request: blurs each of six crop regions in turn and shows how the margin
   between the two leading categories (a fixed A/B pair) changes, with the modified crop and **Restore original**.
5. Lists **nearest training references** from a training-only index (identifiers, labels and similarity; no
   thumbnails, see [Ethics and licences](#ethics-and-licences)).
6. Shows intermediate processing (crop, grayscale, Canny edges, contours, geometric measurements) and offers
   **Download Passport JSON** (hashes, boxes, scores and versions; no image pixels).

Bad inputs (not an image, over 10 MB, over 25 megapixels, empty file) get a typed error with a recovery
action. Without the recognition bundle the app serves detection only and says so.

```mermaid
flowchart LR
    U["POST /predict<br/>app/main.py"] --> D["decode_image<br/>src/vision/decode.py"]
    D --> Y["VehicleDetector + select_target<br/>src/models/detector.py"]
    D -. Classical mode .-> P["propose<br/>src/vision/proposals.py"]
    Y --> C["expand_box + crop_bounds<br/>src/vision/coordinates.py"]
    P -.-> C
    C --> PR["Predictor: learned + classical<br/>score_features<br/>src/models/inference.py"]
    PR --> PP["make_passport<br/>src/passport/decision.py"]
    PR --> RF["references.query<br/>src/passport/references.py"]
    PP --> R["JSON response"]
    RF --> R
    R --> CH["POST /challenge<br/>run_challenge<br/>src/passport/challenge.py"]
```

Full diagrams (runtime, training and evaluation, partitions): [docs/architecture.md](docs/architecture.md).

## Features

| Feature | State |
|---|---|
| Upload, validation, typed errors | Implemented, tested |
| Vehicle detection (YOLO11n, COCO car/bus/truck) with primary-target rule | Implemented, evaluated |
| Verified Stanford Cars manifests and class hierarchy | Implemented, tested |
| Classical features (edges, contours, HOG, LBP, HSV, geometry), classical ROI proposals | Implemented, tested, shown in the app |
| Classical, Deep and Hybrid modes with Top-5 and separate scores | Implemented on the fine-tuned backbone; evaluated on test |
| EfficientNet-B0 fine-tuning | r2 released (`effb0_r2`, full network, [summary](reports/training/effb0_r2_summary.json)); r1 superseded (`effb0_r1`, last stage, [summary](reports/training/effb0_r1_summary.json)) |
| Identity Evidence Passport (candidate set, specificity, decision states, JSON export) | Implemented; final calibration from the untouched calibration split |
| Evidence Challenge (six blurred regions, fixed A/B margin) | Implemented, tested |
| Nearest training references (588-entry training-only index: 3 per class) | Implemented; identifiers, labels and similarity only (no thumbnails, licence decision) |
| Optimization: ONNX FP32 encoder, INT8 encoder attempts, INT8 linear head | Measured; every INT8 encoder failed our accuracy policy; the release keeps the PyTorch FP32 encoder |
| End-to-end evaluator (Top-1/5 with IoU gating, IoU, AP, Passport, calibration) | Run on tune, calibration and (once per iteration, after its freeze) test |
| Robustness: 6 fixed stress conditions on a predeclared 392-image test subset | Run for r1 and for r2 |
| Viewpoint subgroups: 60 viewpoint-labelled test photos | Labelled by an agent before any test result; blind second pass by agents ([reports/viewpoint_label_check.json](reports/viewpoint_label_check.json)); not human-reviewed |
| Negative (unsupported-input) suite: 80 images, 40 tune + 40 test, 10 per bucket | Collected ([artifacts/negative_manifests/](artifacts/negative_manifests/)); used in the r2 freeze and test run |
| Evaluation file [eval_P5_T35.md](eval_P5_T35.md) | Generated from reports by `scripts/report.py` |
| Docker image, offline start, public deployment | Done for r2; checks re-run on the r2 bundle ([reports/container_check.json](reports/container_check.json), [reports/release_acceptance.json](reports/release_acceptance.json)) |

Supported vehicle categories: the 196 Stanford Cars classes (49 makes, model years 1991-2012), listed in
[artifacts/class_metadata.json](artifacts/class_metadata.json).

## Results

All results are in **[eval_P5_T35.md](eval_P5_T35.md)**, in the PS5 metric order: Top-1/Top-5, IoU, mAP,
model size and latency, robustness, and confidence/threshold analysis, followed by ablations and three
rule-selected failure cases. The file is generated by `scripts/report.py` from the report files below; no
number in it or in this README is typed by hand. It is generated for r2 and lists the superseded r1 test
result under "Model iterations".

<!-- RESULTS:START -->
Generated by `scripts/report.py` from `reports/e2e_test_ft_ckpt_4f3584f597d5_9662c2e703.json` (official test split, 8041 images, bundle `ft_ckpt_4f3584f597d5_9662c2e703`). Full results: [eval_P5_T35.md](eval_P5_T35.md).

| Mode | End-to-end top-1 | End-to-end top-5 | Box recall @ IoU 0.5 | 196-class AP50 |
|---|---|---|---|---|
| classical | 6.3% | 14.8% | 63.3% | 0.027 |
| deep | 89.6% | 97.3% | 99.1% | 0.871 |
| hybrid | 89.6% | 97.2% | 99.1% | 0.870 |

Also: unsupported-input false acceptance 20.0% (8/40 held-out negatives, `reports/negatives_test_ft_ckpt_4f3584f597d5_9662c2e703.json`); public-URL latency p50 502 ms / p95 871 ms (`reports/release_acceptance.json`).
<!-- RESULTS:END -->

Report files. File names carry the bundle id (test runs use the released bundle
`ft_ckpt_4f3584f597d5_9662c2e703`; tune-side freeze inputs use the bundle at freeze
`ft_ckpt_4f3584f597d5_c13883508a`) or the backbone tag `ckpt_4f3584f597d5`; the evaluation file always names
the files it was generated from. The links below are the current r2 release.

| What | File pattern | r2 file |
|---|---|---|
| End-to-end test evaluation (all three modes, Passport, calibration) | `reports/e2e_test_<bundle>.json` | [e2e_test_ft_ckpt_4f3584f597d5_9662c2e703.json](reports/e2e_test_ft_ckpt_4f3584f597d5_9662c2e703.json) |
| Robustness on the 392-image test stress subset | `reports/robustness_stress_test_<bundle>.md` / `.json` | [robustness_stress_test_ft_ckpt_4f3584f597d5_9662c2e703.md](reports/robustness_stress_test_ft_ckpt_4f3584f597d5_9662c2e703.md) |
| Viewpoint subgroups and failure cases | `reports/analysis_test_<bundle>.json` | [analysis_test_ft_ckpt_4f3584f597d5_9662c2e703.json](reports/analysis_test_ft_ckpt_4f3584f597d5_9662c2e703.json) |
| End-to-end tune and calibration runs (freeze inputs) | `reports/e2e_<partition>_<bundle>.json` | [e2e_tune_ft_ckpt_4f3584f597d5_c13883508a.json](reports/e2e_tune_ft_ckpt_4f3584f597d5_c13883508a.json), [e2e_calibration_ft_ckpt_4f3584f597d5_c13883508a.json](reports/e2e_calibration_ft_ckpt_4f3584f597d5_c13883508a.json) |
| Head selection and ablations (tune) | `reports/heads_tune_<tag>.json` | [heads_tune_ckpt_4f3584f597d5.json](reports/heads_tune_ckpt_4f3584f597d5.json) |
| Nearest-reference check (tune) | `reports/references_tune_<tag>.json` | [references_tune_ckpt_4f3584f597d5.json](reports/references_tune_ckpt_4f3584f597d5.json) |
| Optimization (ONNX FP32 / INT8 encoder; INT8 linear head) | `reports/optimization_<tag>.json`, `reports/optimization_head_hybrid_<tag>.json` | [optimization_ckpt_4f3584f597d5.json](reports/optimization_ckpt_4f3584f597d5.json), [optimization_head_hybrid_ckpt_4f3584f597d5.json](reports/optimization_head_hybrid_ckpt_4f3584f597d5.json) |
| Fine-tuning run | `reports/training/<run>_summary.json` | [effb0_r2_summary.json](reports/training/effb0_r2_summary.json) |
| Negative suite (unsupported inputs): tune (used at the freeze) and held-out test | `reports/negatives_<partition>_<bundle>.json` | [negatives_tune_ft_ckpt_4f3584f597d5_c13883508a.json](reports/negatives_tune_ft_ckpt_4f3584f597d5_c13883508a.json), [negatives_test_ft_ckpt_4f3584f597d5_9662c2e703.json](reports/negatives_test_ft_ckpt_4f3584f597d5_9662c2e703.json) |
| Viewpoint label check | `reports/viewpoint_label_check.json` | [viewpoint_label_check.json](reports/viewpoint_label_check.json) |
| Container and public-URL release checks (re-run on r2) | `reports/container_check.json`, `reports/release_acceptance.json` | [container_check.json](reports/container_check.json), [release_acceptance.json](reports/release_acceptance.json) |
| Superseded r1 test evaluation (history) | `reports/*_ft_ckpt_2339418979e1_0920511710.json` | [e2e_test_ft_ckpt_2339418979e1_0920511710.json](reports/e2e_test_ft_ckpt_2339418979e1_0920511710.json); r1 freeze, calibration and config in [artifacts/history/r1/](artifacts/history/r1/) |

Note on mode choice: the production mode (Hybrid) was fixed on the tuning split before each iteration's test
run. For r2, Deep and Hybrid were level on tune, and the project rule keeps Hybrid unless Deep is clearly
better on tune; this was recorded in [NEXT.md](NEXT.md) before r2's test run. The Deep/Hybrid comparison on
test is in [eval_P5_T35.md](eval_P5_T35.md) (section 1). We did not switch modes after seeing test results.

## Quick start (Docker)

Prerequisites: Docker with about 4 GB of free disk for the image.

```bash
docker compose up --build
```

Then open <http://localhost:8000>. The build downloads the public weights listed in
[artifacts/release_manifest.json](artifacts/release_manifest.json) (YOLO11n from Ultralytics, plus the
fine-tuned backbone, heads and reference index from GitHub Release `model-v2`) and verifies their SHA256
(`python -m src.release fetch` and `verify` in the [Dockerfile](Dockerfile)); the running container makes
no downloads. A fresh clone built this way serves all three modes, also with `--network none`
([reports/container_check.json](reports/container_check.json)). `compose.yaml` limits the container to
2 CPUs and 4 GB. Readiness: `GET /readyz`.

Without Docker (development; this exact install-from-lock sequence has not been tested from a clean virtual
environment, while `uvicorn` and the tests were run in our pinned venv):

```bash
python -m venv venv && venv/Scripts/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
python -m src.release fetch
python -m uvicorn app.main:app --port 8000
```

Copy [.env.example](.env.example) to `.env` for local paths. Inference needs no secrets.

## Reproduce the results

Training and evaluation need the Stanford Cars data; inference does not. The exact commands (Windows
PowerShell, with environment setup and the checks to run after each step) are in
**[docs/runbook_other_pc.md](docs/runbook_other_pc.md)**. GPU fine-tuning details:
[docs/training_handoff.md](docs/training_handoff.md). The order is:

1. **Data.** Get the original layout (`git clone --depth 1 https://github.com/jhpohovey/StanfordCars.git`),
   set `SEMEVIA_DATA_ROOT` and `SEMEVIA_ARTIFACT_STORE`, then run `scripts/prepare_data.py --config
   configs/data.yaml` and `scripts/declare_subsets.py`. It checks the test annotations against the TFDS
   published SHA256, decodes all 16,185 images, validates boxes, finds duplicates and must reproduce
   [artifacts/manifests/](artifacts/manifests/) byte for byte (train 6,514 / tune 815 / calibration 815 /
   test 8,041). Findings, including 37 duplicate image groups with conflicting labels:
   [reports/data_audit.json](reports/data_audit.json), [docs/checkpoints/P1.md](docs/checkpoints/P1.md).
   Then `python -m pytest`.
2. **Fine-tune.** `scripts/train.py --config configs/train_v2.yaml --resume auto` (r2, the release) or
   `--config configs/train.yaml` (superseded r1). Rerunning the same command resumes.
3. **Features and heads.** `scripts/cache_features.py --features classical --partitions train tune`, then
   `scripts/cache_features.py --features learned --backbone <checkpoint> --partitions train tune` (prints the
   backbone tag `ckpt_<sha12>`), `scripts/fit_heads.py --head classical`, `scripts/fit_heads.py --head
   learned --backbone <tag>`.
4. **Development calibration, references, bundle.** `scripts/calibrate.py --development --mode hybrid
   --backbone <tag>`, `scripts/build_references.py --backbone <tag>`, `scripts/promote_bundle.py --backbone
   <checkpoint>`; optimization evidence with `scripts/optimize.py --backbone <tag>` and
   `scripts/optimize_head.py --mode hybrid --backbone <tag>`; detector check with
   `scripts/evaluate_detector.py --partition tune`.
5. **Negative suite (tune).** `scripts/evaluate_negatives.py --partition tune` with `SEMEVIA_NEG_ROOT` set
   (see [docs/negative_suite.md](docs/negative_suite.md)).
6. **Freeze and final calibration.** `scripts/evaluate.py --partition tune`, `scripts/freeze.py --mode hybrid
   --negatives reports/negatives_tune_<bundle>.json`, `scripts/evaluate.py --partition calibration`,
   `scripts/calibrate.py --final`, `scripts/promote_bundle.py --backbone <checkpoint> --calibration
   artifacts/calibration_final.json`.
   After the freeze nothing in the model, crop, features, runtime or thresholds changes.
7. **Single test run.** `scripts/evaluate.py --partition test --confirm-frozen`, the seven stress runs
   (`--subset stress_test --condition clean|bright_0.5|bright_1.5|down_0.50|down_0.25|occl_0.15|occl_0.30`),
   `scripts/report_robustness.py --subset stress_test`, `scripts/analyze_results.py --partition test`,
   `scripts/evaluate_negatives.py --partition test`.
8. **Report.** `python scripts/report.py --team-id T35 --usns "1RUA24CSE0051,1RUA24CSE0097" --readme` writes
   [eval_P5_T35.md](eval_P5_T35.md) and refreshes the results block in this README.
9. **Release.** Upload the files that `promote_bundle.py` lists with `"url": null` (backbone, heads, reference
   index) to a GitHub Release and put each download URL into
   [artifacts/release_manifest.json](artifacts/release_manifest.json) (SHA256 values unchanged; without the
   URLs a clean build serves detection only). Then `docker compose build`, deploy, and run
   `scripts/check_container.py` and `scripts/check_release.py --url <public url>`.

## Repository layout

| Path | Contents |
|---|---|
| [app/main.py](app/main.py) | FastAPI service: `/predict`, `/challenge`, `/request-image`, `/benchmarks`, `/healthz`, `/readyz`; in-memory request store |
| [app/static/](app/static/) | Web UI (`index.html`, `app.js`, `style.css`) |
| [src/data/](src/data/) | Manifests, class metadata and hierarchy input, crops |
| [src/vision/](src/vision/) | Image decoding and limits, box coordinates, classical features, contour proposals |
| [src/models/](src/models/) | YOLO11n detector wrapper, EfficientNet-B0 backbone, training, heads, feature caches, ONNX export, `Predictor` |
| [src/passport/](src/passport/) | Calibration, conformal sets, decision states, class hierarchy, Evidence Challenge, reference index |
| [src/evaluation/](src/evaluation/) | Metrics, stress conditions and subsets, subgroup and failure analysis, hardware description |
| [src/release.py](src/release.py) | Release manifest: fetch and SHA256-verify model assets |
| [scripts/](scripts/) | Pipeline steps, in the order listed under [Reproduce the results](#reproduce-the-results) |
| [configs/](configs/) | Data, features, heads, calibration, evaluation, runtime and training (`train.yaml` r1, `train_v2.yaml` r2) |
| [artifacts/](artifacts/) | Frozen manifests, negative-suite manifests, class metadata, release manifest and config, calibration, freeze record, archived earlier releases (`artifacts/history/r1/`) (model weights live in `artifacts/store/`, outside Git) |
| [reports/](reports/) | Every generated result file (JSON/Markdown) and data figures |
| [tests/](tests/) | Pytest suite (contracts, metrics, Passport, features, upload limits, bundle parity) |
| [docs/](docs/) | Documentation (see below) |
| [Dockerfile](Dockerfile), [compose.yaml](compose.yaml) | Inference image and local deployment |
| `requirements-infer.txt`, `requirements.lock` | Pinned dependencies (inference image, full environment) |
| [NEXT.md](NEXT.md) | Hand-off log: what was done, commits, next step |

## Documentation

| Document | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Diagrams of the runtime request flow and the training/evaluation flow, matching the code |
| [docs/design.md](docs/design.md) | Design document: goals, data, architecture, interfaces, decisions, Passport semantics, evaluation plan, limitations, ethics |
| [docs/timeline.md](docs/timeline.md) | What was built when during the 24 hours, including the r1 and r2 iterations |
| [docs/demo.md](docs/demo.md) | Three-minute demo script |
| [docs/viva_prep.md](docs/viva_prep.md) | Viva preparation: questions both members should be able to answer |
| [docs/runbook_other_pc.md](docs/runbook_other_pc.md) | Step-by-step reproduction on a GPU machine |
| [docs/hosting_lock.md](docs/hosting_lock.md) | Hosting choice, what runs where, tunnel limits |
| [docs/negative_suite.md](docs/negative_suite.md) | Unsupported-input suite: buckets, rules, how images are recorded |
| [docs/api_schema.json](docs/api_schema.json) | Response contract (`semevia.v1`) |
| [docs/training_handoff.md](docs/training_handoff.md) | Fine-tuning hand-off |
| [docs/checkpoints/](docs/checkpoints/) | Checkpoint notes P0-P5 |

## Limitations

* **Dataset scope.** Stanford Cars covers 196 car categories up to model year 2012, mostly clean,
  catalogue-style photos. Newer models, other vehicles and real street photos are outside its scope.
* **Label noise.** The dataset contains duplicate images with conflicting labels (37 groups, often adjacent
  model years), so some year distinctions are inherently ambiguous.
* **Generic detector.** YOLO11n is a COCO model; it finds cars, buses and trucks, not specific models. Stanford
  Cars annotates one car per image, so real background cars count as false positives in detection AP.
* **Rejection is fallible.** The reject rule is softmax-based. The negative suite (80 images) is a small stress
  set, not an open-world guarantee. The r2 reject threshold was chosen at the freeze with the 40 tune
  negatives and checked on the 40 held-out test negatives
  ([reports/negatives_test_ft_ckpt_4f3584f597d5_9662c2e703.json](reports/negatives_test_ft_ckpt_4f3584f597d5_9662c2e703.json));
  some negatives still pass it (see [eval_P5_T35.md](eval_P5_T35.md)). YOLO's motorcycle class is not mapped,
  so a motorcycle photo returns `no_vehicle`; a bus can pass the threshold, in which case the Passport decision
  is what prevents an identity claim. (r1 was frozen before the suite existed, so its threshold came from known
  tune images only.)
* **Viewpoint labels are not human-reviewed.** The 60 viewpoint labels and the blind second pass were made by
  agents, not by people ([reports/viewpoint_label_check.json](reports/viewpoint_label_check.json)).
* **INT8 encoder.** Every INT8 version of the encoder failed our predeclared accuracy policy
  ([reports/optimization_ckpt_4f3584f597d5.json](reports/optimization_ckpt_4f3584f597d5.json)), so the release
  runs the FP32 encoder. An INT8 linear head was measured separately; it shrinks only a tiny part of the model.
* **No reference thumbnails.** References show identifiers, labels and similarity only (see below), which makes
  them less intuitive to check.
* **Hosting.** The live URL is a quick tunnel to the team laptop. It has no uptime guarantee and stops working
  when the laptop, Docker or `cloudflared` stops; a restart changes the address.
* **Evidence Challenge.** It measures how blurring a region changes the A/B margin; it does not name car parts.

## Ethics and licences

* **Stanford Cars** is licensed for non-commercial research use only. Its images are not redistributed by this
  repository or the container. For the same reason the app shows no reference thumbnails: references show
  identifiers, labels and similarity only.
* **YOLO11n and Ultralytics** are AGPL-3.0.
* **Negative-suite images** come under Creative Commons (CC BY, CC BY-SA, CC0) or public-domain licences. The
  source URL, author and licence of every image are recorded in
  [artifacts/negative_manifests/](artifacts/negative_manifests/); the images themselves are kept outside Git.
* **No user images are stored.** Uploads and derived images live only in an in-memory, request-scoped store
  that expires (`request_ttl_seconds` in [configs/runtime.yaml](configs/runtime.yaml)); they are never written
  to disk or logged. The JSON export contains hashes and coordinates, not pixels.
* The app identifies vehicle models, not people or number plates, and makes no ownership, insurance or legal
  claims. A Passport is not a certificate.
* Code in this repository: licence to be chosen by the team.

## Team and submission

Team ID **T35**, problem statement **PS5** (Computer Vision Hackathon 2026).
Repository: <https://github.com/DuhItzAniket/SEMEVIA> (branch `semevia/implementation`).

| Member | USN |
|---|---|
| Aniket Kumar | 1RUA24CSE0051 |
| C Lakshanaa | 1RUA24CSE0097 |
