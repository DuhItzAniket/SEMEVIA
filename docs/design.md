# SEMEVIA design document

Status: living document, updated at each phase checkpoint. Last update 2026-11-06 (P0-P4 in
progress). Implemented behaviour is described as implemented; planned work is labelled
*(planned)*. Numbers are never typed here by hand: they live in the linked `reports/` files.

## 1. Problem and goals

PS5 asks for a system that detects a vehicle in a real-world photo and identifies its make,
model and year among the 196 Stanford Cars categories, combining classical computer vision with
deep learning, and that handles similar categories, illumination, occlusion and viewpoint.

Goals:
1. Mandatory output: box, make/model/year, separate detector and classifier scores, Top-5.
2. All mandatory classical and deep components, a genuine hybrid, and comparisons/ablations.
3. Distinctive contribution: the **Identity Evidence Passport**, which states how specific an
   answer the evidence supports, and the **Evidence Challenge**, which lets a user test how the
   model's leading distinction reacts to hiding parts of the crop.
4. A public, always-on demo that answers a fresh upload in about ten seconds *(hosting planned)*.

Non-goals: plate OCR, tracking, VIN or build-date verification, vehicles outside the 196 classes.

## 2. Data

* Source: Stanford Cars original layout (mirror recorded in `docs/resource_lock.md`).
* Audit (`scripts/prepare_data.py`, `reports/data_audit.json`): all 16,185 images decode;
  annotations validated; labeled test annotations match the TFDS published SHA256.
* Partitions (seed 20261106): train 6,514 / tune 815 / calibration 815 (from the original train
  pool) and the official test 8,041. Keys are `split/filename` because basenames repeat.
* Known dataset issues: 39 byte-identical image groups, 37 with conflicting labels (mostly
  adjacent model years); 19 test images have exact copies in the train pool. The official test
  set is kept unchanged; a duplicate-excluded diagnostic will be reported beside it.
* Boxes: 1-based inclusive in the source, converted once to zero-based half-open floats.
  Dataset images are used in stored pixel orientation (EXIF ignored, verified); uploads apply EXIF.
* Class hierarchy: make (explicit list incl. multiword makes) -> family (name minus year; only
  year-only variants share a family) -> category. All 196 rows reviewed.

## 3. Architecture

See `docs/architecture.md` for diagrams matching the code. Key decisions:

| Decision | Choice | Reason |
|---|---|---|
| Detector | YOLO11n, COCO weights, car/bus/truck -> vehicle, frozen | Small, established; tune recall measured in `reports/detector_tune.json` |
| Target rule | Largest vehicle box with score >= 0.25 | Deterministic, never uses ground truth |
| Classical ROI | Canny + morphology + contour proposals | A fully classical localisation path for the Classical mode |
| Handcrafted features | HOG 1764, uniform LBP 10, HSV 32, geometry 8, flags 2 | Mandatory classical components; fixed dimensions; finite on blank input |
| CNN | EfficientNet-B0, ImageNet init, 196-way head; last stage fine-tuned | Compact; transfer learning; fine-tuning run on a GPU PC |
| Heads | Multinomial logistic regression on cached features | Cheap, inspectable, enables matched comparisons and ablations |
| Hybrid | Learned 1280 + PCA-64 handcrafted, per-feature standardised | Genuine fusion without a second backbone (see section 5 for the r1 -> r2 change) |
| Passport | Split conformal (LAC score), finite-sample order statistic | No extra model; population-level guarantee under exchangeability |
| Challenge | 2x3 grid, Gaussian blur per cell, same predictor | Observable sensitivity test of the deployed model |
| Serving | FastAPI + static HTML/JS, one container | One origin, few dependencies |

## 4. Interfaces

`docs/api_schema.json` (v1) fixes the response contract. Endpoints: `GET /healthz`,
`GET /readyz`, `POST /predict`, `POST /challenge`, `GET /benchmarks`,
`GET /request-image/{id}/{name}`. Statuses: ok, uncertain, no_vehicle, invalid_image, busy,
error. Passport decision states: exact_candidate, ambiguous, rejected, inadequate,
detection_failed, comparison_only, with the precedence in `src/passport/decision.py`.

## 5. Model development log (tuning data only)

* Classical head: `reports/classical_tune.json`.
* Deep and hybrid heads on the **development** backbone (frozen ImageNet features):
  `reports/heads_tune_imagenet_frozen.json`.
  * r1: the hybrid balanced branches by equal total variance and scored well below the matched
    deep-only head, because each of the 64 handcrafted dimensions then carried about 20x the
    weight of a learned dimension.
  * r2 (one deliberate change, recorded in `configs/heads.yaml`): per-feature standardisation.
    The r1 setting is kept as the variant `hybrid_equal_variance`.
  * Tuning numbers are optimistic for whatever configuration was selected on them; only the
    frozen test evaluation *(planned, P6)* is an unbiased estimate.
* Detector on tune: `reports/detector_tune.json`.

## 6. Passport and challenge semantics

* Probabilities: softmax of raw logits divided by a temperature fitted on tuning data.
* Candidate set: every class with 1 - p <= q, ties included, never truncated; an empty set stays
  empty (insufficient evidence). A missed detection counts as the full 196-label set.
* q: the k-th smallest calibration score, k = ceil((n+1)(1-alpha)), alpha = 0.10. The current
  q is a **development** value from the tuning split; the final q will come from the untouched
  calibration partition after the configuration freeze.
* The 90% target is a population-level property for supported categories under
  exchangeability. It is not the probability that one upload is correct, and it does not hold
  for unsupported vehicles or perturbed images.
* Rejection (max calibrated probability below a threshold) is a separate operational rule; it
  never removes labels from the set. A negative (unsupported-image) suite is still needed to tune
  it *(planned)*.
* Challenge: delta = original (zA - zB) minus the variant's (zA - zB), in raw logits, with A and
  B fixed from the original result. Positive means hiding that cell reduced A's lead. This is
  perturbation sensitivity, not part detection.

## 7. Evaluation plan

Implemented metric code (`src/evaluation/metrics.py`, fixture-tested): Top-1/Top-5,
end-to-end Top-k gated by runtime-box IoU >= 0.5, localization with named denominators,
COCO AP via pycocotools, count-weighted ECE, NLL, rejection contingency, set coverage and size.
Final populations *(planned)*: all 8,041 test images, 392-image stress subset with six fixed
corruptions, 60 viewpoint-labelled photos, held-out negatives.

## 8. Limitations and risks

* Development backbone is not fine-tuned; final numbers will change after the GPU run.
* Stanford Cars covers 196 categories up to 2012 and photos are mostly clean catalogue-style
  images; real-world street photos and newer models are out of distribution.
* Label noise (conflicting duplicates) caps achievable year accuracy for some pairs.
* Only one annotated car per image, so detection AP counts real background cars as false
  positives.
* Softmax-based rejection is a weak open-set detector.

## 9. Ethics and privacy

* Uploads are processed in memory, expire after 15 minutes and are never written to disk or
  logged; exports contain hashes and coordinates, not pixels.
* The app identifies vehicle models, not people or plates; it makes no ownership, insurance or
  legal claims, and every Passport states that it is not a certificate.
* Stanford Cars images are not redistributed. Reference thumbnails need a rights decision.

## 10. Recovery and reproducibility

`NEXT.md` holds the handoff state; `docs/training_handoff.md` the GPU run; caches and heads are
keyed by manifest, crop policy, feature version and backbone hashes and refuse mismatches.
