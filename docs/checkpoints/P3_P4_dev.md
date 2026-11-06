# P3/P4 checkpoint on the development backbone

Date: 2026-11-06. Everything here runs on a **development** backbone (frozen ImageNet
EfficientNet-B0) because the mandatory fine-tuning run is waiting for the team's GPU PC. The
gates `P3_complete_core` and `P4_passport_complete` are **not** tagged: fine-tuning, Docker
container run, public deployment and the reference panel are still open.

## Done and verified

| Item | Evidence |
|---|---|
| Detector evaluated on tune (815) | `reports/detector_tune.json`: generic AP50 0.847, AP50:95 0.771, any-candidate recall 1.000, primary recall@0.5 0.993 at threshold 0.25 (4 misses), p50 140 ms |
| Learned + classical caches (train, tune) | hash-checked; learned cache 8 min on CPU |
| Matched heads, one grid, tune selection | `reports/heads_tune_imagenet_frozen.json` |
| Development calibration (tune only) | `artifacts/development_calibration.json`: T 0.784, k 735, q 0.9775, reject 0.25 (tune retention 0.929) |
| Passport logic | `src/passport/*`, 23 fixture tests (order statistic, ties, empty set, hierarchy, precedence, challenge arithmetic) |
| Metrics | `src/evaluation/metrics.py`, 9 fixture tests (ECE 0.13 example, IoU 0.5, rejection example, AP) |
| App: 3 modes, Passport, challenge, export | `app/`, API contract tests for all modes + challenge; parity test app vs offline caches |
| Browser check (local uvicorn) | Full flow on a test photo; phone width has no overflow |

Tune-split head results (annotation crops, development backbone):

| Head | Top-1 | Top-5 | NLL |
|---|---|---|---|
| classical (handcrafted only) | 0.1865 | 0.3816 | 3.966 |
| deep (learned only) | 0.6429 | 0.9006 | 1.370 |
| hybrid (selected, per-feature) | 0.6491 | 0.8982 | 1.371 |
| hybrid_equal_variance (r1) | 0.5411 | 0.8245 | 1.766 |
| hybrid without HOG | 0.6454 | 0.8933 | 1.379 |
| hybrid without colour + texture | 0.6515 | 0.8994 | 1.369 |
| hybrid without geometry | 0.6503 | 0.8994 | 1.369 |

Reading: with this backbone the handcrafted branch adds no measurable gain (differences of
about 5 images out of 815). The best C for every learned head is the top of the fixed grid.
These are selection-set numbers and are optimistic; test numbers come in P6.

Production Passport mode: **hybrid** (marginally best tune top-1, the PS-required integration,
~20 ms extra). Deep and classical are comparison modes without Passport calibration.

## Observed example (illustrative, not a metric)

A BMW M5 Sedan 2010 test photo: best guess Mitsubishi Lancer Sedan 2012 (wrong), BMW M5 second.
The Passport set had 6 categories across 3 makes and answered "insufficient evidence" instead of
endorsing the wrong guess. The challenge pair (Lancer vs M5) had an original margin of 0.35 logits,
and blurring any region reduced it.

## Open
1. Fine-tuned backbone (training PC), then rerun caches, heads, calibration, promote, tests.
2. Docker container run (disk image move) and public hosting.
3. Reference panel (training-only index) and rights decision for thumbnails.
4. Negative suite for the reject threshold.
5. Public storage for heads so clean builds include recognition.
