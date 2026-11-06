# P0 resource lock

Recorded 2026-11-06. Values below were measured or observed on this date, not assumed.

## Compute

| Item | Observed |
|---|---|
| Development machine | Windows 11 Pro, Intel Core i5-1135G7 (4 cores / 8 threads), 39.8 GB RAM |
| GPU | **None usable for training** (Intel Iris Xe only; no NVIDIA driver / CUDA) |
| Disk | C: 4.1 GB free (too small); **D: 201.6 GB free, used for data, venv, checkpoints** |
| Docker | Docker Desktop 29.8.2 (WSL2 backend), 8 CPUs and ~19.4 GB visible to containers |
| Python | 3.11.9; training venv at `D:/semevia_work/venv` (outside the repository) |

### Pilot throughput (EfficientNet-B0, 224x224, batch 32, synthetic tensors, 4 torch threads)

| Measurement | Seconds |
|---|---|
| Eval forward, batch 32 | 1.22 |
| Train step, last stage + head unfrozen (`features.7`, `features.8`, `classifier`) | 1.79 |
| Train step, full network | 5.48 |

Extrapolated compute per epoch over ~6,514 training crops: about 6 min (last stage) and
19 min (full network), **excluding** JPEG decoding and augmentation. This is an extrapolation
from three timed steps, not a measured epoch. A real-image epoch timing replaces it in P2.

### Training-compute decision (user decision, 2026-11-06)

The user plans to run fine-tuning on another machine (same account) that is not free yet, and
asked to proceed with work that does not require it. Consequence:

* The mandatory "at least one completed last-stage fine-tuning epoch" stays **open** until a
  run completes, on that machine or (given the pilot numbers above) on this CPU if the user
  approves.
* `scripts/train.py` is written to be portable and resumable (`--resume auto`), with
  checkpoints written to `SEMEVIA_ARTIFACT_STORE`.
* Work that only needs frozen ImageNet features (classical head, development heads, Passport
  logic, UI) continues and is labelled **development / frozen backbone**; it does not close the
  fine-tuning requirement.

## Data source

| Item | Result |
|---|---|
| Original Stanford URLs (`ai.stanford.edu/~jkrause/...`) | Connection timeout (unreachable), as the plan anticipated |
| CVUT research mirror named in the plan (R20) | Not located by search; no URL available |
| Mirror used | `https://github.com/jhpohovey/StanfordCars` (git clone, depth 1), original layout: `cars_train/`, `cars_test/`, `devkit/*.mat`, `cars_test_annos_withlabels.mat` |
| Published checksums | TFDS `url_checksums/cars196.txt` (fetched 2026-11-06) |

**Deviation from plan:** the mirror holds extracted images, not the original `.tgz` archives,
so archive SHA256 values cannot be checked. Verification instead uses: SHA256 of
`cars_test_annos_withlabels.mat` against the TFDS published value, exact record counts
(8,144 / 8,041), one-to-one filename coverage between annotations and files, full decode of
every image, box bounds against decoded dimensions, and visual overlay inspection. Results are
in `reports/data_audit.json`.

## Dependencies

* `requirements.lock`: full `pip freeze` of the training environment.
* `requirements-infer.txt`: pinned subset for the inference container.
* Docker base: `python:3.11-slim@sha256:65bd3d9f1d7fa268dd1909973ba7eb4bdf41f6dcf2cfbe76ba9528b170f3fd79`.

## Licences and terms to respect

* Stanford Cars images: research dataset; images are not redistributed by this repository or
  the container. Reference thumbnails in the public UI need a permission decision (P4).
* Ultralytics YOLO11n weights and library: AGPL-3.0.
* Torchvision EfficientNet-B0 ImageNet weights: BSD-3 code licence; weights trained on ImageNet.

## Hosting

See `docs/hosting_lock.md`. Status: **not provisioned (release blocker)**.

## GitHub

* Remote `origin` = `https://github.com/DuhItzAniket/SEMEVIA` (reachable; repository was empty).
* Git author used for commits: existing local config (`C-LAKSHANAA`).
* `gh` CLI is not installed, so opening a pull request needs either `gh` or the web UI.
