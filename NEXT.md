# NEXT: SEMEVIA handoff

## GPU PC run (2026-11-07, host LEGION) — in progress; this PC is the only one committing
* Machine: RTX 4050 Laptop GPU (6 GB), i7-14650HX, 15.7 GB RAM, **C: only (no E: drive)**, so the
  runbook's `E:\semevia` is `C:\semevia` here (data, venv, store, torch cache, tmp). Env: `C:\semevia\env.ps1`.
* Step 1 done: venv Python 3.11.9 with `requirements.lock` + **torch 2.14.1+cu126 / torchvision 0.29.1+cu126**
  (cu124's torch 2.6 conflicts with the locked sympy; runbook updated). Data: jhpohovey/StanfordCars at
  c742195 (8,144 train / 8,041 test). `prepare_data.py` reproduces the manifests (CSV bytes identical;
  `manifest_index.json` identical apart from CRLF line endings); `declare_subsets.py` leaves the tree clean.
  `pytest`: passes (data/bundle-dependent tests skipped before step 3).
* Step 1 commit: 1cdf976 (pushed).
* Step 2 done: `scripts/train.py` on CUDA, 361 s wall, `fine_tuning_requirement_met: true`, 6 last-stage
  epochs with `pretrained_changed` true for features.7/features.8. Evidence copied from the store:
  `reports/training/effb0_r1_summary.json` and `effb0_r1_train_log.jsonl`. Checkpoint (not in Git):
  `C:\semevia\store\runs\effb0_r1\best_finetuned.pt`, tag `ckpt_2339418979e1`.
* Next: step 3, rebuild caches/heads/calibration/references/bundle on that checkpoint.

Updated 2026-11-06 at the P0/P1 checkpoint and partial P2. Branch `semevia/implementation`.
Commit SHA and push status: see the bottom of this file (updated after each push).

## What works (verified)
* Data: verified manifests (train 6,514 / tune 815 / calibration 815 / test 8,041), hierarchy, audit.
* Detection: YOLO11n, evaluated on tune (`reports/detector_tune.json`).
* Classical features + proposals + caches; classical, deep, hybrid and ablation heads on the
  **development backbone (frozen ImageNet)** (`reports/heads_tune_imagenet_frozen.json`).
* Passport (conformal set, hierarchy, decision states), Evidence Challenge, metrics: fixture-tested.
* App: three modes, Passport with **development calibration** (`artifacts/development_calibration.json`),
  challenge, processing views, JSON export. Verified in the browser locally. 77/77 tests pass.
* Details: `docs/checkpoints/P3_P4_dev.md`. Design: `docs/design.md`. Diagrams: `docs/architecture.md`.

## Open blockers
1. **Fine-tuning run** on the training PC (`docs/training_handoff.md`). Required for compliance.
2. **Docker container**: build and run verified locally after two fixes (see `docs/checkpoints/P1.md`);
   the offline (`--network none`) run is still open. Image deleted to free C:; Docker's disk file is on C:.
   **Ask the user before any Docker build** (writes about 3 GB into C:\Users\SCM Testing\AppData\Local\Docker\wsl\disk\docker_data.vhdx).
3. **No public host** (`docs/hosting_lock.md`).
4. **Heads have no public storage**: clean builds are detection-only until they get a URL.
5. Negative (unsupported-image) suite not collected; reject threshold is development-only.
6. Reference thumbnails withheld: image rights undecided (identifiers shown instead).
7. `gh` CLI missing (PR later); team names/USNs unknown.

## When best_finetuned.pt arrives (copy to D:/semevia_work/store/runs/effb0_r1/)
```bash
CK=D:/semevia_work/store/runs/effb0_r1/best_finetuned.pt
python -c "import json;print(json.load(open('D:/semevia_work/store/runs/effb0_r1/summary.json'))['fine_tuning_requirement_met'])"
python scripts/cache_features.py --features learned --backbone $CK --partitions train tune
# set learned.backbone and calibration.yaml backbone to the printed ckpt_<sha12> tag, then:
python scripts/fit_heads.py --head learned
python scripts/calibrate.py --development --mode hybrid      # or deep, if it is clearly better
# extend promote_bundle.py for checkpoint backbones (export state_dict), then promote and run pytest
```
Every cache, head, calibration and the bundle must be regenerated (cache invalidation matrix).

## Environment (outside Git)
* Training venv: `D:/semevia_work/venv` (Python 3.11.9), lock in `requirements.lock`.
* Data root: `D:/semevia_work/raw/StanfordCars/stanford_cars` (git commit c7421951 of
  jhpohovey/StanfordCars). Set `SEMEVIA_DATA_ROOT` to it.
* Set `TMP`/`TEMP` to `D:\semevia_work\tmp` while C: is short of space.
* Release weights: `python -m src.release fetch` (downloads + verifies YOLO11n).

## Exact next commands
```bash
export SEMEVIA_DATA_ROOT=D:/semevia_work/raw/StanfordCars/stanford_cars
D:/semevia_work/venv/Scripts/python -m pytest
D:/semevia_work/venv/Scripts/python scripts/prepare_data.py --config configs/data.yaml   # reproduces manifests byte-for-byte
docker compose up --build    # after Docker storage is on D:
```

## Single next bounded task
**The GPU / large-storage PC takes over the heavy work: follow `docs/runbook_other_pc.md` step by step**
(it ends with a ready prompt for Claude Code on that PC). The laptop stops committing while that PC works.

Code completed for that run (fixture-tested, guarded): checkpoint promotion + `artifacts/release_config.json`,
`scripts/freeze.py` (refuses the development backbone), `scripts/calibrate.py --final`, freeze-fingerprint
guards in `scripts/evaluate.py` (calibration/test locked until the freeze), `scripts/analyze_results.py`
(viewpoint subgroups + rule-based failures), `scripts/evaluate_negatives.py`, negative manifest templates.

### Remaining for completion
| Item | Who / where |
|---|---|
| Fine-tuning run, rebuild bundle on it (runbook 2-3) | GPU PC |
| P5 freeze, final conformal calibration (runbook 5) | GPU PC |
| Official test run, stress test, analysis, eval file (runbook 6) | GPU PC |
| Publish model files as a GitHub Release, fill URLs (runbook 7) | GPU PC (needs repo write + `gh` or web) |
| Docker image, offline check, deployment and release checks (runbook 8-9) | GPU PC + chosen host |
| Choose an always-on public host and billing | Team |
| Collect the negative suite (`docs/negative_suite.md`) | Team (licensed images with provenance) |
| Review agent viewpoint labels (`reports/viewpoint_labels.csv`) | Team |
| Team names and USNs (README, eval file first line) | Team |
| Thumbnail rights decision for references | Team |
| Demo rehearsal (3 minutes) and viva preparation | Team |
| Pull request into `main` (remote has no `main` yet) | Team decision |

## File ownership
See `docs/ownership.md`.

## Commit / push record
(filled in below after each push)
- 2026-11-06 P0/P1 checkpoint: commits 6c265d5, 83a18f4, 15d44d3, b446a65 pushed to origin/semevia/implementation (push succeeded). No gate tag yet: P1 container run and public deploy are open.
- 2026-11-06 partial P2: commits 00a6e22, e275c11, 1a9e948 pushed to origin/semevia/implementation (push succeeded). Fine-tuning still pending on the training PC.
- 2026-11-06 development P3/P4: commits 27b4dec, ecd7c2c, 31aa668, e34f645, 6140788, a738562, 8e612b3 pushed (push succeeded). No gate tags: fine-tuning, container run and hosting open.
