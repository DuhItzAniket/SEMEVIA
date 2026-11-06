# NEXT: SEMEVIA handoff

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
Done since last handoff: D: restore verified (dataset reproduces manifests; heads refit byte-identical);
stress/viewpoint subsets predeclared; six-condition robustness run on the tune subset; viewpoint labels
(agent, team review pending); ONNX encoder runtime (parity-gated, selectable); `scripts/report.py`
generates `eval_P5_TeamID.md` (final values pending, USNs needed).
Next without the GPU run: offline container run (after the user approves a rebuild); negative-suite
collection (needs licensed images: ask the team); README/design polish. Everything else waits for
`best_finetuned.pt` (see the section above) and the P5 freeze, then P6 test runs:
`python scripts/evaluate.py --partition test --resume --confirm-frozen` plus each stress condition.

## File ownership
See `docs/ownership.md`.

## Commit / push record
(filled in below after each push)
- 2026-11-06 P0/P1 checkpoint: commits 6c265d5, 83a18f4, 15d44d3, b446a65 pushed to origin/semevia/implementation (push succeeded). No gate tag yet: P1 container run and public deploy are open.
- 2026-11-06 partial P2: commits 00a6e22, e275c11, 1a9e948 pushed to origin/semevia/implementation (push succeeded). Fine-tuning still pending on the training PC.
- 2026-11-06 development P3/P4: commits 27b4dec, ecd7c2c, 31aa668, e34f645, 6140788, a738562, 8e612b3 pushed (push succeeded). No gate tags: fine-tuning, container run and hosting open.
