# NEXT: SEMEVIA handoff

Updated 2026-11-06 at the P0/P1 checkpoint and partial P2. Branch `semevia/implementation`.
Commit SHA and push status: see the bottom of this file (updated after each push).

## What works (verified)
* Data: verified manifests `artifacts/manifests/*_m1.csv` (train 6,514 / tune 815 /
  calibration 815 / test 8,041), class metadata with make/family hierarchy, audit report.
* Detection: real YOLO11n vehicle detection with bounded uploads and typed errors, through
  the FastAPI app and the browser UI (local uvicorn). 24/24 tests pass.
* Classical features (1816-dim) + classical proposals + versioned caches + classical head
  (tune top-1 0.1865 / top-5 0.3816 on annotation crops; development diagnostic). See
  `docs/checkpoints/P2.md`.
* `scripts/train.py` ready and smoke-tested; real fine-tuning run is waiting for the training PC
  (`docs/training_handoff.md`).
* Not implemented yet: deep/hybrid heads, the 196-class prediction in the app, Passport,
  challenge, references, optimization, evaluation. The UI says so explicitly.

## Open blockers
1. **C: drive full** (Docker Desktop disk image on C:). User is moving it to D:. Then:
   `docker compose up --build` and check `/readyz` + one upload (P1 container check).
2. **No public host** (`docs/hosting_lock.md`). Release blocker; user to decide.
3. **Mandatory fine-tuning epoch not run.** User plans another machine. CPU pilot suggests
   ~6 min/epoch of compute for last-stage fine-tuning here (extrapolated); needs user approval.
4. `gh` CLI not installed (needed later for the pull request, or use the GitHub web UI).
5. Team member names and USNs unknown (needed for README and `eval_P5_TeamID.md`).

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
Development deep/hybrid wiring that does not need fine-tuning: cache frozen ImageNet
EfficientNet-B0 embeddings (labelled development, non-compliant backbone), fit matched
deep-only and hybrid heads, and wire classical/deep/hybrid modes into the app with
"development model" labelling. Swap in `best_finetuned.pt` when the training PC run returns
and regenerate all caches/heads (cache invalidation matrix). Also: rebuild Docker after the
disk-image move and close the P1 container check.

## File ownership
See `docs/ownership.md`.

## Commit / push record
(filled in below after each push)
- 2026-11-06 P0/P1 checkpoint: commits 6c265d5, 83a18f4, 15d44d3, b446a65 pushed to origin/semevia/implementation (push succeeded). No gate tag yet: P1 container run and public deploy are open.
- 2026-11-06 partial P2: commits 00a6e22, e275c11, 1a9e948 pushed to origin/semevia/implementation (push succeeded). Fine-tuning still pending on the training PC.
