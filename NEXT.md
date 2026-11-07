# NEXT: SEMEVIA handoff

## Current status (2026-11-07 ~09:55 IST)
* **Released model: r3** (bundle `ft_ckpt_98b1fb55cc6e_5d755416fd`, GitHub Release `model-v3`), live at
  <https://readings-agency-respond-bacteria.trycloudflare.com>; results in `eval_P5_T35.md`. Zenodo DOI
  10.5281/zenodo.24026963 (v1.0.0).
* **Last step (commit cf2b69e):** Reset button fixed (its id shadowed `form.reset`), upload thumbnail, shorter
  UI text, in-app classical/deep/hybrid comparison panel (PS5 expected output 6), classical-mode whole-image
  fallback, and eval additions (dataset, preprocessing, classical-branch role, accuracy by car size and natural
  occlusion from saved test predictions). Tests pass; `scripts/check_release.py` passed on the public URL.
* **PS5 audit (7 agents, 2026-11-07):** 44 of 49 items met. Still partial: 4.2 illumination/viewpoint (only
  brightness x0.5/x1.5; viewpoint labels need team review), 4.6 optimisation is measured but not deployed
  (INT8 head / ONNX encoder), 4.3 occlusion (synthetic only, natural sample tiny).
* **Next:** team reviews `reports/viewpoint_labels.csv` (drop "review pending"), writes the report PDF (AI-content
  limit 30%; kit in `docs/report_kit.md`), records voice-over on the demo video, fills the submission form.
* **Keep running:** laptop awake and online, Docker Desktop, the `cloudflared` terminal tab.

## Iteration r3 decision record (written before r3 training starts; no r3 result exists)
* r3 = `configs/train_v3.yaml`: r2's recipe at 320 px input instead of 224 (same seed and data).
* **Release rule, fixed now:** r3 replaces r2 only if, on the **tune split**, its hybrid end-to-end top-1
  (`scripts/evaluate.py --partition tune`, same protocol) beats r2's 731/815 by **at least 1.0 percentage point
  (>= 740/815)** and every pipeline check passes (tests, verify, container). Hybrid stays the production mode.
* If the rule is not met: r2 stays the release, r3 is **not** evaluated on the test split, and its tune result is
  reported as a negative result. If met: r3 is frozen (tune negatives, as r2), then evaluated once on test, and
  all three iterations are reported. Decision taken by the team (team Hypermetropia) on 2026-11-07.
* **Rule outcome (recorded before any r3 test run):** r3 tune end-to-end hybrid top-1 = 759/815
  (`reports/e2e_tune_ft_ckpt_98b1fb55cc6e_f99e386191.json`) >= 740/815, so the rule is met. r3 was then frozen with
  the tune negatives (`artifacts/production_freeze.json`) and given its final calibration; r2's freeze is archived
  in `artifacts/history/r2/`. Next: the single r3 test run.

## Iteration r2 decision record (written before any r2 test result exists)
* Decided before r2's test run (this commit precedes it): iteration r2 (`configs/train_v2.yaml`, run effb0_r2)
  **replaces r1 as the release whatever its test result**, as long as its own pipeline checks pass. Its heads,
  calibration and freeze use tune/calibration data only; hybrid stays the production mode (tune tie with deep);
  the freeze uses the 40-photo tune negative suite; the test split and the 40 held-out negatives are evaluated
  once. Both r1 and r2 test results are reported in `eval_P5_T35.md`. r1's freeze and calibration are archived in
  `artifacts/history/r1/`.

## GPU PC run (2026-11-07, host LEGION) â€” in progress; this PC is the only one committing
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
* Step 2 commit: 42e7cbf (pushed).
* Step 3 done: caches, heads (`reports/heads_tune_ckpt_2339418979e1.json`), development calibration,
  references, bundle `ft_ckpt_2339418979e1_7aa5332188` promoted; `src.release verify` passes; pytest 91 passed,
  1 skipped (ONNX encoder parity: release keeps `encoder_runtime: torch`). Production mode stays **hybrid**
  (tune top-1 hybrid 0.7534 vs deep 0.7521: deep not clearly better). Optimization reports regenerated on
  this PC. Fixes made on the way: `test_bundle_parity.py` used a hard-coded `imagenet_frozen` tag; latency
  reports hard-coded the old laptop CPU name (now `src/evaluation/hardware.py`); the detector now runs on
  CPU explicitly (Ultralytics would pick the GPU here, so evaluation would not match the CPU deployment).
* Step 4 (negative suite): skipped, not collected by the team yet.
* Step 3 commit: 8ce22e7 (pushed).
* Step 5 done: tune evaluation (`reports/e2e_tune_ft_ckpt_2339418979e1_7aa5332188.json`), freeze
  (`artifacts/production_freeze.json`, hybrid, reject threshold from known tune images only because no
  negative suite exists), calibration evaluation, final conformal calibration
  (`artifacts/calibration_final.json`, 5 detection failures given the fallback score), bundle re-promoted
  as `ft_ckpt_2339418979e1_0920511710`; verify passes; pytest 91 passed, 1 skipped.
  **Frozen from here:** detector, backbone, heads, crop, features, runtime, thresholds.
* Step 5 commit: a05ee37 (pushed).
* Step 6 done: official test run once (8,041 images, `reports/e2e_test_ft_ckpt_2339418979e1_0920511710.json`),
  7 stress conditions on the 392-image test subset + robustness report, analysis (viewpoints, 3 failures).
  `scripts/report.py` rewritten (it never printed test values and hard-coded dev files); it now picks reports
  by bundle id / backbone tag and writes `eval_P5_T35.md` (old `eval_P5_TeamID.md` removed). A 6-agent check
  recomputed every number from the JSON: no wrong numbers; wording fixes applied. README updated (status,
  results pointer, team T35). On test, deep is +0.6 pt over hybrid top-1; hybrid stays (fixed on tune).
* Step 6 commit: 92ffbae (pushed).
* Step 8 (local) done: `docker compose build` (image 3.05 GB; Docker's disk is on C: on this PC), container ready
  with all three modes + final calibration; `--network none` container also reaches ready with recognition.
  Recorded by `scripts/check_container.py` in `reports/container_check.json`. Fix: `promote_bundle.py` stored the
  Windows path `artifacts\calibration_final.json`, which the Linux container could not open (recognition was
  silently off); paths are now POSIX, bundle id unchanged.
* Step 8 commit: b763e75 (pushed).
* Step 7 done (team approved): GitHub Release `model-v1` holds the backbone, heads and reference index; URLs in
  `artifacts/release_manifest.json` (each download re-checked against its SHA256).
* Step 7 commit: 900f483; clean-build check and Space builder: 2cebf30 (pushed).
* Step 9 done: Hugging Face refused Docker Spaces without PRO (402), so the team chose a Cloudflare quick tunnel
  from this laptop: **https://readings-agency-respond-bacteria.trycloudflare.com** (see `docs/hosting_lock.md`).
  `scripts/check_release.py` through the public URL passes (`reports/release_acceptance.json`: page, readiness,
  20 fresh uploads p50/p95, 4 concurrent clients, 6 bad inputs); phone width 375 px checked in a browser (no
  horizontal scroll, upload flow works). UI fixes: the banner and score type no longer say "development",
  `/benchmarks` serves the current bundle's test reports, static files carry a cache-busting version.
  The eval file now includes the deployed latency.
* Iteration r2 done (commits 4043f4e pre-registration, 38b1a62 freeze, cadcdc3 test, ae9a00b checks): full
  fine-tuning (`configs/train_v2.yaml`), freeze with the tune negatives, single test run, 40 held-out negatives,
  GitHub Release `model-v2`, redeployed behind the same URL. Results: `eval_P5_T35.md` (r1 listed under model
  iterations). Docs rewritten: `docs/architecture.md`, `docs/design.md`, `docs/timeline.md`, `docs/demo.md`,
  `docs/viva_prep.md`, `docs/ownership.md`, README (results block generated by `report.py --readme`).
* **Keep running for judging:** laptop on, plugged in, lid open, sleep off; Docker Desktop with the
  `semevia-semevia-1` container; the `cloudflared` terminal tab. If cloudflared restarts the URL changes:
  update README + submission and rerun `scripts/check_release.py --url <new>`.
* **Still needed from the team:** negative suite (`docs/negative_suite.md`), review of
  `reports/viewpoint_labels.csv`, thumbnail-rights decision, demo rehearsal and viva prep, PR into `main` if wanted.
  Deadline: target 12:00, hard 14:00 on 2026-11-07.

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
