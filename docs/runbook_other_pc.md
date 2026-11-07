# Runbook for the GPU / large-storage PC

This PC finishes everything that needs a GPU or a lot of disk: fine-tuning, the final model bundle,
the P5 freeze, final calibration, all official test runs (P6), the Docker image and deployment.
Every script below already exists and is tested; this runbook only runs them in order.

Rules:
* **One machine commits at a time.** While this PC works, the laptop does not push. Always
  `git pull` before starting and push after each step that produces committed files.
* Never edit `artifacts/manifests/*` or the configs listed as frozen. Never commit weights, datasets,
  caches or `.env` (`.gitignore` already excludes them).
* Stop and report if any check below fails; do not work around it.
* Numbers must come from the generated `reports/*.json`; never type results by hand.

Commands are Windows PowerShell. Replace `E:\semevia` with a folder on the large drive.

## 0. Prerequisites

* NVIDIA GPU with working `nvidia-smi`; Python 3.11; Git; Docker Desktop (step 9).
* About 60 GB free on the large drive (data 4 GB, environment 6 GB, store 5 GB, Docker 30+ GB).

## 1. Code, environment, data

```powershell
mkdir E:\semevia; cd E:\semevia
git clone https://github.com/DuhItzAniket/SEMEVIA; cd SEMEVIA; git checkout semevia/implementation
git clone --depth 1 https://github.com/jhpohovey/StanfordCars.git E:\semevia\StanfordCars
python -m venv E:\semevia\venv
E:\semevia\venv\Scripts\python -m pip install --upgrade pip
Get-Content requirements.lock | Select-String -NotMatch '^(torch|torchvision)==|^#' | Set-Content -Encoding ascii E:\semevia\req_no_torch.txt
E:\semevia\venv\Scripts\python -m pip install -r E:\semevia\req_no_torch.txt
# Install the CUDA build LAST: the lock's other packages otherwise pull the CPU torch from PyPI.
E:\semevia\venv\Scripts\python -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
E:\semevia\venv\Scripts\python -m pip check
E:\semevia\venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Why cu126 and not cu124: the newest cu124 build (torch 2.6) pins `sympy==1.13.1`, which conflicts with
`sympy==1.14.0` in `requirements.lock`, so pip silently replaces it with the CPU-only torch 2.15 from PyPI
(seen on 2026-11-07). cu126 needs an NVIDIA driver that reports CUDA 12.6 or newer in `nvidia-smi`.

`prepare_data.py` writes `manifest_index.json` with Windows line endings, so `git status` can list it
as modified even when its content is identical. `git diff artifacts\manifests` must be empty (the CSVs,
which are hashed, are written byte-exact).

Set these in every new PowerShell window (or a `.env` you do not commit):

```powershell
$env:SEMEVIA_DATA_ROOT = "E:\semevia\StanfordCars\stanford_cars"
$env:SEMEVIA_ARTIFACT_STORE = "E:\semevia\store"
$env:TORCH_HOME = "E:\semevia\torch_home"
$env:TMP = "E:\semevia\tmp"; $env:TEMP = "E:\semevia\tmp"; mkdir E:\semevia\tmp -Force | Out-Null
$PY = "E:\semevia\venv\Scripts\python"
```

Check the data reproduces the frozen manifests (must print nothing for `artifacts\manifests`):

```powershell
& $PY scripts\prepare_data.py --config configs\data.yaml
git status --short artifacts\manifests
git restore reports artifacts\class_metadata.json
& $PY scripts\declare_subsets.py; git status --short artifacts\manifests   # still nothing
```

## 2. Fine-tune (mandatory requirement)

This step as written produced iteration r1 (last-stage fine-tuning, 224 px, now superseded). The released model
is r3: run it with `configs\train_v3.yaml` instead (see step 3b).

```powershell
& $PY scripts\train.py --config configs\train.yaml --resume auto
```

Done when the last line shows `"fine_tuning_requirement_met": true` and stage-2 lines show
`"pretrained_changed": {"features.7": true, "features.8": true}`. Rerun the same command to resume.
`$CK = "E:\semevia\store\runs\effb0_r1\best_finetuned.pt"`

## 3. Rebuild everything on the fine-tuned backbone

```powershell
& $PY scripts\cache_features.py --features classical --partitions train tune
& $PY scripts\cache_features.py --features learned --backbone $CK --partitions train tune   # prints: backbone tag: ckpt_XXXXXXXXXXXX
$TAG = "ckpt_XXXXXXXXXXXX"     # paste the printed tag
& $PY scripts\fit_heads.py --head classical
& $PY scripts\fit_heads.py --head learned --backbone $TAG
& $PY scripts\calibrate.py --development --mode hybrid --backbone $TAG
& $PY scripts\build_references.py --backbone $TAG
& $PY scripts\promote_bundle.py --backbone $CK
& $PY -m pytest
```

Production mode: keep `hybrid` unless `reports\heads_tune_$TAG.json` shows `deep` clearly better
on tune (then use `--mode deep` in this step and in steps 5-6, and say so in the docs).

Optimization evidence on the final model (P5):

```powershell
& $PY scripts\optimize.py --backbone $TAG
& $PY scripts\optimize_head.py --mode hybrid --backbone $TAG
```

Keep `encoder_runtime: torch` for the release unless you rebuild caches and references with the ONNX
encoder; the measured in-app gain was small (for the released r3: `reports\optimization_ckpt_98b1fb55cc6e.json`).

Commit: `reports/*.json`, `artifacts/release_manifest.json`, `artifacts/release_config.json`,
`artifacts/development_calibration.json` (message e.g. `feat(model): promote fine-tuned backbone`).

## 3b. Iterations r2 and r3 (r3 is what was released)

Each iteration is the same steps 2-6 with its own training config. Before rebuilding, archive the current freeze
(`artifacts/history/r1/` before r2, `artifacts/history/r2/` before r3).

| Iteration | Config | Recipe | Status |
|---|---|---|---|
| r1 | `configs/train.yaml` | last-stage fine-tuning, 224 px | superseded (`artifacts/history/r1/`, release `model-v1`) |
| r2 | `configs/train_v2.yaml` | full-network fine-tuning, 224 px | superseded (`artifacts/history/r2/`, release `model-v2`) |
| r3 | `configs/train_v3.yaml` | r2's recipe at 320 px input, same seed | **released** (files to be published as `model-v3`) |

```powershell
& $PY scripts\train.py --config configs\train_v3.yaml --resume auto
$CK = "E:\semevia\store\runs\effb0_r3\best_finetuned.pt"     # then steps 3-6 with this $CK and its printed tag
```

The encoder input size is automatic: it travels with the backbone file (`backbone_input_size` and
`preprocess_version` in `src/models/backbone.py`), so `cache_features.py`, `promote_bundle.py`, `optimize.py` and
the app use 320 px for r3 with no extra flag; 224 px bundles stay byte-identical.

r3 was released only because it met the rule written in `NEXT.md` ("Iteration r3 decision record") before r3
training; the rule was checked on tune and the outcome committed before any r3 test run. Had it failed, r3 would
never have touched the test split. Step 4 below uses the collected negative suite, and step 7 publishes the
iteration's release.

## 4. Negative suite (collected; used for the r2 and r3 freezes)

See `docs/negative_suite.md`. With `$env:SEMEVIA_NEG_ROOT` set:
`& $PY scripts\evaluate_negatives.py --partition tune`

## 5. P5 freeze and final calibration

```powershell
& $PY scripts\evaluate.py --partition tune --resume
& $PY scripts\freeze.py --mode hybrid --negatives reports\negatives_tune_<bundle>.json   # r2 and r3; r1 was frozen without --negatives
& $PY scripts\evaluate.py --partition calibration --resume
& $PY scripts\calibrate.py --final
& $PY scripts\promote_bundle.py --backbone $CK --calibration artifacts\calibration_final.json
& $PY -m pytest
```

After the freeze, do **not** change the detector, backbone, heads, crop, features, runtime or
thresholds. If something must change, delete `artifacts\production_freeze.json` and redo steps 3-5.
Commit `artifacts/production_freeze.json`, `artifacts/calibration_final.json` and the updated manifest/config.

## 6. Official test evaluation (P6)

```powershell
& $PY scripts\evaluate.py --partition test --resume --confirm-frozen
foreach ($c in "clean","bright_0.5","bright_1.5","down_0.50","down_0.25","occl_0.15","occl_0.30") {
  & $PY scripts\evaluate.py --subset stress_test --condition $c --resume --confirm-frozen }
& $PY scripts\report_robustness.py --subset stress_test
& $PY scripts\analyze_results.py --partition test
& $PY scripts\evaluate_negatives.py --partition test          # held-out negatives, once after the freeze
& $PY scripts\evaluate_oracle.py --partition test --confirm-frozen   # standard Stanford Cars protocol (ground-truth crop)
& $PY scripts\report.py --team-id T35 --usns "1RUA24CSE0051,1RUA24CSE0097"
& $PY scripts\make_figures.py                                    # reports\figures\*.png for the team's report
```

The test run is keyed and resumable (`--resume`): if it is interrupted, rerun the same command. Do not re-run
the test to pick a better-looking result: the test split was evaluated end to end once per iteration (three times
in total, r1-r3), and the frozen r2 bundle also had one standard-protocol (ground-truth crop) test run after its
result was known; the status line of `eval_P5_T35.md` discloses this. Commit the new `reports/*` files and
`eval_P5_T35.md`.

## 7. Publish model files (so clean Docker builds include recognition)

Create a GitHub Release on the repository and upload the files listed with `"url": null` in
`artifacts/release_manifest.json` (heads, fine-tuned backbone, reference index). Put each file's
download URL into its manifest entry (the SHA256 values must stay the same) and commit. Releases: `model-v1`
(r1) and `model-v2` (r2) are history; r3's files are to be published as `model-v3` (until then their manifest
URLs are `null`). The CI clean-build container job (`.github/workflows/ci.yml`) downloads these files, so it
fails until the URLs are in the manifest.

## 8-9. Docker and deployment

Move Docker Desktop's disk image to the large drive first (Settings, Resources, Advanced, Disk image
location). Then:

```powershell
docker compose build
docker compose up -d; curl.exe -s localhost:8000/readyz
docker run -d --name semevia_offline --network none semevia:local   # must reach ready without network
```

Deploy the same image to the always-on host the team chooses (see `docs/hosting_lock.md`), then run
the release checks in the plan (phone and laptop, 30-minute idle first visit, 20 fresh uploads with
p50/p95, concurrent clients, bad inputs) and save them in `reports/release_acceptance.json`.

## Prompt for Claude Code on this PC

> You are continuing the SEMEVIA project in this repository (branch `semevia/implementation`). Read
> `NEXT.md`, `docs/runbook_other_pc.md` and `README.md` first. Execute the runbook steps in order on this
> machine, using the large drive for all data, caches and Docker storage. Ask me before downloading
> anything not listed in the runbook, before any step you expect to write more than about 5 GB, and
> before deploying or publishing anything. Stop and report if any check fails. Never fabricate or
> hand-type metrics; every number must come from the generated report files. Commit after each
> completed step with a clear message (only code, reports and small JSON artifacts, never weights or
> data), push to origin, and update `NEXT.md` with what was done, the commit SHA and the next step.
