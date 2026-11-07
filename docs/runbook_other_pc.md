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
is r3-int8: r3's backbone (trained with `configs\train_v3.yaml`) with the hybrid head served as ONNX INT8 (see
step 3b).

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
& $PY scripts\promote_bundle.py --backbone $CK --hybrid-head sklearn --calibration artifacts\development_calibration.json
& $PY scripts\optimize.py --backbone $TAG
& $PY scripts\optimize_head.py --mode hybrid --backbone $TAG     # must run before the INT8 promotion: writes the INT8 sidecar that promote_bundle.py checks
& $PY scripts\promote_bundle.py --backbone $CK --hybrid-head onnx_int8 --calibration artifacts\development_calibration.json
& $PY -m pytest
```

`--hybrid-head` is required: `onnx_int8` (the released r3-int8 setting) or `sklearn` (what r1, r2 and r3 used).
`optimize.py` and `optimize_head.py` read the backbone and hybrid head of the promoted bundle, so with a new
backbone promote once with `sklearn` first. With `onnx_int8`, `promote_bundle.py` refuses to promote unless the
`optimize_head.py` sidecar shows that the INT8 head passed the policy and was made from the same hybrid head; it
rebuilds the INT8 file from `head_hybrid.joblib` and refuses if the sha256 differs from the studied file.

Production mode: keep `hybrid` unless `reports\heads_tune_$TAG.json` shows `deep` clearly better
on tune (then use `--mode deep` in this step and in steps 5-6, and say so in the docs).

Optimization evidence on the final model (P5): `optimize.py` measures the ONNX FP32 and INT8 encoder variants,
`optimize_head.py` the INT8 hybrid head (scored row-wise, as served). For r3 every INT8 encoder variant failed the
accuracy policy, so the encoder stays PyTorch FP32 (`encoder_runtime: torch`, `precision: fp32`); keep it unless
you rebuild caches and references with the ONNX encoder (`reports\optimization_ckpt_98b1fb55cc6e.json`). The INT8
hybrid head passed the policy and is the released head runtime (`reports\optimization_head_hybrid_ckpt_98b1fb55cc6e.json`).

Commit: `reports/*.json`, `artifacts/release_manifest.json`, `artifacts/release_config.json`,
`artifacts/development_calibration.json` (message e.g. `feat(model): promote fine-tuned backbone`).

## 3b. Iterations r2, r3 and r3-int8 (r3-int8 is what is released)

Each iteration is the same steps 2-6 with its own training config (r3-int8 has no training step, see below).
Before rebuilding, archive the current freeze (`artifacts/history/r1/` before r2, `artifacts/history/r2/` before
r3, `artifacts/history/r3/` before r3-int8).

| Iteration | Config | Recipe | Status |
|---|---|---|---|
| r1 | `configs/train.yaml` | last-stage fine-tuning, 224 px | superseded (`artifacts/history/r1/`, release `model-v1`) |
| r2 | `configs/train_v2.yaml` | full-network fine-tuning, 224 px | superseded (`artifacts/history/r2/`, release `model-v2`) |
| r3 | `configs/train_v3.yaml` | r2's recipe at 320 px input, same seed | superseded (`artifacts/history/r3/`, release `model-v3`) |
| r3-int8 | as r3 (no new training) | r3 with the hybrid head's final logistic layer served as a row-wise ONNX INT8 graph | **released** (bundle `ft_ckpt_98b1fb55cc6e_f3734d4f40`; uses the `model-v3` files, no new upload) |

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

r3-int8 follows `NEXT.md` ("Iteration r3-int8 decision record", committed before any per-image, end-to-end,
calibration or test INT8 number existed). It keeps r3's `$CK` and tag: run `optimize_head.py`, then
`promote_bundle.py --backbone $CK --hybrid-head onnx_int8 --calibration artifacts\development_calibration.json`
(step 3), then steps 4-5 so that temperature, reject threshold and conformal q are re-derived from INT8 logits on
tune and calibration data, then the five pipeline checks listed in the record, and only then step 6. The
in-container check is `scripts\int8_parity_host.py` on the host, then `scripts/int8_parity_container.py` inside
the image (usage in the host script's header; result in `reports\int8_container_parity.json`).

## 4. Negative suite (collected; used for the r2, r3 and r3-int8 freezes)

See `docs/negative_suite.md`. With `$env:SEMEVIA_NEG_ROOT` set:
`& $PY scripts\evaluate_negatives.py --partition tune`

## 5. P5 freeze and final calibration

```powershell
& $PY scripts\evaluate.py --partition tune --resume
& $PY scripts\freeze.py --mode hybrid --negatives reports\negatives_tune_<bundle>.json   # r2, r3, r3-int8; r1 was frozen without --negatives
& $PY scripts\evaluate.py --partition calibration --resume
& $PY scripts\calibrate.py --final
& $PY scripts\promote_bundle.py --backbone $CK --hybrid-head onnx_int8 --calibration artifacts\calibration_final.json
& $PY -m pytest
```

Use the same `--hybrid-head` value as in step 3: `promote_bundle.py` refuses a final calibration made for a
different hybrid head runtime, and the freeze and calibration refuse prediction rows from a different runtime.

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
& $PY scripts\analyze_resolution.py                              # accuracy by car size, from the saved test predictions
& $PY scripts\evaluate_negatives.py --partition test          # held-out negatives, once after the freeze
& $PY scripts\evaluate_oracle.py --partition test --confirm-frozen   # standard Stanford Cars protocol (ground-truth crop); r2 and r3 only, not run for r3-int8
& $PY scripts\compare_runs.py --a ft_ckpt_98b1fb55cc6e_5d755416fd --b ft_ckpt_98b1fb55cc6e_f3734d4f40   # r3-int8: INT8 vs FP32 from the two saved prediction files
& $PY scripts\report.py --team-id T35 --usns "1RUA24CSE0051,1RUA24CSE0097"
& $PY scripts\make_figures.py                                    # reports\figures\*.png for the team's report
```

The test run is keyed and resumable (`--resume`): if it is interrupted, rerun the same command. Do not re-run
the test to pick a better-looking result. Test-split use so far, from the files in `reports/`: end to end once per
iteration, four times in total (r1, r2, r3, r3-int8; `e2e_test_*.json`); the stress subset for the same four
bundles (`robustness_stress_test_*.json`); the 40 held-out negatives for three (r2, r3, r3-int8;
`negatives_test_*.json`); the standard-protocol (ground-truth crop) run twice, for the frozen r2 bundle after its
result was known and for r3 (`oracle_test_*.json`; not re-run for r3-int8, as pre-registered). The status line of
`eval_P5_T35.md` discloses this. `analyze_resolution.py` and `compare_runs.py` read saved predictions and run no
model. Commit the new `reports/*` files and `eval_P5_T35.md`.

## 7. Publish model files (so clean Docker builds include recognition)

Create a GitHub Release on the repository and upload the files listed with `"url": null` in
`artifacts/release_manifest.json` (heads, fine-tuned backbone, reference index). Put each file's
download URL into its manifest entry (the SHA256 values must stay the same) and commit. Releases: `model-v1`
(r1) and `model-v2` (r2) are history; `model-v3` holds r3's files, which r3-int8 also uses. The INT8 hybrid
head (`head_hybrid_int8.onnx`) is not uploaded and keeps `"url": null`: `python -m src.release fetch` rebuilds it
from `head_hybrid.joblib`, its sha256 is pinned in the manifest, and `python -m src.release check-derived` in the
Docker build proves the rebuild is byte-identical. The CI clean-build container job (`.github/workflows/ci.yml`)
downloads the published files, so it fails until their URLs are in the manifest.

## 8-9. Docker and deployment

Move Docker Desktop's disk image to the large drive first (Settings, Resources, Advanced, Disk image
location). Then:

```powershell
docker compose build
docker compose up -d; curl.exe -s localhost:8000/readyz
docker run -d --name semevia_offline --network none semevia:local   # must reach ready without network
```

For r3-int8, `/readyz` must also show `"head_runtime": "onnx_int8"`, `"encoder_precision": "fp32"`, the
`head_int8_sha256` of the manifest and a `model_fingerprint` equal to the first 12 characters of the one in
`artifacts\production_freeze.json`.

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
