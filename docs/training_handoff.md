# Fine-tuning run on the training PC

> **Superseded scope (2026-11-06):** the GPU PC now runs the whole remaining heavy pipeline
> (training, final bundle, freeze, calibration, test runs, Docker, deployment) and commits its results.
> Follow `docs/runbook_other_pc.md`; this file remains the detailed reference for the training step.

This closes the mandatory requirement "transfer learning and model fine-tuning": at least one
completed epoch with the final pretrained EfficientNet-B0 stage unfrozen, validated and
checkpointed. The script has been smoke-tested on CPU (plumbing only, 64 images); the real run
has **not** happened yet.

Git: one machine commits at a time (see `docs/runbook_other_pc.md`).

Commands below are Windows PowerShell. Bash equivalents: `export NAME=value` instead of
`$env:NAME = "value"`, and `venv/bin/python` instead of `venv\Scripts\python`.

## 0. Requirements

* NVIDIA GPU with a recent driver (`nvidia-smi` must work), Python 3.11, Git.
* About 10 GB free disk for data, environment and checkpoints.

## 1. Get the code and data (keep data and checkpoints outside the repository)

```powershell
mkdir C:\semevia; cd C:\semevia
git clone https://github.com/DuhItzAniket/SEMEVIA
cd SEMEVIA
git checkout semevia/implementation
git clone --depth 1 https://github.com/jhpohovey/StanfordCars.git C:\semevia\StanfordCars
```

The data clone is about 3.8 GB and must contain `stanford_cars\cars_train` (8,144 jpg),
`stanford_cars\cars_test` (8,041 jpg), `stanford_cars\devkit\*.mat` and
`stanford_cars\cars_test_annos_withlabels.mat`.

## 2. Environment

```powershell
python -m venv C:\semevia\venv
C:\semevia\venv\Scripts\python -m pip install --upgrade pip
# PyTorch with CUDA. cu124 (torch 2.6) conflicts with sympy==1.14.0 in requirements.lock; the
# combination verified on 2026-11-07 is below (needs an nvidia-smi CUDA version of 12.6 or newer).
C:\semevia\venv\Scripts\python -m pip install torch==2.14.1+cu126 torchvision==0.29.1+cu126 --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# Pinned to the development machine's versions (requirements.lock) so the split check is exact
C:\semevia\venv\Scripts\python -m pip install numpy==2.4.6 pillow==13.0.0 scipy==1.17.1 scikit-learn==1.9.1 scikit-image==0.26.0 opencv-python-headless==5.0.0.93 PyYAML==6.0.3 ImageHash==4.3.2
C:\semevia\venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
C:\semevia\venv\Scripts\python -m pip freeze > C:\semevia\freeze_train_pc.txt
```

The check must print `True` and the GPU name. If it prints `False`, the CPU build was
installed: uninstall torch/torchvision and reinstall from the CUDA index.

## 3. Verify the data reproduces the frozen manifests

```powershell
$env:SEMEVIA_DATA_ROOT = "C:\semevia\StanfordCars\stanford_cars"
C:\semevia\venv\Scripts\python scripts\prepare_data.py --config configs\data.yaml
git status --short artifacts\manifests     # must print NOTHING (manifests byte-identical)
git restore reports artifacts\class_metadata.json   # discard regenerated report files (timestamps)
```

If any file under `artifacts\manifests` changed, **stop**: this data copy differs, so results
would not match the frozen splits. Report it instead of training.

## 4. Train

```powershell
$env:SEMEVIA_DATA_ROOT = "C:\semevia\StanfordCars\stanford_cars"
$env:SEMEVIA_ARTIFACT_STORE = "C:\semevia\store"
C:\semevia\venv\Scripts\python scripts\train.py --config configs\train.yaml --resume auto
```

* The first line printed must show `"cuda": true` and the GPU name.
* Stage 1: classifier only, 3 epochs. Stage 2: `features.7`, `features.8` and classifier,
  up to 6 epochs, early stopping patience 3, overall time cap 150 minutes.
* One line per epoch is printed with tune top-1/top-5; stage-2 lines must contain
  `"pretrained_changed": {"features.7": true, "features.8": true}`.
* If interrupted, rerun the same command; it resumes from `C:\semevia\store\runs\effb0_r1\last.pt`.
* If data loading errors appear on Windows, add `--workers 0` (slower, same result).
* Do not change `configs\train.yaml` or the manifests; that would make the run incomparable.

Finished when the last line shows `"fine_tuning_requirement_met": true`.

## 5. Bring back these files (never commit them to Git)

From `C:\semevia\store\runs\effb0_r1\`:

| File | Why |
|---|---|
| `best_finetuned.pt` | compliant backbone used by the app (required) |
| `best.pt` | best checkpoint of any stage (comparison) |
| `summary.json`, `train_log.jsonl` | evidence: epochs, timings, GPU, pretrained-weight change |
| `tune_logits_best_finetuned.npy` | tune predictions for checking on return |
| `C:\semevia\freeze_train_pc.txt` | exact package versions used |

Copy them (USB drive, cloud drive, etc.) to `D:\semevia_work\store\runs\effb0_r1\` on the
development machine (`freeze_train_pc.txt` can go in the same folder).

Acceptance on return (done on the development machine): `summary.json` shows
`fine_tuning_requirement_met: true`, the log shows the pretrained-weight change, and a
re-evaluation of `best_finetuned.pt` on the tune split reproduces the logged top-1 within a
small tolerance.

## If a Claude Code session runs on the training PC

Give it this prompt after step 1:

> Read `docs/training_handoff.md` in this repository and carry out steps 2 to 4 exactly. This
> machine only trains: do not edit tracked files, commit, push, or change configs or manifests.
> Stop and report if the GPU check prints False, if `git status --short artifacts/manifests`
> shows any change, or if training fails. When done, report the summary line and the stage-2
> epoch lines, and list the files from step 5 with their sizes and SHA256 hashes.
