# Fine-tuning run on the training PC

This closes the mandatory requirement "transfer learning and model fine-tuning": at least one
completed epoch with the final pretrained EfficientNet-B0 stage unfrozen, validated and
checkpointed. The script has been smoke-tested on CPU (plumbing only, 64 images); the real run
has **not** happened yet.

## 1. Get the code and data

```bash
git clone https://github.com/DuhItzAniket/SEMEVIA && cd SEMEVIA
git checkout semevia/implementation
git clone --depth 1 https://github.com/jhpohovey/StanfordCars.git ../StanfordCars   # ~3.8 GB
```

## 2. Environment

Python 3.11. Install a PyTorch build matching the GPU (see pytorch.org; for CUDA 12.x for
example `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`),
then the rest:

```bash
pip install numpy pillow scipy scikit-learn scikit-image opencv-python-headless pyyaml imagehash
```

Record the installed versions in the run notes (`pip freeze > freeze_train_pc.txt`).

## 3. Verify the data reproduces the frozen manifests

```bash
export SEMEVIA_DATA_ROOT=../StanfordCars/stanford_cars
python scripts/prepare_data.py --config configs/data.yaml
git status --short artifacts/manifests     # must print nothing: manifests byte-identical
```

If any manifest changed, stop: the data copy differs and results would not be comparable.

## 4. Train

```bash
export SEMEVIA_ARTIFACT_STORE=./store
python scripts/train.py --config configs/train.yaml --resume auto
```

* Stage 1: classifier only, 3 epochs. Stage 2: `features.7`, `features.8` and classifier,
  up to 6 epochs, early stopping patience 3, time cap 150 minutes overall.
* If interrupted, rerun the same command; it resumes from `store/runs/effb0_r1/last.pt`.
* On Windows, if data loading errors appear, add `--workers 0`.

## 5. Bring back these files (do not commit them to Git)

From `store/runs/effb0_r1/`: `best_finetuned.pt`, `best.pt`, `summary.json`,
`train_log.jsonl`, `tune_logits_best_finetuned.npy`. Copy them to
`D:/semevia_work/store/runs/effb0_r1/` on the development machine.

Acceptance (checked on return): `summary.json` shows `fine_tuning_requirement_met: true`, the
log shows `pretrained_changed: {features.7: true, features.8: true}`, and the tune top-1 in the
log matches a re-evaluation of `best_finetuned.pt` here.
