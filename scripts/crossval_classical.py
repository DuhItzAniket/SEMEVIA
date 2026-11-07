"""5-fold stratified cross-validation of the classical (handcrafted-feature) head on the training split.

Usage: python scripts/crossval_classical.py [--folds 5]

Only the classical head is cross-validated: its features do not depend on the fine-tuned CNN. The deep and hybrid
heads are not, because the CNN was trained on these very images, so its features on any training fold are
optimistic and a cross-validation score would be inflated. Uses the cached classical features, the same pipeline
(standard scaling + multinomial logistic regression) and the C selected on tune (reports/classical_tune.json).
Writes reports/crossval_classical.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.models.cache_io import cache_meta, load_cache  # noqa: E402
from src.models.heads import canonical_logits, topk_accuracy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/heads.yaml").read_text(encoding="utf-8"))["classical"]
    sel = json.loads((ROOT / "reports/classical_tune.json").read_text(encoding="utf-8"))["selected"]
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    keys, y, X, meta = load_cache(cache_meta("classical", "train", "annotation", idx))
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=cfg["seed"])
    folds = []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=sel["C"], max_iter=cfg["max_iter"], random_state=cfg["seed"]))
        pipe.fit(X[tr], y[tr])
        z = canonical_logits(pipe, X[va])
        folds.append({"fold": k + 1, "train": int(len(tr)), "validation": int(len(va)),
                      "top1": topk_accuracy(z, y[va], 1), "top5": topk_accuracy(z, y[va], 5)})
        print(json.dumps(folds[-1]), flush=True)
    t1 = np.array([f["top1"] for f in folds]); t5 = np.array([f["top5"] for f in folds])
    out = {"head": "classical (handcrafted features only)", "C": sel["C"], "folds": args.folds,
           "population": {"partition": "train", "count": int(len(y)), "manifest_sha256": meta["manifest_sha256"],
                          "crops": "annotation"},
           "protocol": "StratifiedKFold(shuffle, seed from configs/heads.yaml); scaler and classifier refit per fold; "
                       "C fixed to the tune-selected value",
           "per_fold": folds, "top1_mean": float(t1.mean()), "top1_std": float(t1.std(ddof=1)),
           "top5_mean": float(t5.mean()), "top5_std": float(t5.std(ddof=1)),
           "tune_top1_for_reference": sel["tune_top1"],
           "why_not_deep_hybrid": "The fine-tuned CNN was trained on the training split, so its features on any training "
                                  "fold are optimistic; cross-validating the deep or hybrid head there would overstate "
                                  "accuracy. Their uncertainty is reported with bootstrap intervals on the test predictions "
                                  "(reports/extra_test_<bundle>.json)."}
    (ROOT / "reports/crossval_classical.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"top1 {t1.mean():.4f} +/- {t1.std(ddof=1):.4f}; top5 {t5.mean():.4f} +/- {t5.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
