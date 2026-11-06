"""Fit lightweight heads from versioned caches.

Usage: python scripts/fit_heads.py --head classical
Writes the fitted head to the artifact store (joblib, with SHA256 recorded) and a tuning
report under reports/. Tuning results on annotation crops are an oracle-crop diagnostic,
not an end-to-end or test result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.models.cache_io import cache_meta, load_cache, store_dir  # noqa: E402
from src.models.heads import canonical_logits, softmax, topk_accuracy  # noqa: E402


def fit_classical(cfg: dict, idx: dict) -> None:
    c = cfg["classical"]
    ktr, ytr, Xtr, mtr = load_cache(cache_meta("classical", "train", "annotation", idx))
    ktu, ytu, Xtu, mtu = load_cache(cache_meta("classical", "tune", "annotation", idx))
    if set(ktr) & set(ktu):
        raise SystemExit("train/tune keys overlap")
    runs = []
    best = None
    for C in c["C_grid"]:
        t = time.time()
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=c["max_iter"], random_state=c["seed"]))
        pipe.fit(Xtr, ytr)
        z = canonical_logits(pipe, Xtu)
        p = softmax(z)
        nll = float(-np.mean(np.log(np.clip(p[np.arange(len(ytu)), ytu], 1e-12, None))))
        r = {"C": C, "tune_top1": topk_accuracy(z, ytu, 1), "tune_top5": topk_accuracy(z, ytu, 5),
             "tune_nll": round(nll, 4), "train_top1": topk_accuracy(canonical_logits(pipe, Xtr), ytr, 1),
             "n_iter": int(pipe[-1].n_iter_.max()), "fit_seconds": round(time.time() - t, 1)}
        print(json.dumps(r), flush=True)
        runs.append(r)
        if best is None or r["tune_top1"] > best[0]["tune_top1"]:
            best = (r, pipe, z)
    r, pipe, z = best
    out = store_dir() / "heads"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"classical_{mtr['feature_version']}_C{r['C']}.joblib"
    joblib.dump(pipe, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    pred_dir = store_dir() / "predictions"
    pred_dir.mkdir(exist_ok=True)
    np.savez(pred_dir / f"tune_classical_annotation_{sha[:12]}.npz", keys=ktu, y=ytu, logits=z)
    report = {
        "head": "classical (handcrafted only)", "feature_version": mtr["feature_version"],
        "features_config_sha256": mtr["features_config_sha256"], "crop_policy": mtr["crop_policy"],
        "train": {"partition": "train", "count": int(len(ytr)), "manifest_sha256": mtr["manifest_sha256"]},
        "tune": {"partition": "tune", "count": int(len(ytu)), "manifest_sha256": mtu["manifest_sha256"]},
        "population_note": "Tune partition with annotation (oracle) crops; a development diagnostic, not test or end-to-end accuracy.",
        "grid": runs, "selected": r, "artifact": {"path": str(path.relative_to(store_dir())), "sha256": sha,
                                                   "bytes": path.stat().st_size},
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "classical_tune.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print("selected", json.dumps(r), "->", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["classical"], default="classical")
    ap.add_argument("--config", default="configs/heads.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    if args.head == "classical":
        fit_classical(cfg, idx)


if __name__ == "__main__":
    main()
