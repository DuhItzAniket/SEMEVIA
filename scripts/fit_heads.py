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


def _aligned(idx: dict, part: str, backbone: str):
    kl, yl, Xl, ml = load_cache(cache_meta("learned", part, "annotation", idx, backbone))
    kc, yc, Xc, mc = load_cache(cache_meta("classical", part, "annotation", idx))
    if not (np.array_equal(kl, kc) and np.array_equal(yl, yc)):
        raise SystemExit(f"{part}: learned and classical caches are not row-aligned")
    return kl, yl, np.concatenate([Xl, Xc], axis=1), ml, mc


def fit_learned(cfg: dict, idx: dict) -> None:
    from src.models.heads import FusionFeatures

    c = cfg["learned"]
    bb = c["backbone"]
    ktr, ytr, Xtr, mltr, mctr = _aligned(idx, "train", bb)
    ktu, ytu, Xtu, mltu, _ = _aligned(idx, "tune", bb)
    if set(ktr) & set(ktu):
        raise SystemExit("train/tune keys overlap")
    heads = dict(c["heads"], **c["ablations"])
    results, artifacts = {}, {}
    out = store_dir() / "heads"
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = store_dir() / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for name, spec in heads.items():
        runs, best = [], None
        for C in c["C_grid"]:
            t = time.time()
            pipe = make_pipeline(FusionFeatures(spec["use_learned"], tuple(spec["hc_groups"]), c["pca_components"],
                                                spec.get("branch_balance", "equal_variance")),
                                 LogisticRegression(C=C, max_iter=c["max_iter"], random_state=c["seed"]))
            pipe.fit(Xtr, ytr)
            z = canonical_logits(pipe, Xtu)
            p = softmax(z)
            r = {"C": C, "tune_top1": topk_accuracy(z, ytu, 1), "tune_top5": topk_accuracy(z, ytu, 5),
                 "tune_nll": round(float(-np.mean(np.log(np.clip(p[np.arange(len(ytu)), ytu], 1e-12, None)))), 4),
                 "train_top1": topk_accuracy(canonical_logits(pipe, Xtr), ytr, 1),
                 "input_dim": int(pipe[0].transform(Xtr[:2]).shape[1]), "fit_seconds": round(time.time() - t, 1)}
            print(name, json.dumps(r), flush=True)
            runs.append(r)
            if best is None or r["tune_top1"] > best[0]["tune_top1"]:
                best = (r, pipe, z)
        r, pipe, z = best
        path = out / f"{name}_{bb}_C{r['C']}.joblib"
        joblib.dump(pipe, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        np.savez(pred_dir / f"tune_{name}_{bb}_annotation_{sha[:12]}.npz", keys=ktu, y=ytu, logits=z)
        results[name] = {"spec": spec, "grid": runs, "selected": r}
        artifacts[name] = {"path": str(path.relative_to(store_dir())), "sha256": sha, "bytes": path.stat().st_size}
    report = {
        "backbone": bb,
        "backbone_status": ("DEVELOPMENT: frozen ImageNet EfficientNet-B0, not fine-tuned; does not satisfy the "
                            "fine-tuning requirement" if bb == "imagenet_frozen" else "fine-tuned checkpoint"),
        "learned_feature_version": mltr["feature_version"], "preprocess_version": mltr["preprocess_version"],
        "classical_feature_version": mctr["feature_version"], "crop_policy": mltr["crop_policy"],
        "train": {"count": int(len(ytr)), "manifest_sha256": mltr["manifest_sha256"]},
        "tune": {"count": int(len(ytu)), "manifest_sha256": mltu["manifest_sha256"]},
        "population_note": "Tune partition, annotation (oracle) crops; development diagnostic, not test or end-to-end accuracy.",
        "protocol": "Each head (incl. ablations) refit on train rows; C selected from the same fixed grid by tune top-1; "
                    "PCA and scalers fit on train rows only.",
        "heads": results, "artifacts": artifacts,
    }
    (ROOT / "reports" / f"heads_tune_{bb}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: (v["selected"]["C"], round(v["selected"]["tune_top1"], 4), round(v["selected"]["tune_top5"], 4))
                      for k, v in results.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["classical", "learned"], default="classical")
    ap.add_argument("--config", default="configs/heads.yaml")
    ap.add_argument("--backbone", default=None, help="override learned.backbone (e.g. ckpt_<sha12> from cache_features)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.backbone:
        cfg["learned"]["backbone"] = args.backbone
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    if args.head == "classical":
        fit_classical(cfg, idx)
    else:
        fit_learned(cfg, idx)


if __name__ == "__main__":
    main()
