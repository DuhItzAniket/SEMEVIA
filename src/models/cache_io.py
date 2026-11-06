"""Versioned feature cache I/O. A cache is valid only for its exact manifest, crop policy,
feature version and config hash; load_cache refuses anything else."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def store_dir() -> Path:
    d = Path(os.environ.get("SEMEVIA_ARTIFACT_STORE", "D:/semevia_work/store"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_meta(features: str, partition: str, crop: str, manifest_index: dict, backbone: str | None = None) -> dict:
    import yaml

    fcfg_path = ROOT / "configs" / "features.yaml"
    fcfg = yaml.safe_load(fcfg_path.read_text(encoding="utf-8"))
    meta = {
        "features": features,
        "partition": partition,
        "manifest_sha256": manifest_index["partitions"][partition]["sha256"],
        "crop_policy": f"{crop}_margin{fcfg['crops']['margin']}",
    }
    if features == "classical":
        meta.update(feature_version=fcfg["feature_version"], features_config_sha256=file_sha(fcfg_path))
    else:
        from src.models.backbone import PREPROCESS_VERSION

        if not backbone:
            raise ValueError("learned caches need a backbone tag")
        meta.update(feature_version=f"effb0_pool1280_{backbone}", preprocess_version=PREPROCESS_VERSION,
                    backbone=backbone)
    return meta


def cache_paths(meta: dict) -> tuple[Path, Path]:
    tag = f"{meta['features']}_{meta['feature_version']}_{meta['crop_policy']}_{meta['partition']}_{meta['manifest_sha256'][:10]}"
    d = store_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}.npz", d / f"{tag}.json"


def save_cache(meta: dict, keys: list[str], y: np.ndarray, X: np.ndarray) -> None:
    npz, js = cache_paths(meta)
    if len(keys) != len(set(keys)) or len(keys) != len(X) or len(y) != len(X):
        raise ValueError("cache rows inconsistent")
    tmp = npz.with_suffix(".tmp.npz")
    np.savez(tmp, keys=np.array(keys), y=y, X=X)
    meta = dict(meta, data_sha256=file_sha(tmp))
    tmp.replace(npz)
    js.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def load_cache(expected_meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    npz, js = cache_paths(expected_meta)
    if not npz.exists() or not js.exists():
        raise FileNotFoundError(f"cache missing: {npz.name}; run scripts/cache_features.py")
    meta = json.loads(js.read_text(encoding="utf-8"))
    for k, v in expected_meta.items():
        if meta.get(k) != v:
            raise ValueError(f"incompatible cache {npz.name}: {k} {meta.get(k)!r} != {v!r}")
    if file_sha(npz) != meta["data_sha256"]:
        raise ValueError(f"cache {npz.name} content hash mismatch")
    d = np.load(npz)
    return d["keys"], d["y"], d["X"], meta
