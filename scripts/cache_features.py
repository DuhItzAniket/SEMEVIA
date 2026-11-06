"""Compute versioned feature caches for manifest partitions.

Usage:
  python scripts/cache_features.py --features classical --partitions train tune --crop annotation

Each cache stores image keys, labels and vectors plus a JSON sidecar with the manifest hash,
crop policy, feature version and config hash. Consumers refuse caches whose sidecar does not
match the current configuration (see src/models/cache_io.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.manifest import read_manifest  # noqa: E402
from src.models.cache_io import cache_meta, cache_paths, save_cache, store_dir  # noqa: E402

_DATA_ROOT: Path | None = None


def _init(data_root: str) -> None:
    global _DATA_ROOT
    _DATA_ROOT = Path(data_root)


def _classical_job(args):
    rel, box, margin = args
    from src.data.crops import annotation_crop, load_rgb
    from src.vision.features import extract_classical

    rgb = load_rgb(_DATA_ROOT / rel)
    crop, cbox = annotation_crop(rgb, box, margin)
    return extract_classical(crop, cbox, (rgb.shape[1], rgb.shape[0])).vector


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=["classical"], default="classical")
    ap.add_argument("--partitions", nargs="+", default=["train", "tune"])
    ap.add_argument("--crop", choices=["annotation"], default="annotation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--data-root", default=os.environ.get("SEMEVIA_DATA_ROOT"))
    args = ap.parse_args()
    if not args.data_root or not Path(args.data_root).is_dir():
        raise SystemExit("set SEMEVIA_DATA_ROOT or --data-root")
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    import yaml

    margin = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))["crops"]["margin"]
    for part in args.partitions:
        meta = cache_meta(args.features, part, args.crop, idx)
        npz, _ = cache_paths(meta)
        if npz.exists():
            print(f"skip {part}: cache exists {npz.name}")
            continue
        recs = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"][part]["file"])
        jobs = [(r.rel_path, r.box, margin) for r in recs]
        t = time.time()
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.data_root,)) as ex:
            X = np.stack(list(ex.map(_classical_job, jobs, chunksize=32)))
        if not np.isfinite(X).all():
            raise SystemExit("non-finite features")
        meta.update(count=len(recs), elapsed_s=round(time.time() - t, 1), dim=int(X.shape[1]))
        save_cache(meta, [r.image_key for r in recs], np.array([r.class_id for r in recs]), X)
        print(f"{part}: {X.shape} in {meta['elapsed_s']} s -> {npz}")
    print("store:", store_dir())


if __name__ == "__main__":
    main()
