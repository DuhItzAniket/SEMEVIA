"""Predeclare the stress and viewpoint subsets (image keys only; no model outcomes are read).

Usage: python scripts/declare_subsets.py --config configs/evaluation.yaml
Writes artifacts/manifests/{stress_test,stress_tune,viewpoint_test}_m1.csv and subsets_index.json.
Re-running reproduces the same bytes; the frozen split manifests are not modified.
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

from src.data.manifest import read_manifest, write_manifest  # noqa: E402
from src.evaluation.stress import stratified_subset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/evaluation.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    parts = {p: read_manifest(ROOT / "artifacts/manifests" / v["file"]) for p, v in idx["partitions"].items()}
    seed = cfg["seed"]
    out = {"seed": seed, "source_manifest_version": idx["manifest_version"], "subsets": {}}
    for split in ("test", "tune"):
        recs = parts[split]
        keys = stratified_subset([r.image_key for r in recs], [r.class_id for r in recs], cfg["stress_subset"]["per_class"], seed)
        ks = set(keys)
        sub = [r for r in recs if r.image_key in ks]
        path = ROOT / cfg["stress_subset"]["manifests"][split]
        sha = write_manifest(sub, path)
        out["subsets"][f"stress_{split}"] = {"file": path.name, "count": len(sub), "sha256": sha,
                                             "classes": len({r.class_id for r in sub}), "from": split,
                                             "rule": f"seeded stratified, up to {cfg['stress_subset']['per_class']} per class"}
    stress_test = {r.image_key for r in read_manifest(ROOT / cfg["stress_subset"]["manifests"]["test"])}
    pool = sorted(r.image_key for r in parts["test"] if r.image_key not in stress_test)
    rng = np.random.RandomState(seed)
    vk = set(pool[i] for i in rng.choice(len(pool), cfg["viewpoint_subset"]["count"], replace=False))
    vsub = [r for r in parts["test"] if r.image_key in vk]
    vpath = ROOT / cfg["viewpoint_subset"]["manifest"]
    out["subsets"]["viewpoint_test"] = {"file": vpath.name, "count": len(vsub), "sha256": write_manifest(vsub, vpath),
                                        "from": "test", "rule": "seeded random, disjoint from stress_test"}
    (ROOT / "artifacts/manifests/subsets_index.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
