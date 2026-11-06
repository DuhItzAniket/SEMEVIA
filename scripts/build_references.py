"""Build the curated training-reference index from the learned train cache.

Usage: python scripts/build_references.py --backbone imagenet_frozen
Writes artifacts/store/release/reference_index.npz (+ .json) and validates that only training
keys entered it. The embedding version must equal the deployed encoder's tag; rebuild after any
backbone, preprocessing or precision change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import read_manifest  # noqa: E402
from src.models.cache_io import cache_meta, load_cache  # noqa: E402
from src.passport.references import build_index, save_index, validate_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet_frozen")
    args = ap.parse_args()
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    meta = cache_meta("learned", "train", "annotation", idx, args.backbone)
    keys, y, X, m = load_cache(meta)
    parts = {p: {r.image_key for r in read_manifest(ROOT / "artifacts/manifests" / v["file"])} for p, v in idx["partitions"].items()}
    version = f"{m['feature_version']}|{m['preprocess_version']}|fp32"
    index = build_index(keys, y, X, version)
    validate_index(index, parts["train"], parts["tune"] | parts["calibration"] | parts["test"])
    out = ROOT / "artifacts/store/release/reference_index.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_index(index, out)
    # Retrieval diagnostic on the tune split (annotation crops): does a reference share the true class?
    from src.passport.references import query

    ku, yu, Xu, _ = load_cache(cache_meta("learned", "tune", "annotation", idx, args.backbone))
    hits1 = hits3 = 0
    for q, yy in zip(Xu, yu):
        items = query(index, q, version)["items"]
        hits1 += int(items[0]["class_id"] == yy)
        hits3 += int(any(i["class_id"] == yy for i in items))
    report = {"index_id": index["index_id"], "entries": int(len(index["keys"])), "embedding_version": version,
              "population": {"partition": "tune", "count": int(len(yu)), "crops": "annotation"},
              "nearest_reference_same_class": {"numerator": hits1, "denominator": int(len(yu)), "value": hits1 / len(yu)},
              "any_of_three_same_class": {"numerator": hits3, "denominator": int(len(yu)), "value": hits3 / len(yu)},
              "note": "Descriptive retrieval diagnostic; references are illustrations, not identity evidence."}
    (ROOT / "reports" / f"references_tune_{args.backbone}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("index_id", "entries", "nearest_reference_same_class", "any_of_three_same_class")}))


if __name__ == "__main__":
    main()
