"""Standard Stanford Cars classification protocol on the frozen release: classify the annotated car.

Usage: python scripts/evaluate_oracle.py --partition test --confirm-frozen

Most published Stanford Cars results classify the image region given by the ground-truth box (no detector).
SEMEVIA's headline metric is stricter (end to end: the runtime detector must find the car, IoU >= 0.5, and the
class must be right), so this script adds the comparable number. It uses exactly the frozen bundle (same crop
margin, encoder, heads) and refuses to run unless the release matches artifacts/production_freeze.json. Nothing is
selected or tuned from this output. Writes reports/oracle_<partition>_<bundle>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.data.crops import annotation_crop, load_rgb  # noqa: E402
from src.data.manifest import read_manifest  # noqa: E402
from src.models.heads import topk_accuracy  # noqa: E402
from src.release import load_manifest, recognition_config, verify  # noqa: E402

MODES = ("classical", "deep", "hybrid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="test", choices=["tune", "calibration", "test"])
    ap.add_argument("--confirm-frozen", action="store_true")
    ap.add_argument("--data-root", default=os.environ.get("SEMEVIA_DATA_ROOT"))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    manifest = load_manifest()
    if verify(manifest):
        raise SystemExit("release assets do not verify")
    rc = recognition_config()
    if args.partition in ("calibration", "test"):
        from src.passport.calibration import model_fingerprint

        fz = json.loads((ROOT / "artifacts/production_freeze.json").read_text(encoding="utf-8"))
        if model_fingerprint(manifest, fz["production_mode"], fz["encoder_runtime"]) != fz["model_fingerprint"]:
            raise SystemExit("current model files differ from the frozen pipeline; refusing to touch held-out data")
        if args.partition == "test" and not args.confirm_frozen:
            raise SystemExit("the test partition needs --confirm-frozen")
    from src.models.inference import CropInput, Predictor
    from src.vision.features import extract_classical

    margin = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))["crops"]["margin"]
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"][args.partition]
    recs = read_manifest(ROOT / "artifacts/manifests" / part["file"])
    pred = Predictor({m: ROOT / p for m, p in rc["heads"].items()}, ROOT / rc["backbone"], 4)
    data_root = Path(args.data_root)

    def load(r):
        rgb = load_rgb(data_root / r.rel_path)
        crop, box = annotation_crop(rgb, r.box, margin)
        return CropInput(crop, box, (rgb.shape[1], rgb.shape[0]))

    Z = {m: [] for m in MODES}
    y = np.array([r.class_id for r in recs])
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        for s in range(0, len(recs), args.batch):
            cins = list(pool.map(load, recs[s:s + args.batch]))
            H = np.stack(list(pool.map(lambda c: extract_classical(c.crop, c.box, c.image_size).vector, cins))).astype(np.float64)
            L = pred.learned(cins)
            Z["classical"].append(pred.score_features("classical", None, H))
            Z["deep"].append(pred.score_features("deep", L, H))
            Z["hybrid"].append(pred.score_features("hybrid", L, H))
            if (s // args.batch) % 25 == 0:
                print(f"  {s + len(cins)}/{len(recs)}", flush=True)
    out = {"protocol": "standard Stanford Cars classification: ground-truth box crop (margin %.2f), no detector" % margin,
           "partition": args.partition, "bundle_id": manifest["bundle_id"], "encoder_input_size": pred.input_size,
           "population": {"count": len(recs), "manifest_sha256": part["sha256"]}, "modes": {},
           "note": "Comparable to published classification results; SEMEVIA's headline metric is the stricter "
                   "end-to-end top-1 in reports/e2e_<partition>_<bundle>.json.",
           "wall_seconds": round(time.time() - t0, 1)}
    for m in MODES:
        z = np.concatenate(Z[m])
        t1 = int(round(topk_accuracy(z, y, 1) * len(y)))
        t5 = int(round(topk_accuracy(z, y, 5) * len(y)))
        out["modes"][m] = {"top1": {"numerator": t1, "denominator": len(y), "value": t1 / len(y)},
                           "top5": {"numerator": t5, "denominator": len(y), "value": t5 / len(y)}}
    path = ROOT / "reports" / f"oracle_{args.partition}_{manifest['bundle_id']}.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({m: (v["top1"]["value"], v["top5"]["value"]) for m, v in out["modes"].items()}))


if __name__ == "__main__":
    main()
