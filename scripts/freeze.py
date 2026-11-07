"""P5 freeze: fix the production pipeline's temperature and reject threshold on tuning data only.

Usage: python scripts/freeze.py --mode hybrid [--negatives reports/negatives_tune_<bundle>.json]
Prerequisite: python scripts/evaluate.py --partition tune --resume (for the current bundle).
Temperature is fitted on matched-primary tune rows (runtime box IoU >= 0.5) of the production mode;
the reject threshold is chosen on known tune images (and the negative tuning set when provided).
Writes artifacts/production_freeze.json with a fingerprint of every prediction-changing asset. After
this point any change to detector, backbone, heads, crop, features or runtime invalidates the freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.evaluation.metrics import ece  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.models.heads import softmax  # noqa: E402
from src.passport.calibration import (check_rows_runtime, choose_reject_threshold, fit_temperature,  # noqa: E402
                                      model_fingerprint, nll)
from src.release import load_manifest, recognition_config, sha256_file, verify  # noqa: E402
from src.vision.coordinates import iou  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["deep", "hybrid"])
    ap.add_argument("--negatives", default=None, help="negative tuning result JSON with per-image max probabilities")
    args = ap.parse_args()
    manifest = load_manifest()
    if verify(manifest):
        raise SystemExit("release assets do not verify")
    rc = recognition_config()
    if rc["backbone_tag"] == "imagenet_frozen":
        raise SystemExit("refusing to freeze the development backbone: the production backbone must come from a checkpoint "
                         "with a completed last-stage fine-tuning epoch (promote best_finetuned.pt first)")
    rt = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text(encoding="utf-8"))
    ccfg = yaml.safe_load((ROOT / "configs/calibration.yaml").read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    bundle = manifest["bundle_id"]
    jl = store_dir() / "predictions" / f"e2e_tune_clean_{idx['partitions']['tune']['sha256'][:10]}_{bundle}.jsonl"
    if not jl.exists():
        old = store_dir() / "predictions" / f"e2e_tune_{idx['partitions']['tune']['sha256'][:10]}_{bundle}.jsonl"
        jl = old if old.exists() else jl
    if not jl.exists():
        raise SystemExit(f"missing tune predictions for bundle {bundle}; run scripts/evaluate.py --partition tune --resume")
    rows = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines()]
    if len(rows) != idx["partitions"]["tune"]["count"]:
        raise SystemExit("tune predictions incomplete")
    head_rt = rc.get("hybrid_head_runtime", "sklearn")
    check_rows_runtime(rows, head_rt, jl.name)
    from src.data.manifest import read_manifest

    gt = {r.image_key: r.box for r in read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["tune"]["file"])}
    zkey = f"z_{args.mode}"
    matched = [r for r in rows if r.get("sel_box") is not None and zkey in r and iou(r["sel_box"], gt[r["image_key"]]) >= 0.5]
    Z = np.array([r[zkey] for r in matched], dtype=np.float64)
    Y = np.array([r["y"] for r in matched])
    T = fit_temperature(Z, Y)
    P1, PT = softmax(Z, 1.0), softmax(Z, T)
    known_conf = np.array([softmax(np.array([r[zkey]]), T)[0].max() if zkey in r and r.get("sel_box") is not None else 0.0
                           for r in rows])  # no detection -> cannot be accepted
    neg_conf = None
    if args.negatives:
        neg = json.loads(Path(args.negatives).read_text(encoding="utf-8"))
        if neg.get("partition") != "tune" or neg.get("bundle_id") != bundle:
            raise SystemExit("negative results must be the TUNE negative set for this bundle")
        if neg.get("hybrid_head_runtime", "sklearn") != head_rt:
            raise SystemExit("negative results were scored by a different hybrid head runtime")
        # stored raw production-mode logits; an image with no detection cannot be accepted (confidence 0)
        neg_conf = np.array([softmax(np.array([x["z"]]), T)[0].max() if x.get("z") else 0.0 for x in neg["rows"]])
    rej = choose_reject_threshold(known_conf, ccfg["reject_grid"], ccfg["known_retention_target"], neg_conf)
    runtime = rc.get("encoder_runtime", "torch")
    by_name = {a["name"]: a for a in manifest["assets"]}
    head_asset = next(a for a in manifest["assets"] if a["name"] == f"head_{args.mode}.joblib")
    out = {
        "kind": "frozen", "created_at": datetime.now(timezone.utc).isoformat(), "bundle_id_at_freeze": bundle,
        "model_fingerprint": model_fingerprint(manifest, args.mode, runtime, head_rt),
        "production_mode": args.mode, "backbone": rc["backbone_tag"], "encoder_runtime": runtime, "precision": rc["precision"],
        "head_sha256": head_asset["sha256"], "detector_sha256": by_name["yolo11n.pt"]["sha256"],
        **({"hybrid_head_runtime": head_rt, "head_int8_sha256": by_name["head_hybrid_int8.onnx"]["sha256"]}
           if head_rt != "sklearn" else {}),
        "detector_threshold": rt["target_policy"]["threshold"], "crop_margin": rt["crop"]["margin"],
        "temperature": T, "alpha": ccfg["alpha"],
        "temperature_fit": {"population": "tune matched-primary (runtime box IoU >= 0.5)", "n": int(len(Y)),
                            "excluded": len(rows) - len(Y), "nll_raw": nll(P1, Y), "nll_T": nll(PT, Y),
                            "ece_raw": ece(P1.max(1), P1.argmax(1) == Y)["ece"], "ece_T": ece(PT.max(1), PT.argmax(1) == Y)["ece"],
                            "note": "in-sample on the tune split"},
        "reject": {"rule_name": "max calibrated probability < threshold", **rej,
                   "known_retention_target": ccfg["known_retention_target"]},
        "tune_predictions": {"file": jl.name, "sha256": sha256_file(jl), "count": len(rows)},
        "label_order": "class_id 0..195 (original_class_id - 1)",
        "metadata_sha256": sha256_file(ROOT / "artifacts/class_metadata.json"),
        "features_config_sha256": sha256_file(ROOT / "configs/features.yaml"),
    }
    (ROOT / "artifacts/production_freeze.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("production_mode", "temperature", "model_fingerprint")}, indent=1))
    print("reject:", rej["threshold"], rej["rule"])


if __name__ == "__main__":
    main()
