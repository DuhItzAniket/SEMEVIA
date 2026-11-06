"""Temperature, rejection threshold and conformal quantile for the Passport production mode.

Usage:
  python scripts/calibrate.py --development --mode hybrid
      Uses ONLY tuning-split predictions (annotation crops) and writes
      artifacts/development_calibration.json, labelled kind="development". The untouched
      calibration partition is not read.
  python scripts/calibrate.py --final
      After scripts/freeze.py and `scripts/evaluate.py --partition calibration --resume`: computes the
      conformal quantile from EVERY untouched calibration-partition input run through the frozen pipeline
      (missed detections use the frozen uniform fallback score) and writes artifacts/calibration_final.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from src.evaluation.metrics import ece, nll  # noqa: E402
from src.models.cache_io import store_dir  # noqa: E402
from src.models.heads import softmax  # noqa: E402
from src.passport.conformal import calibrate, calibration_scores  # noqa: E402


def fit_temperature(z: np.ndarray, y: np.ndarray) -> float:
    def f(logt):
        return nll(softmax(z, float(np.exp(logt))), y)
    r = minimize_scalar(f, bounds=(np.log(0.05), np.log(20.0)), method="bounded", options={"xatol": 1e-6})
    return float(np.exp(r.x))


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--development", action="store_true")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--mode", default="hybrid", choices=["deep", "hybrid"])
    ap.add_argument("--config", default="configs/calibration.yaml")
    ap.add_argument("--backbone", default=None, help="override the backbone tag (development mode)")
    args = ap.parse_args()
    if args.final:
        return final_calibration()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    if args.backbone:
        cfg["backbone"] = args.backbone
    heads_report = json.loads((ROOT / "reports" / f"heads_tune_{cfg['backbone']}.json").read_text(encoding="utf-8"))
    art = heads_report["artifacts"][args.mode]
    pred = store_dir() / "predictions" / f"tune_{args.mode}_{cfg['backbone']}_annotation_{art['sha256'][:12]}.npz"
    d = np.load(pred)
    z, y = d["logits"].astype(np.float64), d["y"]
    T = fit_temperature(z, y)
    p_raw, p_cal = softmax(z, 1.0), softmax(z, T)
    conf_raw, conf_cal = p_raw.max(1), p_cal.max(1)
    correct = p_cal.argmax(1) == y
    # Development rejection threshold: largest grid value keeping >= target known retention on tune.
    grid = cfg["reject_grid"]
    feasible = [t for t in grid if (conf_cal >= t).mean() >= cfg["known_retention_target"]]
    reject_t = max(feasible) if feasible else min(grid)
    cq = calibrate(calibration_scores(p_cal, y), cfg["alpha"])
    out = {
        "kind": "development",
        "warning": ("DEVELOPMENT calibration from the tuning split (annotation crops), the same split used to select "
                    "heads. Not the final calibration; coverage claims do not apply. Replaced in P6 by a quantile "
                    "from the untouched calibration partition with the frozen pipeline."),
        "production_mode": args.mode, "backbone": cfg["backbone"], "head_sha256": art["sha256"],
        "label_order": "class_id 0..195 (original_class_id - 1)",
        "temperature": T, "alpha": cfg["alpha"], **cq.to_dict(),
        "score_convention": "1 - p_true with p = softmax(z / T)",
        "reject": {"rule": "max calibrated probability < threshold", "threshold": reject_t,
                   "known_retention_target": cfg["known_retention_target"],
                   "tune_known_retention": float((conf_cal >= reject_t).mean()),
                   "negative_suite": "not yet available; threshold chosen on known tuning images only"},
        "diagnostics_tune": {"n": int(len(y)), "nll_raw": nll(p_raw, y), "nll_calibrated": nll(p_cal, y),
                             "ece_raw": ece(conf_raw, p_raw.argmax(1) == y)["ece"],
                             "ece_calibrated": ece(conf_cal, correct)["ece"],
                             "note": "in-sample for T (fit on the same tune predictions)"},
        "source_predictions": pred.name,
    }
    out["bundle_id"] = "devcal_" + hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:12]
    path = ROOT / "artifacts" / "development_calibration.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("bundle_id", "temperature", "k", "q", "quantile_mode")}, indent=1))
    print("reject", out["reject"]["threshold"], "retention", round(out["reject"]["tune_known_retention"], 3))
    print("diagnostics", json.dumps(out["diagnostics_tune"]))


def final_calibration() -> None:
    from src.data.manifest import read_manifest
    from src.passport.calibration import final_conformal_scores, model_fingerprint
    from src.release import load_manifest, sha256_file, verify

    manifest = load_manifest()
    if verify(manifest):
        raise SystemExit("release assets do not verify")
    fz_path = ROOT / "artifacts/production_freeze.json"
    if not fz_path.exists():
        raise SystemExit("run scripts/freeze.py first (P5 freeze)")
    fz = json.loads(fz_path.read_text(encoding="utf-8"))
    if model_fingerprint(manifest, fz["production_mode"], fz["encoder_runtime"]) != fz["model_fingerprint"]:
        raise SystemExit("the current bundle's model files differ from the frozen pipeline; re-freeze")
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    part = idx["partitions"]["calibration"]
    jl = store_dir() / "predictions" / f"e2e_calibration_clean_{part['sha256'][:10]}_{manifest['bundle_id']}.jsonl"
    if not jl.exists():
        raise SystemExit(f"missing {jl.name}; run scripts/evaluate.py --partition calibration --resume")
    rows = [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines()]
    keys = {r.image_key for r in read_manifest(ROOT / "artifacts/manifests" / part["file"])}
    if len(rows) != len(keys) or {r["image_key"] for r in rows} != keys:
        raise SystemExit("calibration predictions are incomplete or contain other keys")
    zkey = f"z_{fz['production_mode']}"
    scores, failures = final_conformal_scores(rows, zkey, fz["temperature"])
    cq = calibrate(scores, fz["alpha"])
    out = {
        "kind": "final", "warning": None, "production_mode": fz["production_mode"], "backbone": fz["backbone"],
        "head_sha256": fz["head_sha256"], "model_fingerprint": fz["model_fingerprint"],
        "label_order": fz["label_order"], "temperature": fz["temperature"], **cq.to_dict(),
        "score_convention": "1 - p_true with p = softmax(z / T); no-detection inputs use the uniform fallback 1 - 1/196",
        "reject": {"rule": "max calibrated probability < threshold", "threshold": fz["reject"]["threshold"],
                   "selection": fz["reject"]["rule"], "negatives_used": fz["reject"]["negatives_used"]},
        "calibration_population": {"partition": "calibration", "count": len(rows), "manifest_sha256": part["sha256"],
                                   "detection_failures_with_fallback_score": failures,
                                   "predictions_file": jl.name, "predictions_sha256": sha256_file(jl)},
        "freeze_sha256": sha256_file(fz_path),
        "interpretation": ("Nominal marginal coverage 1 - alpha for supported-category inputs under exchangeability with the "
                           "calibration partition. Not a per-image probability; does not cover unsupported vehicles or "
                           "perturbed inputs."),
    }
    out["bundle_id"] = "cal_" + hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:12]
    path = ROOT / "artifacts" / "calibration_final.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("bundle_id", "temperature", "n", "k", "q", "quantile_mode")}, indent=1),
          "failures:", failures)


if __name__ == "__main__":
    main()
