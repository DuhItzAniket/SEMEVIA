"""Temperature, rejection threshold and conformal quantile for the Passport production mode.

Usage:
  python scripts/calibrate.py --development --mode hybrid
      Uses ONLY tuning-split predictions (annotation crops) and writes
      artifacts/development_calibration.json, labelled kind="development". The untouched
      calibration partition is not read.
  (final mode, P6: computed from the calibration partition with the frozen pipeline; not yet implemented)
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
    ap.add_argument("--development", action="store_true", required=True)
    ap.add_argument("--mode", default="hybrid", choices=["deep", "hybrid"])
    ap.add_argument("--config", default="configs/calibration.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    main()
