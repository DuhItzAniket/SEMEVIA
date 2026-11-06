"""Assemble eval_P5_<TeamID>.md from saved report files, in the PS metric order.

Usage: python scripts/report.py --team-id TeamID [--usns "USN1,USN2"]
Every number is read from a reports/*.json file and printed with its population. Final
(official test) values stay "pending" until the P6 test reports exist; development (tune)
diagnostics are shown separately and labelled as such. Nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PENDING = "pending (official test run after the P5 freeze, P6)"


def load(pattern: str):
    files = sorted(glob.glob(str(ROOT / "reports" / pattern)))
    return (json.loads(Path(files[-1]).read_text(encoding="utf-8")), Path(files[-1]).name) if files else (None, None)


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-id", default="TeamID")
    ap.add_argument("--usns", default=None, help="comma-separated USNs of all members (required for submission)")
    args = ap.parse_args()
    test, test_f = load("e2e_test_*.json")
    tune, tune_f = load("e2e_tune_dev_*.json")
    cls, _ = load("classical_tune.json")
    heads, heads_f = load("heads_tune_*.json")
    det, det_f = load("detector_tune.json")
    opt, opt_f = load("optimization_imagenet_frozen.json")
    opth, opth_f = load("optimization_head_hybrid_*.json")
    rob, rob_f = load("robustness_stress_tune_*.json")
    L = []
    L.append(args.usns if args.usns else "USNS_PENDING_TEAM_TO_PROVIDE (first line must list all member USNs, comma-separated)")
    L += ["", f"# SEMEVIA comparative evaluation (PS5, {args.team_id})", "",
          "Status: **development**. The backbone is frozen ImageNet EfficientNet-B0 (fine-tuning pending) and "
          "calibration is development-only. Final values are filled from the official 8,041-image test run; until "
          "then they read *pending*. Tune-split numbers are selection-set diagnostics and are optimistic.", ""]

    L += ["## 1. Classification performance (Top-1 and Top-5)", ""]
    L += [f"Final (official test, end-to-end): {PENDING if not test else ''}", ""]
    if tune:
        L += [f"Development: tune split ({tune['population']['count']} images), runtime box rule, misses count as failures; "
              f"source `reports/{tune_f}`.", "",
              "| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |", "|---|---|---|---|---|"]
        for m, v in tune["modes"].items():
            L.append(f"| {m} | {pct(v['end_to_end_top1']['value'])} | {pct(v['end_to_end_top5']['value'])} | "
                     f"{pct(v['matched_primary_top1']['value'])} | {pct(v['predicted_crop_top1']['value'])} |")
        L.append("")
    if heads and cls:
        L += [f"Oracle-crop diagnostic (annotation crops, tune): classical {pct(cls['selected']['tune_top1'])} / "
              f"{pct(cls['selected']['tune_top5'])}; deep {pct(heads['heads']['deep']['selected']['tune_top1'])} / "
              f"{pct(heads['heads']['deep']['selected']['tune_top5'])}; hybrid {pct(heads['heads']['hybrid']['selected']['tune_top1'])} / "
              f"{pct(heads['heads']['hybrid']['selected']['tune_top5'])} (top-1 / top-5; `reports/{heads_f}`).", ""]

    L += ["## 2. Object localization (IoU)", "", f"Final: {PENDING if not test else ''}", ""]
    if tune:
        L += ["| Path | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | Missing |", "|---|---|---|---|---|"]
        for m in ("classical", "hybrid"):
            loc = tune["modes"][m]["localization"]
            L.append(f"| {tune['modes'][m]['localization_source']} | {loc['mean_iou_valid_selected']['value']:.3f} | "
                     f"{loc['mean_iou_zero_for_failures']['value']:.3f} | {pct(loc['primary_recall_at_iou']['value'])} | {loc['missing']} |")
        L += ["", "(Development: tune split. One-to-one primary target vs the single annotated car.)", ""]

    L += ["## 3. Object detection (mAP)", "", f"Final: {PENDING if not test else ''}", ""]
    if det:
        a = det["generic_vehicle_ap"]
        L += [f"Generic vehicle AP (all YOLO11n vehicle detections, 1 category), tune: AP50 {a['AP50']:.3f}, AP50:95 {a['AP50_95']:.3f} "
              f"(`reports/{det_f}`). Unannotated background cars count as false positives.", ""]
    if tune:
        L += ["196-category AP (max one classified prediction per image, score = localisation score x top-1 probability), tune: " +
              "; ".join(f"{m} AP50 {v['fine_grained_ap']['AP50']:.3f} / AP50:95 {v['fine_grained_ap']['AP50_95']:.3f}"
                        for m, v in tune["modes"].items()), ""]

    L += ["## 4. Model efficiency (model size and inference time)", ""]
    if opt:
        s, lt = opt["size_bytes"], opt["latency_single_crop_encoder_only"]
        L += [f"Encoder, single crop, 2 threads, Intel i5-1135G7 CPU (`reports/{opt_f}`):", "",
              "| Runtime | Size | p50 | p95 | Hybrid tune top-1 |", "|---|---|---|---|---|"]
        for n, key in (("PyTorch FP32", "torch_fp32"), ("ONNX FP32", "onnx_fp32"), ("ONNX INT8 static (MinMax)", "onnx_int8"),
                       ("ONNX INT8 static (percentile)", "onnx_int8_percentile"), ("ONNX INT8 dynamic", "onnx_int8_dynamic")):
            size = s.get("torch_state_dict_pth" if key == "torch_fp32" else key)
            if key in lt and key in opt["accuracy"]:
                L.append(f"| {n} | {size / 1e6:.1f} MB | {lt[key]['p50_ms']:.1f} ms | {lt[key]['p95_ms']:.1f} ms | "
                         f"{pct(opt['accuracy'][key]['hybrid']['tune_top1'])} |")
        L.append("")
    if opth:
        L += [f"INT8 linear head (`reports/{opth_f}`): {opth['size_bytes']['onnx_fp32'] / 1e6:.2f} MB -> "
              f"{opth['size_bytes']['onnx_int8'] / 1e6:.2f} MB, tune top-1 {pct(opth['accuracy']['onnx_fp32']['tune_top1'])} -> "
              f"{pct(opth['accuracy']['onnx_int8']['tune_top1'])}.", ""]
    L += ["Deployed end-to-end latency (p50/p95 over fresh uploads on the public host): pending (P7).", ""]

    L += ["## 5. Robustness under varying conditions", "", f"Final (392-image test stress subset): {PENDING if not test else ''}", ""]
    if rob:
        L += [f"Development: {rob['count']}-image tune stress subset, conditions applied before detection (`reports/{rob_f}`).", "",
              "| Condition | Mode | E2E top-1 | Change vs clean | Box recall@0.5 |", "|---|---|---|---|---|"]
        for r in rob["rows"]:
            if r["mode"] in ("hybrid", "classical"):
                L.append(f"| {r['condition']} | {r['mode']} | {pct(r['e2e_top1'])} | {100 * r['delta_e2e_top1_vs_clean']:+.1f} pt | {pct(r['loc_recall50'])} |")
        L.append("")
    L += ["Natural viewpoint subgroups (60 labelled test photos, `reports/viewpoint_labels.csv`): pending (P6).", ""]

    L += ["## 6. Confidence and threshold analysis", "", f"Final: {PENDING if not test else ''}", ""]
    if tune:
        p = tune["passport_production_mode"]
        c = p["calibration_matched_primary"]
        s = p["sets_all_inputs"]
        L += [f"Development (tune, production mode {p['mode']}, development calibration fitted on this split, so in-sample):", "",
              f"* ECE (10 bins, matched-primary n={c['n']}): raw {c['ece_raw']:.3f}, temperature-scaled {c['ece_calibrated']['ece']:.3f}; "
              f"NLL {c['nll_raw']:.3f} -> {c['nll_calibrated']:.3f}.",
              f"* Reject threshold {p['accepted']['threshold']}: retention {pct(p['accepted']['retention']['value'])}, accepted "
              f"end-to-end accuracy {pct(p['accepted']['accepted_end_to_end_accuracy']['value'])}.",
              f"* Passport sets (all inputs): coverage {pct(s['coverage']['value'])}, median size {s['median_set_size']:.0f}, "
              f"singletons {pct(s['singleton']['value'])} with singleton accuracy {pct(s['singleton_accuracy']['value'])}.",
              "* Negative (unsupported-input) false acceptance: pending (negative suite not collected).", ""]
    L += ["## Ablations and failure analysis", "",
          "Ablations (tune, matched heads): see `reports/heads_tune_imagenet_frozen.json`. Failure cases: pending (selected from "
          "final test predictions in P6).", ""]
    out = ROOT / f"eval_P5_{args.team_id}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.name} ({len(L)} lines)")


if __name__ == "__main__":
    main()
