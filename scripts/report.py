"""Assemble eval_P5_<TeamID>.md from saved report files, in the PS5 metric order.

Usage: python scripts/report.py --team-id T35 --usns "USN1,USN2"

Every number is read from a reports/*.json or artifacts/*.json file and printed with its population
and source file. Reports are selected by the release bundle and backbone tag recorded in
artifacts/release_manifest.json / release_config.json, never by "latest file". Final (official test)
sections stay "pending" until the P6 test reports for the current bundle exist. Nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PENDING = "pending (official test run after the P5 freeze, P6)"
MODES = ("classical", "deep", "hybrid")


def read(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def find(pattern: str):
    """Exactly one match or None; several matches are an error (ambiguous source)."""
    files = sorted(glob.glob(str(ROOT / "reports" / pattern)))
    if len(files) > 1:
        raise SystemExit(f"ambiguous report source for {pattern}: {[Path(f).name for f in files]}")
    return (json.loads(Path(files[0]).read_text(encoding="utf-8")), Path(files[0]).name) if files else (None, None)


def pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def frac(r):
    """A {numerator, denominator, value} ratio as '75.6% (616/815)'."""
    if r is None or r.get("value") is None:
        return "n/a"
    n = r["numerator"]
    n = f"{n:.0f}" if isinstance(n, float) and float(n).is_integer() else (f"{n:.1f}" if isinstance(n, float) else n)
    return f"{pct(r['value'])} ({n}/{r['denominator']})"


def manifest_rows(name: str) -> dict:
    with open(ROOT / "artifacts/manifests" / name, encoding="utf-8") as f:
        return {r["image_key"]: r for r in csv.DictReader(f)}


def mb(b):
    return f"{b / 1e6:.1f} MB" if b >= 1e5 else f"{b / 1e3:.1f} KB"


def classification_table(e2e):
    L = ["| Mode | End-to-end top-1 | End-to-end top-5 | Matched-primary top-1 | Predicted-crop top-1 |",
         "|---|---|---|---|---|"]
    for m in MODES:
        v = e2e["modes"][m]
        L.append(f"| {m} | {frac(v['end_to_end_top1'])} | {frac(v['end_to_end_top5'])} | "
                 f"{frac(v['matched_primary_top1'])} | {frac(v['predicted_crop_top1'])} |")
    return L


def localization_table(e2e):
    L = ["| Path (mode) | Mean IoU (valid boxes) | Mean IoU (0 for misses) | Recall@0.50 | No box | Wrong target |",
         "|---|---|---|---|---|---|"]
    for m in ("classical", "hybrid"):
        loc = e2e["modes"][m]["localization"]
        L.append(f"| {e2e['modes'][m]['localization_source']} ({m}) | {loc['mean_iou_valid_selected']['value']:.3f} | "
                 f"{loc['mean_iou_zero_for_failures']['value']:.3f} | {frac(loc['primary_recall_at_iou'])} | "
                 f"{loc['missing']} | {loc['wrong_target_below_thr']} |")
    return L


def ap_table(e2e):
    L = ["| Mode | 196-category AP50 | 196-category AP50:95 | Predictions / ground truth |", "|---|---|---|---|"]
    for m in MODES:
        a = e2e["modes"][m]["fine_grained_ap"]
        L.append(f"| {m} | {a['AP50']:.3f} | {a['AP50_95']:.3f} | {a['num_predictions']} / {a['num_ground_truth']} |")
    return L


def passport_lines(p):
    c, s, a = p["calibration_matched_primary"], p["sets_all_inputs"], p["accepted"]
    return [f"* Calibration (matched-primary rows, n={c['n']}; {c['excluded_miss_or_wrong_target']} misses/wrong targets "
            f"excluded): ECE (10 bins) raw {c['ece_raw']:.3f}, temperature-scaled {c['ece_calibrated']['ece']:.3f}; "
            f"NLL {c['nll_raw']:.3f} -> {c['nll_calibrated']:.3f}.",
            f"* Reject threshold {a['threshold']} (max calibrated probability): retention {frac(a['retention'])}, "
            f"accepted end-to-end accuracy {frac(a['accepted_end_to_end_accuracy'])}.",
            f"* Passport candidate sets (all inputs): coverage {frac(s['coverage'])}, mean size {s['mean_set_size']:.2f}, "
            f"median size {s['median_set_size']:.0f}, singletons {frac(s['singleton'])} with singleton accuracy "
            f"{frac(s['singleton_accuracy'])}, empty sets {frac(s['empty'])}."]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-id", default="TeamID")
    ap.add_argument("--usns", default=None, help="comma-separated USNs of all members (required for submission)")
    ap.add_argument("--readme", action="store_true", help="also refresh the README results block")
    args = ap.parse_args()

    manifest, rel_cfg = read("artifacts/release_manifest.json"), read("artifacts/release_config.json") or {}
    freeze, cal_final = read("artifacts/production_freeze.json"), read("artifacts/calibration_final.json")
    bundle = manifest["bundle_id"]
    tag = rel_cfg.get("backbone_tag", "imagenet_frozen")
    test, test_f = find(f"e2e_test_{bundle}.json")
    rob, rob_f = find(f"robustness_stress_test_{bundle}.json")
    ana, ana_f = find(f"analysis_test_{bundle}.json")
    tune, tune_f = find(f"e2e_tune_*_{tag}_*.json") if tag != "imagenet_frozen" else find("e2e_tune_dev_*097c99803f.json")
    heads, heads_f = find(f"heads_tune_{tag}.json")
    cls, cls_f = find("classical_tune.json")
    det, det_f = find("detector_tune.json")
    opt, opt_f = find(f"optimization_{tag}.json")
    opth, opth_f = find(f"optimization_head_hybrid_{tag}.json")
    train_sum, train_f = None, None
    for f in sorted(glob.glob(str(ROOT / "reports/training/*_summary.json"))):
        s = json.loads(Path(f).read_text(encoding="utf-8"))
        if f"({s['run_name']}," in rel_cfg.get("backbone_status", ""):
            train_sum, train_f = s, Path(f).name
    final = test is not None

    L = [args.usns if args.usns else "USNS_PENDING_TEAM_TO_PROVIDE (first line must list all member USNs, comma-separated)", ""]
    L += [f"# SEMEVIA comparative evaluation (PS5, {args.team_id})", ""]
    status = [f"Bundle `{bundle}`: YOLO11n detector + {rel_cfg.get('backbone_status', 'unknown backbone')}; "
              f"calibration kind `{rel_cfg.get('calibration_kind', 'unknown')}`."]
    if freeze:
        status.append(f"The production pipeline was frozen in project phase P5 (`artifacts/production_freeze.json`: mode "
                      f"`{freeze.get('production_mode')}`, model fingerprint `{freeze.get('model_fingerprint', '')[:16]}`, "
                      f"bundle `{freeze.get('bundle_id_at_freeze')}`) before any test image was evaluated; the final "
                      f"calibration was then added, which changes the bundle id but not the model fingerprint.")
    status.append("Final values below come from the official Stanford Cars test split (8,041 images), evaluated once "
                  "after the freeze." if final else "Final values read *pending* until the official test run.")
    status.append("Tune-split numbers are selection-set diagnostics and are optimistic.")
    L += ["Status: " + " ".join(status), ""]
    if train_sum:
        unfrozen = sorted({k for h in train_sum["history"] for k, v in (h.get("pretrained_changed") or {}).items() if v})
        stages = sorted({h["stage"] for h in train_sum["history"]}, key=lambda s: [h["stage"] for h in train_sum["history"]].index(s))
        L += [f"Fine-tuning: run `{train_sum['run_name']}` (EfficientNet-B0, ImageNet weights; stages {', '.join(stages)}), "
              f"{train_sum['finetune_epochs_completed']} completed fine-tuning epochs; the final pretrained blocks "
              f"{', '.join(unfrozen)} are verified changed; best tune top-1 {pct(train_sum['best_tune_top1_finetuned'])}; "
              f"fine-tuning requirement met: {train_sum['fine_tuning_requirement_met']} (`reports/training/{train_f}`).", ""]
    if final:
        L += [f"Test protocol: end-to-end on every test image ({test['population']['count']} images, manifest "
              f"`{test['population']['manifest_sha256'][:12]}`); a prediction counts only if the runtime-selected box has "
              f"IoU >= 0.5 with the annotation and the class is right (misses count as failures). Classical mode uses the "
              f"contour-proposal box; deep and hybrid use the YOLO11n primary target. Source `reports/{test_f}`.", ""]

    hist = []
    for f in sorted(glob.glob(str(ROOT / "artifacts/history/*/release_config.json"))):
        rc_old = json.loads(Path(f).read_text(encoding="utf-8"))
        old, old_f = find(f"e2e_test_{rc_old['bundle_id']}.json")
        if old and rc_old["bundle_id"] != bundle:
            hist.append((Path(f).parent.name, rc_old, old, old_f))
    if final and hist:
        L += ["### Model iterations (each evaluated once on test after its own freeze)", "",
              "| Iteration | Backbone | Production mode | Hybrid E2E top-1 | Deep E2E top-1 | Hybrid E2E top-5 | Source |",
              "|---|---|---|---|---|---|---|"]
        for name, rc_old, old, old_f in hist:
            fz_old = read(f"artifacts/history/{name}/production_freeze.json") or {}
            L.append(f"| {name} (superseded) | {rc_old['backbone_status']} | {fz_old.get('production_mode', '?')} | "
                     f"{pct(old['modes']['hybrid']['end_to_end_top1']['value'])} | {pct(old['modes']['deep']['end_to_end_top1']['value'])} | "
                     f"{pct(old['modes']['hybrid']['end_to_end_top5']['value'])} | `reports/{old_f}` |")
        L.append(f"| current | {rel_cfg.get('backbone_status')} | {(freeze or {}).get('production_mode', '?')} | "
                 f"{pct(test['modes']['hybrid']['end_to_end_top1']['value'])} | {pct(test['modes']['deep']['end_to_end_top1']['value'])} | "
                 f"{pct(test['modes']['hybrid']['end_to_end_top5']['value'])} | `reports/{test_f}` |")
        L += ["", "The second iteration changed only the training recipe (full-network instead of last-stage fine-tuning; "
              "`configs/train_v2.yaml` states why), chosen from training and tune behaviour. Every later step (heads, "
              "calibration, freeze) was redone on tune/calibration data before its single test run. Both test results are "
              "reported; all numbers below are for the current iteration.", ""]

    # 1 -----------------------------------------------------------------------------------------
    L += ["## 1. Classification performance (Top-1 and Top-5)", ""]
    if final:
        L += ["Final, official test:", ""] + classification_table(test) + [""]
        tm = test["modes"]
        best_test = max(MODES, key=lambda m: tm[m]["end_to_end_top1"]["value"])
        best_tune = max(MODES, key=lambda m: tune["modes"][m]["end_to_end_top1"]["value"]) if tune else None
        prod = freeze.get("production_mode") if freeze else None
        d1 = 100 * (tm["deep"]["end_to_end_top1"]["value"] - tm["hybrid"]["end_to_end_top1"]["value"])
        d5 = 100 * (tm["deep"]["end_to_end_top5"]["value"] - tm["hybrid"]["end_to_end_top5"]["value"])
        cmp = (f"Comparison: on test the highest end-to-end top-1 is **{best_test}** "
               f"({pct(tm[best_test]['end_to_end_top1']['value'])}); deep minus hybrid is {d1:+.1f} pt top-1 and "
               f"{d5:+.1f} pt top-5. Classical-only is far behind ({pct(tm['classical']['end_to_end_top1']['value'])} top-1): "
               f"its contour box is often wrong (section 2) and handcrafted features alone separate few of the 196 classes.")
        if prod and best_tune:
            cmp += (f" The production mode `{prod}` was fixed on the tune split, where the leader was `{best_tune}`; it was "
                    f"not changed after seeing test results.")
        L += [cmp, ""]
    else:
        L += [f"Final (official test, end-to-end): {PENDING}", ""]
    if tune:
        L += [f"Development, tune split ({tune['population']['count']} images, same protocol; `reports/{tune_f}`):", ""]
        L += classification_table(tune) + [""]
    if heads and cls:
        hd = heads["heads"]
        L += [f"Oracle-crop diagnostic (annotation crops, tune, used for head selection): classical "
              f"{pct(cls['selected']['tune_top1'])} / {pct(cls['selected']['tune_top5'])}; deep "
              f"{pct(hd['deep']['selected']['tune_top1'])} / {pct(hd['deep']['selected']['tune_top5'])}; hybrid "
              f"{pct(hd['hybrid']['selected']['tune_top1'])} / {pct(hd['hybrid']['selected']['tune_top5'])} "
              f"(top-1 / top-5; `reports/{heads_f}`, `reports/{cls_f}`).", ""]

    # 2 -----------------------------------------------------------------------------------------
    L += ["## 2. Object localization (IoU)", ""]
    if final:
        L += ["Final, official test (one-to-one: the runtime-selected box vs the single annotated car):", ""]
        L += localization_table(test) + [""]
    else:
        L += [f"Final: {PENDING}", ""]
    if tune:
        L += [f"Development, tune (`reports/{tune_f}`):", ""] + localization_table(tune) + [""]

    # 3 -----------------------------------------------------------------------------------------
    L += ["## 3. Object detection (mAP)", ""]
    if final:
        L += ["Final, official test: 196-category COCO AP (pycocotools bbox), at most one classified prediction per image, "
              "score = localisation score x top-1 probability:", ""] + ap_table(test) + [""]
    else:
        L += [f"Final: {PENDING}", ""]
    if det:
        a = det["generic_vehicle_ap"]
        L += [f"Generic vehicle AP (all YOLO11n car/bus/truck detections as one category), tune only (the detector "
              f"evaluator does not run on test): AP50 {a['AP50']:.3f}, AP50:95 {a['AP50_95']:.3f} (`reports/{det_f}`). "
              f"Unannotated background cars count as false positives.", ""]

    # 4 -----------------------------------------------------------------------------------------
    L += ["## 4. Model efficiency (model size and inference time)", ""]
    sizes = {a["name"]: a["bytes"] for a in manifest["assets"] if "bytes" in a}
    if sizes:
        L += [f"Release bundle files (`artifacts/release_manifest.json`, bundle `{bundle}`):", "",
              "| File | Size |", "|---|---|"] + [f"| {n} | {mb(b)} |" for n, b in sizes.items()]
        L += [f"| **total** | **{mb(sum(sizes.values()))}** |", ""]
    if final and "latency_ms" in test:
        lt = test["latency_ms"]
        L += [f"Offline test-run timing ({lt['hardware']}; {lt['note']}): detector p50 {lt['detector_p50']:.1f} ms, "
              f"p95 {lt['detector_p95']:.1f} ms per image; features + heads {lt['features_and_heads_per_crop_mean_batched']:.1f} "
              f"ms per crop (mean, batched).", ""]
    if opt:
        s, lt = opt["size_bytes"], opt["latency_single_crop_encoder_only"]
        pop = opt.get("population", {})
        L += [f"Optimization of the fine-tuned encoder (`reports/{opt_f}`). Latency: encoder forward only, batch 1, warm, "
              f"{lt['hardware']}. Accuracy: hybrid head on {pop.get('count')} {pop.get('partition')} "
              f"{pop.get('crops')} (oracle) crops, so not end-to-end:", "",
              "| Runtime | Size | p50 | p95 | Hybrid top-1 (tune, annotation crops) |", "|---|---|---|---|---|"]
        for n, key in (("PyTorch FP32 (released)", "torch_fp32"), ("ONNX FP32", "onnx_fp32"),
                       ("ONNX INT8 static (MinMax)", "onnx_int8"), ("ONNX INT8 static (percentile)", "onnx_int8_percentile"),
                       ("ONNX INT8 dynamic", "onnx_int8_dynamic")):
            size = s.get("torch_state_dict_pth" if key == "torch_fp32" else key)
            if key in lt and key in opt["accuracy"] and size:
                L.append(f"| {n} | {mb(size)} | {lt[key]['p50_ms']:.1f} ms | {lt[key]['p95_ms']:.1f} ms | "
                         f"{pct(opt['accuracy'][key]['hybrid']['tune_top1'])} |")
        L += ["", f"Selection policy: {opt['selection_policy']}. Encoder INT8 variants passing it: "
              f"{', '.join(k for k, v in opt['int8_passes_policy'].items() if v) or 'none'}. ONNX FP32 keeps the same "
              f"tune accuracy at lower encoder latency.", ""]
    if opth:
        L += [f"INT8 linear head (`reports/{opth_f}`): the hybrid head's final linear layer exported to ONNX, "
              f"{opth['size_bytes']['onnx_fp32'] / 1e6:.2f} MB (FP32) -> {opth['size_bytes']['onnx_int8'] / 1e6:.2f} MB (INT8), "
              f"tune top-1 {pct(opth['accuracy']['onnx_fp32']['tune_top1'])} -> {pct(opth['accuracy']['onnx_int8']['tune_top1'])} "
              f"(passes the policy: {opth.get('int8_passes_policy')}). {opth.get('note', '')}", ""]
    if freeze:
        L += [f"Released and frozen runtime: encoder `{freeze.get('encoder_runtime')}`, precision `{freeze.get('precision')}` "
              f"(the optimized variants above are measured, not deployed).", ""]
    acc = read("reports/release_acceptance.json")
    if acc and acc.get("passed"):
        fu, cc = acc["fresh_uploads"], acc["concurrent"]
        L += [f"Deployed end-to-end latency through the public URL (`reports/release_acceptance.json`, {acc['created_at'][:10]}; "
              f"{acc['host_note']}): {fu['count']} fresh uploads, client wall time including the network, p50 "
              f"{fu['p50_ms']:.0f} ms, p95 {fu['p95_ms']:.0f} ms, max {fu['max_ms']:.0f} ms; {cc['clients']} simultaneous "
              f"clients all answered in {cc['wall_ms']:.0f} ms; {len(acc['bad_inputs'])} bad-input cases answered with "
              f"typed errors.", ""]
    else:
        L += ["Deployed end-to-end latency (p50/p95 over fresh uploads on the public host): pending (no passing "
              "`reports/release_acceptance.json`).", ""]

    # 5 -----------------------------------------------------------------------------------------
    L += ["## 5. Robustness under varying conditions", ""]
    if rob:
        L += [f"Final: {rob['count']}-image predeclared test stress subset (seeded, stratified, up to 2 images per class), "
              f"conditions applied to stored pixels before detection (`reports/{rob_f}`). Caveat from the report: "
              f"{rob.get('note', '')}", "",
              "| Condition | Mode | E2E top-1 | Change vs clean | Box recall@0.5 |", "|---|---|---|---|---|"]
        for r in rob["rows"]:
            L.append(f"| {r['condition']} | {r['mode']} | {pct(r['e2e_top1'])} | {100 * r['delta_e2e_top1_vs_clean']:+.1f} pt | "
                     f"{pct(r['loc_recall50'])} |")
        L.append("")
    else:
        L += [f"Final (test stress subset): {PENDING}", ""]
    vp = (ana or {}).get("viewpoint_subgroups")
    if vp:
        n_vp = sum(c["count"] for c in next(iter(vp.values())).values())
        L += [f"Natural viewpoint subgroups: {n_vp} seeded random test photos (disjoint from the stress subset), viewpoint "
              f"labelled by an agent before any test result existed, team review pending (`reports/viewpoint_labels.csv`; "
              f"`reports/{ana_f}`). {ana.get('viewpoint_note', '')}", "",
              "| Viewpoint | Photos | " + " | ".join(f"{m} E2E top-1 | {m} E2E top-5" for m in vp) + " |",
              "|---|---|" + "---|---|" * len(vp)]
        for g in next(iter(vp.values())):
            cells = [f"{pct(vp[m][g]['top1_rate'])} ({vp[m][g]['top1']}) | {pct(vp[m][g]['top5_rate'])} ({vp[m][g]['top5']})"
                     for m in vp]
            L.append(f"| {g} | {next(iter(vp.values()))[g]['count']} | " + " | ".join(cells) + " |")
        L.append("")
    else:
        L += ["Natural viewpoint subgroups (labelled test photos): pending.", ""]

    # 6 -----------------------------------------------------------------------------------------
    L += ["## 6. Confidence and threshold analysis", ""]
    if final:
        p = test["passport_production_mode"]
        L += [f"Final, official test, production mode `{p['mode']}`, calibration `{p['calibration_kind']}`"
              + (f" (temperature {cal_final['temperature']:.3f} from the frozen tune fit, conformal quantile q "
                 f"{cal_final['q']:.4f} from the untouched {cal_final['n']}-image calibration partition, alpha "
                 f"{cal_final['alpha']})" if cal_final else "") + ":", ""] + passport_lines(p)
        rj = (freeze or {}).get("reject")
        if rj:
            L += ["", f"Threshold choice (`artifacts/production_freeze.json`, tune known images, before the test run): rule "
                  f"'{rj['rule']}', target known retention {pct(rj['known_retention_target'])}, chosen threshold "
                  f"{rj['threshold']}.", "", "| Threshold | Tune known retention |", "|---|---|"]
            L += [f"| {r['threshold']} | {pct(r['known_retention'])} |" for r in rj["table"]]
            L.append("")
    else:
        L += [f"Final: {PENDING}"]
    neg, neg_f = find(f"negatives_test_{bundle}.json")
    if neg:
        o = neg["overall_false_acceptance"]
        L += ["", f"Unsupported inputs (held-out negative suite, {neg['count']} licensed Wikimedia Commons photos, evaluated "
              f"after the freeze; `reports/{neg_f}`, sources in `artifacts/negative_manifests/negatives_test.csv`). A photo is "
              f"falsely accepted when the system returns a confident car identity (calibrated max probability >= "
              f"{neg['reject_threshold']}); a detector miss counts as a correct rejection. {neg.get('note', '')}", "",
              "| Bucket | Photos | Reached classifier | Falsely accepted | False acceptance |", "|---|---|---|---|---|"]
        for b, v in neg["per_bucket"].items():
            L.append(f"| {b} | {v['count']} | {v['reached_classifier']} | {v['accepted']} | {pct(v['false_acceptance'])} |")
        L += [f"| **all** | {o['denominator']} | {sum(v['reached_classifier'] for v in neg['per_bucket'].values())} | "
              f"{o['numerator']} | {pct(o['numerator'] / o['denominator'])} |", ""]
        if freeze and (freeze.get("reject") or {}).get("negatives_used"):
            L += ["The reject threshold was chosen on the separate 40-photo tune negative set together with known tune "
                  "images (`artifacts/production_freeze.json`).", ""]
    else:
        L += ["* Negative (unsupported-input) false acceptance: pending (negative suite not evaluated for this bundle).", ""]

    # Ablations / failures ------------------------------------------------------------------------
    L += ["## Ablations", ""]
    if heads:
        import yaml

        hcfg = yaml.safe_load((ROOT / "configs/heads.yaml").read_text(encoding="utf-8"))["learned"]
        specs = {**hcfg["heads"], **hcfg["ablations"]}
        L += [f"Component ablations (tune, annotation crops; every variant refit on train with the same C grid and selected "
              f"by tune top-1; `reports/{heads_f}`, groups from `configs/heads.yaml`). The handcrafted branch is reduced by "
              f"PCA to at most {hcfg['pca_components']} components, so removing a group can leave the input size unchanged.", "",
              "| Variant | Handcrafted groups | Branch balance | Input dim | C | Tune top-1 | Tune top-5 | Tune NLL |",
              "|---|---|---|---|---|---|---|---|"]
        for name, h in heads["heads"].items():
            s, sp = h["selected"], specs.get(name, {})
            L.append(f"| {name} | {', '.join(sp.get('hc_groups', [])) or 'none'} | {sp.get('branch_balance', '-')} | "
                     f"{s['input_dim']} | {s['C']} | {pct(s['tune_top1'])} | {pct(s['tune_top5'])} | {s['tune_nll']:.3f} |")
        if cls:
            L.append(f"| classical only (no learned features) | all | - | - | {cls['selected']['C']} | "
                     f"{pct(cls['selected']['tune_top1'])} | {pct(cls['selected']['tune_top5'])} | {cls['selected']['tune_nll']:.3f} |")
        dev, dev_f = find("heads_tune_imagenet_frozen.json")
        if dev:
            ds = dev["heads"]["hybrid"]["selected"]
            L.append(f"| hybrid on the frozen ImageNet backbone (no fine-tuning; `reports/{dev_f}`) | as hybrid | per_feature | "
                     f"{ds['input_dim']} | {ds['C']} | {pct(ds['tune_top1'])} | {pct(ds['tune_top5'])} | {ds['tune_nll']:.3f} |")
        hd = heads["heads"]
        base = hd["hybrid"]["selected"]["tune_top1"]
        deltas = "; ".join(f"{n}: {100 * (hd[n]['selected']['tune_top1'] - base):+.1f} pt" for n in hd if n != "hybrid")
        ft_gain = (f"; fine-tuning the backbone: {100 * (base - dev['heads']['hybrid']['selected']['tune_top1']):+.1f} pt "
                   f"(hybrid, fine-tuned vs frozen)") if dev else ""
        L += ["", f"Change in tune top-1 versus the full hybrid head: {deltas}{ft_gain}. Reading: fine-tuning gives the "
              "largest gain; with the fine-tuned features, removing any single handcrafted group changes tune top-1 by well "
              "under one point, so the handcrafted branch adds little accuracy here, while equal-variance branch weighting "
              "clearly hurts.", ""]
    L += ["## Failure analysis", ""]
    fails = (ana or {}).get("failures")
    if fails:
        L += [f"Selected by fixed rules from the test predictions (`reports/{ana_f}`). Image sizes from "
              f"`artifacts/manifests/test_m1.csv`; duplicate lists from `reports/data_audit.json`.", ""]
        mrows = manifest_rows("test_m1.csv")
        audit = read("reports/data_audit.json") or {}
        dup_text = json.dumps(audit.get("duplicates", {}))
        for i, f in enumerate(fails, 1):
            rt = f.get("runtime_top5") or []
            at = f.get("annotation_crop_top5") or []
            box = ("no vehicle box selected" if f.get("selected_box") is None else
                   f"selected box IoU {f['iou']:.3f}, detector score {f['detector_score']:.3f}")
            L += [f"{i}. **{f['rule']}**: `{f['image_key']}`, true class *{f['true']}*; {box}.",
                  f"   * Runtime top-1: " + (f"*{rt[0][0]}* (p = {rt[0][1]:.4f})" if rt else "none (no crop)") +
                  f"; annotation-crop top-1: *{at[0][0]}* (p = {at[0][1]:.4f}), "
                  f"{'correct' if f['annotation_crop_correct'] else 'still wrong'}.",
                  f"   * {f['hypothesis']}"]
            mr = mrows.get(f["image_key"])
            ab = f["annotation_box"]
            why = []
            if mr:
                W, H = int(mr["width"]), int(mr["height"])
                bw, bh = ab[2] - ab[0], ab[3] - ab[1]
                why.append(f"image {W}x{H} px, annotated car {bw:.0f}x{bh:.0f} px ({100 * bw * bh / (W * H):.0f}% of the image)")
            names = [c for c, _ in rt]
            if rt:
                why.append("true class " + (f"ranked {names.index(f['true']) + 1} in the runtime top-5" if f["true"] in names
                                            else "not in the runtime top-5"))
                same_make = rt[0][0].split(" ")[0] == f["true"].split(" ")[0]
                why.append("prediction is a sibling model of the same make" if same_make else "prediction is a different make")
            why.append("not listed as a duplicate in the data audit" if f["image_key"] not in dup_text
                       else "listed in the data audit's duplicate groups")
            if f.get("selected_box") is None and mr and max(int(mr["width"]), int(mr["height"])) < 200:
                why.append("likely cause: the whole photo is a small thumbnail, so the car has too little detail for "
                           "YOLO11n to produce a vehicle box above its score floor")
            elif rt and not f["annotation_crop_correct"]:
                why.append("likely cause: classifier confusion, not localisation (the box is good and the annotation crop gives "
                           "the same wrong answer); the very high probability shows over-confidence on visually close "
                           "classes even after temperature scaling")
            L.append("   * Evidence: " + "; ".join(why) + ".")
        L.append("")
    else:
        L += ["Failure cases: pending (selected from final test predictions in P6).", ""]

    out = ROOT / f"eval_P5_{args.team_id}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.name} ({len(L)} lines; final={'yes' if final else 'no'})")
    if args.readme and final:
        write_readme_block(test, test_f, neg, neg_f, acc, out.name, bundle)


def write_readme_block(test, test_f, neg, neg_f, acc, eval_name, bundle) -> None:
    """Replace the README block between the RESULTS markers with a table generated from the reports."""
    readme = ROOT / "README.md"
    t = readme.read_text(encoding="utf-8")
    a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if a not in t or b not in t:
        print("README has no RESULTS markers; skipped")
        return
    B = [a, f"Generated by `scripts/report.py` from `reports/{test_f}` (official test split, "
            f"{test['population']['count']} images, bundle `{bundle}`). Full results: [{eval_name}]({eval_name}).", "",
         "| Mode | End-to-end top-1 | End-to-end top-5 | Box recall @ IoU 0.5 | 196-class AP50 |", "|---|---|---|---|---|"]
    for m in MODES:
        v = test["modes"][m]
        B.append(f"| {m} | {pct(v['end_to_end_top1']['value'])} | {pct(v['end_to_end_top5']['value'])} | "
                 f"{pct(v['localization']['primary_recall_at_iou']['value'])} | {v['fine_grained_ap']['AP50']:.3f} |")
    extra = []
    if neg:
        o = neg["overall_false_acceptance"]
        extra.append(f"unsupported-input false acceptance {pct(o['numerator'] / o['denominator'])} "
                     f"({o['numerator']}/{o['denominator']} held-out negatives, `reports/{neg_f}`)")
    if acc and acc.get("passed"):
        extra.append(f"public-URL latency p50 {acc['fresh_uploads']['p50_ms']:.0f} ms / p95 "
                     f"{acc['fresh_uploads']['p95_ms']:.0f} ms (`reports/release_acceptance.json`)")
    if extra:
        B += ["", "Also: " + "; ".join(extra) + "."]
    B.append(b)
    t = t[: t.index(a)] + "\n".join(B) + t[t.index(b) + len(b):]
    readme.write_text(t, encoding="utf-8", newline="\n")
    print("README results block updated")


if __name__ == "__main__":
    main()
