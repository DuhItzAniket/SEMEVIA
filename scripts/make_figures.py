"""Draw the report figures from the saved report files (no number is typed in): reports/figures/*.png.

Usage: python scripts/make_figures.py
Figures: model iterations on test, training curves on tune, robustness on the test stress subset, reject-threshold
trade-off at the freeze, negative-suite outcomes, and component ablations. Iterations are discovered from
artifacts/history/*/ plus the current release.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "reports/figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3 (light mode, all pairs)
MODES = ("classical", "deep", "hybrid")


def read(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def style(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, name, source):
    fig.text(0.01, 0.01, f"Source: {source}", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def iterations():
    """[(label, bundle_id, release_config, freeze)] oldest first, current last."""
    its = []
    for f in sorted(glob.glob(str(ROOT / "artifacts/history/*/release_config.json"))):
        rc = json.loads(Path(f).read_text(encoding="utf-8"))
        fz = read(f"artifacts/history/{Path(f).parent.name}/production_freeze.json")
        name = Path(f).parent.name
        its.append((name, rc["bundle_id"], rc, fz))
    rc = read("artifacts/release_config.json")
    run = rc["backbone_status"].split("(")[1].split(",")[0] if "(" in rc["backbone_status"] else "current"
    run = run.replace("effb0_", "")
    if rc.get("hybrid_head_runtime") == "onnx_int8":   # same backbone as the archived FP32 iteration: name both heads
        run += "-int8"
        its = [(f"{n} FP32 head" if n == run[:-5] else n, b, r, z) for n, b, r, z in its]
    its.append((f"{run} (released)", rc["bundle_id"], rc, read("artifacts/production_freeze.json")))
    return its[-3:]


def bar_labels(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)


def fig_iterations(its):
    rows = [(lab, read(f"reports/e2e_test_{b}.json")) for lab, b, _, _ in its]
    rows = [(lab, r) for lab, r in rows if r]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.8 / len(rows)
    for i, (lab, r) in enumerate(rows):
        vals = [100 * r["modes"][m]["end_to_end_top1"]["value"] for m in MODES]
        bars = ax.bar([x + (i - (len(rows) - 1) / 2) * w for x in range(3)], vals, w * 0.92, label=lab, color=SERIES[i])
        bar_labels(ax, bars)
    ax.set_xticks(range(3), [m.capitalize() for m in MODES])
    ax.set_ylim(0, 105)
    style(ax, "End-to-end top-1 on the official test split (8,041 images), per model iteration", "End-to-end top-1 (%)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, "iterations_test_top1.png", ", ".join(f"reports/e2e_test_{b}.json" for _, b, _, _ in its))


def fig_training():
    runs = sorted(glob.glob(str(ROOT / "reports/training/effb0_r*_summary.json")))[-3:]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, f in enumerate(runs):
        s = json.loads(Path(f).read_text(encoding="utf-8"))
        ep = [h["global_epoch"] for h in s["history"]]
        t1 = [100 * h["tune"]["top1"] for h in s["history"]]
        ax.plot(ep, t1, color=SERIES[i], linewidth=2, marker="o", markersize=4, label=s["run_name"])
        ax.annotate(f"{s['run_name']} {t1[-1]:.1f}%", (ep[-1], t1[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK, va="center")
    style(ax, "Tune top-1 of the network's own classifier, per training epoch", "Tune top-1 (%)")
    ax.set_xlabel("Epoch (head stage, then fine-tuning)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    save(fig, "training_tune_top1.png", ", ".join("reports/training/" + Path(f).name for f in runs))


def fig_robustness(its):
    rows = [(lab, read(f"reports/robustness_stress_test_{b}.json")) for lab, b, _, _ in its]
    rows = [(lab, r) for lab, r in rows if r]
    if not rows:
        return
    conds = list(rows[0][1]["conditions"].keys())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(rows)
    for i, (lab, r) in enumerate(rows):
        by = {x["condition"]: 100 * x["e2e_top1"] for x in r["rows"] if x["mode"] == "hybrid"}
        bars = ax.bar([x + (i - (len(rows) - 1) / 2) * w for x in range(len(conds))], [by[c] for c in conds], w * 0.92,
                      label=lab, color=SERIES[i])
        bar_labels(ax, bars, "{:.0f}")
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylim(0, 105)
    style(ax, f"Hybrid end-to-end top-1 under fixed stress conditions ({rows[0][1]['count']}-image test subset)",
          "End-to-end top-1 (%)")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, "robustness_hybrid.png", ", ".join(f"reports/robustness_stress_test_{b}.json" for _, b, _, _ in its))


def fig_threshold():
    fz = read("artifacts/production_freeze.json")
    rj = (fz or {}).get("reject")
    if not rj:
        return
    th = [r["threshold"] for r in rj["table"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(th, [100 * r["known_retention"] for r in rj["table"]], color=SERIES[0], linewidth=2, marker="o",
            markersize=4, label="Known tune images kept")
    if rj.get("negatives_used"):
        ax.plot(th, [100 * r["negative_false_acceptance"] for r in rj["table"]], color=SERIES[1], linewidth=2,
                marker="o", markersize=4, label="Tune negatives falsely accepted")
    ax.axvline(rj["threshold"], color=INK2, linewidth=1, linestyle="--")
    ax.annotate(f"chosen {rj['threshold']}", (rj["threshold"], 50), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK)
    ax.axhline(100 * rj["known_retention_target"], color=GRID, linewidth=1)
    ax.annotate(f"retention target {100 * rj['known_retention_target']:.0f}%", (th[0], 100 * rj["known_retention_target"]),
                xytext=(0, -12), textcoords="offset points", fontsize=8, color=INK2)
    style(ax, "Reject threshold chosen at the freeze (tune split only)", "Share of images (%)")
    ax.set_xlabel("Threshold on the calibrated max probability", color=INK2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    save(fig, "reject_threshold.png", "artifacts/production_freeze.json")


def fig_negatives():
    rc = read("artifacts/release_config.json")
    neg = read(f"reports/negatives_test_{rc['bundle_id']}.json")
    if not neg:
        return
    buckets = list(neg["per_bucket"].keys())
    det = [neg["per_bucket"][b]["rejected_by_detector"] for b in buckets]
    acc = [neg["per_bucket"][b]["accepted"] for b in buckets]
    thr = [neg["per_bucket"][b]["count"] - d - a for b, d, a in zip(buckets, det, acc)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(buckets))
    left = [0] * len(buckets)
    for i, (vals, lab) in enumerate(((det, "Rejected: no car/bus/truck detected"), (thr, "Rejected: below reject threshold"),
                                     (acc, "Falsely accepted"))):
        ax.barh(y, vals, 0.6, left=left, color=SERIES[i], label=lab, edgecolor=SURFACE, linewidth=2)
        for k, v in enumerate(vals):
            if v:
                ax.text(left[k] + v / 2, k, str(v), ha="center", va="center", fontsize=8, color=SURFACE if i != 2 else INK)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(list(y), [b.replace("_", " ") for b in buckets])
    ax.invert_yaxis()
    style(ax, f"Held-out unsupported photos ({neg['count']}): what stopped each one", None)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Photos", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    save(fig, "negatives_test.png", f"reports/negatives_test_{rc['bundle_id']}.json")


def fig_ablation():
    rc = read("artifacts/release_config.json")
    h = read(f"reports/heads_tune_{rc['backbone_tag']}.json")
    cls = read("reports/classical_tune.json")
    dev = read("reports/heads_tune_imagenet_frozen.json")
    if not h:
        return
    rows = [(n.replace("_", " "), 100 * v["selected"]["tune_top1"]) for n, v in h["heads"].items()]
    if dev:
        rows.append(("hybrid, frozen ImageNet backbone", 100 * dev["heads"]["hybrid"]["selected"]["tune_top1"]))
    if cls:
        rows.append(("classical only", 100 * cls["selected"]["tune_top1"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(range(len(rows)), [v for _, v in rows], 0.6, color=SERIES[0])
    for b in bars:
        ax.text(b.get_width() + 0.8, b.get_y() + b.get_height() / 2, f"{b.get_width():.1f}%", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    style(ax, "Component ablations: tune top-1 on annotation crops", None)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Tune top-1 (%)", color=INK2, fontsize=9)
    save(fig, "ablations_tune.png", f"reports/heads_tune_{rc['backbone_tag']}.json, reports/heads_tune_imagenet_frozen.json, reports/classical_tune.json")


def fig_reliability():
    rc = read("artifacts/release_config.json")
    ex = read(f"reports/extra_test_{rc['bundle_id']}.json")
    if not ex:
        return
    rl = ex["reliability"]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 100], [0, 100], color=GRID, linewidth=1.5, linestyle="--")
    ax.annotate("perfect calibration", (62, 58), color=INK2, fontsize=8, rotation=38)
    for i, (key, lab) in enumerate((("raw", "Raw softmax"), ("calibrated", "Temperature-scaled (released)"))):
        bins = [b for b in rl[key]["bins"] if b["count"]]
        ax.plot([100 * b["mean_confidence"] for b in bins], [100 * b["accuracy"] for b in bins], color=SERIES[i], linewidth=2,
                marker="o", markersize=5, label=f"{lab}  ECE {rl[key]['ece']:.3f}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    style(ax, f"Reliability on {rl['n']} correctly localised test images", "Accuracy in bin (%)")
    ax.set_xlabel("Mean confidence in bin (%)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper left")
    save(fig, "reliability_test.png", f"reports/extra_test_{rc['bundle_id']}.json")


def fig_confusions():
    rc = read("artifacts/release_config.json")
    ex = read(f"reports/extra_test_{rc['bundle_id']}.json")
    if not ex:
        return
    cf = ex["confusions"]
    rows = cf["top_pairs"][:10][::-1]
    kinds = list(dict.fromkeys(r["kind"] for r in cf["top_pairs"][:10]))
    colour = {k: SERIES[i % 3] for i, k in enumerate(kinds)}
    fig, ax = plt.subplots(figsize=(9, 5.4))
    bars = ax.barh(range(len(rows)), [r["count"] for r in rows], 0.6, color=[colour[r["kind"]] for r in rows])
    for b in bars:
        ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2, f"{int(b.get_width())}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), [f"{r['true']}  →  {r['predicted']}" for r in rows], fontsize=8)
    style(ax, "Most frequent mix-ups on the test split", None)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=colour[k], label=k) for k in kinds], frameon=False, fontsize=8, labelcolor=INK,
              loc="upper center", bbox_to_anchor=(0.35, -0.12), ncol=len(kinds))
    ax.set_xlabel(f"Test images ({cf['wrong']} wrong top-1 among correctly localised images)", color=INK2, fontsize=9)
    save(fig, "confusions_test.png", f"reports/extra_test_{rc['bundle_id']}.json")


def main() -> None:
    its = iterations()
    fig_reliability()
    fig_confusions()
    fig_iterations(its)
    fig_training()
    fig_robustness(its)
    fig_threshold()
    fig_negatives()
    fig_ablation()


if __name__ == "__main__":
    main()
