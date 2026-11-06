"""Assemble the robustness table from per-condition end-to-end reports.

Usage: python scripts/report_robustness.py --subset stress_tune
Reads reports/e2e_<subset>_<condition>_<bundle>.json for every configured condition (all must
exist for the same bundle and subset manifest) and writes reports/robustness_<subset>_<bundle>.json
plus a Markdown table. Each condition is compared with the clean result on the same images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

MODES = ("classical", "deep", "hybrid")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="stress_tune")
    args = ap.parse_args()
    ecfg = yaml.safe_load((ROOT / "configs/evaluation.yaml").read_text(encoding="utf-8"))
    bundle = json.loads((ROOT / "artifacts/release_manifest.json").read_text(encoding="utf-8"))["bundle_id"]
    reps = {}
    for c in ecfg["conditions"]:
        p = ROOT / "reports" / f"e2e_{args.subset}_{c}_{bundle}.json"
        if not p.exists():
            raise SystemExit(f"missing condition report {p.name}")
        reps[c] = json.loads(p.read_text(encoding="utf-8"))
    shas = {r["population"]["manifest_sha256"] for r in reps.values()}
    counts = {r["population"]["count"] for r in reps.values()}
    if len(shas) != 1 or len(counts) != 1:
        raise SystemExit("condition reports cover different populations")
    rows = []
    clean = reps["clean"]["modes"]
    for c, r in reps.items():
        for m in MODES:
            mm = r["modes"][m]
            row = {"condition": c, "mode": m,
                   "e2e_top1": mm["end_to_end_top1"]["value"], "e2e_top5": mm["end_to_end_top5"]["value"],
                   "loc_recall50": mm["localization"]["primary_recall_at_iou"]["value"],
                   "missing_box": mm["localization"]["missing"],
                   "matched_top1": mm["matched_primary_top1"]["value"]}
            row["delta_e2e_top1_vs_clean"] = row["e2e_top1"] - clean[m]["end_to_end_top1"]["value"]
            rows.append(row)
    out = {"subset": args.subset, "bundle_id": bundle, "count": counts.pop(), "manifest_sha256": shas.pop(),
           "conditions": {c: ecfg["conditions"][c] for c in reps},
           "note": "Synthetic, fixed conditions applied before detection (design choices, not a standard benchmark). "
                   "Artificial darkening is not real night imagery; downsampling is not distance.",
           "rows": rows}
    name = f"robustness_{args.subset}_{bundle}"
    (ROOT / "reports" / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    lines = [f"# Robustness on `{args.subset}` ({out['count']} images, bundle `{bundle}`)", "",
             out["note"], "", "| Condition | Mode | E2E top-1 | Change vs clean | E2E top-5 | Box recall@0.5 | No box |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['mode']} | {r['e2e_top1']:.3f} | {r['delta_e2e_top1_vs_clean']:+.3f} | "
                     f"{r['e2e_top5']:.3f} | {r['loc_recall50']:.3f} | {r['missing_box']} |")
    (ROOT / "reports" / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
