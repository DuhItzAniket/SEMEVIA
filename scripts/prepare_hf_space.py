"""Assemble a Hugging Face Docker Space folder from this repository.

Usage: python scripts/prepare_hf_space.py <out_dir>

Copies only what the Dockerfile needs (Dockerfile, requirements-infer.txt, src, app, configs, artifacts without
the local store, reports) and writes the Space README with the YAML header Hugging Face requires. Model files are
not copied: the image build downloads them from the URLs in artifacts/release_manifest.json and checks SHA256.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ["Dockerfile", ".dockerignore", "requirements-infer.txt", "requirements-derive.txt", "src", "app", "configs", "artifacts", "reports"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", "store", "cache")

README = """---
title: SEMEVIA
emoji: 🚗
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Fine-grained car recognition with an evidence passport (PS5)
---

# SEMEVIA: vehicle identity through evidence

Upload a car photo: YOLO11n finds the vehicle, a fine-tuned EfficientNet-B0 with classical features names the
make, model and year among the 196 Stanford Cars categories, and the Identity Evidence Passport shows a calibrated
candidate set and how sure the system is.

Source code, design, architecture and evaluation: https://github.com/DuhItzAniket/SEMEVIA
(branch `semevia/implementation`; results in `eval_P5_T35.md`). Release bundle `{bundle}`.

Team T35, Computer Vision Hackathon 2026, problem statement PS5.
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1]).resolve()
    if out == ROOT or ROOT in out.parents:
        raise SystemExit("choose an output folder outside the repository")
    manifest = json.loads((ROOT / "artifacts/release_manifest.json").read_text(encoding="utf-8"))
    by_name = {a["name"]: a for a in manifest["assets"]}
    # a derived asset (the INT8 head) is rebuilt during the image build; its source must be downloadable
    missing = [a["name"] for a in manifest["assets"] if not a.get("url") and not a.get("in_git")
               and not (a.get("derived_from") and by_name.get(a["derived_from"], {}).get("url"))]
    if missing:
        raise SystemExit(f"publish these assets first (no URL): {missing}")
    out.mkdir(parents=True, exist_ok=True)
    for item in ITEMS:
        src, dst = ROOT / item, out / item
        if dst.is_dir():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=SKIP)
        else:
            shutil.copy2(src, dst)
    (out / "README.md").write_text(README.replace("{bundle}", manifest["bundle_id"]), encoding="utf-8", newline="\n")
    print(f"Space folder ready: {out} (bundle {manifest['bundle_id']})")


if __name__ == "__main__":
    main()
