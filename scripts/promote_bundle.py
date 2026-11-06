"""Promote selected heads + backbone into the release folder and record them in the manifest.

Usage: python scripts/promote_bundle.py --backbone imagenet_frozen
Copies the heads named in reports/heads_tune_<backbone>.json and reports/classical_tune.json
from the artifact store to artifacts/store/release/heads/<mode>.joblib, and the backbone weights
to the configured path. Entries are written as optional assets (recognition is disabled, not
the whole app, when they are missing). Assets without a public URL cannot be fetched by a clean
Docker build; this is recorded in each entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.models.cache_io import store_dir  # noqa: E402

IMAGENET_URL = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet_frozen")
    args = ap.parse_args()
    rc = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text(encoding="utf-8"))["recognition"]
    learned = json.loads((ROOT / f"reports/heads_tune_{args.backbone}.json").read_text(encoding="utf-8"))
    classical = json.loads((ROOT / "reports/classical_tune.json").read_text(encoding="utf-8"))
    src = {"classical": classical["artifact"], "deep": learned["artifacts"]["deep"], "hybrid": learned["artifacts"]["hybrid"]}
    entries = []
    for mode, art in src.items():
        s = store_dir() / art["path"]
        if sha(s) != art["sha256"]:
            raise SystemExit(f"{mode}: store file does not match its report hash")
        dst = ROOT / rc["heads"][mode]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, dst)
        entries.append({"name": f"head_{mode}.joblib", "path": rc["heads"][mode], "sha256": art["sha256"],
                        "bytes": dst.stat().st_size, "url": None, "optional": True,
                        "source": f"{art['path']} (fit_heads.py, backbone {args.backbone})",
                        "public_storage": "not yet published; clean builds run detection-only until a public URL is added"})
    bb_dst = ROOT / rc["backbone"]
    if args.backbone == "imagenet_frozen":
        th = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")) / "hub/checkpoints" / bb_dst.name
        if not bb_dst.exists():
            shutil.copyfile(th, bb_dst)
        entries.append({"name": bb_dst.name, "path": rc["backbone"], "sha256": sha(bb_dst), "bytes": bb_dst.stat().st_size,
                        "url": IMAGENET_URL, "optional": True,
                        "license": "torchvision ImageNet-1K weights (BSD-3 code licence)",
                        "status": "development backbone: frozen ImageNet, not fine-tuned"})
    onnx_enc = ROOT / rc["onnx_encoder"]
    if onnx_enc.exists():
        info = json.loads(onnx_enc.with_suffix(".json").read_text(encoding="utf-8"))
        if not info.get("parity_ok") or info["sha256"] != sha(onnx_enc) or info["source_backbone_sha256"] != sha(bb_dst):
            raise SystemExit("ONNX encoder does not match its parity record or the current backbone; re-export it")
        entries.append({"name": onnx_enc.name, "path": rc["onnx_encoder"], "sha256": info["sha256"], "bytes": info["bytes"],
                        "url": None, "optional": True,
                        "source": "python -m src.models.export_onnx (parity-gated export of the backbone above)",
                        "parity_max_abs_diff": info["parity_max_abs_diff"]})
    ref = ROOT / rc["reference_index"]
    for p in (ref, ref.with_suffix(".json")):
        if p.exists():
            entries.append({"name": p.name, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size,
                            "url": None, "optional": True, "source": "scripts/build_references.py (training split only)"})
    cal = ROOT / rc["calibration"]
    if cal.exists():
        entries.append({"name": "development_calibration.json", "path": rc["calibration"], "sha256": sha(cal),
                        "bytes": cal.stat().st_size, "url": None, "optional": True, "in_git": True})
    mpath = ROOT / "artifacts/release_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    keep = [a for a in manifest["assets"] if a["name"] == "yolo11n.pt"]
    manifest["assets"] = keep + entries
    manifest["bundle_id"] = "dev_" + args.backbone + "_" + hashlib.sha256(
        json.dumps([a["sha256"] for a in manifest["assets"]]).encode()).hexdigest()[:10]
    manifest["note"] = ("Development bundle: YOLO11n detector (final) + frozen ImageNet EfficientNet-B0 + classical/deep/hybrid "
                        "heads + development calibration. Not the fine-tuned release.")
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"bundle_id": manifest["bundle_id"], "assets": [(a["name"], a["sha256"][:12]) for a in manifest["assets"]]}, indent=1))


if __name__ == "__main__":
    main()
