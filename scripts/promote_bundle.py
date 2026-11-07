"""Promote selected heads + backbone into the release folder and record them in the manifest.

Usage:
  python scripts/promote_bundle.py --backbone imagenet_frozen
  python scripts/promote_bundle.py --backbone D:/semevia_work/store/runs/effb0_r1/best_finetuned.pt \
      [--calibration artifacts/calibration_final.json]

Heads come from reports/heads_tune_<tag>.json and reports/classical_tune.json (hash-checked against
the artifact store). A training checkpoint is reduced to its model state_dict and written to
artifacts/store/release/backbone_<tag>.pth; its tag (ckpt_<sha12> of the checkpoint file) matches the
cache/head tag produced by scripts/cache_features.py. artifacts/release_config.json records the
backbone, tag, status text and calibration file the app must use. All recognition assets are optional
(missing -> detection-only app). Assets without a public URL cannot be fetched by a clean Docker build.
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
IMAGENET_FILE = "efficientnet_b0_rwightman-7f5810bc.pth"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="imagenet_frozen", help="imagenet_frozen or a training checkpoint path")
    ap.add_argument("--calibration", default=None, help="calibration JSON for the production mode (default: development)")
    ap.add_argument("--hybrid-head", required=True, choices=["sklearn", "onnx_int8"],
                    help="hybrid head runtime of this release (onnx_int8 = INT8 graph derived from the hybrid joblib)")
    args = ap.parse_args()
    rc = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text(encoding="utf-8"))["recognition"]
    # Check the calibration's runtime before anything is copied or written (a refused promote must change nothing).
    _cal = json.loads((ROOT / Path(args.calibration or rc["calibration"]).as_posix()).read_text(encoding="utf-8"))
    if _cal.get("kind") == "final" and _cal.get("hybrid_head_runtime", "sklearn") != args.hybrid_head:
        raise SystemExit(f"final calibration was made for hybrid head {_cal.get('hybrid_head_runtime', 'sklearn')}, "
                         f"not {args.hybrid_head}")
    release = ROOT / "artifacts/store/release"
    release.mkdir(parents=True, exist_ok=True)
    entries = []

    # --- backbone
    if args.backbone == "imagenet_frozen":
        tag, status = "imagenet_frozen", "development: frozen ImageNet EfficientNet-B0 (not fine-tuned)"
        bb_dst = release / IMAGENET_FILE
        if not bb_dst.exists():
            th = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")) / "hub/checkpoints" / IMAGENET_FILE
            shutil.copyfile(th, bb_dst)
        bb_entry = {"url": IMAGENET_URL, "license": "torchvision ImageNet-1K weights (BSD-3 code licence)", "status": status}
        prefix = "dev"
    else:
        import torch

        ck_path = Path(args.backbone)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        state = ck["state"]
        if state.get("finetune_epochs_completed", 0) < 1:
            raise SystemExit("checkpoint has no completed last-stage fine-tuning epoch; it cannot be the production backbone")
        tag = "ckpt_" + sha(ck_path)[:12]
        status = (f"fine-tuned EfficientNet-B0 ({ck['config']['run_name']}, {state['finetune_epochs_completed']} fine-tuning "
                  f"epoch(s), best tune top-1 {state['best_ft_top1']:.4f})")
        bb_dst = release / f"backbone_{tag}.pth"
        payload = {"model": ck["model"], "source_checkpoint_sha256": sha(ck_path), "run_name": ck["config"]["run_name"]}
        if ck["config"].get("input_size", 224) != 224:   # 224 bundles stay byte-identical to earlier releases
            payload["input_size"] = int(ck["config"]["input_size"])
        torch.save(payload, bb_dst)
        bb_entry = {"url": None, "status": status, "source_checkpoint": ck_path.name, "source_checkpoint_sha256": sha(ck_path),
                    "public_storage": "not yet published"}
        prefix = "ft"
    bb_rel = bb_dst.relative_to(ROOT).as_posix()
    entries.append({"name": bb_dst.name, "path": bb_rel, "sha256": sha(bb_dst), "bytes": bb_dst.stat().st_size,
                    "optional": True, **bb_entry})

    # --- heads
    learned = json.loads((ROOT / f"reports/heads_tune_{tag}.json").read_text(encoding="utf-8"))
    classical = json.loads((ROOT / "reports/classical_tune.json").read_text(encoding="utf-8"))
    src = {"classical": classical["artifact"], "deep": learned["artifacts"]["deep"], "hybrid": learned["artifacts"]["hybrid"]}
    for mode, art in src.items():
        s = store_dir() / art["path"].replace("\\", "/")
        if sha(s) != art["sha256"]:
            raise SystemExit(f"{mode}: store file does not match its report hash")
        dst = ROOT / rc["heads"][mode]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s, dst)
        entries.append({"name": f"head_{mode}.joblib", "path": rc["heads"][mode], "sha256": art["sha256"],
                        "bytes": dst.stat().st_size, "url": None, "optional": True,
                        "source": f"{art['path'].replace(chr(92), '/')} (fit_heads.py, backbone {tag})",
                        "public_storage": "not yet published; clean builds run detection-only until a public URL is added"})

    # --- INT8 hybrid head: derived deterministically from the hybrid joblib above, pinned by sha256; never uploaded
    #     separately (src.release fetch rebuilds it and check-derived proves the rebuild is byte-identical).
    if args.hybrid_head == "onnx_int8":
        from src.models.head_onnx import INT8_NAME, build_int8

        hyb = next(e for e in entries if e["name"] == "head_hybrid.joblib")
        side_p = store_dir() / "optimized" / f"head_hybrid_{tag}_int8.json"
        if not side_p.exists():
            raise SystemExit(f"missing {side_p}; run scripts/optimize_head.py --mode hybrid --backbone {tag} first")
        side = json.loads(side_p.read_text(encoding="utf-8"))
        if not (side.get("int8_passes_policy") and side.get("fp32_parity_ok")):
            raise SystemExit("the INT8 head did not pass the optimisation policy; it cannot be released")
        if side["source_head_sha256"] != hyb["sha256"]:
            raise SystemExit("the INT8 study was made from a different hybrid head")
        dst = ROOT / rc["head_int8"]
        built = build_int8(ROOT / hyb["path"], dst)
        if built["sha256"] != side["sha256"]:
            raise SystemExit(f"INT8 rebuild {built['sha256'][:12]} differs from the studied file {side['sha256'][:12]}")
        entries.append({"name": INT8_NAME, "path": rc["head_int8"], "sha256": built["sha256"], "bytes": built["bytes"],
                        "url": None, "optional": True, "derived_from": hyb["name"], "derived_from_sha256": hyb["sha256"],
                        "recipe": built["recipe"], "source": f"src/models/head_onnx.py (policy report reports/{side['report']})"})

    # --- optional ONNX encoder (must be exported from this exact backbone)
    onnx_enc = ROOT / rc["onnx_encoder"]
    if onnx_enc.exists():
        info = json.loads(onnx_enc.with_suffix(".json").read_text(encoding="utf-8"))
        if info.get("parity_ok") and info["sha256"] == sha(onnx_enc) and info["source_backbone_sha256"] == sha(bb_dst):
            entries.append({"name": onnx_enc.name, "path": rc["onnx_encoder"], "sha256": info["sha256"], "bytes": info["bytes"],
                            "url": None, "optional": True, "parity_max_abs_diff": info["parity_max_abs_diff"],
                            "source": "python -m src.models.export_onnx (parity-gated export of the backbone above)"})
        else:
            print("note: ONNX encoder skipped (not exported from this backbone); re-run src.models.export_onnx")

    # --- reference index (embedding version must match; the app checks it at load time)
    ref = ROOT / rc["reference_index"]
    for p in (ref, ref.with_suffix(".json")):
        if p.exists():
            entries.append({"name": p.name, "path": p.relative_to(ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size,
                            "url": None, "optional": True, "source": "scripts/build_references.py (training split only)"})

    # --- calibration for the production mode
    # Stored paths are read inside the Linux container, so always record them with forward slashes
    # (a Windows shell passes artifacts\calibration_final.json).
    cal_rel = Path(args.calibration or rc["calibration"]).as_posix()
    cal = json.loads((ROOT / cal_rel).read_text(encoding="utf-8"))
    if cal.get("backbone") != tag:
        raise SystemExit(f"calibration {cal_rel} was made for backbone {cal.get('backbone')}, not {tag}")
    entries.append({"name": Path(cal_rel).name, "path": cal_rel, "sha256": sha(ROOT / cal_rel),
                    "bytes": (ROOT / cal_rel).stat().st_size, "url": None, "optional": True, "in_git": True,
                    "kind": cal.get("kind")})

    mpath = ROOT / "artifacts/release_manifest.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    keep = [a for a in manifest["assets"] if a["name"] == "yolo11n.pt"]
    # Re-promoting the same file keeps its public download details (url, licence, storage note).
    prev = {(a["name"], a["sha256"]): a for a in manifest["assets"]}
    for e in entries:
        old = prev.get((e["name"], e["sha256"]))
        if old and old.get("url"):
            for k in ("url", "license", "public_storage"):
                if k in old:
                    e[k] = old[k]
                elif k in e and k != "url":
                    del e[k]
    manifest["assets"] = keep + entries
    ids = [a["sha256"] for a in manifest["assets"]] + (["hybrid_head_runtime=onnx_int8"] if args.hybrid_head == "onnx_int8" else [])
    manifest["bundle_id"] = f"{prefix}_{tag}_" + hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:10]
    manifest["note"] = (f"{'Development' if prefix == 'dev' else 'Fine-tuned'} bundle: YOLO11n detector + {status} + "
                        f"classical/deep/hybrid heads + {cal.get('kind')} calibration.")
    if cal.get("kind") == "final" and cal.get("model_fingerprint"):
        from src.passport.calibration import model_fingerprint

        if args.hybrid_head == "onnx_int8" and cal.get("head_int8_sha256") != built["sha256"]:
            raise SystemExit("final calibration was made with a different INT8 head file; nothing written")
        if model_fingerprint(manifest, cal["production_mode"], rc.get("encoder_runtime", "torch"),
                             args.hybrid_head) != cal["model_fingerprint"]:
            raise SystemExit("final calibration does not match this bundle's model files; nothing written")
    release_cfg ={"bundle_id": manifest["bundle_id"], "backbone": bb_rel, "backbone_tag": tag, "backbone_status": status,
                   "calibration": cal_rel, "calibration_kind": cal.get("kind"), "precision": rc["precision"]}
    if args.hybrid_head != "sklearn":
        release_cfg["hybrid_head_runtime"] = args.hybrid_head
    # Write both files only after every check has passed, each through a temporary file.
    for path, obj in ((mpath, manifest), (ROOT / "artifacts/release_config.json", release_cfg)):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(path)
    print(json.dumps({"bundle_id": manifest["bundle_id"], "assets": [(a["name"], a["sha256"][:12]) for a in manifest["assets"]]}, indent=1))


if __name__ == "__main__":
    main()
