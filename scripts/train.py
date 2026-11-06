"""Train the deep baseline (EfficientNet-B0, 196 classes) with automatic resume.

Usage:
  python scripts/train.py --config configs/train.yaml --resume auto
  python scripts/train.py --smoke 64        # tiny plumbing check; never a metric

Environment: SEMEVIA_DATA_ROOT (original-layout data), SEMEVIA_ARTIFACT_STORE (checkpoints).
Runs on CUDA when available, otherwise CPU. Writes runs/<run_name>/ under the artifact store
with last.pt, best.pt, best_finetuned.pt, train_log.jsonl and summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.data.manifest import read_manifest  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--resume", choices=["auto", "none"], default="auto")
    ap.add_argument("--smoke", type=int, default=0, help="use N train/N tune images and 2 steps per epoch")
    ap.add_argument("--workers", type=int, default=None, help="override data-loader workers")
    ap.add_argument("--data-root", default=os.environ.get("SEMEVIA_DATA_ROOT"))
    args = ap.parse_args()
    if not args.data_root or not Path(args.data_root).is_dir():
        raise SystemExit("set SEMEVIA_DATA_ROOT or --data-root")
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    split_hashes = {p: idx["partitions"][p]["sha256"] for p in ("train", "tune")}
    train = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["train"]["file"])
    tune = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["tune"]["file"])
    store = Path(os.environ.get(cfg["checkpoint"]["dir_env"], "D:/semevia_work/store"))
    run_name = cfg["run_name"] + (f"_SMOKE{args.smoke}" if args.smoke else "")
    max_steps = None
    if args.smoke:
        train, tune = train[: args.smoke], tune[: args.smoke]
        cfg = dict(cfg, num_workers=0 if args.workers is None else args.workers, stages=[dict(s, epochs=1) for s in cfg["stages"]])
        max_steps = 2
    if args.workers is not None:
        cfg = dict(cfg, num_workers=args.workers)
    run_dir = store / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train_log.jsonl"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "msg": msg}) + "\n")

    import torch
    import torchvision

    from src.models.training import run

    log(json.dumps({"start": run_name, "torch": torch.__version__, "torchvision": torchvision.__version__,
                    "cuda": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "host": platform.node(), "python": platform.python_version(), "split_hashes": split_hashes,
                    "smoke": bool(args.smoke)}))
    t = time.time()
    state = run(cfg, train, tune, Path(args.data_root), run_dir, split_hashes, args.resume == "auto", log, max_steps)
    summary = {"run_name": run_name, "smoke": bool(args.smoke), "wall_seconds": round(time.time() - t, 1),
               "best_tune_top1_any": state["best_top1"], "best_tune_top1_finetuned": state["best_ft_top1"],
               "finetune_epochs_completed": state["finetune_epochs_completed"],
               "fine_tuning_requirement_met": state["finetune_epochs_completed"] >= 1 and not args.smoke,
               "history": state["history"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(json.dumps({k: v for k, v in summary.items() if k != "history"}))


if __name__ == "__main__":
    main()
