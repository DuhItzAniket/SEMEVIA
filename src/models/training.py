"""Resumable two-stage training: head first, then last pretrained stage fine-tuning.

Checkpoints (every epoch): last.pt and, when improved, best.pt; plus best_finetuned.pt, the
best checkpoint that has completed at least one epoch with the final pretrained stage
unfrozen (the compliant backbone). Each checkpoint holds model, optimizer, epoch counters,
best scores, config, split hashes, RNG states and AMP scaler state.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.data.crops import load_rgb
from src.models.backbone import (
    MEAN, STD, build_model, param_digest, set_trainable, train_mode_respecting_freeze,
)
from src.vision.coordinates import crop_bounds, expand_box


class CropDataset(Dataset):
    def __init__(self, records, data_root: Path, cfg: dict, train: bool, seed: int):
        self.recs, self.root, self.cfg, self.train, self.seed = records, Path(data_root), cfg, train, seed
        self.size = cfg["input_size"]

    def __len__(self):
        return len(self.recs)

    def _box(self, r, w, h, rng):
        box = r.box
        if self.train:
            j = self.cfg["crop"]["jitter"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            box = [box[0] + rng.uniform(-j, j) * bw, box[1] + rng.uniform(-j, j) * bh,
                   box[2] + rng.uniform(-j, j) * bw, box[3] + rng.uniform(-j, j) * bh]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                box = r.box
        eb = expand_box(box, self.cfg["crop"]["margin"], w, h)
        return crop_bounds(eb, w, h) or crop_bounds(expand_box(r.box, 0.0, w, h), w, h)

    def __getitem__(self, i):
        r = self.recs[i]
        rng = random.Random(hash((self.seed, i, torch.initial_seed())) if self.train else 0)
        rgb = load_rgb(self.root / r.rel_path)
        h, w = rgb.shape[:2]
        l, t, rr, b = self._box(r, w, h, rng)
        im = Image.fromarray(rgb[t:b, l:rr]).resize((self.size, self.size), Image.BILINEAR)
        if self.train:
            a = self.cfg["augment"]
            if rng.random() < a["hflip"]:
                im = TF.hflip(im)
            im = TF.adjust_brightness(im, 1 + rng.uniform(-a["brightness"], a["brightness"]))
            im = TF.adjust_contrast(im, 1 + rng.uniform(-a["contrast"], a["contrast"]))
            if rng.random() < a["blur_p"]:
                im = im.filter(ImageFilter.GaussianBlur(radius=1.0))
        x = TF.normalize(TF.to_tensor(im), MEAN, STD)
        return x, r.class_id


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


@torch.no_grad()
def evaluate(model, loader, device, amp: bool):
    model.eval()
    n, loss, c1, c5, all_logits = 0, 0.0, 0, 0, []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device.type, enabled=amp):
            z = model(x)
        z = z.float()
        loss += torch.nn.functional.cross_entropy(z, y, reduction="sum").item()
        top5 = z.topk(5, dim=1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += len(y)
        all_logits.append(z.cpu())
    return {"loss": loss / n, "top1": c1 / n, "top5": c5 / n}, torch.cat(all_logits)


def _optimizer(model, stage, cfg):
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("classifier")]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier")]
    groups = [{"params": head, "lr": stage.get("lr_head", stage.get("lr"))}]
    if body:
        groups.append({"params": body, "lr": stage["lr_backbone"]})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def _base_lrs(stage, opt):
    """Initial LR of each optimizer group, in _optimizer's order (head first, then body)."""
    base = [stage.get("lr_head", stage.get("lr"))]
    if len(opt.param_groups) > 1:
        base.append(stage["lr_backbone"])
    return base


def run(cfg: dict, train_recs, tune_recs, data_root: Path, run_dir: Path, split_hashes: dict,
        resume: bool, log=print, max_steps_per_epoch: int | None = None) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.get("amp")) and device.type == "cuda"
    torch.manual_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    model = build_model(cfg["num_classes"], pretrained=True).to(device)
    pre_digest = {p: param_digest(model, p) for p in ("features.7", "features.8")}
    state = {"stage_idx": 0, "epoch_in_stage": 0, "global_epoch": 0, "best_top1": -1.0,
             "best_ft_top1": -1.0, "bad_epochs": 0, "finetune_epochs_completed": 0, "history": [],
             "pretrained_digest": pre_digest}
    ckpt_last = run_dir / "last.pt"
    opt = scaler = None
    if resume and ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        if ck["split_hashes"] != split_hashes:
            raise SystemExit("resume refused: split manifests changed")
        model.load_state_dict(ck["model"])
        state = ck["state"]
        log(f"resumed from {ckpt_last} at stage {state['stage_idx']} epoch {state['epoch_in_stage']}")
    workers = cfg["num_workers"]
    tune_loader = DataLoader(CropDataset(tune_recs, data_root, cfg, False, cfg["seed"]), batch_size=cfg["batch_size"],
                             shuffle=False, num_workers=workers, persistent_workers=workers > 0)
    t_start = time.time()
    cap = cfg["checkpoint"]["time_cap_minutes"] * 60
    while state["stage_idx"] < len(cfg["stages"]):
        stage = cfg["stages"][state["stage_idx"]]
        trainable = set_trainable(model, stage["trainable"])
        is_ft = any(not t.startswith("classifier") for t in stage["trainable"])
        opt = _optimizer(model, stage, cfg)
        scaler = torch.amp.GradScaler(enabled=amp)
        if resume and ckpt_last.exists() and "optimizer" in (ck := torch.load(ckpt_last, map_location=device, weights_only=False)) \
                and ck["state"]["stage_idx"] == state["stage_idx"] and ck.get("optimizer") is not None:
            opt.load_state_dict(ck["optimizer"])
            if ck.get("scaler"):
                scaler.load_state_dict(ck["scaler"])
            _set_rng_state(ck["rng"])
        resume = False
        while state["epoch_in_stage"] < stage["epochs"]:
            if time.time() - t_start > cap:
                log("time cap reached; stopping with saved checkpoints")
                return state
            g = torch.Generator().manual_seed(cfg["seed"] + state["global_epoch"])
            loader = DataLoader(CropDataset(train_recs, data_root, cfg, True, cfg["seed"] + state["global_epoch"]),
                                batch_size=cfg["batch_size"], shuffle=True, generator=g, num_workers=workers,
                                drop_last=False)
            if stage.get("schedule") == "cosine":
                # Per-epoch cosine decay computed from the epoch counter, so a resumed run gets the same LR.
                f = 0.5 * (1 + math.cos(math.pi * state["epoch_in_stage"] / stage["epochs"]))
                for grp, base in zip(opt.param_groups, _base_lrs(stage, opt)):
                    grp["lr"] = base * f
            train_mode_respecting_freeze(model, stage["trainable"])
            t0, seen, tl = time.time(), 0, 0.0
            for step, (x, y) in enumerate(loader):
                if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
                    break
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=amp):
                    loss = torch.nn.functional.cross_entropy(model(x), y, label_smoothing=cfg.get("label_smoothing", 0.0))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tl += loss.item() * len(y)
                seen += len(y)
            metrics, logits = evaluate(model, tune_loader, device, amp)
            state["epoch_in_stage"] += 1
            state["global_epoch"] += 1
            if is_ft:
                state["finetune_epochs_completed"] += 1
            rec = {"stage": stage["name"], "epoch_in_stage": state["epoch_in_stage"], "global_epoch": state["global_epoch"],
                   "train_loss": tl / max(seen, 1), "train_images": seen, "tune": metrics,
                   "epoch_seconds": round(time.time() - t0, 1), "trainable_tensors": len(trainable), "device": device.type,
                   "lr": [round(g["lr"], 8) for g in opt.param_groups]}
            if is_ft:
                rec["pretrained_changed"] = {p: param_digest(model, p) != state["pretrained_digest"][p]
                                             for p in ("features.7", "features.8")}
            state["history"].append(rec)
            log(json.dumps(rec))
            improved = metrics["top1"] > state["best_top1"]
            if improved:
                state["best_top1"] = metrics["top1"]
            ck = {"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                  "state": state, "config": cfg, "split_hashes": split_hashes, "rng": _rng_state()}
            torch.save(ck, run_dir / "last.tmp")
            (run_dir / "last.tmp").replace(ckpt_last)
            if improved:
                torch.save(ck, run_dir / "best.pt")
                np.save(run_dir / "tune_logits_best.npy", logits.numpy())
            if is_ft:
                if metrics["top1"] > state["best_ft_top1"]:
                    state["best_ft_top1"] = metrics["top1"]
                    state["bad_epochs"] = 0
                    torch.save(ck, run_dir / "best_finetuned.pt")
                    np.save(run_dir / "tune_logits_best_finetuned.npy", logits.numpy())
                else:
                    state["bad_epochs"] += 1
                    if state["bad_epochs"] >= cfg["early_stopping_patience"]:
                        log("early stopping in fine-tuning stage")
                        break
        state["stage_idx"] += 1
        state["epoch_in_stage"] = 0
        state["bad_epochs"] = 0
        torch.save({"model": model.state_dict(), "optimizer": None, "scaler": None, "state": state, "config": cfg,
                    "split_hashes": split_hashes, "rng": _rng_state()}, ckpt_last)
    return state
