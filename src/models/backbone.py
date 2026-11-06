"""EfficientNet-B0 backbone with a 196-class head, plus freezing helpers.

The same deterministic crop preprocessing (`crop_to_tensor`) is used for validation,
feature caching and inference so every consumer sees identical inputs.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch
import torchvision
from PIL import Image

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESS_VERSION = "crop_squash224_bilinear_imagenetnorm_v1"   # the 224 px version (r1, r2)


def preprocess_version(size: int = 224) -> str:
    """Version string of crop_to_tensor at a given square input size (224 gives PREPROCESS_VERSION)."""
    return f"crop_squash{int(size)}_bilinear_imagenetnorm_v1"


_SIZE_CACHE: dict = {}


def backbone_input_size(spec) -> int:
    """Input size a backbone was trained at: 'input_size' in a released backbone file, config['input_size'] in a
    training checkpoint, else 224 (torchvision ImageNet weights and every bundle before r3)."""
    spec = str(spec)
    if spec == "imagenet_frozen":
        return 224
    if spec not in _SIZE_CACHE:
        import torch

        ck = torch.load(spec, map_location="cpu", weights_only=False)
        size = 224
        if isinstance(ck, dict):
            size = ck.get("input_size") or (ck.get("config") or {}).get("input_size") or 224
        _SIZE_CACHE[spec] = int(size)
    return _SIZE_CACHE[spec]


def build_model(num_classes: int = 196, pretrained: bool = True) -> torch.nn.Module:
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.efficientnet_b0(weights=weights)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    return m


def set_trainable(model: torch.nn.Module, prefixes: list[str]) -> list[str]:
    names = []
    for n, p in model.named_parameters():
        p.requires_grad = any(n == pre or n.startswith(pre + ".") for pre in prefixes)
        if p.requires_grad:
            names.append(n)
    return names


def train_mode_respecting_freeze(model: torch.nn.Module, prefixes: list[str]) -> None:
    """Train mode for trainable blocks; frozen blocks (incl. their BatchNorm stats) stay in eval."""
    model.train()
    for n, mod in model.named_modules():
        if n and not any(n == pre or n.startswith(pre + ".") or pre.startswith(n + ".") for pre in prefixes):
            if not any(pre.startswith(n + ".") for pre in prefixes):
                mod.eval()


def param_digest(model: torch.nn.Module, prefix: str) -> str:
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n.startswith(prefix):
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def crop_to_tensor(crop_rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    im = Image.fromarray(crop_rgb).resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def embed_and_logits(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (1280-d pooled embedding, logits) for a batch, in eval mode."""
    model.eval()
    f = model.avgpool(model.features(x)).flatten(1)
    return f, model.classifier(f)
