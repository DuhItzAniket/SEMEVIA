"""Authoritative crop inference shared by the app, evaluation and the Evidence Challenge.

score_crops(mode, crops) recomputes every branch the mode's head uses (learned embedding and/or
handcrafted features) for each crop, so challenge variants are scored by exactly the deployed
pipeline. Modes: classical (handcrafted head), deep (matched learned-only head), hybrid (fused).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.models.heads import canonical_logits
from src.vision.features import extract_classical

MODES = ("classical", "deep", "hybrid")


@dataclass
class CropInput:
    crop: np.ndarray                 # HxWx3 uint8 RGB
    box: list[float]                 # crop location in the source image (canonical ltrb)
    image_size: tuple[int, int]      # (w, h) of the source image


class Predictor:
    def __init__(self, heads: dict[str, Path], backbone: str | Path, torch_threads: int = 2,
                 encoder_runtime: str = "torch", onnx_encoder: str | Path | None = None):
        import joblib
        import torch

        torch.set_num_threads(torch_threads)
        self.heads = {m: joblib.load(p) for m, p in heads.items()}
        unknown = set(self.heads) - set(MODES)
        if unknown:
            raise ValueError(f"unknown modes {unknown}")
        if encoder_runtime not in ("torch", "onnx_fp32"):
            raise ValueError("encoder_runtime must be torch or onnx_fp32")
        self.backbone_spec = str(backbone)
        self.encoder_runtime = encoder_runtime
        self.input_size = 224
        self.model = None
        self.session = None
        if any(m in self.heads for m in ("deep", "hybrid")):
            from src.models.backbone import backbone_input_size

            self.input_size = backbone_input_size(backbone)
            if encoder_runtime == "onnx_fp32":
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = torch_threads
                self.session = ort.InferenceSession(str(onnx_encoder), so, providers=["CPUExecutionProvider"])
            else:
                self.model = _load_backbone(backbone)

    @property
    def has_encoder(self) -> bool:
        return self.model is not None or self.session is not None

    @property
    def modes(self) -> list[str]:
        return [m for m in MODES if m in self.heads]

    def learned(self, crops: list[CropInput]) -> np.ndarray:
        import torch

        from src.models.backbone import crop_to_tensor, embed_and_logits

        x = torch.stack([crop_to_tensor(c.crop, self.input_size) for c in crops])
        if self.session is not None:
            return self.session.run(None, {"input": x.numpy()})[0].astype(np.float64)
        f, _ = embed_and_logits(self.model, x)
        return f.numpy().astype(np.float64)

    def classical(self, crops: list[CropInput], want_views: bool = False):
        res = [extract_classical(c.crop, c.box, c.image_size, want_views=want_views) for c in crops]
        return np.stack([r.vector for r in res]).astype(np.float64), res

    def score_features(self, mode: str, learned: np.ndarray | None, hc: np.ndarray) -> np.ndarray:
        """Score precomputed branch features (used by batch evaluation to avoid recomputing)."""
        X = hc if mode == "classical" else np.concatenate([learned, hc], axis=1)
        z = canonical_logits(self.heads[mode], X)
        if not np.isfinite(z).all():
            raise RuntimeError("non-finite logits")
        return z

    def score_crops(self, mode: str, crops: list[CropInput], want_views: bool = False) -> tuple[np.ndarray, list]:
        if mode not in self.heads:
            raise KeyError(f"mode {mode} not available")
        hc, hc_res = self.classical(crops, want_views)
        if mode == "classical":
            X = hc
        else:
            X = np.concatenate([self.learned(crops), hc], axis=1)
        z = canonical_logits(self.heads[mode], X)
        if not np.isfinite(z).all():
            raise RuntimeError("non-finite logits")
        return z, hc_res


def _load_backbone(spec):
    import torch

    from src.models.backbone import build_model

    spec = str(spec)
    m = build_model(196, pretrained=False)
    sd = torch.load(spec, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
        m.load_state_dict(sd)
    else:  # torchvision ImageNet state dict: load features only; our 196-class classifier is unused here
        missing, unexpected = m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("classifier")}, strict=False)
        if any(not k.startswith("classifier") for k in missing) or unexpected:
            raise RuntimeError("backbone weights do not match EfficientNet-B0 features")
    return m.eval()
