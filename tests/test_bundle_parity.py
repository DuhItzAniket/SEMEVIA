"""Contract check: the deployed Predictor reproduces the offline caches and head logits on real
tuning images (same crop, preprocessing, class order). Skipped without data/caches/bundle."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SEMEVIA_DATA_ROOT", "D:/semevia_work/raw/StanfordCars/stanford_cars"))
RCFG = yaml.safe_load((ROOT / "configs/runtime.yaml").read_text(encoding="utf-8"))["recognition"]


def _have_bundle():
    return all((ROOT / p).exists() for p in [RCFG["backbone"], *RCFG["heads"].values()])


@pytest.mark.skipif(not (DATA_ROOT.is_dir() and _have_bundle()), reason="needs data and recognition bundle")
def test_predictor_matches_cached_features_and_logits():
    from src.data.crops import annotation_crop, load_rgb
    from src.data.manifest import read_manifest
    from src.models.cache_io import cache_meta, load_cache
    from src.models.heads import canonical_logits
    from src.models.inference import CropInput, Predictor

    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    recs = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["tune"]["file"])[:8]
    kl, _, XL, _ = load_cache(cache_meta("learned", "tune", "annotation", idx, "imagenet_frozen"))
    kc, _, XC, _ = load_cache(cache_meta("classical", "tune", "annotation", idx))
    pred = Predictor({m: ROOT / p for m, p in RCFG["heads"].items()}, ROOT / RCFG["backbone"], 2)
    margin = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))["crops"]["margin"]
    crops = []
    for r in recs:
        rgb = load_rgb(DATA_ROOT / r.rel_path)
        crop, box = annotation_crop(rgb, r.box, margin)
        crops.append(CropInput(crop, box, (rgb.shape[1], rgb.shape[0])))
    rows = [list(kl).index(r.image_key) for r in recs]
    assert np.allclose(pred.learned(crops), XL[rows], atol=1e-4, rtol=1e-4)
    hc, _ = pred.classical(crops)
    assert np.allclose(hc, XC[rows], atol=1e-5)
    for mode in pred.modes:
        z, _ = pred.score_crops(mode, crops)
        X = XC[rows] if mode == "classical" else np.concatenate([XL[rows], XC[rows]], axis=1)
        z_off = canonical_logits(pred.heads[mode], X)
        zc, zo = z - z.mean(1, keepdims=True), z_off - z_off.mean(1, keepdims=True)
        assert np.allclose(zc, zo, atol=1e-3)
        assert (z.argmax(1) == z_off.argmax(1)).all()


@pytest.mark.skipif(not (DATA_ROOT.is_dir() and _have_bundle() and (ROOT / RCFG["onnx_encoder"]).exists()),
                    reason="needs data, bundle and exported ONNX encoder")
def test_onnx_runtime_matches_torch_on_real_crops():
    from src.data.crops import annotation_crop, load_rgb
    from src.data.manifest import read_manifest
    from src.models.inference import CropInput, Predictor

    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    recs = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["tune"]["file"])[8:16]
    heads = {m: ROOT / p for m, p in RCFG["heads"].items()}
    pt = Predictor(heads, ROOT / RCFG["backbone"], 2, "torch")
    po = Predictor(heads, ROOT / RCFG["backbone"], 2, "onnx_fp32", ROOT / RCFG["onnx_encoder"])
    crops = []
    for r in recs:
        rgb = load_rgb(DATA_ROOT / r.rel_path)
        crop, box = annotation_crop(rgb, r.box, 0.05)
        crops.append(CropInput(crop, box, (rgb.shape[1], rgb.shape[0])))
    assert np.allclose(pt.learned(crops), po.learned(crops), atol=1e-4, rtol=1e-4)
    for mode in ("deep", "hybrid"):
        zt, _ = pt.score_crops(mode, crops)
        zo, _ = po.score_crops(mode, crops)
        assert (zt.argmax(1) == zo.argmax(1)).all()
        assert np.allclose(zt - zt.mean(1, keepdims=True), zo - zo.mean(1, keepdims=True), atol=1e-3)
