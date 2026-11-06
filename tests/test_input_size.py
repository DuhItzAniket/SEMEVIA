"""The encoder input size travels with the backbone file; 224 stays the default."""
import torch

from src.models.backbone import PREPROCESS_VERSION, backbone_input_size, crop_to_tensor, preprocess_version


def test_default_version_unchanged():
    assert preprocess_version() == PREPROCESS_VERSION
    assert preprocess_version(320) == "crop_squash320_bilinear_imagenetnorm_v1"


def test_size_read_from_files(tmp_path):
    released = tmp_path / "released.pth"
    torch.save({"model": {}, "input_size": 320}, released)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model": {}, "config": {"input_size": 288}}, checkpoint)
    legacy = tmp_path / "legacy.pth"
    torch.save({"model": {}, "run_name": "effb0_r2"}, legacy)
    assert backbone_input_size(released) == 320
    assert backbone_input_size(checkpoint) == 288
    assert backbone_input_size(legacy) == 224
    assert backbone_input_size("imagenet_frozen") == 224


def test_crop_to_tensor_size():
    import numpy as np

    crop = np.zeros((50, 80, 3), np.uint8)
    assert tuple(crop_to_tensor(crop, 320).shape) == (3, 320, 320)
    assert tuple(crop_to_tensor(crop).shape) == (3, 224, 224)
