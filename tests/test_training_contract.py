import numpy as np
import torch

from src.models.backbone import build_model, crop_to_tensor, embed_and_logits, set_trainable
from src.models.heads import canonical_logits, softmax, topk_accuracy


def test_model_has_196_outputs_and_1280_embedding():
    m = build_model(196, pretrained=False)
    f, z = embed_and_logits(m, torch.zeros(2, 3, 224, 224))
    assert f.shape == (2, 1280) and z.shape == (2, 196)


def test_set_trainable_last_stage_only():
    m = build_model(196, pretrained=False)
    names = set_trainable(m, ["classifier", "features.7", "features.8"])
    assert names and all(n.startswith(("classifier", "features.7", "features.8")) for n in names)
    assert not any(p.requires_grad for n, p in m.named_parameters() if n.startswith("features.6"))


def test_crop_to_tensor_is_normalised_224():
    x = crop_to_tensor(np.full((50, 120, 3), 255, np.uint8))
    assert x.shape == (3, 224, 224) and torch.isfinite(x).all()


def test_topk_ties_break_by_lower_class_id():
    z = np.array([[1.0, 3.0, 3.0, 0.0]])
    assert topk_accuracy(z, np.array([1]), 1) == 1.0
    assert topk_accuracy(z, np.array([2]), 1) == 0.0


def test_softmax_stable_and_rejects_bad_temperature():
    p = softmax(np.array([[1000.0, 0.0]]))
    assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-12
    for T in (0.0, -1.0, float("nan")):
        try:
            softmax(np.zeros((1, 2)), T)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_canonical_logits_reorders_by_classes():
    class Fake:
        classes_ = np.array([2, 0, 1])

        def decision_function(self, X):
            return np.array([[20.0, 0.0, 10.0]])

    z = canonical_logits(Fake(), np.zeros((1, 1)), n_classes=3)
    assert z.tolist() == [[0.0, 10.0, 20.0]]


def test_fusion_features_dims_and_branch_balance():
    from src.models.heads import FusionFeatures

    rng = np.random.RandomState(0)
    X = np.concatenate([rng.randn(300, 1280), rng.rand(300, 1816) * 5], axis=1)
    hyb = FusionFeatures(True, ("hog", "lbp", "hsv", "geometry", "flags"), 64).fit(X)
    Z = hyb.transform(X)
    assert Z.shape == (300, 1344) and np.isfinite(Z).all()
    var_learned, var_hc = Z[:, :1280].var(0).sum(), Z[:, 1280:].var(0).sum()
    assert abs(var_learned - var_hc) / var_learned < 0.01      # equal total branch variance
    deep = FusionFeatures(True, (), 64).fit(X)
    assert deep.transform(X).shape == (300, 1280)
    no_hog = FusionFeatures(True, ("lbp", "hsv", "geometry", "flags"), 64).fit(X)
    assert no_hog._hc(X).shape[1] == 52                         # 10 + 32 + 8 + 2 columns


def test_fusion_per_feature_mode_keeps_unit_variance():
    from src.models.heads import FusionFeatures

    rng = np.random.RandomState(1)
    X = np.concatenate([rng.randn(300, 1280), rng.rand(300, 1816) * 5], axis=1)
    Z = FusionFeatures(True, ("hog", "lbp", "hsv", "geometry", "flags"), 64, "per_feature").fit(X).transform(X)
    assert np.allclose(Z.var(0), 1.0, atol=1e-6)
    try:
        FusionFeatures(True, ("hog",), 8, "bogus").fit(X)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
