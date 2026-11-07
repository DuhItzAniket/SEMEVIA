"""INT8 hybrid head: deterministic build, FP32 parity, row-wise batch invariance, fingerprint binding, runtime guards.

The first group needs no data. The last test uses real tuning crops and the released bundle (skipped without them).
Thresholds of the real-crop test are pre-registered in NEXT.md (r3-int8 decision record, check 3).
"""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def toy_head(tmp_path_factory):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(196), 3)
    X = rng.normal(size=(len(y), 12)) + 0.05 * y[:, None]
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=60))]).fit(X, y)
    path = tmp_path_factory.mktemp("head") / "toy.joblib"
    joblib.dump(pipe, path)
    return pipe, path, X


def test_fp32_graph_matches_sklearn(toy_head):
    from src.models.head_onnx import build_linear_graph, row_wise_logits
    from src.models.heads import canonical_logits

    pipe, _, X = toy_head
    sess = ort.InferenceSession(build_linear_graph(pipe).SerializeToString(), providers=["CPUExecutionProvider"])
    F = pipe[:-1].transform(X[:40])
    z = row_wise_logits(sess, F)
    ref = canonical_logits(pipe, X[:40])
    assert np.allclose(z - z.mean(1, keepdims=True), ref - ref.mean(1, keepdims=True), atol=1e-4)


def test_int8_build_is_deterministic_and_quantised(toy_head, tmp_path):
    from src.models.head_onnx import build_int8

    _, path, _ = toy_head
    a = build_int8(path, tmp_path / "a.onnx")
    b = build_int8(path, tmp_path / "b.onnx")
    assert a["sha256"] == b["sha256"] == hashlib.sha256((tmp_path / "a.onnx").read_bytes()).hexdigest()
    assert a["source_head_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    ops = {n.op_type for n in onnx.load(str(tmp_path / "a.onnx")).graph.node}
    assert {"MatMulInteger", "DynamicQuantizeLinear"} <= ops


def test_row_wise_scoring_is_batch_invariant(toy_head, tmp_path):
    from src.models.head_onnx import build_int8, row_wise_logits, session

    pipe, path, X = toy_head
    build_int8(path, tmp_path / "h.onnx")
    sess = session(tmp_path / "h.onnx", 1)
    F = pipe[:-1].transform(X[:16]).astype(np.float32)
    F[3] *= 50.0   # an outlier row would change a shared activation scale
    alone = row_wise_logits(sess, F[5:6])
    batch = row_wise_logits(sess, F)
    assert np.array_equal(alone[0], batch[5])


def test_class_order_is_enforced(toy_head):
    from src.models.head_onnx import build_linear_graph

    pipe = copy.deepcopy(toy_head[0])
    pipe[-1].classes_ = np.arange(1, 197)   # 196 ids but not 0..195
    with pytest.raises(ValueError):
        build_linear_graph(pipe)


def _legacy_fingerprint(manifest, mode, runtime):
    keep = sorted((a["name"], a["sha256"]) for a in manifest["assets"]
                  if a["name"].endswith((".pt", ".pth", ".joblib")) or (runtime == "onnx_fp32" and a["name"].endswith(".onnx")))
    return hashlib.sha256(json.dumps({"assets": keep, "mode": mode, "runtime": runtime}).encode()).hexdigest()


MANIFEST = {"assets": [{"name": "yolo11n.pt", "sha256": "a"}, {"name": "backbone.pth", "sha256": "b"},
                       {"name": "head_hybrid.joblib", "sha256": "c"}, {"name": "encoder_fp32.onnx", "sha256": "d"}]}


def test_sklearn_fingerprint_is_unchanged():
    from src.passport.calibration import model_fingerprint

    for rt in ("torch", "onnx_fp32"):
        assert model_fingerprint(MANIFEST, "hybrid", rt) == _legacy_fingerprint(MANIFEST, "hybrid", rt)


@pytest.mark.parametrize("it", ["r1", "r2", "r3"])
def test_archived_freezes_still_verify(it):
    from src.passport.calibration import freeze_head_runtime, model_fingerprint

    d = ROOT / "artifacts/history" / it
    if not (d / "production_freeze.json").exists():
        pytest.skip(f"no archived freeze for {it}")
    m = json.loads((d / "release_manifest.json").read_text(encoding="utf-8"))
    fz = json.loads((d / "production_freeze.json").read_text(encoding="utf-8"))
    assert model_fingerprint(m, fz["production_mode"], fz["encoder_runtime"], freeze_head_runtime(fz)) == fz["model_fingerprint"]


def test_int8_fingerprint_binds_the_int8_file():
    from src.passport.calibration import model_fingerprint

    m = copy.deepcopy(MANIFEST)
    with pytest.raises(ValueError):
        model_fingerprint(m, "hybrid", "torch", "onnx_int8")    # no INT8 asset
    m["assets"].append({"name": "head_hybrid_int8.onnx", "sha256": "e"})
    base = model_fingerprint(m, "hybrid", "torch", "sklearn")
    f1 = model_fingerprint(m, "hybrid", "torch", "onnx_int8")
    m["assets"][-1]["sha256"] = "f"
    f2 = model_fingerprint(m, "hybrid", "torch", "onnx_int8")
    assert len({base, f1, f2}) == 3
    # with the sklearn head the INT8 file is ignored, also when an ONNX encoder is in use
    assert model_fingerprint(m, "hybrid", "onnx_fp32") == _legacy_fingerprint(MANIFEST, "hybrid", "onnx_fp32")


def test_runtime_guards_refuse_mismatches():
    from src.passport.calibration import check_rows_runtime, check_runtime_matches

    check_runtime_matches({"encoder_runtime": "torch"}, {"encoder_runtime": "torch"})
    with pytest.raises(SystemExit):
        check_runtime_matches({"encoder_runtime": "torch", "hybrid_head_runtime": "onnx_int8"}, {"encoder_runtime": "torch"})
    with pytest.raises(SystemExit):
        check_runtime_matches({"encoder_runtime": "torch"}, {"encoder_runtime": "torch", "hybrid_head_runtime": "onnx_int8"})
    rows = [{"z_hybrid": [0.0], "hybrid_rt": "onnx_int8"}, {"z_hybrid": [0.0]}, {"sel_box": None}]
    with pytest.raises(SystemExit):
        check_rows_runtime(rows, "onnx_int8", "mixed")
    check_rows_runtime(rows[:1] + rows[2:], "onnx_int8", "int8 only")
    check_rows_runtime(rows[1:], "sklearn", "sklearn only")


DATA_ROOT = Path(os.environ.get("SEMEVIA_DATA_ROOT", "D:/semevia_work/raw/StanfordCars/stanford_cars"))


def _bundle_ok():
    from src.release import recognition_config

    rc = recognition_config()
    return all((ROOT / p).exists() for p in [rc["backbone"], *rc["heads"].values()])


@pytest.mark.skipif(not (DATA_ROOT.is_dir() and _bundle_ok()), reason="needs data and recognition bundle")
def test_int8_head_parity_on_real_tune_crops(tmp_path):
    """Pre-registered check 3: tune[0:16] annotation crops, max centred logit difference <= 0.20 and the same
    argmax on >= 15 of 16; deep and classical heads unchanged."""
    import yaml

    from src.data.crops import annotation_crop, load_rgb
    from src.data.manifest import read_manifest
    from src.models.head_onnx import build_int8
    from src.models.inference import CropInput, Predictor
    from src.release import recognition_config

    rc = recognition_config()
    int8 = ROOT / rc["head_int8"]
    if not int8.exists():
        build_int8(ROOT / rc["heads"]["hybrid"], tmp_path / "h.onnx")
        int8 = tmp_path / "h.onnx"
    heads = {m: ROOT / p for m, p in rc["heads"].items()}
    p32 = Predictor(heads, ROOT / rc["backbone"], 2)
    p8 = Predictor(heads, ROOT / rc["backbone"], 2, hybrid_head_runtime="onnx_int8", head_int8=int8)
    assert p8.head_runtime == {"classical": "sklearn", "deep": "sklearn", "hybrid": "onnx_int8"}
    idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
    recs = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["tune"]["file"])[:16]
    margin = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))["crops"]["margin"]
    crops = []
    for r in recs:
        rgb = load_rgb(DATA_ROOT / r.rel_path)
        crop, box = annotation_crop(rgb, r.box, margin)
        crops.append(CropInput(crop, box, (rgb.shape[1], rgb.shape[0])))
    L = p32.learned(crops)
    H, _ = p32.classical(crops)
    for mode in ("classical", "deep"):
        assert np.array_equal(p32.score_features(mode, L, H), p8.score_features(mode, L, H))
    a, b = p32.score_features("hybrid", L, H), p8.score_features("hybrid", L, H)
    # the served path is batch-invariant: a crop scored alone gets exactly its row of the batch
    assert np.array_equal(p8.score_features("hybrid", L[5:6], H[5:6])[0], b[5])
    diff = np.abs((a - a.mean(1, keepdims=True)) - (b - b.mean(1, keepdims=True))).max()
    assert diff <= 0.20
    assert int((a.argmax(1) == b.argmax(1)).sum()) >= 15
