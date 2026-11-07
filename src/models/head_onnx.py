"""INT8 hybrid head: the logistic-regression layer of a released head, as a quantised ONNX graph.

The fitted coefficients are copied into an ONNX MatMul+Add graph (columns in class-id order 0..195) and quantised
with onnxruntime dynamic quantisation (QInt8 per-tensor weights; activations quantised per call). The fusion
transform before it (scalers + PCA) stays the sklearn FP32 one. The build is deterministic: the same joblib gives a
byte-identical file, which is how a release pins it by sha256 without publishing the file separately.

Dynamic quantisation picks one activation scale per input tensor, so the graph is always run one row at a time
(row_wise_logits): each image's logits then depend only on that image's features, not on the batch it is in.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np

INT8_NAME = "head_hybrid_int8.onnx"
RUNTIMES = ("sklearn", "onnx_int8")
RECIPE = ("logistic coefficients -> ONNX MatMul+Add (opset 17, IR 10); onnxruntime quantize_dynamic QInt8 per-tensor "
          "weights (MatMulInteger); onnx 1.23.2 / onnxruntime 1.31.1")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_linear_graph(pipe, n_classes: int = 196):
    """FP32 ONNX graph of the pipeline's final linear classifier. Graph, tensor and initializer names and node
    order are part of the pinned sha256: do not change them."""
    from onnx import TensorProto, checker, helper, numpy_helper

    clf = pipe[-1]
    classes = np.asarray(clf.classes_)
    if classes.shape[0] != n_classes or set(classes.tolist()) - set(range(n_classes)):
        raise ValueError(f"head must have exactly the class ids 0..{n_classes - 1}")
    W = np.zeros((n_classes, clf.coef_.shape[1]), np.float32)
    b = np.zeros(n_classes, np.float32)
    W[classes], b[classes] = clf.coef_, clf.intercept_
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["features", "W"], ["mm"]), helper.make_node("Add", ["mm", "b"], ["logits"])],
        "semevia_linear_head",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, W.shape[1]])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, n_classes])],
        [numpy_helper.from_array(W.T.copy(), "W"), numpy_helper.from_array(b, "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10  # onnxruntime 1.31 loads IR <= 13; onnx 1.23 defaults to 14
    checker.check_model(model)
    return model


def build_int8(joblib_path: Path, out_path: Path, fp32_out: Path | None = None) -> dict:
    """Write the INT8 graph for a released joblib head; return its sha256 and provenance."""
    import joblib
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    pipe = joblib.load(joblib_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        p32 = Path(fp32_out) if fp32_out else Path(td) / "head_fp32.onnx"
        onnx.save(build_linear_graph(pipe), str(p32))
        tmp8 = Path(td) / "head_int8.onnx"
        quantize_dynamic(str(p32), str(tmp8), weight_type=QuantType.QInt8, per_channel=False)
        out_path.write_bytes(tmp8.read_bytes())
    return {"sha256": _sha(out_path), "bytes": out_path.stat().st_size,
            "source_head_sha256": _sha(joblib_path), "recipe": RECIPE}


def session(path: Path, threads: int = 2):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def row_wise_logits(sess, F: np.ndarray) -> np.ndarray:
    """Score each row on its own (one dynamic activation scale per image)."""
    F = np.asarray(F, dtype=np.float32)
    return np.vstack([sess.run(None, {"features": F[i:i + 1]})[0] for i in range(F.shape[0])]).astype(np.float64)
