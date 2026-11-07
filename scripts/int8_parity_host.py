"""Host side of the r3-int8 check 5: fused input X for tune[0:16] annotation crops, and the host INT8 hybrid logits.

Usage: python scripts/int8_parity_host.py   (writes C:/semevia/tmp/parity/X.npy and z_host.npy), then
  docker run --rm --network none -v C:/semevia/tmp/parity:/p -v <repo>/scripts:/s semevia:local python /s/int8_parity_container.py
The comparison is recorded in reports/int8_container_parity.json.
"""
import json, os, sys
from pathlib import Path
import numpy as np, yaml
ROOT = Path(r"C:\Users\Luikz\Downloads\SEMEVIA"); sys.path.insert(0, str(ROOT))
from src.data.crops import annotation_crop, load_rgb
from src.data.manifest import read_manifest
from src.models.inference import CropInput, Predictor
from src.release import recognition_config
rc = recognition_config()
p = Predictor.from_config(rc, ROOT, 2)
assert p.head_runtime["hybrid"] == "onnx_int8"
idx = json.loads((ROOT / "artifacts/manifests/manifest_index.json").read_text(encoding="utf-8"))
recs = read_manifest(ROOT / "artifacts/manifests" / idx["partitions"]["tune"]["file"])[:16]
margin = yaml.safe_load((ROOT / "configs/features.yaml").read_text(encoding="utf-8"))["crops"]["margin"]
D = Path(os.environ["SEMEVIA_DATA_ROOT"]); crops = []
for r in recs:
    rgb = load_rgb(D / r.rel_path); c, b = annotation_crop(rgb, r.box, margin); crops.append(CropInput(c, b, (rgb.shape[1], rgb.shape[0])))
X = np.concatenate([p.learned(crops), p.classical(crops)[0]], axis=1)
np.save(r"C:\semevia\tmp\parity\X.npy", X)
np.save(r"C:\semevia\tmp\parity\z_host.npy", p._logits("hybrid", X))
print("host done", X.shape)
