"""Container side of the r3-int8 check 5: INT8 hybrid logits for /p/X.npy with the image's own onnxruntime."""
import sys; sys.path.insert(0, "/srv")
from pathlib import Path
import numpy as np
from src.models.inference import Predictor
from src.release import recognition_config
rc = recognition_config()
p = Predictor({m: Path("/srv") / v for m, v in rc["heads"].items()}, Path("/srv") / rc["backbone"], 2,
              hybrid_head_runtime=rc["hybrid_head_runtime"], head_int8=Path("/srv") / rc["head_int8"])
np.save("/p/z_container.npy", p._logits("hybrid", np.load("/p/X.npy")))
print("container done")
