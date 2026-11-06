"""Annotation loading, image audit and split manifests.

Image identity is (source_split, filename): the original train and test directories reuse
basenames such as 00001.jpg. Records keep filename, split, class and box together.
"""
from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.vision.coordinates import BBOX_FORMAT_VERSION, box_valid, stanford_to_canonical

MANIFEST_FIELDS = [
    "image_key", "source_split", "partition", "rel_path", "width", "height", "class_id",
    "original_class_id", "left", "top", "right", "bottom", "bbox_format", "content_sha256",
    "exif_orientation", "orientation_transform", "metadata_version",
]


@dataclass
class Record:
    image_key: str
    source_split: str          # "train" | "test" (original split)
    partition: str             # "train" | "tune" | "calibration" | "test"
    rel_path: str              # relative to data root; never returned by the public API
    width: int
    height: int
    class_id: int
    original_class_id: int
    left: float
    top: float
    right: float
    bottom: float
    bbox_format: str
    content_sha256: str
    exif_orientation: int      # EXIF tag value (1 = none/normal)
    orientation_transform: str
    metadata_version: str

    @property
    def box(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]


def load_annotations(mat_path: Path) -> list[dict]:
    """Return raw annotation rows (1-based inclusive boxes, 1-based class) from a devkit .mat."""
    from scipy.io import loadmat

    mat = loadmat(str(mat_path), squeeze_me=True)
    annos = mat["annotations"]
    rows = []
    for a in annos:
        rows.append({
            "x1": int(a["bbox_x1"]), "y1": int(a["bbox_y1"]),
            "x2": int(a["bbox_x2"]), "y2": int(a["bbox_y2"]),
            "class": int(a["class"]), "fname": str(a["fname"]),
        })
    return rows


def load_class_names(meta_path: Path) -> list[str]:
    from scipy.io import loadmat

    mat = loadmat(str(meta_path), squeeze_me=True)
    return [str(n) for n in mat["class_names"]]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_image(data: bytes) -> dict:
    """Decode fully; return dimensions (stored frame), mode and EXIF orientation."""
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        exif = im.getexif()
        orientation = int(exif.get(0x0112, 1)) if exif else 1
        rgb = im.convert("RGB")
        w, h = rgb.size
        arr = np.asarray(rgb.resize((32, 32)))
    return {"width": w, "height": h, "mode": im.mode, "exif_orientation": orientation,
            "std": float(arr.std())}


def audit_path(path: str) -> dict:
    """Worker for parallel audits: hash bytes, decode, perceptual hash. Never raises."""
    try:
        data = Path(path).read_bytes()
        info = audit_image(data)
        import imagehash

        with Image.open(io.BytesIO(data)) as im:
            ph = int(str(imagehash.phash(ImageOps.exif_transpose(im).convert("L"))), 16)
        return {"path": path, "ok": True, "sha256": sha256_bytes(data), "bytes": len(data), "phash": ph, **info}
    except Exception as exc:  # recorded as an audit failure, never silently dropped
        return {"path": path, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def make_record(source_split: str, row: dict, info: dict, sha: str, metadata_version: str) -> Record:
    box = stanford_to_canonical(row["x1"], row["y1"], row["x2"], row["y2"])
    return Record(
        image_key=f"{source_split}/{row['fname']}",
        source_split=source_split,
        partition="test" if source_split == "test" else "unassigned",
        rel_path=f"cars_{source_split}/{row['fname']}",
        width=info["width"], height=info["height"],
        class_id=row["class"] - 1, original_class_id=row["class"],
        left=box[0], top=box[1], right=box[2], bottom=box[3],
        bbox_format=BBOX_FORMAT_VERSION,
        content_sha256=sha,
        exif_orientation=info["exif_orientation"],
        orientation_transform="none" if info["exif_orientation"] == 1 else f"exif_{info['exif_orientation']}_not_applied_boxes_in_stored_frame",
        metadata_version=metadata_version,
    )


def assign_partitions(train_records: list[Record], seed: int, calib_size: int, tune_size: int) -> None:
    """Calibration: random from the train pool (no model outcomes). Tune: class-stratified from the rest."""
    from sklearn.model_selection import train_test_split

    keys = sorted(r.image_key for r in train_records)
    by_key = {r.image_key: r for r in train_records}
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(keys))
    calib = {keys[i] for i in perm[:calib_size]}
    rest = [k for k in keys if k not in calib]
    labels = [by_key[k].class_id for k in rest]
    train_keys, tune_keys = train_test_split(rest, test_size=tune_size, random_state=seed, stratify=labels)
    for k in calib:
        by_key[k].partition = "calibration"
    for k in tune_keys:
        by_key[k].partition = "tune"
    for k in train_keys:
        by_key[k].partition = "train"


def write_manifest(records: list[Record], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(records, key=lambda r: r.image_key)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(asdict(r))
    data = buf.getvalue().encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def read_manifest(path: Path) -> list[Record]:
    types = {f.name: f.type for f in fields(Record)}
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kw = {}
            for k, v in row.items():
                t = types[k]
                kw[k] = int(v) if t in ("int", int) else float(v) if t in ("float", float) else v
            out.append(Record(**kw))
    return out


def validate_record(r: Record) -> list[str]:
    problems = []
    if not (0 <= r.class_id <= 195) or r.original_class_id != r.class_id + 1:
        problems.append("class_id")
    if not box_valid(r.box, r.width, r.height):
        problems.append("box_out_of_bounds_or_degenerate")
    return problems
