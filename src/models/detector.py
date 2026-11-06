"""YOLO11n vehicle detector adapter and fixed primary-target policy.

Ultralytics returns boxes already restored to the input image frame, so this adapter
does not apply a second inverse letterbox transform. Numpy inputs to Ultralytics are
interpreted as BGR, so the canonical RGB array is channel-reversed here and only here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.vision.coordinates import box_area, box_valid


def disable_ultralytics_pil_autoinstall() -> None:
    """Ultralytics monkey-patches PIL.Image.open and, on an unidentified file, tries to pip-install
    pi-heif at request time. Mark its plugin step as done so the patch just calls Pillow's original
    open: no runtime package installs or downloads, and bad uploads stay ordinary decode errors."""
    try:
        from ultralytics.utils import patches

        patches._pil_plugins_registered = True
    except Exception:
        pass


@dataclass
class Detection:
    box: list[float]          # canonical ltrb, input-image pixels
    score: float
    coco_class: str
    vehicle: bool


@dataclass
class DetectionResult:
    candidates: list[Detection]          # vehicle-mapped detections at or above conf_floor
    selected: Detection | None
    selected_index: int | None
    rule: str
    threshold: float
    detector_version: str
    all_count: int = 0
    invalid_count: int = 0
    extra: dict = field(default_factory=dict)


class VehicleDetector:
    def __init__(self, weights: str | Path, imgsz: int, conf_floor: float, nms_iou: float, max_det: int,
                 vehicle_classes: dict, weights_sha256: str):
        from ultralytics import YOLO

        disable_ultralytics_pil_autoinstall()
        self.model = YOLO(str(weights), task="detect")
        self.imgsz, self.conf_floor, self.nms_iou, self.max_det = imgsz, conf_floor, nms_iou, max_det
        self.vehicle_classes = {int(k): v for k, v in vehicle_classes.items()}
        self.version = f"yolo11n:{weights_sha256[:12]}"

    def detect(self, rgb: np.ndarray) -> tuple[list[Detection], int, int]:
        h, w = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        # CPU always: the deployed app is CPU-only, and Ultralytics would otherwise pick a GPU when one
        # exists, so offline evaluation on a GPU machine would not measure the deployed pipeline.
        res = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf_floor, iou=self.nms_iou,
                                 max_det=self.max_det, verbose=False, device="cpu")[0]
        names = res.names
        out, invalid, total = [], 0, 0
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy().astype(float)
            conf = res.boxes.conf.cpu().numpy().astype(float)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            for b, s, c in zip(xyxy, conf, cls):
                total += 1
                if c not in self.vehicle_classes:
                    continue
                box = [max(0.0, b[0]), max(0.0, b[1]), min(float(w), b[2]), min(float(h), b[3])]
                if not box_valid(box, w, h):
                    invalid += 1
                    continue
                out.append(Detection(box=box, score=float(s), coco_class=names[int(c)], vehicle=True))
        out.sort(key=lambda d: (-d.score, d.box))
        return out, total, invalid


def select_target(candidates: list[Detection], threshold: float) -> tuple[Detection | None, int | None]:
    """Fixed runtime rule: largest box area among vehicle candidates with score >= threshold.
    Ties broken by higher score, then lower index. Never uses ground truth."""
    best, best_i = None, None
    for i, d in enumerate(candidates):
        if d.score < threshold:
            continue
        if best is None or (box_area(d.box), d.score) > (box_area(best.box), best.score):
            best, best_i = d, i
    return best, best_i
