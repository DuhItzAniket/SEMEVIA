"""Bounded upload decoding to a canonical RGB image."""
from __future__ import annotations

import hashlib
import io
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


def _pil_open(fp):
    """Pillow's original Image.open, bypassing Ultralytics' global monkey-patch if it is loaded."""
    patches = sys.modules.get("ultralytics.utils.patches")
    original = getattr(patches, "_image_open", None) if patches else None
    return (original or Image.open)(fp)


class DecodeError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class DecodedImage:
    rgb: np.ndarray                 # HxWx3 uint8, canonical frame (EXIF applied, maybe resized)
    sha256: str
    original_size: tuple[int, int]  # (w, h) after EXIF orientation, before resize
    orientation_applied: bool
    resized: bool
    source_format: str

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])


def decode_image(data: bytes, max_bytes: int, max_pixels: int, max_long_edge: int,
                 allowed_formats: list[str]) -> DecodedImage:
    if not data:
        raise DecodeError("empty_file", "The uploaded file is empty. Choose an image file.")
    if len(data) > max_bytes:
        raise DecodeError("file_too_large", f"File exceeds {max_bytes // (1024 * 1024)} MB. Choose a smaller image.")
    try:
        im = _pil_open(io.BytesIO(data))
        fmt = (im.format or "").upper()
        if fmt not in allowed_formats:
            raise DecodeError("unsupported_format", f"Unsupported image format '{fmt or 'unknown'}'. Use JPEG, PNG, WEBP or BMP.")
        w, h = im.size
        if w <= 0 or h <= 0:
            raise DecodeError("invalid_image", "Image has no pixels.")
        if w * h > max_pixels:
            raise DecodeError("too_many_pixels", f"Image is {w}x{h}; the limit is {max_pixels // 1_000_000} megapixels. Resize it and retry.")
        im.load()
        oriented = ImageOps.exif_transpose(im)
        orientation_applied = oriented is not im and oriented.size != (0, 0) and (im.getexif().get(0x0112, 1) != 1)
        im = oriented
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, rgba).convert("RGB")
        else:
            im = im.convert("RGB")
    except DecodeError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, SyntaxError) as exc:
        raise DecodeError("invalid_image", "The file could not be read as an image. Choose a valid JPEG or PNG.") from exc
    original_size = im.size
    resized = False
    if max(im.size) > max_long_edge:
        s = max_long_edge / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
        resized = True
    return DecodedImage(
        rgb=np.ascontiguousarray(np.asarray(im, dtype=np.uint8)),
        sha256=hashlib.sha256(data).hexdigest(),
        original_size=original_size,
        orientation_applied=bool(orientation_applied),
        resized=resized,
        source_format=fmt,
    )
