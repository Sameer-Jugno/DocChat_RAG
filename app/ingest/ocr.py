"""OCR helpers for scanned PDF pages and embedded images (RapidOCR / ONNX)."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger("pdf_chat.ocr")


@lru_cache
def get_ocr():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_image(image: Image.Image | np.ndarray) -> str:
    """Run OCR on a PIL image or numpy array; return plain text."""
    if isinstance(image, Image.Image):
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        arr = np.array(image)
    else:
        arr = image

    try:
        result, _ = get_ocr()(arr)
    except Exception:  # noqa: BLE001
        logger.exception("OCR failed")
        return ""

    if not result:
        return ""

    # RapidOCR returns list of [box, text, score]
    lines: list[str] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        text = str(item[1]).strip()
        if text:
            lines.append(text)
    return "\n".join(lines).strip()


def ocr_pixmap_png_bytes(png_bytes: bytes) -> str:
    from io import BytesIO

    img = Image.open(BytesIO(png_bytes))
    return ocr_image(img)
