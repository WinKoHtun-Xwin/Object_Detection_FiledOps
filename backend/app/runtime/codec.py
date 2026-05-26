"""Image codec helpers — JPEG bytes <-> numpy arrays."""
from __future__ import annotations

import cv2
import numpy as np


def jpeg_to_bgr(jpeg: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed -- not a valid JPEG")
    return img
