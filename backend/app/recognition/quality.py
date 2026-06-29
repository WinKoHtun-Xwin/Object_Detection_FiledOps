"""Face quality gate.

Drops blurry, tiny, or low-confidence detections before they reach the review
queue or a person's gallery. Thresholds live in `Settings` (face_min_*), so the
gate is tuned in config.py, not here. Sharpness is measured on the 112px aligned
crop so the live gate and the offline review-queue purge use the same metric.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import settings
from app.recognition.engine import FaceResult


def blur_variance(crop_bgr: np.ndarray) -> float:
    """Variance of the Laplacian — a standard sharpness score. Higher = sharper."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_acceptable_face(face: FaceResult) -> bool:
    """True if a detected face is large, confident, and sharp enough to keep."""
    x1, y1, x2, y2 = face.bbox
    if min(x2 - x1, y2 - y1) < settings.face_min_px:
        return False
    if face.det_score < settings.face_min_det_score:
        return False
    return blur_variance(face.crop_bgr) >= settings.face_min_blur_var
