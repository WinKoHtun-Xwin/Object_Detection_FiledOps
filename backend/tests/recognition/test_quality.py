"""Tests for the face quality gate."""
from __future__ import annotations

import numpy as np

from app.recognition.engine import FaceResult
from app.recognition.quality import blur_variance, is_acceptable_face


def _sharp() -> np.ndarray:
    """High-contrast checkerboard — large Laplacian variance."""
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    for i in range(0, 112, 16):
        for j in range(0, 112, 16):
            if (i // 16 + j // 16) % 2 == 0:
                img[i:i + 16, j:j + 16] = 255
    return img


def _flat() -> np.ndarray:
    return np.full((112, 112, 3), 128, dtype=np.uint8)


def _face(crop: np.ndarray, bbox=(0, 0, 100, 100), det_score: float = 0.9) -> FaceResult:
    return FaceResult(
        crop_bgr=crop,
        embedding=np.zeros(512, dtype=np.float32),
        bbox=bbox,
        det_score=det_score,
    )


def test_blur_variance_high_for_textured_zero_for_flat() -> None:
    assert blur_variance(_sharp()) > 50.0
    assert blur_variance(_flat()) == 0.0


def test_accepts_sharp_confident_large_face() -> None:
    assert is_acceptable_face(_face(_sharp()))


def test_rejects_blurry_face() -> None:
    assert not is_acceptable_face(_face(_flat()))


def test_rejects_tiny_face() -> None:
    assert not is_acceptable_face(_face(_sharp(), bbox=(0, 0, 20, 20)))


def test_rejects_low_confidence_face() -> None:
    assert not is_acceptable_face(_face(_sharp(), det_score=0.3))
