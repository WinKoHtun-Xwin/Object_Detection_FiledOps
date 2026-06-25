"""FaceEngine carries a configurable det_size (no model load needed)."""
from __future__ import annotations

from app.recognition.engine import FaceEngine


def test_default_det_size_is_640():
    assert FaceEngine()._det_size == 640


def test_custom_det_size():
    assert FaceEngine(det_size=320)._det_size == 320
