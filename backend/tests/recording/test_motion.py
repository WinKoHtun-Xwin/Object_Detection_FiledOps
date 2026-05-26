"""Tests for MotionDetector."""
from __future__ import annotations

from app.recording.motion import MotionDetector


def _box(x: float, y: float, label: str = "person") -> dict:
    return {"x": x, "y": y, "w": 0.1, "h": 0.2, "label": label, "conf": 0.9}


def test_empty_detections_is_no_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    assert det.update([]) is False


def test_single_static_frame_is_no_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([_box(0.5, 0.5)])
    assert det.update([_box(0.5, 0.5)]) is False


def test_box_drift_above_threshold_is_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([_box(0.5, 0.5)])
    assert det.update([_box(0.6, 0.5)]) is True   # 0.10 shift on x — well above 0.01


def test_box_drift_below_threshold_is_no_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.02)
    det.update([_box(0.500, 0.500)])
    assert det.update([_box(0.505, 0.500)]) is False  # 0.005 < 0.02


def test_non_allowlisted_class_ignored() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([_box(0.5, 0.5, label="car")])
    assert det.update([_box(0.7, 0.5, label="car")]) is False


def test_allowlist_class_appears_after_nothing_is_not_motion() -> None:
    """A class entering the frame for the first time is presence, not motion."""
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([])
    assert det.update([_box(0.5, 0.5)]) is False
