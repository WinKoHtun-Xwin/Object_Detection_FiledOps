"""Tests for MotionRecorder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.recording.recorder import MotionRecorder


def _box(x: float, y: float) -> dict:
    return {"x": x, "y": y, "w": 0.1, "h": 0.2, "label": "person", "conf": 0.9}


def _frame() -> np.ndarray:
    # 240×320 BGR frame
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def rec_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg
    import dataclasses
    # Settings is a frozen dataclass; replace the module-level singleton with a
    # patched copy so tests write to tmp_path instead of the real clips dir.
    patched = dataclasses.replace(cfg.settings, clips_dir=tmp_path)
    monkeypatch.setattr(cfg, "settings", patched)
    import app.recording.recorder as rec_mod
    monkeypatch.setattr(rec_mod, "settings", patched)
    return tmp_path


def test_no_motion_produces_no_files(rec_root: Path) -> None:
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240))
    # 30 static frames — same box in same place
    for i in range(30):
        rec.feed(_frame(), [_box(0.5, 0.5)], ts=i / 15.0)
    rec.close()
    files = list(rec_root.rglob("*"))
    assert all(f.is_dir() for f in files), f"expected no clip files, got {files}"


def test_motion_event_creates_mp4_and_jpg(rec_root: Path) -> None:
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240))
    # 5 static frames (warm up the detector with a baseline), then 30 frames of
    # drifting motion, then 60 static frames to drive post-roll + close.
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        rec.feed(_frame(), [_box(0.5, 0.5)], ts=t); t += dt
    for i in range(30):
        rec.feed(_frame(), [_box(0.5 + 0.02 * i, 0.5)], ts=t); t += dt
    for _ in range(60):
        rec.feed(_frame(), [_box(0.8, 0.5)], ts=t); t += dt   # static, no motion
    rec.close()

    mp4s = sorted(rec_root.rglob("*.mp4"))
    jpgs = sorted(rec_root.rglob("*.jpg"))
    assert len(mp4s) >= 1, f"expected ≥1 mp4, got {mp4s}"
    assert len(jpgs) == 1, f"expected exactly one event snapshot, got {jpgs}"
    assert mp4s[0].stat().st_size > 0
    assert jpgs[0].stat().st_size > 0


def test_sustained_motion_rotates_15s_files(rec_root: Path) -> None:
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240))
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        rec.feed(_frame(), [_box(0.5, 0.5)], ts=t); t += dt
    # ~25 seconds of continuous motion → expect 2 mp4 files in the event.
    # Alternating drift of 0.02 each frame (above the 0.015 threshold).
    for i in range(25 * 15):
        rec.feed(_frame(), [_box(0.5 + 0.02 * (i % 2 + 1), 0.5)], ts=t); t += dt
    for _ in range(60):
        rec.feed(_frame(), [_box(0.9, 0.5)], ts=t); t += dt   # let post-roll close
    rec.close()

    mp4s = sorted(rec_root.rglob("*.mp4"))
    jpgs = sorted(rec_root.rglob("*.jpg"))
    assert len(mp4s) >= 2, f"expected ≥2 rotation files, got {mp4s}"
    assert len(jpgs) == 1, "snapshot is per-event, not per-rotation"
