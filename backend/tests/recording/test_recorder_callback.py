"""Test that MotionRecorder fires on_clip_closed once per event."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.recording.recorder import MotionRecorder


def _box(x: float) -> dict:
    return {"x": x, "y": 0.5, "w": 0.1, "h": 0.2, "label": "person", "conf": 0.9}


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def rec_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg
    import dataclasses
    patched = dataclasses.replace(cfg.settings, clips_dir=tmp_path)
    monkeypatch.setattr(cfg, "settings", patched)
    import app.recording.recorder as rec_mod
    monkeypatch.setattr(rec_mod, "settings", patched)
    return tmp_path


def test_callback_fires_once_per_event(rec_root: Path) -> None:
    callback = MagicMock()
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240), on_clip_closed=callback)
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        rec.feed(_frame(), [_box(0.5)], ts=t); t += dt
    for i in range(25 * 15):
        rec.feed(_frame(), [_box(0.5 + 0.02 * (i % 2 + 1))], ts=t); t += dt
    for _ in range(60):
        rec.feed(_frame(), [_box(0.9)], ts=t); t += dt
    rec.close()
    assert callback.call_count == 1
    first_clip = callback.call_args[0][0]
    assert isinstance(first_clip, Path)
    assert first_clip.suffix == ".mp4"


def test_no_callback_when_no_event(rec_root: Path) -> None:
    callback = MagicMock()
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240), on_clip_closed=callback)
    for i in range(30):
        rec.feed(_frame(), [_box(0.5)], ts=i / 15.0)
    rec.close()
    assert callback.call_count == 0


def test_callback_called_with_first_clip_on_rotation(rec_root: Path) -> None:
    seen_paths: list[Path] = []
    def cb(p: Path) -> None:
        seen_paths.append(p)
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240), on_clip_closed=cb)
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        rec.feed(_frame(), [_box(0.5)], ts=t); t += dt
    for i in range(25 * 15):
        rec.feed(_frame(), [_box(0.5 + 0.02 * (i % 2 + 1))], ts=t); t += dt
    for _ in range(60):
        rec.feed(_frame(), [_box(0.9)], ts=t); t += dt
    rec.close()
    assert len(seen_paths) == 1
    mp4s = sorted(rec_root.rglob("*.mp4"))
    assert len(mp4s) >= 2
    assert seen_paths[0] == mp4s[0]
