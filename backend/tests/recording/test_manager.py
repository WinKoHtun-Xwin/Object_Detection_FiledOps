"""Tests for RecorderManager."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.recording.manager import RecorderManager


def _box(x: float) -> dict:
    return {"x": x, "y": 0.5, "w": 0.1, "h": 0.2, "label": "person", "conf": 0.9}


def _frame() -> np.ndarray:
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


def test_per_camera_subfolders(rec_root: Path) -> None:
    mgr = RecorderManager()
    # interleave frames from two cameras, each with its own motion event
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        mgr.feed("cam-A", _frame(), [_box(0.5)])
        mgr.feed("cam-B", _frame(), [_box(0.5)])
        t += dt
    for i in range(30):
        mgr.feed("cam-A", _frame(), [_box(0.5 + 0.01 * i)])
        mgr.feed("cam-B", _frame(), [_box(0.5 + 0.01 * i)])
        t += dt
    for _ in range(60):
        mgr.feed("cam-A", _frame(), [_box(0.85)])
        mgr.feed("cam-B", _frame(), [_box(0.85)])
        t += dt
    mgr.close_all()

    assert (rec_root / "cam-A").exists()
    assert (rec_root / "cam-B").exists()
    a_mp4s = list((rec_root / "cam-A").rglob("*.mp4"))
    b_mp4s = list((rec_root / "cam-B").rglob("*.mp4"))
    assert a_mp4s, "cam-A produced no clips"
    assert b_mp4s, "cam-B produced no clips"


def test_unknown_camera_creates_recorder_lazily(rec_root: Path) -> None:
    mgr = RecorderManager()
    assert "cam-new" not in mgr._recorders   # noqa: SLF001
    mgr.feed("cam-new", _frame(), [])
    assert "cam-new" in mgr._recorders   # noqa: SLF001
