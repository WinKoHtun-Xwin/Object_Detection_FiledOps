"""Tests for the track-pinned LiveRecognizer."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.recognition import db as dbmod
from app.recognition.engine import FaceResult


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg

    patched = dataclasses.replace(
        cfg.settings,
        db_path=tmp_path / "app.db",
        faces_dir=tmp_path / "faces",
    )
    monkeypatch.setattr(cfg, "settings", patched)
    monkeypatch.setattr(dbmod, "settings", patched)
    dbmod._reset_for_tests()
    dbmod.init_schema()
    return tmp_path


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _pb(track_id: int | None) -> dict:
    return {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "label": "person", "track_id": track_id}


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _engine_returning(emb: np.ndarray, det_score: float = 0.99) -> MagicMock:
    engine = MagicMock()
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=emb,
            bbox=(10, 10, 110, 110),
            det_score=det_score,
        )
    ]
    return engine


def _gallery_with(name: str, emb: np.ndarray):
    from app.recognition.gallery import Gallery

    pid = dbmod.add_person(name)
    dbmod.add_face(pid, f"faces/{pid}/a.jpg", emb.tobytes(), source="upload")
    return Gallery()


def test_new_track_enqueues_once_then_resolves(env: Path) -> None:
    from app.recognition.live import LiveRecognizer

    alice = _unit(1)
    g = _gallery_with("Alice", alice)
    lr = LiveRecognizer(engine=_engine_returning(alice), gallery=g, clock=FakeClock())

    first = lr.annotate("cam-A", _frame(), [_pb(1)])
    assert first == {}                       # brand-new track: not known yet
    assert lr._q.qsize() == 1

    lr._drain_for_tests()
    second = lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._engine.detect_and_embed.call_count == 1
    assert second[1].name == "Alice"
    assert second[1].kind == "high"
    assert lr._q.qsize() == 0                # confirmed → no re-enqueue


def test_confirmed_track_refreshes_only_after_interval(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.config import settings

    alice = _unit(1)
    clock = FakeClock()
    g = _gallery_with("Alice", alice)
    lr = LiveRecognizer(engine=_engine_returning(alice), gallery=g, clock=clock)

    lr.annotate("cam-A", _frame(), [_pb(1)])
    lr._drain_for_tests()

    clock.t += settings.live_recognition_refresh - 0.1
    lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._q.qsize() == 0                # not yet due

    clock.t += 0.2
    lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._q.qsize() == 1                # refresh due


def test_no_face_is_tentative_and_retries_after_backoff(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.config import settings

    clock = FakeClock()
    g = _gallery_with("Alice", _unit(1))
    engine = MagicMock()
    engine.detect_and_embed.return_value = []      # person facing away
    lr = LiveRecognizer(engine=engine, gallery=g, clock=clock)

    lr.annotate("cam-A", _frame(), [_pb(1)])
    lr._drain_for_tests()
    shown = lr.annotate("cam-A", _frame(), [_pb(1)])
    assert shown[1].name == "Unknown"
    assert shown[1].kind == "unknown"
    assert lr._q.qsize() == 0                # within retry backoff

    clock.t += settings.live_recognition_retry + 0.01
    lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._q.qsize() == 1                # retry due


def test_track_without_id_is_ignored(env: Path) -> None:
    from app.recognition.live import LiveRecognizer

    g = _gallery_with("Alice", _unit(1))
    lr = LiveRecognizer(engine=_engine_returning(_unit(1)), gallery=g, clock=FakeClock())
    out = lr.annotate("cam-A", _frame(), [_pb(None)])
    assert out == {}
    assert lr._q.qsize() == 0


def test_eviction_after_ttl(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.config import settings

    clock = FakeClock()
    g = _gallery_with("Alice", _unit(1))
    lr = LiveRecognizer(engine=_engine_returning(_unit(1)), gallery=g, clock=clock)

    lr.annotate("cam-A", _frame(), [_pb(1)])
    lr._drain_for_tests()
    assert 1 in lr._state["cam-A"]

    clock.t += settings.track_ttl + 0.1
    lr.annotate("cam-A", _frame(), [_pb(2)])     # any later frame triggers eviction
    assert 1 not in lr._state.get("cam-A", {})


def test_queue_cap_drops_excess(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import config as cfg
    from app.recognition.live import LiveRecognizer

    capped = dataclasses.replace(cfg.settings, recognition_queue_max=2)
    monkeypatch.setattr("app.recognition.live.settings", capped)

    g = _gallery_with("Alice", _unit(1))
    lr = LiveRecognizer(engine=_engine_returning(_unit(1)), gallery=g, clock=FakeClock())
    lr.annotate("cam-A", _frame(), [_pb(1), _pb(2), _pb(3), _pb(4), _pb(5)])
    assert lr._q.qsize() == 2                # capped; extras dropped, not buffered


def test_per_camera_isolation(env: Path) -> None:
    from app.recognition.live import LiveRecognizer

    alice = _unit(1)
    bob = _unit(2)
    dbmod.add_face(dbmod.add_person("Alice"), "faces/a.jpg", alice.tobytes(), source="upload")
    dbmod.add_face(dbmod.add_person("Bob"), "faces/b.jpg", bob.tobytes(), source="upload")
    from app.recognition.gallery import Gallery

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.side_effect = [
        [FaceResult(np.zeros((112, 112, 3), np.uint8), alice, (0, 0, 1, 1), 0.99)],
        [FaceResult(np.zeros((112, 112, 3), np.uint8), bob, (0, 0, 1, 1), 0.99)],
    ]
    lr = LiveRecognizer(engine=engine, gallery=g, clock=FakeClock())

    lr.annotate("cam-A", _frame(), [_pb(1)])   # cam-A track 1
    lr.annotate("cam-B", _frame(), [_pb(1)])   # cam-B track 1 (same id, different camera)
    lr._drain_for_tests()

    a = lr.annotate("cam-A", _frame(), [_pb(1)])
    b = lr.annotate("cam-B", _frame(), [_pb(1)])
    assert a[1].name == "Alice"
    assert b[1].name == "Bob"
