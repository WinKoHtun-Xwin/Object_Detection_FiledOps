"""Tests for the RecognitionWorker."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.recognition import db as dbmod
from app.recognition.engine import FaceResult


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg
    import dataclasses
    patched = dataclasses.replace(
        cfg.settings,
        db_path=tmp_path / "app.db",
        faces_dir=tmp_path / "faces",
        clips_dir=tmp_path / "clips",
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


def _crop() -> np.ndarray:
    return np.zeros((112, 112, 3), dtype=np.uint8)


def test_worker_writes_sighting_for_known_face(env: Path, monkeypatch) -> None:
    from app.recognition.worker import RecognitionWorker
    from app.recognition.gallery import Gallery

    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    w = RecognitionWorker(engine=MagicMock(), gallery=g)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(w, "_sample_frames", lambda path: iter([(0.0, frame), (1.0, frame)]))

    w._engine.detect_and_embed.return_value = [
        FaceResult(crop_bgr=_crop(), embedding=alice_emb, bbox=(0, 0, 100, 100), det_score=0.99)
    ]

    clip_path = env / "clips" / "cam-A" / "2026-05-26" / "100000.mp4"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.touch()
    w._process(clip_path)

    sightings = dbmod.clips_for_person(alice)
    assert len(sightings) == 1
    assert sightings[0]["confidence"] > 0.99
    assert dbmod.list_queue(status="pending") == []


def test_worker_writes_review_for_unknown_face(env: Path, monkeypatch) -> None:
    from app.recognition.worker import RecognitionWorker
    from app.recognition.gallery import Gallery

    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    w = RecognitionWorker(engine=MagicMock(), gallery=g)
    monkeypatch.setattr(
        w, "_sample_frames",
        lambda path: iter([(0.0, np.zeros((480, 640, 3), dtype=np.uint8))]),
    )
    unknown_emb = _unit(999)
    w._engine.detect_and_embed.return_value = [
        FaceResult(crop_bgr=_crop(), embedding=unknown_emb, bbox=(0, 0, 100, 100), det_score=0.99)
    ]

    clip_path = env / "clips" / "cam-A" / "2026-05-26" / "100000.mp4"
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    clip_path.touch()
    w._process(clip_path)

    assert dbmod.clips_for_person(alice) == []
    queue = dbmod.list_queue(status="pending")
    assert len(queue) == 1
    assert queue[0]["suggested_id"] is None or queue[0]["suggested_score"] < 0.55


def test_worker_drains_queue_in_background_thread(env: Path) -> None:
    from app.recognition.worker import RecognitionWorker
    from app.recognition.gallery import Gallery
    import time

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.return_value = []
    w = RecognitionWorker(engine=engine, gallery=g)
    w.start()
    try:
        w.enqueue(env / "clips" / "fake.mp4")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if w._q.empty():
                break
            time.sleep(0.05)
        assert w._q.empty()
    finally:
        w.stop()
