"""Tests for LiveRecognizer."""
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
        live_recognition_interval=1.0,
        live_recognition_cache_ttl=3.0,
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


def _person_box() -> dict:
    return {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "label": "person", "conf": 0.9}


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_annotate_runs_on_first_call(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.recognition.gallery import Gallery

    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=alice_emb,
            bbox=(10, 10, 110, 110),
            det_score=0.99,
        )
    ]
    lr = LiveRecognizer(engine=engine, gallery=g)

    matches = lr.annotate("cam-A", _frame(), [_person_box()])
    assert engine.detect_and_embed.called
    assert len(matches) == 1
    assert matches[0].name == "Alice"
    assert matches[0].kind == "high"
    x, y, w, h = matches[0].bbox
    assert 0.0 <= x < 1.0 and 0.0 <= y < 1.0
    assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def test_annotate_throttles_subsequent_calls(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.recognition.gallery import Gallery

    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=alice_emb,
            bbox=(10, 10, 110, 110),
            det_score=0.99,
        )
    ]
    lr = LiveRecognizer(engine=engine, gallery=g)

    first = lr.annotate("cam-A", _frame(), [_person_box()])
    second = lr.annotate("cam-A", _frame(), [_person_box()])
    assert engine.detect_and_embed.call_count == 1
    assert len(second) == 1
    assert second[0].name == first[0].name


def test_unknown_face_gets_unknown_kind(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.recognition.gallery import Gallery

    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", _unit(1).tobytes(), source="upload")

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=_unit(999),
            bbox=(10, 10, 110, 110),
            det_score=0.99,
        )
    ]
    lr = LiveRecognizer(engine=engine, gallery=g)

    matches = lr.annotate("cam-A", _frame(), [_person_box()])
    assert len(matches) == 1
    assert matches[0].name == "Unknown"
    assert matches[0].kind == "unknown"
