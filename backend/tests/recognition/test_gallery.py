"""Tests for the in-memory Gallery."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.recognition import db as dbmod
from app.recognition.gallery import Gallery


@pytest.fixture
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg
    import dataclasses
    p = tmp_path / "app.db"
    patched = dataclasses.replace(cfg.settings, db_path=p, faces_dir=tmp_path / "faces")
    monkeypatch.setattr(cfg, "settings", patched)
    monkeypatch.setattr(dbmod, "settings", patched)
    dbmod._reset_for_tests()
    dbmod.init_schema()
    return p


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_empty_gallery_returns_none(fresh_db: Path) -> None:
    g = Gallery()
    pid, score = g.match(_unit(1))
    assert pid is None
    assert score == 0.0


def test_match_finds_closest_person(fresh_db: Path) -> None:
    alice_emb = _unit(1)
    bob_emb = _unit(2)
    alice = dbmod.add_person("Alice")
    bob = dbmod.add_person("Bob")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")
    dbmod.add_face(bob, "faces/2/b.jpg", bob_emb.tobytes(), source="upload")
    g = Gallery()

    noisy_alice = alice_emb + 0.01 * _unit(99)
    noisy_alice = noisy_alice / np.linalg.norm(noisy_alice)
    pid, score = g.match(noisy_alice)
    assert pid == alice
    assert score > 0.95


def test_match_returns_low_score_for_unrelated(fresh_db: Path) -> None:
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", _unit(1).tobytes(), source="upload")
    g = Gallery()

    far = _unit(500)
    pid, score = g.match(far)
    assert pid == alice
    assert abs(score) < 0.2


def test_add_updates_in_memory_without_reload(fresh_db: Path) -> None:
    g = Gallery()
    pid = dbmod.add_person("Alice")
    dbmod.add_face(pid, "faces/1/a.jpg", _unit(1).tobytes(), source="upload")
    g.add(pid, _unit(1))
    pid2, score = g.match(_unit(1))
    assert pid2 == pid
    assert score > 0.99


def test_reload_picks_up_external_rows(fresh_db: Path) -> None:
    g = Gallery()
    pid = dbmod.add_person("Alice")
    dbmod.add_face(pid, "faces/1/a.jpg", _unit(1).tobytes(), source="upload")
    p, _ = g.match(_unit(1))
    assert p is None
    g.reload()
    p, _ = g.match(_unit(1))
    assert p == pid
