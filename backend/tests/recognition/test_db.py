"""Tests for the recognition SQLite wrapper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.recognition import db as dbmod


@pytest.fixture
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point settings.db_path at a temp file and reset the connection cache."""
    from app import config as cfg
    import dataclasses
    p = tmp_path / "app.db"
    patched = dataclasses.replace(cfg.settings, db_path=p, faces_dir=tmp_path / "faces")
    monkeypatch.setattr(cfg, "settings", patched)
    monkeypatch.setattr(dbmod, "settings", patched)
    dbmod._reset_for_tests()
    dbmod.init_schema()
    return p


def _emb(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tobytes()


def test_init_schema_is_idempotent(fresh_db: Path) -> None:
    dbmod.init_schema()
    persons = dbmod.list_persons()
    assert persons == []


def test_add_and_list_persons(fresh_db: Path) -> None:
    pid = dbmod.add_person("Alice")
    persons = dbmod.list_persons()
    assert len(persons) == 1
    assert persons[0]["id"] == pid
    assert persons[0]["name"] == "Alice"
    assert persons[0]["face_count"] == 0


def test_face_count_reflects_faces(fresh_db: Path) -> None:
    pid = dbmod.add_person("Alice")
    dbmod.add_face(pid, crop_path="faces/1/a.jpg", embedding=_emb(1), source="upload")
    dbmod.add_face(pid, crop_path="faces/1/b.jpg", embedding=_emb(2), source="upload")
    persons = dbmod.list_persons()
    assert persons[0]["face_count"] == 2


def test_delete_person_cascades_faces_and_sightings(fresh_db: Path) -> None:
    pid = dbmod.add_person("Alice")
    dbmod.add_face(pid, crop_path="faces/1/a.jpg", embedding=_emb(1), source="upload")
    dbmod.add_sighting("cam-A/2026-05-26/100000.mp4", pid, 0.9, frame_ts=1.0)
    dbmod.delete_person(pid)
    assert dbmod.list_persons() == []
    assert dbmod.list_faces_for(pid) == []
    assert dbmod.clips_for_person(pid) == []


def test_review_queue_lifecycle(fresh_db: Path) -> None:
    qid = dbmod.enqueue_review(
        crop_path="queue/abc.jpg", embedding=_emb(7),
        source_clip="cam-A/2026-05-26/100000.mp4",
        suggested_id=None, suggested_score=None,
    )
    pending = dbmod.list_queue(status="pending")
    assert len(pending) == 1
    assert pending[0]["id"] == qid
    dbmod.update_queue_status(qid, "dismissed")
    assert dbmod.list_queue(status="pending") == []


def test_embedding_blob_round_trip(fresh_db: Path) -> None:
    pid = dbmod.add_person("Alice")
    blob = _emb(42)
    dbmod.add_face(pid, crop_path="faces/1/x.jpg", embedding=blob, source="upload")
    rows = dbmod.list_faces_for(pid)
    assert len(rows) == 1
    assert rows[0]["embedding"] == blob


def test_load_all_embeddings(fresh_db: Path) -> None:
    a = dbmod.add_person("A")
    b = dbmod.add_person("B")
    dbmod.add_face(a, crop_path="faces/1/x.jpg", embedding=_emb(1), source="upload")
    dbmod.add_face(b, crop_path="faces/2/y.jpg", embedding=_emb(2), source="upload")
    rows = dbmod.load_all_embeddings()
    person_ids = sorted(r["person_id"] for r in rows)
    assert person_ids == sorted([a, b])
    assert all(len(r["embedding"]) == 2048 for r in rows)
