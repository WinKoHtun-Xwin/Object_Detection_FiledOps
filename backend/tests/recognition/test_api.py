"""Tests for the recognition REST API."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


FIXTURE = Path(__file__).parent.parent / "fixtures" / "face_test.jpg"


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from app import config as cfg
    from app.recognition import db as dbmod
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

    # Patch FaceEngine to return a deterministic fake face for any input.
    import app.recognition as rec_pkg
    import app.recognition.engine as engmod

    fake_emb = np.zeros(512, dtype=np.float32); fake_emb[0] = 1.0
    fake_crop = np.full((112, 112, 3), 200, dtype=np.uint8)

    def fake_detect(self, frame_bgr):
        return [engmod.FaceResult(crop_bgr=fake_crop, embedding=fake_emb, bbox=(0,0,1,1), det_score=0.99)]

    monkeypatch.setattr(engmod.FaceEngine, "detect_and_embed", fake_detect)
    rec_pkg.gallery.reload()

    from app.main import create_app
    return TestClient(create_app())


def test_post_people_creates_person_with_face(client: TestClient) -> None:
    with FIXTURE.open("rb") as f:
        r = client.post(
            "/api/people",
            data={"name": "Alice"},
            files={"files": ("face.jpg", f, "image/jpeg")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "Alice"
    assert body["face_count"] == 1

    r2 = client.get("/api/people")
    assert r2.status_code == 200
    people = r2.json()
    assert len(people) == 1
    assert people[0]["name"] == "Alice"


def test_delete_person_cascades(client: TestClient) -> None:
    with FIXTURE.open("rb") as f:
        r = client.post(
            "/api/people",
            data={"name": "Bob"},
            files={"files": ("face.jpg", f, "image/jpeg")},
        )
    pid = r.json()["id"]
    r = client.delete(f"/api/people/{pid}")
    assert r.status_code == 200
    r = client.get("/api/people")
    assert r.json() == []


def test_label_review_item_promotes_to_gallery(client: TestClient, tmp_path: Path) -> None:
    from app.recognition import db as dbmod
    crop_rel = "queue/test.jpg"
    (tmp_path / "faces" / "queue").mkdir(parents=True, exist_ok=True)
    (tmp_path / "faces" / crop_rel).write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    emb = np.zeros(512, dtype=np.float32); emb[0] = 1.0
    qid = dbmod.enqueue_review(
        crop_path=crop_rel,
        embedding=emb.tobytes(),
        source_clip="cam-A/2026-05-26/100000.mp4",
        suggested_id=None, suggested_score=None,
    )
    r = client.post(f"/api/review/{qid}/label", json={"new_name": "Carol"})
    assert r.status_code == 200
    person_id = r.json()["person_id"]
    persons = client.get("/api/people").json()
    assert any(p["id"] == person_id and p["name"] == "Carol" for p in persons)
    pending = dbmod.list_queue(status="pending")
    assert pending == []
