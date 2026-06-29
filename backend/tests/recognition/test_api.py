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
    import app.api.recognition as recmod
    monkeypatch.setattr(recmod, "settings", patched)
    dbmod._reset_for_tests()
    dbmod.init_schema()

    # Patch FaceEngine to return a deterministic fake face for any input.
    import app.recognition as rec_pkg
    import app.recognition.engine as engmod

    fake_emb = np.zeros(512, dtype=np.float32); fake_emb[0] = 1.0
    # Sharp, large, confident face so it passes the quality gate.
    fake_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    fake_crop[::2] = 255

    def fake_detect(self, frame_bgr):
        return [engmod.FaceResult(crop_bgr=fake_crop, embedding=fake_emb, bbox=(0,0,100,100), det_score=0.99)]

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


def test_purge_lowquality_dismisses_blurry_keeps_sharp(client: TestClient, tmp_path: Path) -> None:
    import cv2
    from app.recognition import db as dbmod

    qdir = tmp_path / "faces" / "queue"
    qdir.mkdir(parents=True, exist_ok=True)
    sharp = np.zeros((112, 112, 3), dtype=np.uint8)
    for i in range(0, 112, 16):
        for j in range(0, 112, 16):
            if (i // 16 + j // 16) % 2 == 0:
                sharp[i:i + 16, j:j + 16] = 255
    flat = np.full((112, 112, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(qdir / "sharp.jpg"), sharp)
    cv2.imwrite(str(qdir / "flat.jpg"), flat)

    emb = np.zeros(512, dtype=np.float32); emb[0] = 1.0
    dbmod.enqueue_review("queue/sharp.jpg", emb.tobytes(), None, None, None)
    dbmod.enqueue_review("queue/flat.jpg", emb.tobytes(), None, None, None)

    r = client.post("/api/review/purge-lowquality")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scanned"] == 2
    assert body["dismissed"] == 1

    pending = dbmod.list_queue(status="pending")
    assert len(pending) == 1
    assert pending[0]["crop_path"] == "queue/sharp.jpg"
