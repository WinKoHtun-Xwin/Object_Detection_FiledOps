# Face Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add InsightFace-based face recognition that tags recorded clips with the people who appear in them, with upload-based and Frigate-style review-queue registration, plus a search UX that lists clips by person.

**Architecture:** Recognition runs in a background worker thread after each motion event closes. The worker samples ~1 frame/sec from the clip, runs InsightFace detect+embed, matches each face against a SQLite-backed gallery via cosine similarity, and writes either a `clip_sightings` row (high match) or a `review_queue` row (low/borderline match). Frontend gains `react-router-dom` with three pages: Live, People, Review.

**Tech Stack:** Python 3.12 / FastAPI / SQLite (stdlib) / InsightFace `buffalo_l` (ONNX) / OpenCV. React + TypeScript + Zustand + react-router-dom on the frontend.

**Spec:** [docs/superpowers/specs/2026-05-26-face-recognition-design.md](../specs/2026-05-26-face-recognition-design.md)

---

## File structure

**New (backend):**
- `backend/app/recognition/__init__.py` — exports `face_engine`, `gallery`, `recognition_worker` singletons
- `backend/app/recognition/db.py` — SQLite wrapper (connection, schema, CRUD)
- `backend/app/recognition/engine.py` — `FaceEngine` wrapping InsightFace
- `backend/app/recognition/gallery.py` — `Gallery` in-memory matcher
- `backend/app/recognition/worker.py` — `RecognitionWorker` background thread
- `backend/app/api/recognition.py` — REST endpoints

**New (backend tests):**
- `backend/tests/recognition/__init__.py`
- `backend/tests/recognition/test_db.py`
- `backend/tests/recognition/test_gallery.py`
- `backend/tests/recognition/test_worker.py`
- `backend/tests/recognition/test_api.py`
- `backend/tests/recording/test_recorder_callback.py` (extends Phase 1 tests)

**Modified (backend):**
- `backend/pyproject.toml` — add `insightface`, `onnxruntime-gpu`
- `backend/app/config.py` — new fields
- `backend/app/main.py` — lifespan startup/shutdown, `/faces` static mount, include recognition router
- `backend/app/recording/recorder.py` — `on_clip_closed` callback
- `backend/app/recording/manager.py` — pass callback to new recorders

**New (frontend):**
- `frontend/src/pages/LivePage.tsx` — current App body (just moved)
- `frontend/src/pages/PeoplePage.tsx`
- `frontend/src/pages/PersonDetailPage.tsx`
- `frontend/src/pages/ReviewPage.tsx`
- `frontend/src/api/recognition.ts` — fetch helpers
- `frontend/src/layout/NavBar.tsx` — top tabs

**Modified (frontend):**
- `frontend/package.json` — add `react-router-dom`
- `frontend/src/main.tsx` — wrap in `<BrowserRouter>`
- `frontend/src/App.tsx` — thin layout shell with `<NavBar />` + `<Outlet />`
- `frontend/src/clips/ClipsPanel.tsx` — split into a fetching hook + pure renderer

---

## Task 0: Dependencies + scratch fixtures

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `frontend/package.json`

- [ ] **Step 1:** Add Python deps to `backend/pyproject.toml`. Find the `dependencies = [...]` block and add `"insightface>=0.7"`, `"onnxruntime-gpu>=1.18"` (with `onnxruntime` fallback if GPU build unavailable). Run `cd backend && pip install -e .`.

- [ ] **Step 2:** Verify install:

```bash
cd backend
python -c "import insightface; print(insightface.__version__)"
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```
Expected: a version number for insightface; a list including `'CUDAExecutionProvider'` (or at least `'CPUExecutionProvider'`).

- [ ] **Step 3:** Add frontend dep:
```bash
cd frontend
npm install react-router-dom
```
Verify `package.json` now lists `"react-router-dom"`.

- [ ] **Step 4:** Commit.

```bash
git add backend/pyproject.toml frontend/package.json frontend/package-lock.json
git commit -m "chore: add insightface, onnxruntime-gpu, react-router-dom"
```

---

## Task 1: Config additions

**Files:**
- Modify: `backend/app/config.py`

- [ ] **Step 1:** Read the current `backend/app/config.py`. Then add a `FACES_DIR`, `DB_PATH` module constants and new fields to the `Settings` dataclass. The final file must read:

```python
"""Application settings — single source of truth for paths, device, defaults."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BACKEND_ROOT / "weights"
CLIPS_DIR = BACKEND_ROOT / "data" / "clips"
FACES_DIR = BACKEND_ROOT / "data" / "faces"
DB_PATH = BACKEND_ROOT / "data" / "app.db"


@dataclass(frozen=True)
class Settings:
    weights_dir: Path = WEIGHTS_DIR
    default_yolo_detect: str = "yolo26n.pt"
    default_yolo_pose: str = "yolo26n-pose.pt"
    jpeg_quality_hint: int = 70
    frame_max_side: int = 640
    cors_origins: tuple[str, ...] = ("http://localhost:5173","http://localhost:8000", "http://localhost:8001")

    # --- recording ---
    clips_dir: Path = CLIPS_DIR
    clip_seconds: int = 15
    pre_roll_seconds: float = 2.0
    post_roll_seconds: float = 2.0
    motion_threshold: float = 0.015
    motion_classes: tuple[str, ...] = ("person",)
    clip_fps: int = 15

    # --- recognition ---
    faces_dir: Path = FACES_DIR
    db_path: Path = DB_PATH
    match_high: float = 0.55
    match_low: float = 0.40
    recognition_sample_fps: float = 1.0
    recognition_device: str = "cuda"


settings = Settings()
```

- [ ] **Step 2:** Run `cd backend && pytest -q` — expect existing 17 tests still pass.

- [ ] **Step 3:** Commit.

```bash
git add backend/app/config.py
git commit -m "feat(recognition): add config fields"
```

---

## Task 2: SQLite wrapper — failing test

**Files:**
- Create: `backend/tests/recognition/__init__.py` (empty)
- Create: `backend/tests/recognition/test_db.py`

- [ ] **Step 1:** Create empty `backend/tests/recognition/__init__.py`.

- [ ] **Step 2:** Write `backend/tests/recognition/test_db.py`:

```python
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
    # Second call must not error.
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
    assert all(len(r["embedding"]) == 2048 for r in rows)   # 512 * 4
```

- [ ] **Step 3:** Run, expect ImportError.

```bash
cd backend && pytest tests/recognition/test_db.py -v
```

---

## Task 3: SQLite wrapper — implementation

**Files:**
- Create: `backend/app/recognition/__init__.py` (empty for now)
- Create: `backend/app/recognition/db.py`

- [ ] **Step 1:** Create empty `backend/app/recognition/__init__.py`.

- [ ] **Step 2:** Create `backend/app/recognition/db.py`:

```python
"""SQLite wrapper for the recognition subsystem.

One connection per thread (sqlite3 isn't thread-safe). Schema is created
idempotently on init.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any

from app.config import settings

_local = threading.local()
_init_lock = threading.Lock()
_initialized = False

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS persons (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT UNIQUE NOT NULL,
  created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  person_id    INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  crop_path    TEXT NOT NULL,
  embedding    BLOB NOT NULL,
  source       TEXT NOT NULL,
  created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS faces_person_idx ON faces(person_id);

CREATE TABLE IF NOT EXISTS review_queue (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  crop_path       TEXT NOT NULL,
  embedding       BLOB NOT NULL,
  source_clip     TEXT,
  suggested_id    INTEGER REFERENCES persons(id) ON DELETE SET NULL,
  suggested_score REAL,
  status          TEXT NOT NULL DEFAULT 'pending',
  created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS review_status_idx ON review_queue(status, created_at);

CREATE TABLE IF NOT EXISTS clip_sightings (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  clip_path   TEXT NOT NULL,
  person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
  confidence  REAL NOT NULL,
  frame_ts    REAL NOT NULL,
  created_at  REAL NOT NULL,
  UNIQUE(clip_path, person_id)
);
CREATE INDEX IF NOT EXISTS sightings_clip_idx ON clip_sightings(clip_path);
CREATE INDEX IF NOT EXISTS sightings_person_idx ON clip_sightings(person_id);
"""


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(settings.db_path), check_same_thread=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        _local.conn = conn
    return conn


def init_schema() -> None:
    global _initialized
    with _init_lock:
        conn = get_conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        _initialized = True


def _reset_for_tests() -> None:
    """Clear thread-local connection cache. Test-only."""
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
    global _initialized
    _initialized = False


# --- persons ---

def add_person(name: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO persons(name, created_at) VALUES(?, ?)",
        (name, time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_persons() -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT p.id, p.name, p.created_at,
               (SELECT COUNT(*) FROM faces f WHERE f.person_id = p.id) AS face_count,
               (SELECT crop_path FROM faces f WHERE f.person_id = p.id
                ORDER BY id DESC LIMIT 1) AS latest_crop_path
        FROM persons p
        ORDER BY p.id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_person(person_id: int) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    return dict(row) if row else None


def delete_person(person_id: int) -> None:
    conn = get_conn()
    conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.commit()


# --- faces ---

def add_face(person_id: int, crop_path: str, embedding: bytes, source: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO faces(person_id, crop_path, embedding, source, created_at) "
        "VALUES(?, ?, ?, ?, ?)",
        (person_id, crop_path, embedding, source, time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_faces_for(person_id: int) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, person_id, crop_path, embedding, source, created_at "
        "FROM faces WHERE person_id = ? ORDER BY id DESC",
        (person_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def load_all_embeddings() -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute("SELECT person_id, embedding FROM faces").fetchall()
    return [dict(r) for r in rows]


# --- review queue ---

def enqueue_review(
    crop_path: str,
    embedding: bytes,
    source_clip: str | None,
    suggested_id: int | None,
    suggested_score: float | None,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO review_queue(crop_path, embedding, source_clip, "
        "suggested_id, suggested_score, status, created_at) "
        "VALUES(?, ?, ?, ?, ?, 'pending', ?)",
        (crop_path, embedding, source_clip, suggested_id, suggested_score, time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_queue(status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT q.id, q.crop_path, q.source_clip, q.suggested_id,
               q.suggested_score, q.status, q.created_at, p.name AS suggested_name
        FROM review_queue q
        LEFT JOIN persons p ON p.id = q.suggested_id
        WHERE q.status = ?
        ORDER BY q.created_at DESC
        LIMIT ?
        """,
        (status, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_queue_item(queue_id: int) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, crop_path, embedding, source_clip, suggested_id, "
        "suggested_score, status, created_at FROM review_queue WHERE id = ?",
        (queue_id,),
    ).fetchone()
    return dict(row) if row else None


def update_queue_status(queue_id: int, status: str) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE review_queue SET status = ? WHERE id = ?",
        (status, queue_id),
    )
    conn.commit()


# --- sightings ---

def add_sighting(
    clip_path: str,
    person_id: int,
    confidence: float,
    frame_ts: float,
) -> None:
    conn = get_conn()
    # INSERT OR REPLACE keeps the highest-confidence sighting per (clip, person).
    existing = conn.execute(
        "SELECT confidence FROM clip_sightings "
        "WHERE clip_path = ? AND person_id = ?",
        (clip_path, person_id),
    ).fetchone()
    if existing is not None and existing["confidence"] >= confidence:
        return
    conn.execute(
        "INSERT INTO clip_sightings(clip_path, person_id, confidence, frame_ts, created_at) "
        "VALUES(?, ?, ?, ?, ?) "
        "ON CONFLICT(clip_path, person_id) DO UPDATE SET "
        "confidence = excluded.confidence, frame_ts = excluded.frame_ts",
        (clip_path, person_id, confidence, frame_ts, time.time()),
    )
    conn.commit()


def clips_for_person(person_id: int) -> list[dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT clip_path, confidence, frame_ts, created_at "
        "FROM clip_sightings WHERE person_id = ? "
        "ORDER BY created_at DESC",
        (person_id,),
    ).fetchall()
    return [dict(r) for r in rows]
```

- [ ] **Step 3:** Run the test file.

```bash
cd backend && pytest tests/recognition/test_db.py -v
```
Expected: 7 passed.

- [ ] **Step 4:** Commit.

```bash
git add backend/app/recognition/__init__.py backend/app/recognition/db.py backend/tests/recognition/__init__.py backend/tests/recognition/test_db.py
git commit -m "feat(recognition): SQLite schema + CRUD"
```

---

## Task 4: Gallery — failing test

**Files:**
- Create: `backend/tests/recognition/test_gallery.py`

- [ ] **Step 1:** Write:

```python
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

    # Query close to Alice
    noisy_alice = alice_emb + 0.01 * _unit(99)
    noisy_alice = noisy_alice / np.linalg.norm(noisy_alice)
    pid, score = g.match(noisy_alice)
    assert pid == alice
    assert score > 0.95


def test_match_returns_low_score_for_unrelated(fresh_db: Path) -> None:
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", _unit(1).tobytes(), source="upload")
    g = Gallery()

    far = _unit(500)   # very different from seed 1
    pid, score = g.match(far)
    # The closest is still alice, but cosine should be near 0.
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
    # g has not been told yet
    p, _ = g.match(_unit(1))
    assert p is None
    g.reload()
    p, _ = g.match(_unit(1))
    assert p == pid
```

- [ ] **Step 2:** Run, expect ImportError.

```bash
cd backend && pytest tests/recognition/test_gallery.py -v
```

---

## Task 5: Gallery — implementation

**Files:**
- Create: `backend/app/recognition/gallery.py`

- [ ] **Step 1:** Write:

```python
"""In-memory face gallery — brute-force cosine match against all stored embeddings."""
from __future__ import annotations

import logging
import threading

import numpy as np

from app.recognition import db as dbmod

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512


class Gallery:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._matrix: np.ndarray = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self._person_ids: list[int] = []
        self.reload()

    def reload(self) -> None:
        rows = dbmod.load_all_embeddings()
        with self._lock:
            self._person_ids = [int(r["person_id"]) for r in rows]
            if rows:
                buf = b"".join(r["embedding"] for r in rows)
                self._matrix = np.frombuffer(buf, dtype=np.float32).reshape(
                    len(rows), EMBEDDING_DIM
                ).copy()
            else:
                self._matrix = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        log.info("gallery reloaded: %d faces", len(self._person_ids))

    def add(self, person_id: int, embedding: np.ndarray) -> None:
        if embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(f"expected ({EMBEDDING_DIM},) embedding, got {embedding.shape}")
        with self._lock:
            self._matrix = np.vstack([self._matrix, embedding.astype(np.float32)])
            self._person_ids.append(int(person_id))

    def match(self, query: np.ndarray) -> tuple[int | None, float]:
        if query.shape != (EMBEDDING_DIM,):
            raise ValueError(f"expected ({EMBEDDING_DIM},) query, got {query.shape}")
        with self._lock:
            if self._matrix.shape[0] == 0:
                return None, 0.0
            scores = self._matrix @ query.astype(np.float32)
            idx = int(np.argmax(scores))
            return self._person_ids[idx], float(scores[idx])
```

- [ ] **Step 2:** Run, expect 5 passed.

```bash
cd backend && pytest tests/recognition/test_gallery.py -v
```

- [ ] **Step 3:** Add the gallery singleton to `backend/app/recognition/__init__.py`:

```python
"""Recognition subsystem."""
from app.recognition.gallery import Gallery

gallery = Gallery()

__all__ = ["gallery"]
```

- [ ] **Step 4:** Commit.

```bash
git add backend/app/recognition/gallery.py backend/app/recognition/__init__.py backend/tests/recognition/test_gallery.py
git commit -m "feat(recognition): Gallery cosine matcher"
```

---

## Task 6: FaceEngine wrapper

**Files:**
- Create: `backend/app/recognition/engine.py`

(No unit test — InsightFace download is slow and GPU-bound; engine is verified end-to-end via the worker and API tests where it is stubbed.)

- [ ] **Step 1:** Write `backend/app/recognition/engine.py`:

```python
"""InsightFace wrapper. Lazy-loads buffalo_l on first call.

Returns 512-d L2-normalized embeddings per detected face."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceResult:
    crop_bgr: np.ndarray              # 112×112×3 aligned crop
    embedding: np.ndarray             # 512-d float32, L2-normalized
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in source pixels
    det_score: float


class FaceEngine:
    def __init__(self, device: str | None = None) -> None:
        self._device = device or settings.recognition_device
        self._app: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        log.info("loading InsightFace buffalo_l (providers=%s)", providers)
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if self._device == "cuda" else -1, det_size=(640, 640))
        self._app = app

    def detect_and_embed(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        self._ensure_loaded()
        faces = self._app.get(frame_bgr)
        results: list[FaceResult] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            # Resize to 112×112 for storage consistency.
            import cv2
            crop_112 = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            results.append(
                FaceResult(
                    crop_bgr=crop_112,
                    embedding=emb,
                    bbox=(x1, y1, x2, y2),
                    det_score=float(f.det_score),
                )
            )
        return results
```

- [ ] **Step 2:** Update `backend/app/recognition/__init__.py` to also export `face_engine`:

```python
"""Recognition subsystem."""
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery

face_engine = FaceEngine()
gallery = Gallery()

__all__ = ["face_engine", "gallery"]
```

- [ ] **Step 3:** Verify imports work without crashing (engine is lazy, no models downloaded yet):

```bash
cd backend && python -c "from app.recognition import face_engine, gallery; print('ok')"
```
Expected: `ok`.

- [ ] **Step 4:** Commit.

```bash
git add backend/app/recognition/engine.py backend/app/recognition/__init__.py
git commit -m "feat(recognition): FaceEngine wrapping InsightFace buffalo_l"
```

---

## Task 7: RecognitionWorker — failing test

**Files:**
- Create: `backend/tests/recognition/test_worker.py`

- [ ] **Step 1:** Write:

```python
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

    # Seed gallery with Alice
    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    w = RecognitionWorker(engine=MagicMock(), gallery=g)
    # Stub _sample_frames to yield two synthetic frames at ts 0.0 and 1.0
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    monkeypatch.setattr(w, "_sample_frames", lambda path: iter([(0.0, frame), (1.0, frame)]))

    # Stub engine.detect_and_embed: every call returns one face matching Alice
    w._engine.detect_and_embed.return_value = [
        FaceResult(crop_bgr=_crop(), embedding=alice_emb, bbox=(0, 0, 100, 100), det_score=0.99)
    ]

    clip_path = env / "clips" / "cam-A" / "2026-05-26" / "100000.mp4"
    w._process(clip_path)

    sightings = dbmod.clips_for_person(alice)
    assert len(sightings) == 1
    assert sightings[0]["confidence"] > 0.99
    assert dbmod.list_queue(status="pending") == []


def test_worker_writes_review_for_unknown_face(env: Path, monkeypatch) -> None:
    from app.recognition.worker import RecognitionWorker
    from app.recognition.gallery import Gallery

    # Gallery has Alice
    alice_emb = _unit(1)
    alice = dbmod.add_person("Alice")
    dbmod.add_face(alice, "faces/1/a.jpg", alice_emb.tobytes(), source="upload")

    g = Gallery()
    w = RecognitionWorker(engine=MagicMock(), gallery=g)
    monkeypatch.setattr(
        w, "_sample_frames",
        lambda path: iter([(0.0, np.zeros((480, 640, 3), dtype=np.uint8))]),
    )
    # Return an embedding very unlike Alice
    unknown_emb = _unit(999)
    w._engine.detect_and_embed.return_value = [
        FaceResult(crop_bgr=_crop(), embedding=unknown_emb, bbox=(0, 0, 100, 100), det_score=0.99)
    ]

    clip_path = env / "clips" / "cam-A" / "2026-05-26" / "100000.mp4"
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
        # Wait up to 2s for queue to drain.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if w._q.empty():
                break
            time.sleep(0.05)
        assert w._q.empty()
    finally:
        w.stop()
```

- [ ] **Step 2:** Run, expect ImportError.

```bash
cd backend && pytest tests/recognition/test_worker.py -v
```

---

## Task 8: RecognitionWorker — implementation

**Files:**
- Create: `backend/app/recognition/worker.py`

- [ ] **Step 1:** Write:

```python
"""Background worker that processes closed clips for face recognition."""
from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery

log = logging.getLogger(__name__)


class RecognitionWorker:
    def __init__(self, engine: FaceEngine, gallery: Gallery) -> None:
        self._engine = engine
        self._gallery = gallery
        self._q: queue.Queue[Path] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        t = threading.Thread(target=self._run, name="RecognitionWorker", daemon=True)
        t.start()
        self._thread = t
        log.info("recognition worker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("recognition worker stopped")

    def enqueue(self, clip_path: Path) -> None:
        self._q.put(clip_path)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                path = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(path)
            except Exception:
                log.exception("recognition failed for %s", path)
            finally:
                self._q.task_done()

    def _process(self, clip_path: Path) -> None:
        if not clip_path.exists():
            log.warning("clip not found, skipping: %s", clip_path)
            return
        log.info("processing clip: %s", clip_path)
        best_per_person: dict[int, tuple[float, float]] = {}
        rel_clip = self._relative_clip_path(clip_path)

        for ts, frame in self._sample_frames(clip_path):
            for face in self._engine.detect_and_embed(frame):
                pid, score = self._gallery.match(face.embedding)
                if pid is not None and score >= settings.match_high:
                    prev = best_per_person.get(pid)
                    if prev is None or score > prev[0]:
                        best_per_person[pid] = (score, ts)
                elif settings.match_low <= score < settings.match_high and pid is not None:
                    self._save_review(face, rel_clip, suggested_id=pid, suggested_score=score)
                else:
                    self._save_review(face, rel_clip, suggested_id=None, suggested_score=None)

        for pid, (score, ts) in best_per_person.items():
            dbmod.add_sighting(rel_clip, pid, score, ts)
        log.info("clip %s: %d sightings, queue may have grown", clip_path.name, len(best_per_person))

    def _sample_frames(self, clip_path: Path) -> Iterator[tuple[float, np.ndarray]]:
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            log.warning("VideoCapture failed: %s", clip_path)
            return
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or settings.clip_fps
            step = max(1, int(round(fps / settings.recognition_sample_fps)))
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    yield (i / fps, frame)
                i += 1
        finally:
            cap.release()

    def _save_review(
        self,
        face,
        source_clip: str,
        suggested_id: int | None,
        suggested_score: float | None,
    ) -> None:
        crop_name = f"queue_{uuid.uuid4().hex}.jpg"
        crop_rel = Path("queue") / crop_name
        crop_abs = settings.faces_dir / crop_rel
        crop_abs.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_abs), face.crop_bgr)
        dbmod.enqueue_review(
            crop_path=str(crop_rel).replace("\\", "/"),
            embedding=face.embedding.tobytes(),
            source_clip=source_clip,
            suggested_id=suggested_id,
            suggested_score=suggested_score,
        )

    @staticmethod
    def _relative_clip_path(clip_path: Path) -> str:
        try:
            rel = clip_path.resolve().relative_to(settings.clips_dir.resolve())
            return rel.as_posix()
        except ValueError:
            return clip_path.name
```

- [ ] **Step 2:** Update `backend/app/recognition/__init__.py`:

```python
"""Recognition subsystem."""
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery
from app.recognition.worker import RecognitionWorker

face_engine = FaceEngine()
gallery = Gallery()
recognition_worker = RecognitionWorker(engine=face_engine, gallery=gallery)

__all__ = ["face_engine", "gallery", "recognition_worker"]
```

- [ ] **Step 3:** Run tests.

```bash
cd backend && pytest tests/recognition/ -v
```
Expected: all recognition tests pass (db: 7, gallery: 5, worker: 3 = 15).

- [ ] **Step 4:** Commit.

```bash
git add backend/app/recognition/worker.py backend/app/recognition/__init__.py backend/tests/recognition/test_worker.py
git commit -m "feat(recognition): RecognitionWorker background thread"
```

---

## Task 9: Wire `on_clip_closed` callback into MotionRecorder

**Files:**
- Modify: `backend/app/recording/recorder.py`
- Create: `backend/tests/recording/test_recorder_callback.py`

- [ ] **Step 1:** Write the failing test `backend/tests/recording/test_recorder_callback.py`:

```python
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
    # warm up
    for _ in range(5):
        rec.feed(_frame(), [_box(0.5)], ts=t); t += dt
    # 25s of motion (rotates once)
    for i in range(25 * 15):
        rec.feed(_frame(), [_box(0.5 + 0.02 * (i % 2 + 1))], ts=t); t += dt
    # post-roll to force close
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
    # multiple .mp4s exist on disk
    mp4s = sorted(rec_root.rglob("*.mp4"))
    assert len(mp4s) >= 2
    # callback received the FIRST clip
    assert seen_paths[0] == mp4s[0]
```

- [ ] **Step 2:** Run — expect failure (constructor doesn't accept `on_clip_closed`).

```bash
cd backend && pytest tests/recording/test_recorder_callback.py -v
```

- [ ] **Step 3:** Modify `backend/app/recording/recorder.py`. Read the existing file, then apply these specific changes:

(a) Imports at top — add `Callable`:

```python
from collections.abc import Callable
```

(b) `MotionRecorder.__init__` signature and body — add `on_clip_closed` param and track the event's first clip path:

```python
    def __init__(
        self,
        camera_id: str,
        frame_size: tuple[int, int],
        on_clip_closed: Callable[[Path], None] | None = None,
    ) -> None:
        self._camera_id = camera_id
        self._frame_w, self._frame_h = frame_size
        self._on_clip_closed = on_clip_closed

        self._detector = MotionDetector(
            allowed_classes=settings.motion_classes,
            threshold=settings.motion_threshold,
        )
        self._fps = settings.clip_fps
        ring_len = max(1, int(round(settings.pre_roll_seconds * self._fps)))
        self._ring: deque[tuple[float, np.ndarray]] = deque(maxlen=ring_len)

        self._state = _State.IDLE
        self._writer: cv2.VideoWriter | None = None
        self._clip_started_at: float = 0.0
        self._last_motion_ts: float = 0.0
        self._event_open = False
        self._event_first_clip: Path | None = None
```

(c) `_begin_event` — record the first clip path of the event:

```python
    def _begin_event(self, trigger_frame: np.ndarray, trigger_ts: float) -> None:
        first_ts = self._ring[0][0] if self._ring else trigger_ts
        path = self._make_clip_path(first_ts)
        self._open_writer(path, first_ts)
        for _ts, f in self._ring:
            self._writer_write(f)
        snap_path = path.with_suffix(".jpg")
        cv2.imwrite(str(snap_path), trigger_frame)
        self._event_open = True
        self._event_first_clip = path
        self._last_motion_ts = trigger_ts
```

(d) The post-roll-close branch in `feed`: when transitioning RECORDING → IDLE, invoke the callback with `_event_first_clip`, then clear it. The current branch is:

```python
                if ts - self._last_motion_ts > settings.post_roll_seconds:
                    self._close_writer()
                    self._event_open = False
                    self._state = _State.IDLE
                    return
```

Change to:

```python
                if ts - self._last_motion_ts > settings.post_roll_seconds:
                    self._close_writer()
                    self._event_open = False
                    self._state = _State.IDLE
                    self._fire_close_callback()
                    return
```

(e) Public `close()` also needs to fire if an event was active:

```python
    def close(self) -> None:
        was_event = self._event_open
        self._close_writer()
        self._event_open = False
        self._state = _State.IDLE
        if was_event:
            self._fire_close_callback()
```

(f) Add a helper at the bottom of the class (before `_make_clip_path` is fine):

```python
    def _fire_close_callback(self) -> None:
        first = self._event_first_clip
        self._event_first_clip = None
        if first is not None and self._on_clip_closed is not None:
            try:
                self._on_clip_closed(first)
            except Exception:
                log.exception("on_clip_closed callback failed")
```

- [ ] **Step 4:** Run all recording tests.

```bash
cd backend && pytest tests/recording/ -v
```
Expected: all (3 prior recorder tests + 3 manager tests + 3 callback tests = 9) pass.

- [ ] **Step 5:** Commit.

```bash
git add backend/app/recording/recorder.py backend/tests/recording/test_recorder_callback.py
git commit -m "feat(recording): on_clip_closed callback fired once per motion event"
```

---

## Task 10: Wire RecorderManager → worker.enqueue

**Files:**
- Modify: `backend/app/recording/manager.py`

- [ ] **Step 1:** Read the current `backend/app/recording/manager.py` and modify `feed()` to create new `MotionRecorder`s with the callback. Final body:

```python
"""Per-process recorder registry — one MotionRecorder per camera_id."""
from __future__ import annotations

import logging

import numpy as np

from app.recognition import recognition_worker
from app.recording.recorder import MotionRecorder

log = logging.getLogger(__name__)


class RecorderManager:
    def __init__(self) -> None:
        self._recorders: dict[str, MotionRecorder] = {}

    def feed(self, camera_id: str, frame_bgr: np.ndarray, detections: list[dict]) -> None:
        if not camera_id:
            return
        rec = self._recorders.get(camera_id)
        if rec is None:
            h, w = frame_bgr.shape[:2]
            rec = MotionRecorder(
                camera_id=camera_id,
                frame_size=(w, h),
                on_clip_closed=recognition_worker.enqueue,
            )
            self._recorders[camera_id] = rec
            log.info("recorder created: %s (%dx%d)", camera_id, w, h)
        rec.feed(frame_bgr, detections)

    def close(self, camera_id: str) -> None:
        rec = self._recorders.pop(camera_id, None)
        if rec is not None:
            rec.close()
            log.info("recorder closed: %s", camera_id)

    def close_all(self) -> None:
        for rec in self._recorders.values():
            rec.close()
        self._recorders.clear()


recorder_manager = RecorderManager()
```

- [ ] **Step 2:** Run full recording test suite — they pass `on_clip_closed=callback` explicitly so they're not affected by manager changes, but verify.

```bash
cd backend && pytest tests/recording/ tests/recognition/ -v
```
Expected: all pass.

- [ ] **Step 3:** Commit.

```bash
git add backend/app/recording/manager.py
git commit -m "feat(recording): forward closed clips to recognition worker"
```

---

## Task 11: Recognition API — failing test

**Files:**
- Create: `backend/tests/recognition/test_api.py`
- Create: `backend/tests/fixtures/face_test.jpg` (synthetic — see step 1)

- [ ] **Step 1:** Create a tiny synthetic JPEG used as a "face image" by tests. We will stub the FaceEngine so its content doesn't matter:

```bash
cd backend && python -c "import cv2, numpy as np; img = np.full((224,224,3), 128, dtype=np.uint8); cv2.imwrite('tests/fixtures/face_test.jpg', img)" 2>/dev/null || (mkdir -p tests/fixtures && python -c "import cv2, numpy as np; img = np.full((224,224,3), 128, dtype=np.uint8); cv2.imwrite('tests/fixtures/face_test.jpg', img)")
```

- [ ] **Step 2:** Write `backend/tests/recognition/test_api.py`:

```python
"""Tests for the recognition REST API."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

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

    # Patch FaceEngine to return a deterministic fake face for any input
    import app.recognition as rec_pkg
    import app.recognition.engine as engmod

    fake_emb = np.zeros(512, dtype=np.float32); fake_emb[0] = 1.0  # unit vector
    fake_crop = np.full((112, 112, 3), 200, dtype=np.uint8)

    def fake_detect(self, frame_bgr):
        return [engmod.FaceResult(crop_bgr=fake_crop, embedding=fake_emb, bbox=(0,0,1,1), det_score=0.99)]

    monkeypatch.setattr(engmod.FaceEngine, "detect_and_embed", fake_detect)
    rec_pkg.gallery.reload()  # clear any state

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
    # Manually insert a queue item.
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
```

- [ ] **Step 3:** Run, expect ImportError or 404s (router not yet wired).

```bash
cd backend && pytest tests/recognition/test_api.py -v
```

---

## Task 12: Recognition API — implementation

**Files:**
- Create: `backend/app/api/recognition.py`
- Modify: `backend/app/main.py`

- [ ] **Step 1:** Create `backend/app/api/recognition.py`:

```python
"""Recognition REST endpoints."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.recognition import db as dbmod
from app.recognition import face_engine, gallery

log = logging.getLogger(__name__)
router = APIRouter()


def _person_to_dict(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "face_count": row.get("face_count", 0),
        "latest_face_url": (
            f"/faces/{row['latest_crop_path']}"
            if row.get("latest_crop_path") else None
        ),
    }


@router.get("/people")
def list_people() -> list[dict]:
    return [_person_to_dict(p) for p in dbmod.list_persons()]


@router.post("/people")
async def create_person(
    name: Annotated[str, Form(min_length=1, max_length=64)],
    files: Annotated[list[UploadFile], File()],
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="at least one image required")

    pid = dbmod.add_person(name)
    faces_added = 0

    person_dir = settings.faces_dir / str(pid)
    person_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        data = await upload.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("could not decode upload %s", upload.filename)
            continue
        for face in face_engine.detect_and_embed(img):
            crop_name = f"upload_{uuid.uuid4().hex}.jpg"
            crop_rel = f"{pid}/{crop_name}"
            cv2.imwrite(str(settings.faces_dir / crop_rel), face.crop_bgr)
            dbmod.add_face(pid, crop_path=crop_rel, embedding=face.embedding.tobytes(), source="upload")
            gallery.add(pid, face.embedding)
            faces_added += 1

    return {"id": pid, "name": name, "face_count": faces_added}


@router.delete("/people/{person_id}")
def delete_person(person_id: int) -> dict:
    person = dbmod.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="person not found")
    dbmod.delete_person(person_id)
    gallery.reload()
    return {"ok": True}


@router.get("/people/{person_id}")
def get_person(person_id: int) -> dict:
    person = dbmod.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="person not found")
    faces = dbmod.list_faces_for(person_id)
    return {
        "id": person["id"],
        "name": person["name"],
        "faces": [
            {
                "id": f["id"],
                "crop_url": f"/faces/{f['crop_path']}",
                "source": f["source"],
                "created_at": f["created_at"],
            }
            for f in faces
        ],
    }


@router.get("/people/{person_id}/clips")
def list_person_clips(person_id: int) -> list[dict]:
    person = dbmod.get_person(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="person not found")
    rows = dbmod.clips_for_person(person_id)
    out = []
    for r in rows:
        clip_rel = r["clip_path"]
        mp4_url = f"/clips/{clip_rel}"
        jpg_url = mp4_url.rsplit(".", 1)[0] + ".jpg"
        out.append(
            {
                "clip_path": clip_rel,
                "mp4_url": mp4_url,
                "jpg_url": jpg_url,
                "confidence": r["confidence"],
                "frame_ts": r["frame_ts"],
                "created_at": r["created_at"],
            }
        )
    return out


@router.get("/review")
def list_review(limit: int = 50) -> list[dict]:
    rows = dbmod.list_queue(status="pending", limit=limit)
    return [
        {
            "id": r["id"],
            "crop_url": f"/faces/{r['crop_path']}",
            "suggested": (
                {
                    "id": r["suggested_id"],
                    "name": r["suggested_name"],
                    "score": r["suggested_score"],
                }
                if r["suggested_id"] else None
            ),
            "source_clip": r["source_clip"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.post("/review/{queue_id}/label")
def label_review(queue_id: int, body: dict) -> dict:
    item = dbmod.get_queue_item(queue_id)
    if not item:
        raise HTTPException(status_code=404, detail="queue item not found")

    person_id = body.get("person_id")
    new_name = body.get("new_name")
    if person_id is None and not new_name:
        raise HTTPException(status_code=400, detail="provide person_id or new_name")

    if person_id is None:
        person_id = dbmod.add_person(str(new_name))

    # Move/copy the crop file into the person's folder.
    src = settings.faces_dir / item["crop_path"]
    dest_dir = settings.faces_dir / str(person_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = f"queue_{uuid.uuid4().hex}.jpg"
    dest_rel = f"{person_id}/{dest_name}"
    dest_abs = settings.faces_dir / dest_rel
    if src.exists():
        dest_abs.write_bytes(src.read_bytes())

    embedding = np.frombuffer(item["embedding"], dtype=np.float32)
    dbmod.add_face(person_id, crop_path=dest_rel, embedding=item["embedding"], source="queue")
    gallery.add(person_id, embedding)
    dbmod.update_queue_status(queue_id, "labeled")

    return {"person_id": person_id}


@router.post("/review/{queue_id}/dismiss")
def dismiss_review(queue_id: int) -> dict:
    item = dbmod.get_queue_item(queue_id)
    if not item:
        raise HTTPException(status_code=404, detail="queue item not found")
    dbmod.update_queue_status(queue_id, "dismissed")
    return {"ok": True}
```

- [ ] **Step 2:** Modify `backend/app/main.py` to register the new router, mount `/faces`, init the DB on startup, and start the worker.

Final `backend/app/main.py`:

```python
"""FastAPI app factory + route registration."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import clips, health, recognition, stream
from app.config import settings
from app.recognition import db as dbmod
from app.recognition import gallery, recognition_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.clips_dir.mkdir(parents=True, exist_ok=True)
    settings.faces_dir.mkdir(parents=True, exist_ok=True)
    dbmod.init_schema()
    gallery.reload()
    recognition_worker.start()
    try:
        yield
    finally:
        recognition_worker.stop()


def create_app() -> FastAPI:
    app = FastAPI(title="Object Detection Playground", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(clips.router, prefix="/api")
    app.include_router(recognition.router, prefix="/api")
    app.include_router(stream.router)

    settings.clips_dir.mkdir(parents=True, exist_ok=True)
    settings.faces_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/clips", StaticFiles(directory=settings.clips_dir), name="clips")
    app.mount("/faces", StaticFiles(directory=settings.faces_dir), name="faces")
    return app


app = create_app()
```

- [ ] **Step 3:** Run all tests.

```bash
cd backend && pytest -q
```
Expected: all pass (17 prior + db 7 + gallery 5 + worker 3 + recorder callback 3 + api 3 = 38).

- [ ] **Step 4:** Commit.

```bash
git add backend/app/api/recognition.py backend/app/main.py backend/tests/recognition/test_api.py backend/tests/fixtures/face_test.jpg
git commit -m "feat(recognition): REST API + lifespan integration"
```

---

## Task 13: Frontend — routing shell + extract LivePage

**Files:**
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/pages/LivePage.tsx`
- Create: `frontend/src/layout/NavBar.tsx`

- [ ] **Step 1:** Wrap with router. Replace `frontend/src/main.tsx`:

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import { LivePage } from './pages/LivePage';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<LivePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
```

(People + Review routes will be added in Task 14.)

- [ ] **Step 2:** Create `frontend/src/layout/NavBar.tsx`:

```tsx
import { NavLink } from 'react-router-dom';

const navStyle = (active: boolean): React.CSSProperties => ({
  padding: '8px 16px',
  color: active ? '#fff' : '#aaa',
  background: active ? '#222' : 'transparent',
  borderRadius: 4,
  textDecoration: 'none',
  fontSize: 14,
});

export function NavBar() {
  return (
    <nav style={{
      display: 'flex',
      gap: 8,
      padding: '8px 16px',
      background: '#0c0c0c',
      borderBottom: '1px solid #222',
    }}>
      <NavLink to="/" end style={({ isActive }) => navStyle(isActive)}>Live</NavLink>
      <NavLink to="/people" style={({ isActive }) => navStyle(isActive)}>People</NavLink>
      <NavLink to="/review" style={({ isActive }) => navStyle(isActive)}>Review</NavLink>
    </nav>
  );
}
```

- [ ] **Step 3:** Create `frontend/src/pages/LivePage.tsx`. Take the *entire current body* of `App.tsx` (the JSX returned from the `App` function, plus all the hooks and helper logic at the top of `App`) and move it into a new `LivePage` function. Keep all imports it needs in the new file. The new file should export `function LivePage() { ... }`.

You may copy the file content verbatim, then rename `App` → `LivePage` and update the export to `export function LivePage`. The `sizeLabel` and `modeLabel` helpers at the bottom also move with it.

- [ ] **Step 4:** Replace `frontend/src/App.tsx` with a thin shell:

```tsx
import { Outlet } from 'react-router-dom';
import { NavBar } from './layout/NavBar';
import './App.css';

export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <NavBar />
      <div style={{ flex: 1, minHeight: 0 }}>
        <Outlet />
      </div>
    </div>
  );
}
```

- [ ] **Step 5:** Verify build.

```bash
cd frontend && npm run build
```
Expected: success, no TS errors. The Live tab works exactly as before; People/Review routes 404 for now (they're added next task).

- [ ] **Step 6:** Commit.

```bash
git add frontend/src/main.tsx frontend/src/App.tsx frontend/src/pages/LivePage.tsx frontend/src/layout/NavBar.tsx
git commit -m "feat(frontend): introduce react-router shell with LivePage"
```

---

## Task 14: Frontend — recognition API helpers + PeoplePage

**Files:**
- Create: `frontend/src/api/recognition.ts`
- Create: `frontend/src/pages/PeoplePage.tsx`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1:** Create `frontend/src/api/recognition.ts`:

```ts
const API_BASE = 'http://localhost:8000';

export interface PersonSummary {
  id: number;
  name: string;
  face_count: number;
  latest_face_url: string | null;
}

export interface PersonFace {
  id: number;
  crop_url: string;
  source: 'upload' | 'queue';
  created_at: number;
}

export interface PersonDetail {
  id: number;
  name: string;
  faces: PersonFace[];
}

export interface PersonClip {
  clip_path: string;
  mp4_url: string;
  jpg_url: string;
  confidence: number;
  frame_ts: number;
  created_at: number;
}

export interface ReviewItem {
  id: number;
  crop_url: string;
  suggested: { id: number; name: string; score: number } | null;
  source_clip: string | null;
  created_at: number;
}

export function absUrl(path: string): string {
  return path.startsWith('http') ? path : `${API_BASE}${path}`;
}

export async function listPeople(): Promise<PersonSummary[]> {
  const r = await fetch(`${API_BASE}/api/people`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function getPerson(id: number): Promise<PersonDetail> {
  const r = await fetch(`${API_BASE}/api/people/${id}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function listPersonClips(id: number): Promise<PersonClip[]> {
  const r = await fetch(`${API_BASE}/api/people/${id}/clips`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function createPerson(name: string, files: File[]): Promise<PersonSummary> {
  const fd = new FormData();
  fd.set('name', name);
  for (const f of files) fd.append('files', f);
  const r = await fetch(`${API_BASE}/api/people`, { method: 'POST', body: fd });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function deletePerson(id: number): Promise<void> {
  const r = await fetch(`${API_BASE}/api/people/${id}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
}

export async function listReview(limit = 50): Promise<ReviewItem[]> {
  const r = await fetch(`${API_BASE}/api/review?limit=${limit}`);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function labelReview(id: number, body: { person_id?: number; new_name?: string }): Promise<{ person_id: number }> {
  const r = await fetch(`${API_BASE}/api/review/${id}/label`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

export async function dismissReview(id: number): Promise<void> {
  const r = await fetch(`${API_BASE}/api/review/${id}/dismiss`, { method: 'POST' });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
}
```

- [ ] **Step 2:** Create `frontend/src/pages/PeoplePage.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  absUrl, createPerson, listPeople, type PersonSummary,
} from '../api/recognition';

export function PeoplePage() {
  const [people, setPeople] = useState<PersonSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [showAdd, setShowAdd] = useState(false);

  async function refresh() {
    try { setPeople(await listPeople()); setError(null); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
  }

  useEffect(() => { void refresh(); }, []);

  return (
    <div style={{ padding: 16, color: '#eee', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>People ({people.length})</h2>
        <button
          onClick={() => setShowAdd(true)}
          style={{ padding: '6px 12px', background: '#2e6cdf', color: '#fff', border: 0, borderRadius: 4, cursor: 'pointer' }}
        >
          + Add person
        </button>
      </div>

      {error && <p style={{ color: '#f87171' }}>err: {error}</p>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12, marginTop: 16 }}>
        {people.map((p) => (
          <Link key={p.id} to={`/people/${p.id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
            <div style={{ background: '#141414', border: '1px solid #222', borderRadius: 6, padding: 12 }}>
              {p.latest_face_url ? (
                <img src={absUrl(p.latest_face_url)} alt={p.name}
                  style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4 }} />
              ) : (
                <div style={{ width: '100%', aspectRatio: '1', background: '#222', borderRadius: 4 }} />
              )}
              <div style={{ marginTop: 8, fontWeight: 600 }}>{p.name}</div>
              <div style={{ fontSize: 11, color: '#888' }}>{p.face_count} face(s)</div>
            </div>
          </Link>
        ))}
      </div>

      {showAdd && (
        <AddPersonModal onClose={() => setShowAdd(false)} onCreated={() => { void refresh(); setShowAdd(false); }} />
      )}
    </div>
  );
}

function AddPersonModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit() {
    if (!name.trim() || files.length === 0) {
      setErr('name and at least one image required');
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      await createPerson(name.trim(), files);
      onCreated();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)',
      display: 'grid', placeItems: 'center', zIndex: 1000,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{
        background: '#141414', padding: 24, borderRadius: 8, minWidth: 320,
        display: 'flex', flexDirection: 'column', gap: 12,
      }}>
        <h3 style={{ margin: 0 }}>Add person</h3>
        <input
          type="text" placeholder="Name" value={name} onChange={(e) => setName(e.target.value)}
          style={{ padding: 8, background: '#0c0c0c', color: '#eee', border: '1px solid #333', borderRadius: 4 }}
        />
        <input
          type="file" multiple accept="image/*"
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          style={{ color: '#aaa' }}
        />
        <div style={{ fontSize: 11, color: '#888' }}>{files.length} file(s) selected</div>
        {err && <p style={{ color: '#f87171', fontSize: 12 }}>{err}</p>}
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button onClick={onClose} disabled={busy}
            style={{ padding: '6px 12px', background: 'transparent', color: '#aaa', border: '1px solid #333', borderRadius: 4, cursor: 'pointer' }}>
            Cancel
          </button>
          <button onClick={() => void submit()} disabled={busy}
            style={{ padding: '6px 12px', background: '#2e6cdf', color: '#fff', border: 0, borderRadius: 4, cursor: 'pointer' }}>
            {busy ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3:** Register the route in `frontend/src/main.tsx`:

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import { LivePage } from './pages/LivePage';
import { PeoplePage } from './pages/PeoplePage';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<LivePage />} />
          <Route path="people" element={<PeoplePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
```

- [ ] **Step 4:** `cd frontend && npm run build` — expect success.

- [ ] **Step 5:** Commit.

```bash
git add frontend/src/api/recognition.ts frontend/src/pages/PeoplePage.tsx frontend/src/main.tsx
git commit -m "feat(frontend): People list + add-person flow"
```

---

## Task 15: Frontend — PersonDetailPage

**Files:**
- Create: `frontend/src/pages/PersonDetailPage.tsx`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1:** Create `frontend/src/pages/PersonDetailPage.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  absUrl, deletePerson, getPerson, listPersonClips,
  type PersonClip, type PersonDetail,
} from '../api/recognition';

export function PersonDetailPage() {
  const { id } = useParams<{ id: string }>();
  const nav = useNavigate();
  const pid = id ? Number(id) : NaN;
  const [person, setPerson] = useState<PersonDetail | null>(null);
  const [clips, setClips] = useState<PersonClip[]>([]);
  const [openClip, setOpenClip] = useState<PersonClip | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!Number.isFinite(pid)) return;
    (async () => {
      try {
        const [p, c] = await Promise.all([getPerson(pid), listPersonClips(pid)]);
        setPerson(p);
        setClips(c);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [pid]);

  async function onDelete() {
    if (!person) return;
    if (!confirm(`Delete ${person.name} and all their faces?`)) return;
    try {
      await deletePerson(person.id);
      nav('/people');
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  if (err) return <div style={{ padding: 16, color: '#f87171' }}>err: {err}</div>;
  if (!person) return <div style={{ padding: 16, color: '#aaa' }}>loading...</div>;

  return (
    <div style={{ padding: 16, color: '#eee', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ margin: 0 }}>{person.name}</h2>
        <button onClick={() => void onDelete()}
          style={{ padding: '6px 12px', background: '#c2410c', color: '#fff', border: 0, borderRadius: 4, cursor: 'pointer' }}>
          Delete person
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginTop: 16 }}>
        <section>
          <h3>Gallery ({person.faces.length})</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))', gap: 6 }}>
            {person.faces.map((f) => (
              <img key={f.id} src={absUrl(f.crop_url)} alt={`face ${f.id}`}
                style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4 }}
                title={f.source} />
            ))}
          </div>
        </section>

        <section>
          <h3>Clips ({clips.length})</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {clips.length === 0 && <p style={{ color: '#777', fontSize: 12 }}>No clips yet.</p>}
            {clips.map((c) => (
              <button key={c.clip_path} onClick={() => setOpenClip(c)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: 4,
                  background: '#0c0c0c', border: '1px solid #222', borderRadius: 4,
                  cursor: 'pointer', textAlign: 'left', color: '#eee',
                }}>
                <img src={absUrl(c.jpg_url)} alt={c.clip_path}
                  style={{ width: 80, height: 45, objectFit: 'cover', borderRadius: 2 }} />
                <div style={{ fontSize: 11, lineHeight: 1.3 }}>
                  <div style={{ fontFamily: 'ui-monospace, monospace' }}>{c.clip_path}</div>
                  <div style={{ color: '#888' }}>conf {c.confidence.toFixed(2)} · t={c.frame_ts.toFixed(1)}s</div>
                </div>
              </button>
            ))}
          </div>
        </section>
      </div>

      {openClip && <ClipModal clip={openClip} onClose={() => setOpenClip(null)} />}
    </div>
  );
}

function ClipModal({ clip, onClose }: { clip: PersonClip; onClose: () => void }) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') onClose(); }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div onClick={onClose} style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.85)',
      display: 'grid', placeItems: 'center', zIndex: 1000,
    }}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: '#141414', padding: 16, borderRadius: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>{clip.clip_path}</div>
          <button onClick={onClose}
            style={{ background: 'transparent', color: '#eee', border: '1px solid #333', borderRadius: 4, padding: '4px 10px', cursor: 'pointer' }}>
            close
          </button>
        </div>
        <video controls autoPlay src={absUrl(clip.mp4_url)}
          style={{ maxWidth: '80vw', maxHeight: '70vh', borderRadius: 4 }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2:** Add the route in `main.tsx` — insert between `people` and `review`:

```tsx
          <Route path="people" element={<PeoplePage />} />
          <Route path="people/:id" element={<PersonDetailPage />} />
```

(Don't forget the import at top: `import { PersonDetailPage } from './pages/PersonDetailPage';`)

- [ ] **Step 3:** Build.

```bash
cd frontend && npm run build
```

- [ ] **Step 4:** Commit.

```bash
git add frontend/src/pages/PersonDetailPage.tsx frontend/src/main.tsx
git commit -m "feat(frontend): PersonDetailPage with gallery + clips"
```

---

## Task 16: Frontend — ReviewPage

**Files:**
- Create: `frontend/src/pages/ReviewPage.tsx`
- Modify: `frontend/src/main.tsx`

- [ ] **Step 1:** Create `frontend/src/pages/ReviewPage.tsx`:

```tsx
import { useEffect, useState } from 'react';
import {
  absUrl, dismissReview, labelReview, listPeople, listReview,
  type PersonSummary, type ReviewItem,
} from '../api/recognition';

const POLL_MS = 10000;

export function ReviewPage() {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [people, setPeople] = useState<PersonSummary[]>([]);
  const [err, setErr] = useState<string | null>(null);

  async function refresh() {
    try {
      const [it, ppl] = await Promise.all([listReview(), listPeople()]);
      setItems(it);
      setPeople(ppl);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), POLL_MS);
    return () => clearInterval(t);
  }, []);

  async function onConfirm(item: ReviewItem) {
    if (!item.suggested) return;
    try {
      await labelReview(item.id, { person_id: item.suggested.id });
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  async function onLabelNew(item: ReviewItem) {
    const name = prompt('Name for new person?');
    if (!name?.trim()) return;
    try {
      await labelReview(item.id, { new_name: name.trim() });
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  async function onLabelExisting(item: ReviewItem, person_id: number) {
    try {
      await labelReview(item.id, { person_id });
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  async function onDismiss(item: ReviewItem) {
    try {
      await dismissReview(item.id);
      await refresh();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
  }

  return (
    <div style={{ padding: 16, color: '#eee', height: '100%', overflowY: 'auto' }}>
      <h2 style={{ margin: 0 }}>Review queue ({items.length})</h2>
      {err && <p style={{ color: '#f87171' }}>err: {err}</p>}
      {items.length === 0 && <p style={{ color: '#777' }}>No pending faces. Triggers fill this queue automatically.</p>}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12, marginTop: 16 }}>
        {items.map((it) => (
          <div key={it.id} style={{ background: '#141414', border: '1px solid #222', borderRadius: 6, padding: 12, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <img src={absUrl(it.crop_url)} alt={`q${it.id}`}
              style={{ width: '100%', aspectRatio: '1', objectFit: 'cover', borderRadius: 4 }} />
            {it.suggested ? (
              <div style={{ fontSize: 12 }}>
                Maybe <b>{it.suggested.name}</b> ({it.suggested.score.toFixed(2)})
              </div>
            ) : (
              <div style={{ fontSize: 12, color: '#888' }}>Unknown face</div>
            )}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {it.suggested && (
                <button onClick={() => void onConfirm(it)}
                  style={btn('#2e6cdf')}>Confirm</button>
              )}
              <button onClick={() => void onLabelNew(it)} style={btn('#16a34a')}>New...</button>
              <select
                onChange={(e) => { if (e.target.value) void onLabelExisting(it, Number(e.target.value)); e.target.value=''; }}
                defaultValue=""
                style={{ padding: '4px 6px', background: '#0c0c0c', color: '#eee', border: '1px solid #333', borderRadius: 4, fontSize: 11 }}>
                <option value="">Label existing...</option>
                {people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <button onClick={() => void onDismiss(it)} style={btn('#444')}>Dismiss</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function btn(bg: string): React.CSSProperties {
  return {
    padding: '4px 8px', background: bg, color: '#fff', border: 0,
    borderRadius: 4, cursor: 'pointer', fontSize: 11,
  };
}
```

- [ ] **Step 2:** Register the route in `frontend/src/main.tsx`. The final file:

```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App';
import { LivePage } from './pages/LivePage';
import { PeoplePage } from './pages/PeoplePage';
import { PersonDetailPage } from './pages/PersonDetailPage';
import { ReviewPage } from './pages/ReviewPage';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<LivePage />} />
          <Route path="people" element={<PeoplePage />} />
          <Route path="people/:id" element={<PersonDetailPage />} />
          <Route path="review" element={<ReviewPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
);
```

- [ ] **Step 3:** Build + manual smoke.

```bash
cd frontend && npm run build
```

- [ ] **Step 4:** Commit.

```bash
git add frontend/src/pages/ReviewPage.tsx frontend/src/main.tsx
git commit -m "feat(frontend): ReviewPage for labeling unknown faces"
```

---

## Task 17: Manual end-to-end verification

**Files:** none

- [ ] **Step 1:** Restart backend (it picks up `lifespan` so face engine + worker start):

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```
First request will trigger a ~50MB InsightFace model download.

- [ ] **Step 2:** Start frontend:

```powershell
cd frontend; npm run dev
```

- [ ] **Step 3:** Open `http://localhost:5173`. Confirm:
  - The new top NavBar shows `Live` / `People` / `Review`.
  - Live tab works exactly as before (camera picker, inference, recent clips).

- [ ] **Step 4:** Click `People` → `+ Add person`. Enter your name. Upload 2-3 photos of yourself. Expected:
  - On submit, modal closes.
  - Your card appears with a thumbnail and `2 face(s)` (or however many faces InsightFace detected).
  - Click the card → PersonDetailPage shows the gallery thumbnails.

- [ ] **Step 5:** Go back to `Live`. Wave at the camera. Wait for a clip to record. Within a few seconds of the clip closing, the recognition worker should process it. Backend logs print `processing clip: ...` and either `add_sighting` or queue insertion.

- [ ] **Step 6:** Click `People` → your card. The "Clips" pane should now show the clip(s) you appeared in.

- [ ] **Step 7:** Have someone else (or a printout/phone photo of a different face) appear on camera. Their face should land in the `Review` queue.

- [ ] **Step 8:** Click `Review`. Verify:
  - The unknown face appears as a card.
  - Use `Label existing...` to map them to a known person, or `New...` to create one.
  - After labeling, the card disappears from the queue.

- [ ] **Step 9:** Spot-check the DB:

```powershell
cd backend\data; python -c "import sqlite3; c=sqlite3.connect('app.db'); print(list(c.execute('select * from persons'))); print(list(c.execute('select id, person_id, source from faces'))); print(list(c.execute('select * from clip_sightings')))"
```

- [ ] **Step 10:** Commit a final marker:

```bash
git commit --allow-empty -m "docs: face recognition verified end-to-end"
```

---

## Self-review notes

- Spec section 2 decisions are covered: post-clip worker (Tasks 5/7/8/10), InsightFace `buffalo_l` (Task 6), hybrid registration (Tasks 11/12 upload + 16 queue), crop + embedding storage (Tasks 3 schema + 8 worker + 12 API), Person→clips search (Task 12 endpoint + Task 15 page), SQLite (Tasks 2/3), 1 fps sampling (Task 8 `_sample_frames`), match thresholds (Task 8 `_process` branches), separate routes (Tasks 13–16).
- No placeholders or "TBD".
- Type/name consistency: `RecognitionWorker`, `recognition_worker`, `face_engine`, `gallery`, `FaceResult`, `Gallery.match`, `Gallery.add`, `Gallery.reload` are used uniformly across tasks. `on_clip_closed` callback signature `Callable[[Path], None]` is consistent in Tasks 9, 10, 11.
- The frontend tests are not added since the existing repo has none. Build success + manual verification (Task 17) is the standard.
