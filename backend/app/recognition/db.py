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
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
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
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None
    global _initialized
    _initialized = False


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


def add_sighting(
    clip_path: str,
    person_id: int,
    confidence: float,
    frame_ts: float,
) -> None:
    conn = get_conn()
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
