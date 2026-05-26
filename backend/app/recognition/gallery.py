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
