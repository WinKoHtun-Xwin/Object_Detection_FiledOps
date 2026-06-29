"""Background worker that processes closed clips for face recognition."""
from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery
from app.recognition.quality import is_acceptable_face

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
                    continue
                # Below high-confidence → candidate for the review queue. Skip
                # blurry / tiny / low-confidence faces so they never get queued.
                if not is_acceptable_face(face):
                    continue
                if settings.match_low <= score < settings.match_high and pid is not None:
                    self._save_review(face, rel_clip, suggested_id=pid, suggested_score=score)
                else:
                    self._save_review(face, rel_clip, suggested_id=None, suggested_score=None)

        for pid, (score, ts) in best_per_person.items():
            dbmod.add_sighting(rel_clip, pid, score, ts)
        log.info("clip %s: %d sightings", clip_path.name, len(best_per_person))

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
