"""Track-pinned live face recognition.

A recognized name is pinned to (camera_id, track_id) and reused every frame at
zero cost. Face detect/embed/match runs on a single background worker thread,
off the WebSocket event loop. The per-frame `annotate` call only reads the cache
and enqueues misses — it never calls the face engine inline.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceMatch:
    name: str       # "Alice" | "Unknown"
    score: float
    kind: str       # "high" | "mid" | "unknown"


@dataclass
class _TrackName:
    name: str = "Unknown"
    score: float = 0.0
    kind: str = "unknown"
    resolved_at: float = 0.0       # last successful recognition (0.0 = never)
    last_seen: float = 0.0
    next_attempt_at: float = 0.0   # earliest clock time we may (re)recognize


@dataclass
class _Job:
    camera_id: str
    track_id: int
    crop: np.ndarray


class LiveRecognizer:
    def __init__(
        self,
        engine: FaceEngine,
        gallery: Gallery,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._gallery = gallery
        self._clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, dict[int, _TrackName]] = {}
        self._inflight: set[tuple[str, int]] = set()
        self._q: queue.Queue[_Job] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- worker lifecycle ----
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        t = threading.Thread(target=self._run, name="LiveRecognizer", daemon=True)
        t.start()
        self._thread = t
        log.info("live recognizer worker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("live recognizer worker stopped")

    def clear_camera(self, camera_id: str) -> None:
        with self._lock:
            self._state.pop(camera_id, None)
            self._inflight = {k for k in self._inflight if k[0] != camera_id}

    # ---- per-frame (event loop thread) ----
    def annotate(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        person_boxes: list[dict],
    ) -> dict[int, FaceMatch]:
        now = self._clock()
        h, w = frame_bgr.shape[:2]
        result: dict[int, FaceMatch] = {}
        with self._lock:
            cam = self._state.setdefault(camera_id, {})
            for pb in person_boxes:
                tid = pb.get("track_id")
                if tid is None:
                    continue
                st = cam.get(tid)
                if st is None:
                    st = _TrackName(last_seen=now, next_attempt_at=now)
                    cam[tid] = st
                st.last_seen = now
                if st.resolved_at > 0.0:
                    result[tid] = FaceMatch(st.name, st.score, st.kind)
                if now >= st.next_attempt_at:
                    crop = self._crop(frame_bgr, pb, w, h)
                    if crop is not None and self._enqueue(camera_id, tid, crop):
                        st.next_attempt_at = now + settings.live_recognition_retry
            self._evict(now)
        return result

    def _enqueue(self, camera_id: str, track_id: int, crop: np.ndarray) -> bool:
        key = (camera_id, track_id)
        if key in self._inflight:
            return False
        if self._q.qsize() >= settings.recognition_queue_max:
            return False
        self._inflight.add(key)
        self._q.put(_Job(camera_id, track_id, crop))
        return True

    @staticmethod
    def _crop(frame_bgr: np.ndarray, pb: dict, w: int, h: int) -> np.ndarray | None:
        x1 = max(0, int(round(float(pb["x"]) * w)))
        y1 = max(0, int(round(float(pb["y"]) * h)))
        x2 = min(w, int(round((float(pb["x"]) + float(pb["w"])) * w)))
        y2 = min(h, int(round((float(pb["y"]) + float(pb["h"])) * h)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop.copy()   # worker owns it, independent of the live frame buffer

    def _evict(self, now: float) -> None:
        ttl = settings.track_ttl
        for cam_id, cam in list(self._state.items()):
            for tid, st in list(cam.items()):
                if now - st.last_seen > ttl:
                    del cam[tid]
            if not cam:
                del self._state[cam_id]

    # ---- background worker ----
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_job(job)
            except Exception:
                log.exception("live recognition job failed")
                self._mark_tentative(job)
            finally:
                with self._lock:
                    self._inflight.discard((job.camera_id, job.track_id))
                self._q.task_done()

    def _drain_for_tests(self) -> None:
        """Process all queued jobs on the calling thread (deterministic tests)."""
        while True:
            try:
                job = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                self._process_job(job)
            finally:
                with self._lock:
                    self._inflight.discard((job.camera_id, job.track_id))
                self._q.task_done()

    def _process_job(self, job: _Job) -> None:
        now = self._clock()
        faces = self._engine.detect_and_embed(job.crop)
        if not faces:
            self._write(job, "Unknown", 0.0, "unknown", now)
            return
        face = max(faces, key=lambda f: f.det_score)
        pid, score = self._gallery.match(face.embedding)
        kind = self._classify(score, pid)
        if kind == "unknown" or pid is None:
            name = "Unknown"
        else:
            person = dbmod.get_person(pid)
            name = person["name"] if person else "Unknown"
        self._write(job, name, float(score), kind, now)

    def _write(self, job: _Job, name: str, score: float, kind: str, now: float) -> None:
        with self._lock:
            cam = self._state.setdefault(job.camera_id, {})
            st = cam.get(job.track_id) or _TrackName()
            st.name = name
            st.score = score
            st.kind = kind
            st.resolved_at = now
            st.last_seen = max(st.last_seen, now)
            st.next_attempt_at = now + (
                settings.live_recognition_refresh
                if kind == "high"
                else settings.live_recognition_retry
            )
            cam[job.track_id] = st

    def _mark_tentative(self, job: _Job) -> None:
        now = self._clock()
        with self._lock:
            cam = self._state.setdefault(job.camera_id, {})
            st = cam.get(job.track_id) or _TrackName()
            st.next_attempt_at = now + settings.live_recognition_retry
            cam[job.track_id] = st

    @staticmethod
    def _classify(score: float, pid: int | None) -> str:
        if pid is None:
            return "unknown"
        if score >= settings.match_high:
            return "high"
        if score >= settings.match_low:
            return "mid"
        return "unknown"
