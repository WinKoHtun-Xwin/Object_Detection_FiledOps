"""Throttled per-camera live face recognition.

Cropping happens here, not in the recognition worker — the live path needs to
operate on a single shared frame across many person detections, while the
worker reads back a saved video.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceMatch:
    bbox: tuple[float, float, float, float]   # normalized (x, y, w, h) of face crop
    name: str                                  # "Alice" | "Unknown"
    score: float
    kind: str                                  # "high" | "mid" | "unknown"


@dataclass
class _CamState:
    last_run_ts: float = 0.0
    cached: list[FaceMatch] = field(default_factory=list)
    cached_at: float = 0.0


class LiveRecognizer:
    def __init__(self, engine: FaceEngine, gallery: Gallery) -> None:
        self._engine = engine
        self._gallery = gallery
        self._state: dict[str, _CamState] = {}

    def annotate(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        person_boxes: list[dict],
    ) -> list[FaceMatch]:
        cam = self._state.setdefault(camera_id, _CamState())
        now = time.time()

        if now - cam.last_run_ts < settings.live_recognition_interval:
            return self._filter_stale(cam, now)

        cam.last_run_ts = now
        cam.cached = list(self._run(frame_bgr, person_boxes))
        cam.cached_at = now
        return cam.cached

    def _run(self, frame_bgr: np.ndarray, person_boxes: list[dict]):
        h, w = frame_bgr.shape[:2]
        for pb in person_boxes:
            x1 = int(round(float(pb["x"]) * w))
            y1 = int(round(float(pb["y"]) * h))
            x2 = int(round((float(pb["x"]) + float(pb["w"])) * w))
            y2 = int(round((float(pb["y"]) + float(pb["h"])) * h))
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            faces = self._engine.detect_and_embed(crop)
            for face in faces:
                pid, score = self._gallery.match(face.embedding)
                kind = self._classify(score, pid)
                if kind == "unknown":
                    name = "Unknown"
                else:
                    from app.recognition import db as dbmod
                    person = dbmod.get_person(pid) if pid is not None else None
                    name = person["name"] if person else "Unknown"

                fx1, fy1, fx2, fy2 = face.bbox
                ax1 = x1 + fx1
                ay1 = y1 + fy1
                ax2 = x1 + fx2
                ay2 = y1 + fy2
                yield FaceMatch(
                    bbox=(ax1 / w, ay1 / h, (ax2 - ax1) / w, (ay2 - ay1) / h),
                    name=name,
                    score=float(score),
                    kind=kind,
                )

    @staticmethod
    def _classify(score: float, pid: int | None) -> str:
        if pid is None:
            return "unknown"
        if score >= settings.match_high:
            return "high"
        if score >= settings.match_low:
            return "mid"
        return "unknown"

    def _filter_stale(self, cam: _CamState, now: float) -> list[FaceMatch]:
        if now - cam.cached_at > settings.live_recognition_cache_ttl:
            return []
        return cam.cached
