"""Per-process recorder registry — one MotionRecorder per camera_id."""
from __future__ import annotations

import logging

import numpy as np

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
            rec = MotionRecorder(camera_id=camera_id, frame_size=(w, h))
            self._recorders[camera_id] = rec
            log.info("recorder created: %s (%dx%d)", camera_id, w, h)
        rec.feed(frame_bgr, detections)

    def close_all(self) -> None:
        for rec in self._recorders.values():
            rec.close()
        self._recorders.clear()


recorder_manager = RecorderManager()
