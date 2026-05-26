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
