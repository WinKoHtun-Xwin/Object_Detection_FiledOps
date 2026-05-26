"""Per-camera motion-triggered MP4 + snapshot writer.

Holds a 2s ring buffer of recent frames. On the first motion frame:
- flush ring buffer → MP4 (pre-roll)
- write snapshot JPG (the *trigger* frame, not the pre-roll start)
- continue writing while motion persists
- rotate to a new MP4 every 15s without closing the event
- when motion has been absent for > post_roll_seconds, close the file
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.recording.motion import MotionDetector

log = logging.getLogger(__name__)


class _State(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class MotionRecorder:
    def __init__(self, camera_id: str, frame_size: tuple[int, int]) -> None:
        self._camera_id = camera_id
        self._frame_w, self._frame_h = frame_size

        self._detector = MotionDetector(
            allowed_classes=settings.motion_classes,
            threshold=settings.motion_threshold,
        )
        self._fps = settings.clip_fps
        ring_len = max(1, int(round(settings.pre_roll_seconds * self._fps)))
        self._ring: deque[tuple[float, np.ndarray]] = deque(maxlen=ring_len)

        self._state = _State.IDLE
        self._writer: cv2.VideoWriter | None = None
        self._clip_started_at: float = 0.0     # wall time of clip's first frame
        self._last_motion_ts: float = 0.0
        self._event_open = False               # True between trigger and final close

    # ---- public ----

    def feed(self, frame_bgr: np.ndarray, detections: list[dict], ts: float | None = None) -> None:
        if ts is None:
            ts = time.time()

        # Detect the frame's resolution drift (rare but possible).
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self._frame_w, self._frame_h):
            log.info("frame size changed %sx%s -> %sx%s; closing current clip",
                     self._frame_w, self._frame_h, w, h)
            self._close_writer()
            self._event_open = False
            self._state = _State.IDLE
            self._frame_w, self._frame_h = w, h

        self._ring.append((ts, frame_bgr.copy()))
        moving = self._detector.update(detections)

        if self._state is _State.IDLE:
            if moving:
                self._begin_event(trigger_frame=frame_bgr, trigger_ts=ts)
                self._state = _State.RECORDING
        else:  # RECORDING
            self._writer_write(frame_bgr)
            if moving:
                self._last_motion_ts = ts
            else:
                if ts - self._last_motion_ts > settings.post_roll_seconds:
                    self._close_writer()
                    self._event_open = False
                    self._state = _State.IDLE
                    return
            # rotate every clip_seconds
            if ts - self._clip_started_at >= settings.clip_seconds:
                self._rotate_clip(ts)

    def close(self) -> None:
        self._close_writer()
        self._event_open = False
        self._state = _State.IDLE

    # ---- internals ----

    def _begin_event(self, trigger_frame: np.ndarray, trigger_ts: float) -> None:
        # Pre-roll: the ring buffer's earliest timestamp becomes the file's start.
        first_ts = self._ring[0][0] if self._ring else trigger_ts
        path = self._make_clip_path(first_ts)
        self._open_writer(path, first_ts)
        # Flush ring (pre-roll). The current frame is the last entry of the ring
        # because we appended it above — don't double-write it.
        for _ts, f in self._ring:
            self._writer_write(f)
        # Snapshot is the *trigger* frame, paired with this file's basename.
        snap_path = path.with_suffix(".jpg")
        cv2.imwrite(str(snap_path), trigger_frame)
        self._event_open = True
        self._last_motion_ts = trigger_ts

    def _rotate_clip(self, ts: float) -> None:
        self._close_writer()
        path = self._make_clip_path(ts)
        self._open_writer(path, ts)
        # No new snapshot — snapshot is per event, not per rotation.

    def _open_writer(self, path: Path, started_at: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Try H.264 first (browser-playable). Fall back to mp4v if codec missing.
        self._writer = None
        for tag in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*tag)
            writer = cv2.VideoWriter(
                str(path), fourcc, self._fps, (self._frame_w, self._frame_h)
            )
            if writer.isOpened():
                self._writer = writer
                log.info("clip opened (%s): %s", tag, path)
                break
            writer.release()
        if self._writer is None:
            log.error("VideoWriter failed to open: %s", path)
            self._state = _State.IDLE
            return
        self._clip_started_at = started_at

    def _writer_write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            log.info("clip closed")
            self._writer = None

    def _make_clip_path(self, ts: float) -> Path:
        dt = datetime.fromtimestamp(ts)
        day = dt.strftime("%Y-%m-%d")
        name = dt.strftime("%H%M%S")
        return settings.clips_dir / self._camera_id / day / f"{name}.mp4"
