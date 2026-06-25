"""Per-camera ByteTrack tracking, driven off the shared YOLO detections.

Detection stays a single warm `.predict()` (one model in VRAM); this layer
assigns a stable `track_id` to each detection, isolated per camera.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class CameraTrackers:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trackers: dict[str, Any] = {}
        self._args: Any | None = None

    def _ensure_args(self) -> Any:
        if self._args is None:
            from ultralytics.utils import YAML, IterableSimpleNamespace
            from ultralytics.utils.checks import check_yaml

            self._args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        return self._args

    def _tracker_for(self, camera_id: str) -> Any:
        tracker = self._trackers.get(camera_id)
        if tracker is not None:
            return tracker
        # BYTETracker.__init__ calls reset_id(), zeroing the class-global STrack
        # counter. Save/restore it so IDs stay monotonic (and disjoint) across cameras.
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = self._ensure_args()
        prev = BaseTrack._count
        tracker = BYTETracker(args=args)
        BaseTrack._count = prev
        self._trackers[camera_id] = tracker
        log.info("created ByteTracker for camera %s", camera_id)
        return tracker

    def assign(self, camera_id: str, det: Any, frame: np.ndarray) -> list[int | None]:
        n = len(det)
        ids: list[int | None] = [None] * n
        if n == 0:
            return ids
        with self._lock:
            args = self._ensure_args()
            # BYTETracker internally drops detections below track_high_thresh and
            # numbers the survivors' idx over that filtered subset. Feed it only the
            # high-confidence detections so each returned idx maps cleanly back to our
            # original positions via `keep`.
            mask = np.asarray(det.conf) >= args.track_high_thresh
            keep = np.nonzero(mask)[0]
            if len(keep) == 0:
                return ids
            tracker = self._tracker_for(camera_id)
            tracks = tracker.update(det[mask], frame)
        # Each row is [x1, y1, x2, y2, track_id, score, cls, idx]; idx is the index
        # within the filtered subset we passed in.
        for row in tracks:
            j = int(row[-1])
            if 0 <= j < len(keep):
                ids[int(keep[j])] = int(row[4])
        return ids

    def drop(self, camera_id: str) -> None:
        with self._lock:
            self._trackers.pop(camera_id, None)


camera_trackers = CameraTrackers()
