"""YOLO26 object-detection engine across all sizes.

One class wraps ultralytics.YOLO and serializes detections into the JSON shape
the frontend expects. The instance is keyed in the registry by size so each
size lazy-loads once and stays warm.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from typing import Any, Literal

import numpy as np

from app.config import settings
from app.domain import Box

log = logging.getLogger(__name__)

Size = Literal["n", "s", "m", "l", "x"]


def _weights_filename(size: Size) -> str:
    return f"yolo26{size}.pt"


class YoloEngine:
    name = "yolo"

    def __init__(self, size: Size = "n", device: str = "cuda") -> None:
        from ultralytics import YOLO

        self.size = size
        self.device = device
        self.weights_name = _weights_filename(size)
        weights_path = settings.weights_dir / self.weights_name

        log.info("loading YOLO detect size=%s -> %s on %s", size, self.weights_name, device)
        # Ultralytics auto-downloads if file missing — chdir to weights_dir so it lands there.
        cwd = os.getcwd()
        try:
            os.chdir(settings.weights_dir)
            self.model = YOLO(
                self.weights_name if not weights_path.exists() else str(weights_path)
            )
        finally:
            os.chdir(cwd)

        # warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=device, verbose=False)

    def infer(self, frame_bgr: np.ndarray, header: dict[str, Any]) -> dict[str, Any]:
        conf = float(header.get("conf", 0.25))
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        result = self.model.predict(
            frame_bgr, device=self.device, conf=conf, verbose=False
        )[0]
        dt = (time.perf_counter() - t0) * 1000.0
        log.debug("yolo detect/%s %.0fms", self.size, dt)

        track_ids = None
        camera_id = header.get("camera_id")
        # Live face recognition pins names to track IDs, so recognition
        # implicitly requires tracking — stamp track IDs whenever tracking is
        # requested OR recognition is on.
        if camera_id and (header.get("tracking") or header.get("recognize")):
            from app.inference.tracking import camera_trackers

            det = result.boxes.cpu().numpy()
            track_ids = camera_trackers.assign(camera_id, det, frame_bgr)
        return {"type": "detect", "boxes": _extract_boxes(result, w, h, track_ids)}


def _extract_boxes(
    result: Any,
    w: int,
    h: int,
    track_ids: list[int | None] | None = None,
) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    out: list[dict[str, Any]] = []
    for i, ((x1, y1, x2, y2), c, p) in enumerate(zip(xyxy, cls, conf, strict=True)):
        tid = track_ids[i] if track_ids is not None and i < len(track_ids) else None
        out.append(
            asdict(
                Box(
                    x=float(x1) / w,
                    y=float(y1) / h,
                    w=float(x2 - x1) / w,
                    h=float(y2 - y1) / h,
                    label=names.get(int(c), str(c)),
                    conf=float(p),
                    track_id=tid,
                )
            )
        )
    return out
