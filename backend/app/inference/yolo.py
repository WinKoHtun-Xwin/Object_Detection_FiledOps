"""Unified YOLO26 engine — handles detect, seg, pose, obb, cls tasks across all sizes.

One class wraps ultralytics.YOLO and serializes results into the appropriate
JSON shape for the frontend. The instance is keyed in the registry by
(task, size) so each (task, size) combo lazy-loads once and stays warm.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from dataclasses import asdict
from typing import Any, Literal

import numpy as np
from PIL import Image

from app.config import settings
from app.domain import Box, Keypoint, Person

log = logging.getLogger(__name__)

Task = Literal["detect", "seg", "pose", "obb", "cls"]
Size = Literal["n", "s", "m", "l", "x"]

_TASK_SUFFIX: dict[Task, str] = {
    "detect": "",
    "seg": "-seg",
    "pose": "-pose",
    "obb": "-obb",
    "cls": "-cls",
}


def _weights_filename(task: Task, size: Size) -> str:
    return f"yolo26{size}{_TASK_SUFFIX[task]}.pt"


class YoloEngine:
    name = "yolo"

    def __init__(self, task: Task, size: Size = "n", device: str = "cuda") -> None:
        from ultralytics import YOLO

        self.task = task
        self.size = size
        self.device = device
        self.weights_name = _weights_filename(task, size)
        weights_path = settings.weights_dir / self.weights_name

        log.info("loading YOLO %s size=%s -> %s on %s", task, size, self.weights_name, device)
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
        log.debug("yolo %s/%s %.0fms", self.task, self.size, dt)

        if self.task == "detect":
            return {"type": "detect", "boxes": _extract_boxes(result, w, h)}
        if self.task == "seg":
            return {"type": "sam3", "masks": _extract_seg_masks(result, w, h)}
        if self.task == "pose":
            return {"type": "pose", "people": _extract_pose(result, w, h)}
        if self.task == "obb":
            return {"type": "obb", "obboxes": _extract_obb(result, w, h)}
        if self.task == "cls":
            return {"type": "cls", "topk": _extract_cls(result)}
        raise AssertionError(f"unknown task {self.task}")


def _extract_boxes(result: Any, w: int, h: int) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    out: list[dict[str, Any]] = []
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf, strict=True):
        out.append(
            asdict(
                Box(
                    x=float(x1) / w,
                    y=float(y1) / h,
                    w=float(x2 - x1) / w,
                    h=float(y2 - y1) / h,
                    label=names.get(int(c), str(c)),
                    conf=float(p),
                )
            )
        )
    return out


def _extract_pose(result: Any, w: int, h: int) -> list[dict[str, Any]]:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf_arr = result.boxes.conf.cpu().numpy()
    kpts_xy = result.keypoints.xy.cpu().numpy()
    kpts_conf = result.keypoints.conf.cpu().numpy()
    people: list[dict[str, Any]] = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        box = Box(
            x=float(x1) / w, y=float(y1) / h,
            w=float(x2 - x1) / w, h=float(y2 - y1) / h,
            label="person", conf=float(conf_arr[i]),
        )
        kps = [
            Keypoint(x=float(px) / w, y=float(py) / h, vis=float(kpts_conf[i, j]))
            for j, (px, py) in enumerate(kpts_xy[i])
        ]
        people.append(
            {"box": asdict(box), "keypoints": [asdict(k) for k in kps]}
        )
    _ = Person  # appease unused-import lint paranoia
    return people


def _extract_seg_masks(result: Any, w: int, h: int) -> list[dict[str, Any]]:
    """Return per-instance segmentation masks in the same shape as SAM 3 results."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    masks_np = result.masks.data.cpu().numpy().astype(np.uint8)  # (N, H, W)
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    out: list[dict[str, Any]] = []
    for i, m in enumerate(masks_np):
        x1, y1, x2, y2 = xyxy[i]
        box = Box(
            x=float(x1) / w, y=float(y1) / h,
            w=float(x2 - x1) / w, h=float(y2 - y1) / h,
        )
        out.append(
            {
                "rle": _mask_to_png_b64(m),
                "score": float(conf[i]),
                "label": names.get(int(cls[i]), str(cls[i])),
                "box": asdict(box),
            }
        )
    return out


def _extract_obb(result: Any, w: int, h: int) -> list[dict[str, Any]]:
    obb = getattr(result, "obb", None)
    if obb is None or len(obb) == 0:
        return []
    names = result.names
    # xyxyxyxy: (N, 4, 2) corners in image-pixel coords
    corners = obb.xyxyxyxy.cpu().numpy()
    cls = obb.cls.cpu().numpy().astype(int)
    conf = obb.conf.cpu().numpy()
    out: list[dict[str, Any]] = []
    for i, poly in enumerate(corners):
        out.append(
            {
                "points": [[float(p[0]) / w, float(p[1]) / h] for p in poly],
                "label": names.get(int(cls[i]), str(cls[i])),
                "conf": float(conf[i]),
            }
        )
    return out


def _extract_cls(result: Any) -> list[dict[str, Any]]:
    probs = getattr(result, "probs", None)
    if probs is None:
        return []
    names = result.names
    top5 = probs.top5
    top5conf = probs.top5conf.cpu().numpy()
    return [
        {"label": names.get(int(idx), str(idx)), "conf": float(c)}
        for idx, c in zip(top5, top5conf, strict=True)
    ]


def _mask_to_png_b64(mask: np.ndarray) -> str:
    """Encode binary (H, W) mask as RGBA PNG with shape in the alpha channel."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    alpha = (mask > 0).astype(np.uint8) * 255
    rgba[..., 0:3] = 255
    rgba[..., 3] = alpha
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
