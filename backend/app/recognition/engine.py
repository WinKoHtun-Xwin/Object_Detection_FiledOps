"""InsightFace wrapper. Lazy-loads buffalo_l on first call.

Returns 512-d L2-normalized embeddings per detected face."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceResult:
    crop_bgr: np.ndarray              # 112×112×3 aligned crop
    embedding: np.ndarray             # 512-d float32, L2-normalized
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in source pixels
    det_score: float


class FaceEngine:
    def __init__(self, device: str | None = None) -> None:
        self._device = device or settings.recognition_device
        self._app: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        log.info("loading InsightFace buffalo_l (providers=%s)", providers)
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if self._device == "cuda" else -1, det_size=(640, 640))
        self._app = app

    def detect_and_embed(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        self._ensure_loaded()
        faces = self._app.get(frame_bgr)
        results: list[FaceResult] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2); y2 = min(frame_bgr.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            import cv2
            crop_112 = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
            emb = np.asarray(f.normed_embedding, dtype=np.float32)
            results.append(
                FaceResult(
                    crop_bgr=crop_112,
                    embedding=emb,
                    bbox=(x1, y1, x2, y2),
                    det_score=float(f.det_score),
                )
            )
        return results
