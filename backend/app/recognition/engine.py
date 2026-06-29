"""InsightFace wrapper. Lazy-loads buffalo_l on first call.

Returns 512-d L2-normalized embeddings per detected face."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

_cuda_dll_dir_added = False


def _ensure_cuda_dll_path() -> None:
    """Put torch's bundled CUDA/cuDNN DLLs on the Windows DLL search path.

    onnxruntime-gpu needs the CUDA 12 runtime + cuDNN 9 DLLs (cudnn64_9.dll,
    cublas64_12.dll, cudart64_12.dll, ...) to bind CUDAExecutionProvider;
    without them it silently falls back to CPU. torch (installed for YOLO) ships
    exactly those DLLs in its ``lib`` dir, so we make that dir discoverable
    before InsightFace creates its sessions. No-op off Windows / without torch.
    """
    global _cuda_dll_dir_added
    if _cuda_dll_dir_added or not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(lib):
            os.add_dll_directory(lib)
            _cuda_dll_dir_added = True
            log.info("added torch CUDA lib dir to DLL search path: %s", lib)
    except Exception:
        log.warning("could not add torch CUDA lib dir to DLL path", exc_info=True)


@dataclass(frozen=True)
class FaceResult:
    crop_bgr: np.ndarray              # 112×112×3 aligned crop
    embedding: np.ndarray             # 512-d float32, L2-normalized
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in source pixels
    det_score: float


class FaceEngine:
    def __init__(self, device: str | None = None, det_size: int = 640) -> None:
        self._device = device or settings.recognition_device
        self._det_size = int(det_size)
        self._app: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        if self._device == "cuda":
            _ensure_cuda_dll_path()
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        log.info(
            "loading InsightFace buffalo_l (providers=%s, det_size=%d)",
            providers,
            self._det_size,
        )
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(
            ctx_id=0 if self._device == "cuda" else -1,
            det_size=(self._det_size, self._det_size),
        )
        self._app = app

    def warmup(self) -> None:
        """Load the models and run one dummy pass through each so the first real
        call is fast.

        The dummy frame has no face, so ``.get()`` only exercises the detector;
        the landmark / genderage / recognition sub-models would otherwise pay
        their one-time cuDNN autotuning on the first *real* face (a multi-second
        stall). Run each sub-model's session on a correctly-shaped zero input so
        that cost is paid here, at startup, instead of mid-stream.
        """
        self._ensure_loaded()
        dummy = np.zeros((self._det_size, self._det_size, 3), dtype=np.uint8)
        self._app.get(dummy)  # warms the detector
        for model in self._app.models.values():
            if getattr(model, "taskname", None) == "detection":
                continue  # already warmed via .get() above
            session = getattr(model, "session", None)
            if session is None:
                continue
            try:
                inp = session.get_inputs()[0]
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                session.run(None, {inp.name: np.zeros(shape, dtype=np.float32)})
            except Exception:
                log.debug(
                    "sub-model warmup skipped for %s",
                    getattr(model, "taskname", "?"),
                    exc_info=True,
                )

    def detect_and_embed(self, frame_bgr: np.ndarray) -> list[FaceResult]:
        self._ensure_loaded()
        faces = self._app.get(frame_bgr)
        results: list[FaceResult] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1], x2)
            y2 = min(frame_bgr.shape[0], y2)
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
