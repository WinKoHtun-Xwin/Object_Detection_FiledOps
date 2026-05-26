"""SAM 3 image segmentation via Ultralytics wrapper.

Supports text prompts (e.g. "person", "laptop"). One backbone pass per frame.
"""
from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import asdict
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.domain import Box, MaskResult

log = logging.getLogger(__name__)


class Sam3ImageEngine:
    name = "sam3_image"

    def __init__(self, weights: str | None = None, device: str = "cuda") -> None:
        from ultralytics.models.sam import SAM3SemanticPredictor

        weights_path = str(settings.weights_dir / (weights or "sam3.pt"))
        log.info("loading SAM 3 image predictor: %s on %s", weights_path, device)
        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model=weights_path,
            half=(device == "cuda"),
            save=False,
            verbose=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.device = device
        # warmup
        dummy = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
        self.predictor.set_image(dummy)
        self.predictor(text=["thing"])

    def infer(self, frame_bgr: np.ndarray, header: dict[str, Any]) -> dict[str, Any]:
        text = header.get("text") or "person"
        # SAM3SemanticPredictor accepts a list of phrases
        prompts = [p.strip() for p in str(text).split(",") if p.strip()]
        if not prompts:
            prompts = ["person"]

        h, w = frame_bgr.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        t0 = time.perf_counter()
        self.predictor.set_image(pil)
        results = self.predictor(text=prompts)
        dt = (time.perf_counter() - t0) * 1000.0
        log.debug("sam3 infer text=%s ms=%.0f", prompts, dt)

        r = results[0] if isinstance(results, list) else results

        masks_out: list[dict[str, Any]] = []
        mdata = getattr(r.masks, "data", None) if r.masks is not None else None
        bdata = r.boxes if r.boxes is not None else None

        if mdata is not None and len(mdata) > 0:
            masks_np = mdata.cpu().numpy().astype(np.uint8)  # (N, H, W) 0/1
            xyxy = bdata.xyxy.cpu().numpy() if bdata is not None else None
            confs = bdata.conf.cpu().numpy() if bdata is not None else None
            cls = bdata.cls.cpu().numpy().astype(int) if bdata is not None and bdata.cls is not None else None

            for i, m in enumerate(masks_np):
                box = None
                if xyxy is not None and i < len(xyxy):
                    x1, y1, x2, y2 = xyxy[i]
                    box = Box(
                        x=float(x1) / w,
                        y=float(y1) / h,
                        w=float(x2 - x1) / w,
                        h=float(y2 - y1) / h,
                    )
                label = ""
                if cls is not None and i < len(cls) and 0 <= cls[i] < len(prompts):
                    label = prompts[cls[i]]
                score = float(confs[i]) if confs is not None and i < len(confs) else 0.0

                masks_out.append(
                    {
                        **asdict(MaskResult(rle=_mask_to_png_b64(m), score=score, label=label, box=box)),
                    }
                )

        return {"type": "sam3", "masks": masks_out}


def _mask_to_png_b64(mask: np.ndarray) -> str:
    """Encode a binary (H, W) mask as an RGBA PNG with the mask in the ALPHA channel.

    The shape must be in alpha (not luminance) so the frontend's composite-based
    outline trick (dilate then destination-out) works correctly.
    """
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    alpha = (mask > 0).astype(np.uint8) * 255
    rgba[..., 0] = 255  # white RGB so any tint shows white by default
    rgba[..., 1] = 255
    rgba[..., 2] = 255
    rgba[..., 3] = alpha
    pil = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
