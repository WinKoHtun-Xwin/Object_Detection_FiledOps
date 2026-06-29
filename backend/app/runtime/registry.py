"""Lazy model registry — load on first use, keep warm. Keyed by (mode, size)."""
from __future__ import annotations

import logging
from typing import Any

from app.runtime.device import probe

log = logging.getLogger(__name__)

MODE_TO_TASK = {
    "yolo_detect": "detect",
}


class ModelRegistry:
    def __init__(self) -> None:
        self._engines: dict[tuple[str, str], Any] = {}
        self.device = probe()
        self.torch_device = "cuda" if self.device.kind == "cuda" else "cpu"

    def get(self, mode: str, size: str = "n") -> Any:
        key = (mode, size)
        if key in self._engines:
            return self._engines[key]

        log.info("lazy-loading engine: %s (size=%s)", mode, size)
        if mode in MODE_TO_TASK:
            from app.inference.yolo import YoloEngine
            engine = YoloEngine(size=size, device=self.torch_device)
        else:
            raise ValueError(f"unknown mode: {mode}")

        self._engines[key] = engine
        return engine


registry = ModelRegistry()
