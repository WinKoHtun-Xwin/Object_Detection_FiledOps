"""Inference engine base class — pure, no I/O."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class InferenceEngine(Protocol):
    name: str

    def infer(self, frame_bgr: np.ndarray, header: dict[str, Any]) -> dict[str, Any]:
        """Run model on a BGR frame. Returns JSON-serializable result dict."""
        ...
