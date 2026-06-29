"""Shared dataclasses used by inference engines and result serialization.

Coordinates are normalized to [0, 1] in image space so the frontend can scale
to its display without knowing the inference resolution.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str = ""
    conf: float = 0.0
    track_id: int | None = None
