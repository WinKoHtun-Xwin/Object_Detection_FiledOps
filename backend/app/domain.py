"""Shared dataclasses used by inference engines and result serialization.

Coordinates are normalized to [0, 1] in image space so the frontend can scale
to its display without knowing the inference resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str = ""
    conf: float = 0.0
    track_id: int | None = None


@dataclass
class Keypoint:
    x: float
    y: float
    vis: float = 1.0  # visibility / confidence


@dataclass
class Person:
    box: Box
    keypoints: list[Keypoint] = field(default_factory=list)


@dataclass
class MaskResult:
    rle: str           # COCO RLE-encoded mask
    score: float
    label: str = ""
    box: Box | None = None
