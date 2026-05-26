"""Bounding-box-displacement motion detector.

Compares the latest frame's detection centers against the previous frame's
(by class, greedy nearest-match). Returns True if any allowlisted class moved
more than `threshold` in normalized coordinates.
"""
from __future__ import annotations

from collections.abc import Iterable


class MotionDetector:
    def __init__(self, allowed_classes: Iterable[str], threshold: float) -> None:
        self._allowed = set(allowed_classes)
        self._threshold = float(threshold)
        # last seen centers, grouped by class: {label: [(cx, cy), ...]}
        self._prev: dict[str, list[tuple[float, float]]] = {}

    def update(self, detections: list[dict]) -> bool:
        # Group current frame's allowlisted detections by label.
        current: dict[str, list[tuple[float, float]]] = {}
        for d in detections:
            label = d.get("label", "")
            if label not in self._allowed:
                continue
            cx = float(d["x"]) + float(d["w"]) / 2.0
            cy = float(d["y"]) + float(d["h"]) / 2.0
            current.setdefault(label, []).append((cx, cy))

        moved = False
        for label, centers in current.items():
            prev_centers = self._prev.get(label, [])
            if not prev_centers:
                # No prior data for this class — presence only, not motion.
                continue
            if self._any_drift(centers, prev_centers) > self._threshold:
                moved = True
                break

        # Always update history (even on no-motion frames) so the next frame
        # has a baseline.
        self._prev = current
        return moved

    @staticmethod
    def _any_drift(
        current: list[tuple[float, float]],
        previous: list[tuple[float, float]],
    ) -> float:
        """Return the largest min-distance shift from any current center to its
        nearest previous center. Greedy 1-to-1 is unnecessary for a max-drift
        signal."""
        max_drift = 0.0
        for cx, cy in current:
            best = min(
                ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                for px, py in previous
            )
            if best > max_drift:
                max_drift = best
        return max_drift
