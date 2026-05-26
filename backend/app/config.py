"""Application settings — single source of truth for paths, device, defaults."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BACKEND_ROOT / "weights"


@dataclass(frozen=True)
class Settings:
    weights_dir: Path = WEIGHTS_DIR
    default_yolo_detect: str = "yolo26n.pt"
    default_yolo_pose: str = "yolo26n-pose.pt"
    jpeg_quality_hint: int = 70
    frame_max_side: int = 640
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)


settings = Settings()
