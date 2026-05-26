"""Application settings — single source of truth for paths, device, defaults."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BACKEND_ROOT / "weights"
CLIPS_DIR = BACKEND_ROOT / "data" / "clips"
FACES_DIR = BACKEND_ROOT / "data" / "faces"
DB_PATH = BACKEND_ROOT / "data" / "app.db"


@dataclass(frozen=True)
class Settings:
    weights_dir: Path = WEIGHTS_DIR
    default_yolo_detect: str = "yolo26n.pt"
    default_yolo_pose: str = "yolo26n-pose.pt"
    jpeg_quality_hint: int = 70
    frame_max_side: int = 640
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://localhost:8001", "http://localhost:8002")

    # --- recording ---
    clips_dir: Path = CLIPS_DIR
    clip_seconds: int = 15
    pre_roll_seconds: float = 2.0
    post_roll_seconds: float = 2.0
    motion_threshold: float = 0.015
    motion_classes: tuple[str, ...] = ("person",)
    clip_fps: int = 15

    # --- recognition ---
    faces_dir: Path = FACES_DIR
    db_path: Path = DB_PATH
    match_high: float = 0.55
    match_low: float = 0.40
    recognition_sample_fps: float = 1.0
    recognition_device: str = "cuda"


settings = Settings()
