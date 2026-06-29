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
    jpeg_quality_hint: int = 70
    frame_max_side: int = 640
    cors_origins: tuple[str, ...] = ("http://localhost:5173","http://localhost:8000", "http://localhost:8001", "http://localhost:8002")

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
    # live recognition (track-pinned)
    live_recognition_refresh: float = 10.0   # re-confirm a CONFIRMED (high) track this often
    live_recognition_retry: float = 1.5      # backoff before retrying a TENTATIVE track
    track_ttl: float = 5.0                   # evict a track's cached name after this unseen gap (s)
    recognition_queue_max: int = 16          # max queued live face jobs before dropping
    live_det_size: int = 320                 # InsightFace det_size for live crops (offline stays 640)
    # face quality gate (review queue + uploads) — drop blurry / tiny / low-confidence faces
    face_min_px: int = 60                    # min detected-face bbox side, source pixels
    face_min_det_score: float = 0.65         # InsightFace detector confidence
    face_min_blur_var: float = 50.0          # variance of Laplacian on the 112px crop; lower = blurrier


settings = Settings()
