# Motion-Triggered Clip Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-camera motion-triggered MP4 clip recording (15s rotating segments, with snapshots and 2s pre/post-roll) to the existing FastAPI + React inference pipeline, plus a frontend camera-picker that sends a `camera_id` over the WebSocket.

**Architecture:** New `backend/app/recording/` package (motion detector, per-camera recorder, manager singleton) hooks into the WebSocket handler after YOLO inference. Frontend adds a camera picker + `camera_id` field in the packet header. No changes to the binary protocol, inference engines, or runtime registry.

**Tech Stack:** Python 3.12 / FastAPI / OpenCV (`cv2.VideoWriter`, `mp4v` fourcc) / pytest. React + TypeScript + Zustand on the frontend.

**Spec:** [docs/superpowers/specs/2026-05-26-motion-clip-recording-design.md](../specs/2026-05-26-motion-clip-recording-design.md)

**Refinement vs spec:** Spec called the threshold `motion_px_threshold` (integer pixels). Backend inference results are already in **normalized [0,1]** coords, so the implementation uses `motion_threshold: float = 0.015` (≈ 1.5% of image width, ≈ 10 px at 640-wide). Same outcome, fewer conversions.

---

## File structure

**New (backend):**
- `backend/app/recording/__init__.py` — exports `recorder_manager`
- `backend/app/recording/motion.py` — `MotionDetector`
- `backend/app/recording/recorder.py` — `MotionRecorder`
- `backend/app/recording/manager.py` — `RecorderManager` + singleton
- `backend/tests/recording/__init__.py`
- `backend/tests/recording/test_motion.py`
- `backend/tests/recording/test_recorder.py`
- `backend/tests/recording/test_manager.py`

**Modified (backend):**
- `backend/app/config.py` — add 7 settings
- `backend/app/api/stream.py` — pull `camera_id` from header, call `recorder_manager.feed(...)`

**New (frontend):**
- `frontend/src/camera/CameraPicker.tsx`

**Modified (frontend):**
- `frontend/src/state/appState.ts` — add `selectedDeviceId`, `cameraId`, setters
- `frontend/src/camera/useWebcam.ts` — accept `deviceId` option
- `frontend/src/stream/useFrameSender.ts` — include `camera_id` in header
- `frontend/src/App.tsx` — mount `CameraPicker`, pass `deviceId` to `useWebcam`

**Root:**
- `.gitignore` — add `backend/data/`

---

## Task 0: Initialize git (if not already)

**Files:** repo root

- [ ] **Step 1:** Check whether repo is already a git repo.

Run: `git -C "c:/Users/win-work/Github/Object_detection" rev-parse --is-inside-work-tree`
Expected: prints `true`. If it errors with "not a git repository", proceed to Step 2; otherwise skip the rest of this task.

- [ ] **Step 2:** Initialize the repo.

Run: `git -C "c:/Users/win-work/Github/Object_detection" init`
Expected: `Initialized empty Git repository in …`

- [ ] **Step 3:** Stage existing files and create the baseline commit.

```bash
cd "c:/Users/win-work/Github/Object_detection"
git add -A
git commit -m "chore: import existing project as initial commit"
```
Expected: a commit summary listing all current files.

---

## Task 1: Add recording config

**Files:**
- Modify: `backend/app/config.py`

- [ ] **Step 1:** Open [backend/app/config.py](../../../backend/app/config.py) and add the seven new fields to the `Settings` dataclass, plus a `CLIPS_DIR` module constant.

Replace the file's body with:

```python
"""Application settings — single source of truth for paths, device, defaults."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BACKEND_ROOT / "weights"
CLIPS_DIR = BACKEND_ROOT / "data" / "clips"


@dataclass(frozen=True)
class Settings:
    weights_dir: Path = WEIGHTS_DIR
    default_yolo_detect: str = "yolo26n.pt"
    default_yolo_pose: str = "yolo26n-pose.pt"
    jpeg_quality_hint: int = 70
    frame_max_side: int = 640
    cors_origins: tuple[str, ...] = ("http://localhost:5173",)

    # --- recording ---
    clips_dir: Path = CLIPS_DIR
    clip_seconds: int = 15
    pre_roll_seconds: float = 2.0
    post_roll_seconds: float = 2.0
    motion_threshold: float = 0.015          # fraction of image width; ~10 px @ 640w
    motion_classes: tuple[str, ...] = ("person",)
    clip_fps: int = 15


settings = Settings()
```

- [ ] **Step 2:** Commit.

```bash
git add backend/app/config.py
git commit -m "feat(recording): add config for motion-triggered clip recording"
```

---

## Task 2: MotionDetector — failing test

**Files:**
- Create: `backend/tests/recording/__init__.py` (empty)
- Create: `backend/tests/recording/test_motion.py`

- [ ] **Step 1:** Create the empty package init.

`backend/tests/recording/__init__.py` — empty file.

- [ ] **Step 2:** Write `backend/tests/recording/test_motion.py`:

```python
"""Tests for MotionDetector."""
from __future__ import annotations

from app.recording.motion import MotionDetector


def _box(x: float, y: float, label: str = "person") -> dict:
    return {"x": x, "y": y, "w": 0.1, "h": 0.2, "label": label, "conf": 0.9}


def test_empty_detections_is_no_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    assert det.update([]) is False


def test_single_static_frame_is_no_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([_box(0.5, 0.5)])
    assert det.update([_box(0.5, 0.5)]) is False


def test_box_drift_above_threshold_is_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([_box(0.5, 0.5)])
    assert det.update([_box(0.6, 0.5)]) is True   # 0.10 shift on x — well above 0.01


def test_box_drift_below_threshold_is_no_motion() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.02)
    det.update([_box(0.500, 0.500)])
    assert det.update([_box(0.505, 0.500)]) is False  # 0.005 < 0.02


def test_non_allowlisted_class_ignored() -> None:
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([_box(0.5, 0.5, label="car")])
    assert det.update([_box(0.7, 0.5, label="car")]) is False


def test_allowlist_class_appears_after_nothing_is_not_motion() -> None:
    """A class entering the frame for the first time is presence, not motion."""
    det = MotionDetector(allowed_classes={"person"}, threshold=0.01)
    det.update([])
    assert det.update([_box(0.5, 0.5)]) is False
```

- [ ] **Step 3:** Run the test, expect failure.

```bash
cd backend
pytest tests/recording/test_motion.py -v
```
Expected: ImportError / ModuleNotFoundError for `app.recording.motion`.

---

## Task 3: MotionDetector — implementation

**Files:**
- Create: `backend/app/recording/__init__.py` (empty for now)
- Create: `backend/app/recording/motion.py`

- [ ] **Step 1:** Create `backend/app/recording/__init__.py` — empty.

- [ ] **Step 2:** Create `backend/app/recording/motion.py`:

```python
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
```

- [ ] **Step 3:** Run tests, expect pass.

```bash
cd backend
pytest tests/recording/test_motion.py -v
```
Expected: 6 passed.

- [ ] **Step 4:** Commit.

```bash
git add backend/app/recording/__init__.py backend/app/recording/motion.py backend/tests/recording/__init__.py backend/tests/recording/test_motion.py
git commit -m "feat(recording): MotionDetector + tests"
```

---

## Task 4: MotionRecorder — failing test

**Files:**
- Create: `backend/tests/recording/test_recorder.py`

- [ ] **Step 1:** Write the test file:

```python
"""Tests for MotionRecorder."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.recording.recorder import MotionRecorder


def _box(x: float, y: float) -> dict:
    return {"x": x, "y": y, "w": 0.1, "h": 0.2, "label": "person", "conf": 0.9}


def _frame() -> np.ndarray:
    # 240×320 BGR frame
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def rec_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg
    monkeypatch.setattr(cfg.settings, "clips_dir", tmp_path, raising=False)
    return tmp_path


def test_no_motion_produces_no_files(rec_root: Path) -> None:
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240))
    # 30 static frames — same box in same place
    for i in range(30):
        rec.feed(_frame(), [_box(0.5, 0.5)], ts=i / 15.0)
    rec.close()
    files = list(rec_root.rglob("*"))
    assert all(f.is_dir() for f in files), f"expected no clip files, got {files}"


def test_motion_event_creates_mp4_and_jpg(rec_root: Path) -> None:
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240))
    # 5 static frames (warm up the detector with a baseline), then 30 frames of
    # drifting motion, then 60 static frames to drive post-roll + close.
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        rec.feed(_frame(), [_box(0.5, 0.5)], ts=t); t += dt
    for i in range(30):
        rec.feed(_frame(), [_box(0.5 + 0.01 * i, 0.5)], ts=t); t += dt
    for _ in range(60):
        rec.feed(_frame(), [_box(0.8, 0.5)], ts=t); t += dt   # static, no motion
    rec.close()

    mp4s = sorted(rec_root.rglob("*.mp4"))
    jpgs = sorted(rec_root.rglob("*.jpg"))
    assert len(mp4s) >= 1, f"expected ≥1 mp4, got {mp4s}"
    assert len(jpgs) == 1, f"expected exactly one event snapshot, got {jpgs}"
    assert mp4s[0].stat().st_size > 0
    assert jpgs[0].stat().st_size > 0


def test_sustained_motion_rotates_15s_files(rec_root: Path) -> None:
    rec = MotionRecorder(camera_id="cam-test", frame_size=(320, 240))
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        rec.feed(_frame(), [_box(0.5, 0.5)], ts=t); t += dt
    # ~25 seconds of continuous motion → expect 2 mp4 files in the event.
    for i in range(25 * 15):
        rec.feed(_frame(), [_box(0.5 + 0.005 * (i % 2 + 1), 0.5)], ts=t); t += dt
    for _ in range(60):
        rec.feed(_frame(), [_box(0.9, 0.5)], ts=t); t += dt   # let post-roll close
    rec.close()

    mp4s = sorted(rec_root.rglob("*.mp4"))
    jpgs = sorted(rec_root.rglob("*.jpg"))
    assert len(mp4s) >= 2, f"expected ≥2 rotation files, got {mp4s}"
    assert len(jpgs) == 1, "snapshot is per-event, not per-rotation"
```

- [ ] **Step 2:** Run, expect failure (ImportError for `app.recording.recorder`).

```bash
cd backend
pytest tests/recording/test_recorder.py -v
```

---

## Task 5: MotionRecorder — implementation

**Files:**
- Create: `backend/app/recording/recorder.py`

- [ ] **Step 1:** Create `backend/app/recording/recorder.py`:

```python
"""Per-camera motion-triggered MP4 + snapshot writer.

Holds a 2s ring buffer of recent frames. On the first motion frame:
- flush ring buffer → MP4 (pre-roll)
- write snapshot JPG (the *trigger* frame, not the pre-roll start)
- continue writing while motion persists
- rotate to a new MP4 every 15s without closing the event
- when motion has been absent for > post_roll_seconds, close the file
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.recording.motion import MotionDetector

log = logging.getLogger(__name__)


class _State(Enum):
    IDLE = "idle"
    RECORDING = "recording"


class MotionRecorder:
    def __init__(self, camera_id: str, frame_size: tuple[int, int]) -> None:
        self._camera_id = camera_id
        self._frame_w, self._frame_h = frame_size

        self._detector = MotionDetector(
            allowed_classes=settings.motion_classes,
            threshold=settings.motion_threshold,
        )
        self._fps = settings.clip_fps
        ring_len = max(1, int(round(settings.pre_roll_seconds * self._fps)))
        self._ring: deque[tuple[float, np.ndarray]] = deque(maxlen=ring_len)

        self._state = _State.IDLE
        self._writer: cv2.VideoWriter | None = None
        self._clip_started_at: float = 0.0     # wall time of clip's first frame
        self._last_motion_ts: float = 0.0
        self._event_open = False               # True between trigger and final close

    # ---- public ----

    def feed(self, frame_bgr: np.ndarray, detections: list[dict], ts: float | None = None) -> None:
        if ts is None:
            ts = time.time()

        # Detect the frame's resolution drift (rare but possible).
        h, w = frame_bgr.shape[:2]
        if (w, h) != (self._frame_w, self._frame_h):
            log.info("frame size changed %sx%s -> %sx%s; closing current clip",
                     self._frame_w, self._frame_h, w, h)
            self._close_writer()
            self._event_open = False
            self._state = _State.IDLE
            self._frame_w, self._frame_h = w, h

        self._ring.append((ts, frame_bgr.copy()))
        moving = self._detector.update(detections)

        if self._state is _State.IDLE:
            if moving:
                self._begin_event(trigger_frame=frame_bgr, trigger_ts=ts)
                self._state = _State.RECORDING
        else:  # RECORDING
            self._writer_write(frame_bgr)
            if moving:
                self._last_motion_ts = ts
            else:
                if ts - self._last_motion_ts > settings.post_roll_seconds:
                    self._close_writer()
                    self._event_open = False
                    self._state = _State.IDLE
                    return
            # rotate every clip_seconds
            if ts - self._clip_started_at >= settings.clip_seconds:
                self._rotate_clip(ts)

    def close(self) -> None:
        self._close_writer()
        self._event_open = False
        self._state = _State.IDLE

    # ---- internals ----

    def _begin_event(self, trigger_frame: np.ndarray, trigger_ts: float) -> None:
        # Pre-roll: the ring buffer's earliest timestamp becomes the file's start.
        first_ts = self._ring[0][0] if self._ring else trigger_ts
        path = self._make_clip_path(first_ts)
        self._open_writer(path, first_ts)
        # Flush ring (pre-roll). The current frame is the last entry of the ring
        # because we appended it above — don't double-write it.
        for _ts, f in self._ring:
            self._writer_write(f)
        # Snapshot is the *trigger* frame, paired with this file's basename.
        snap_path = path.with_suffix(".jpg")
        cv2.imwrite(str(snap_path), trigger_frame)
        self._event_open = True
        self._last_motion_ts = trigger_ts

    def _rotate_clip(self, ts: float) -> None:
        self._close_writer()
        path = self._make_clip_path(ts)
        self._open_writer(path, ts)
        # No new snapshot — snapshot is per event, not per rotation.

    def _open_writer(self, path: Path, started_at: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            str(path), fourcc, self._fps, (self._frame_w, self._frame_h)
        )
        if not self._writer.isOpened():
            log.error("VideoWriter failed to open: %s", path)
            self._writer = None
            return
        self._clip_started_at = started_at
        log.info("clip opened: %s", path)

    def _writer_write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            self._writer.write(frame)

    def _close_writer(self) -> None:
        if self._writer is not None:
            self._writer.release()
            log.info("clip closed")
            self._writer = None

    def _make_clip_path(self, ts: float) -> Path:
        dt = datetime.fromtimestamp(ts)
        day = dt.strftime("%Y-%m-%d")
        name = dt.strftime("%H%M%S")
        return settings.clips_dir / self._camera_id / day / f"{name}.mp4"
```

- [ ] **Step 2:** Run tests.

```bash
cd backend
pytest tests/recording/test_recorder.py -v
```
Expected: 3 passed.

- [ ] **Step 3:** Commit.

```bash
git add backend/app/recording/recorder.py backend/tests/recording/test_recorder.py
git commit -m "feat(recording): MotionRecorder with 15s rotation + pre/post-roll"
```

---

## Task 6: RecorderManager — failing test

**Files:**
- Create: `backend/tests/recording/test_manager.py`

- [ ] **Step 1:** Write:

```python
"""Tests for RecorderManager."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.recording.manager import RecorderManager


def _box(x: float) -> dict:
    return {"x": x, "y": 0.5, "w": 0.1, "h": 0.2, "label": "person", "conf": 0.9}


def _frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def rec_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg
    monkeypatch.setattr(cfg.settings, "clips_dir", tmp_path, raising=False)
    return tmp_path


def test_per_camera_subfolders(rec_root: Path) -> None:
    mgr = RecorderManager()
    # interleave frames from two cameras, each with its own motion event
    t = 0.0
    dt = 1.0 / 15.0
    for _ in range(5):
        mgr.feed("cam-A", _frame(), [_box(0.5)])
        mgr.feed("cam-B", _frame(), [_box(0.5)])
        t += dt
    for i in range(30):
        mgr.feed("cam-A", _frame(), [_box(0.5 + 0.01 * i)])
        mgr.feed("cam-B", _frame(), [_box(0.5 + 0.01 * i)])
        t += dt
    for _ in range(60):
        mgr.feed("cam-A", _frame(), [_box(0.85)])
        mgr.feed("cam-B", _frame(), [_box(0.85)])
        t += dt
    mgr.close_all()

    assert (rec_root / "cam-A").exists()
    assert (rec_root / "cam-B").exists()
    a_mp4s = list((rec_root / "cam-A").rglob("*.mp4"))
    b_mp4s = list((rec_root / "cam-B").rglob("*.mp4"))
    assert a_mp4s, "cam-A produced no clips"
    assert b_mp4s, "cam-B produced no clips"


def test_unknown_camera_creates_recorder_lazily(rec_root: Path) -> None:
    mgr = RecorderManager()
    assert "cam-new" not in mgr._recorders   # noqa: SLF001
    mgr.feed("cam-new", _frame(), [])
    assert "cam-new" in mgr._recorders   # noqa: SLF001
```

- [ ] **Step 2:** Run, expect ImportError.

```bash
cd backend
pytest tests/recording/test_manager.py -v
```

---

## Task 7: RecorderManager — implementation

**Files:**
- Create: `backend/app/recording/manager.py`
- Modify: `backend/app/recording/__init__.py` — export singleton

- [ ] **Step 1:** Create `backend/app/recording/manager.py`:

```python
"""Per-process recorder registry — one MotionRecorder per camera_id."""
from __future__ import annotations

import logging

import numpy as np

from app.recording.recorder import MotionRecorder

log = logging.getLogger(__name__)


class RecorderManager:
    def __init__(self) -> None:
        self._recorders: dict[str, MotionRecorder] = {}

    def feed(self, camera_id: str, frame_bgr: np.ndarray, detections: list[dict]) -> None:
        if not camera_id:
            return
        rec = self._recorders.get(camera_id)
        if rec is None:
            h, w = frame_bgr.shape[:2]
            rec = MotionRecorder(camera_id=camera_id, frame_size=(w, h))
            self._recorders[camera_id] = rec
            log.info("recorder created: %s (%dx%d)", camera_id, w, h)
        rec.feed(frame_bgr, detections)

    def close_all(self) -> None:
        for rec in self._recorders.values():
            rec.close()
        self._recorders.clear()


recorder_manager = RecorderManager()
```

- [ ] **Step 2:** Set `backend/app/recording/__init__.py` to:

```python
"""Motion-triggered clip recording."""
from app.recording.manager import recorder_manager

__all__ = ["recorder_manager"]
```

- [ ] **Step 3:** Run tests.

```bash
cd backend
pytest tests/recording/ -v
```
Expected: all 11 tests pass.

- [ ] **Step 4:** Commit.

```bash
git add backend/app/recording/manager.py backend/app/recording/__init__.py backend/tests/recording/test_manager.py
git commit -m "feat(recording): RecorderManager singleton"
```

---

## Task 8: Wire recorder into the WebSocket handler

**Files:**
- Modify: `backend/app/api/stream.py`

- [ ] **Step 1:** Replace the contents of `backend/app/api/stream.py` with:

```python
"""WebSocket inference endpoint."""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.protocol.frame import FrameDecodeError, parse
from app.recording import recorder_manager
from app.runtime.codec import jpeg_to_bgr
from app.runtime.registry import registry

log = logging.getLogger(__name__)
router = APIRouter()

MODE_BY_ID = {
    0: "yolo_detect",
    1: "yolo_pose",
    2: "sam3_image",
    3: "sam3_video",
    4: "yolo_seg",
    5: "yolo_obb",
    6: "yolo_cls",
}

SIZE_BY_VARIANT = {0: "n", 1: "s", 2: "m", 3: "l", 4: "x"}


def _detections_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten any inference result into a list of Box-shaped dicts.

    Only modes that surface bounding boxes contribute to motion. For modes
    without boxes (cls, sam3, obb without box equivalents), returns []."""
    if "boxes" in result:
        return result["boxes"]
    if "people" in result:
        return [p["box"] for p in result["people"]]
    return []


@router.websocket("/ws")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    log.info("ws connected: %s", ws.client)
    frame_id = 0
    try:
        while True:
            packet = await ws.receive_bytes()
            t0 = time.perf_counter()
            try:
                fp = parse(packet)
                img = jpeg_to_bgr(fp.jpeg)
            except (FrameDecodeError, ValueError) as e:
                await ws.send_json({"type": "error", "frame_id": frame_id, "message": str(e)})
                frame_id += 1
                continue

            mode = MODE_BY_ID.get(fp.mode_id)
            if mode is None:
                await ws.send_json({"type": "error", "frame_id": frame_id, "message": f"unknown mode {fp.mode_id}"})
                frame_id += 1
                continue

            size = SIZE_BY_VARIANT.get(fp.variant_id, "n") if mode.startswith("yolo_") else "n"
            try:
                engine = registry.get(mode, size=size)
                result = engine.infer(img, fp.header)
            except (ValueError, NotImplementedError) as e:
                await ws.send_json({"type": "error", "frame_id": frame_id, "message": str(e)})
                frame_id += 1
                continue

            ms = (time.perf_counter() - t0) * 1000.0
            result["frame_id"] = frame_id
            result["ms"] = round(ms, 2)
            await ws.send_json(result)

            camera_id = fp.header.get("camera_id") if isinstance(fp.header, dict) else None
            if camera_id:
                recorder_manager.feed(camera_id, img, _detections_from_result(result))

            frame_id += 1
    except WebSocketDisconnect:
        log.info("ws disconnected: %s (after %d frames)", ws.client, frame_id)
```

- [ ] **Step 2:** Run the full backend test suite to confirm nothing else regressed.

```bash
cd backend
pytest -v
```
Expected: all previous tests + the 11 new recording tests pass.

- [ ] **Step 3:** Commit.

```bash
git add backend/app/api/stream.py
git commit -m "feat(recording): wire recorder_manager into /ws after inference"
```

---

## Task 9: .gitignore for runtime data

**Files:**
- Create or modify: `.gitignore` at repo root

- [ ] **Step 1:** Check current contents.

Run: `cat .gitignore 2>/dev/null || echo "(missing)"`

- [ ] **Step 2:** Append the new line (or create the file with it).

If file is missing, create it with:

```gitignore
# runtime-generated
backend/data/
```

If it exists and already has unrelated entries, append the two lines above.

- [ ] **Step 3:** Commit.

```bash
git add .gitignore
git commit -m "chore: gitignore backend/data/"
```

---

## Task 10: Frontend — Zustand state for camera selection

**Files:**
- Modify: `frontend/src/state/appState.ts`

- [ ] **Step 1:** Replace the file with:

```ts
import { create } from 'zustand';
import type { InferenceResult, ModeId, Sam3PromptKind, YoloSize } from '../types';

interface AppState {
  mode: ModeId;
  yoloSize: YoloSize;
  yoloConf: number;
  sam3Prompt: Sam3PromptKind;
  sam3Text: string;
  tracking: boolean;
  mirror: boolean;
  paused: boolean;

  selectedDeviceId: string | null;
  cameraId: string;

  fps: number;
  inferenceMs: number;
  lastResult: InferenceResult | null;

  setMode: (m: ModeId) => void;
  setYoloSize: (s: YoloSize) => void;
  setYoloConf: (c: number) => void;
  setSam3Prompt: (p: Sam3PromptKind) => void;
  setSam3Text: (t: string) => void;
  setTracking: (b: boolean) => void;
  setMirror: (b: boolean) => void;
  setPaused: (b: boolean) => void;

  setSelectedDeviceId: (id: string | null) => void;
  setCameraId: (id: string) => void;

  pushResult: (r: InferenceResult) => void;
  setFps: (n: number) => void;
}

export const useAppState = create<AppState>((set) => ({
  mode: 'yolo_detect',
  yoloSize: 'n',
  yoloConf: 0.25,
  sam3Prompt: 'text',
  sam3Text: 'person',
  tracking: false,
  mirror: true,
  paused: false,

  selectedDeviceId: null,
  cameraId: 'cam-default',

  fps: 0,
  inferenceMs: 0,
  lastResult: null,

  setMode: (mode) => set({ mode }),
  setYoloSize: (yoloSize) => set({ yoloSize }),
  setYoloConf: (yoloConf) => set({ yoloConf }),
  setSam3Prompt: (sam3Prompt) => set({ sam3Prompt }),
  setSam3Text: (sam3Text) => set({ sam3Text }),
  setTracking: (tracking) => set({ tracking }),
  setMirror: (mirror) => set({ mirror }),
  setPaused: (paused) => set({ paused }),

  setSelectedDeviceId: (selectedDeviceId) => set({ selectedDeviceId }),
  setCameraId: (cameraId) => set({ cameraId }),

  pushResult: (r) => set({ lastResult: r, inferenceMs: r.ms }),
  setFps: (fps) => set({ fps }),
}));
```

- [ ] **Step 2:** Commit.

```bash
git add frontend/src/state/appState.ts
git commit -m "feat(frontend): add camera selection state"
```

---

## Task 11: Frontend — useWebcam accepts deviceId

**Files:**
- Modify: `frontend/src/camera/useWebcam.ts`

- [ ] **Step 1:** Replace the file with:

```ts
import { useEffect, useRef, useState } from 'react';

interface UseWebcamOptions {
  width?: number;
  height?: number;
  facingMode?: 'user' | 'environment';
  deviceId?: string | null;
}

interface UseWebcamResult {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  ready: boolean;
  error: string | null;
}

export function useWebcam(opts: UseWebcamOptions = {}): UseWebcamResult {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let cancelled = false;
    setReady(false);

    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: opts.width ?? 1280 },
            height: { ideal: opts.height ?? 720 },
            facingMode: opts.facingMode ?? 'user',
            ...(opts.deviceId ? { deviceId: { exact: opts.deviceId } } : {}),
          },
          audio: false,
        });
        if (cancelled || !videoRef.current) {
          stream?.getTracks().forEach((t) => t.stop());
          return;
        }
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setReady(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [opts.width, opts.height, opts.facingMode, opts.deviceId]);

  return { videoRef, ready, error };
}
```

- [ ] **Step 2:** Commit.

```bash
git add frontend/src/camera/useWebcam.ts
git commit -m "feat(frontend): useWebcam accepts deviceId option"
```

---

## Task 12: Frontend — CameraPicker component

**Files:**
- Create: `frontend/src/camera/CameraPicker.tsx`

- [ ] **Step 1:** Write:

```tsx
import { useEffect, useState } from 'react';
import { useAppState } from '../state/appState';

interface DeviceOption {
  deviceId: string;
  label: string;
}

export function CameraPicker() {
  const selectedDeviceId = useAppState((s) => s.selectedDeviceId);
  const setSelectedDeviceId = useAppState((s) => s.setSelectedDeviceId);
  const cameraId = useAppState((s) => s.cameraId);
  const setCameraId = useAppState((s) => s.setCameraId);

  const [devices, setDevices] = useState<DeviceOption[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        // Request a temporary stream so device labels are populated.
        const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
        tmp.getTracks().forEach((t) => t.stop());

        const all = await navigator.mediaDevices.enumerateDevices();
        const cams = all
          .filter((d) => d.kind === 'videoinput')
          .map((d, i) => ({
            deviceId: d.deviceId,
            label: d.label || `Camera ${i + 1}`,
          }));
        setDevices(cams);
        if (cams.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(cams[0].deviceId);
          setCameraId(`cam-${cams[0].deviceId.slice(0, 6) || 'default'}`);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    // selectedDeviceId/setters omitted intentionally — run once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <h4>Camera</h4>
      <select
        value={selectedDeviceId ?? ''}
        onChange={(e) => {
          setSelectedDeviceId(e.target.value);
          const dev = devices.find((d) => d.deviceId === e.target.value);
          if (dev) setCameraId(`cam-${dev.deviceId.slice(0, 6)}`);
        }}
        style={{
          width: '100%', padding: 6, background: '#0c0c0c',
          color: '#eee', border: '1px solid #333', borderRadius: 4,
        }}
      >
        {devices.length === 0 && <option value="">(no cameras)</option>}
        {devices.map((d) => (
          <option key={d.deviceId} value={d.deviceId}>{d.label}</option>
        ))}
      </select>

      <label style={{ display: 'block', marginTop: 8, fontSize: 12, color: '#aaa' }}>
        Camera id (used for clip folder name)
        <input
          type="text"
          value={cameraId}
          onChange={(e) => setCameraId(e.target.value)}
          style={{
            display: 'block', width: '100%', marginTop: 4, padding: 6,
            background: '#0c0c0c', color: '#eee', border: '1px solid #333', borderRadius: 4,
          }}
        />
      </label>

      {error && (
        <p style={{ color: '#f87171', fontSize: 12, marginTop: 6 }}>{error}</p>
      )}
    </div>
  );
}
```

- [ ] **Step 2:** Commit.

```bash
git add frontend/src/camera/CameraPicker.tsx
git commit -m "feat(frontend): CameraPicker component"
```

---

## Task 13: Frontend — send camera_id in packet header

**Files:**
- Modify: `frontend/src/stream/useFrameSender.ts`

- [ ] **Step 1:** Edit the file. Two specific changes:

(a) Subscribe to `cameraId` and add it to the ref. Replace the existing block:

```ts
  const yoloConf = useAppState((s) => s.yoloConf);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
```

with:

```ts
  const yoloConf = useAppState((s) => s.yoloConf);
  const cameraId = useAppState((s) => s.cameraId);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
```

(b) Replace the `stateRef` initialization line:

```ts
    const stateRef = useRef({ mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf });
    stateRef.current = { mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf };
```

with:

```ts
    const stateRef = useRef({ mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf, cameraId });
    stateRef.current = { mode, yoloSize, sam3Prompt, sam3Text, tracking, paused, yoloConf, cameraId };
```

(c) Replace the `buildHeader` function with:

```ts
    function buildHeader(s: typeof stateRef.current): Record<string, unknown> {
      const base: Record<string, unknown> = { camera_id: s.cameraId };
      if (s.mode.startsWith('yolo_')) {
        return { ...base, conf: s.yoloConf };
      }
      if (s.mode === 'sam3_image' || s.mode === 'sam3_video') {
        return {
          ...base,
          prompt: s.sam3Prompt,
          text: s.sam3Text,
          tracking: s.tracking,
        };
      }
      return base;
    }
```

Note: `buildHeader` is no longer ever `undefined` — every packet now carries `camera_id` at minimum. The signature change to non-optional return is intentional. `protocol.ts`'s `PacketParts.header` is already optional and `buildPacket` already handles both cases, so this is a non-breaking change.

- [ ] **Step 2:** Commit.

```bash
git add frontend/src/stream/useFrameSender.ts
git commit -m "feat(frontend): include camera_id in every WS packet header"
```

---

## Task 14: Frontend — mount CameraPicker in App.tsx

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1:** Two specific changes.

(a) Add an import near the top, after the `useWebcam` import:

```tsx
import { CameraPicker } from './camera/CameraPicker';
```

(b) Replace the `useWebcam` call and add `selectedDeviceId`:

```tsx
  const { videoRef, ready, error } = useWebcam();
```

with:

```tsx
  const selectedDeviceId = useAppState((s) => s.selectedDeviceId);
  const { videoRef, ready, error } = useWebcam({ deviceId: selectedDeviceId });
```

(c) Insert `<CameraPicker />` in the sidebar, right after the `<h2>` heading:

```tsx
        <h2 style={{ marginTop: 0 }}>Object Detection</h2>

        <CameraPicker />

        <h4>YOLO26 task</h4>
```

- [ ] **Step 2:** Verify the frontend still builds.

```bash
cd frontend
npm run build
```
Expected: build succeeds, no TypeScript errors.

- [ ] **Step 3:** Commit.

```bash
git add frontend/src/App.tsx
git commit -m "feat(frontend): wire CameraPicker into App"
```

---

## Task 15: Manual end-to-end verification

**Files:** none (manual test)

- [ ] **Step 1:** Start backend.

```bash
cd backend
.\.venv\Scripts\Activate.ps1   # PowerShell
uvicorn app.main:app --reload
```

- [ ] **Step 2:** Start frontend in a second shell.

```bash
cd frontend
npm run dev
```

- [ ] **Step 3:** Open `http://localhost:5173`. Pick a camera in the sidebar. Confirm:
  - Camera picker shows at least one device.
  - The default `cameraId` populates (e.g. `cam-abc123`).
  - Live inference still works as before (FPS reading is non-zero, overlays render).

- [ ] **Step 4:** Wave at the camera for ~10 seconds. Stop. Wait ~5 seconds.

- [ ] **Step 5:** Inspect the recordings folder.

```powershell
Get-ChildItem -Recurse backend\data\clips
```
Expected: at least one `.mp4` and one `.jpg` under `backend\data\clips\<cameraId>\<today>\`.

- [ ] **Step 6:** Open the `.mp4` in a player. Expected: ~12–17 seconds of video covering your wave (2s pre-roll + activity + 2s post-roll). The `.jpg` shows the trigger frame.

- [ ] **Step 7:** Hold still in front of the camera for 30 seconds. Expected: **no new files** while you're still. (One existing clip from when you sat down is acceptable.)

- [ ] **Step 8:** Wave for ~40 continuous seconds. Expected: 3 MP4 files (~15s + ~15s + ~10s) plus exactly **one** `.jpg` (per event, not per file).

- [ ] **Step 9:** Open a second browser tab to `http://localhost:5173`. (Same page is fine — the camera picker should let you keep the same camera or pick a different one if you have two webcams.) Set a different `cameraId` in the second tab (e.g. `cam-secondary`). Wave. Confirm a new `cam-secondary` folder appears next to the first.

- [ ] **Step 10:** Commit a stub note in the spec marking this feature as shipped.

```bash
git add docs/superpowers/specs/2026-05-26-motion-clip-recording-design.md
git commit --allow-empty -m "docs: motion-clip recording verified end-to-end"
```

---

## Self-review notes

- Spec section 2 decisions are each covered by a task: backend recording (Tasks 3/5/7/8), snapshot at trigger frame (Task 5 `_begin_event`), motion via bbox displacement (Tasks 2/3), one WS per camera + `camera_id` header (Tasks 10/12/13), back-to-back 15s rotation (Task 4 test + Task 5 `_rotate_clip`), 2s pre/post-roll (Task 5 `_begin_event` + post-roll branch), class allowlist defaulting to `["person"]` (Task 1), `mp4v` MP4 + `cv2.imwrite` JPG (Task 5), and the `data/clips/<camera_id>/<date>/<HHMMSS>.{mp4,jpg}` layout (Task 5 `_make_clip_path`).
- No placeholders, no "TBD".
- The class name `RecorderManager` and method `feed(camera_id, frame_bgr, detections)` are used consistently in Tasks 6–8.
- Threshold semantics intentionally diverged from spec (px → normalized) and the divergence is called out in the plan header.
