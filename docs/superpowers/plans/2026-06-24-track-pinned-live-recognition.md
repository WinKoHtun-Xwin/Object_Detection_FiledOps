# Track-Pinned Live Face Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live face names appear fast and stay glued to people by pinning recognized names to per-camera tracking IDs and running all face work off the WebSocket event loop.

**Architecture:** Drive Ultralytics' standalone `BYTETracker` once per camera off the shared, warm YOLO `.predict()`; stamp a `track_id` onto each detection box; keep a per-`(camera_id, track_id)` name cache filled by a single background worker thread; warm InsightFace at startup. The WebSocket loop only reads the cache and enqueues misses — it never blocks on the face engine.

**Tech Stack:** Python 3.12, FastAPI/WebSocket, Ultralytics (YOLO26 + bundled ByteTrack), InsightFace (`buffalo_l`), NumPy, pytest, ruff.

## Global Constraints

- Coordinates crossing the protocol are normalized `[0, 1]`; only the overlay layer converts to pixels.
- Backend behavior is tuned through `Settings` in `backend/app/config.py` — prefer a setting over a hardcoded constant.
- One YOLO model stays warm in VRAM; do not create per-camera YOLO model instances.
- The WebSocket inference loop must never block on face work and must never crash a stream on a recognition error.
- Live recognition is **display-only** — no DB writes from the live path (the clip worker still owns sightings / review-queue).
- Tests must run without a GPU or real model weights: use fakes/mocks for YOLO results and the face engine.
- Lint clean: `ruff check app` (line-length 100, target py312).
- Spec: [docs/superpowers/specs/2026-06-24-track-pinned-live-recognition-design.md](../specs/2026-06-24-track-pinned-live-recognition-design.md).
- All `pytest` / `ruff` commands run from `backend/` with the venv active (`.\.venv\Scripts\Activate.ps1`). All `git` commands run from the repo root.

---

## File Structure

- `backend/app/config.py` — **modify**: swap live-recognition settings.
- `backend/app/domain.py` — **modify**: add `Box.track_id`.
- `backend/app/inference/yolo.py` — **modify**: `_extract_boxes` takes `track_ids`; detect path runs the per-camera tracker.
- `backend/app/inference/tracking.py` — **create**: `CameraTrackers` + `camera_trackers` singleton.
- `backend/app/recognition/engine.py` — **modify**: configurable `det_size` + `warmup()`.
- `backend/app/recognition/live.py` — **rewrite**: track-pinned cache + background worker; new `FaceMatch`.
- `backend/app/recognition/__init__.py` — **modify**: add `live_face_engine`; wire `live_recognizer`.
- `backend/app/main.py` — **modify**: warm InsightFace + start/stop the live worker in `lifespan`.
- `backend/app/api/stream.py` — **modify**: `apply_live_names` helper; attach by `track_id`; disconnect cleanup.
- `backend/tests/inference/__init__.py` — **create**.
- `backend/tests/inference/test_tracking.py` — **create**.
- `backend/tests/inference/test_yolo_track_ids.py` — **create**.
- `backend/tests/recognition/test_live.py` — **rewrite**.
- `backend/tests/api/__init__.py` — **create**.
- `backend/tests/api/test_stream_attach.py` — **create**.

---

### Task 1: Config — swap live-recognition settings

Add the new track-pinned settings. (The two obsolete ones are removed in Task 5, where their last readers are rewritten, so every task stays green.)

**Files:**
- Modify: `backend/app/config.py`

**Interfaces:**
- Produces: `settings.live_recognition_refresh: float`, `settings.live_recognition_retry: float`, `settings.track_ttl: float`, `settings.recognition_queue_max: int`, `settings.live_det_size: int`.

- [ ] **Step 1: Add the new settings**

In `backend/app/config.py`, the current `# --- recognition ---` block ends with:

```python
    recognition_device: str = "cuda"
    live_recognition_interval: float = 1.0     # seconds between runs per camera
    live_recognition_cache_ttl: float = 3.0    # seconds a cached match stays valid
```

Insert the new fields immediately **after** that block (leave the two old fields in place for now):

```python
    # live recognition (track-pinned)
    live_recognition_refresh: float = 10.0   # re-confirm a CONFIRMED (high) track this often
    live_recognition_retry: float = 1.5      # backoff before retrying a TENTATIVE track
    track_ttl: float = 5.0                   # evict a track's cached name after this unseen gap (s)
    recognition_queue_max: int = 16          # max queued live face jobs before dropping
    live_det_size: int = 320                 # InsightFace det_size for live crops (offline stays 640)
```

- [ ] **Step 2: Verify settings import cleanly**

Run: `python -c "from app.config import settings; print(settings.live_recognition_refresh, settings.live_recognition_retry, settings.track_ttl, settings.recognition_queue_max, settings.live_det_size)"`
Expected: `10.0 1.5 5.0 16 320`

- [ ] **Step 3: Lint**

Run: `ruff check app`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add backend/app/config.py
git commit -m "feat(config): add track-pinned live recognition settings"
```

---

### Task 2: `Box.track_id` + serialization

Add an optional track ID to the `Box` domain object and let `_extract_boxes` stamp it.

**Files:**
- Modify: `backend/app/domain.py`
- Modify: `backend/app/inference/yolo.py` (`_extract_boxes`)
- Test: `backend/tests/inference/test_yolo_track_ids.py`
- Create: `backend/tests/inference/__init__.py`

**Interfaces:**
- Produces: `Box.track_id: int | None = None`; `_extract_boxes(result, w, h, track_ids: list[int | None] | None = None) -> list[dict]` where each dict gains a `track_id` key (`None` when no tracking).

- [ ] **Step 1: Create the test package init**

Create `backend/tests/inference/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test**

Create `backend/tests/inference/test_yolo_track_ids.py`:

```python
"""_extract_boxes stamps track IDs aligned to detection order."""
from __future__ import annotations

import numpy as np

from app.inference.yolo import _extract_boxes


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = _FakeTensor(xyxy)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)

    def __len__(self):
        return len(self.conf.arr)


class _FakeResult:
    def __init__(self, xyxy, cls, conf, names):
        self.boxes = _FakeBoxes(xyxy, cls, conf)
        self.names = names


def _result():
    return _FakeResult(
        xyxy=[[0.0, 0.0, 50.0, 50.0], [50.0, 50.0, 100.0, 100.0]],
        cls=[0, 16],
        conf=[0.9, 0.8],
        names={0: "person", 16: "dog"},
    )


def test_track_ids_default_none():
    boxes = _extract_boxes(_result(), w=100, h=100)
    assert [b["track_id"] for b in boxes] == [None, None]


def test_track_ids_applied_in_order():
    boxes = _extract_boxes(_result(), w=100, h=100, track_ids=[7, 3])
    assert boxes[0]["track_id"] == 7
    assert boxes[1]["track_id"] == 3
    assert boxes[0]["label"] == "person"
    assert boxes[1]["label"] == "dog"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/inference/test_yolo_track_ids.py -v`
Expected: FAIL — `_extract_boxes()` got an unexpected keyword `track_ids` / KeyError `track_id`.

- [ ] **Step 4: Add `track_id` to `Box`**

In `backend/app/domain.py`, change the `Box` dataclass to:

```python
@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    label: str = ""
    conf: float = 0.0
    track_id: int | None = None
```

- [ ] **Step 5: Thread `track_ids` through `_extract_boxes`**

In `backend/app/inference/yolo.py`, replace the whole `_extract_boxes` function with:

```python
def _extract_boxes(
    result: Any,
    w: int,
    h: int,
    track_ids: list[int | None] | None = None,
) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    names = result.names
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    out: list[dict[str, Any]] = []
    for i, ((x1, y1, x2, y2), c, p) in enumerate(zip(xyxy, cls, conf, strict=True)):
        tid = track_ids[i] if track_ids is not None and i < len(track_ids) else None
        out.append(
            asdict(
                Box(
                    x=float(x1) / w,
                    y=float(y1) / h,
                    w=float(x2 - x1) / w,
                    h=float(y2 - y1) / h,
                    label=names.get(int(c), str(c)),
                    conf=float(p),
                    track_id=tid,
                )
            )
        )
    return out
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/inference/test_yolo_track_ids.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Full suite + lint (no regressions from the new field)**

Run: `pytest -q && ruff check app`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add backend/app/domain.py backend/app/inference/yolo.py backend/tests/inference/__init__.py backend/tests/inference/test_yolo_track_ids.py
git commit -m "feat(inference): add Box.track_id and thread track ids through _extract_boxes"
```

---

### Task 3: `CameraTrackers` — per-camera ByteTrack

Drive one `BYTETracker` per camera, with a guard that keeps track IDs globally monotonic across cameras (Ultralytics' ID counter is class-global).

**Files:**
- Create: `backend/app/inference/tracking.py`
- Test: `backend/tests/inference/test_tracking.py`

**Interfaces:**
- Produces:
  - `class CameraTrackers` with `assign(camera_id: str, det, frame) -> list[int | None]` (one ID per detection, aligned to detection order; `None` where ByteTrack returned no track) and `drop(camera_id: str) -> None`.
  - module singleton `camera_trackers = CameraTrackers()`.
  - `det` is an Ultralytics `Boxes` (numpy-backed) exposing `.xywh`, `.conf`, `.cls`, `__len__`, and boolean `__getitem__`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/inference/test_tracking.py`:

```python
"""CameraTrackers: stable IDs across frames, isolation across cameras."""
from __future__ import annotations

import numpy as np

from app.inference.tracking import CameraTrackers


class _Dets:
    """Minimal stand-in for an Ultralytics numpy Boxes (what BYTETracker reads)."""

    def __init__(self, xywh, conf, cls):
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, m):
        return _Dets(self.xywh[m], self.conf[m], self.cls[m])


def _two_people():
    # xywh = center-x, center-y, width, height (pixels)
    return _Dets(
        xywh=[[100.0, 100.0, 40.0, 80.0], [400.0, 300.0, 40.0, 80.0]],
        conf=[0.9, 0.9],
        cls=[0.0, 0.0],
    )


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_ids_assigned_and_stable_across_frames():
    ct = CameraTrackers()
    ids1 = ct.assign("cam-A", _two_people(), _frame())
    ids2 = ct.assign("cam-A", _two_people(), _frame())
    assert len(ids1) == 2 and all(i is not None for i in ids1)
    assert ids1 == ids2            # same detections → same IDs next frame
    assert ids1[0] != ids1[1]      # two distinct people → distinct IDs


def test_ids_are_isolated_across_cameras():
    ct = CameraTrackers()
    a = ct.assign("cam-A", _two_people(), _frame())
    b = ct.assign("cam-B", _two_people(), _frame())
    # Independent trackers, but the monotonic-counter guard keeps IDs disjoint.
    assert set(a).isdisjoint(set(b))


def test_empty_detections_returns_empty():
    ct = CameraTrackers()
    out = ct.assign("cam-A", _Dets([], [], []), _frame())
    assert out == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/inference/test_tracking.py -v`
Expected: FAIL — `ModuleNotFoundError: app.inference.tracking`.

- [ ] **Step 3: Implement `CameraTrackers`**

Create `backend/app/inference/tracking.py`:

```python
"""Per-camera ByteTrack tracking, driven off the shared YOLO detections.

Detection stays a single warm `.predict()` (one model in VRAM); this layer
assigns a stable `track_id` to each detection, isolated per camera.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class CameraTrackers:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trackers: dict[str, Any] = {}
        self._args: Any | None = None

    def _ensure_args(self) -> Any:
        if self._args is None:
            from ultralytics.utils import YAML, IterableSimpleNamespace
            from ultralytics.utils.checks import check_yaml

            self._args = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
        return self._args

    def _tracker_for(self, camera_id: str) -> Any:
        tracker = self._trackers.get(camera_id)
        if tracker is not None:
            return tracker
        # BYTETracker.__init__ calls reset_id(), zeroing the class-global STrack
        # counter. Save/restore it so IDs stay monotonic (and disjoint) across cameras.
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = self._ensure_args()
        prev = BaseTrack._count
        tracker = BYTETracker(args=args)
        BaseTrack._count = prev
        self._trackers[camera_id] = tracker
        log.info("created ByteTracker for camera %s", camera_id)
        return tracker

    def assign(self, camera_id: str, det: Any, frame: np.ndarray) -> list[int | None]:
        n = len(det)
        ids: list[int | None] = [None] * n
        if n == 0:
            return ids
        with self._lock:
            tracker = self._tracker_for(camera_id)
            tracks = tracker.update(det, frame)
        # Each row is [x1, y1, x2, y2, track_id, score, cls, idx]; idx maps back to
        # the original detection order.
        for row in tracks:
            idx = int(row[-1])
            if 0 <= idx < n:
                ids[idx] = int(row[4])
        return ids

    def drop(self, camera_id: str) -> None:
        with self._lock:
            self._trackers.pop(camera_id, None)


camera_trackers = CameraTrackers()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/inference/test_tracking.py -v`
Expected: PASS (3 passed).

> If `test_ids_are_isolated_across_cameras` fails with overlapping IDs, the `BaseTrack._count` save/restore guard is missing or wrong — re-check Step 3.

- [ ] **Step 5: Lint**

Run: `ruff check app`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/app/inference/tracking.py backend/tests/inference/test_tracking.py
git commit -m "feat(inference): per-camera ByteTrack tracker with monotonic id guard"
```

---

### Task 4: Wire tracking into the YOLO detect path

When the frame header carries `tracking` + `camera_id`, run the per-camera tracker over the detect result and stamp track IDs.

**Files:**
- Modify: `backend/app/inference/yolo.py` (`YoloEngine.infer`)
- Test: `backend/tests/inference/test_yolo_track_ids.py` (add a case)

**Interfaces:**
- Consumes: `camera_trackers.assign` (Task 3), `_extract_boxes(..., track_ids=...)` (Task 2).
- Produces: detect results whose `boxes[i]["track_id"]` is populated when tracking is requested.

- [ ] **Step 1: Add the failing wiring test**

Append to `backend/tests/inference/test_yolo_track_ids.py`:

```python
def test_infer_detect_stamps_track_ids(monkeypatch):
    from unittest.mock import MagicMock

    from app.inference.yolo import YoloEngine
    import app.inference.tracking as trackmod

    eng = object.__new__(YoloEngine)   # bypass __init__ (no model load / GPU)
    eng.task = "detect"
    eng.size = "n"
    eng.device = "cpu"
    eng.model = MagicMock()
    eng.model.predict.return_value = [_result()]

    captured = {}

    def fake_assign(camera_id, det, frame):
        captured["camera_id"] = camera_id
        return [11, 22]

    monkeypatch.setattr(trackmod.camera_trackers, "assign", fake_assign)

    out = eng.infer(
        np.zeros((100, 100, 3), dtype=np.uint8),
        {"tracking": True, "camera_id": "cam-A", "conf": 0.25},
    )
    assert out["type"] == "detect"
    assert [b["track_id"] for b in out["boxes"]] == [11, 22]
    assert captured["camera_id"] == "cam-A"


def test_infer_detect_no_tracking_leaves_ids_none(monkeypatch):
    from unittest.mock import MagicMock

    from app.inference.yolo import YoloEngine
    import app.inference.tracking as trackmod

    eng = object.__new__(YoloEngine)
    eng.task = "detect"
    eng.size = "n"
    eng.device = "cpu"
    eng.model = MagicMock()
    eng.model.predict.return_value = [_result()]

    def boom(*a, **k):
        raise AssertionError("assign must not be called without tracking")

    monkeypatch.setattr(trackmod.camera_trackers, "assign", boom)

    out = eng.infer(np.zeros((100, 100, 3), dtype=np.uint8), {"conf": 0.25})
    assert [b["track_id"] for b in out["boxes"]] == [None, None]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/inference/test_yolo_track_ids.py -k infer -v`
Expected: FAIL — `track_id` is `None` even with tracking (assign not wired yet).

- [ ] **Step 3: Wire the detect branch**

In `backend/app/inference/yolo.py`, replace the `infer` method body's tail (the `if self.task == ...` chain) so the **detect** branch runs the tracker. The full method becomes:

```python
    def infer(self, frame_bgr: np.ndarray, header: dict[str, Any]) -> dict[str, Any]:
        conf = float(header.get("conf", 0.25))
        h, w = frame_bgr.shape[:2]
        t0 = time.perf_counter()
        result = self.model.predict(
            frame_bgr, device=self.device, conf=conf, verbose=False
        )[0]
        dt = (time.perf_counter() - t0) * 1000.0
        log.debug("yolo %s/%s %.0fms", self.task, self.size, dt)

        if self.task == "detect":
            track_ids = None
            camera_id = header.get("camera_id")
            if header.get("tracking") and camera_id:
                from app.inference.tracking import camera_trackers

                det = result.boxes.cpu().numpy()
                track_ids = camera_trackers.assign(camera_id, det, frame_bgr)
            return {"type": "detect", "boxes": _extract_boxes(result, w, h, track_ids)}
        if self.task == "seg":
            return {"type": "sam3", "masks": _extract_seg_masks(result, w, h)}
        if self.task == "pose":
            return {"type": "pose", "people": _extract_pose(result, w, h)}
        if self.task == "obb":
            return {"type": "obb", "obboxes": _extract_obb(result, w, h)}
        if self.task == "cls":
            return {"type": "cls", "topk": _extract_cls(result)}
        raise AssertionError(f"unknown task {self.task}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/inference/test_yolo_track_ids.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Lint**

Run: `ruff check app`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add backend/app/inference/yolo.py backend/tests/inference/test_yolo_track_ids.py
git commit -m "feat(inference): run per-camera tracker in YOLO detect path when tracking enabled"
```

---

### Task 5: Rework `LiveRecognizer` — track-pinned cache + worker

Replace the throttle/snapshot recognizer with a `(camera_id, track_id)` name cache filled by a background worker. Redefine `FaceMatch`, rewrite its tests, and remove the obsolete config fields.

**Files:**
- Rewrite: `backend/app/recognition/live.py`
- Rewrite: `backend/tests/recognition/test_live.py`
- Modify: `backend/app/config.py` (remove the two obsolete fields)

**Interfaces:**
- Produces:
  - `class FaceMatch` (frozen): `name: str`, `score: float`, `kind: str` (`"high" | "mid" | "unknown"`).
  - `class LiveRecognizer(engine, gallery, clock=time.monotonic)` with:
    - `annotate(camera_id: str, frame_bgr, person_boxes: list[dict]) -> dict[int, FaceMatch]` — O(1); reads cache, enqueues misses; key = `track_id`.
    - `start() -> None`, `stop() -> None`, `clear_camera(camera_id: str) -> None`.
    - `_drain_for_tests() -> None` — process all queued jobs synchronously (test-only, mirrors `dbmod._reset_for_tests`).
  - Each `person_boxes` dict carries `track_id`, `x`, `y`, `w`, `h`.

- [ ] **Step 1: Rewrite the tests (failing)**

Replace the entire contents of `backend/tests/recognition/test_live.py`:

```python
"""Tests for the track-pinned LiveRecognizer."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.recognition import db as dbmod
from app.recognition.engine import FaceResult


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from app import config as cfg

    patched = dataclasses.replace(
        cfg.settings,
        db_path=tmp_path / "app.db",
        faces_dir=tmp_path / "faces",
    )
    monkeypatch.setattr(cfg, "settings", patched)
    monkeypatch.setattr(dbmod, "settings", patched)
    dbmod._reset_for_tests()
    dbmod.init_schema()
    return tmp_path


def _unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _pb(track_id: int | None) -> dict:
    return {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5, "label": "person", "track_id": track_id}


def _frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _engine_returning(emb: np.ndarray, det_score: float = 0.99) -> MagicMock:
    engine = MagicMock()
    engine.detect_and_embed.return_value = [
        FaceResult(
            crop_bgr=np.zeros((112, 112, 3), dtype=np.uint8),
            embedding=emb,
            bbox=(10, 10, 110, 110),
            det_score=det_score,
        )
    ]
    return engine


def _gallery_with(name: str, emb: np.ndarray):
    from app.recognition.gallery import Gallery

    pid = dbmod.add_person(name)
    dbmod.add_face(pid, f"faces/{pid}/a.jpg", emb.tobytes(), source="upload")
    return Gallery()


def test_new_track_enqueues_once_then_resolves(env: Path) -> None:
    from app.recognition.live import LiveRecognizer

    alice = _unit(1)
    g = _gallery_with("Alice", alice)
    lr = LiveRecognizer(engine=_engine_returning(alice), gallery=g, clock=FakeClock())

    first = lr.annotate("cam-A", _frame(), [_pb(1)])
    assert first == {}                       # brand-new track: not known yet
    assert lr._q.qsize() == 1

    lr._drain_for_tests()
    second = lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._engine.detect_and_embed.call_count == 1
    assert second[1].name == "Alice"
    assert second[1].kind == "high"
    assert lr._q.qsize() == 0                # confirmed → no re-enqueue


def test_confirmed_track_refreshes_only_after_interval(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.config import settings

    alice = _unit(1)
    clock = FakeClock()
    g = _gallery_with("Alice", alice)
    lr = LiveRecognizer(engine=_engine_returning(alice), gallery=g, clock=clock)

    lr.annotate("cam-A", _frame(), [_pb(1)])
    lr._drain_for_tests()

    clock.t += settings.live_recognition_refresh - 0.1
    lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._q.qsize() == 0                # not yet due

    clock.t += 0.2
    lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._q.qsize() == 1                # refresh due


def test_no_face_is_tentative_and_retries_after_backoff(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.config import settings

    clock = FakeClock()
    g = _gallery_with("Alice", _unit(1))
    engine = MagicMock()
    engine.detect_and_embed.return_value = []      # person facing away
    lr = LiveRecognizer(engine=engine, gallery=g, clock=clock)

    lr.annotate("cam-A", _frame(), [_pb(1)])
    lr._drain_for_tests()
    shown = lr.annotate("cam-A", _frame(), [_pb(1)])
    assert shown[1].name == "Unknown"
    assert shown[1].kind == "unknown"
    assert lr._q.qsize() == 0                # within retry backoff

    clock.t += settings.live_recognition_retry + 0.01
    lr.annotate("cam-A", _frame(), [_pb(1)])
    assert lr._q.qsize() == 1                # retry due


def test_track_without_id_is_ignored(env: Path) -> None:
    from app.recognition.live import LiveRecognizer

    g = _gallery_with("Alice", _unit(1))
    lr = LiveRecognizer(engine=_engine_returning(_unit(1)), gallery=g, clock=FakeClock())
    out = lr.annotate("cam-A", _frame(), [_pb(None)])
    assert out == {}
    assert lr._q.qsize() == 0


def test_eviction_after_ttl(env: Path) -> None:
    from app.recognition.live import LiveRecognizer
    from app.config import settings

    clock = FakeClock()
    g = _gallery_with("Alice", _unit(1))
    lr = LiveRecognizer(engine=_engine_returning(_unit(1)), gallery=g, clock=clock)

    lr.annotate("cam-A", _frame(), [_pb(1)])
    lr._drain_for_tests()
    assert 1 in lr._state["cam-A"]

    clock.t += settings.track_ttl + 0.1
    lr.annotate("cam-A", _frame(), [_pb(2)])     # any later frame triggers eviction
    assert 1 not in lr._state.get("cam-A", {})


def test_queue_cap_drops_excess(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app import config as cfg
    from app.recognition.live import LiveRecognizer

    capped = dataclasses.replace(cfg.settings, recognition_queue_max=2)
    monkeypatch.setattr("app.recognition.live.settings", capped)

    g = _gallery_with("Alice", _unit(1))
    lr = LiveRecognizer(engine=_engine_returning(_unit(1)), gallery=g, clock=FakeClock())
    lr.annotate("cam-A", _frame(), [_pb(1), _pb(2), _pb(3), _pb(4), _pb(5)])
    assert lr._q.qsize() == 2                # capped; extras dropped, not buffered


def test_per_camera_isolation(env: Path) -> None:
    from app.recognition.live import LiveRecognizer

    alice = _unit(1)
    bob = _unit(2)
    dbmod.add_face(dbmod.add_person("Alice"), "faces/a.jpg", alice.tobytes(), source="upload")
    dbmod.add_face(dbmod.add_person("Bob"), "faces/b.jpg", bob.tobytes(), source="upload")
    from app.recognition.gallery import Gallery

    g = Gallery()
    engine = MagicMock()
    engine.detect_and_embed.side_effect = [
        [FaceResult(np.zeros((112, 112, 3), np.uint8), alice, (0, 0, 1, 1), 0.99)],
        [FaceResult(np.zeros((112, 112, 3), np.uint8), bob, (0, 0, 1, 1), 0.99)],
    ]
    lr = LiveRecognizer(engine=engine, gallery=g, clock=FakeClock())

    lr.annotate("cam-A", _frame(), [_pb(1)])   # cam-A track 1
    lr.annotate("cam-B", _frame(), [_pb(1)])   # cam-B track 1 (same id, different camera)
    lr._drain_for_tests()

    a = lr.annotate("cam-A", _frame(), [_pb(1)])
    b = lr.annotate("cam-B", _frame(), [_pb(1)])
    assert a[1].name == "Alice"
    assert b[1].name == "Bob"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/recognition/test_live.py -v`
Expected: FAIL — old `LiveRecognizer`/`FaceMatch` shape (e.g. `_q`, `_state`, `_drain_for_tests`, dict return all absent).

- [ ] **Step 3: Rewrite `live.py`**

Replace the entire contents of `backend/app/recognition/live.py`:

```python
"""Track-pinned live face recognition.

A recognized name is pinned to (camera_id, track_id) and reused every frame at
zero cost. Face detect/embed/match runs on a single background worker thread,
off the WebSocket event loop. The per-frame `annotate` call only reads the cache
and enqueues misses — it never calls the face engine inline.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.config import settings
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceMatch:
    name: str       # "Alice" | "Unknown"
    score: float
    kind: str       # "high" | "mid" | "unknown"


@dataclass
class _TrackName:
    name: str = "Unknown"
    score: float = 0.0
    kind: str = "unknown"
    resolved_at: float = 0.0       # last successful recognition (0.0 = never)
    last_seen: float = 0.0
    next_attempt_at: float = 0.0   # earliest clock time we may (re)recognize


@dataclass
class _Job:
    camera_id: str
    track_id: int
    crop: np.ndarray


class LiveRecognizer:
    def __init__(
        self,
        engine: FaceEngine,
        gallery: Gallery,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._gallery = gallery
        self._clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, dict[int, _TrackName]] = {}
        self._inflight: set[tuple[str, int]] = set()
        self._q: queue.Queue[_Job] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- worker lifecycle ----
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        t = threading.Thread(target=self._run, name="LiveRecognizer", daemon=True)
        t.start()
        self._thread = t
        log.info("live recognizer worker started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        log.info("live recognizer worker stopped")

    def clear_camera(self, camera_id: str) -> None:
        with self._lock:
            self._state.pop(camera_id, None)
            self._inflight = {k for k in self._inflight if k[0] != camera_id}

    # ---- per-frame (event loop thread) ----
    def annotate(
        self,
        camera_id: str,
        frame_bgr: np.ndarray,
        person_boxes: list[dict],
    ) -> dict[int, FaceMatch]:
        now = self._clock()
        h, w = frame_bgr.shape[:2]
        result: dict[int, FaceMatch] = {}
        with self._lock:
            cam = self._state.setdefault(camera_id, {})
            for pb in person_boxes:
                tid = pb.get("track_id")
                if tid is None:
                    continue
                st = cam.get(tid)
                if st is None:
                    st = _TrackName(last_seen=now, next_attempt_at=now)
                    cam[tid] = st
                st.last_seen = now
                if st.resolved_at > 0.0:
                    result[tid] = FaceMatch(st.name, st.score, st.kind)
                if now >= st.next_attempt_at:
                    crop = self._crop(frame_bgr, pb, w, h)
                    if crop is not None and self._enqueue(camera_id, tid, crop):
                        st.next_attempt_at = now + settings.live_recognition_retry
            self._evict(now)
        return result

    def _enqueue(self, camera_id: str, track_id: int, crop: np.ndarray) -> bool:
        key = (camera_id, track_id)
        if key in self._inflight:
            return False
        if self._q.qsize() >= settings.recognition_queue_max:
            return False
        self._inflight.add(key)
        self._q.put(_Job(camera_id, track_id, crop))
        return True

    @staticmethod
    def _crop(frame_bgr: np.ndarray, pb: dict, w: int, h: int) -> np.ndarray | None:
        x1 = max(0, int(round(float(pb["x"]) * w)))
        y1 = max(0, int(round(float(pb["y"]) * h)))
        x2 = min(w, int(round((float(pb["x"]) + float(pb["w"])) * w)))
        y2 = min(h, int(round((float(pb["y"]) + float(pb["h"])) * h)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop.copy()   # worker owns it, independent of the live frame buffer

    def _evict(self, now: float) -> None:
        ttl = settings.track_ttl
        for cam_id, cam in list(self._state.items()):
            for tid, st in list(cam.items()):
                if now - st.last_seen > ttl:
                    del cam[tid]
            if not cam:
                del self._state[cam_id]

    # ---- background worker ----
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process_job(job)
            except Exception:
                log.exception("live recognition job failed")
                self._mark_tentative(job)
            finally:
                with self._lock:
                    self._inflight.discard((job.camera_id, job.track_id))
                self._q.task_done()

    def _drain_for_tests(self) -> None:
        """Process all queued jobs on the calling thread (deterministic tests)."""
        while True:
            try:
                job = self._q.get_nowait()
            except queue.Empty:
                return
            try:
                self._process_job(job)
            finally:
                with self._lock:
                    self._inflight.discard((job.camera_id, job.track_id))

    def _process_job(self, job: _Job) -> None:
        now = self._clock()
        faces = self._engine.detect_and_embed(job.crop)
        if not faces:
            self._write(job, "Unknown", 0.0, "unknown", now)
            return
        face = max(faces, key=lambda f: f.det_score)
        pid, score = self._gallery.match(face.embedding)
        kind = self._classify(score, pid)
        if kind == "unknown" or pid is None:
            name = "Unknown"
        else:
            person = dbmod.get_person(pid)
            name = person["name"] if person else "Unknown"
        self._write(job, name, float(score), kind, now)

    def _write(self, job: _Job, name: str, score: float, kind: str, now: float) -> None:
        with self._lock:
            cam = self._state.setdefault(job.camera_id, {})
            st = cam.get(job.track_id) or _TrackName()
            st.name = name
            st.score = score
            st.kind = kind
            st.resolved_at = now
            st.last_seen = max(st.last_seen, now)
            st.next_attempt_at = now + (
                settings.live_recognition_refresh
                if kind == "high"
                else settings.live_recognition_retry
            )
            cam[job.track_id] = st

    def _mark_tentative(self, job: _Job) -> None:
        now = self._clock()
        with self._lock:
            cam = self._state.setdefault(job.camera_id, {})
            st = cam.get(job.track_id) or _TrackName()
            st.next_attempt_at = now + settings.live_recognition_retry
            cam[job.track_id] = st

    @staticmethod
    def _classify(score: float, pid: int | None) -> str:
        if pid is None:
            return "unknown"
        if score >= settings.match_high:
            return "high"
        if score >= settings.match_low:
            return "mid"
        return "unknown"
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/recognition/test_live.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Remove the obsolete config fields**

In `backend/app/config.py`, delete these two now-unused lines from the `# --- recognition ---` block:

```python
    live_recognition_interval: float = 1.0     # seconds between runs per camera
    live_recognition_cache_ttl: float = 3.0    # seconds a cached match stays valid
```

- [ ] **Step 6: Verify nothing else referenced them**

Run: `grep -rn "live_recognition_interval\|live_recognition_cache_ttl" app tests`
Expected: no matches.

- [ ] **Step 7: Full suite + lint**

Run: `pytest -q && ruff check app`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add backend/app/recognition/live.py backend/tests/recognition/test_live.py backend/app/config.py
git commit -m "feat(recognition): track-pinned live recognizer with async worker"
```

---

### Task 6: Warm InsightFace + wire the live engine and worker

Give `FaceEngine` a configurable `det_size` + `warmup()`, add a dedicated live engine (320²), and warm/start it in `lifespan`.

**Files:**
- Modify: `backend/app/recognition/engine.py`
- Modify: `backend/app/recognition/__init__.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/recognition/test_engine_config.py` (create)

**Interfaces:**
- Consumes: `LiveRecognizer.start/stop` (Task 5), `settings.live_det_size` (Task 1).
- Produces: `FaceEngine(device=None, det_size: int = 640)` with `warmup() -> None`; singletons `live_face_engine` and `live_recognizer` (now backed by `live_face_engine`).

- [ ] **Step 1: Write the failing test**

Create `backend/tests/recognition/test_engine_config.py`:

```python
"""FaceEngine carries a configurable det_size (no model load needed)."""
from __future__ import annotations

from app.recognition.engine import FaceEngine


def test_default_det_size_is_640():
    assert FaceEngine()._det_size == 640


def test_custom_det_size():
    assert FaceEngine(det_size=320)._det_size == 320
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/recognition/test_engine_config.py -v`
Expected: FAIL — `FaceEngine` has no `_det_size`.

- [ ] **Step 3: Add `det_size` + `warmup()` to `FaceEngine`**

In `backend/app/recognition/engine.py`, replace the `__init__` and `_ensure_loaded` methods and add `warmup`:

```python
    def __init__(self, device: str | None = None, det_size: int = 640) -> None:
        self._device = device or settings.recognition_device
        self._det_size = int(det_size)
        self._app: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        log.info(
            "loading InsightFace buffalo_l (providers=%s, det_size=%d)",
            providers,
            self._det_size,
        )
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(
            ctx_id=0 if self._device == "cuda" else -1,
            det_size=(self._det_size, self._det_size),
        )
        self._app = app

    def warmup(self) -> None:
        """Load the model and run one dummy pass so the first real call is fast."""
        self._ensure_loaded()
        dummy = np.zeros((self._det_size, self._det_size, 3), dtype=np.uint8)
        self._app.get(dummy)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/recognition/test_engine_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Wire the live engine singleton**

Replace the body of `backend/app/recognition/__init__.py`:

```python
"""Recognition subsystem."""
from app.config import settings
from app.recognition import db as dbmod
from app.recognition.engine import FaceEngine
from app.recognition.gallery import Gallery
from app.recognition.live import LiveRecognizer
from app.recognition.worker import RecognitionWorker

dbmod.init_schema()
face_engine = FaceEngine()                                   # 640² — offline clip worker
live_face_engine = FaceEngine(det_size=settings.live_det_size)  # smaller — live crops
gallery = Gallery()
recognition_worker = RecognitionWorker(engine=face_engine, gallery=gallery)
live_recognizer = LiveRecognizer(engine=live_face_engine, gallery=gallery)

__all__ = [
    "face_engine",
    "live_face_engine",
    "gallery",
    "recognition_worker",
    "live_recognizer",
]
```

- [ ] **Step 6: Warm + start/stop in `lifespan`**

In `backend/app/main.py`, update the imports and `lifespan`:

```python
import logging

from app.recognition import db as dbmod
from app.recognition import gallery, live_face_engine, live_recognizer, recognition_worker

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.clips_dir.mkdir(parents=True, exist_ok=True)
    settings.faces_dir.mkdir(parents=True, exist_ok=True)
    dbmod.init_schema()
    gallery.reload()
    recognition_worker.start()
    live_recognizer.start()
    try:
        live_face_engine.warmup()
    except Exception:
        log.exception("live face engine warmup failed")
    try:
        yield
    finally:
        recognition_worker.stop()
        live_recognizer.stop()
```

- [ ] **Step 7: App imports cleanly + full suite + lint**

Run: `python -c "import app.main" && pytest -q && ruff check app`
Expected: import OK, all tests green, no lint errors.

- [ ] **Step 8: Commit**

```bash
git add backend/app/recognition/engine.py backend/app/recognition/__init__.py backend/app/main.py backend/tests/recognition/test_engine_config.py
git commit -m "feat(recognition): dedicated warm live face engine started in lifespan"
```

---

### Task 7: Stream attach-by-track_id + disconnect cleanup + final verification

Replace the drifting index mapping with a pure, tested `apply_live_names` helper keyed by `track_id`, and clean up tracker + cache on disconnect.

**Files:**
- Modify: `backend/app/api/stream.py`
- Create: `backend/tests/api/__init__.py`
- Test: `backend/tests/api/test_stream_attach.py`

**Interfaces:**
- Consumes: `live_recognizer.annotate -> dict[int, FaceMatch]` (Task 5), `camera_trackers.drop` (Task 3), `live_recognizer.clear_camera` (Task 5).
- Produces: `apply_live_names(boxes: list[dict], names: dict[int, FaceMatch]) -> None` — mutates each box's `label` in place.

- [ ] **Step 1: Create the api test package init**

Create `backend/tests/api/__init__.py` (empty file).

- [ ] **Step 2: Write the failing test**

Create `backend/tests/api/test_stream_attach.py`:

```python
"""apply_live_names attaches cached names to boxes by track_id (no drift)."""
from __future__ import annotations

from app.api.stream import apply_live_names
from app.recognition.live import FaceMatch


def test_attaches_high_match_with_score():
    boxes = [{"label": "person", "track_id": 2}]
    apply_live_names(boxes, {2: FaceMatch("Bob", 0.7, "high")})
    assert boxes[0]["label"] == "person · Bob 0.70"


def test_unknown_kind_gets_plain_suffix():
    boxes = [{"label": "person", "track_id": 5}]
    apply_live_names(boxes, {5: FaceMatch("Unknown", 0.0, "unknown")})
    assert boxes[0]["label"] == "person · Unknown"


def test_untracked_and_unmatched_boxes_unchanged():
    boxes = [{"label": "person"}, {"label": "dog", "track_id": 9}]
    apply_live_names(boxes, {2: FaceMatch("Bob", 0.7, "high")})
    assert boxes[0]["label"] == "person"
    assert boxes[1]["label"] == "dog"


def test_no_drift_with_multiple_people():
    boxes = [{"label": "person", "track_id": 7}, {"label": "person", "track_id": 3}]
    names = {3: FaceMatch("Cara", 0.8, "high"), 7: FaceMatch("Dan", 0.6, "mid")}
    apply_live_names(boxes, names)
    assert boxes[0]["label"] == "person · Dan 0.60"
    assert boxes[1]["label"] == "person · Cara 0.80"
```

- [ ] **Step 3: Run to verify it fails**

Run: `pytest tests/api/test_stream_attach.py -v`
Expected: FAIL — `cannot import name 'apply_live_names'`.

- [ ] **Step 4: Add the helper and rewire the stream**

In `backend/app/api/stream.py`, update the imports at the top:

```python
from app.protocol.frame import FrameDecodeError, parse
from app.inference.tracking import camera_trackers
from app.recognition import live_recognizer
from app.recording import recorder_manager
from app.runtime.codec import jpeg_to_bgr
from app.runtime.registry import registry
```

Add this helper near the top of the module (after `SIZE_BY_VARIANT`):

```python
def apply_live_names(boxes: list[dict[str, Any]], names: dict[int, Any]) -> None:
    """Append recognized names to box labels, matched by track_id."""
    for b in boxes:
        tid = b.get("track_id")
        if tid is None:
            continue
        m = names.get(tid)
        if m is None:
            continue
        suffix = f" · {m.name} {m.score:.2f}" if m.kind != "unknown" else " · Unknown"
        b["label"] = f"{b.get('label', 'person')}{suffix}"
```

Replace the recognize block (the `if camera_id and isinstance(fp.header, dict) and fp.header.get("recognize"):` section) with:

```python
            if camera_id and isinstance(fp.header, dict) and fp.header.get("recognize"):
                result_boxes = result.get("boxes") or []
                person_boxes = [b for b in result_boxes if b.get("label") == "person"]
                if person_boxes:
                    try:
                        names = live_recognizer.annotate(camera_id, img, person_boxes)
                        apply_live_names(result_boxes, names)
                    except Exception:
                        log.exception("live recognition failed")
```

Update the disconnect handler to also drop the tracker and live cache:

```python
    except WebSocketDisconnect:
        log.info("ws disconnected: %s (after %d frames)", ws.client, frame_id)
        if current_camera_id is not None:
            recorder_manager.close(current_camera_id)
            camera_trackers.drop(current_camera_id)
            live_recognizer.clear_camera(current_camera_id)
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/api/test_stream_attach.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Full suite + lint**

Run: `pytest -q && ruff check app`
Expected: all green, no lint errors.

- [ ] **Step 7: Manual smoke verification (real GPU)**

Start the backend (`uvicorn app.main:app --reload`) and the frontend (`npm run dev`). With **Tracking** and **Recognize faces** both on:
- A new person's name appears within roughly a second of entering frame (not several seconds).
- The name stays on the correct person as they move and as a second person enters — names do not swap.
- Backend log shows `live recognizer worker started` and `created ByteTracker for camera ...` once per camera.
- Confirm the server→client `boxes` payload now includes a `track_id` field (browser devtools → WS frames, or add a temporary log).

- [ ] **Step 8: Commit**

```bash
git add backend/app/api/stream.py backend/tests/api/__init__.py backend/tests/api/test_stream_attach.py
git commit -m "feat(stream): attach live names by track_id and clean up trackers on disconnect"
```

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:
- New `tracking.py` / `CameraTrackers` → Task 3; per-camera isolation + class-global ID guard → Task 3 (`_tracker_for`).
- `Box.track_id` + serialization → Task 2.
- `YoloEngine` detect-path tracking → Task 4.
- Reworked `LiveRecognizer` (cache + state machine + async worker + cadence) → Task 5.
- Startup warmup + worker lifecycle → Task 6.
- `stream.py` attach-by-track_id + disconnect cleanup → Task 7.
- Config changes (add new / remove obsolete) → Task 1 (add) + Task 5 (remove).
- Edge cases: no-track-id ignored (Task 5 test `test_track_without_id_is_ignored`, Task 7 helper), no-face → tentative (Task 5), eviction (Task 5), queue cap (Task 5), per-camera isolation (Tasks 3 & 5), disconnect cleanup (Task 7).
- Error handling: worker try/except → `_mark_tentative` (Task 5); stream try/except (Task 7).
- Testing plan: tracking, live, stream-attach, yolo track-ids, engine-config — all present.
- Out of scope honored: no frontend overlay change, detect-only tracking, no live DB writes.

**2. Placeholder scan** — no TBD/TODO; every code step contains full code; manual GPU smoke (Task 7 Step 7) is an explicit, concrete checklist, used only where unit-testing a real model/GPU is impractical (the wiring itself is unit-tested in Task 4 with fakes).

**3. Type consistency** — `FaceMatch(name, score, kind)` is defined in Task 5 and consumed identically in Tasks 5 & 7. `annotate(...) -> dict[int, FaceMatch]` (Task 5) matches its caller in Task 7. `_extract_boxes(result, w, h, track_ids=None)` (Task 2) matches its caller in Task 4. `camera_trackers.assign(camera_id, det, frame)` / `.drop(camera_id)` (Task 3) match callers in Tasks 4 & 7. `FaceEngine(device, det_size)` + `warmup()` (Task 6) match callers in `__init__`/`main`. `clear_camera` / `start` / `stop` consistent across Tasks 5–7.
