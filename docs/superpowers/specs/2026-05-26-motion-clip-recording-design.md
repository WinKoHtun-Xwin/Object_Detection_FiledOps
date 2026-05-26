# Motion-Triggered Clip Recording — Design

**Date:** 2026-05-26
**Status:** Drafted from brainstorming session, awaiting user review
**Scope:** Backend + frontend changes to record 15-second MP4 clips (plus one snapshot) per motion event, per camera.

---

## 1. Goal

Record video from the live inference stream **only when there is meaningful activity**. Specifically:

- Save a `.mp4` clip when an allowlisted object (default: `person`) is detected **and** its bounding box is moving.
- Save a single `.jpg` snapshot at the start of each motion event, matching the clip's basename.
- When the scene is empty or static, write nothing.
- Support multiple cameras independently — each camera has its own recorder and its own folder of clips.

Non-goals (deferred):

- No retention / cleanup policy. Disk fills until manually purged.
- No UI for browsing clips.
- No audio.
- No backend-pulled (RTSP) cameras — browser webcams only.
- No multi-stream in a single browser tab. One tab = one camera.

## 2. Decisions (resolved in brainstorm)

| # | Decision | Choice |
|---|---|---|
| 1 | Where recording happens | Backend records (uses the JPEGs already arriving over the WS) |
| 2 | Snapshot policy | One `.jpg` per motion event, captured at motion start |
| 3 | Motion definition | YOLO bbox center displacement > threshold across recent frames |
| 4 | Multi-camera transport | One WebSocket connection per camera; `camera_id` carried in packet header |
| 5 | Clip duration during sustained motion | Back-to-back 15s files until motion ends |
| 6 | Pre/post-roll | 2s pre-roll buffer, 2s post-roll grace |
| 7 | Class filter | Configurable allowlist, default `["person"]` |
| 8 | Container/codec | `.mp4` via OpenCV `VideoWriter` with `mp4v` fourcc |
| 9 | On-disk layout | `data/clips/<camera_id>/<YYYY-MM-DD>/<HHMMSS>.{mp4,jpg}` |

## 3. Architecture

The recorder hooks into the existing WebSocket inference path **after** YOLO inference, per camera. Inference output (boxes / people) is the input to motion detection. No change to the protocol, no change to inference, no change to the frontend's render pipeline.

```
client (browser tab = one camera)
    │  binary packet  (header now includes camera_id)
    ▼
/ws ──► parse ──► jpeg_to_bgr ──► YOLO ──► JSON result back to client
                                    │
                                    ▼
                          RecorderManager.feed(camera_id, frame_bgr, detections)
                          └── MotionRecorder[camera_id]
                              ├── 2s ring buffer of (timestamp, bgr_frame)
                              ├── MotionDetector (bbox displacement)
                              ├── state machine: IDLE → RECORDING → POST_ROLL → IDLE
                              └── cv2.VideoWriter for current clip
```

Disk writes run synchronously inside the WS handler after the response has been sent. Measured cost is expected to be a few ms per frame; if profiling later shows this hurting inference FPS, move the writer to a background thread via `queue.Queue`. YAGNI until measured.

## 4. Backend components

### 4.1 New files

#### `backend/app/recording/motion.py` — `MotionDetector`

Per-camera class. Stateless across cameras.

```python
class MotionDetector:
    def __init__(self, allowed_classes: set[str], px_threshold: int, history: int = 5) -> None: ...
    def update(self, detections: list[dict]) -> bool:
        """Return True if any allowlisted class's bbox center moved > px_threshold
        vs its nearest match in the previous frame."""
```

- Keeps the last `history` frames of detection centers per class.
- "Nearest match" = nearest center of the same class in the previous frame (greedy 1-1 assignment is fine for v1; ByteTrack-style identity is overkill here).
- A frame with no allowlisted detections always returns `False`.

#### `backend/app/recording/recorder.py` — `MotionRecorder`

One instance per camera. Owns ring buffer, motion detector, current writer.

```python
class MotionRecorder:
    def __init__(self, camera_id: str, frame_size: tuple[int, int]) -> None: ...
    def feed(self, frame_bgr: np.ndarray, detections: list[dict], ts: float) -> None: ...
    def close(self) -> None: ...  # flush + close on shutdown / camera disconnect
```

**State machine:**

| State | On motion frame | On no-motion frame |
|---|---|---|
| `IDLE` | → `RECORDING`: open MP4, write ring buffer (pre-roll), save snapshot JPG, write current frame | stay `IDLE` |
| `RECORDING` | write frame; if 15s elapsed, close file, open next 15s file (no new snapshot), reset timer | → `POST_ROLL`: keep writing, mark `last_motion_ts` |
| `POST_ROLL` | → `RECORDING` (re-enter, same file) | write frame; if `now - last_motion_ts > 2.0s`, close file, → `IDLE` |

**Notes:**
- Ring buffer holds `pre_roll_seconds * clip_fps` frames (rounded up). Discarded when motion starts (its contents are flushed to the new MP4).
- Snapshot is the **frame on which motion was first detected** (the trigger frame, not the pre-roll start). The MP4 starts 2s earlier from the ring buffer, but the `.jpg` shows the moment the trigger fired.
- File basename for a new event = `HHMMSS` from the MP4's first-frame timestamp (i.e. the start of the pre-roll). Continuation files (15s rotations during the same event) get a fresh `HHMMSS` from their own start, no `.jpg`.
- Resolution is locked from the first frame of a clip. If frame size changes mid-clip, current clip is closed and a new one opens on the next motion frame.

#### `backend/app/recording/manager.py` — `RecorderManager`

```python
class RecorderManager:
    def __init__(self) -> None:
        self._recorders: dict[str, MotionRecorder] = {}
    def feed(self, camera_id: str, frame_bgr: np.ndarray, detections: list[dict]) -> None: ...
    def close_all(self) -> None: ...  # called on shutdown
```

- Lazy-creates a `MotionRecorder` on first frame from a `camera_id`.
- Singleton instance exported as `recorder_manager`.

### 4.2 Existing files — minimal edits

#### `backend/app/api/stream.py`

After `result = engine.infer(...)` and after `await ws.send_json(result)`:

```python
camera_id = fp.header.get("camera_id")
if camera_id:
    detections = result.get("boxes") or _boxes_from_people(result.get("people"))
    recorder_manager.feed(camera_id, img, detections or [])
```

- If `camera_id` is missing, recording is skipped silently. Inference still runs (backward-compatible).
- `_boxes_from_people` is a 3-line helper inside `stream.py` (or `recording/__init__.py`) that flattens `pose` results' `box` field into the same shape as `detect` results, so `MotionDetector` only needs to know one shape.
- For modes that don't produce boxes (`sam3`, `cls`, `obb`), nothing is fed — those modes don't trigger recording. (Acceptable for v1.)

#### `backend/app/config.py`

Add to `Settings`:

```python
clips_dir: Path = BACKEND_ROOT / "data" / "clips"
clip_seconds: int = 15
pre_roll_seconds: float = 2.0
post_roll_seconds: float = 2.0
motion_px_threshold: int = 12
motion_classes: tuple[str, ...] = ("person",)
clip_fps: int = 15  # nominal FPS written into the MP4
```

`clip_fps` is the FPS metadata stamped into the MP4 container. Actual frame rate from the WS will drift around this; that's fine — playback uses container FPS.

### 4.3 No edits

- `protocol/frame.py`: untouched. `camera_id` is just a new optional field inside the existing JSON header.
- `inference/*`: untouched.
- `runtime/*`: untouched.

## 5. Frontend components

### 5.1 New files

#### `frontend/src/camera/CameraPicker.tsx`

- On mount: `navigator.mediaDevices.enumerateDevices()` → filter `kind === "videoinput"`.
- Renders a `<select>` of cameras (label = device label or `"Camera <n>"`).
- Stores chosen `deviceId` in `appState`.
- Renders a text input for **camera id** (the human-readable name sent to the backend; e.g. `front-door`). Default: `cam-<first 6 chars of deviceId>`.

### 5.2 Existing files — edits

#### `frontend/src/state/appState.ts`

Add:

```ts
selectedDeviceId: string | null
cameraId: string                // sent to backend
setSelectedDeviceId(id: string): void
setCameraId(id: string): void
```

#### `frontend/src/camera/useWebcam.ts`

Accept a `deviceId?: string` option. Pass it through:

```ts
video: {
  deviceId: opts.deviceId ? { exact: opts.deviceId } : undefined,
  width: { ideal: ... },
  height: { ideal: ... },
}
```

Restart the stream when `deviceId` changes.

#### `frontend/src/stream/useFrameSender.ts` (and/or `protocol.ts` call sites)

When building the packet header, include `camera_id` from `appState`:

```ts
const header = { camera_id: cameraId, conf, /* ...existing fields */ };
```

#### `frontend/src/App.tsx`

Mount `<CameraPicker />` somewhere in the UI chrome. Pass the selected `deviceId` into `useWebcam`.

## 6. Data flow on one frame

```
1. Frontend captures webcam frame → JPEG blob
2. Frontend builds packet, header = { camera_id, conf, ... } → /ws
3. Backend protocol.parse → header, jpeg
4. Backend jpeg_to_bgr → BGR frame
5. Backend engine.infer(frame, header) → result
6. Backend ws.send_json(result) (existing)
   ─── new step 7 ───
7. recorder_manager.feed(camera_id, frame, detections_from_result):
   a. lookup/create MotionRecorder[camera_id]
   b. push frame onto 2s ring buffer
   c. motion_detector.update(detections) → bool
   d. state machine reacts (open/rotate/close MP4, save snapshot)
```

## 7. File layout on disk

```
backend/data/clips/
└── cam-front/                   ← camera_id from frontend
    └── 2026-05-26/
        ├── 143052.mp4           ← 15s clip, motion event A starts here
        ├── 143052.jpg           ← snapshot of event A
        ├── 143107.mp4           ← rotation: event A still going
        ├── 143122.mp4           ← rotation: event A still going
        ├── 143209.mp4           ← motion event B (new event after gap)
        └── 143209.jpg
```

- `data/clips/` is gitignored.
- Folders are created lazily (`mkdir parents=True, exist_ok=True`).

## 8. Configuration summary

All in `backend/app/config.py`. Hard-coded; no CLI flags, no env vars in v1.

| Constant | Default | Purpose |
|---|---|---|
| `clips_dir` | `backend/data/clips` | Root for all recordings |
| `clip_seconds` | `15` | Max length of one MP4 file |
| `pre_roll_seconds` | `2.0` | Ring-buffer length flushed at motion start |
| `post_roll_seconds` | `2.0` | Grace period after motion ends before closing |
| `motion_px_threshold` | `12` | Min bbox-center displacement (in source pixels) to count as moving |
| `motion_classes` | `("person",)` | Class allowlist that can trigger recording |
| `clip_fps` | `15` | FPS stamped into MP4 container |

## 9. Testing

Three offline tests (no webcam, no WS).

1. **`tests/test_motion.py`**
   - Box drifting > threshold across frames → `update` returns `True`.
   - Identical boxes across frames → `False`.
   - Boxes of a non-allowlisted class moving → `False`.
   - Empty detection list → `False`.

2. **`tests/test_recorder.py`**
   - Synthesize 60 BGR frames at 15 fps; frames 5–35 contain motion; rest are static.
   - Assert: at least one `.mp4` produced, file exists and is > 0 bytes.
   - Assert: one `.jpg` snapshot exists with the same basename as the first `.mp4` of the event.
   - Assert: motion stretching across the 15s boundary produced ≥ 2 MP4 files in the same event.

3. **`tests/test_manager.py`**
   - Interleave frames from `cam-A` and `cam-B`.
   - Assert each camera produces its own subfolder.
   - Assert no cross-contamination (recorder for A is separate object from B's).

Frontend gets no tests in this pass (no test infrastructure in the repo).

## 10. .gitignore additions

```
backend/data/
```

(Already implied by README but not present yet.)

## 11. Open questions / future work

- **Cleanup / retention.** Cron-style deletion of clips older than N days. Not in scope here.
- **H.264 encoding.** Try `cv2.VideoWriter_fourcc(*"avc1")` for better compression; falls back to `mp4v` if codec missing.
- **Browse UI.** A read-only HTTP route + small frontend page to list clips and play them.
- **Tracking-based motion.** Use ByteTrack IDs once we add them (Phase 1 of the broader design doc), so motion attribution survives occlusions cleanly.
- **Background thread for writes** if disk I/O ever shows up in profiling.

---

*End of design.*
