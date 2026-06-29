# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Live webcam inference playground. The browser captures webcam frames, streams JPEG-encoded frames over a binary WebSocket protocol to a FastAPI backend, which runs **YOLO26 (Ultralytics)** or **SAM 3** inference and returns normalized detection results that the frontend overlays on the live video. Layered on top: motion-triggered clip recording and InsightFace-based face recognition (both live overlay and offline clip processing).

- **Backend**: Python 3.12 + FastAPI + WebSocket, CUDA-accelerated (RTX 4070).
- **Frontend**: Vite + React + TypeScript + Zustand.
- SAM 3 weights require HuggingFace access approval for `facebook/sam3`; YOLO26 weights auto-download via Ultralytics on first use.

## Commands

### Backend (`backend/`)
```powershell
.\.venv\Scripts\Activate.ps1            # venv already created with py -3.12
uvicorn app.main:app --reload           # serves on http://127.0.0.1:8000
pip install -e ".[dev]"                 # install incl. pytest/ruff (first-time / after dep change)

pytest                                  # all tests
pytest tests/recording/test_motion.py   # single file
pytest tests/recording/test_motion.py::test_name   # single test
ruff check app                          # lint (line-length 100, target py312)
```

### Frontend (`frontend/`)
```powershell
npm run dev      # Vite dev server on http://localhost:5173
npm run build    # tsc -b && vite build (type-check is part of build)
npm run lint     # eslint .
```

Open http://localhost:5173 and allow camera access. The frontend has **no Vite proxy** — REST base URL and WebSocket URL are hardcoded to `http://localhost:8000` / `ws://localhost:8000/ws`. If you change the backend port, update [frontend/src/api/recognition.ts](frontend/src/api/recognition.ts), [frontend/src/pages/LivePage.tsx](frontend/src/pages/LivePage.tsx), and `cors_origins` in [backend/app/config.py](backend/app/config.py).

## The frame protocol (the core contract)

This binary packet is the seam between frontend and backend — changes must be made on both sides in lockstep.

Client → server packet ([frontend/src/stream/protocol.ts](frontend/src/stream/protocol.ts) builds it, [backend/app/protocol/frame.py](backend/app/protocol/frame.py) parses it):
```
[u8 mode_id][u8 variant_id][u32 LE header_len][header_json (optional)][jpeg_bytes]
```
- `mode_id`: 0=yolo_detect, 1=yolo_pose, 2=sam3_image, 3=sam3_video, 4=yolo_seg, 5=yolo_obb, 6=yolo_cls
- `variant_id`: YOLO size n/s/m/l/x = 0–4 (or SAM3 prompt kind)
- `header_json`: `{camera_id, conf, text, recognize, tracking}` — all optional

Server → client is JSON: `{type, boxes?, people?, masks?, faces?, frame_id, ms}`. **All coordinates are normalized `[0,1]`**; the frontend scales them to canvas pixels. `faces` is an optional array of `FaceMatch` nested into detect/pose responses when `recognize` is set.

Back-pressure: the sender drops frames if the WebSocket isn't OPEN or a frame is already in-flight (no queueing) — see [frontend/src/stream/useFrameSender.ts](frontend/src/stream/useFrameSender.ts).

## Backend architecture (`backend/app/`)

- **[config.py](backend/app/config.py)** — frozen `Settings` dataclass, instantiated as module-level singleton `settings`. **Single source of truth** for all paths, ports/CORS, model defaults, recording params, and recognition thresholds. Change behavior here, not scattered constants.
- **[main.py](backend/app/main.py)** — app factory + `lifespan`: on startup creates dirs, runs `db.init_schema()`, `gallery.reload()`, starts the recognition worker; stops the worker on shutdown.
- **api/** — routers. `stream.py` is the WebSocket inference endpoint (`/ws`, no prefix); the rest mount under `/api`: `health.py` (`/health`, `/device`), `clips.py` (`/clips`), `recognition.py` (people CRUD, `/review` queue).
- **runtime/registry.py** — `ModelRegistry` singleton. Lazy-loads an engine per `(mode, size)` on first request and keeps it warm in-process. This is why the first frame of a new mode is slow.
- **inference/** — `yolo.py` (`YoloEngine` wrapping `ultralytics.YOLO`, weights `yolo26{size}{suffix}.pt` in `weights/`) and `sam3_image.py` (text-prompted segmentation). Both serialize to normalized boxes; masks/seg are PNG-encoded base64 in the `rle` field. `tracking.py` — `CameraTrackers` runs one Ultralytics ByteTrack instance per `camera_id` off the shared detections; the YOLO detect path stamps a `track_id` on each box when `tracking` or `recognize` is set (live recognition pins names to track ids, so it implies tracking).

### Recognition subsystem (`recognition/`)
Singletons wired in [recognition/__init__.py](backend/app/recognition/__init__.py): `face_engine`, `gallery`, `recognition_worker`, `live_recognizer`.
- **engine.py** — `FaceEngine` wraps InsightFace `buffalo_l`; produces 512-d L2-normalized embeddings + 112×112 aligned crops.
- **gallery.py** — in-memory `(N, 512)` matrix; `match()` is brute-force cosine similarity (lock-protected). Rebuilt from DB via `reload()`.
- **Two-tier matching** (thresholds in config): `score >= match_high` (0.55) → confident sighting recorded; `match_low` (0.40) `<= score < match_high` → enqueued to `review_queue` for human labeling; below → unknown.
- **live.py** — `LiveRecognizer`: track-pinned live overlay recognition. A recognized name is pinned to `(camera_id, track_id)` and reused every frame at zero cost; the per-frame `annotate()` only reads the cache and enqueues misses onto a background worker thread (off the WS event loop) that runs detect/embed/match. Confirmed (high) tracks re-confirm every `live_recognition_refresh`s, tentative ones back off `live_recognition_retry`s, and a track's cached name is evicted after `track_ttl`s unseen (`recognition_queue_max` bounds the job queue).
- **worker.py** — `RecognitionWorker`: background thread triggered by the `on_clip_closed` callback; samples a closed clip at `recognition_sample_fps`, matches faces, records sightings / enqueues review items.
- **db.py** — SQLite schema: `persons`, `faces` (embedding stored as 2048-byte BLOB), `review_queue`, `clip_sightings`. DB at `backend/data/app.db`.

### Recording subsystem (`recording/`)
- **motion.py** — `MotionDetector`: flags motion when an allowlisted class's bbox center (normalized) drifts beyond `motion_threshold`, using greedy nearest-neighbor matching across frames.
- **recorder.py** — `MotionRecorder`: `IDLE → RECORDING → POST_ROLL` state machine with a pre-roll ring buffer (`pre_roll_seconds` × `clip_fps`). Writes MP4 (H.264 with mp4v fallback) + a trigger snapshot JPG, rotates every `clip_seconds`. Files: `data/clips/{camera_id}/{YYYY-MM-DD}/{HHMMSS}.{mp4,jpg}`.
- **manager.py** — `RecorderManager`: one recorder per `camera_id`, lazily created, wired to the recognition worker's clip callback.

The stream handler feeds each frame's detections into the recorder when `camera_id` is present, and into the live recognizer when `recognize` is set.

## Frontend architecture (`frontend/src/`)

Data flow:
```
useWebcam → CameraView → useFrameSender (canvas→JPEG→buildPacket) → useInferenceWS.send
  → server → useInferenceWS.lastMessage → LivePage.pushResult → appState.lastResult → OverlayCanvas
```

- **state/appState.ts** — single Zustand store holding all inference config (`mode`, `yoloSize`, `yoloConf`, `sam3*`, `tracking`, `recognizeFaces`, `mirror`, `paused`), camera selection, and live results (`fps`, `inferenceMs`, `lastResult`). `useFrameSender` reads config via a ref and packs it into the packet header each tick.
- **stream/** — `protocol.ts` (packet builder + server message types), `useFrameSender.ts` (rAF capture loop, 640px max side, JPEG q0.7, drop-on-backpressure, FPS counter), `useInferenceWS.ts` (connection + JSON parsing).
- **overlay/OverlayCanvas.tsx** — canvas sized to video intrinsic resolution; routes `lastResult.type` to `drawBoxes` / `drawPose` (COCO-17) / `drawMasks` / `drawOBB` / `drawClassification`. `drawFaceNames.ts` exists for `FaceMatch` overlays.
- **pages/** — routes (React Router v7, defined in `main.tsx`): `/` LivePage, `/people` PeoplePage, `/people/:id` PersonDetailPage, `/review` ReviewPage.
- **api/recognition.ts** — REST client for people/clips/review; `absUrl()` resolves relative media paths against the backend base URL.

## Conventions

- Coordinates crossing the protocol are always normalized `[0,1]`; only the overlay layer converts to pixels.
- Backend behavior is tuned through `Settings` in `config.py` — prefer adding a setting over hardcoding.
- Models load lazily and stay warm; don't add eager loading without reason (startup cost, VRAM).
